import numpy as np

import torch
import torch.nn as nn
import torchaudio
from torch.nn import functional as F

class HybridLoss(nn.Module):
    """
    Experimental. 
    """
    def __init__(self, args):
        super().__init__()
        # stft loss
        n_ffts = args.loss.n_ffts
        self.loss_mss_func = MSSLoss(n_ffts)
        
        # mel loss
        sample_rate  = args.data.sampling_rate
        n_fft     = args.data.win_length
        hop_length     = args.data.block_size
        n_mels = args.data.n_mels
        f_min = args.data.mel_fmin
        f_max = args.data.mel_fmax
        self.loss_mel_func = MelLoss(sample_rate, n_fft, hop_length, n_mels, f_min, f_max)

    def forward(self, y_pred, y_true):
        loss_mss = self.loss_mss_func(y_pred, y_true)
        loss_mel = self.loss_mel_func(y_pred, y_true)
        return loss_mss + loss_mel

class MelLoss(nn.Module):
    def __init__(self, sample_rate, n_fft, hop_length, n_mels, f_min=0, f_max=None, eps=1e-7):
        super().__init__()
        self.eps = eps 
        self.melspec = torchaudio.transforms.MelSpectrogram(
                                                sample_rate=sample_rate, 
                                                n_fft=n_fft, 
                                                hop_length=hop_length,
                                                n_mels=n_mels,
                                                f_min=f_min,
                                                f_max=f_max,
                                                mel_scale='slaney')
                                                
    def forward(self, x_pred, x_true):
        min_len = np.min([x_true.shape[1], x_pred.shape[1]])
        
        # print('--------')
        # print(min_len)
        # print('x_pred:', x_pred.shape)
        # print('x_true:', x_true.shape)

        x_true = x_true[:, -min_len:]
        x_pred = x_pred[:, -min_len:]

        # print('x_pred:', x_pred.shape)
        # print('x_true:', x_true.shape)
        # print('--------\n\n\n')

        S_true = self.melspec(x_true)
        S_pred = self.melspec(x_pred)
        converge_term = torch.linalg.norm(S_true - S_pred) / (torch.linalg.norm(S_true) + self.eps)
        log_term = F.l1_loss((S_true + self.eps).log2(), (S_pred + self.eps).log2())

        loss = converge_term + log_term
        return loss

class SSSLoss(nn.Module):
    """
    Single-scale Spectral Loss. 
    """

    def __init__(self, n_fft=111, alpha=1.0, overlap=0.75, eps=1e-7):
        super().__init__()
        self.n_fft = n_fft
        self.alpha = alpha
        self.eps = eps
        self.hop_length = int(n_fft * (1 - overlap))  # 25% of the length
        self.spec = torchaudio.transforms.Spectrogram(n_fft=self.n_fft, hop_length=self.hop_length)
        
    def forward(self, x_true, x_pred):
        min_len = np.min([x_true.shape[1], x_pred.shape[1]])
        
        # print('--------')
        # print(min_len)
        # print('x_pred:', x_pred.shape)
        # print('x_true:', x_true.shape)

        x_true = x_true[:, -min_len:]
        x_pred = x_pred[:, -min_len:]

        # print('x_pred:', x_pred.shape)
        # print('x_true:', x_true.shape)
        # print('--------\n\n\n')

        S_true = self.spec(x_true)
        S_pred = self.spec(x_pred)
        converge_term = torch.linalg.norm(S_true - S_pred) / (torch.linalg.norm(S_true) + self.eps)
        log_term = F.l1_loss((S_true + self.eps).log(), (S_pred + self.eps).log())

        loss = converge_term + self.alpha * log_term
        return loss
        

class MSSLoss(nn.Module):
    """
    Multi-scale Spectral Loss.
    Usage ::
    mssloss = MSSLoss([2048, 1024, 512, 256], alpha=1.0, overlap=0.75)
    mssloss(y_pred, y_gt)
    input(y_pred, y_gt) : two of torch.tensor w/ shape(batch, 1d-wave)
    output(loss) : torch.tensor(scalar)

    48k: n_ffts=[2048, 1024, 512, 256]
    24k: n_ffts=[1024, 512, 256, 128]
    """

    def __init__(self, n_ffts, alpha=1.0, ratio = 1.0, overlap=0.75, eps=1e-7):
        super().__init__()
        self.losses = nn.ModuleList([SSSLoss(n_fft, alpha, overlap, eps) for n_fft in n_ffts])
        self.ratio = ratio
        
    def forward(self, x_pred, x_true):
        x_pred = x_pred[..., :x_true.shape[-1]]
        value = 0.
        for loss in self.losses:
            value += loss(x_true, x_pred)
        return self.ratio * value

