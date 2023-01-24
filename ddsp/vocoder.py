import os
import numpy as np
import yaml
import torch
from librosa.filters import mel as librosa_mel_fn
from .mel2control import Mel2Control
from .modules import SawtoothGenerator, HarmonicOscillator
from .core import frequency_filter, upsample, scale_function

class DotDict(dict):
    def __getattr__(*args):         
        val = dict.get(*args)         
        return DotDict(val) if type(val) is dict else val   

    __setattr__ = dict.__setitem__    
    __delattr__ = dict.__delitem__
    
def load_model(
        model_path,
        device='cpu'):
    config_file = os.path.join(os.path.split(model_path)[0], 'config.yaml')
    with open(config_file, "r") as config:
        args = yaml.safe_load(config)
    args = DotDict(args)
    
    # load model
    model = None

    if args.model.type == 'Sins':
        model = Sins(
            sampling_rate=args.data.sampling_rate,
            block_size=args.data.block_size,
            n_harmonics=args.model.n_harmonics,
            n_mag_noise=args.model.n_mag_noise,
            n_mels=args.data.n_mels)

    elif args.model.type == 'SawSub':
        model = SawSub(
            sampling_rate=args.data.sampling_rate,
            block_size=args.data.block_size,
            n_mag_harmonic=args.model.n_mag_harmonic,
            n_mag_noise=args.model.n_mag_noise,
            n_mels=args.data.n_mels)
            
    elif args.model.type == 'Full':
        model = Full(
            sampling_rate=args.data.sampling_rate,
            block_size=args.data.block_size,
            n_mag_harmonic=args.model.n_mag_harmonic,
            n_mag_noise=args.model.n_mag_noise,
            n_harmonics=args.model.n_harmonics,
            n_sub_harmonics=args.model.n_sub_harmonics,
            n_mels=args.data.n_mels)
            
    else:
        raise ValueError(f" [x] Unknown Model: {args.model.type}")
    
    print(' [Loading] ' + model_path)
    ckpt = torch.load(model_path, map_location=torch.device(device))
    model.to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model, args
    
class Audio2Mel(torch.nn.Module):
    def __init__(
        self,
        hop_length,
        sampling_rate,
        n_mel_channels,
        win_length,
        n_fft=None,
        mel_fmin=0,
        mel_fmax=None,
        clamp = 1e-5
    ):
        super().__init__()
        n_fft = win_length if n_fft is None else n_fft
        window = torch.hann_window(win_length).float()
        mel_basis = librosa_mel_fn(
            sr=sampling_rate,
            n_fft=n_fft, 
            n_mels=n_mel_channels, 
            fmin=mel_fmin, 
            fmax=mel_fmax)
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer("mel_basis", mel_basis)
        self.register_buffer("window", window)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.sampling_rate = sampling_rate
        self.n_mel_channels = n_mel_channels
        self.clamp = 1e-5

    def forward(self, audio):
        '''
              audio: B x C x T
        og_mel_spec: B x T_ x C x n_mel 
        '''
        B, C, T = audio.shape
        audio = audio.reshape(B * C, T)
        fft = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=True,
            return_complex=False)
        real_part, imag_part = fft.unbind(-1)
        magnitude = torch.sqrt(real_part ** 2 + imag_part ** 2)
        mel_output = torch.matmul(self.mel_basis, magnitude)
        log_mel_spec = torch.log10(torch.clamp(mel_output, min=self.clamp))

        # log_mel_spec: B x C, M, T
        T_ = log_mel_spec.shape[-1]
        log_mel_spec = log_mel_spec.reshape(B, C, self.n_mel_channels ,T_)
        log_mel_spec = log_mel_spec.permute(0, 3, 1, 2)

        # print('og_mel_spec:', log_mel_spec.shape)
        log_mel_spec = log_mel_spec.squeeze(2) # mono
        return log_mel_spec
        
class Full(torch.nn.Module):
    def __init__(self, 
            sampling_rate,
            block_size,
            n_mag_harmonic,
            n_mag_noise,
            n_harmonics,
            n_sub_harmonics,
            n_mels=80):
        super().__init__()

        print(' [DDSP Model] Sinusoids + Sawtooth Subtractive Synthesiser ')

        # params
        self.register_buffer("sampling_rate", torch.tensor(sampling_rate))
        self.register_buffer("block_size", torch.tensor(block_size))
        
        # Mel2Control
        split_map = {
            'A': 1,
            'amplitudes': n_harmonics,
            'harmonic_magnitude': n_mag_harmonic,
            'noise_magnitude': n_mag_noise
        }
        self.mel2ctrl = Mel2Control(n_mels, split_map)

        # Harmonic Synthsizer
        self.harmonic_synthsizer_low = HarmonicOscillator(
                                            sampling_rate, 
                                            block_size, 
                                            level_start=1, 
                                            is_remove_above_nyquist=False)
                                            
        self.harmonic_synthsizer_high = HarmonicOscillator(
                                            sampling_rate, 
                                            block_size, 
                                            level_start=n_sub_harmonics+1, 
                                            is_remove_above_nyquist=True)
        
        self.n_sub_harmonics = n_sub_harmonics
        

    def forward(self, mel, f0, initial_phase=None, max_upsample_dim=32):
        '''
            mel: B x n_frames x n_mels
            f0: B x n_frames x 1
        '''

        ctrls = self.mel2ctrl(mel, f0)
        
        A           = scale_function(ctrls['A'])
        amplitudes  = scale_function(ctrls['amplitudes'])
        src_param = scale_function(ctrls['harmonic_magnitude'])
        noise_param = scale_function(ctrls['noise_magnitude'])

        amplitudes /= amplitudes.sum(-1, keepdim=True) # to distribution
        amplitudes *= A
        
        # harmonic
        harmonic_low, final_phase = self.harmonic_synthsizer_low(
                                            f0, 
                                            amplitudes[:,:,:self.n_sub_harmonics], 
                                            initial_phase, 
                                            max_upsample_dim)
        
        harmonic_low = frequency_filter(
                        harmonic_low,
                        src_param)
                        
        harmonic_high, final_phase = self.harmonic_synthsizer_high(
                                            f0, 
                                            amplitudes[:,:,self.n_sub_harmonics:], 
                                            initial_phase, 
                                            max_upsample_dim)
        
        harmonic = harmonic_low + harmonic_high
            
        # noise part
        noise = torch.rand_like(harmonic).to(noise_param) * 2 - 1
        noise = frequency_filter(
                        noise,
                        noise_param)
        signal = harmonic + noise

        return signal, final_phase, (harmonic, noise)

class Sins(torch.nn.Module):
    def __init__(self, 
            sampling_rate,
            block_size,
            n_harmonics,
            n_mag_noise,
            n_mels=80):
        super().__init__()

        print(' [DDSP Model] Sinusoids Additive Synthesiser')

        # params
        self.register_buffer("sampling_rate", torch.tensor(sampling_rate))
        self.register_buffer("block_size", torch.tensor(block_size))
        
        # Mel2Control
        split_map = {
            'A': 1,
            'amplitudes': n_harmonics,
            'noise_magnitude': n_mag_noise,
        }
        self.mel2ctrl = Mel2Control(n_mels, split_map)

        # Harmonic Synthsizer
        self.harmonic_synthsizer = HarmonicOscillator(sampling_rate, block_size)

    def forward(self, mel, f0, initial_phase=None, max_upsample_dim=32):
        '''
            mel: B x n_frames x n_mels
            f0: B x n_frames x 1
        '''

        ctrls = self.mel2ctrl(mel, f0)

        A           = scale_function(ctrls['A'])
        amplitudes  = scale_function(ctrls['amplitudes'])
        noise_param = scale_function(ctrls['noise_magnitude'])

        amplitudes /= amplitudes.sum(-1, keepdim=True) # to distribution
        amplitudes *= A
        
        # harmonic
        harmonic, final_phase = self.harmonic_synthsizer(f0, amplitudes, initial_phase, max_upsample_dim)
                
        # noise part
        noise = torch.rand_like(harmonic).to(noise_param) * 2 - 1
        noise = frequency_filter(
                        noise,
                        noise_param)
        signal = harmonic + noise

        return signal, final_phase, (harmonic, noise) #, (noise_param, noise_param)

class SawSub(torch.nn.Module):
    def __init__(self, 
            sampling_rate,
            block_size,
            n_mag_harmonic,
            n_mag_noise,
            n_mels=80):
        super().__init__()

        print(' [DDSP Model] Sawtooth Subtractive Synthesiser')
        # params
        self.register_buffer("sampling_rate", torch.tensor(sampling_rate))
        self.register_buffer("block_size", torch.tensor(block_size))
        # Mel2Control
        split_map = {
            'harmonic_magnitude': n_mag_harmonic, # 1024 for 48k, 512 for 24k 
            'noise_magnitude': n_mag_noise
        }
        self.mel2ctrl = Mel2Control(n_mels, split_map)

        # Harmonic Synthsizer
        self.harmonic_synthsizer = SawtoothGenerator(sampling_rate)

    def forward(self, mel, f0, initial_phase=None, **kwargs):
        '''
            mel: B x n_frames x n_mels
            f0: B x n_frames x 1
        '''

        ctrls = self.mel2ctrl(mel, f0)
        
        src_param = scale_function(ctrls['harmonic_magnitude'])
        noise_param = scale_function(ctrls['noise_magnitude'])

        # exciter signal
        B, n_frames, _ = f0.shape

        # upsample
        pitch = upsample(f0, self.block_size)

        # harmonic
        harmonic, final_phase = self.harmonic_synthsizer(pitch, initial_phase)
        harmonic= frequency_filter(
                        harmonic,
                        src_param)

        # noise part
        noise = torch.rand_like(harmonic).to(noise_param) * 2 - 1
        noise = frequency_filter(
                        noise,
                        noise_param)
        signal = harmonic + noise

        return signal, final_phase, (harmonic, noise)
