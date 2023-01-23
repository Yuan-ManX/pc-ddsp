import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from .core import fft_convolve, linear_lookup, upsample, remove_above_nyquist, upsample


class HarmonicOscillator(nn.Module):
    """synthesize audio with a bank of harmonic oscillators"""
    def __init__(self, fs, block_size, level_start=1, oscillator=torch.sin, is_remove_above_nyquist=True):
        super().__init__()
        self.fs = fs
        self.oscillator = oscillator
        self.block_size = block_size
        self.level_start = level_start
        self.is_remove_above_nyquist = is_remove_above_nyquist

    def forward(self, f0_frames, amplitudes_frames, initial_phase=None, max_upsample_dim=32):
        '''
                    f0: B x n_frames x 1 (Hz)
            amplitudes: B x n_frames x n_harmonic
         initial_phase: B x 1 x 1
          ---
              signal: B x T
         final_phase: B x 1 x 1
        '''
        # anti-aliasing
        if self.is_remove_above_nyquist:
            amplitudes_frames = remove_above_nyquist(amplitudes_frames, f0_frames, self.fs, self.level_start)
        
        # phase
        f0 = upsample(f0_frames, self.block_size)
        if initial_phase is None:
            initial_phase = torch.zeros(f0.shape[0], 1, 1).to(f0)    
        phase = 2 * np.pi * ((torch.cumsum(f0.double() / self.fs, axis=1) + initial_phase.double()) % 1)
        phase = phase.float() 
        
        # sinusoids
        n_harmonic = amplitudes_frames.shape[-1]
        level_harmonic = torch.arange(self.level_start, n_harmonic + self.level_start).to(phase)
        signal = 0.
        for n in range(( n_harmonic - 1) // max_upsample_dim + 1):
            start = n * max_upsample_dim
            end = (n + 1) * max_upsample_dim
            phases = phase * level_harmonic[start:end]
            amplitudes = upsample(amplitudes_frames[:,:,start:end], self.block_size)
            signal += (self.oscillator(phases) * amplitudes).sum(-1)
                
        final_phase = phase[:, -1:, :]
        return signal, final_phase.detach()

class SawtoothGenerator(nn.Module):
    """synthesize audio with a sawtooth oscillator"""
    def __init__(self, fs, is_reversed=True):
        super().__init__()
        self.fs = fs
        self.is_reversed = is_reversed

    def forward(self, f0, initial_phase=None):
        '''
                    f0: B x T x 1 (Hz)
            amplitudes: B x T x 1
         initial_phase: B x 1 x 1
           ---
              signal: B x T
         final_phase: B x 1 x 1
        '''
        if initial_phase is None:
            initial_phase = torch.zeros(f0.shape[0], 1, 1).to(f0)
        
        # phase
        phase = 2 * np.pi * ((torch.cumsum(f0.double() / self.fs, axis=1) + initial_phase.double()) % 1)
        phase = phase.float() 
        if self.is_reversed:
            phase = 2 * np.pi - phase
        
        # sawtooth        
        signal = (phase / np.pi) - 1
        signal = signal.squeeze(-1)
        
        final_phase = phase[:, -1:, :]
        return signal, final_phase.detach()