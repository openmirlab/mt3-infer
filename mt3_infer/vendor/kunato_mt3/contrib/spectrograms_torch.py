"""
PyTorch-based Spectrogram Processing for MT3

Replaces TensorFlow/DDSP spectrograms.py with PyTorch implementation.
Compatible with kunato/mt3-pytorch architecture.

Original Copyright 2022 The MT3 Authors (Apache 2.0 License)
PyTorch port for mt3-infer project
"""

import dataclasses
import torch
import torchaudio.transforms as T
import numpy as np

# Defaults for spectrogram config
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_HOP_WIDTH = 128
DEFAULT_NUM_MEL_BINS = 512

# Fixed constants
FFT_SIZE = 2048
MEL_LO_HZ = 20.0


@dataclasses.dataclass
class SpectrogramConfig:
    """Spectrogram configuration parameters."""
    sample_rate: int = DEFAULT_SAMPLE_RATE
    hop_width: int = DEFAULT_HOP_WIDTH
    num_mel_bins: int = DEFAULT_NUM_MEL_BINS

    @property
    def abbrev_str(self):
        s = ''
        if self.sample_rate != DEFAULT_SAMPLE_RATE:
            s += 'sr%d' % self.sample_rate
        if self.hop_width != DEFAULT_HOP_WIDTH:
            s += 'hw%d' % self.hop_width
        if self.num_mel_bins != DEFAULT_NUM_MEL_BINS:
            s += 'mb%d' % self.num_mel_bins
        return s

    @property
    def frames_per_second(self):
        return self.sample_rate / self.hop_width


class SpectrogramProcessor:
    """PyTorch-based mel spectrogram processor for MT3."""

    def __init__(self, config: SpectrogramConfig = None):
        self.config = config or SpectrogramConfig()

        # Create mel spectrogram transform
        self.mel_transform = T.MelSpectrogram(
            sample_rate=self.config.sample_rate,
            n_fft=FFT_SIZE,
            hop_length=self.config.hop_width,
            n_mels=self.config.num_mel_bins,
            f_min=MEL_LO_HZ,
            power=2.0,  # Power spectrogram
            normalized=False,
        )

    def split_audio(self, samples):
        """Split audio into frames.

        Args:
            samples: Audio samples (numpy array or torch tensor)

        Returns:
            Framed audio (batch, frames, hop_width)
        """
        if isinstance(samples, np.ndarray):
            samples = torch.from_numpy(samples).float()

        # Pad to make divisible by hop_width
        remainder = len(samples) % self.config.hop_width
        if remainder != 0:
            padding = self.config.hop_width - remainder
            samples = torch.nn.functional.pad(samples, (0, padding))

        # Reshape into frames
        num_frames = len(samples) // self.config.hop_width
        frames = samples.reshape(num_frames, self.config.hop_width)

        return frames.numpy()

    def compute_spectrogram(self, samples):
        """Compute mel spectrogram.

        Args:
            samples: Audio samples (numpy array or torch tensor)

        Returns:
            Log-mel spectrogram (time, freq)
        """
        if isinstance(samples, np.ndarray):
            samples = torch.from_numpy(samples).float()

        # Ensure 1D
        if samples.dim() == 0:
            samples = samples.unsqueeze(0)

        # Compute mel spectrogram
        mel_spec = self.mel_transform(samples)

        # Convert to log scale (add small epsilon for numerical stability)
        log_mel_spec = torch.log(mel_spec + 1e-6)

        # Transpose to (time, freq) format
        log_mel_spec = log_mel_spec.squeeze(0).T

        return log_mel_spec.numpy()

    def flatten_frames(self, frames):
        """Convert frames back into a flat array of samples.

        Args:
            frames: Framed audio (batch, hop_width)

        Returns:
            Flattened samples
        """
        if isinstance(frames, np.ndarray):
            frames = torch.from_numpy(frames)

        return frames.reshape(-1).numpy()


# Module-level functions for backwards compatibility
def split_audio(samples, spectrogram_config: SpectrogramConfig):
    """Split audio into frames."""
    processor = SpectrogramProcessor(spectrogram_config)
    return processor.split_audio(samples)


def compute_spectrogram(samples, spectrogram_config: SpectrogramConfig):
    """Compute a mel spectrogram."""
    processor = SpectrogramProcessor(spectrogram_config)
    return processor.compute_spectrogram(samples)


def flatten_frames(frames):
    """Convert frames back into a flat array of samples."""
    if isinstance(frames, np.ndarray):
        frames = torch.from_numpy(frames)
    return frames.reshape(-1).numpy()


def input_depth(spectrogram_config: SpectrogramConfig):
    return spectrogram_config.num_mel_bins
