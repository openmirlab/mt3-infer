"""
MR-MT3 Adapter for mt3-infer

Extracted from: https://github.com/gudgud96/MR-MT3
Commit: 826ea84a933f93cd707d11e91af711f1d19c8d79
License: MIT

Copyright (c) 2024 Hao Hao Tan
Adapted for mt3-infer by extracting PyTorch-only inference code.

This adapter implements the MT3Base interface using MR-MT3's PyTorch implementation.
Key changes from upstream:
- Extracted PyTorch-only spectrogram code (skip TensorFlow/DDSP dependencies)
- Extracted vocabulary/codec decoding logic (skip seqio/TensorFlow dependencies)
- Modernized to torch 2.7.1 (replaced torch.no_grad with torch.inference_mode)
- Removed all training code (datasets, losses, optimizers)
- Adapted to MT3Base interface (load_model, preprocess, forward, decode, transcribe)
- Proper codec-based MIDI decoding with accurate pitches, timing, and velocities
"""

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, List, Optional

import mido
import numpy as np
import sys
import torch
import torch.nn as nn
from torchaudio.transforms import MelSpectrogram
from transformers import T5Config

from mt3_infer.base import MT3Base
from mt3_infer.exceptions import CheckpointError, AudioError, InferenceError
from mt3_infer.adapters import vocab_utils  # Proper codec-based decoding

def _import_t5_model():
    """Import MR-MT3's custom T5 model."""
    from mt3_infer.models.mr_mt3.t5 import T5ForConditionalGeneration
    return T5ForConditionalGeneration


# Spectrogram constants (from contrib/spectrograms.py)
MIN_LOG_MEL = -12
MAX_LOG_MEL = 5
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_HOP_WIDTH = 128
DEFAULT_NUM_MEL_BINS = 512
FFT_SIZE = 2048
MEL_LO_HZ = 20.0


class SpectrogramConfig:
    """Spectrogram configuration (PyTorch-only, no TF deps)."""
    def __init__(
        self,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        hop_width: int = DEFAULT_HOP_WIDTH,
        num_mel_bins: int = DEFAULT_NUM_MEL_BINS,
    ):
        self.sample_rate = sample_rate
        self.hop_width = hop_width
        self.num_mel_bins = num_mel_bins

    @property
    def frames_per_second(self):
        return self.sample_rate / self.hop_width


def pad_end(samples: torch.Tensor, n_fft: int, hop_size: int) -> torch.Tensor:
    """Pad audio to ensure complete frames."""
    n_samples = samples.shape[-1]
    n_frames = -(-n_samples // hop_size)  # ceiling division
    pad_samples = max(0, n_fft + hop_size * (n_frames - 1) - n_samples)
    return torch.nn.functional.pad(samples, (0, pad_samples))


def safe_log(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Avoid taking log of non-positive numbers."""
    safe_x = torch.where(x <= 0.0, torch.tensor(eps), x)
    return torch.log(safe_x)


def compute_spectrogram(samples: np.ndarray, config: SpectrogramConfig) -> np.ndarray:
    """Compute mel spectrogram using PyTorch (no TensorFlow)."""
    transform = MelSpectrogram(
        sample_rate=config.sample_rate,
        n_fft=FFT_SIZE,
        hop_length=config.hop_width,
        n_mels=config.num_mel_bins,
        f_min=MEL_LO_HZ,
        f_max=7600,
        power=1.0,
        center=False
    )
    samples_tensor = torch.from_numpy(samples).float()
    S = transform(pad_end(samples_tensor, FFT_SIZE, config.hop_width))
    S = safe_log(S)
    return S.numpy()


def split_audio(samples: np.ndarray, hop_width: int) -> np.ndarray:
    """Split audio into frames (PyTorch version)."""
    samples_tensor = torch.from_numpy(samples)
    frames = samples_tensor.unfold(0, hop_width, hop_width)
    return frames.numpy()


def flatten_frames(frames: np.ndarray) -> np.ndarray:
    """Flatten frames back to 1D audio."""
    return frames.flatten()


class MRMT3Adapter(MT3Base):
    """
    MR-MT3 adapter implementing MT3Base interface.

    Uses PyTorch-only inference path (no TensorFlow/DDSP dependencies).
    Loads pretrained checkpoint from HuggingFace or local path.
    """

    def __init__(self):
        super().__init__()
        self.model: Optional[nn.Module] = None
        self.device_str: str = "cpu"
        self.spectrogram_config = SpectrogramConfig()

        # Vocabulary and codec will be initialized in load_model
        self.vocab = None
        self.codec = None

        # Model config (MT3 standard)
        self.model_config = {
            'd_model': 512,
            'd_ff': 1024,
            'd_kv': 64,
            'num_heads': 6,
            'num_layers': 8,
            'num_decoder_layers': 8,
            'vocab_size': 1536,
            'encoder_vocab_size': 1024,
            'decoder_start_token_id': 0,
            'pad_token_id': 0,
            'eos_token_id': 1,
            'unk_token_id': 2,
            'feed_forward_proj': 'gated-gelu',
            'dropout_rate': 0.1,
            'layer_norm_epsilon': 1e-6,
            'is_encoder_decoder': True,
            'tie_word_embeddings': False,
            'use_cache': False,
        }

    def load_model(self, checkpoint_path: str, device: str = "auto") -> None:
        """
        Load MR-MT3 model from checkpoint.

        Args:
            checkpoint_path: Path to .pth checkpoint file
            device: Device to load model on ('cuda', 'cpu', or 'auto')

        Raises:
            CheckpointError: If checkpoint loading fails
        """
        from mt3_infer.utils.framework import get_device, check_torch_version

        # Verify PyTorch version
        check_torch_version()

        # Resolve device
        self.device_str = get_device(device)

        # Validate checkpoint path
        if not os.path.exists(checkpoint_path):
            raise CheckpointError(
                f"Checkpoint not found: {checkpoint_path}\n"
                f"Download from: https://huggingface.co/gudgud1014/MR-MT3/resolve/main/mt3.pth"
            )

        try:
            # Lazy import MR-MT3's custom T5 model (avoids contrib dependencies)
            T5ForConditionalGeneration = _import_t5_model()

            # Create T5 config from model_config dict
            config = T5Config.from_dict(self.model_config)

            # Initialize model architecture using MR-MT3's custom T5
            self.model = T5ForConditionalGeneration(config)

            # Load checkpoint weights
            # Note: weights_only=False required for pickle compatibility with PyTorch 2.6+
            state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            self.model.load_state_dict(state_dict, strict=False)

            # Move to device and set eval mode
            self.model.to(self.device_str)
            self.model.eval()

            # Initialize vocabulary and codec
            self._initialize_vocab()

            self._model_loaded = True

        except Exception as e:
            raise CheckpointError(
                f"Failed to load checkpoint from {checkpoint_path}: {str(e)}"
            ) from e

    def _initialize_vocab(self):
        """Initialize vocabulary and codec for MIDI decoding."""
        # Build codec (1 velocity bin for MT3 pretrained checkpoint)
        self.codec = vocab_utils.build_codec(num_velocity_bins=1)
        # Vocabulary size = codec classes + 3 special tokens (PAD, EOS, UNK)
        self.vocab_size = self.codec.num_classes + 3

    def preprocess(self, audio: np.ndarray, sr: int) -> Any:
        """
        Preprocess audio to model input format.

        Args:
            audio: Audio array (float32, [-1, 1] range)
            sr: Sample rate

        Returns:
            Preprocessed features ready for model forward pass
        """
        # Resample if needed
        if sr != self.spectrogram_config.sample_rate:
            raise AudioError(
                f"Audio sample rate {sr} does not match expected "
                f"{self.spectrogram_config.sample_rate}. "
                f"Use mt3_infer.utils.audio.load_audio() for automatic resampling."
            )

        # Split audio into frames
        frames = split_audio(audio, self.spectrogram_config.hop_width)

        # Compute frame times
        num_frames = len(frames)
        frame_times = np.arange(num_frames) / self.spectrogram_config.frames_per_second

        # Split into batches (max 256 frames per batch)
        max_length = 256
        num_segments = math.ceil(num_frames / max_length)

        batches = []
        frame_times_batches = []
        paddings = []

        for i in range(num_segments):
            batch = np.zeros((max_length, self.spectrogram_config.hop_width))
            frame_times_batch = np.zeros(max_length)

            start_idx = i * max_length
            end_idx = min((i + 1) * max_length, num_frames)
            length = end_idx - start_idx

            batch[:length] = frames[start_idx:end_idx]
            frame_times_batch[:length] = frame_times[start_idx:end_idx]

            batches.append(batch)
            frame_times_batches.append(frame_times_batch)
            paddings.append(length)

        # Compute spectrograms for each batch
        spectrograms = []
        for batch in batches:
            # Flatten frames back to audio
            audio_segment = flatten_frames(batch)
            # Compute mel spectrogram (returns shape: freq_bins x time_frames)
            spec = compute_spectrogram(audio_segment, self.spectrogram_config)
            # Transpose to (time_frames x freq_bins) for model input
            spec = spec.T
            spectrograms.append(spec)

        spectrograms = np.stack(spectrograms, axis=0)  # (batch, time, freq)

        # Apply normalization (not used for pretrained MT3)
        # spectrograms = np.clip(spectrograms, MIN_LOG_MEL, MAX_LOG_MEL)
        # spectrograms = (spectrograms - MIN_LOG_MEL) / (MAX_LOG_MEL - MIN_LOG_MEL)

        # Zero out padding (zero from padding position onwards in time dimension)
        for i, p in enumerate(paddings):
            spectrograms[i, p:, :] = 0

        return {
            'inputs': torch.from_numpy(spectrograms).float(),
            'frame_times': np.stack(frame_times_batches, axis=0),
            'paddings': paddings
        }

    @torch.inference_mode()
    def forward(self, features: Any) -> Any:
        """
        Run model inference.

        Args:
            features: Preprocessed features from preprocess()

        Returns:
            Model outputs (token predictions)
        """
        inputs = features['inputs'].to(self.device_str)

        try:
            # Run generation
            outputs = self.model.generate(
                inputs=inputs,
                max_length=1024,
                num_beams=1,
                do_sample=False,
                length_penalty=0.4,
                eos_token_id=self.model.config.eos_token_id,
                early_stopping=False,
                use_cache=False,
            )

            # Post-process: remove EOS tokens and special tokens
            after_eos = torch.cumsum((outputs == self.model.config.eos_token_id).float(), dim=-1)
            # Subtract 3 special tokens (PAD, EOS, UNK) to get codec token indices
            outputs = outputs - 3
            outputs = torch.where(after_eos.bool(), -1, outputs)
            outputs = outputs[:, 1:]  # Remove BOS token

            return {
                'tokens': outputs.cpu().numpy(),
                'frame_times': features['frame_times']
            }

        except Exception as e:
            raise InferenceError(
                f"Model forward pass failed: {str(e)}"
            ) from e

    def decode(self, outputs: Any) -> mido.MidiFile:
        """
        Decode model outputs to MIDI file using proper codec-based decoding.

        Args:
            outputs: Model outputs from forward()

        Returns:
            mido.MidiFile object
        """
        tokens = outputs['tokens']
        frame_times = outputs['frame_times']

        # Prepare predictions for decoder (one per batch)
        predictions = []
        for batch_idx, (batch_tokens, batch_frame_times) in enumerate(zip(tokens, frame_times)):
            # Extract valid tokens (filter padding/EOS marked as -1)
            valid_tokens = batch_tokens[batch_tokens >= 0]
            if len(valid_tokens) > 0:
                predictions.append({
                    'est_tokens': valid_tokens,
                    'start_time': float(batch_frame_times[0])
                })

        # Decode using codec
        note_sequence, invalid_events, dropped_events = vocab_utils.decode_and_combine_predictions(
            predictions, self.codec
        )

        # Convert NoteSequence to mido.MidiFile
        midi = mido.MidiFile(ticks_per_beat=note_sequence.ticks_per_quarter)
        track = mido.MidiTrack()
        midi.tracks.append(track)

        # Set tempo (default 120 BPM)
        track.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(120)))

        # Sort notes by start time
        sorted_notes = sorted(note_sequence.notes, key=lambda n: n.start_time)

        # Convert notes to MIDI messages
        events = []
        for note in sorted_notes:
            # Note on
            events.append({
                'time': note.start_time,
                'type': 'note_on',
                'note': note.pitch,
                'velocity': note.velocity,
                'channel': 9 if note.is_drum else 0
            })
            # Note off
            events.append({
                'time': note.end_time,
                'type': 'note_off',
                'note': note.pitch,
                'velocity': 0,
                'channel': 9 if note.is_drum else 0
            })

        # Sort all events by time
        events.sort(key=lambda e: e['time'])

        # Convert absolute times to delta times and add to track
        current_time = 0.0
        for event in events:
            # Calculate delta time in seconds
            delta_time_seconds = event['time'] - current_time
            # Convert to MIDI ticks
            delta_ticks = int(delta_time_seconds * note_sequence.ticks_per_quarter * 2)  # 2 = 120 BPM / 60
            delta_ticks = max(0, delta_ticks)  # Ensure non-negative

            # Add MIDI message
            if event['type'] == 'note_on':
                track.append(mido.Message(
                    'note_on',
                    note=event['note'],
                    velocity=event['velocity'],
                    time=delta_ticks,
                    channel=event['channel']
                ))
            else:  # note_off
                track.append(mido.Message(
                    'note_off',
                    note=event['note'],
                    velocity=event['velocity'],
                    time=delta_ticks,
                    channel=event['channel']
                ))

            current_time = event['time']

        # Add end of track
        track.append(mido.MetaMessage('end_of_track', time=0))

        return midi
