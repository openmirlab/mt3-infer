"""
MT3-PyTorch Adapter - Official MT3 Architecture (PyTorch Port)

This adapter wraps the kunato/mt3-pytorch implementation, which is a PyTorch
port of Google's official Magenta MT3 model. Uses HuggingFace Transformers
instead of JAX/T5X, avoiding dependency conflicts.

Architecture:
- Custom T5 with continuous (audio) input encoder
- Standard T5 decoder for token generation
- Event codec for MIDI transcription

Repository: https://github.com/kunato/mt3-pytorch
License: Apache 2.0 (compatible with mt3-infer MIT license)
Model implementation: mt3_infer/models/mt3_pytorch/
"""

import json
import math
import sys
from pathlib import Path
from typing import Optional, List

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import mido
import librosa

from mt3_infer.base import MT3Base
from mt3_infer.exceptions import FrameworkError, ModelNotFoundError


class MT3PyTorchAdapter(MT3Base):
    """
    Adapter for kunato/mt3-pytorch (official MT3 architecture in PyTorch).

    This implementation uses:
    - Custom T5 model with continuous input projection
    - PyTorch-only spectrogram processing (no TensorFlow)
    - Original MT3 codec for event decoding
    - Automatic instrument leakage filtering (optional)

    Known Issue:
        MT3-PyTorch can incorrectly assign drum sounds to melodic instruments
        (bass, chromatic percussion) when transcribing drum tracks. Enable
        auto_filter to automatically correct this.

    Args:
        auto_filter: Enable automatic instrument leakage filtering (default: True)
    """

    SAMPLE_RATE = 16000
    MAX_LENGTH = 256  # Maximum input length (frames)
    MAX_GENERATION_LENGTH = 1024  # Maximum output sequence length

    def __init__(self, auto_filter: bool = True) -> None:
        """
        Initialize MT3-PyTorch adapter.

        Args:
            auto_filter: Whether to automatically filter instrument leakage in drum tracks.
                        MT3-PyTorch has a known issue where it assigns drum sounds to
                        melodic instruments. Set to True to automatically fix this.
                        Default: True
        """
        super().__init__()
        self._model = None
        self._codec = None
        self._vocab = None
        self._spectrogram_processor = None
        self._device = None
        self.auto_filter = auto_filter

    def load_model(
        self,
        checkpoint_path: Optional[str] = None,
        device: str = "auto"
    ) -> None:
        """
        Load MT3-PyTorch model from checkpoint.

        Args:
            checkpoint_path: Path to directory with config.json and mt3.pth
            device: Device placement ('auto', 'cpu', 'cuda')

        Raises:
            ModelNotFoundError: If checkpoint files not found
            FrameworkError: If PyTorch not available
        """
        if checkpoint_path is None:
            raise ValueError(
                "checkpoint_path required for MT3-PyTorch adapter.\n"
                "Download from: https://github.com/kunato/mt3-pytorch/tree/main/pretrained\n"
                "Or use git lfs: cd refs/kunato-mt3-pytorch && git lfs pull"
            )

        checkpoint_path = Path(checkpoint_path)
        config_file = checkpoint_path / "config.json"
        weights_file = checkpoint_path / "mt3.pth"

        # Check if files exist
        if not config_file.exists():
            raise ModelNotFoundError(
                f"Config file not found: {config_file}\n"
                f"Expected structure: {checkpoint_path}/(config.json, mt3.pth)"
            )

        if not weights_file.exists():
            raise ModelNotFoundError(
                f"Weights file not found: {weights_file}\n"
                f"If using git lfs: cd {checkpoint_path.parent} && git lfs pull"
            )

        # Set device
        if device == "auto":
            self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self._device = torch.device(device)

        try:
            # Import from models directory
            from mt3_infer.models.mt3_pytorch.t5 import T5ForConditionalGeneration, T5Config
            from mt3_infer.models.mt3_pytorch.contrib import vocabularies, note_sequences, metrics_utils
            from mt3_infer.models.mt3_pytorch.contrib.spectrograms_torch import SpectrogramProcessor, SpectrogramConfig

            # Load config
            with open(config_file) as f:
                config_dict = json.load(f)

            # Create model
            config = T5Config.from_dict(config_dict)
            self._model: nn.Module = T5ForConditionalGeneration(config)

            # Load weights
            # Note: weights_only=False required for pickle compatibility with PyTorch 2.6+
            state_dict = torch.load(weights_file, map_location='cpu', weights_only=False)
            self._model.load_state_dict(state_dict, strict=True)
            self._model.eval()
            self._model.to(self._device)

            # Initialize spectrogram processor (PyTorch-only)
            spectrogram_config = SpectrogramConfig()
            self._spectrogram_processor = SpectrogramProcessor(spectrogram_config)

            # Build codec and vocabulary
            vocab_config = vocabularies.VocabularyConfig(num_velocity_bins=1)
            self._codec = vocabularies.build_codec(vocab_config)
            self._vocab = vocabularies.vocabulary_from_codec(self._codec)

            # Store encoding spec for decoding
            self._encoding_spec = note_sequences.NoteEncodingWithTiesSpec

            self._model_loaded = True

        except ImportError as e:
            raise FrameworkError(
                f"Failed to import mt3_pytorch modules: {e}\n"
                "Check that mt3_infer/models/mt3_pytorch/ exists."
            ) from e

    def preprocess(self, audio: np.ndarray, sr: int = 16000) -> np.ndarray:
        """
        Convert audio to spectrogram features and chunk into segments.

        Args:
            audio: Audio waveform (float32, mono)
            sr: Sample rate (will be resampled to 16kHz if different)

        Returns:
            Spectrogram features (batch, time, freq)
        """
        if not self._model_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Resample if needed
        if sr != self.SAMPLE_RATE:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.SAMPLE_RATE)

        # Compute frames
        frames, frame_times = self._audio_to_frames(audio)

        # Split into chunks
        frames_chunked, frame_times_chunked, paddings = self._split_into_chunks(
            frames, frame_times, max_length=self.MAX_LENGTH
        )

        # Compute spectrograms for each chunk
        # Note: All chunks are MAX_LENGTH frames, so spectrograms will have same shape
        spectrograms = []
        for chunk_frames in frames_chunked:
            # Flatten frames to audio
            audio_chunk = self._spectrogram_processor.flatten_frames(chunk_frames)

            # Compute spectrogram
            spec = self._spectrogram_processor.compute_spectrogram(audio_chunk)
            spectrograms.append(spec)

        # Stack into batch
        features = np.stack(spectrograms, axis=0)  # (batch, time, freq)

        # Clear frames beyond the valid region. Even when `p` equals MAX_LENGTH,
        # the spectrogram includes one extra frame from the FFT window overlap, so
        # we always zero everything from `p` onward to mirror the reference model.
        for i, p in enumerate(paddings):
            features[i, p:] = 0

        # Store frame times for decoding
        self._last_frame_times = frame_times_chunked

        return features

    def forward(self, features: np.ndarray) -> np.ndarray:
        """
        Run MT3 model inference on spectrograms.

        Args:
            features: Spectrogram features (batch, time, freq)

        Returns:
            Token predictions (batch, seq_len)
        """
        if not self._model_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Convert to tensor
        inputs_tensor = torch.from_numpy(features).float().to(self._device)

        # Run generation
        # Note: Each batch may produce different sequence lengths
        all_outputs = []
        for batch in tqdm(inputs_tensor, desc="Generating MIDI"):
            batch = batch.unsqueeze(0)  # Add batch dimension
            with torch.no_grad():
                result = self._model.generate(
                    inputs=batch,
                    max_length=self.MAX_GENERATION_LENGTH,
                    num_beams=1,
                    do_sample=False,
                    length_penalty=0.4,
                    eos_token_id=self._model.config.eos_token_id,
                    early_stopping=False
                )
            # Postprocess immediately (remove special tokens)
            processed = self._postprocess_batch(result)
            all_outputs.append(processed.cpu().numpy())

        # Return list of outputs (each may have different length)
        # Concatenate along batch dimension by padding to max length
        max_seq_len = max(out.shape[1] for out in all_outputs)
        padded_outputs = []
        for out in all_outputs:
            if out.shape[1] < max_seq_len:
                # Pad with -1 (invalid token marker)
                padding = np.full((out.shape[0], max_seq_len - out.shape[1]), -1, dtype=out.dtype)
                out = np.concatenate([out, padding], axis=1)
            padded_outputs.append(out)

        outputs = np.concatenate(padded_outputs, axis=0)  # (batch, max_seq_len)
        return outputs

    def decode(self, outputs: np.ndarray) -> mido.MidiFile:
        """
        Decode model outputs to MIDI.

        Args:
            outputs: Token predictions (batch, seq_len)

        Returns:
            MIDI file
        """
        if not self._model_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        try:
            from mt3_infer.models.mt3_pytorch.contrib import vocabularies, metrics_utils
            import note_seq

            # Convert tokens to predictions format
            predictions = []
            for i in range(outputs.shape[0]):
                tokens = outputs[i]

                # Find EOS token
                eos_idx = np.where(tokens == vocabularies.DECODED_EOS_ID)[0]
                if len(eos_idx) > 0:
                    tokens = tokens[:eos_idx[0]]

                # Get start time for this chunk
                start_time = self._last_frame_times[i][0]
                start_time -= start_time % (1 / self._codec.steps_per_second)

                predictions.append({
                    'est_tokens': tokens,
                    'start_time': start_time,
                    'raw_inputs': []
                })

            # Decode to note sequence
            result = metrics_utils.event_predictions_to_ns(
                predictions,
                codec=self._codec,
                encoding_spec=self._encoding_spec
            )

            note_sequence = result['est_ns']

            # Convert to MIDI
            midi = self._note_sequence_to_midi(note_sequence)

            # Apply automatic filtering for drum-heavy content if enabled
            if self.auto_filter:
                midi = self._apply_auto_filtering(midi)

            return midi

        except ImportError as e:
            raise FrameworkError(f"Failed to import mt3_pytorch modules: {e}") from e

    def _audio_to_frames(self, audio: np.ndarray):
        """Split audio into frames."""
        frame_size = self._spectrogram_processor.config.hop_width
        remainder = len(audio) % frame_size
        padding = frame_size - remainder if remainder != 0 else frame_size
        audio = np.pad(audio, (0, padding), mode='constant')

        frames = self._spectrogram_processor.split_audio(audio)
        num_frames = len(audio) // frame_size
        times = np.arange(num_frames) / self._spectrogram_processor.config.frames_per_second

        return frames, times

    def _split_into_chunks(
        self,
        frames: np.ndarray,
        frame_times: np.ndarray,
        max_length: int = 256
    ):
        """Split frames into fixed-length chunks."""
        num_segments = math.ceil(frames.shape[0] / max_length)
        chunks = []
        time_chunks = []
        paddings = []

        for i in range(num_segments):
            start_idx = i * max_length
            end_idx = min(start_idx + max_length, frames.shape[0])
            actual_length = end_idx - start_idx

            # Create chunk with padding
            chunk = np.zeros((max_length, *frames.shape[1:]))
            time_chunk = np.zeros(max_length)

            chunk[:actual_length] = frames[start_idx:end_idx]
            time_chunk[:actual_length] = frame_times[start_idx:end_idx]

            chunks.append(chunk)
            time_chunks.append(time_chunk)
            paddings.append(actual_length)

        return np.array(chunks), np.array(time_chunks), paddings

    def _postprocess_batch(self, result: torch.Tensor) -> torch.Tensor:
        """Postprocess model outputs (remove BOS, handle EOS)."""
        # Mark tokens after EOS as invalid
        after_eos = torch.cumsum(
            (result == self._model.config.eos_token_id).float(), dim=-1
        )

        # Subtract special token offset
        result = result - self._vocab.num_special_tokens()

        # Mask tokens after EOS
        result = torch.where(after_eos.bool(), -1, result)

        # Remove BOS token
        result = result[:, 1:]

        return result

    def _apply_auto_filtering(self, midi: mido.MidiFile) -> mido.MidiFile:
        """
        Apply automatic filtering to remove instrument leakage in drum tracks.

        MT3-PyTorch has a known issue where it incorrectly assigns drum sounds
        to melodic instruments (particularly bass and synth lead). This method
        automatically detects and filters such cases.

        Args:
            midi: Input MIDI file

        Returns:
            Filtered MIDI file
        """
        # Count notes by instrument type
        drum_notes = 0
        bass_notes = 0  # Programs 32-39 (bass instruments)
        chromatic_notes = 0  # Programs 8-15 (chromatic percussion)
        other_notes = 0

        program_by_channel = {}

        for track in midi.tracks:
            for msg in track:
                # Track program changes
                if msg.type == 'program_change':
                    program_by_channel[msg.channel] = msg.program

                # Count notes
                elif msg.type == 'note_on' and msg.velocity > 0:
                    if hasattr(msg, 'channel'):
                        if msg.channel == 9:
                            drum_notes += 1
                        else:
                            program = program_by_channel.get(msg.channel, 0)
                            if 32 <= program <= 39:
                                bass_notes += 1
                            elif 8 <= program <= 15:
                                chromatic_notes += 1
                            else:
                                other_notes += 1

        total_notes = drum_notes + bass_notes + chromatic_notes + other_notes

        if total_notes == 0:
            return midi

        # Detect drum tracks with leakage patterns
        # Pattern 1: Drums with bass leakage (common)
        # Pattern 2: Chromatic percussion misclassification (FDNB case)
        # Pattern 3: Mostly drums (>60%)

        drum_ratio = drum_notes / total_notes
        bass_ratio = bass_notes / total_notes
        chromatic_ratio = chromatic_notes / total_notes

        # If we have drums + bass/chromatic (typical leakage pattern)
        if drum_notes > 0 and (bass_ratio > 0.2 or chromatic_ratio > 0.5):
            # This looks like drum leakage - filter to drums only
            return self._filter_drums_only(midi)

        # If mostly drums (>60%), filter out non-drums
        if drum_ratio > 0.6:
            return self._filter_drums_only(midi)

        # If chromatic percussion dominates (>70%), likely misclassified drums
        if chromatic_ratio > 0.7:
            # Convert to drums (this is a misclassification of drums)
            return self._remap_to_drums(midi)

        return midi

    def _filter_drums_only(self, midi: mido.MidiFile) -> mido.MidiFile:
        """
        Keep only MIDI channel 10 (drums), removing leaked melodic instruments.

        Args:
            midi: Input MIDI file

        Returns:
            MIDI file with only drum channel
        """
        filtered_mid = mido.MidiFile()
        filtered_mid.ticks_per_beat = midi.ticks_per_beat

        for track in midi.tracks:
            new_track = mido.MidiTrack()
            has_messages = False

            for msg in track:
                # Keep non-channel messages (tempo, time signature, etc.)
                if not hasattr(msg, 'channel'):
                    new_track.append(msg)
                    has_messages = True
                # Keep only drum channel (9 = channel 10 in MIDI)
                elif msg.channel == 9:
                    new_track.append(msg)
                    has_messages = True
                # Skip program changes and notes for non-drum channels
                elif msg.channel != 9 and msg.type in ['program_change', 'note_on', 'note_off']:
                    continue

            if has_messages:
                filtered_mid.tracks.append(new_track)

        return filtered_mid

    def _remap_to_drums(self, midi: mido.MidiFile) -> mido.MidiFile:
        """
        Remap all notes to drum channel when chromatic percussion dominates.

        This handles cases where drum sounds are misclassified as chromatic
        percussion instruments (e.g., vibraphone, marimba).

        Args:
            midi: Input MIDI file

        Returns:
            MIDI file with all notes remapped to drum channel
        """
        remapped = mido.MidiFile()
        remapped.ticks_per_beat = midi.ticks_per_beat

        for track_idx, track in enumerate(midi.tracks):
            new_track = mido.MidiTrack()

            for msg in track:
                # Skip program changes (we're remapping everything to drums)
                if msg.type == 'program_change':
                    continue

                # Remap all notes to drum channel
                elif msg.type in ['note_on', 'note_off'] and hasattr(msg, 'channel'):
                    # Change channel to 9 (drum channel)
                    new_msg = msg.copy(channel=9)
                    new_track.append(new_msg)
                else:
                    # Keep other messages as-is
                    new_track.append(msg)

            remapped.tracks.append(new_track)

        return remapped

    def _note_sequence_to_midi(self, note_sequence) -> mido.MidiFile:
        """Convert note-seq NoteSequence to mido.MidiFile."""
        import note_seq
        import tempfile

        # Write to temporary MIDI file
        with tempfile.NamedTemporaryFile(suffix='.mid', delete=False) as tmp:
            tmp_path = tmp.name

        note_seq.sequence_proto_to_midi_file(note_sequence, tmp_path)

        # Load with mido
        midi = mido.MidiFile(tmp_path)

        # Clean up
        Path(tmp_path).unlink()

        return midi
