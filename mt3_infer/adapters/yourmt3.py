"""
YourMT3 Adapter for MT3-Infer

Minimal inference-only adapter for YourMT3 (PyTorch + Lightning).
Uses vendored YourMT3 code for self-contained distribution.

Original Authors: Taegyun Kwon, et al.
Original Repository: https://huggingface.co/spaces/mimbres/YourMT3
License: Apache 2.0

This adapter vendors the YourMT3 code in mt3_infer/vendor/yourmt3/
for easy installation via PyPI/uv without external dependencies.
"""
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import mido
import numpy as np
import torch

from mt3_infer.base import MT3Base
from mt3_infer.exceptions import CheckpointError, InferenceError, ModelNotFoundError

# Checkpoint configurations from app.py
CHECKPOINT_CONFIGS = {
    "ymt3plus": {
        "name": "YMT3+",
        "checkpoint": "notask_all_cross_v6_xk2_amp0811_gm_ext_plus_nops_b72@model.ckpt",
        "args": [],
        "description": "Base model, no pitch shift (518MB)"
    },
    "yptf_single": {
        "name": "YPTF+Single (noPS)",
        "checkpoint": "ptf_all_cross_rebal5_mirst_xk2_edr005_attend_c_full_plus_b100@model.ckpt",
        "args": ['-enc', 'perceiver-tf', '-ac', 'spec', '-hop', '300', '-atc', '1'],
        "description": "PerceiverTF encoder, single-track (345MB)"
    },
    "yptf_multi": {
        "name": "YPTF+Multi (PS)",
        "checkpoint": "mc13_256_all_cross_v6_xk5_amp0811_edr005_attend_c_full_plus_2psn_nl26_sb_b26r_800k@model.ckpt",
        "args": ['-tk', 'mc13_full_plus_256', '-dec', 'multi-t5', '-nl', '26',
                 '-enc', 'perceiver-tf', '-ac', 'spec', '-hop', '300', '-atc', '1'],
        "description": "Multi-track with pitch shift (517MB)"
    },
    "yptf_moe_nops": {
        "name": "YPTF.MoE+Multi (noPS)",
        "checkpoint": "mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b36_nops@last.ckpt",
        "args": ['-tk', 'mc13_full_plus_256', '-dec', 'multi-t5', '-nl', '26',
                 '-enc', 'perceiver-tf', '-sqr', '1', '-ff', 'moe', '-wf', '4',
                 '-nmoe', '8', '-kmoe', '2', '-act', 'silu', '-epe', 'rope',
                 '-rp', '1', '-ac', 'spec', '-hop', '300', '-atc', '1'],
        "description": "Mixture of Experts, no pitch shift (536MB)"
    },
    "yptf_moe_ps": {
        "name": "YPTF.MoE+Multi (PS)",
        "checkpoint": "mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b80_ps2@model.ckpt",
        "args": ['-tk', 'mc13_full_plus_256', '-dec', 'multi-t5', '-nl', '26',
                 '-enc', 'perceiver-tf', '-sqr', '1', '-ff', 'moe', '-wf', '4',
                 '-nmoe', '8', '-kmoe', '2', '-act', 'silu', '-epe', 'rope',
                 '-rp', '1', '-ac', 'spec', '-hop', '300', '-atc', '1'],
        "description": "Mixture of Experts with pitch shift (724MB)"
    }
}


class YourMT3Adapter(MT3Base):
    """
    Minimal inference-only adapter for YourMT3.

    Wraps the upstream YourMT3 PyTorch Lightning module for inference.
    Strips out training code and provides MT3Base interface.

    Args:
        model_key: One of the checkpoint keys: ymt3plus, yptf_single, yptf_multi,
                   yptf_moe_nops, yptf_moe_ps. Defaults to the public MoE model
                   (`yptf_moe_nops`).
        verbose: Enable verbose output for adaptive transcription
    """

    def __init__(self, model_key: str | None = None, verbose: bool = False):
        super().__init__()

        self.model = None
        self.device_str = "cpu"
        self.verbose = verbose
        self._set_model_key(model_key)

    def _set_model_key(self, model_key: Optional[str]) -> None:
        resolved_key = model_key or "yptf_moe_nops"
        if resolved_key not in CHECKPOINT_CONFIGS:
            available = ", ".join(CHECKPOINT_CONFIGS.keys())
            raise ModelNotFoundError(
                f"Unknown YourMT3 model key '{resolved_key}'. "
                f"Available: {available}"
            )
        self.model_key = resolved_key
        self.config = CHECKPOINT_CONFIGS[resolved_key]

    def _checkpoint_components(self, key: str) -> Tuple[str, str]:
        checkpoint_spec = CHECKPOINT_CONFIGS[key]["checkpoint"]
        if "@" in checkpoint_spec:
            return tuple(checkpoint_spec.split("@", 1))  # (directory, filename)
        return checkpoint_spec, ""

    def _infer_model_key_from_checkpoint(self, checkpoint_path: Path) -> str | None:
        checkpoint_str = str(checkpoint_path)
        for key, cfg in CHECKPOINT_CONFIGS.items():
            directory, filename = self._checkpoint_components(key)
            if directory and directory in checkpoint_str:
                return key
            if filename and filename in checkpoint_path.name:
                return key
        return None

    def _resolve_task_name(self) -> str:
        args = self.config.get("args", [])
        if "-tk" in args:
            idx = args.index("-tk")
            if idx + 1 < len(args):
                return args[idx + 1]
        return "mt3_full_plus"

    def load_model(
        self,
        checkpoint_path: str | None = None,
        device: str = "auto"
    ) -> None:
        """
        Load YourMT3 model checkpoint using inference-only loader.

        Args:
            checkpoint_path: Path to .ckpt checkpoint file. If None, looks for default
                           checkpoint in checkpoints/ directory.
            device: Device to load model on ('auto', 'cuda', 'cpu')
        """
        import sys

        try:
            from mt3_infer.models.yourmt3.inference_loader import load_model_for_inference

            # Determine checkpoint path
            if checkpoint_path is None:
                checkpoint_name, checkpoint_file = self._checkpoint_components(self.model_key)

                checkpoints_dir = Path(__file__).parent.parent.parent / "checkpoints" / "yourmt3"
                checkpoint_path = checkpoints_dir / checkpoint_name / checkpoint_file

                if not checkpoint_path.exists():
                    refs_checkpoint = (
                        Path(__file__).parent.parent.parent
                        / "refs"
                        / "yourmt3"
                        / "amt"
                        / "logs"
                        / "2024"
                        / checkpoint_name
                        / "checkpoints"
                        / checkpoint_file
                    )
                    if refs_checkpoint.exists():
                        checkpoint_path = refs_checkpoint
                    else:
                        raise CheckpointError(
                            f"Checkpoint not found. Tried:\n"
                            f"  1. {checkpoints_dir / checkpoint_name / checkpoint_file}\n"
                            f"  2. {refs_checkpoint}\n"
                            f"Please download YourMT3 checkpoints or provide checkpoint_path."
                        )

            checkpoint_path = Path(checkpoint_path)

            # Update model key if checkpoint points to a different variant
            inferred_key = self._infer_model_key_from_checkpoint(checkpoint_path)
            if inferred_key is not None and inferred_key != self.model_key:
                self._set_model_key(inferred_key)

            # Determine device
            if device == "auto":
                self.device_str = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self.device_str = device

            # Load model using inference-only loader
            print(f"Loading YourMT3 model: {self.config['name']}")
            print(f"Checkpoint: {checkpoint_path}")
            print(f"Device: {self.device_str}")

            task_name = self._resolve_task_name()

            self.model = load_model_for_inference(
                checkpoint_path=str(checkpoint_path),
                device=self.device_str,
                task_name=task_name
            )

            # Verify model is loaded
            if self.model is None:
                raise CheckpointError("Model is None after loading!")

            # Set loaded flag for MT3Base
            self._model_loaded = True

            print(f"Model loaded successfully! (type: {type(self.model).__name__})")

        except Exception as e:
            self.model = None  # Reset on error
            self._model_loaded = False
            raise CheckpointError(f"Failed to load YourMT3 checkpoint: {e}") from e

    @torch.inference_mode()
    def preprocess(
        self,
        audio: np.ndarray,
        sr: int
    ) -> torch.Tensor:
        """
        Preprocess audio for YourMT3 model.

        Converts audio to torch tensor and segments it according to model's
        input_frames configuration.

        Args:
            audio: Audio array (mono, float32)
            sr: Sample rate

        Returns:
            Audio segments tensor (n_segments, 1, segment_length)
        """
        import sys

        import torchaudio
        from mt3_infer.models.yourmt3.utils.audio import slice_padded_array

        # Convert to torch tensor
        audio_tensor = torch.from_numpy(audio.astype('float32')).unsqueeze(0)  # (1, n_samples)

        # Resample to model's sample rate
        target_sr = self.model.audio_cfg['sample_rate']
        if sr != target_sr:
            audio_tensor = torchaudio.functional.resample(audio_tensor, sr, target_sr)

        # Segment audio
        input_frames = self.model.audio_cfg['input_frames']
        audio_segments = slice_padded_array(
            audio_tensor.numpy(),
            input_frames,
            input_frames
        )  # (n_segments, segment_length)

        # Convert to (n_segments, 1, segment_length) format
        audio_segments = torch.from_numpy(audio_segments.astype('float32')).unsqueeze(1)

        return audio_segments

    @torch.inference_mode()
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Run inference on audio segments.

        Args:
            features: Audio segments (n_segments, 1, segment_length)

        Returns:
            Token predictions array (list of batch predictions)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Move to device
        features = features.to(self.device_str)

        # Run inference
        try:
            bsz = 8  # Batch size for inference
            print(f"Running inference on {features.shape[0]} segments...")
            result = self.model.inference_file(
                bsz=bsz,
                audio_segments=features
            )
            print(f"Inference result type: {type(result)}")
            if isinstance(result, tuple):
                pred_token_arr, loss = result
                print(f"Got {len(pred_token_arr)} prediction batches")
            else:
                pred_token_arr = result
            return pred_token_arr

        except Exception as e:
            import traceback
            traceback.print_exc()
            raise InferenceError(f"YourMT3 inference failed: {e}") from e

    def decode(self, outputs: Any) -> mido.MidiFile:
        """
        Decode model outputs to MIDI file.

        Args:
            outputs: Token predictions from forward()

        Returns:
            MIDI file object
        """
        import sys
        import tempfile
        from collections import Counter
        from mt3_infer.models.yourmt3.utils.event2note import merge_zipped_note_events_and_ties_to_notes
        from mt3_infer.models.yourmt3.utils.note2event import mix_notes
        from mt3_infer.models.yourmt3.utils.utils import write_model_output_as_midi

        pred_token_arr = outputs

        # Calculate start times for each segment
        # pred_token_arr is a list of batches, each batch is (B, C, L)
        # We need to flatten to get the total number of segments
        total_segments = sum(arr.shape[0] for arr in pred_token_arr)
        input_frames = self.model.audio_cfg['input_frames']
        sample_rate = self.model.audio_cfg['sample_rate']
        start_secs_file = [input_frames * i / sample_rate for i in range(total_segments)]

        # Detokenize predictions for each channel
        num_channels = self.model.task_manager.num_decoding_channels
        pred_notes_in_file = []
        n_err_cnt = Counter()

        for ch in range(num_channels):
            # Extract channel predictions
            pred_token_arr_ch = [arr[:, ch, :] for arr in pred_token_arr]  # (B, L)

            # Detokenize
            zipped_note_events_and_tie, list_events, ne_err_cnt = \
                self.model.task_manager.detokenize_list_batches(
                    pred_token_arr_ch,
                    start_secs_file,
                    return_events=True
                )

            # Convert to notes
            pred_notes_ch, n_err_cnt_ch = merge_zipped_note_events_and_ties_to_notes(
                zipped_note_events_and_tie
            )
            pred_notes_in_file.append(pred_notes_ch)
            n_err_cnt += n_err_cnt_ch

        # Mix notes from all channels
        pred_notes = mix_notes(pred_notes_in_file)

        # Write to temporary MIDI file
        with tempfile.TemporaryDirectory() as tmpdir:
            track_name = "yourmt3_output"
            write_model_output_as_midi(
                pred_notes,
                tmpdir,
                track_name,
                self.model.midi_output_inverse_vocab
            )

            midi_path = Path(tmpdir) / "model_output" / f"{track_name}.mid"
            if not midi_path.exists():
                raise InferenceError(f"MIDI file not created at {midi_path}")

            # Load and return MIDI file
            midi_file = mido.MidiFile(str(midi_path))

        return midi_file

    def transcribe(
        self,
        audio: np.ndarray,
        sr: int = 16000,
        adaptive: bool = False,
        stretch_factors: Optional[List[float]] = None,
        stretch_method: str = "auto",
        num_attempts: int = 1,
        return_all: bool = False
    ) -> Union[mido.MidiFile, List[Tuple[mido.MidiFile, int, float]]]:
        """
        Transcribe audio to MIDI with optional adaptive preprocessing.

        YourMT3 has known variability on dense drum patterns. Use adaptive=True
        to enable automatic preprocessing and multi-attempt strategy for more
        reliable results.

        Args:
            audio: Audio waveform (n_samples,), float32, range [-1.0, 1.0]
            sr: Sample rate in Hz
            adaptive: Enable adaptive preprocessing for dense patterns
            stretch_factors: List of time-stretch factors to try (e.g., [1.0, 0.9, 0.7])
                           If None, uses [1.0, 0.9, 0.8, 0.7] for dense patterns
            stretch_method: Time-stretch method:
                           "librosa" - phase vocoder (faster)
                           "pyrubberband" - Rubber Band library (better quality)
                           "auto" - automatically choose best method
            num_attempts: Number of transcription attempts per configuration
            return_all: If True, return all results for manual selection

        Returns:
            If return_all=False: Best MIDI transcription
            If return_all=True: List of (midi, note_count, stretch_factor) tuples

        Example:
            >>> # Normal transcription
            >>> midi = model.transcribe(audio, sr=16000)

            >>> # Adaptive mode for dense drums
            >>> midi = model.transcribe(
            ...     audio, sr=16000,
            ...     adaptive=True,
            ...     num_attempts=3
            ... )

            >>> # Manual selection of best result
            >>> results = model.transcribe(
            ...     audio, sr=16000,
            ...     adaptive=True,
            ...     return_all=True
            ... )
            >>> for midi, notes, factor in results:
            ...     print(f"{notes} notes at {factor}x stretch")
        """
        if not adaptive:
            # Standard transcription using base implementation
            return super().transcribe(audio, sr)

        # Import preprocessing utilities
        try:
            import librosa
            HAS_LIBROSA = True
        except ImportError:
            HAS_LIBROSA = False

        try:
            import pyrubberband as pyrb
            HAS_PYRUBBERBAND = True
        except ImportError:
            HAS_PYRUBBERBAND = False

        # Determine stretch method
        if stretch_method == "auto":
            use_method = "pyrubberband" if HAS_PYRUBBERBAND else "librosa"
        else:
            use_method = stretch_method

        if use_method == "pyrubberband" and not HAS_PYRUBBERBAND:
            import warnings
            warnings.warn("pyrubberband not available, falling back to librosa")
            use_method = "librosa"

        if use_method == "librosa" and not HAS_LIBROSA:
            raise ImportError("librosa required for adaptive preprocessing. Install with: uv add librosa")

        # Analyze audio density
        analysis = self._analyze_audio_density(audio, sr)

        # Determine stretch factors to try
        if stretch_factors is None:
            if analysis['is_dense']:
                # Dense pattern: try multiple factors
                stretch_factors = [1.0, 0.9, 0.8, 0.7]
            else:
                # Normal pattern: just use original
                stretch_factors = [1.0]

        if self.verbose:
            print(f"\nAdaptive transcription mode:")
            print(f"  Onset density: {analysis['onset_density']:.2f} onsets/sec")
            print(f"  Pattern type: {'DENSE' if analysis['is_dense'] else 'normal'}")
            print(f"  Stretch factors: {stretch_factors}")
            print(f"  Method: {use_method}")
            print(f"  Attempts per config: {num_attempts}")

        # Try multiple configurations
        all_results = []

        for stretch_factor in stretch_factors:
            # Prepare audio
            if stretch_factor == 1.0:
                processed_audio = audio
                config_name = "original"
            else:
                # Apply time-stretching
                if use_method == "pyrubberband":
                    processed_audio = pyrb.time_stretch(audio, sr, rate=stretch_factor)
                else:  # librosa
                    processed_audio = librosa.effects.time_stretch(
                        audio, rate=1.0/stretch_factor
                    )
                config_name = f"{use_method}_{stretch_factor:.2f}x"

            # Multiple attempts for this configuration
            for attempt in range(num_attempts):
                try:
                    # Transcribe using base implementation
                    midi = super().transcribe(processed_audio, sr)

                    # Count notes
                    note_count = self._count_notes(midi)

                    # Restore timing if stretched
                    if stretch_factor != 1.0:
                        midi = self._restore_timing(midi, stretch_factor)

                    all_results.append((midi, note_count, stretch_factor, config_name))

                    if self.verbose:
                        print(f"  {config_name} attempt {attempt+1}: {note_count} notes")

                except Exception as e:
                    if self.verbose:
                        print(f"  {config_name} attempt {attempt+1}: Error - {e}")
                    continue

        if not all_results:
            raise InferenceError("All adaptive transcription attempts failed")

        # Return results
        if return_all:
            return [(midi, notes, factor) for midi, notes, factor, _ in all_results]

        # Select best result (most notes)
        best_midi, best_notes, best_factor, best_config = max(all_results, key=lambda x: x[1])

        if self.verbose:
            print(f"\n✅ Best result: {best_config} with {best_notes} notes")

        return best_midi

    def _analyze_audio_density(self, audio: np.ndarray, sr: int) -> dict:
        """Analyze audio for dense pattern detection."""
        try:
            import librosa
        except ImportError:
            # Fallback: assume not dense if can't analyze
            return {
                'onset_density': 0.0,
                'tempo': 120.0,
                'is_dense': False
            }

        # Detect onsets
        onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
        onset_frames = librosa.onset.onset_detect(
            onset_envelope=onset_env,
            sr=sr,
            units='frames',
            hop_length=512,
            backtrack=False
        )

        duration = len(audio) / sr
        onset_density = len(onset_frames) / duration if duration > 0 else 0

        # Estimate tempo
        try:
            tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, trim=False)
            tempo = float(tempo) if tempo else 120.0
        except:
            tempo = 120.0

        # Determine if dense
        is_dense = (
            onset_density > 6.0 or  # High onset density
            (tempo > 150 and onset_density > 5) or  # Fast tempo
            (tempo > 160)  # Very fast tempo
        )

        return {
            'onset_density': onset_density,
            'tempo': tempo,
            'is_dense': is_dense
        }

    def _count_notes(self, midi: mido.MidiFile) -> int:
        """Count total notes in MIDI file."""
        note_count = 0
        for track in midi.tracks:
            for msg in track:
                if msg.type == 'note_on' and msg.velocity > 0:
                    note_count += 1
        return note_count

    def _restore_timing(self, midi: mido.MidiFile, stretch_factor: float) -> mido.MidiFile:
        """Restore MIDI timing to original tempo after time-stretched transcription."""
        if stretch_factor == 1.0:
            return midi

        # Create new MIDI with scaled timing
        restored = mido.MidiFile(ticks_per_beat=midi.ticks_per_beat)

        for track in midi.tracks:
            new_track = mido.MidiTrack()
            for msg in track:
                if msg.is_meta or hasattr(msg, 'time'):
                    # Scale delta time back to original
                    new_msg = msg.copy(time=int(msg.time * stretch_factor))
                    new_track.append(new_msg)
                else:
                    new_track.append(msg)
            restored.tracks.append(new_track)

        return restored

    @classmethod
    def list_available_models(cls) -> dict[str, str]:
        """List all available YourMT3 model configurations."""
        return {
            key: config["description"]
            for key, config in CHECKPOINT_CONFIGS.items()
        }
