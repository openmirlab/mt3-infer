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
from typing import Any

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
    """

    def __init__(self, model_key: str | None = None):
        super().__init__()

        self.model = None
        self.device_str = "cpu"
        self._set_model_key(model_key)

    def _set_model_key(self, model_key: str | None) -> None:
        resolved_key = model_key or "yptf_moe_nops"
        if resolved_key not in CHECKPOINT_CONFIGS:
            available = ", ".join(CHECKPOINT_CONFIGS.keys())
            raise ModelNotFoundError(
                f"Unknown YourMT3 model key '{resolved_key}'. "
                f"Available: {available}"
            )
        self.model_key = resolved_key
        self.config = CHECKPOINT_CONFIGS[resolved_key]

    def _checkpoint_components(self, key: str) -> tuple[str, str]:
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

        vendor_root = Path(__file__).parent.parent / "vendor" / "yourmt3"
        sys.path.insert(0, str(vendor_root))

        try:
            from inference_loader import load_model_for_inference

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
        finally:
            # Always remove vendor path from sys.path after loading
            if str(vendor_root) in sys.path:
                sys.path.remove(str(vendor_root))

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

        # Temporarily add vendor path for imports
        vendor_root = Path(__file__).parent.parent / "vendor" / "yourmt3"
        sys.path.insert(0, str(vendor_root))
        try:
            from utils.audio import slice_padded_array
        finally:
            if str(vendor_root) in sys.path:
                sys.path.remove(str(vendor_root))

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

        # Temporarily add vendor path for imports
        vendor_root = Path(__file__).parent.parent / "vendor" / "yourmt3"
        sys.path.insert(0, str(vendor_root))
        try:
            from utils.event2note import merge_zipped_note_events_and_ties_to_notes
            from utils.note2event import mix_notes
            from utils.utils import write_model_output_as_midi
        finally:
            if str(vendor_root) in sys.path:
                sys.path.remove(str(vendor_root))

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

    @classmethod
    def list_available_models(cls) -> dict[str, str]:
        """List all available YourMT3 model configurations."""
        return {
            key: config["description"]
            for key, config in CHECKPOINT_CONFIGS.items()
        }
