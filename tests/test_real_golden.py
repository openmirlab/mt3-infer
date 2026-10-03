"""Replay the untouched three-backend transcription baseline with real weights.

The fixture records the original default branch before any compatibility or
inference-boundary edits. It is opt-in because the checkpoints total ~886 MB.
Reads: assets/HappySounds_120bpm_Drums_drum_120BPM_BANDLAB.wav,
tests/golden/mt3/manifest.json, mt3_infer.api, the model checkpoint cache.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import platform
from pathlib import Path

import librosa
import numpy as np
import pytest
import soundfile as sf
import torch

from mt3_infer import transcribe
from mt3_infer.api import _load_registry


_ROOT = Path(__file__).parents[1]
_GOLDEN = Path(__file__).parent / "golden" / "mt3"
_MANIFEST = json.loads((_GOLDEN / "manifest.json").read_text())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@pytest.mark.real_weights
@pytest.mark.parametrize("model", ["mr_mt3", "mt3_pytorch", "yourmt3"])
def test_original_real_checkpoint_midi(model: str) -> None:
    if os.environ.get("MT3_REAL_GOLDEN") != "1":
        pytest.skip("set MT3_REAL_GOLDEN=1 and MT3_CHECKPOINT_DIR to run real weights")

    profile = _MANIFEST["profile"]
    current = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "librosa": librosa.__version__,
    }
    mismatch = {key: (profile[key], value) for key, value in current.items() if profile[key] != value}
    if mismatch:
        pytest.skip(f"recorded CPU float/MIDI profile differs: {mismatch}")

    cache = os.environ.get("MT3_CHECKPOINT_DIR")
    if not cache:
        pytest.fail("MT3_CHECKPOINT_DIR is required with MT3_REAL_GOLDEN=1")
    relative = Path(_load_registry()["models"][model]["checkpoint"]["path"])
    checkpoint = Path(cache) / relative.relative_to(".mt3_checkpoints")
    checkpoint_file = checkpoint if checkpoint.is_file() else checkpoint / "mt3.pth"
    assert _sha256(checkpoint_file) == _MANIFEST["models"][model]["checkpoint_sha256"]

    source = _ROOT / _MANIFEST["source_audio"]
    assert _sha256(source) == _MANIFEST["source_sha256"]
    waveform, source_rate = sf.read(source, dtype="float32", always_2d=True)
    waveform = waveform[: source_rate * 4].mean(axis=1)
    waveform = librosa.resample(waveform, orig_sr=source_rate, target_sr=16000)
    assert waveform.shape == (_MANIFEST["audio_samples"],)
    assert hashlib.sha256(waveform.tobytes()).hexdigest() == _MANIFEST["audio_float_sha256"]

    prior_threads = torch.get_num_threads()
    torch.set_num_threads(profile["torch_threads"])
    try:
        midi = transcribe(
            waveform, model=model, sr=16000, checkpoint_path=str(checkpoint),
            device="cpu", auto_download=False,
        )
    finally:
        torch.set_num_threads(prior_threads)

    messages = [
        {"track": track_number, "message": str(message)}
        for track_number, track in enumerate(midi.tracks)
        for message in track
    ]
    assert messages == _MANIFEST["models"][model]["messages"]
    stream = io.BytesIO()
    midi.save(file=stream)
    assert hashlib.sha256(stream.getvalue()).hexdigest() == _MANIFEST["models"][model]["midi_sha256"]
