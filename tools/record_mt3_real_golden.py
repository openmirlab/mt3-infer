"""Capture a reviewable MIDI candidate from the public real-weight API.

Write to a scratch directory; compare the result with the committed original
before deliberately replacing any golden fixture or its manifest entry.
Reads: tests/golden/mt3/manifest.json, the tracked HappySounds audio,
mt3_infer.api and the caller's MT3_CHECKPOINT_DIR.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
import transformers

from mt3_infer import transcribe
from mt3_infer.api import _load_registry


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=("mr_mt3", "mt3_pytorch", "yourmt3"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cache = os.environ.get("MT3_CHECKPOINT_DIR")
    if not cache:
        parser.error("set MT3_CHECKPOINT_DIR to the local real-checkpoint cache")
    root = Path(__file__).parents[1]
    manifest = json.loads((root / "tests/golden/mt3/manifest.json").read_text())
    relative = Path(_load_registry()["models"][args.model]["checkpoint"]["path"])
    checkpoint = Path(cache) / relative.relative_to(".mt3_checkpoints")
    checkpoint_file = checkpoint if checkpoint.is_file() else checkpoint / "mt3.pth"
    source = root / manifest["source_audio"]
    audio, rate = sf.read(source, dtype="float32", always_2d=True)
    audio = librosa.resample(audio[: rate * 4].mean(axis=1), orig_sr=rate, target_sr=16000)
    torch.set_num_threads(4)
    midi = transcribe(audio, model=args.model, sr=16000, checkpoint_path=str(checkpoint),
                      device="cpu", auto_download=False)

    args.output.mkdir(parents=True, exist_ok=True)
    midi_path = args.output / f"{args.model}.mid"
    midi.save(midi_path)
    record = {
        "source_head_of_original_baseline": manifest["source_head"],
        "model": args.model,
        "source_sha256": sha256(source),
        "audio_float_sha256": hashlib.sha256(audio.tobytes()).hexdigest(),
        "checkpoint_sha256": sha256(checkpoint_file),
        "midi_sha256": sha256(midi_path),
        "messages": [{"track": index, "message": str(message)}
                     for index, track in enumerate(midi.tracks) for message in track],
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "numpy": np.__version__, "librosa": librosa.__version__,
                        "transformers": transformers.__version__, "device": "cpu", "torch_threads": 4},
    }
    (args.output / f"{args.model}.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"{args.model}: {len(record['messages'])} MIDI messages, SHA-256 {record['midi_sha256']}")


if __name__ == "__main__":
    main()
