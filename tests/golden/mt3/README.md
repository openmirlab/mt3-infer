# MT3 real-checkpoint baseline

These MIDI files and `manifest.json` were captured from untouched default
commit `3675ad8` on a four-second excerpt of the tracked HappySounds drum WAV.
The CPU profile was Python 3.10.18, Torch 2.13.0+cu130, NumPy 2.2.6, and
librosa 0.11.0, with four Torch threads. `mr_mt3` and `mt3_pytorch` used
Transformers 5.13.1; `yourmt3` used 4.43.4 because the default 5.13.1
environment fails before inference. YourMT3's reference MIDI was reproduced
twice with the same SHA-256. Each checkpoint hash is recorded in the manifest.

To replay without downloading, point to the three-checkpoint cache and opt in:

```bash
MT3_CHECKPOINT_DIR=/path/to/.mt3_checkpoints MT3_REAL_GOLDEN=1 \
  python -m pytest tests/test_real_golden.py -q
```

The recorder writes a candidate and its full provenance outside the repo:

```bash
MT3_CHECKPOINT_DIR=/path/to/.mt3_checkpoints \
  python -m tools.record_mt3_real_golden mr_mt3 --output /tmp/mt3-candidate
```

Repeat for `mt3_pytorch` and `yourmt3`. Run YourMT3 under Transformers
4.43.4 when recreating the original baseline; compare a newer-version repair
against that same MIDI. Review each candidate and its JSON against the
committed source/head and manifest before replacing any fixture. The original
fixture stays immutable during compatibility work.
