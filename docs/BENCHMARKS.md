# Baseline Benchmark Plan

This document defines how we validate `mt3_infer` adapters against the original MT3-family repositories.

## Goals (REVISED POLICY)
- **Primary:** Validate mt3-infer adapters produce reasonable MIDI output
- **Secondary:** Compare against upstream implementations ONLY if adapter results look weird
- **Rationale:** Baseline setup has dependency conflicts (TensorFlow/LLVM). Adapter-first approach avoids blockers and gets working MT3 faster.

This document describes baseline setup for **validation purposes** (not required upfront).

## Shared Audio Assets
All reference audio clips live in `assets/` inside this repo. Each clip should be short (<30s) and cover a specific musical texture.

| File | Length | Use Case | Notes |
|------|--------|----------|-------|
| `assets/HappySounds_120bpm_Drums_drum_120BPM_BANDLAB.wav` | ≈8s | Percussive baseline | 120 BPM drum loop from BandLab "HappySounds" pack |

When adding new clips:
1. Place the file under `assets/` (uncompressed WAV preferred).
2. Update the table above with length, focus, and provenance.
3. Keep total asset size under ~10 MB to avoid bloating the repo.

## Per-Repository Virtual Environments
Create isolated `uv` environments inside `refs/<repo>/` only when you need to reproduce upstream behaviour.

```bash
cd refs/mr-mt3
uv venv .venv
uv sync  # run only when comparison is required
```

Guidelines:
- Never commit `.venv/` directories; they are local-only.
- Record any non-default install commands in `refs/<repo>/BENCHMARK.md` when you do set them up.
- For repos that rely on Git LFS checkpoints, document manual download steps the first time you pull them.

## Upstream Runner Scripts
Each reference repo should expose a script that can be invoked from the project root to generate baselines:

```
refs/<repo>/scripts/run_baseline.py --audio ../../assets/HappySounds_120bpm...wav --output ../outputs
```

The script responsibilities:
1. Activate the repo-specific `uv` environment (`uv run ...`).
2. Load the repo's inference pipeline and run on the provided WAV.
3. Save outputs to `refs/<repo>/outputs/<clip>.mid` and a summary JSON containing:
   - SHA256 hash of the MIDI file
   - Note count
   - Total duration (seconds)
   - Any backend-specific metadata (e.g., beam settings, temperature)

## Comparison Harness (mt3_infer)
During Phase 6 we can add a runner under `scripts/` or `tests/benchmarks/` that, when triggered,
1. Regenerates upstream baselines for the clips under investigation.
2. Runs the corresponding `mt3_infer` adapter on the same audio clips.
3. Produces a diff report summarising hash/note/duration differences.
4. Surfaces discrepancies without blocking normal development when everything looks good.

When stored, keep outputs in `tests/baselines/` (hash JSON) and `tests/fixtures/midi/` (reference MIDIs); regenerate only when behaviour changes.

## Open Questions
- How many additional audio clips do we need to cover melodic and polyphonic content?
- Where should we document any manual checkpoint download steps for Magenta/YourMT3? (Likely alongside the runner script instructions.)
- Should benchmark scripts run automatically in CI, or only on-demand?

## Suggested When Needed
1. Provision `uv` environments for `refs/mr-mt3` and `refs/mt3-pytorch` if an adapter output needs upstream comparison.
2. Draft/run baseline scripts to reproduce the specific scenario.
3. Capture hashes/notes/durations and store in `refs/<repo>/outputs/` + BENCHMARK.md.
4. Expand the audio asset set once multiple adapters produce high-fidelity MIDI.
