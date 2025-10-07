# MT3-Infer Progress Review

## TL;DR
- Phase 1 + 2 infrastructure is stable; smoke tests (34, 65% coverage) still guard the core.
- **Big win:** MR-MT3 adapter now runs end-to-end via PyTorch-only preprocessing and codec-based decoding (116 drum notes on HappySounds clip).
- Benchmark plan remains adapter-first: upstream baselines gathered only when behaviour is suspicious.
- Immediate focus shifts to finishing MR-MT3 decoding, wiring the registry/verification utilities, and deepening utility coverage.

## Evidence Collected
- `mt3_infer/adapters/mr_mt3.py` (323 lines) + `mt3_infer/adapters/vocab_utils.py` (codec helper) load checkpoints, preprocess with `torchaudio` MelSpectrogram, run the custom T5, and emit fully decoded MIDI.
- Manual run (see `PROGRESS.md`) shows HappySounds drum clip successfully producing a MIDI file via the new adapter.
- `docs/dev/TODO.md:180-229` documents the adapter progress (80% complete) and remaining decode tasks.
- `docs/dev/PLAN.md:243-266` + `docs/dev/BENCHMARKS.md:1-74` reflect the adapter-first benchmark policy.
- Coverage remains ~65% (utility coverage still ~44%); MR-MT3 decode now emits accurate drum MIDI.

## Status Notes
- ✅ MR-MT3 adapter fully functional (codec-based decoding) while still avoiding TensorFlow/DDSP dependencies.
- ✅ Smoke suite still green; documentation synced with policy shift and adapter progress.
- ⚠️ MR-MT3 automated tests still missing; need assertions around decoded MIDI quality before closing Phase 4.
- ⚠️ Registry loader/checkpoint verification utilities still outstanding; MT3-PyTorch adapter untouched.
- ⚠️ Utility coverage gaps (resampling + CUDA detection paths) remain.

## Immediate Follow-Up Items
- Add automated MR-MT3 adapter tests (smoke + decoded MIDI checks) and integrate into CI.
- Implement registry YAML loader and checkpoint verification helpers, integrating them with `load_model` workflow.
- Add integration-level tests covering `load_audio` resampling (with/without SciPy) and `get_device` GPU detection to raise coverage.
- Keep upstream baseline runs optional; document any investigations in `refs/<repo>/BENCHMARK.md` if behaviour regresses.

## Review Cadence & Exit Criteria
- Maintain weekly reviews; trigger a check-in once MR-MT3 decoding + registry utilities land.
- v0.1.0 exit still requires two PyTorch adapters, on-demand regression comparisons, ≥80% coverage, dependency alignment, and updated docs.

## Outstanding Questions
- Which sections of `contrib/vocabularies.py` / `note_sequences.py` map cleanly into our adapter without dragging heavy deps?
- Do we plan to snapshot upstream MIDI hashes after MR-MT3 decode stabilises, or defer until MT3-PyTorch is ready?
- What additional audio fixtures (melodic/polyphonic) should we introduce once decoding is complete?
