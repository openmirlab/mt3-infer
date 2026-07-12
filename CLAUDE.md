# CLAUDE.md

Guidance for Claude Code (or any agent) working in this repository. Follows
the [openmirlab org constitution](https://github.com/openmirlab/openmirlab-skills/blob/main/plugins/openmirlab/CLAUDE.md).

## What this is

`mt3_infer` is a unified, inference-only Python toolkit wrapping three
independent MT3 (music transcription) model implementations behind one
`MT3Base` interface (`mt3_infer/base.py`) and one public API
(`mt3_infer/api.py`: `transcribe()`, `load_model()`, `list_models()`,
`get_model_info()`, `download_model()`). Entry point: `mt3_infer/__init__.py`.

**Scope / status:** shipped on PyPI, all three PyTorch backends (`mr_mt3`,
`mt3_pytorch`, `yourmt3`) working. Magenta MT3 (JAX/Flax) is intentionally
not wrapped — see README's Scope section; the `jax` extra in
`pyproject.toml` is commented out, not a committed roadmap item. No
training path, no batch/streaming/ONNX API exist in this repo; if that
framing shows up anywhere (docs, comments, planning notes), it's aspirational
dev-phase bookkeeping, not a README-facing promise — keep it out of
README.md per the org constitution's no-roadmap-language rule.

```
mt3_infer/
  api.py, base.py, cli.py, exceptions.py   -- public surface
  adapters/                                -- one adapter per backend, implements MT3Base
    mr_mt3.py, mt3_pytorch.py, yourmt3.py
  models/
    mr_mt3/        -- 605 LOC, MIT, vendored from gudgud96/MR-MT3
    mt3_pytorch/    -- 2193 LOC, license-FROZEN (see below), vendored from kunato/mt3-pytorch
    yourmt3/        -- 17K+ LOC, Apache-2.0, vendored from mimbres/YourMT3 (the main worktree)
  config/checkpoints.yaml            -- model registry: paths, download sources, sha256, metadata
  config/external_integrations.yaml  -- provenance log (source repo/commit/license per model)
  tests/                              -- pytest, see "Testing" below
```

## Hard constraint: `models/mt3_pytorch/` is license-frozen

`kunato/mt3-pytorch` (the upstream this tree is vendored from) has **no
license declared**. Per project decision, the vendored code under
`mt3_infer/models/mt3_pytorch/` is kept exactly as extracted, unmodified --
no refactors, no "creative reuse," no restructuring, even for things that
would otherwise be an easy win (dead code, style, whatever). It is not
advertised as more than it is. See the root `LICENSE` file for the full
honesty note. `mt3_infer/adapters/mt3_pytorch.py` (the *adapter*, not the
vendored tree) is fair game -- exports, docstrings, error handling there can
be fixed like any other file.

`contrib/preprocessor.py` inside that tree was a dead, un-importable orphan
(broken deps, zero importers) and was removed as pure dead-code deletion,
not a modification of working vendored code.

## Backends at a glance

| Backend | LOC | License | Status |
|---|---|---|---|
| `mr_mt3` | 605 | MIT | Clean, actively maintained |
| `mt3_pytorch` | 2,193 | none upstream (frozen, see above) | Works; transformers v4.44+/v5 compat fixed (commit 83180b7) |
| `yourmt3` | ~10,400 (post-strip) | Apache-2.0 | Main worktree; see gotcha below |

**Gotcha: yourmt3 needs an older `transformers`.** As of this writing the
default install resolves `transformers>=4.35.0` (currently 4.57.x in a fresh
`uv sync`), which works for `mr_mt3` and `mt3_pytorch` but breaks
`yourmt3`'s vendored T5 forward pass (`cache_position` ends up `None` ->
`TypeError` in `transformers/models/t5/modeling_t5.py`). This is a
pre-existing gap, not something touched by the Lightning-shim work -- commit
83180b7 fixed the same class of transformers-API-drift issue for `mr_mt3`
and `mt3_pytorch` but never touched `yourmt3/model/t5mod.py`. Until someone
patches that, testing yourmt3 end-to-end needs an older pin, e.g.:
```bash
uv run --with "transformers==4.43.4" python your_script.py
```

**yourmt3 no longer needs `pytorch_lightning`/`lightning` at all.** Its
model class extends a vendored `LightningModuleShim`
(`mt3_infer/models/yourmt3/model/lightning_shim.py`) instead of
`pl.LightningModule` -- only the `nn.Module` contract and a `.device`
property were ever used for inference. The `full` extra no longer lists
`lightning`; it still carries `matplotlib`, `pyloudnorm`, `pyrubberband`
(the last is for YourMT3's adaptive-transcription mode).

**Checkpoint unpickling still needs one scoped module alias.** YourMT3
checkpoints were pickled while this repo had a flat `utils`/`model`/`config`
layout (before the `mt3_infer.models.yourmt3` nesting).
`hyper_parameters['task_manager']` unpickles as a
`utils.task_manager.TaskManager` instance, so loading needs `utils`
resolvable as a top-level module name for that one `torch.load()` call. See
`inference_loader.py`'s `_checkpoint_unpickling_module_alias()` -- it's a
context manager scoped to just that call (not a permanent `sys.modules`
hijack), and only `utils` is needed (verified against the real checkpoint;
`model`/`config`/`config.vocabulary`/`utils.task_manager` turned out to be
unnecessary once `utils` itself resolves).

## Testing

- `pytest mt3_infer/tests/` runs the fast unit/smoke suite (no checkpoint
  downloads, no GPU). A `network` pytest marker gates checkpoint-source
  liveness checks; they're excluded by default (`-m "not network"` in
  `pyproject.toml`'s addopts) and run explicitly with `-m network`.
- CI: `.github/workflows/test.yml` runs the suite on push/PR;
  `.github/workflows/publish.yml` has a `test` job that `publish` depends on
  via `needs: [test]`.
- **Accuracy is exact-equality, not "close enough."** MIDI note events are
  discrete (pitch/velocity/onset/offset ticks), so for any change touching a
  backend's model code, adapter, or checkpoint-loading path: capture a
  baseline (transcribe a short fixed clip, e.g.
  `assets/HappySounds_120bpm_Drums_drum_120BPM_BANDLAB.wav`, hash the
  resulting note events), make the change, re-run, and diff. A refactor that
  changes even one note is a regression, not a rounding error. `.mt3_checkpoints/`,
  `.env_checkpoints/`, and `checkpoints/` (all gitignored) already have all
  three backends' weights cached locally in this environment -- no download
  needed for baselining.

## Dev workflow

Uses `uv` (see `docs/dev/PRINCIPLES.md`). Use `uv run`, not bare `python`.
`uv sync --extra dev --extra full` gets you pytest + matplotlib/pyloudnorm/
pyrubberband; core dependencies alone are enough for `mr_mt3` and
`mt3_pytorch`. Package version is single-sourced in `mt3_infer/__about__.py`
(`pyproject.toml` reads it via `[tool.hatch.version]`) -- don't hand-edit a
version number in `pyproject.toml` or `mt3_infer/__init__.py` directly.
