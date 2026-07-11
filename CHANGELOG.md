# Changelog

All notable changes to MT3-Infer will be documented in this file.

## [0.2.0] - 2026-07-11

### Changed
- **YourMT3 no longer depends on pytorch_lightning/lightning at runtime.** Its
  model class (`YourMT3`) now extends a vendored `LightningModuleShim`
  (`mt3_infer/models/yourmt3/model/lightning_shim.py`, ~55 lines) instead of
  `pl.LightningModule` -- inference only ever needed the `nn.Module` contract
  and a `.device` property, both of which the shim provides with identical
  semantics. Removed `lightning>=2.3.0` from the `full` extra accordingly;
  `matplotlib`, `pyloudnorm`, and `pyrubberband` remain (the latter is used by
  YourMT3's adaptive-transcription mode).
- **Checkpoint-unpickling module aliasing is now scoped, not global.**
  Loading a YourMT3 checkpoint used to permanently alias five top-level
  module names (`utils`, `model`, `config`, `config.vocabulary`,
  `utils.task_manager`) in `sys.modules` for the lifetime of the process, as
  soon as the loader module was imported. Only `utils` is actually needed
  (verified against the real checkpoint), and it's now aliased only around
  the single `torch.load()` call that needs it, then restored.
- Package version is now single-sourced: `mt3_infer/__about__.py` is the one
  place `__version__` lives; `pyproject.toml` reads it via
  `[tool.hatch.version]` and `mt3_infer/__init__.py` imports it.
- `MT3PyTorchAdapter` is now exported from `mt3_infer.adapters` (previously
  only reachable via `mt3_infer.adapters.mt3_pytorch`).

### Fixed
- `adapters/yourmt3.py`'s `load_model()` no longer wraps a missing-dependency
  `ModuleNotFoundError` as a misleading `CheckpointError` ("your checkpoint is
  bad") -- it now raises `FrameworkError` with an actionable message,
  matching the existing convention in `adapters/mt3_pytorch.py`.
- Corrected `mt3-pytorch`'s attribution across LICENSE, adapter docstrings,
  and `config/external_integrations.yaml`: it was citing
  `github.com/rlax59us/MT3-pytorch` with "License: To be verified"/"Apache
  2.0 (compatible...)", but the code has always used
  `github.com/kunato/mt3-pytorch`, which has no license declared upstream.
  The vendored code itself is unchanged; only the (previously false)
  attribution is fixed.
- Removed the root LICENSE's credit for a vendored Magenta MT3 that doesn't
  exist anywhere in this repo, and fixed stale `mt3_infer/vendor/yourmt3/`
  path references (actual location: `mt3_infer/models/yourmt3/`).
- Recorded verified SHA-256 checksums for all three checkpoints in
  `checkpoints.yaml` (mr_mt3's is enforced at download time; mt3_pytorch's
  and yourmt3's are recorded for provenance -- the git-lfs download path
  doesn't verify checksums yet). Added a `network`-marked liveness test for
  each checkpoint's download source (excluded from the default test run).

### Removed
- ~6,700 LOC of orphaned YourMT3 dataset/eval/preprocessing code with zero
  reachable importers from the public API: `utils/datasets_eval.py`,
  `utils/datasets_helper.py`, `utils/metrics.py`, `utils/metrics_helper.py`,
  `utils/mirdata_dev/`, `utils/preprocess/` (19 files), plus the now-dead
  `note_event2token2note_event_sanity_check()` in `utils/utils.py` and four
  unreachable embedded `test()` functions in `model/perceiver_mod.py`,
  `model/t5mod.py`, `model/ff_layer.py`, `model/conformer_mod.py`.
- `mt3_infer/models/mt3_pytorch/contrib/preprocessor.py`: dead, un-importable
  (missing `immutabledict`/`note_seq` deps) with zero importers. Deleting a
  dead file is not a modification of the license-frozen `models/mt3_pytorch/`
  tree -- everything else under that tree is untouched.

### Testing
- Added a CI test workflow (`.github/workflows/test.yml`) and a `test` job
  gating `publish` in `.github/workflows/publish.yml` -- previously no CI job
  ran the test suite at all.
- Filled the two empty test stub files (`test_mr_mt3.py`, `test_mt3_pytorch.py`)
  and added `test_yourmt3.py` and `test_checkpoint_registry.py`.
- mr_mt3, mt3_pytorch, and yourmt3 were each baselined end-to-end (exact MIDI
  event equality, verified via sha256 of the extracted note events) before
  and after every change in this pass; all three remained byte-identical
  throughout.

## [0.1.3] - 2026-01-14

### Added
- **Google Colab notebook** (`notebooks/quickstart_colab.ipynb`)
  - 7-section workflow: install, download model, upload audio, run transcription, check outputs, preview results, download results
  - Support for both CLI and Python API methods

### Changed
- **Relax dependency version constraints** for better compatibility
  - `transformers`: `~=4.30.2` → `>=4.35.0` (fixes tokenizers build error on Python 3.12)
  - `torch`: `==2.7.1` → `>=2.0.0` (allows Colab's pre-installed PyTorch)
  - `torchvision`: `==0.22.1` → `>=0.15.0`
  - `torchaudio`: `==2.7.1` → `>=2.0.0`
- **Update minimum Python version**: 3.8 → 3.9 (required by matplotlib>=3.8.0)

### Fixed
- **Google Colab compatibility**: Fixed "Failed building wheel for tokenizers" error
  - Old transformers required tokenizers<0.14 which lacked Python 3.12 wheels
  - New transformers uses tokenizers>=0.14 with pre-built wheels

## [0.1.2] - 2025-11-29

### Removed
- **Training-related code**: Removed all training code to make mt3-infer a pure inference-only library
  - Removed `wandb` dependency requirement (was already commented out)
  - Removed training methods from YourMT3 (`training_step`, `validation_step`, `test_step`, `configure_optimizers`, etc.)
  - Removed training-specific files: `init_train.py`, `model_helper.py`, `optimizers.py`, `lr_scheduler.py`, `datasets_train.py`, `augment.py`, `data_modules.py`
  - Removed unused `T5Adversarial` class from MR-MT3 (adversarial training code)
  - Cleaned up training-related parameters from YourMT3 `__init__` method

### Verified
- **MR-MT3 and MT3-PyTorch**: Confirmed both adapters are already inference-only with no training dependencies
  - Both models set to `.eval()` mode
  - No Lightning modules or training methods
  - No optimizers or schedulers

## [0.1.1] - 2025-11-14

### Fixed
- **Package distribution**: YAML configuration files (`checkpoints.yaml`, `external_integrations.yaml`) are now properly included in the built package
  - Fixed `FileNotFoundError` when calling `list_models()` or `get_model_info()` on installed packages
  - Updated `pyproject.toml` to explicitly include `mt3_infer/**/*.yaml` and `mt3_infer/**/*.yml` in build

## [0.2.0] - Planned

### Added
- **Automatic instrument leakage filtering for MT3-PyTorch**
  - New `auto_filter` parameter (default: `True`) for `MT3PyTorchAdapter`
  - Configurable via `load_model()` and `transcribe()` API functions
  - Detects and corrects drum tracks misclassified as melodic instruments
  - Comprehensive documentation in `docs/INSTRUMENT_LEAKAGE_INVESTIGATION.md`

### Fixed
- **MT3-PyTorch instrument leakage issue** - Drum tracks no longer incorrectly transcribed with bass and chromatic percussion instruments
  - HappySounds drums: Fixed from 38.5% leakage to 0%
  - FDNB drums: Fixed from 98.4% leakage to 0%

### Changed
- Updated `MT3PyTorchAdapter` to include filtering logic in `decode()` method
- API functions now accept `**kwargs` for model-specific parameters
- Default behavior provides clean drum transcriptions out of the box

### Documentation
- Added `docs/dev/INSTRUMENT_LEAKAGE_INVESTIGATION.md` - Complete investigation report
- Updated `README.md` with What's New section and auto_filter information
- Updated `CLAUDE.md` with latest project status and MT3-PyTorch filtering details
- Updated `docs/dev/TROUBLESHOOTING.md` with instrument leakage section

### Development
- Maintained backward compatibility - existing code continues to work

## [0.1.0] - 2025-11-14

### Initial Release
- Three production-ready models: MR-MT3, MT3-PyTorch, YourMT3
- Unified API with `transcribe()` and `load_model()` functions
- CLI tool with download, list, and transcribe commands
- Automatic checkpoint downloading
- MIDI synthesis capabilities
- Comprehensive documentation and examples
- Environment variable support for checkpoint directory (`MT3_CHECKPOINT_DIR`)
- Full PyTorch backend for all models