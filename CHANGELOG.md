# Changelog

All notable changes to MT3-Infer will be documented in this file.

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