# Changelog

All notable changes to MT3-Infer will be documented in this file.

## [0.2.0] - 2024-10-08

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

## [0.1.0] - 2024-10-01

### Initial Release
- Three production-ready models: MR-MT3, MT3-PyTorch, YourMT3
- Unified API with `transcribe()` and `load_model()` functions
- CLI tool with download, list, and transcribe commands
- Automatic checkpoint downloading
- MIDI synthesis capabilities
- Comprehensive documentation and examples
- Environment variable support for checkpoint directory (`MT3_CHECKPOINT_DIR`)
- Full PyTorch backend for all models