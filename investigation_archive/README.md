# Investigation Archive

This directory contains scripts and outputs from various investigations conducted during the development of MT3-Infer, particularly the instrument leakage investigation for MT3-PyTorch.

## Contents

### Investigation Scripts

These scripts were used to identify and fix the MT3-PyTorch instrument leakage issue:

- **`debug_mt3_pytorch_decoding.py`** - Step-by-step token debugging to trace the source of leakage
- **`investigate_instrument_leakage.py`** - Initial analysis of instrument distribution in MIDI outputs
- **`test_all_files_leakage.py`** - Comprehensive testing across multiple audio files
- **`fix_instrument_leakage.py`** - Development of filtering strategies
- **`mt3_pytorch_patched.py`** - Prototype implementation of the patched adapter
- **`test_mt3_pytorch_fix.py`** - Validation of the filtering solution
- **`test_filter_parameter.py`** - Testing the auto_filter parameter
- **`test_api_auto_filter.py`** - API integration testing
- **`verify_auto_filter_integration.py`** - Final comprehensive verification

### Other Investigation Scripts

Scripts from related investigations:

- **`debug_transcription.py`** - General transcription debugging
- **`fix_loudness_normalization.py`** - Investigation into loudness normalization
- **`fix_with_pyloudnorm.py`** - Testing pyloudnorm for audio normalization
- **`test_all_files.py`** - Testing all models on all available audio files
- **`test_simple.py`** - Simple transcription tests

### Outputs Directory

Contains various outputs generated during investigations:

- **MIDI files** (`.mid`) - Transcription outputs with and without filtering
- **Log files** (`.txt`) - Detailed debugging and analysis logs
- **JSON files** - Structured data from tests
- **Visualizations** - Charts and graphs from comprehensive testing

### Comprehensive Leakage Test

The `comprehensive_leakage_test/` subdirectory contains:
- Detailed test results comparing all models
- Visualization of instrument distributions
- Statistical analysis of leakage patterns

## Key Findings

1. **MT3-PyTorch Leakage**: 38-98% of drum notes were incorrectly assigned to melodic instruments
2. **Root Cause**: Model generates incorrect program tokens in raw output
3. **Solution**: Automatic post-processing filter with configurable parameter
4. **Other Models**: MR-MT3 and YourMT3 showed no leakage issues

## Usage

These scripts are preserved for reference and should not be needed for normal operation. The fixes have been integrated into the main codebase:

```python
# The fix is now part of the adapter
from mt3_infer.adapters.mt3_pytorch import MT3PyTorchAdapter

# Filtering is enabled by default
adapter = MT3PyTorchAdapter()  # auto_filter=True

# Can be disabled if needed
adapter = MT3PyTorchAdapter(auto_filter=False)
```

## Related Documentation

- `docs/INSTRUMENT_LEAKAGE_INVESTIGATION.md` - Comprehensive investigation report
- `docs/MT3_PYTORCH_INSTRUMENT_LEAKAGE.md` - Technical analysis
- `docs/MT3_PYTORCH_AUTO_FILTER.md` - Usage guide for the fix

## Note

These files are archived for historical reference and debugging purposes. For production use, please use the main MT3-Infer API with the integrated fixes.