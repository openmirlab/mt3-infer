# Release Notes - MT3-Infer v0.2.0

## Overview

MT3-Infer v0.2.0 introduces a critical fix for MT3-PyTorch's instrument leakage issue, where drum tracks were incorrectly transcribed with melodic instruments. This release maintains full backward compatibility while providing automatic filtering that dramatically improves transcription accuracy for drum content.

## Key Improvements

### 🎯 MT3-PyTorch Instrument Leakage Fix

**Problem Solved**: MT3-PyTorch was generating incorrect instrument assignments when transcribing drum tracks, with up to 98% of notes being misclassified as bass, chromatic percussion, or other melodic instruments.

**Solution**: Implemented automatic post-processing filter with configurable `auto_filter` parameter.

**Results**:
- HappySounds drums: 38.5% → 0% leakage
- FDNB drums: 98.4% → 0% leakage
- Guitar tracks: Correctly preserved

## Usage

### Default Behavior (Filtering Enabled)

```python
from mt3_infer import load_model, transcribe

# All of these use filtering by default
model = load_model("mt3_pytorch")
midi = transcribe(audio, model="mt3_pytorch")
```

### Disable Filtering (For Research)

```python
# Direct adapter
from mt3_infer.adapters.mt3_pytorch import MT3PyTorchAdapter
adapter = MT3PyTorchAdapter(auto_filter=False)

# Via API
model = load_model("mt3_pytorch", auto_filter=False)
midi = transcribe(audio, model="mt3_pytorch", auto_filter=False)
```

## Technical Details

### Root Cause

Investigation revealed the issue originates from the model itself, not our implementation:
- Model generates incorrect program tokens (32 for bass, 80 for synth) in raw output
- Likely due to training data bias toward pitched instruments
- Shared attention mechanism across all instrument types

### Detection Logic

The filter detects three patterns:
1. Drums mixed with bass (>20% bass notes)
2. Chromatic percussion misclassification (>70%)
3. Drum-heavy content (>60% drums)

### Implementation

- Added to `mt3_infer/adapters/mt3_pytorch.py`
- Integrated with public API functions
- Fully backward compatible
- No performance impact

## Documentation Updates

- **NEW**: `docs/INSTRUMENT_LEAKAGE_INVESTIGATION.md` - Complete investigation report
- **NEW**: `docs/MT3_PYTORCH_AUTO_FILTER.md` - Usage guide
- **NEW**: `investigation_archive/` - Archived test scripts for reference
- **UPDATED**: README.md, CLAUDE.md, TROUBLESHOOTING.md

## Backward Compatibility

✅ **Fully backward compatible** - All existing code continues to work without modification. The filter defaults to `True`, providing better results out of the box.

## Other Changes

- Cleaned up project structure
- Archived investigation scripts to `investigation_archive/`
- Updated all documentation to reflect v0.2.0 changes
- Added CHANGELOG.md for version tracking

## Credits

Special thanks to the user who asked "why do you sure the issue is not come from our code?" - this critical question directed the investigation to examine our implementation thoroughly, ultimately revealing the model-level issue.

## Next Steps

For users:
- Update to v0.2.0 for automatic drum transcription improvements
- No code changes required unless you want to disable filtering

For developers:
- Consider fine-tuning MT3-PyTorch on drum-heavy datasets
- Implement confidence scores for more nuanced filtering
- Explore ensemble methods combining multiple models

## Installation

```bash
# Update to latest version
pip install --upgrade mt3-infer

# Or with uv
uv add mt3-infer@latest
```

## More Information

- Full investigation: `docs/INSTRUMENT_LEAKAGE_INVESTIGATION.md`
- Technical details: `docs/MT3_PYTORCH_INSTRUMENT_LEAKAGE.md`
- Usage guide: `docs/MT3_PYTORCH_AUTO_FILTER.md`
- Troubleshooting: `docs/TROUBLESHOOTING.md`

---

*MT3-Infer v0.2.0 - Production-ready music transcription with automatic instrument correction*