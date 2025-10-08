# MT3-PyTorch Auto-Filter Parameter

## Overview

The MT3-PyTorch adapter now includes an `auto_filter` parameter to control automatic instrument leakage filtering. This addresses a known issue where MT3-PyTorch incorrectly assigns drum sounds to melodic instruments.

## Usage

### Via Adapter (Direct)

```python
from mt3_infer.adapters.mt3_pytorch import MT3PyTorchAdapter

# Default: filtering enabled
adapter = MT3PyTorchAdapter(auto_filter=True)  # or just MT3PyTorchAdapter()

# Disable filtering
adapter = MT3PyTorchAdapter(auto_filter=False)
```

### Via Public API

```python
from mt3_infer import load_model

# Default: filtering enabled
model = load_model("mt3_pytorch")  # auto_filter=True by default

# Explicitly enable filtering
model = load_model("mt3_pytorch", auto_filter=True)

# Disable filtering
model = load_model("mt3_pytorch", auto_filter=False)
```

## What It Does

When `auto_filter=True` (default), the adapter automatically:

1. **Detects drum-heavy content** - If >60% of notes are drums
2. **Identifies leakage patterns** - Drums mixed with bass (>20%) or chromatic percussion (>70%)
3. **Applies appropriate filtering**:
   - Removes non-drum channels for drum tracks
   - Remaps misclassified chromatic percussion to drums
   - Preserves non-drum content for actual melodic tracks

## Results

| Test File | Without Filter | With Filter |
|-----------|---------------|-------------|
| HappySounds drums | 38.5% leakage (47/122 notes) | ✅ 0% leakage (75/75 notes) |
| FDNB drums | 98.4% leakage (190/193 notes) | ✅ 0% leakage (3/3 notes) |
| Guitar | Correctly preserved | Correctly preserved |

## When to Disable

You might want to set `auto_filter=False` if:

- You're analyzing the raw model output for research
- You're working with mixed drum/melodic content and want full output
- You have your own post-processing pipeline
- You want to see the original model behavior

## Technical Details

The filtering logic examines MIDI program numbers:
- Programs 32-39: Bass instruments (common leakage)
- Programs 8-15: Chromatic percussion (misclassified drums)
- Channel 9: Drum channel (preserved)

The filter is conservative and only activates when clear patterns of misclassification are detected.

## Example: Comparing Outputs

```python
from mt3_infer import load_model
from mt3_infer.utils.audio import load_audio

# Load drum audio
audio, sr = load_audio("drums.wav", sr=16000)

# With filter (default)
model_filtered = load_model("mt3_pytorch")
midi_clean = model_filtered.transcribe(audio, sr)
midi_clean.save("drums_clean.mid")

# Without filter
model_raw = load_model("mt3_pytorch", auto_filter=False)
midi_raw = model_raw.transcribe(audio, sr)
midi_raw.save("drums_raw.mid")

# The filtered version will have only drum notes
# The raw version may have bass and other instruments mixed in
```

## Implementation

The filtering is implemented in `mt3_infer/adapters/mt3_pytorch.py`:
- `_apply_auto_filtering()`: Main detection logic
- `_filter_drums_only()`: Removes non-drum channels
- `_remap_to_drums()`: Converts misclassified notes to drums

The parameter defaults to `True` to provide the best out-of-box experience for most users.