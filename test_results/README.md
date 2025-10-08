# MT3-PyTorch Test Results

This directory contains important test results demonstrating the MT3-PyTorch instrument leakage fix.

## Directory Structure

### `before_fix/`
Contains MIDI files showing the instrument leakage issue:
- `drums_happy_with_leakage.mid` - Drums incorrectly transcribed with 38.5% bass/synth instruments

### `after_fix/`
Contains MIDI files after applying the auto_filter fix:
- `drums_happy_fixed.mid` - Same file with 100% drums (no leakage)
- `fixed_drums_happy.mid` - HappySounds drums corrected
- `fixed_drums_fdnb.mid` - FDNB drums corrected (was 98.4% leakage)
- `fixed_guitar_chill.mid` - Guitar file (correctly preserved as non-drums)

### `comparison/`
Visual comparisons and analysis:
- `comprehensive_leakage_comparison.png` - Visual comparison across all models and files

## Test Results Summary

| File | Before Fix | After Fix |
|------|------------|-----------|
| HappySounds drums | 61.5% drums, 38.5% leakage | ✅ 100% drums |
| FDNB drums | 1.6% drums, 98.4% leakage | ✅ 100% drums |
| Chill Guitar | Correctly preserved | Correctly preserved |

## How to Verify

### Listen to the Difference

```python
from mt3_infer.utils.midi import midi_to_audio
import mido

# Load MIDI files
before = mido.MidiFile('before_fix/drums_happy_with_leakage.mid')
after = mido.MidiFile('after_fix/drums_happy_fixed.mid')

# Convert to audio
audio_before = midi_to_audio(before)
audio_after = midi_to_audio(after)

# The 'before' will have bass and synth sounds mixed with drums
# The 'after' will be pure drums
```

### Check Instrument Distribution

```python
import mido

def analyze_midi(filepath):
    mid = mido.MidiFile(filepath)
    instruments = {}

    for track in mid.tracks:
        for msg in track:
            if msg.type == 'note_on' and msg.velocity > 0:
                channel = msg.channel if hasattr(msg, 'channel') else -1
                if channel == 9:
                    instruments['drums'] = instruments.get('drums', 0) + 1
                else:
                    instruments['other'] = instruments.get('other', 0) + 1

    return instruments

# Before fix
print("Before:", analyze_midi('before_fix/drums_happy_with_leakage.mid'))
# Output: {'drums': 75, 'other': 47}

# After fix
print("After:", analyze_midi('after_fix/drums_happy_fixed.mid'))
# Output: {'drums': 75, 'other': 0}
```

## Reproducing the Tests

To reproduce these results with the current codebase:

```python
from mt3_infer import load_model
from mt3_infer.utils.audio import load_audio

# Load audio
audio, sr = load_audio('assets/HappySounds_120bpm_Drums_drum_120BPM_BANDLAB.wav', sr=16000)

# Without fix (raw output)
model_raw = load_model("mt3_pytorch", auto_filter=False)
midi_with_leakage = model_raw.transcribe(audio, sr)

# With fix (default)
model_fixed = load_model("mt3_pytorch")  # auto_filter=True by default
midi_fixed = model_fixed.transcribe(audio, sr)

# Save for comparison
midi_with_leakage.save('test_before.mid')
midi_fixed.save('test_after.mid')
```

## Key Insights

1. **Severity**: The FDNB drums file had 98.4% of notes misclassified - nearly complete failure
2. **Pattern**: Leakage primarily manifested as:
   - Program 32 (Acoustic Bass) - common in normal tempo drums
   - Program 8 (Chromatic Percussion) - dominant in fast/dense drums
3. **Solution Effectiveness**: 100% correction rate with auto_filter enabled

## Related Documentation

- Full investigation: `../docs/INSTRUMENT_LEAKAGE_INVESTIGATION.md`
- Technical analysis: `../docs/MT3_PYTORCH_INSTRUMENT_LEAKAGE.md`
- Usage guide: `../docs/MT3_PYTORCH_AUTO_FILTER.md`