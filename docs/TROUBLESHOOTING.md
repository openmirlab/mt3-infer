# MT3-Infer Troubleshooting Guide

## Common Issues and Solutions

### 1. Instrument Leakage in Drum Tracks (MT3-PyTorch)

#### Symptoms
- Drum tracks transcribed with bass or other melodic instruments
- MIDI contains multiple channels when only drums expected
- Chromatic percussion (vibraphone) appearing in drum transcriptions

#### Solution
This is automatically fixed in v0.2.0+ with the `auto_filter` parameter (enabled by default):

```python
# Default behavior (filtering enabled)
model = load_model("mt3_pytorch")  # auto_filter=True

# To see raw output (for research)
model = load_model("mt3_pytorch", auto_filter=False)
```

**Note**: This only affects MT3-PyTorch. MR-MT3 and YourMT3 don't have this issue.

For technical details, see `docs/MT3_PYTORCH_INSTRUMENT_LEAKAGE.md`.

### 2. Model Returns 0 Notes / Empty Transcription

#### Symptoms
- MIDI file is generated but contains no notes
- Synthesized audio is very short (< 2 seconds)
- `transcribe()` returns empty or near-empty MIDI

#### Causes and Solutions

##### A. Wrong Sample Rate
**Cause**: MT3 models require 16kHz audio input, not 44.1kHz or 48kHz.

**Solution**:
```python
from mt3_infer.utils.audio import load_audio

# This automatically resamples to 16kHz
audio, sr = load_audio('your_file.wav', sr=16000)
midi = transcribe(audio, model='mr_mt3', sr=sr)
```

##### B. Audio Too Loud/Quiet
**Cause**: Extreme loudness levels can confuse the models.

**Diagnosis**:
```python
import numpy as np
peak_db = 20 * np.log10(np.abs(audio).max())
print(f"Peak level: {peak_db:.1f} dB")

# Problematic if:
# - peak_db > -3 (too loud, near clipping)
# - peak_db < -40 (too quiet)
```

**Solution**:
```python
# Simple peak normalization
def normalize_audio(audio, target_peak_db=-6.0):
    peak = np.abs(audio).max()
    target_peak = 10**(target_peak_db / 20)
    scale = target_peak / (peak + 1e-10)
    return audio * scale

audio = normalize_audio(audio, target_peak_db=-6.0)
```

##### C. Model-Specific Limitations

**YourMT3 fails on dense drum patterns**:
- Files with > 170 BPM tempo
- < 5% silence ratio
- High spectral centroid (> 3000 Hz)

**Solution**: Use MR-MT3 instead
```python
# Automatic model selection
if "drum" in filename and "174" in filename:  # Fast drums
    model = "mr_mt3"  # More robust
else:
    model = "yourmt3"  # Better for multi-instrument
```

### 2. Synthesis Duration Mismatch

#### Symptoms
- Synthesized audio is much shorter/longer than original
- MIDI duration doesn't match audio duration

#### Solution
Check MIDI tempo and note timing:
```python
import pretty_midi

pm = pretty_midi.PrettyMIDI('output.mid')
print(f"MIDI duration: {pm.get_end_time():.2f}s")
print(f"Number of instruments: {len(pm.instruments)}")
print(f"Total notes: {sum(len(i.notes) for i in pm.instruments)}")
```

### 3. Installation Issues

#### numpy Compatibility
**Error**: `AttributeError: module 'numpy' has no attribute 'int'`

**Solution**: This is from pretty_midi, not mt3-infer. Use mido instead:
```python
import mido

mid = mido.MidiFile('output.mid')
note_count = sum(1 for track in mid.tracks
                 for msg in track
                 if msg.type == 'note_on' and msg.velocity > 0)
```

#### CUDA Out of Memory
**Error**: `CUDA out of memory`

**Solution**: Use CPU or reduce batch size:
```python
# Force CPU
midi = transcribe(audio, model='yourmt3', device='cpu')

# Or clear GPU cache
import torch
torch.cuda.empty_cache()
```

### 4. Model-Specific Issues

#### MR-MT3
- Generally most robust
- May produce more false positives on quiet sections
- Best for: drums, simple melodies

#### YourMT3
- More sophisticated but pickier about input
- Fails on very dense drum patterns
- Best for: multi-instrument, vocals, complex music

#### MT3-PyTorch
- Middle ground between MR-MT3 and YourMT3
- Uses official Magenta weights
- Good general-purpose choice

### 5. Audio Preprocessing Best Practices

```python
from mt3_infer.utils.audio import load_audio
import pyloudnorm as pyln
import numpy as np

def preprocess_audio(filepath):
    """Robust audio preprocessing pipeline."""

    # 1. Load and resample to 16kHz
    audio, sr = load_audio(filepath, sr=16000)

    # 2. Apply LUFS normalization
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(audio)

    if loudness > -14.0 or loudness < -30.0:
        # Normalize to -16 LUFS (streaming standard)
        audio = pyln.normalize.loudness(audio, loudness, -16.0)

    # 3. Prevent clipping
    peak = np.abs(audio).max()
    if peak > 0.99:
        audio = audio * (0.99 / peak)

    return audio, sr

# Use preprocessed audio
audio, sr = preprocess_audio('your_file.wav')
midi = transcribe(audio, model='mr_mt3', sr=sr)
```

### 6. Debugging Transcription Issues

#### Step-by-Step Debugging
```python
import numpy as np
from mt3_infer import load_model

# 1. Load model explicitly
model = load_model('mr_mt3')

# 2. Check audio properties
print(f"Audio shape: {audio.shape}")
print(f"Duration: {len(audio)/sr:.2f}s")
print(f"Peak: {20*np.log10(np.abs(audio).max()):.1f}dB")
print(f"RMS: {20*np.log10(np.sqrt(np.mean(audio**2))):.1f}dB")

# 3. Test transcription
midi = model.transcribe(audio, sr)

# 4. Analyze output
import mido
mid = mido.MidiFile('output.mid')
for i, track in enumerate(mid.tracks):
    notes = [msg for msg in track if msg.type == 'note_on']
    print(f"Track {i}: {len(notes)} notes")
```

### 7. Known Limitations

#### All Models
- Require 16kHz sample rate
- Work best with -20 to -14 LUFS loudness
- May struggle with very noisy recordings
- Limited to 127 MIDI programs (no custom instruments)

#### YourMT3 Specific
- **Cannot transcribe**: FDNB-style fast, dense drums (> 170 BPM)
- May fail on heavily distorted audio
- Requires more GPU memory than other models

#### MR-MT3 Specific
- Less accurate on complex harmonies
- May miss quiet background instruments
- Simpler architecture = less nuanced transcription

### 8. Performance Optimization

#### Speed vs Accuracy Trade-offs
```python
# Fastest (57x realtime on GPU)
midi = transcribe(audio, model='mr_mt3')

# Most accurate
midi = transcribe(audio, model='mt3_pytorch')

# Best for multi-instrument
midi = transcribe(audio, model='yourmt3')
```

#### Batch Processing
```python
def batch_transcribe(file_list, model='mr_mt3'):
    """Process multiple files efficiently."""

    # Load model once
    model_instance = load_model(model)

    results = {}
    for filepath in file_list:
        audio, sr = load_audio(filepath, sr=16000)
        midi = model_instance.transcribe(audio, sr)
        results[filepath] = midi

    return results
```

### 9. Alternative Approaches for Difficult Files

If a file consistently fails:

1. **Try all models**:
```python
for model in ['mr_mt3', 'mt3_pytorch', 'yourmt3']:
    try:
        midi = transcribe(audio, model=model)
        if count_notes(midi) > 0:
            print(f"Success with {model}")
            break
    except:
        continue
```

2. **Preprocess aggressively**:
```python
# Percussive/harmonic separation
import librosa
harmonic, percussive = librosa.effects.hpss(audio)

# Try transcribing each component
midi_harm = transcribe(harmonic, model='mr_mt3')
midi_perc = transcribe(percussive, model='mr_mt3')
```

3. **Use external tools** for very difficult cases:
- Melodyne for professional transcription
- Logic Pro X's MIDI extraction
- Ableton's Convert Audio to MIDI

### 10. Getting Help

If issues persist:

1. **Check the audio file**:
```bash
ffprobe -v error -show_entries format=duration,bit_rate:stream=codec_name,sample_rate,channels -of json input.wav
```

2. **Provide diagnostic info**:
- Audio duration, sample rate, channels
- Peak and RMS levels
- Model used and error messages
- First 10 seconds of audio (if possible)

3. **File an issue** with:
- Minimal reproducible example
- Audio file characteristics
- Expected vs actual output

## Quick Reference

| Issue | Quick Fix |
|-------|-----------|
| No notes detected | Resample to 16kHz, normalize to -16 LUFS |
| YourMT3 fails on drums | Use MR-MT3 instead |
| Short synthesis | Check MIDI tempo and note timing |
| Out of memory | Use CPU or reduce file length |
| Noisy transcription | Preprocess with noise reduction |
| Missing instruments | Try YourMT3 or MT3-PyTorch |