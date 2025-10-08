# MT3-Infer Audio Requirements and Best Practices

## Required Audio Format

### Essential Requirements
- **Sample Rate**: 16,000 Hz (16 kHz)
- **Channels**: Mono (single channel)
- **Duration**: 2-30 seconds per segment (longer files are automatically segmented)
- **Format**: Any format supported by librosa (WAV, MP3, FLAC, etc.)

### Automatic Handling
The `load_audio()` utility automatically handles:
- Resampling to 16 kHz
- Converting stereo to mono
- Loading various audio formats

```python
from mt3_infer.utils.audio import load_audio

# Automatically converts to 16kHz mono
audio, sr = load_audio('any_audio.mp3', sr=16000)
```

## Optimal Audio Characteristics

### Loudness Levels

| Level | Range | Status | Notes |
|-------|-------|--------|-------|
| **Optimal** | -20 to -14 LUFS | ✅ Best | Standard for streaming platforms |
| Acceptable | -30 to -10 LUFS | ✅ Good | May need normalization |
| Too Quiet | < -30 LUFS | ⚠️ Warning | May miss quiet notes |
| Too Loud | > -10 LUFS | ⚠️ Warning | Risk of clipping artifacts |
| Clipping | > -3 LUFS | ❌ Problem | YourMT3 likely to fail |

### Peak Levels

| Peak Level | Status | Action Required |
|------------|--------|-----------------|
| -12 to -6 dB | ✅ Optimal | None |
| -20 to -12 dB | ✅ Good | None |
| -40 to -20 dB | ⚠️ Quiet | Consider amplification |
| -6 to -3 dB | ⚠️ Loud | Monitor for issues |
| > -3 dB | ❌ Clipping | Normalize required |

## Audio Content Characteristics

### Successfully Transcribed

#### Works Well
- Clear, separated instruments
- Moderate tempo (60-140 BPM)
- Good dynamic range
- Minimal background noise
- Standard tuning (A440)

#### Specific Examples
- Jazz trios
- Classical piano
- Pop/rock with clear mix
- Folk acoustic guitar
- Simple drum patterns

### Challenging Content

#### May Have Issues
- Very fast tempo (> 170 BPM)
- Dense orchestration
- Heavy distortion/effects
- Extreme frequency content
- Non-standard tuning

#### Known Problematic Cases
1. **Dense Fast Drums** (YourMT3 fails)
   - BPM > 170
   - Continuous patterns with < 5% silence
   - Example: FDNB_Bombs_Drum_Full_174

2. **Extreme Metal**
   - Heavy distortion
   - Blast beats
   - Low-tuned guitars

3. **Electronic Music**
   - Heavy sidechain compression
   - Extreme bass (< 40 Hz)
   - Glitch/IDM patterns

## Preprocessing Recommendations

### Basic Preprocessing Pipeline

```python
import numpy as np
import pyloudnorm as pyln
from mt3_infer.utils.audio import load_audio

def preprocess_for_mt3(filepath, target_lufs=-16.0):
    """Prepare audio for optimal MT3 transcription."""

    # Step 1: Load and convert to 16kHz mono
    audio, sr = load_audio(filepath, sr=16000)

    # Step 2: Remove DC offset
    audio = audio - np.mean(audio)

    # Step 3: Apply LUFS normalization
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(audio)

    if abs(loudness - target_lufs) > 2.0:  # If more than 2 LU off target
        audio = pyln.normalize.loudness(audio, loudness, target_lufs)

    # Step 4: Soft limiting to prevent clipping
    audio = np.tanh(audio * 0.95) / 0.95

    # Step 5: Final peak check
    peak = np.abs(audio).max()
    if peak > 0.99:
        audio = audio * (0.99 / peak)

    return audio, sr
```

### Advanced Preprocessing

```python
import librosa

def advanced_preprocess(filepath):
    """Advanced preprocessing for difficult audio."""

    audio, sr = load_audio(filepath, sr=16000)

    # 1. Noise reduction (if needed)
    if is_noisy(audio):
        audio = reduce_noise(audio, sr)

    # 2. Harmonic/percussive separation
    harmonic, percussive = librosa.effects.hpss(audio)

    # 3. Process separately if needed
    if is_drum_heavy(audio):
        # Use percussive component for drums
        audio = percussive
    elif is_pitched_only(audio):
        # Use harmonic for melodic content
        audio = harmonic

    # 4. Dynamic range compression
    audio = compress_dynamics(audio, threshold=0.7, ratio=4)

    # 5. Normalize
    audio = normalize_lufs(audio, sr, target=-16)

    return audio, sr
```

## Model-Specific Requirements

### MR-MT3
- **Most forgiving** with audio quality
- Handles loud audio well
- Works with dense patterns
- Less sensitive to frequency extremes

### YourMT3
- **Most demanding** on audio quality
- Fails on very dense drums (> 170 BPM)
- Requires good separation between instruments
- Best with -20 to -14 LUFS

### MT3-PyTorch
- **Balanced** requirements
- Similar to MR-MT3 but more accurate
- Good with standard audio
- Benefits from normalization

## Quick Diagnostic

```python
def diagnose_audio(filepath):
    """Quick diagnostic for audio issues."""

    audio, sr = load_audio(filepath, sr=16000)

    # Calculate metrics
    peak_db = 20 * np.log10(np.abs(audio).max() + 1e-10)
    rms_db = 20 * np.log10(np.sqrt(np.mean(audio**2)) + 1e-10)

    meter = pyln.Meter(sr)
    lufs = meter.integrated_loudness(audio)

    silence_ratio = np.sum(np.abs(audio) < 0.001) / len(audio)

    # Diagnose
    issues = []

    if peak_db > -3:
        issues.append(f"⚠️ Near clipping: {peak_db:.1f}dB peak")
    if lufs > -10:
        issues.append(f"⚠️ Very loud: {lufs:.1f} LUFS")
    if lufs < -30:
        issues.append(f"⚠️ Very quiet: {lufs:.1f} LUFS")
    if silence_ratio > 0.7:
        issues.append(f"⚠️ Mostly silence: {silence_ratio*100:.1f}%")

    if not issues:
        print("✅ Audio is in good condition for transcription")
    else:
        print("Issues found:")
        for issue in issues:
            print(f"  {issue}")

    # Recommend model
    if silence_ratio < 0.05 and "drum" in filepath.lower():
        print("💡 Recommendation: Use MR-MT3 (YourMT3 may fail)")
    elif lufs > -10:
        print("💡 Recommendation: Apply LUFS normalization first")

    return {
        'peak_db': peak_db,
        'rms_db': rms_db,
        'lufs': lufs,
        'silence_ratio': silence_ratio,
        'issues': issues
    }
```

## Summary Checklist

### ✅ Do's
- Use 16 kHz sample rate
- Normalize to -16 LUFS for best results
- Use mono audio
- Remove DC offset
- Apply soft limiting if needed

### ❌ Don'ts
- Don't use 44.1/48 kHz directly
- Don't submit clipped audio (> -3dB peak)
- Don't use extremely quiet audio (< -40dB)
- Don't expect YourMT3 to handle dense fast drums
- Don't forget to check silence ratio

### 🔧 Quick Fixes
| Problem | Quick Solution |
|---------|---------------|
| No notes detected | Resample to 16kHz |
| Too loud | Normalize to -16 LUFS |
| Too quiet | Amplify by 10-20 dB |
| Dense drums failing | Switch to MR-MT3 |
| Noisy recording | Apply noise reduction first |