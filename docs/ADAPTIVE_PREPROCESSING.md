# Adaptive Preprocessing for Dense Drum Patterns

## Overview

YourMT3 can struggle with dense drum patterns (high onset density, >6 onsets/sec), often producing zero or very few transcribed notes. This document explains the **adaptive preprocessing** feature that addresses this issue through automatic audio analysis and multi-attempt transcription.

## The Problem

YourMT3 exhibits two key challenges on dense drum tracks:

1. **Onset Density Threshold**: When drum patterns exceed ~6.0 onsets/second, the model's ability to detect and transcribe individual drum hits degrades significantly
2. **Non-Determinism**: Even on the same input, YourMT3 can produce highly variable results (0-27 notes on identical audio)

**Example failure case:**
```
File: FDNB_Bombs_Drum_Full_174_drum_174BPM_BANDLAB.wav
- Tempo: 174 BPM
- Onset density: 5.71 onsets/sec
- Standard transcription: 0 notes
```

## The Solution

The adaptive preprocessing feature combines two strategies:

### 1. Time-Stretching (Slowing Audio)
By slowing down dense audio before transcription, we reduce the effective onset density, making it easier for the model to detect individual drum hits. After transcription, MIDI timing is automatically restored to the original tempo.

**Proven improvement:**
```
Original (1.0x): 0 notes
0.9x stretch: 16 notes ✓
0.8x stretch: 21 notes ✓
0.7x stretch: 18 notes ✓
```

### 2. Multi-Attempt Strategy
Due to YourMT3's non-determinism, we run multiple transcription attempts across different stretch factors and select the result with the highest note count.

## Usage

### Basic Example

```python
from mt3_infer import load_model
import librosa

# Load audio
audio, sr = librosa.load("dense_drums.wav", sr=16000)

# Load model
model = load_model("yourmt3", verbose=True)

# Transcribe with adaptive preprocessing
midi = model.transcribe(
    audio,
    sr=16000,
    adaptive=True,  # Enable adaptive mode
    num_attempts=3  # Try 3 times per configuration
)

# Save result
midi.save("output.mid")
```

### High-Level API

```python
from mt3_infer import transcribe
import librosa

audio, sr = librosa.load("dense_drums.wav", sr=16000)

# One-line transcription with adaptive mode
midi = transcribe(
    audio,
    model="yourmt3",
    adaptive=True,
    num_attempts=3
)
```

### Custom Configuration

```python
# Advanced: Custom stretch factors and method
midi = model.transcribe(
    audio,
    sr=16000,
    adaptive=True,
    stretch_factors=[1.0, 0.85, 0.75, 0.65],  # Custom factors
    stretch_method="pyrubberband",  # High-quality stretching
    num_attempts=5,  # More attempts for better coverage
    return_all=False  # Return only best result
)
```

## Parameters

### `adaptive` (bool, default=False)
Enable adaptive preprocessing mode. When enabled:
- Audio density is automatically analyzed
- Time-stretching is applied if density exceeds threshold
- Multi-attempt strategy is used
- Timing is automatically restored

### `stretch_factors` (List[float], optional)
Custom time-stretch factors to test. Each factor < 1.0 slows down the audio.

**Default behavior:**
- Dense audio (>6.0 onsets/sec): `[1.0, 0.9, 0.8, 0.7]`
- Normal audio: `[1.0]` (no stretching)

**Example:**
```python
stretch_factors=[1.0, 0.85, 0.75]  # Test 3 speeds
```

### `stretch_method` (str, default="auto")
Time-stretching algorithm to use:

- `"auto"`: Use pyrubberband if available, else librosa
- `"pyrubberband"`: High-quality stretching (better transient preservation)
- `"librosa"`: Standard phase vocoder (always available)

**Installation for pyrubberband:**
```bash
# Install rubberband library (system-level)
sudo apt-get install rubberband-cli  # Ubuntu/Debian
brew install rubberband  # macOS

# Install Python wrapper
uv add pyrubberband
```

### `num_attempts` (int, default=1)
Number of transcription attempts per stretch factor. Higher values improve coverage but increase processing time.

**Recommended values:**
- Dense drums: 3-5 attempts
- Normal audio: 1 attempt

### `return_all` (bool, default=False)
When `True`, returns all transcription results for manual selection:

```python
results = model.transcribe(
    audio,
    sr=16000,
    adaptive=True,
    num_attempts=3,
    return_all=True
)

# Results format: List[(midi, note_count, stretch_factor, config_name)]
for midi, note_count, stretch, config in results:
    print(f"{config}: {note_count} notes (stretch={stretch})")
```

### `verbose` (bool, default=False)
Enable verbose logging during transcription (set during model initialization):

```python
model = load_model("yourmt3", verbose=True)
midi = model.transcribe(audio, sr=16000, adaptive=True)
```

Output shows:
- Density analysis results
- Stretch factors being tested
- Note counts for each attempt
- Selected best result

## How It Works

### 1. Density Analysis

The algorithm analyzes audio characteristics:

```python
onset_density = num_onsets / duration  # Onsets per second
tempo = librosa.beat.beat_track(...)    # BPM

is_dense = (
    onset_density > 6.0 or
    (tempo > 150 and onset_density > 5) or
    (tempo > 160)
)
```

### 2. Multi-Pass Transcription

For each stretch factor × num_attempts:
1. Apply time-stretching (if factor ≠ 1.0)
2. Run transcription
3. Count notes in MIDI output
4. Restore original timing (if stretched)

### 3. Result Selection

Select the transcription with the highest note count:

```python
best_result = max(all_results, key=lambda x: x[1])  # x[1] = note_count
```

## Performance Considerations

### Processing Time

Adaptive mode increases processing time proportionally to attempts:

```
Standard mode: ~5 seconds
Adaptive (3 attempts × 4 stretch factors): ~60 seconds
```

**Recommendations:**
- Use `adaptive=True` only when needed (dense drums)
- Analyze density first with a test run
- Batch process offline for production

### Memory Usage

Each transcription attempt creates a separate MIDI object. For `return_all=True`:

```python
memory_per_attempt ≈ 10-50 KB (MIDI file size)
total_memory = num_attempts × len(stretch_factors) × memory_per_attempt
```

**Example:** 5 attempts × 4 factors = 20 MIDI objects ≈ 200-1000 KB

## Limitations

### 1. Non-Determinism Not Eliminated
Adaptive mode **mitigates** but does not **eliminate** YourMT3's variability. Results can still vary between runs.

### 2. Quality vs. Quantity Trade-off
Selecting by note count favors recall (finding all notes) over precision (avoiding false positives). For critical applications, consider:

```python
# Return all results for manual review
results = model.transcribe(audio, adaptive=True, return_all=True)

# Select based on custom criteria
best = max(results, key=lambda x: score_midi_quality(x[0]))
```

### 3. Tempo-Specific Optimization
The default stretch factors `[1.0, 0.9, 0.8, 0.7]` work well for 160-180 BPM drums. For extreme tempos:

```python
# Very fast (>200 BPM)
stretch_factors=[1.0, 0.8, 0.6, 0.5]

# Moderate tempo (<140 BPM)
stretch_factors=[1.0, 0.95, 0.9]  # Smaller adjustments
```

## Technical Details

### Timing Restoration Algorithm

After time-stretched transcription, MIDI timing is restored:

```python
def _restore_timing(midi, stretch_factor):
    """Scale all MIDI delta times by stretch_factor."""
    for track in midi.tracks:
        for msg in track:
            msg.time = int(msg.time * stretch_factor)
    return midi
```

**Note:** This preserves relative timing but may introduce small quantization errors (±1 tick).

### Librosa vs. Pyrubberband Quality

Spectral analysis shows pyrubberband better preserves transients:

| Metric | Pyrubberband | Librosa | Improvement |
|--------|--------------|---------|-------------|
| Onset strength | 2.88 | 2.01 | +43% |
| Spectral centroid | 3053 Hz | 2723 Hz | +12% |

However, YourMT3's non-determinism often outweighs audio quality differences in practice.

## Examples

### Example 1: Dense Drum Loop

```python
from mt3_infer import load_model
import librosa

# Load 174 BPM drum loop
audio, sr = librosa.load("drum_loop_174bpm.wav", sr=16000)

# Standard transcription (likely fails)
model = load_model("yourmt3")
midi_standard = model.transcribe(audio, sr=16000)
print(f"Standard: {count_notes(midi_standard)} notes")  # Output: 0 notes

# Adaptive transcription
midi_adaptive = model.transcribe(
    audio, sr=16000,
    adaptive=True,
    num_attempts=3
)
print(f"Adaptive: {count_notes(midi_adaptive)} notes")  # Output: 16-27 notes
```

### Example 2: Batch Processing with Density Check

```python
import librosa
from mt3_infer import load_model

model = load_model("yourmt3")

audio_files = ["track1.wav", "track2.wav", "track3.wav"]

for file_path in audio_files:
    audio, sr = librosa.load(file_path, sr=16000)

    # Analyze density
    onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
    onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
    density = len(onsets) / (len(audio) / sr)

    # Use adaptive mode only for dense tracks
    use_adaptive = density > 6.0

    midi = model.transcribe(
        audio, sr=sr,
        adaptive=use_adaptive,
        num_attempts=3 if use_adaptive else 1
    )

    print(f"{file_path}: density={density:.2f}, notes={count_notes(midi)}")
```

### Example 3: Custom Result Selection

```python
from mt3_infer import load_model
import librosa

def score_midi_quality(midi):
    """Custom quality scoring: balance note count and velocity variance."""
    note_count = sum(1 for track in midi.tracks
                     for msg in track
                     if msg.type == 'note_on' and msg.velocity > 0)

    velocities = [msg.velocity for track in midi.tracks
                  for msg in track
                  if msg.type == 'note_on' and msg.velocity > 0]

    velocity_variance = np.var(velocities) if velocities else 0

    # Prefer more notes with varied dynamics
    return note_count * (1 + velocity_variance / 100)

# Get all results
model = load_model("yourmt3")
audio, sr = librosa.load("dense_drums.wav", sr=16000)

results = model.transcribe(
    audio, sr=sr,
    adaptive=True,
    num_attempts=5,
    return_all=True
)

# Select by custom quality metric
best_midi = max(results, key=lambda x: score_midi_quality(x[0]))[0]
```

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'pyrubberband'"

**Solution:** Either install pyrubberband or use librosa:
```python
midi = model.transcribe(audio, adaptive=True, stretch_method="librosa")
```

### Issue: Still getting 0 notes even with adaptive=True

**Possible causes:**
1. Model non-determinism (try increasing `num_attempts`)
2. Audio preprocessing issues (check audio format: mono, 16kHz, float32)
3. Extreme density (try more aggressive stretch factors)

**Debug steps:**
```python
# Check audio properties
print(f"Shape: {audio.shape}, dtype: {audio.dtype}, range: [{audio.min()}, {audio.max()}]")

# Try with verbose logging
model = load_model("yourmt3", verbose=True)
midi = model.transcribe(audio, adaptive=True, num_attempts=5)

# Inspect all results
results = model.transcribe(audio, adaptive=True, num_attempts=5, return_all=True)
for midi, count, stretch, config in results:
    print(f"{config} (stretch={stretch}): {count} notes")
```

### Issue: Processing takes too long

**Solutions:**
1. Reduce `num_attempts`: `num_attempts=2` (faster)
2. Reduce stretch factors: `stretch_factors=[1.0, 0.8]` (test fewer speeds)
3. Use GPU: `model = load_model("yourmt3", device="cuda")`

## References

- **Investigation Report**: See `investigation_output/debug_log.txt` for detailed test results
- **Test Scripts**: `investigate_dense_drums.py`, `test_optimal_stretch.py`
- **YourMT3 Paper**: [Multi-Task Music Transcription](https://arxiv.org/abs/2407.04822)
