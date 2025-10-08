#!/usr/bin/env python3
"""
Test YourMT3 integration with real audio.

This script tests:
1. Loading YourMT3 checkpoint
2. Transcribing audio
3. Saving MIDI output
4. Performance measurement
"""

import time
import soundfile as sf
import numpy as np
from pathlib import Path

print("=" * 70)
print("YourMT3 Integration Test")
print("=" * 70)

# Test audio file
audio_path = "assets/HappySounds_120bpm_Drums_drum_120BPM_BANDLAB.wav"
checkpoint_path = ".mt3_checkpoints/yourmt3/mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b36_nops/last.ckpt"
output_path = "test_outputs/yourmt3_output.mid"

# Create output directory
Path("test_outputs").mkdir(exist_ok=True)

print(f"\n📁 Test Audio: {audio_path}")
print(f"📁 Checkpoint: {checkpoint_path}")
print(f"📁 Output: {output_path}")

# Step 1: Load audio
print("\n" + "=" * 70)
print("Step 1: Loading Audio")
print("=" * 70)

audio, sr = sf.read(audio_path)
if audio.ndim > 1:
    audio = audio.mean(axis=1)  # Convert to mono

duration = len(audio) / sr
print(f"✅ Audio loaded: {duration:.2f} seconds, {sr} Hz")
print(f"   Shape: {audio.shape}, dtype: {audio.dtype}")

# Step 2: Load YourMT3 model
print("\n" + "=" * 70)
print("Step 2: Loading YourMT3 Model")
print("=" * 70)

try:
    from mt3_infer.adapters.yourmt3 import YourMT3Adapter

    adapter = YourMT3Adapter(model_key='yptf_moe_nops')
    print(f"✅ Adapter created: {adapter.config['name']}")
    print(f"   Description: {adapter.config['description']}")

    # Load checkpoint
    load_start = time.time()
    adapter.load_model(checkpoint_path, device='auto')
    load_time = time.time() - load_start

    print(f"✅ Model loaded in {load_time:.2f}s")
    print(f"   Device: {adapter.device_str}")
    print(f"   Model type: {type(adapter.model).__name__}")

except Exception as e:
    print(f"❌ Failed to load model: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Step 3: Transcribe audio
print("\n" + "=" * 70)
print("Step 3: Transcribing Audio")
print("=" * 70)

try:
    transcribe_start = time.time()
    midi = adapter.transcribe(audio, sr)
    transcribe_time = time.time() - transcribe_start

    print(f"✅ Transcription complete in {transcribe_time:.2f}s")
    print(f"   Speed: {duration / transcribe_time:.2f}x real-time")

    # Count notes
    total_notes = sum(len(track.notes) for track in midi.tracks if hasattr(track, 'notes'))
    print(f"   Notes detected: {total_notes}")

except Exception as e:
    print(f"❌ Transcription failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Step 4: Save MIDI
print("\n" + "=" * 70)
print("Step 4: Saving MIDI Output")
print("=" * 70)

try:
    midi.save(output_path)

    # Verify output
    output_size = Path(output_path).stat().st_size
    print(f"✅ MIDI saved: {output_path}")
    print(f"   File size: {output_size} bytes")
    print(f"   Tracks: {len(midi.tracks)}")

except Exception as e:
    print(f"❌ Failed to save MIDI: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Summary
print("\n" + "=" * 70)
print("✅ TEST PASSED - YourMT3 Integration Successful!")
print("=" * 70)
print(f"\n📊 Performance Summary:")
print(f"   Audio duration:    {duration:.2f}s")
print(f"   Load time:         {load_time:.2f}s")
print(f"   Transcription time: {transcribe_time:.2f}s")
print(f"   Speed:             {duration / transcribe_time:.2f}x real-time")
print(f"   Notes detected:    {total_notes}")
print(f"   Device:            {adapter.device_str}")
print(f"\n💾 Output saved to: {output_path}")
print("\n" + "=" * 70)
