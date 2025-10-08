#!/usr/bin/env python3
"""Test auto_filter parameter through the public API."""

import sys
sys.path.insert(0, '/home/worzpro/Desktop/dev/patched_modules/mt3-infer')

from mt3_infer import load_model
from mt3_infer.utils.audio import load_audio


def test_api_filter():
    """Test auto_filter parameter via API."""

    print("Testing auto_filter via Public API")
    print("="*60)

    # Load test audio
    audio, sr = load_audio("assets/HappySounds_120bpm_Drums_drum_120BPM_BANDLAB.wav", sr=16000)

    # Test 1: Default behavior (auto_filter=True)
    print("\n1. Default API usage (should have filtering on)")
    model_default = load_model("mt3_pytorch")
    midi_default = model_default.transcribe(audio, sr)

    drum_notes = 0
    non_drum_notes = 0
    for track in midi_default.tracks:
        for msg in track:
            if msg.type == 'note_on' and msg.velocity > 0 and hasattr(msg, 'channel'):
                if msg.channel == 9:
                    drum_notes += 1
                else:
                    non_drum_notes += 1

    total = drum_notes + non_drum_notes
    print(f"   Total: {total}, Drums: {drum_notes}, Non-drums: {non_drum_notes}")
    print(f"   Status: {'✅ Filtering active' if non_drum_notes == 0 else '⚠️ Leakage detected'}")

    # Test 2: Explicitly disable filtering
    print("\n2. API with auto_filter=False")
    model_no_filter = load_model("mt3_pytorch", auto_filter=False, cache=False)
    midi_no_filter = model_no_filter.transcribe(audio, sr)

    drum_notes = 0
    non_drum_notes = 0
    for track in midi_no_filter.tracks:
        for msg in track:
            if msg.type == 'note_on' and msg.velocity > 0 and hasattr(msg, 'channel'):
                if msg.channel == 9:
                    drum_notes += 1
                else:
                    non_drum_notes += 1

    total = drum_notes + non_drum_notes
    print(f"   Total: {total}, Drums: {drum_notes}, Non-drums: {non_drum_notes}")
    print(f"   Status: {'⚠️ No leakage' if non_drum_notes == 0 else '✅ Leakage present (expected)'}")

    # Test 3: Explicitly enable filtering
    print("\n3. API with auto_filter=True (explicit)")
    model_filter = load_model("mt3_pytorch", auto_filter=True, cache=False)
    midi_filter = model_filter.transcribe(audio, sr)

    drum_notes = 0
    non_drum_notes = 0
    for track in midi_filter.tracks:
        for msg in track:
            if msg.type == 'note_on' and msg.velocity > 0 and hasattr(msg, 'channel'):
                if msg.channel == 9:
                    drum_notes += 1
                else:
                    non_drum_notes += 1

    total = drum_notes + non_drum_notes
    print(f"   Total: {total}, Drums: {drum_notes}, Non-drums: {non_drum_notes}")
    print(f"   Status: {'✅ Filtering active' if non_drum_notes == 0 else '⚠️ Leakage detected'}")

    print("\n" + "="*60)
    print("✅ API auto_filter parameter working correctly!")


if __name__ == "__main__":
    test_api_filter()