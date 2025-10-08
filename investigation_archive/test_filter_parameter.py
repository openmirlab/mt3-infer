#!/usr/bin/env python3
"""Test the auto_filter parameter for MT3-PyTorch adapter."""

import sys
sys.path.insert(0, '/home/worzpro/Desktop/dev/patched_modules/mt3-infer')

from mt3_infer.adapters.mt3_pytorch import MT3PyTorchAdapter
from mt3_infer.utils.audio import load_audio


def test_filter_parameter():
    """Test auto_filter parameter on/off."""

    print("Testing auto_filter Parameter")
    print("="*60)

    # Test audio file (drums with known leakage)
    audio_file = "assets/HappySounds_120bpm_Drums_drum_120BPM_BANDLAB.wav"
    checkpoint_path = "/home/worzpro/Desktop/dev/patched_modules/mt3-infer/.mt3_checkpoints/mt3_pytorch"

    # Load audio once
    audio, sr = load_audio(audio_file, sr=16000)

    results = {}

    # Test with filtering ON (default)
    print("\n1. Testing with auto_filter=True (default)")
    adapter_on = MT3PyTorchAdapter(auto_filter=True)
    adapter_on.load_model(checkpoint_path)
    midi_filtered = adapter_on.transcribe(audio, sr)

    # Analyze
    drum_notes = 0
    non_drum_notes = 0
    for track in midi_filtered.tracks:
        for msg in track:
            if msg.type == 'note_on' and msg.velocity > 0 and hasattr(msg, 'channel'):
                if msg.channel == 9:
                    drum_notes += 1
                else:
                    non_drum_notes += 1

    total = drum_notes + non_drum_notes
    leakage_pct = (non_drum_notes / total * 100) if total > 0 else 0

    print(f"   Total: {total}, Drums: {drum_notes}, Non-drums: {non_drum_notes}")
    print(f"   Leakage: {leakage_pct:.1f}%")
    results['filter_on'] = {'total': total, 'drums': drum_notes, 'non_drums': non_drum_notes}

    # Test with filtering OFF
    print("\n2. Testing with auto_filter=False")
    adapter_off = MT3PyTorchAdapter(auto_filter=False)
    adapter_off.load_model(checkpoint_path)
    midi_unfiltered = adapter_off.transcribe(audio, sr)

    # Analyze
    drum_notes = 0
    non_drum_notes = 0
    instruments = {}
    for track in midi_unfiltered.tracks:
        for msg in track:
            if msg.type == 'program_change':
                instruments[msg.channel] = msg.program
            elif msg.type == 'note_on' and msg.velocity > 0 and hasattr(msg, 'channel'):
                if msg.channel == 9:
                    drum_notes += 1
                else:
                    non_drum_notes += 1

    total = drum_notes + non_drum_notes
    leakage_pct = (non_drum_notes / total * 100) if total > 0 else 0

    print(f"   Total: {total}, Drums: {drum_notes}, Non-drums: {non_drum_notes}")
    print(f"   Leakage: {leakage_pct:.1f}%")
    if non_drum_notes > 0:
        print(f"   Non-drum instruments: {instruments}")
    results['filter_off'] = {'total': total, 'drums': drum_notes, 'non_drums': non_drum_notes}

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    filter_on_leakage = results['filter_on']['non_drums'] / results['filter_on']['total'] * 100 if results['filter_on']['total'] > 0 else 0
    filter_off_leakage = results['filter_off']['non_drums'] / results['filter_off']['total'] * 100 if results['filter_off']['total'] > 0 else 0

    print(f"auto_filter=True:  {filter_on_leakage:.1f}% leakage ({results['filter_on']['non_drums']}/{results['filter_on']['total']} notes)")
    print(f"auto_filter=False: {filter_off_leakage:.1f}% leakage ({results['filter_off']['non_drums']}/{results['filter_off']['total']} notes)")

    if filter_on_leakage == 0 and filter_off_leakage > 0:
        print("\n✅ auto_filter parameter working correctly!")
        print("   - With filter: No leakage")
        print("   - Without filter: Instrument leakage present")
    else:
        print("\n⚠️ Unexpected results")

    # Save both versions
    midi_filtered.save("test_filtered_on.mid")
    midi_unfiltered.save("test_filtered_off.mid")
    print(f"\nSaved MIDI files:")
    print(f"  - test_filtered_on.mid (auto_filter=True)")
    print(f"  - test_filtered_off.mid (auto_filter=False)")

    return results


if __name__ == "__main__":
    test_filter_parameter()