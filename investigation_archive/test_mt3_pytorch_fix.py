#!/usr/bin/env python3
"""Test the fixed MT3-PyTorch adapter."""

import sys
sys.path.insert(0, '/home/worzpro/Desktop/dev/patched_modules/mt3-infer')

from mt3_infer.adapters.mt3_pytorch import MT3PyTorchAdapter
from mt3_infer.utils.audio import load_audio


def test_fixed_adapter():
    """Test the fixed MT3-PyTorch adapter."""

    print("Testing Fixed MT3-PyTorch Adapter")
    print("="*60)

    # Load model
    adapter = MT3PyTorchAdapter()
    checkpoint_path = "/home/worzpro/Desktop/dev/patched_modules/mt3-infer/.mt3_checkpoints/mt3_pytorch"
    adapter.load_model(checkpoint_path)

    # Test with all three audio files
    test_files = [
        ("assets/HappySounds_120bpm_Drums_drum_120BPM_BANDLAB.wav", "drums_happy"),
        ("assets/FDNB_Bombs_Drum_Full_174_drum_174BPM_BANDLAB.wav", "drums_fdnb"),
        ("assets/Chill_Guitar_loop_BANDLAB.wav", "guitar_chill")
    ]

    results = {}

    for audio_file, label in test_files:
        print(f"\nTranscribing: {label}")

        try:
            audio, sr = load_audio(audio_file, sr=16000)
            midi = adapter.transcribe(audio, sr)

            # Analyze result
            drum_notes = 0
            non_drum_notes = 0
            instruments = {}

            for track in midi.tracks:
                for msg in track:
                    if msg.type == 'note_on' and msg.velocity > 0:
                        if hasattr(msg, 'channel'):
                            if msg.channel == 9:
                                drum_notes += 1
                                instruments['drums'] = instruments.get('drums', 0) + 1
                            else:
                                non_drum_notes += 1
                                # Get program for this channel
                                program = 'unknown'
                                for prev_msg in track:
                                    if prev_msg.type == 'program_change' and prev_msg.channel == msg.channel:
                                        program = prev_msg.program
                                        break
                                instruments[f'ch{msg.channel}_p{program}'] = instruments.get(f'ch{msg.channel}_p{program}', 0) + 1

            total_notes = drum_notes + non_drum_notes

            if total_notes > 0:
                leakage_pct = (non_drum_notes / total_notes) * 100
            else:
                leakage_pct = 0

            results[label] = {
                'total': total_notes,
                'drums': drum_notes,
                'non_drums': non_drum_notes,
                'leakage_pct': leakage_pct,
                'instruments': instruments
            }

            print(f"  Total: {total_notes}, Drums: {drum_notes}, Non-drums: {non_drum_notes}")
            print(f"  Leakage: {leakage_pct:.1f}%")

            # Save MIDI
            output_file = f"fixed_{label}.mid"
            midi.save(output_file)

        except Exception as e:
            print(f"  Error: {e}")
            results[label] = {'error': str(e)}

    # Summary
    print("\n" + "="*60)
    print("SUMMARY - Fixed MT3-PyTorch")
    print("="*60)

    for label, result in results.items():
        if 'error' in result:
            print(f"{label}: ERROR - {result['error']}")
        else:
            status = "✅ FIXED" if result['leakage_pct'] == 0 else f"⚠️ {result['leakage_pct']:.1f}% leakage"
            print(f"{label}: {result['total']} notes, {status}")
            if result['non_drums'] > 0:
                print(f"  Instruments: {result['instruments']}")

    return results


if __name__ == "__main__":
    test_fixed_adapter()