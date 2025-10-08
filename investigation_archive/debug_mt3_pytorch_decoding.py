#!/usr/bin/env python3
"""Debug MT3-PyTorch decoding to identify source of instrument leakage."""

import sys
import numpy as np
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, '/home/worzpro/Desktop/dev/patched_modules/mt3-infer')

from mt3_infer import load_model
from mt3_infer.utils.audio import load_audio
from mt3_infer.adapters.mt3_pytorch import MT3PyTorchAdapter

# Add vendored code to path
vendor_root = Path('/home/worzpro/Desktop/dev/patched_modules/mt3-infer/mt3_infer/vendor/kunato_mt3')
sys.path.insert(0, str(vendor_root))

try:
    from contrib import vocabularies, metrics_utils, event_codec, note_sequences
    import note_seq
except ImportError as e:
    print(f"Failed to import: {e}")
    sys.exit(1)


def debug_decoding():
    """Debug the decoding process step by step."""

    print("="*60)
    print("DEBUGGING MT3-PYTORCH DECODING")
    print("="*60)

    # Load model
    print("\n1. Loading model...")
    # Use auto_filter=False to see raw model output for debugging
    adapter = MT3PyTorchAdapter(auto_filter=False)
    checkpoint_path = "/home/worzpro/Desktop/dev/patched_modules/mt3-infer/.mt3_checkpoints/mt3_pytorch"
    adapter.load_model(checkpoint_path)

    # Load drum audio
    print("\n2. Loading drum audio...")
    audio_file = "assets/HappySounds_120bpm_Drums_drum_120BPM_BANDLAB.wav"
    audio, sr = load_audio(audio_file, sr=16000)

    # Preprocess
    print("\n3. Preprocessing...")
    features = adapter.preprocess(audio, sr)
    print(f"   Features shape: {features.shape}")

    # Forward pass
    print("\n4. Running model...")
    outputs = adapter.forward(features)
    print(f"   Outputs shape: {outputs.shape}")

    # Analyze raw tokens before decoding
    print("\n5. Analyzing raw tokens...")

    # Get codec for analysis
    vocab_config = vocabularies.VocabularyConfig(num_velocity_bins=1)
    codec = vocabularies.build_codec(vocab_config)

    # Check token ranges
    print(f"\n   Token ranges in codec:")
    print(f"   - Total classes: {codec.num_classes}")

    # Get event type ranges
    event_ranges = {}
    for event_type in ['shift', 'pitch', 'velocity', 'tie', 'program', 'drum']:
        try:
            min_id, max_id = codec.event_type_range(event_type)
            event_ranges[event_type] = (min_id, max_id)
            print(f"   - {event_type}: {min_id}-{max_id}")
        except:
            pass

    # Analyze token distribution in first chunk
    print(f"\n6. Token distribution in first chunk:")
    first_chunk = outputs[0]
    valid_tokens = first_chunk[first_chunk >= 0]  # Remove padding

    # Count tokens by type
    token_types = {'unknown': 0}
    for token in valid_tokens:
        found = False
        for event_type, (min_id, max_id) in event_ranges.items():
            if min_id <= token <= max_id:
                if event_type not in token_types:
                    token_types[event_type] = 0
                token_types[event_type] += 1
                found = True
                break
        if not found:
            token_types['unknown'] += 1

    print(f"   Token type counts:")
    for event_type, count in sorted(token_types.items(), key=lambda x: x[1], reverse=True):
        print(f"   - {event_type}: {count}")

    # Decode tokens manually to see what's happening
    print(f"\n7. Decoding first few tokens manually:")

    for i, token in enumerate(valid_tokens[:20]):
        try:
            event = codec.decode_event_index(int(token))
            print(f"   Token {i}: {token} -> {event.type}={event.value}")

            # If it's a program event, identify the instrument
            if event.type == 'program':
                print(f"      -> Instrument program {event.value}")
            elif event.type == 'drum':
                print(f"      -> Drum note {event.value}")
            elif event.type == 'pitch':
                print(f"      -> Pitch {event.value}")

        except Exception as e:
            print(f"   Token {i}: {token} -> Error: {e}")

    # Now decode properly and analyze the note_sequence
    print(f"\n8. Decoding to NoteSequence...")

    predictions = []
    for i in range(outputs.shape[0]):
        tokens = outputs[i]

        # Find EOS token
        eos_idx = np.where(tokens == vocabularies.DECODED_EOS_ID)[0]
        if len(eos_idx) > 0:
            tokens = tokens[:eos_idx[0]]

        # Get start time for this chunk
        start_time = adapter._last_frame_times[i][0]
        start_time -= start_time % (1 / codec.steps_per_second)

        predictions.append({
            'est_tokens': tokens,
            'start_time': start_time,
            'raw_inputs': []
        })

    # Decode to note sequence
    result = metrics_utils.event_predictions_to_ns(
        predictions,
        codec=codec,
        encoding_spec=note_sequences.NoteEncodingWithTiesSpec
    )

    note_sequence = result['est_ns']

    print(f"\n9. Analyzing NoteSequence:")
    print(f"   Total notes: {len(note_sequence.notes)}")

    # Count by instrument
    instrument_counts = {}
    for note in note_sequence.notes:
        key = f"is_drum={note.is_drum}, program={note.program}"
        if key not in instrument_counts:
            instrument_counts[key] = 0
        instrument_counts[key] += 1

    print(f"\n   Notes by instrument:")
    for key, count in sorted(instrument_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"   - {key}: {count}")

    # Sample some notes
    print(f"\n   First 10 notes:")
    for i, note in enumerate(note_sequence.notes[:10]):
        print(f"   {i}: pitch={note.pitch}, is_drum={note.is_drum}, program={note.program}, "
              f"velocity={note.velocity}, time={note.start_time:.3f}-{note.end_time:.3f}")

    # Check if the issue is in note_seq to MIDI conversion
    print(f"\n10. Checking note_seq to MIDI conversion...")

    # Convert to MIDI and analyze
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.mid', delete=False) as tmp:
        tmp_path = tmp.name

    note_seq.sequence_proto_to_midi_file(note_sequence, tmp_path)

    # Load with mido and check
    import mido
    mid = mido.MidiFile(tmp_path)

    print(f"\n   MIDI file analysis:")
    print(f"   - Tracks: {len(mid.tracks)}")

    channel_programs = {}
    note_counts_by_channel = {}

    for track_idx, track in enumerate(mid.tracks):
        for msg in track:
            if msg.type == 'program_change':
                channel_programs[msg.channel] = msg.program
            elif msg.type == 'note_on' and msg.velocity > 0:
                if msg.channel not in note_counts_by_channel:
                    note_counts_by_channel[msg.channel] = 0
                note_counts_by_channel[msg.channel] += 1

    print(f"\n   Channel analysis:")
    for channel, count in sorted(note_counts_by_channel.items()):
        program = channel_programs.get(channel, "default")
        print(f"   - Channel {channel}: {count} notes, program={program}")
        if channel == 9:
            print(f"     -> DRUM CHANNEL")
        else:
            print(f"     -> MELODIC CHANNEL")

    # Clean up
    Path(tmp_path).unlink()

    return note_sequence, instrument_counts


if __name__ == "__main__":
    note_seq, counts = debug_decoding()