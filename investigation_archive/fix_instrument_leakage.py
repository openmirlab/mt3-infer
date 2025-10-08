#!/usr/bin/env python3
"""Fix MT3-PyTorch instrument leakage with advanced filtering strategies."""

import sys
import json
import numpy as np
import torch
import mido
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, '/home/worzpro/Desktop/dev/patched_modules/mt3-infer')

from mt3_infer import transcribe, load_model
from mt3_infer.utils.audio import load_audio
from mt3_infer.adapters.mt3_pytorch import MT3PyTorchAdapter
from mt3_infer.adapters import vocab_utils

# Create output directory
output_dir = Path("instrument_leakage_fix")
output_dir.mkdir(exist_ok=True)


def analyze_mt3_pytorch_decoding():
    """Deep dive into MT3-PyTorch's decoding logic."""
    print("\n" + "="*60)
    print("ANALYZING MT3-PYTORCH DECODING LOGIC")
    print("="*60)

    # Load and examine the adapter
    # Use auto_filter=False to test our own filtering strategies
    adapter = MT3PyTorchAdapter(auto_filter=False)

    # Check the vocab configuration
    print("\n--- Vocabulary Configuration ---")
    codec = vocab_utils.build_codec(num_velocity_bins=1)
    print(f"Codec num_classes: {codec.num_classes}")
    print(f"Max shift steps: {codec.max_shift_steps}")

    # Check event type ranges
    for event_type in ['shift', 'pitch', 'velocity', 'tie', 'program', 'drum']:
        try:
            min_id, max_id = codec.event_type_range(event_type)
            print(f"{event_type}: IDs {min_id}-{max_id} (count: {max_id-min_id+1})")
        except:
            pass

    return codec


def create_advanced_filter(midi_path, output_path, strategy='smart'):
    """Apply advanced filtering strategies to reduce instrument leakage."""

    mid = mido.MidiFile(midi_path)
    filtered_mid = mido.MidiFile()
    filtered_mid.ticks_per_beat = mid.ticks_per_beat

    # Statistics tracking
    stats = {
        'total_notes': 0,
        'filtered_notes': 0,
        'kept_notes': 0,
        'filters_applied': []
    }

    # Analyze note patterns first
    note_events = []
    for track_idx, track in enumerate(mid.tracks):
        current_time = 0
        for msg in track:
            current_time += msg.time
            if msg.type == 'note_on' and msg.velocity > 0:
                note_events.append({
                    'time': current_time,
                    'pitch': msg.note,
                    'velocity': msg.velocity,
                    'channel': msg.channel,
                    'track': track_idx
                })
                stats['total_notes'] += 1

    # Calculate statistics for filtering
    if note_events:
        velocities = [n['velocity'] for n in note_events]
        velocity_median = np.median(velocities)
        velocity_std = np.std(velocities)

        # Drum pitch range (typical drum MIDI notes)
        drum_pitch_range = (35, 81)  # From bass drum to open triangle

    print(f"\nApplying filter strategy: {strategy}")

    for track_idx, track in enumerate(mid.tracks):
        new_track = mido.MidiTrack()
        current_time = 0

        for msg in track:
            current_time += msg.time
            keep_message = True
            reason = ""

            if msg.type == 'note_on' and msg.velocity > 0:
                # Apply filtering based on strategy
                if strategy == 'drums_only':
                    # Keep only drum channel
                    if msg.channel != 9:
                        keep_message = False
                        reason = "Non-drum channel"
                        stats['filtered_notes'] += 1

                elif strategy == 'smart':
                    # Smart filtering based on multiple criteria
                    if msg.channel != 9:  # Non-drum channel
                        # Check if it's likely a mis-classified drum
                        is_likely_drum = False

                        # Criterion 1: Pitch in typical drum range
                        if drum_pitch_range[0] <= msg.note <= drum_pitch_range[1]:
                            is_likely_drum = True

                        # Criterion 2: Low velocity (ghost notes often misclassified)
                        if msg.velocity < velocity_median - velocity_std:
                            keep_message = False
                            reason = "Low velocity non-drum"

                        # Criterion 3: Isolated notes (drums are usually repetitive)
                        time_window = 500  # milliseconds
                        nearby_notes = [n for n in note_events
                                      if abs(n['time'] - current_time) < time_window
                                      and n['channel'] == 9]
                        if len(nearby_notes) < 2 and not is_likely_drum:
                            keep_message = False
                            reason = "Isolated non-drum note"

                        if not keep_message:
                            stats['filtered_notes'] += 1

                elif strategy == 'velocity_threshold':
                    # Filter by velocity
                    threshold = velocity_median - 0.5 * velocity_std
                    if msg.velocity < threshold and msg.channel != 9:
                        keep_message = False
                        reason = f"Below velocity threshold ({threshold:.1f})"
                        stats['filtered_notes'] += 1

                elif strategy == 'pitch_based':
                    # Filter non-drum channels with unusual pitches
                    if msg.channel != 9:
                        # Bass instruments often leak - filter high bass notes
                        if msg.channel == 0 and msg.note > 60:  # High bass notes
                            keep_message = False
                            reason = "High bass note (likely false positive)"
                            stats['filtered_notes'] += 1

            # Add message to new track
            if keep_message:
                new_track.append(msg)
                if msg.type == 'note_on' and msg.velocity > 0:
                    stats['kept_notes'] += 1
            else:
                # Replace with note_off to maintain timing
                if msg.type == 'note_on':
                    new_track.append(mido.Message('note_off', note=msg.note,
                                                 velocity=0, time=msg.time,
                                                 channel=msg.channel))

        if new_track:
            filtered_mid.tracks.append(new_track)

    # Save filtered MIDI
    filtered_mid.save(str(output_path))

    # Print statistics
    print(f"\nFiltering Statistics:")
    print(f"  Total notes: {stats['total_notes']}")
    print(f"  Filtered out: {stats['filtered_notes']}")
    print(f"  Kept: {stats['kept_notes']}")
    print(f"  Filter rate: {stats['filtered_notes']/stats['total_notes']*100:.1f}%")

    return stats


def patch_mt3_pytorch_adapter():
    """Create a patched version of MT3PyTorchAdapter with leakage mitigation."""

    patch_code = '''
# Patch for mt3_infer/adapters/mt3_pytorch.py

import numpy as np
from typing import Dict, Any

class MT3PyTorchAdapterPatched(MT3PyTorchAdapter):
    """Patched MT3-PyTorch adapter with instrument leakage mitigation."""

    def __init__(self, filter_non_drums=False, confidence_threshold=0.5):
        super().__init__()
        self.filter_non_drums = filter_non_drums
        self.confidence_threshold = confidence_threshold

    def decode(self, outputs: np.ndarray) -> Any:
        """Decode model outputs with filtering."""
        # Get original MIDI
        midi = super().decode(outputs)

        if not self.filter_non_drums:
            return midi

        # Apply filtering
        filtered_mid = mido.MidiFile()
        filtered_mid.ticks_per_beat = midi.ticks_per_beat

        for track in midi.tracks:
            new_track = mido.MidiTrack()

            for msg in track:
                # Filter logic
                keep = True

                if hasattr(msg, 'channel'):
                    # Option 1: Keep only drums
                    if self.filter_non_drums and msg.channel != 9:
                        if msg.type in ['note_on', 'note_off']:
                            keep = False

                    # Option 2: Filter by velocity (confidence proxy)
                    if msg.type == 'note_on' and msg.velocity > 0:
                        normalized_velocity = msg.velocity / 127.0
                        if normalized_velocity < self.confidence_threshold:
                            if msg.channel != 9:  # Don't filter drums
                                keep = False

                if keep:
                    new_track.append(msg)
                elif msg.type == 'note_on':
                    # Replace with note_off to maintain structure
                    new_track.append(mido.Message('note_off',
                                                 note=msg.note,
                                                 velocity=0,
                                                 time=msg.time,
                                                 channel=msg.channel))

            filtered_mid.tracks.append(new_track)

        return filtered_mid

    def transcribe_with_filtering(self, audio, sr, filter_strategy='smart'):
        """Transcribe with automatic filtering."""
        # Get raw transcription
        midi = self.transcribe(audio, sr)

        # Apply post-processing filter
        if filter_strategy == 'drums_only':
            self.filter_non_drums = True
            midi = self.decode(self.forward(self.preprocess(audio, sr)))

        return midi
'''

    # Save patch
    patch_path = output_dir / "mt3_pytorch_patch.py"
    with open(patch_path, 'w') as f:
        f.write(patch_code)

    print(f"\nPatch saved to: {patch_path}")
    return patch_code


def test_filtering_strategies():
    """Test different filtering strategies on the drum file."""

    print("\n" + "="*60)
    print("TESTING FILTERING STRATEGIES")
    print("="*60)

    # Load drum audio
    drum_file = "assets/HappySounds_120bpm_Drums_drum_120BPM_BANDLAB.wav"
    audio, sr = load_audio(drum_file, sr=16000)

    # Get original transcription
    print("\n1. Getting original transcription...")
    midi_original = transcribe(audio, model='mt3_pytorch', sr=sr)
    original_path = output_dir / "original.mid"
    midi_original.save(str(original_path))

    # Test different filtering strategies
    strategies = ['drums_only', 'smart', 'velocity_threshold', 'pitch_based']
    results = {}

    for strategy in strategies:
        print(f"\n2. Testing {strategy} filter...")
        filtered_path = output_dir / f"filtered_{strategy}.mid"
        stats = create_advanced_filter(original_path, filtered_path, strategy)
        results[strategy] = stats

    # Save results
    with open(output_dir / "filtering_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    return results


def compare_frequency_overlap():
    """Analyze frequency overlap between drums and leaked instruments."""

    print("\n" + "="*60)
    print("ANALYZING FREQUENCY OVERLAP")
    print("="*60)

    # This would require spectral analysis of the specific notes
    # that are being misclassified

    drum_file = "assets/HappySounds_120bpm_Drums_drum_120BPM_BANDLAB.wav"
    audio, sr = load_audio(drum_file, sr=16000)

    # Get transcription
    midi = transcribe(audio, model='mt3_pytorch', sr=sr)
    midi_path = output_dir / "for_frequency_analysis.mid"
    midi.save(str(midi_path))

    # Analyze which drum hits correspond to piano notes
    mid = mido.MidiFile(midi_path)

    bass_notes = []
    drum_notes = []

    for track in mid.tracks:
        current_time = 0
        for msg in track:
            current_time += msg.time
            if msg.type == 'note_on' and msg.velocity > 0:
                if msg.channel == 9:
                    drum_notes.append({
                        'time': current_time,
                        'pitch': msg.note,
                        'drum_name': DRUM_NOTES.get(msg.note, f'Drum {msg.note}')
                    })
                elif msg.channel == 0:  # Bass channel
                    bass_notes.append({
                        'time': current_time,
                        'pitch': msg.note,
                        'freq': 440 * 2**((msg.note - 69) / 12)  # MIDI to Hz
                    })

    # Find temporal correlations
    print("\nTemporal correlations:")
    time_window = 50  # milliseconds

    correlations = []
    for bass in bass_notes[:10]:  # Check first 10 bass notes
        nearby_drums = [d for d in drum_notes
                       if abs(d['time'] - bass['time']) < time_window]
        if nearby_drums:
            print(f"Bass note (pitch {bass['pitch']}, {bass['freq']:.1f}Hz) at {bass['time']}ms")
            print(f"  Nearby drums: {[d['drum_name'] for d in nearby_drums]}")
            correlations.append({
                'bass': bass,
                'drums': nearby_drums
            })

    return correlations


# Drum note mapping
DRUM_NOTES = {
    35: 'Acoustic Bass Drum', 36: 'Bass Drum 1', 38: 'Acoustic Snare',
    42: 'Closed Hi Hat', 46: 'Open Hi-Hat'
}


def main():
    """Main routine to fix instrument leakage."""

    print("MT3-PYTORCH INSTRUMENT LEAKAGE FIX")
    print("="*60)

    # Step 1: Analyze the decoding logic
    codec = analyze_mt3_pytorch_decoding()

    # Step 2: Test filtering strategies
    filtering_results = test_filtering_strategies()

    # Step 3: Analyze frequency overlap
    correlations = compare_frequency_overlap()

    # Step 4: Create patch
    patch_mt3_pytorch_adapter()

    # Step 5: Generate final report
    print("\n" + "="*60)
    print("FINAL RECOMMENDATIONS")
    print("="*60)

    print("\n1. IMMEDIATE FIX: Use 'drums_only' filter")
    print("   This removes all non-drum channel notes")

    print("\n2. SMART FIX: Use 'smart' filter")
    print("   This applies context-aware filtering")

    print("\n3. LONG-TERM FIX: Retrain or fine-tune model")
    print("   The model needs drum-specific training")

    # Find best strategy
    best_strategy = min(filtering_results.keys(),
                       key=lambda k: filtering_results[k]['filtered_notes'])

    print(f"\n💡 RECOMMENDED STRATEGY: {best_strategy}")
    print(f"   Achieves {filtering_results[best_strategy]['kept_notes']} retained notes")
    print(f"   with {filtering_results[best_strategy]['filtered_notes']} filtered")

    return filtering_results


if __name__ == "__main__":
    results = main()