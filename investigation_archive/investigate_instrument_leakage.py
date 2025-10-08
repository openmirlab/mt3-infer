#!/usr/bin/env python3
"""Investigate MT3-PyTorch instrument leakage issue with drum tracks."""

import sys
import json
import numpy as np
import mido
import matplotlib.pyplot as plt
import librosa
import librosa.display
from pathlib import Path
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, '/home/worzpro/Desktop/dev/patched_modules/mt3-infer')

from mt3_infer import transcribe, load_model
from mt3_infer.utils.audio import load_audio
from mt3_infer.adapters.mt3_pytorch import MT3PyTorchAdapter
from mt3_infer.adapters import vocab_utils

# Create output directory
output_dir = Path("instrument_leakage_analysis")
output_dir.mkdir(exist_ok=True)

# MIDI program to instrument mapping
PROGRAM_TO_INSTRUMENT = {
    # Piano
    0: 'Acoustic Grand Piano', 1: 'Bright Acoustic Piano', 2: 'Electric Grand Piano',
    3: 'Honky-tonk Piano', 4: 'Electric Piano 1', 5: 'Electric Piano 2',
    6: 'Harpsichord', 7: 'Clavi',

    # Chromatic Percussion
    8: 'Celesta', 9: 'Glockenspiel', 10: 'Music Box', 11: 'Vibraphone',
    12: 'Marimba', 13: 'Xylophone', 14: 'Tubular Bells', 15: 'Dulcimer',

    # Organ
    16: 'Drawbar Organ', 17: 'Percussive Organ', 18: 'Rock Organ', 19: 'Church Organ',
    20: 'Reed Organ', 21: 'Accordion', 22: 'Harmonica', 23: 'Tango Accordion',

    # Guitar
    24: 'Acoustic Guitar (nylon)', 25: 'Acoustic Guitar (steel)', 26: 'Electric Guitar (jazz)',
    27: 'Electric Guitar (clean)', 28: 'Electric Guitar (muted)', 29: 'Overdriven Guitar',
    30: 'Distortion Guitar', 31: 'Guitar harmonics',

    # Bass
    32: 'Acoustic Bass', 33: 'Electric Bass (finger)', 34: 'Electric Bass (pick)',
    35: 'Fretless Bass', 36: 'Slap Bass 1', 37: 'Slap Bass 2', 38: 'Synth Bass 1', 39: 'Synth Bass 2',

    # Drums (channel 10, but programs for other uses)
    128: 'Drum Kit',  # Special designation for channel 10
}

# Drum note mapping (MIDI note numbers for drums on channel 10)
DRUM_NOTES = {
    35: 'Acoustic Bass Drum', 36: 'Bass Drum 1', 37: 'Side Stick', 38: 'Acoustic Snare',
    39: 'Hand Clap', 40: 'Electric Snare', 41: 'Low Floor Tom', 42: 'Closed Hi Hat',
    43: 'High Floor Tom', 44: 'Pedal Hi-Hat', 45: 'Low Tom', 46: 'Open Hi-Hat',
    47: 'Low-Mid Tom', 48: 'Hi-Mid Tom', 49: 'Crash Cymbal 1', 50: 'High Tom',
    51: 'Ride Cymbal 1', 52: 'Chinese Cymbal', 53: 'Ride Bell', 54: 'Tambourine',
    55: 'Splash Cymbal', 56: 'Cowbell', 57: 'Crash Cymbal 2', 58: 'Vibraslap',
    59: 'Ride Cymbal 2', 60: 'Hi Bongo', 61: 'Low Bongo', 62: 'Mute Hi Conga',
    63: 'Open Hi Conga', 64: 'Low Conga', 65: 'High Timbale', 66: 'Low Timbale',
}


def analyze_midi_file(midi_path, label=""):
    """Analyze MIDI file for instrument distribution."""
    mid = mido.MidiFile(midi_path)

    instrument_notes = defaultdict(list)
    channel_programs = {}  # Track program changes per channel

    for track_idx, track in enumerate(mid.tracks):
        current_time = 0
        for msg in track:
            current_time += msg.time

            if msg.type == 'program_change':
                channel_programs[msg.channel] = msg.program

            elif msg.type == 'note_on' and msg.velocity > 0:
                # Determine instrument
                if msg.channel == 9:  # Drum channel
                    instrument = 'Drums'
                    instrument_id = 128
                else:
                    instrument_id = channel_programs.get(msg.channel, 0)  # Default to piano
                    instrument = PROGRAM_TO_INSTRUMENT.get(instrument_id, f'Program {instrument_id}')

                note_info = {
                    'pitch': msg.note,
                    'velocity': msg.velocity,
                    'channel': msg.channel,
                    'program': instrument_id,
                    'time': current_time,
                    'track': track_idx,
                    'instrument_name': instrument,
                    'is_drum': msg.channel == 9
                }

                # Add drum-specific info
                if msg.channel == 9:
                    note_info['drum_name'] = DRUM_NOTES.get(msg.note, f'Drum {msg.note}')

                instrument_notes[instrument].append(note_info)

    # Generate statistics
    stats = {
        'label': label,
        'total_notes': sum(len(notes) for notes in instrument_notes.values()),
        'instruments': {}
    }

    for instrument, notes in instrument_notes.items():
        stats['instruments'][instrument] = {
            'count': len(notes),
            'percentage': len(notes) / stats['total_notes'] * 100 if stats['total_notes'] > 0 else 0,
            'pitch_range': [min(n['pitch'] for n in notes), max(n['pitch'] for n in notes)] if notes else [0, 0],
            'channels': list(set(n['channel'] for n in notes)),
            'programs': list(set(n['program'] for n in notes)),
            'sample_notes': notes[:5]  # First 5 notes as examples
        }

    return stats, instrument_notes


def analyze_raw_model_output(model, audio, sr):
    """Analyze raw model outputs before MIDI conversion."""
    print("\n--- Analyzing Raw Model Output ---")

    # Get raw predictions
    features = model.preprocess(audio, sr)
    outputs = model.forward(features)

    # Analyze output structure
    if isinstance(outputs, dict):
        print(f"Output type: dict with keys: {outputs.keys()}")
        if 'predictions' in outputs:
            pred = outputs['predictions']
            print(f"Predictions shape: {pred.shape if hasattr(pred, 'shape') else type(pred)}")
    elif isinstance(outputs, (list, tuple)):
        print(f"Output type: {type(outputs).__name__} with {len(outputs)} elements")
    else:
        print(f"Output type: {type(outputs)}")

    return outputs


def test_with_filtering(audio, sr, model_name='mt3_pytorch'):
    """Test transcription with various filtering strategies."""
    results = {}

    # Load model
    model = load_model(model_name)

    # 1. Original transcription
    print("\n1. Original Transcription")
    midi_original = transcribe(audio, model=model_name, sr=sr)
    midi_original_path = output_dir / "original_transcription.mid"
    midi_original.save(str(midi_original_path))

    stats_original, notes_original = analyze_midi_file(midi_original_path, "Original")
    results['original'] = stats_original

    # 2. Try to access model internals for filtering
    print("\n2. Attempting to filter by confidence/velocity")

    # Create filtered MIDI by removing low-velocity notes
    mid_filtered = mido.MidiFile()
    mid_filtered.ticks_per_beat = midi_original.ticks_per_beat

    velocity_threshold = 40  # Filter out quiet notes

    for track_idx, track in enumerate(midi_original.tracks):
        new_track = mido.MidiTrack()
        for msg in track:
            if msg.type == 'note_on':
                if msg.velocity >= velocity_threshold or msg.channel == 9:  # Keep drums
                    new_track.append(msg)
                else:
                    # Replace with note_off
                    new_track.append(mido.Message('note_off', note=msg.note,
                                                 velocity=0, time=msg.time))
            else:
                new_track.append(msg)
        mid_filtered.tracks.append(new_track)

    midi_filtered_path = output_dir / "velocity_filtered_transcription.mid"
    mid_filtered.save(str(midi_filtered_path))

    stats_filtered, notes_filtered = analyze_midi_file(midi_filtered_path, "Velocity Filtered")
    results['velocity_filtered'] = stats_filtered

    # 3. Create drum-only version
    print("\n3. Creating drum-only version")

    mid_drums_only = mido.MidiFile()
    mid_drums_only.ticks_per_beat = midi_original.ticks_per_beat

    for track_idx, track in enumerate(midi_original.tracks):
        new_track = mido.MidiTrack()
        for msg in track:
            # Only keep messages on drum channel (9) or meta messages
            if hasattr(msg, 'channel'):
                if msg.channel == 9:
                    new_track.append(msg)
                elif msg.type == 'program_change':
                    # Change all programs to drums
                    new_msg = msg.copy(program=0, channel=9)
                    new_track.append(new_msg)
                else:
                    # Skip non-drum channel messages
                    pass
            else:
                # Keep meta messages (tempo, time signature, etc.)
                new_track.append(msg)
        if new_track:
            mid_drums_only.tracks.append(new_track)

    midi_drums_path = output_dir / "drums_only_transcription.mid"
    mid_drums_only.save(str(midi_drums_path))

    stats_drums, notes_drums = analyze_midi_file(midi_drums_path, "Drums Only")
    results['drums_only'] = stats_drums

    return results


def compare_models(audio, sr):
    """Compare instrument leakage across different MT3 models."""
    print("\n" + "="*60)
    print("COMPARING MODELS")
    print("="*60)

    models = ['mt3_pytorch', 'mr_mt3', 'yourmt3']
    comparison = {}

    for model_name in models:
        print(f"\n--- Testing {model_name} ---")
        try:
            midi = transcribe(audio, model=model_name, sr=sr)
            midi_path = output_dir / f"{model_name}_transcription.mid"
            midi.save(str(midi_path))

            stats, notes = analyze_midi_file(midi_path, model_name)
            comparison[model_name] = stats

            # Print summary
            print(f"{model_name} results:")
            for instrument, info in stats['instruments'].items():
                print(f"  {instrument}: {info['count']} notes ({info['percentage']:.1f}%)")

        except Exception as e:
            print(f"Error with {model_name}: {e}")
            comparison[model_name] = {'error': str(e)}

    return comparison


def create_visualizations(audio, sr, midi_path):
    """Create diagnostic visualizations."""
    print("\n--- Creating Visualizations ---")

    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    # 1. Waveform
    times = np.linspace(0, len(audio)/sr, len(audio))
    axes[0].plot(times, audio, linewidth=0.5)
    axes[0].set_title('Audio Waveform')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Amplitude')
    axes[0].grid(True, alpha=0.3)

    # 2. Spectrogram with MIDI overlay
    D = librosa.stft(audio, n_fft=2048, hop_length=512)
    D_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    img = librosa.display.specshow(D_db, sr=sr, x_axis='time', y_axis='hz',
                                   hop_length=512, ax=axes[1])
    axes[1].set_title('Spectrogram with Transcribed Notes')
    axes[1].set_ylim(0, 2000)  # Focus on lower frequencies for drums

    # Overlay MIDI notes
    mid = mido.MidiFile(midi_path)
    current_time = 0
    colors = {9: 'red', 0: 'blue', 1: 'green'}  # Channel colors

    for track in mid.tracks:
        current_time = 0
        for msg in track:
            current_time += msg.time * (60 / (500000 * 2))  # Convert to seconds (approximate)

            if msg.type == 'note_on' and msg.velocity > 0:
                freq = librosa.midi_to_hz(msg.note)
                color = colors.get(msg.channel, 'yellow')
                alpha = 0.7 if msg.channel == 9 else 0.3  # Highlight drums

                axes[1].scatter(current_time, freq, c=color, alpha=alpha, s=20)

    axes[1].legend(['Drums (ch10)', 'Piano (ch1)', 'Other'], loc='upper right')

    # 3. Instrument distribution over time
    time_bins = np.linspace(0, len(audio)/sr, 50)
    instrument_timeline = defaultdict(lambda: np.zeros(len(time_bins)-1))

    mid = mido.MidiFile(midi_path)
    for track in mid.tracks:
        current_time = 0
        for msg in track:
            current_time += msg.time * (60 / (500000 * 2))

            if msg.type == 'note_on' and msg.velocity > 0:
                instrument = 'Drums' if msg.channel == 9 else 'Piano/Other'
                bin_idx = np.searchsorted(time_bins, current_time) - 1
                if 0 <= bin_idx < len(time_bins) - 1:
                    instrument_timeline[instrument][bin_idx] += 1

    # Stack plot
    bottom = np.zeros(len(time_bins)-1)
    for instrument, counts in instrument_timeline.items():
        axes[2].bar(time_bins[:-1], counts, width=time_bins[1]-time_bins[0],
                   bottom=bottom, label=instrument, alpha=0.7)
        bottom += counts

    axes[2].set_title('Instrument Distribution Over Time')
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Note Count')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    viz_path = output_dir / "spectrogram_with_labels.png"
    plt.savefig(viz_path, dpi=150)
    plt.close()

    print(f"Visualization saved to: {viz_path}")


def main():
    """Main investigation routine."""
    print("MT3-PYTORCH INSTRUMENT LEAKAGE INVESTIGATION")
    print("="*60)

    # Load drum audio
    drum_file = "assets/HappySounds_120bpm_Drums_drum_120BPM_BANDLAB.wav"
    print(f"Test file: {drum_file}")

    audio, sr = load_audio(drum_file, sr=16000)
    print(f"Audio loaded: {len(audio)/sr:.2f}s @ {sr}Hz")

    # Initialize debug log
    debug_log = []
    debug_log.append(f"Investigation started for: {drum_file}")
    debug_log.append(f"Audio duration: {len(audio)/sr:.2f}s, Sample rate: {sr}Hz")

    # Step 1: Reproduce issue
    print("\n" + "="*60)
    print("STEP 1: REPRODUCING ISSUE")
    print("="*60)

    midi_original = transcribe(audio, model='mt3_pytorch', sr=sr)
    midi_path = output_dir / "transcription_full.mid"
    midi_original.save(str(midi_path))

    stats, notes = analyze_midi_file(midi_path, "MT3-PyTorch Original")

    print("\nInstrument Distribution:")
    for instrument, info in stats['instruments'].items():
        print(f"  {instrument}: {info['count']} notes ({info['percentage']:.1f}%)")
        debug_log.append(f"{instrument}: {info['count']} notes ({info['percentage']:.1f}%)")

        if instrument != 'Drums' and info['count'] > 0:
            print(f"    ⚠️  Non-drum instrument detected!")
            print(f"    Channels: {info['channels']}")
            print(f"    Programs: {info['programs']}")
            print(f"    Pitch range: {info['pitch_range']}")

    # Save raw transcription data
    with open(output_dir / "transcription_full.json", 'w') as f:
        json.dump(stats, f, indent=2, default=str)

    # Step 2: Analyze raw model output
    print("\n" + "="*60)
    print("STEP 2: ANALYZING RAW MODEL OUTPUT")
    print("="*60)

    model = load_model('mt3_pytorch')
    raw_output = analyze_raw_model_output(model, audio, sr)

    # Step 3: Test filtering strategies
    print("\n" + "="*60)
    print("STEP 3: TESTING FILTERING STRATEGIES")
    print("="*60)

    filtering_results = test_with_filtering(audio, sr)

    # Save filtered results
    with open(output_dir / "transcription_filtered.json", 'w') as f:
        json.dump(filtering_results, f, indent=2, default=str)

    # Step 4: Compare with other models
    print("\n" + "="*60)
    print("STEP 4: COMPARING WITH OTHER MODELS")
    print("="*60)

    model_comparison = compare_models(audio, sr)

    # Step 5: Create visualizations
    print("\n" + "="*60)
    print("STEP 5: CREATING VISUALIZATIONS")
    print("="*60)

    create_visualizations(audio, sr, midi_path)

    # Generate summary report
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    # Write instrument distribution summary
    with open(output_dir / "instrument_distribution.txt", 'w') as f:
        f.write("INSTRUMENT DISTRIBUTION ANALYSIS\n")
        f.write("="*60 + "\n\n")

        f.write("MT3-PyTorch Original Transcription:\n")
        f.write("-"*40 + "\n")
        for instrument, info in stats['instruments'].items():
            f.write(f"{instrument}: {info['count']} notes ({info['percentage']:.1f}%)\n")

        f.write("\n\nModel Comparison:\n")
        f.write("-"*40 + "\n")
        for model_name, model_stats in model_comparison.items():
            if 'error' not in model_stats:
                f.write(f"\n{model_name}:\n")
                for instrument, info in model_stats.get('instruments', {}).items():
                    f.write(f"  {instrument}: {info['count']} notes ({info['percentage']:.1f}%)\n")

        f.write("\n\nFiltering Results:\n")
        f.write("-"*40 + "\n")
        for filter_name, filter_stats in filtering_results.items():
            f.write(f"\n{filter_name}:\n")
            f.write(f"  Total notes: {filter_stats['total_notes']}\n")
            for instrument, info in filter_stats.get('instruments', {}).items():
                f.write(f"  {instrument}: {info['count']} notes ({info['percentage']:.1f}%)\n")

    # Write debug log
    with open(output_dir / "debug_log.txt", 'w') as f:
        f.write("MT3-PYTORCH INSTRUMENT LEAKAGE DEBUG LOG\n")
        f.write("="*60 + "\n\n")
        for line in debug_log:
            f.write(line + "\n")

        f.write("\n\nKey Findings:\n")
        f.write("-"*40 + "\n")

        # Calculate leakage percentage
        total_notes = stats['total_notes']
        drum_notes = stats['instruments'].get('Drums', {}).get('count', 0)
        non_drum_notes = total_notes - drum_notes
        leakage_percentage = (non_drum_notes / total_notes * 100) if total_notes > 0 else 0

        f.write(f"Total notes transcribed: {total_notes}\n")
        f.write(f"Drum notes: {drum_notes}\n")
        f.write(f"Non-drum notes (leakage): {non_drum_notes}\n")
        f.write(f"Leakage percentage: {leakage_percentage:.1f}%\n")

        if leakage_percentage > 10:
            f.write("\n⚠️  SIGNIFICANT INSTRUMENT LEAKAGE DETECTED\n")
            f.write("The model is incorrectly classifying drum sounds as melodic instruments.\n")

    print(f"\nResults saved to: {output_dir}/")
    print(f"Leakage detected: {leakage_percentage:.1f}% non-drum notes in drum-only audio")

    return stats, filtering_results, model_comparison


if __name__ == "__main__":
    stats, filtering, comparison = main()