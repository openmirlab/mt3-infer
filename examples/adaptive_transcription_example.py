"""
Example: Adaptive Preprocessing for Dense Drum Transcription

This script demonstrates how to use YourMT3's adaptive preprocessing feature
to transcribe dense drum patterns that would normally produce zero or very few notes.
"""

import numpy as np
import librosa
from pathlib import Path
import mido

from mt3_infer import load_model, transcribe


def count_notes(midi: mido.MidiFile) -> int:
    """Count total notes in MIDI file."""
    note_count = 0
    for track in midi.tracks:
        for msg in track:
            if msg.type == 'note_on' and msg.velocity > 0:
                note_count += 1
    return note_count


def analyze_density(audio: np.ndarray, sr: int) -> dict:
    """Analyze audio density to determine if adaptive mode is needed."""
    onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
    duration = len(audio) / sr
    onset_density = len(onset_frames) / duration

    tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)

    is_dense = (
        onset_density > 6.0 or
        (tempo > 150 and onset_density > 5) or
        (tempo > 160)
    )

    return {
        'onset_density': onset_density,
        'tempo': tempo,
        'duration': duration,
        'is_dense': is_dense
    }


def example_1_basic_usage():
    """Example 1: Basic adaptive transcription."""
    print("\n" + "="*80)
    print("Example 1: Basic Adaptive Transcription")
    print("="*80)

    # Load test audio (dense drum pattern)
    audio_path = "assets/FDNB_Bombs_Drum_Full_174_drum_174BPM_BANDLAB.wav"

    if not Path(audio_path).exists():
        print(f"⚠️  Test file not found: {audio_path}")
        print("Please provide a dense drum audio file to test.")
        return

    print(f"\nLoading: {audio_path}")
    audio, sr = librosa.load(audio_path, sr=16000, mono=True)
    print(f"Duration: {len(audio)/sr:.2f}s, Sample rate: {sr} Hz")

    # Analyze density
    analysis = analyze_density(audio, sr)
    print(f"\nAudio Analysis:")
    print(f"  Onset density: {analysis['onset_density']:.2f} onsets/sec")
    print(f"  Tempo: {analysis['tempo']:.1f} BPM")
    print(f"  Is dense: {analysis['is_dense']}")

    # Load model
    print("\nLoading YourMT3 model...")
    model = load_model("yourmt3", verbose=True)

    # Standard transcription (likely fails on dense drums)
    print("\n--- Standard Transcription ---")
    midi_standard = model.transcribe(audio, sr=sr)
    note_count_standard = count_notes(midi_standard)
    print(f"Result: {note_count_standard} notes")

    # Adaptive transcription
    print("\n--- Adaptive Transcription ---")
    midi_adaptive = model.transcribe(
        audio,
        sr=sr,
        adaptive=True,
        num_attempts=3
    )
    note_count_adaptive = count_notes(midi_adaptive)
    print(f"Result: {note_count_adaptive} notes")

    # Save results
    output_dir = Path("investigation_output")
    output_dir.mkdir(exist_ok=True)

    midi_standard.save(output_dir / "standard_transcription.mid")
    midi_adaptive.save(output_dir / "adaptive_transcription.mid")

    print(f"\n✓ Results saved to {output_dir}/")
    print(f"  Standard: {note_count_standard} notes")
    print(f"  Adaptive: {note_count_adaptive} notes")
    print(f"  Improvement: {note_count_adaptive - note_count_standard:+d} notes")


def example_2_high_level_api():
    """Example 2: Using the high-level transcribe() API."""
    print("\n" + "="*80)
    print("Example 2: High-Level API (One-Line Transcription)")
    print("="*80)

    # Load audio
    audio_path = "assets/FDNB_Bombs_Drum_Full_174_drum_174BPM_BANDLAB.wav"

    if not Path(audio_path).exists():
        print(f"⚠️  Test file not found: {audio_path}")
        return

    print(f"\nLoading: {audio_path}")
    audio, sr = librosa.load(audio_path, sr=16000, mono=True)

    # One-line transcription with adaptive mode
    print("\nTranscribing with adaptive mode...")
    midi = transcribe(
        audio,
        model="yourmt3",
        adaptive=True,
        num_attempts=3
    )

    note_count = count_notes(midi)
    print(f"\n✓ Transcribed: {note_count} notes")


def example_3_custom_configuration():
    """Example 3: Custom stretch factors and methods."""
    print("\n" + "="*80)
    print("Example 3: Custom Configuration")
    print("="*80)

    audio_path = "assets/FDNB_Bombs_Drum_Full_174_drum_174BPM_BANDLAB.wav"

    if not Path(audio_path).exists():
        print(f"⚠️  Test file not found: {audio_path}")
        return

    print(f"\nLoading: {audio_path}")
    audio, sr = librosa.load(audio_path, sr=16000, mono=True)

    # Load model
    model = load_model("yourmt3", verbose=True)

    # Custom configuration: aggressive stretching for very dense patterns
    print("\nTranscribing with custom stretch factors...")
    midi = model.transcribe(
        audio,
        sr=sr,
        adaptive=True,
        stretch_factors=[1.0, 0.85, 0.75, 0.65],  # More aggressive stretching
        stretch_method="auto",  # Use pyrubberband if available
        num_attempts=5,  # More attempts for better coverage
        return_all=False
    )

    note_count = count_notes(midi)
    print(f"\n✓ Transcribed: {note_count} notes")


def example_4_compare_all_results():
    """Example 4: Compare all transcription attempts."""
    print("\n" + "="*80)
    print("Example 4: Compare All Results")
    print("="*80)

    audio_path = "assets/FDNB_Bombs_Drum_Full_174_drum_174BPM_BANDLAB.wav"

    if not Path(audio_path).exists():
        print(f"⚠️  Test file not found: {audio_path}")
        return

    print(f"\nLoading: {audio_path}")
    audio, sr = librosa.load(audio_path, sr=16000, mono=True)

    # Load model
    model = load_model("yourmt3")

    # Get all results for comparison
    print("\nRunning multiple transcription attempts...")
    print("(This may take a minute...)\n")

    results = model.transcribe(
        audio,
        sr=sr,
        adaptive=True,
        stretch_factors=[1.0, 0.9, 0.8, 0.7],
        num_attempts=3,
        return_all=True  # Return all results
    )

    # Display results table
    print(f"{'Stretch Factor':<15} {'Attempt':<10} {'Notes':<10} {'Config':<30}")
    print("-" * 70)

    for midi, note_count, stretch_factor, config_name in results:
        attempt_num = config_name.split('_')[-1].replace('attempt', '')
        print(f"{stretch_factor:<15.2f} {attempt_num:<10} {note_count:<10} {config_name:<30}")

    # Find best result
    best_midi, best_count, best_stretch, best_config = max(results, key=lambda x: x[1])
    print("-" * 70)
    print(f"Best result: {best_config} with {best_count} notes (stretch={best_stretch})")

    # Save best result
    output_dir = Path("investigation_output")
    output_dir.mkdir(exist_ok=True)
    best_midi.save(output_dir / "best_transcription.mid")
    print(f"\n✓ Best result saved to {output_dir}/best_transcription.mid")


def example_5_batch_processing():
    """Example 5: Batch processing with automatic density detection."""
    print("\n" + "="*80)
    print("Example 5: Batch Processing with Density Detection")
    print("="*80)

    # Simulate multiple audio files
    audio_files = [
        "assets/FDNB_Bombs_Drum_Full_174_drum_174BPM_BANDLAB.wav",
        # Add more files here
    ]

    model = load_model("yourmt3")

    print("\nProcessing audio files...")
    print(f"{'File':<40} {'Density':<12} {'Mode':<12} {'Notes':<10}")
    print("-" * 80)

    for file_path in audio_files:
        if not Path(file_path).exists():
            print(f"{file_path:<40} SKIPPED (not found)")
            continue

        # Load audio
        audio, sr = librosa.load(file_path, sr=16000, mono=True)

        # Analyze density
        analysis = analyze_density(audio, sr)
        density = analysis['onset_density']

        # Use adaptive mode only for dense tracks
        use_adaptive = analysis['is_dense']
        mode = "adaptive" if use_adaptive else "standard"

        # Transcribe
        midi = model.transcribe(
            audio, sr=sr,
            adaptive=use_adaptive,
            num_attempts=3 if use_adaptive else 1
        )

        note_count = count_notes(midi)

        # Display result
        filename = Path(file_path).name[:37] + "..." if len(Path(file_path).name) > 40 else Path(file_path).name
        print(f"{filename:<40} {density:>8.2f}/s  {mode:<12} {note_count:<10}")

        # Save result
        output_dir = Path("investigation_output") / "batch"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_name = Path(file_path).stem + f"_{mode}.mid"
        midi.save(output_dir / output_name)

    print(f"\n✓ Results saved to investigation_output/batch/")


def main():
    """Run all examples."""
    print("\n" + "="*80)
    print("YourMT3 Adaptive Preprocessing Examples")
    print("="*80)

    # Run examples
    try:
        example_1_basic_usage()
    except Exception as e:
        print(f"⚠️  Example 1 failed: {e}")

    try:
        example_2_high_level_api()
    except Exception as e:
        print(f"⚠️  Example 2 failed: {e}")

    try:
        example_3_custom_configuration()
    except Exception as e:
        print(f"⚠️  Example 3 failed: {e}")

    try:
        example_4_compare_all_results()
    except Exception as e:
        print(f"⚠️  Example 4 failed: {e}")

    try:
        example_5_batch_processing()
    except Exception as e:
        print(f"⚠️  Example 5 failed: {e}")

    print("\n" + "="*80)
    print("All examples completed!")
    print("="*80)


if __name__ == "__main__":
    main()
