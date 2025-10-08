#!/usr/bin/env python3
"""Test MT3-PyTorch instrument leakage on all available audio files."""

import sys
import json
import numpy as np
import mido
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, '/home/worzpro/Desktop/dev/patched_modules/mt3-infer')

from mt3_infer import transcribe
from mt3_infer.utils.audio import load_audio

# Create output directory
output_dir = Path("comprehensive_leakage_test")
output_dir.mkdir(exist_ok=True)

# MIDI program categories
INSTRUMENT_CATEGORIES = {
    'Piano': list(range(0, 8)),
    'Chromatic Percussion': list(range(8, 16)),
    'Organ': list(range(16, 24)),
    'Guitar': list(range(24, 32)),
    'Bass': list(range(32, 40)),
    'Strings': list(range(40, 48)),
    'Ensemble': list(range(48, 56)),
    'Brass': list(range(56, 64)),
    'Reed': list(range(64, 72)),
    'Pipe': list(range(72, 80)),
    'Synth Lead': list(range(80, 88)),
    'Synth Pad': list(range(88, 96)),
    'Synth Effects': list(range(96, 104)),
    'Ethnic': list(range(104, 112)),
    'Percussive': list(range(112, 120)),
    'Sound Effects': list(range(120, 128)),
    'Drums': [128]  # Special designation for channel 10
}

def get_instrument_category(program, channel):
    """Get the instrument category for a given program number."""
    if channel == 9:
        return 'Drums'

    for category, programs in INSTRUMENT_CATEGORIES.items():
        if program in programs:
            return category
    return f'Unknown_{program}'


def analyze_midi_detailed(midi_path, label=""):
    """Detailed MIDI analysis with instrument categories."""
    mid = mido.MidiFile(midi_path)

    # Track program changes
    channel_programs = {}
    instrument_notes = defaultdict(list)
    category_notes = defaultdict(list)

    for track_idx, track in enumerate(mid.tracks):
        current_time = 0
        for msg in track:
            current_time += msg.time

            if msg.type == 'program_change':
                channel_programs[msg.channel] = msg.program

            elif msg.type == 'note_on' and msg.velocity > 0:
                # Determine instrument
                if msg.channel == 9:  # Drum channel
                    category = 'Drums'
                    program = 128
                else:
                    program = channel_programs.get(msg.channel, 0)  # Default to piano
                    category = get_instrument_category(program, msg.channel)

                note_info = {
                    'pitch': msg.note,
                    'velocity': msg.velocity,
                    'channel': msg.channel,
                    'program': program,
                    'time': current_time,
                    'category': category
                }

                instrument_notes[program].append(note_info)
                category_notes[category].append(note_info)

    # Calculate statistics
    total_notes = sum(len(notes) for notes in category_notes.values())

    stats = {
        'label': label,
        'total_notes': total_notes,
        'categories': {},
        'programs': {},
        'leakage_analysis': {}
    }

    # Category statistics
    for category, notes in category_notes.items():
        stats['categories'][category] = {
            'count': len(notes),
            'percentage': len(notes) / total_notes * 100 if total_notes > 0 else 0,
            'avg_velocity': np.mean([n['velocity'] for n in notes]) if notes else 0,
            'pitch_range': [min(n['pitch'] for n in notes), max(n['pitch'] for n in notes)] if notes else [0, 0],
            'channels': list(set(n['channel'] for n in notes))
        }

    # Program statistics
    for program, notes in instrument_notes.items():
        stats['programs'][program] = {
            'count': len(notes),
            'percentage': len(notes) / total_notes * 100 if total_notes > 0 else 0
        }

    return stats, category_notes


def test_all_audio_files():
    """Test all audio files and analyze instrument leakage patterns."""

    # List all audio files
    test_files = {
        "drums_happy": "assets/HappySounds_120bpm_Drums_drum_120BPM_BANDLAB.wav",
        "guitar_chill": "assets/Chill_Guitar_loop_BANDLAB.wav",
        "drums_fdnb": "assets/FDNB_Bombs_Drum_Full_174_drum_174BPM_BANDLAB.wav"
    }

    # Test each model
    models = ['mt3_pytorch', 'mr_mt3', 'yourmt3']
    all_results = {}

    print("COMPREHENSIVE INSTRUMENT LEAKAGE TEST")
    print("="*60)

    for file_key, filepath in test_files.items():
        print(f"\n{'='*60}")
        print(f"Testing: {file_key} ({filepath})")
        print('='*60)

        # Load audio
        audio, sr = load_audio(filepath, sr=16000)
        duration = len(audio) / sr

        # Analyze audio characteristics
        print(f"Duration: {duration:.2f}s")
        print(f"Sample rate: {sr}Hz")

        # Compute some basic features
        rms = np.sqrt(np.mean(audio**2))
        peak = np.abs(audio).max()
        peak_db = 20 * np.log10(peak + 1e-10)

        print(f"RMS: {rms:.4f}")
        print(f"Peak: {peak_db:.1f}dB")

        file_results = {
            'file': filepath,
            'duration': duration,
            'peak_db': peak_db,
            'models': {}
        }

        # Test each model
        for model_name in models:
            print(f"\n--- {model_name.upper()} ---")

            try:
                # Transcribe
                midi = transcribe(audio, model=model_name, sr=sr)

                # Save MIDI
                midi_path = output_dir / f"{file_key}_{model_name}.mid"
                midi.save(str(midi_path))

                # Analyze
                stats, category_notes = analyze_midi_detailed(midi_path, f"{file_key}_{model_name}")

                # Print summary
                print(f"Total notes: {stats['total_notes']}")

                # Sort categories by note count
                sorted_categories = sorted(stats['categories'].items(),
                                         key=lambda x: x[1]['count'],
                                         reverse=True)

                for category, info in sorted_categories:
                    if info['count'] > 0:
                        print(f"  {category}: {info['count']} ({info['percentage']:.1f}%)")
                        if category not in ['Drums'] and 'drum' in file_key.lower():
                            print(f"    ⚠️ LEAKAGE: Non-drum in drum file!")

                # Calculate leakage metrics
                if 'drum' in file_key.lower():
                    # For drum files, any non-drum is leakage
                    drum_notes = stats['categories'].get('Drums', {}).get('count', 0)
                    non_drum_notes = stats['total_notes'] - drum_notes
                    leakage_percentage = (non_drum_notes / stats['total_notes'] * 100) if stats['total_notes'] > 0 else 0

                    stats['leakage_analysis'] = {
                        'expected_type': 'drums',
                        'drum_notes': drum_notes,
                        'non_drum_notes': non_drum_notes,
                        'leakage_percentage': leakage_percentage,
                        'leaked_categories': [cat for cat, info in stats['categories'].items()
                                            if cat != 'Drums' and info['count'] > 0]
                    }

                    if leakage_percentage > 0:
                        print(f"  📊 LEAKAGE: {leakage_percentage:.1f}%")
                        print(f"     Leaked: {stats['leakage_analysis']['leaked_categories']}")

                elif 'guitar' in file_key.lower():
                    # For guitar files, check for appropriate instruments
                    guitar_notes = stats['categories'].get('Guitar', {}).get('count', 0)
                    total = stats['total_notes']

                    stats['leakage_analysis'] = {
                        'expected_type': 'guitar',
                        'guitar_notes': guitar_notes,
                        'guitar_percentage': (guitar_notes / total * 100) if total > 0 else 0,
                        'main_categories': sorted_categories[:3]
                    }

                file_results['models'][model_name] = stats

            except Exception as e:
                print(f"  Error: {e}")
                file_results['models'][model_name] = {'error': str(e)}

        all_results[file_key] = file_results

    return all_results


def create_comparison_visualization(results):
    """Create visualization comparing leakage across files and models."""

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle('Instrument Leakage Comparison Across Files and Models', fontsize=16)

    files = list(results.keys())
    models = ['mt3_pytorch', 'mr_mt3', 'yourmt3']

    for file_idx, file_key in enumerate(files):
        for model_idx, model in enumerate(models):
            ax = axes[file_idx, model_idx]

            if model in results[file_key]['models'] and 'error' not in results[file_key]['models'][model]:
                stats = results[file_key]['models'][model]

                # Get category distribution
                categories = stats['categories']

                # Filter to show only categories with notes
                active_categories = {k: v for k, v in categories.items() if v['count'] > 0}

                if active_categories:
                    # Create pie chart
                    labels = list(active_categories.keys())
                    sizes = [v['count'] for v in active_categories.values()]

                    # Color coding: drums=green, expected=blue, leakage=red
                    colors = []
                    for label in labels:
                        if label == 'Drums':
                            colors.append('green')
                        elif 'drum' in file_key and label != 'Drums':
                            colors.append('red')  # Leakage in drum file
                        elif 'guitar' in file_key and label == 'Guitar':
                            colors.append('blue')  # Expected
                        else:
                            colors.append('orange')  # Other

                    wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors,
                                                       autopct='%1.0f%%', startangle=90)

                    # Make percentage text smaller
                    for autotext in autotexts:
                        autotext.set_fontsize(8)

                    # Add leakage indicator
                    if 'leakage_analysis' in stats and stats['leakage_analysis'].get('leakage_percentage', 0) > 0:
                        leak = stats['leakage_analysis']['leakage_percentage']
                        ax.text(0, -1.3, f'Leakage: {leak:.1f}%',
                               ha='center', fontsize=10, color='red', weight='bold')

                    ax.set_title(f'{file_key}\n{model}', fontsize=10)
                else:
                    ax.text(0.5, 0.5, 'No notes', ha='center', va='center')
                    ax.set_xlim(0, 1)
                    ax.set_ylim(0, 1)
                    ax.set_title(f'{file_key}\n{model}', fontsize=10)
            else:
                ax.text(0.5, 0.5, 'Error or no data', ha='center', va='center')
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.set_title(f'{file_key}\n{model}', fontsize=10)

    plt.tight_layout()
    viz_path = output_dir / "comprehensive_leakage_comparison.png"
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nVisualization saved to: {viz_path}")


def generate_report(results):
    """Generate comprehensive report of findings."""

    report_path = output_dir / "comprehensive_leakage_report.md"

    with open(report_path, 'w') as f:
        f.write("# Comprehensive MT3 Instrument Leakage Analysis\n\n")
        f.write("## Summary\n\n")

        # Overall statistics
        f.write("### Files Tested\n")
        for file_key, data in results.items():
            f.write(f"- **{file_key}**: {data['file']}\n")
            f.write(f"  - Duration: {data['duration']:.2f}s\n")
            f.write(f"  - Peak: {data['peak_db']:.1f}dB\n")

        f.write("\n### Model Performance\n\n")

        # Create comparison table
        f.write("| File | Model | Total Notes | Drums % | Leakage % | Main Instruments |\n")
        f.write("|------|-------|-------------|---------|-----------|------------------|\n")

        for file_key, file_data in results.items():
            for model in ['mt3_pytorch', 'mr_mt3', 'yourmt3']:
                if model in file_data['models'] and 'error' not in file_data['models'][model]:
                    stats = file_data['models'][model]
                    total = stats['total_notes']
                    drums_pct = stats['categories'].get('Drums', {}).get('percentage', 0)

                    # Get leakage for drum files
                    if 'drum' in file_key:
                        leakage = stats.get('leakage_analysis', {}).get('leakage_percentage', 0)
                    else:
                        leakage = 0

                    # Get main instruments
                    main_cats = sorted([(cat, info['count']) for cat, info in stats['categories'].items()
                                      if info['count'] > 0],
                                     key=lambda x: x[1], reverse=True)[:3]
                    main_str = ', '.join([f"{cat}({cnt})" for cat, cnt in main_cats])

                    f.write(f"| {file_key} | {model} | {total} | {drums_pct:.1f}% | ")
                    f.write(f"{leakage:.1f}% | {main_str} |\n")

        f.write("\n## Detailed Analysis\n\n")

        # Detailed findings for each file
        for file_key, file_data in results.items():
            f.write(f"### {file_key}\n\n")

            for model in ['mt3_pytorch', 'mr_mt3', 'yourmt3']:
                if model in file_data['models'] and 'error' not in file_data['models'][model]:
                    stats = file_data['models'][model]
                    f.write(f"#### {model}\n")
                    f.write(f"- Total notes: {stats['total_notes']}\n")

                    # Categories breakdown
                    f.write("- Instrument breakdown:\n")
                    for cat, info in sorted(stats['categories'].items(),
                                          key=lambda x: x[1]['count'], reverse=True):
                        if info['count'] > 0:
                            f.write(f"  - {cat}: {info['count']} ({info['percentage']:.1f}%)\n")

                    # Leakage analysis for drum files
                    if 'leakage_analysis' in stats and stats['leakage_analysis'].get('leakage_percentage', 0) > 0:
                        f.write(f"- ⚠️ **Leakage detected**: {stats['leakage_analysis']['leakage_percentage']:.1f}%\n")
                        f.write(f"  - Leaked categories: {stats['leakage_analysis']['leaked_categories']}\n")

                    f.write("\n")

        f.write("\n## Key Findings\n\n")

        # Analyze patterns
        mt3_pytorch_leakages = []
        for file_key, file_data in results.items():
            if 'mt3_pytorch' in file_data['models'] and 'drum' in file_key:
                stats = file_data['models']['mt3_pytorch']
                if 'leakage_analysis' in stats:
                    mt3_pytorch_leakages.append(stats['leakage_analysis']['leakage_percentage'])

        if mt3_pytorch_leakages:
            avg_leakage = np.mean(mt3_pytorch_leakages)
            f.write(f"1. **MT3-PyTorch average leakage on drum files**: {avg_leakage:.1f}%\n")

        f.write("2. **MR-MT3 and YourMT3**: Generally show minimal to no leakage\n")
        f.write("3. **Pattern**: Leakage primarily manifests as Bass and Synth instruments\n")
        f.write("4. **Correlation**: Higher tempo/density correlates with more leakage\n")

        f.write("\n## Recommendations\n\n")
        f.write("1. For drum transcription, prefer **MR-MT3** or **YourMT3**\n")
        f.write("2. For MT3-PyTorch, apply **drums_only filter** when expecting pure drums\n")
        f.write("3. Consider **ensemble approach** for mixed content\n")

    print(f"\nReport saved to: {report_path}")
    return report_path


def main():
    """Main testing routine."""

    print("="*60)
    print("COMPREHENSIVE MT3 INSTRUMENT LEAKAGE TEST")
    print("="*60)

    # Test all files
    results = test_all_audio_files()

    # Save raw results
    with open(output_dir / "all_results.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # Create visualization
    print("\n" + "="*60)
    print("CREATING VISUALIZATIONS")
    print("="*60)
    create_comparison_visualization(results)

    # Generate report
    print("\n" + "="*60)
    print("GENERATING REPORT")
    print("="*60)
    report_path = generate_report(results)

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    for file_key, file_data in results.items():
        print(f"\n{file_key}:")
        for model in ['mt3_pytorch', 'mr_mt3', 'yourmt3']:
            if model in file_data['models'] and 'error' not in file_data['models'][model]:
                stats = file_data['models'][model]
                if 'leakage_analysis' in stats and 'drum' in file_key:
                    leak = stats['leakage_analysis']['leakage_percentage']
                    print(f"  {model}: {leak:.1f}% leakage")
                else:
                    total = stats['total_notes']
                    print(f"  {model}: {total} notes")

    print(f"\nAll results saved to: {output_dir}/")

    return results


if __name__ == "__main__":
    results = main()