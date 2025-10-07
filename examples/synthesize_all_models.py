"""
Synthesize audio from all three MT3 models for comparison.

Transcribes HappySounds test audio with MR-MT3, MT3-PyTorch, and YourMT3,
then synthesizes each MIDI back to audio for quality checking.
"""

from pathlib import Path
from mt3_infer import transcribe, load_model
from mt3_infer.utils.audio import load_audio
from mt3_infer.utils.midi import midi_to_audio, count_notes, get_midi_duration
import soundfile as sf

def main():
    print("=" * 70)
    print("MT3 Model Comparison: Transcription + Synthesis")
    print("=" * 70)
    
    # Load full audio
    audio_path = Path(__file__).parent.parent / "assets" / "HappySounds_120bpm_Drums_drum_120BPM_BANDLAB.wav"
    print(f"\nLoading audio: {audio_path.name}")
    audio, sr = load_audio(str(audio_path))
    print(f"✓ Loaded {len(audio)/sr:.2f}s of audio at {sr} Hz")
    
    # Models to test
    models = [
        ("mr_mt3", "MR-MT3", "Speed-optimized (57x real-time)"),
        ("mt3_pytorch", "MT3-PyTorch", "Official baseline (12x real-time)"),
        ("yourmt3", "YourMT3", "Multi-task (8-stem separation)"),
    ]
    
    results = []
    
    for model_id, model_name, description in models:
        print(f"\n{'=' * 70}")
        print(f"Model: {model_name}")
        print(f"Description: {description}")
        print(f"{'=' * 70}")
        
        try:
            # Transcribe
            print(f"[1/3] Transcribing with {model_name}...")
            midi = transcribe(audio, model=model_id, sr=sr)
            
            # Get MIDI stats
            notes = count_notes(midi)
            duration = get_midi_duration(midi)
            print(f"✓ Transcribed {notes} notes ({duration:.2f}s duration)")
            
            # Save MIDI
            midi_path = Path(__file__).parent.parent / "test_outputs" / f"comparison_{model_id}.mid"
            midi.save(str(midi_path))
            print(f"✓ Saved MIDI: {midi_path.name}")
            
            # Synthesize
            print(f"[2/3] Synthesizing MIDI to audio...")
            synth_audio = midi_to_audio(midi, sr=44100)
            
            # Save synthesized audio
            audio_path = Path(__file__).parent.parent / "test_outputs" / f"comparison_{model_id}.wav"
            sf.write(str(audio_path), synth_audio, 44100)
            
            synth_duration = len(synth_audio) / 44100
            print(f"✓ Synthesized {synth_duration:.2f}s of audio")
            print(f"✓ Saved audio: {audio_path.name}")
            
            # Store results
            results.append({
                'model': model_name,
                'model_id': model_id,
                'notes': notes,
                'midi_duration': duration,
                'synth_duration': synth_duration,
                'midi_file': midi_path.name,
                'audio_file': audio_path.name,
                'success': True
            })
            
            print(f"✓ {model_name} complete!")
            
        except Exception as e:
            print(f"✗ {model_name} failed: {e}")
            results.append({
                'model': model_name,
                'model_id': model_id,
                'success': False,
                'error': str(e)
            })
    
    # Summary
    print(f"\n{'=' * 70}")
    print("COMPARISON SUMMARY")
    print(f"{'=' * 70}")
    
    print(f"\n{'Model':<20} {'Notes':<10} {'MIDI File':<30}")
    print("-" * 70)
    
    for result in results:
        if result['success']:
            print(f"{result['model']:<20} {result['notes']:<10} {result['midi_file']:<30}")
        else:
            print(f"{result['model']:<20} {'FAILED':<10} {result.get('error', 'Unknown error'):<30}")
    
    # Audio files
    successful = [r for r in results if r['success']]
    if successful:
        print(f"\n{'=' * 70}")
        print("SYNTHESIZED AUDIO FILES (for quality checking):")
        print(f"{'=' * 70}")
        for result in successful:
            print(f"  🎵 {result['audio_file']}")
            print(f"     → {result['notes']} notes, {result['synth_duration']:.2f}s duration")
    
    print(f"\n{'=' * 70}")
    print(f"✓ Completed {len(successful)}/{len(models)} models successfully")
    print(f"{'=' * 70}")
    
    if len(successful) > 1:
        print("\n💡 TIP: Listen to the synthesized .wav files to compare transcription quality!")
        print("   The audio files are in test_outputs/comparison_*.wav")

if __name__ == "__main__":
    main()
