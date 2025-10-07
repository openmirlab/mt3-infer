"""
MIDI-to-Audio Synthesis Demo

Shows how to use midi_to_audio() for quality checking transcriptions.
"""

from pathlib import Path
from mt3_infer import transcribe
from mt3_infer.utils.audio import load_audio
from mt3_infer.utils.midi import midi_to_audio, count_notes
import soundfile as sf

def main():
    print("=" * 60)
    print("MIDI-to-Audio Synthesis Demo")
    print("=" * 60)
    
    # Load test audio
    audio_path = Path(__file__).parent.parent / "assets" / "HappySounds_120bpm_Drums_drum_120BPM_BANDLAB.wav"
    print(f"\nLoading audio: {audio_path.name}")
    audio, sr = load_audio(str(audio_path))
    
    # Use first 5 seconds for quick test
    audio_short = audio[:16000*5]
    
    # Transcribe with different models
    models = [
        ("fast", "MR-MT3 (57x real-time)"),
        ("accurate", "MT3-PyTorch (12x real-time)")
    ]
    
    for model_alias, model_name in models:
        print(f"\n{'=' * 60}")
        print(f"Model: {model_name}")
        print(f"{'=' * 60}")
        
        # Transcribe
        print(f"Transcribing...")
        midi = transcribe(audio_short, model=model_alias, sr=sr)
        notes = count_notes(midi)
        print(f"✓ Transcribed {notes} notes")
        
        # Save MIDI
        midi_path = Path(__file__).parent.parent / "test_outputs" / f"synth_demo_{model_alias}.mid"
        midi.save(str(midi_path))
        print(f"✓ Saved MIDI to: {midi_path.name}")
        
        # Synthesize to audio
        print(f"Synthesizing MIDI to audio (using fluidsynth)...")
        synth_audio = midi_to_audio(midi, sr=44100)
        
        # Save synthesized audio
        audio_path = Path(__file__).parent.parent / "test_outputs" / f"synth_demo_{model_alias}.wav"
        sf.write(str(audio_path), synth_audio, 44100)
        
        duration = len(synth_audio) / 44100
        print(f"✓ Synthesized {duration:.2f}s of audio")
        print(f"✓ Saved to: {audio_path.name}")
        print(f"\n💡 Listen to {audio_path.name} to hear the transcription!")
    
    print(f"\n{'=' * 60}")
    print("Demo complete!")
    print("=" * 60)
    print("\nCompare the synthesized audio files to check transcription quality:")
    print("  - synth_demo_fast.wav (MR-MT3)")
    print("  - synth_demo_accurate.wav (MT3-PyTorch)")

if __name__ == "__main__":
    main()
