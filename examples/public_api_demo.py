"""
MT3-Infer Public API Demo

Demonstrates the high-level public API for music transcription.
"""

import numpy as np
import soundfile as sf
from pathlib import Path

# Import public API
from mt3_infer import (
    transcribe,
    load_model,
    list_models,
    get_model_info,
    clear_cache
)
from mt3_infer.utils.midi import midi_to_audio
from mt3_infer.utils.audio import load_audio

def demo_simple_transcription():
    """Demo 1: Simplest possible usage."""
    print("=" * 60)
    print("Demo 1: Simple Transcription")
    print("=" * 60)
    
    # Load test audio
    audio_path = Path(__file__).parent.parent / "assets" / "HappySounds_120bpm_Drums_drum_120BPM_BANDLAB.wav"
    audio, sr = load_audio(str(audio_path))
    
    # Transcribe with default model (one line!)
    print("Transcribing with default model...")
    midi = transcribe(audio, sr=sr)
    
    # Save result
    output_path = Path(__file__).parent.parent / "test_outputs" / "api_demo_default.mid"
    midi.save(str(output_path))
    print(f"✓ Saved to: {output_path}")
    
    # Count notes
    from mt3_infer.utils.midi import count_notes
    print(f"✓ Detected {count_notes(midi)} notes")
    print()


def demo_model_aliases():
    """Demo 2: Using model aliases."""
    print("=" * 60)
    print("Demo 2: Model Aliases")
    print("=" * 60)
    
    # Load test audio
    audio_path = Path(__file__).parent.parent / "assets" / "HappySounds_120bpm_Drums_drum_120BPM_BANDLAB.wav"
    audio, sr = load_audio(str(audio_path))
    
    # Try different model aliases (skip multitask - known YourMT3 checkpoint path issue)
    aliases = ["fast", "accurate"]
    
    for alias in aliases:
        print(f"\nTranscribing with '{alias}' model...")
        midi = transcribe(audio, model=alias, sr=sr)
        
        from mt3_infer.utils.midi import count_notes
        notes = count_notes(midi)
        
        # Get model info
        info = get_model_info(alias)
        print(f"  → Model: {info['name']}")
        print(f"  → Framework: {info['framework']}")
        print(f"  → Notes detected: {notes}")
        
        # Save
        output_path = Path(__file__).parent.parent / "test_outputs" / f"api_demo_{alias}.mid"
        midi.save(str(output_path))
        print(f"  → Saved to: {output_path}")
    
    print()


def demo_advanced_usage():
    """Demo 3: Advanced usage with explicit model loading."""
    print("=" * 60)
    print("Demo 3: Advanced Usage")
    print("=" * 60)
    
    # Load test audio
    audio_path = Path(__file__).parent.parent / "assets" / "HappySounds_120bpm_Drums_drum_120BPM_BANDLAB.wav"
    audio, sr = load_audio(str(audio_path))
    
    # Load model explicitly
    print("Loading MT3-PyTorch model...")
    model = load_model("mt3_pytorch", device="cuda")
    
    # Use model multiple times (cached, no reloading)
    print("Transcribing (1st call - model already loaded)...")
    midi1 = model.transcribe(audio, sr=sr)
    
    print("Transcribing (2nd call - using cached model)...")
    midi2 = model.transcribe(audio, sr=sr)
    
    # Verify determinism
    from mt3_infer.utils.midi import midi_to_hash
    hash1 = midi_to_hash(midi1)
    hash2 = midi_to_hash(midi2)
    print(f"\n✓ Deterministic: {hash1 == hash2}")
    print(f"  Hash: {hash1[:16]}...")
    print()


def demo_midi_synthesis():
    """Demo 4: Synthesize MIDI to audio for quality checking."""
    print("=" * 60)
    print("Demo 4: MIDI Synthesis (Quality Check)")
    print("=" * 60)
    
    # Load test audio
    audio_path = Path(__file__).parent.parent / "assets" / "HappySounds_120bpm_Drums_drum_120BPM_BANDLAB.wav"
    audio, sr = load_audio(str(audio_path))
    
    # Transcribe
    print("Transcribing with accurate model...")
    midi = transcribe(audio, model="accurate", sr=sr)
    
    # Synthesize MIDI back to audio
    print("Synthesizing MIDI to audio...")
    synth_audio = midi_to_audio(midi, sr=44100)
    
    # Save synthesized audio
    output_path = Path(__file__).parent.parent / "test_outputs" / "api_demo_synthesis.wav"
    sf.write(str(output_path), synth_audio, 44100)
    print(f"✓ Saved synthesized audio to: {output_path}")
    print(f"✓ Duration: {len(synth_audio) / 44100:.2f}s")
    print("\nYou can now listen to the synthesized MIDI to check transcription quality!")
    print()


def demo_model_registry():
    """Demo 5: Explore the model registry."""
    print("=" * 60)
    print("Demo 5: Model Registry")
    print("=" * 60)
    
    # List all available models
    models = list_models()
    
    print(f"Available models: {len(models)}\n")
    
    for model_id, info in models.items():
        print(f"{model_id}:")
        print(f"  Name: {info['name']}")
        print(f"  Description: {info['description']}")
        print(f"  Framework: {info['framework']}")
        print(f"  Checkpoint: {info['checkpoint']['path']}")
        print(f"  Speed: {info['metadata']['performance']['speed_x_realtime']}x real-time")
        print(f"  GPU Memory: {info['metadata']['performance']['peak_gpu_memory_mb']} MB")
        print(f"  Notes (test): {info['metadata']['performance']['notes_detected']}")
        print()


def demo_cleanup():
    """Demo 6: Memory management."""
    print("=" * 60)
    print("Demo 6: Memory Management")
    print("=" * 60)
    
    # Load multiple models (they get cached)
    print("Loading 2 models...")
    model1 = load_model("mr_mt3")
    model2 = load_model("mt3_pytorch")
    # Note: YourMT3 skipped due to known checkpoint path issue

    print("✓ All models loaded and cached")
    
    # Clear cache to free memory
    print("\nClearing model cache...")
    clear_cache()
    print("✓ Cache cleared, memory freed")
    print()


if __name__ == "__main__":
    # Run all demos
    demo_simple_transcription()
    demo_model_aliases()
    demo_advanced_usage()
    # demo_midi_synthesis()  # Skipped: pretty_midi has numpy compatibility issues
    demo_model_registry()
    demo_cleanup()
    
    print("=" * 60)
    print("All demos completed successfully!")
    print("=" * 60)
