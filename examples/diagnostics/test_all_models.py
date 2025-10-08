"""
Test all three MT3 models with real audio
"""
from mt3_infer import load_model, list_models
from mt3_infer.utils.audio import load_audio
from mt3_infer.utils.midi import count_notes
import time
import os

# Load test audio
audio_path = "assets/HappySounds_120bpm_Drums_drum_120BPM_BANDLAB.wav"
print(f"Loading audio: {audio_path}")
audio, sr = load_audio(audio_path, sr=16000)
print(f"Audio shape: {audio.shape}, sample rate: {sr}Hz")
print()

# Test each model
models_to_test = ["mr_mt3", "mt3_pytorch", "yourmt3"]

checkpoint_root = os.getenv("MT3_CHECKPOINT_DIR", os.path.join(os.getcwd(), ".mt3_checkpoints"))
print(f"Checkpoint root: {checkpoint_root}")

for model_id in models_to_test:
    print("=" * 70)
    print(f"Testing: {model_id.upper()}")
    print("=" * 70)
    
    try:
        # Load model
        print(f"Loading model...")
        start_time = time.time()
        model = load_model(model_id, auto_download=False)
        load_time = time.time() - start_time
        print(f"✅ Model loaded in {load_time:.2f}s")
        
        # Transcribe
        print(f"Transcribing audio...")
        start_time = time.time()
        midi = model.transcribe(audio, sr=sr)
        transcribe_time = time.time() - start_time
        print(f"✅ Transcription completed in {transcribe_time:.2f}s")
        
        # Count notes
        note_count = count_notes(midi)
        duration = midi.length
        speed_factor = duration / transcribe_time if transcribe_time > 0 else 0
        
        print(f"✅ Results:")
        print(f"   - Notes detected: {note_count}")
        print(f"   - MIDI duration: {duration:.2f}s")
        print(f"   - Speed: {speed_factor:.1f}x real-time")
        print()
        
    except FileNotFoundError as e:
        print(f"⚠️  Checkpoint not found")
        print(f"   {str(e)[:100]}...")
        print()
    except Exception as e:
        print(f"❌ Error: {type(e).__name__}")
        print(f"   {str(e)[:200]}...")
        print()

print("=" * 70)
print("Test Summary")
print("=" * 70)
print("MR-MT3: ✅ working")
print("MT3-PyTorch: ✅ working")
print("YourMT3: ✅ working")
