# kunato/mt3-pytorch Comprehensive Analysis

**Date:** 2025-10-06
**Repository:** https://github.com/kunato/mt3-pytorch
**Status:** ✅ **RECOMMENDED for adapter implementation**

## Executive Summary

kunato/mt3-pytorch is a **PyTorch port of Google's official Magenta MT3** that successfully avoids the T5X/JAX dependency hell we encountered. It provides:

1. ✅ Complete PyTorch implementation using HuggingFace Transformers
2. ✅ Pre-converted model weights (183MB, via Git LFS)
3. ✅ **Conversion script to load official Magenta checkpoints** (if needed)
4. ✅ Complete inference pipeline (audio → MIDI)
5. ⚠️ TensorFlow/DDSP dependency for spectrograms (solvable - same as MR-MT3)

**Recommendation:** Implement adapter using kunato's architecture. We can reuse our PyTorch-only spectrogram from MR-MT3 adapter to eliminate TensorFlow dependency.

---

## Repository Structure

```
kunato-mt3-pytorch/
├── inference.py              # InferenceHandler class (164 lines)
├── models/
│   └── t5.py                 # Custom T5 for continuous inputs (HuggingFace-based)
├── contrib/                  # MT3 core utilities (from Magenta)
│   ├── event_codec.py        # Event encoding (pitch, velocity, etc.)
│   ├── vocabularies.py       # Codec builder, vocab config
│   ├── note_sequences.py     # Note sequence handling
│   ├── spectrograms.py       # Mel spectrogram (TensorFlow/DDSP)
│   ├── run_length_encoding.py
│   └── metrics_utils.py
├── tools/
│   └── convert_weight.py     # T5X → PyTorch weight conversion (169 lines)
├── pretrained/
│   ├── config.json           # T5 config (512 d_model, 8 layers)
│   └── mt3.pth              # Model weights (183MB, Git LFS)
├── requirements.txt          # Dependencies
└── README.md
```

---

## Key Technical Details

### 1. Model Architecture

**Custom T5 for Audio Input** (`models/t5.py`):
- Based on HuggingFace `T5PreTrainedModel`
- **Key difference:** Encoder uses `proj = nn.Linear()` for continuous (audio) inputs instead of token embeddings
- Decoder uses standard token embeddings
- Config:
  ```json
  {
    "d_model": 512,
    "d_ff": 1024,
    "num_layers": 8,
    "num_decoder_layers": 8,
    "num_heads": 6,
    "vocab_size": 1536
  }
  ```

**Inference Pipeline** (`inference.py`):
```python
class InferenceHandler:
    def __init__(self, weight_path, device='cuda'):
        # Load config and PyTorch weights
        config = T5Config.from_dict(json.load(config_path))
        model = T5ForConditionalGeneration(config)
        model.load_state_dict(torch.load(weight_path))

    def inference(self, audio_path, outpath=None):
        # 1. Load audio (librosa, 16kHz)
        # 2. Preprocess (spectrogram + chunking)
        # 3. Model generation (batched)
        # 4. Decode to MIDI (note-seq)
```

---

### 2. Weight Conversion from Magenta MT3

**Critical Finding:** `tools/convert_weight.py` provides **T5X → PyTorch conversion**

**Process:**
1. Load Magenta T5X checkpoint (JAX/Flax pickle file)
2. Flatten T5X state dict
3. Map keys: `target/encoder/layers_0/attention/key/kernel` → `encoder.block.0.layer.0.SelfAttention.k.weight`
4. Transpose weights (JAX convention → PyTorch convention)
5. Save as PyTorch `.pth` file

**Key Code:**
```python
def convert_t5x_to_pt(config, flatten_statedict):
    # Maps ~100 keys from T5X format to PyTorch format
    # Handles kernel → weight renaming
    # Transposes linear layer weights (.T)
    # Returns PyTorch state dict

# Usage:
state_dict = load_t5x_statedict('mt3_flax_state_dict.pk')
pt_state_dict = convert_t5x_to_pt(config, state_dict)
torch.save(pt_state_dict, 'pretrained/mt3.pth')
```

**Implication:** We CAN load official Magenta weights without T5X at inference time! Just need T5X for one-time conversion.

---

### 3. Pretrained Weights

**File:** `pretrained/mt3.pth`
- **Size:** 183,672,643 bytes (183.7 MB)
- **Storage:** Git LFS pointer file
- **SHA256:** `b8a3807ed265059abd25ad7f68142c06c35e8f6144dcaa45bd55946a3745398f`

**Status:** Weights exist but need to be pulled via Git LFS:
```bash
cd refs/kunato-mt3-pytorch/
git lfs pull
```

---

### 4. Dependencies

**From `requirements.txt`:**
```
transformers==4.18.0   # HuggingFace (older version)
torch                  # PyTorch (no version specified)
librosa==0.9.1         # Audio loading
ddsp==3.3.4            # ⚠️ TensorFlow-based (for spectrograms)
t5==0.9.3              # ⚠️ Google's T5 library (TensorFlow)
note-seq==0.0.3        # MIDI handling
pretty-midi==0.2.9     # MIDI utilities
einops==0.4.1          # Tensor operations
```

**Dependency Issues:**
1. ❌ **ddsp==3.3.4**: TensorFlow-based, used in `contrib/spectrograms.py`
2. ❌ **t5==0.9.3**: Google's TensorFlow T5, used in `contrib/vocabularies.py` (line 24: `import t5.data`)
3. ⚠️ **transformers==4.18.0**: Old version (we use ~=4.30.2)

---

### 5. Spectrogram Implementation

**Problem:** `contrib/spectrograms.py` uses TensorFlow + DDSP:

```python
from ddsp import spectral_ops
import tensorflow as tf

def compute_spectrogram(samples, spectrogram_config):
    overlap = 1 - (spectrogram_config.hop_width / FFT_SIZE)
    return spectral_ops.compute_logmel(
        samples,
        bins=spectrogram_config.num_mel_bins,
        lo_hz=MEL_LO_HZ,
        overlap=overlap,
        fft_size=FFT_SIZE,
        sample_rate=spectrogram_config.sample_rate
    )
```

**Config:**
- Sample rate: 16000 Hz
- FFT size: 2048
- Hop width: 128 samples
- Mel bins: 512
- Low freq: 20 Hz

**Solution:** Replace with PyTorch implementation (same approach as MR-MT3 adapter):
```python
# Use torchaudio.transforms.MelSpectrogram
import torchaudio.transforms as T

mel_transform = T.MelSpectrogram(
    sample_rate=16000,
    n_fft=2048,
    hop_length=128,
    n_mels=512,
    f_min=20.0
)
```

---

## Comparison with Other Implementations

| Feature | kunato/mt3-pytorch | MR-MT3 | YourMT3 | Magenta MT3 |
|---------|-------------------|---------|----------|-------------|
| **Framework** | PyTorch | PyTorch | PyTorch | JAX/Flax |
| **T5 Source** | HuggingFace (custom) | HuggingFace (custom) | HuggingFace | T5X |
| **Spectrogram** | TensorFlow/DDSP | PyTorch | TensorFlow/DDSP | JAX/DDSP |
| **Weights** | 183 MB (converted) | 176 MB | 5 models (2.6GB) | 874 KB (T5X ckpt) |
| **Architecture** | Official MT3 | MT3 variant | MT3 extended | Official MT3 |
| **Status** | Active (2024) | Archived | Active (2024) | Reference |
| **Inference-only** | ✅ | ✅ | ❌ (includes training) | ❌ (research) |

---

## Adapter Implementation Strategy

### Approach: Extract + Replace Spectrogram

**Steps:**

1. **Vendor T5 Model** (`models/t5.py`)
   - Copy to `mt3_infer/vendor/kunato_mt3/t5.py`
   - Self-contained, ~500 lines
   - Depends only on HuggingFace transformers

2. **Vendor Codec Utilities** (`contrib/`)
   - event_codec.py
   - vocabularies.py (modify to remove `t5.data` import)
   - note_sequences.py
   - run_length_encoding.py
   - **Skip** spectrograms.py (replace with PyTorch)

3. **Create PyTorch Spectrogram** (reuse from MR-MT3)
   ```python
   class MT3SpectrogramProcessor:
       def __init__(self):
           self.mel_transform = T.MelSpectrogram(
               sample_rate=16000,
               n_fft=2048,
               hop_length=128,
               n_mels=512,
               f_min=20.0
           )

       def compute_spectrogram(self, audio):
           spec = self.mel_transform(torch.tensor(audio))
           log_spec = torch.log(spec + 1e-6)
           return log_spec
   ```

4. **Create Adapter** (`mt3_infer/adapters/mt3_pytorch.py`)
   ```python
   class MT3PyTorchAdapter(MT3Base):
       def load_model(self, checkpoint_path, device='auto'):
           # Load config.json
           # Load mt3.pth (or convert from T5X if needed)
           # Initialize T5ForConditionalGeneration

       def preprocess(self, audio, sr):
           # Resample to 16kHz if needed
           # Compute mel spectrogram (PyTorch)
           # Chunk into 256-frame segments

       def forward(self, features):
           # Run T5 generation
           # Uses HuggingFace generate() method

       def decode(self, outputs):
           # Use codec to decode tokens → note events
           # Convert to MIDI via note-seq
   ```

5. **Pull Pretrained Weights**
   ```bash
   cd refs/kunato-mt3-pytorch/
   git lfs pull
   cp pretrained/mt3.pth ../../mt3_infer/checkpoints/
   cp pretrained/config.json ../../mt3_infer/checkpoints/
   ```

---

## Pros and Cons

### ✅ Advantages

1. **Official Architecture:** Uses same T5 architecture as Magenta MT3
2. **PyTorch Native:** No JAX/Flax dependencies at inference time
3. **Lightweight:** 183 MB weights (vs 2.6 GB for YourMT3)
4. **Pre-converted Weights:** Avoid one-time T5X conversion
5. **Proven Code:** Active repository with working inference
6. **Codec-based Decoding:** Uses proper event codec (not heuristics)
7. **Conversion Script:** Can load official Magenta weights if needed

### ⚠️ Challenges

1. **TensorFlow Dependency:** Spectrogram uses DDSP (solvable - replace with PyTorch)
2. **Old Dependencies:** transformers==4.18.0, t5==0.9.3 (need version updates)
3. **Git LFS:** 183 MB download required
4. **Documentation:** Minimal (but code is clean)

---

## Recommendation: IMPLEMENT

**Yes, we should implement this adapter because:**

1. ✅ **Avoids T5X Dependency Hell:** No JAX/Flax/T5X at inference time
2. ✅ **Official MT3 Baseline:** Provides reference implementation for comparison
3. ✅ **Clean Architecture:** Well-structured, inference-focused code
4. ✅ **Solvable Challenges:** TensorFlow dependency can be replaced (same as MR-MT3)
5. ✅ **Completes MT3 Family:**
   - MR-MT3: Multi-instrument variant (PyTorch)
   - YourMT3: Extended multi-task (PyTorch)
   - MT3-PyTorch: **Official baseline** (PyTorch)
6. ✅ **Path to Magenta Weights:** Conversion script available if users want official weights

**Next Steps:**
1. Pull Git LFS weights (`git lfs pull`)
2. Vendor T5 model and codec utilities
3. Implement PyTorch spectrogram (reuse MR-MT3 code)
4. Create `mt3_infer/adapters/mt3_pytorch.py`
5. Test with HappySounds audio
6. Compare all three adapters (MR-MT3, YourMT3, MT3-PyTorch)

---

## Alternative: Defer to v0.2.0

**If we want to go public faster:**
- MR-MT3 and YourMT3 adapters are already working
- Can defer MT3-PyTorch to v0.2.0 alongside Magenta JAX implementation
- Focus on public API and documentation for initial release

**However:** Implementing MT3-PyTorch now gives us 3 working adapters for stronger initial release.

---

## Appendix: Code Quality Assessment

**Strengths:**
- Clean separation of concerns (model, inference, utilities)
- Apache 2.0 license (compatible with our MIT license)
- Follows Magenta MT3 architecture closely
- Well-commented key functions

**Concerns:**
- Minimal documentation (README is 27 lines)
- No tests included
- Training code incomplete ("training not done yet" - README line 15)
- Dependency versions outdated

**Overall:** Good quality for inference-only adaptation. Code is readable and well-structured.

---

## Conclusion

kunato/mt3-pytorch is **the solution we were looking for** to avoid T5X dependency issues while still getting official MT3 architecture. With PyTorch-only spectrogram replacement, we can create a clean, dependency-minimal adapter that complements our existing MR-MT3 and YourMT3 implementations.

**Status:** ✅ Ready for implementation
