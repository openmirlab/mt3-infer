# YourMT3 Integration Options - Feature Comparison

**Date:** 2025-10-07

## Overview

YourMT3 has **94 Python files** across multiple directories. Here's what would be available with each integration approach.

---

## Option 1: Full Vendor Integration

### ✅ Features Kept (Everything)

**All 5 Model Variants:**
1. ✅ YPTF.MoE+Multi (noPS) - 536MB - Default multi-task model
2. ✅ YPTF.MoE+Multi (PS) - 724MB - With pitch shift augmentation
3. ✅ YPTF+Multi (PS) - 517MB - Multi-task with pitch shift
4. ✅ YPTF+Single (noPS) - 345MB - Single-task model
5. ✅ YMT3+ - 518MB - No task conditioning

**Advanced Model Architectures:**
- ✅ Perceiver-TF encoder (cross-attention based)
- ✅ Multi-T5 decoder (26 layers)
- ✅ Mixture of Experts (MoE) feed-forward layers
- ✅ Standard T5 encoder/decoder
- ✅ Conformer encoder support

**Multi-Task Capabilities:**
- ✅ 8-stem separation (drums, bass, guitar, piano, strings, winds, vocals, other)
- ✅ Multi-instrument transcription
- ✅ Simultaneous transcription of multiple instruments
- ✅ Per-channel note extraction

**Advanced Audio Features:**
- ✅ Multiple audio codecs (spectrogram, mel-spectrogram)
- ✅ Configurable hop lengths (128, 300 frames)
- ✅ Variable input frame sizes
- ✅ Pitch shift augmentation (during training/testing)

**Positional Encoding Options:**
- ✅ RoPE (Rotary Position Embedding)
- ✅ ALiBi (Attention with Linear Biases)
- ✅ Trainable positional encoding
- ✅ Sinusoidal encoding
- ✅ Task-dependent positional encoding

**Configuration Flexibility:**
- ✅ Task-conditional encoder/decoder
- ✅ Multiple vocabulary sizes
- ✅ Configurable model dimensions
- ✅ Variable number of layers
- ✅ Different activation functions (GELU, SiLU, ReLU)

**Training Features (if needed later):**
- ✅ PyTorch Lightning training loop
- ✅ Data augmentation pipeline
- ✅ Multiple dataset support
- ✅ Distributed training support
- ✅ Wandb logging integration

### ❌ Drawbacks

**Package Size:**
- ❌ +94 Python files (~50KB total source code)
- ❌ +2.6GB checkpoints (all 5 models)
- ❌ Complex dependency tree

**Complexity:**
- ❌ PyTorch Lightning dependency (large framework)
- ❌ Many configuration files to maintain
- ❌ Complex initialization logic
- ❌ Harder to debug issues
- ❌ Potential dependency conflicts

**Maintenance:**
- ❌ Need to keep vendor/ directory in sync with upstream
- ❌ More surface area for bugs
- ❌ Harder to understand codebase for contributors

---

## Option 2: Simplified Adapter Integration

### ✅ Features Kept (Essential Inference Only)

**Single Default Model:**
- ✅ YPTF.MoE+Multi (noPS) - 536MB
- ✅ Perceiver-TF encoder + Multi-T5 decoder
- ✅ MoE feed-forward layers
- ✅ RoPE positional encoding

**Core Transcription:**
- ✅ Multi-instrument transcription
- ✅ 8-stem separation capability
- ✅ Audio → MIDI conversion
- ✅ Note onset/offset detection
- ✅ Velocity estimation
- ✅ Program (instrument) detection

**Audio Processing:**
- ✅ Spectrogram feature extraction
- ✅ 300-frame hop length
- ✅ Audio segmentation
- ✅ Automatic resampling to 16kHz

**Post-Processing:**
- ✅ Multi-channel detokenization
- ✅ Note event merging
- ✅ Tie note handling
- ✅ MIDI file generation

**Integration:**
- ✅ Fits MT3Base interface
- ✅ Auto-download checkpoint
- ✅ Device auto-detection (CPU/GPU)
- ✅ Simple transcribe() API

### ⚠️ Features Lost (Advanced/Training Features)

**Model Variants:**
- ❌ Can't switch to other 4 checkpoint variants
- ❌ No pitch shift model option
- ❌ No single-task model option
- ❌ Can't use YMT3+ variant

**Architecture Flexibility:**
- ❌ Can't switch encoder types (T5, Conformer)
- ❌ Can't switch decoder types
- ❌ No runtime architecture configuration
- ❌ Fixed positional encoding type

**Advanced Features:**
- ❌ No pitch shift augmentation
- ❌ Can't change audio codec at runtime
- ❌ Can't adjust hop length dynamically
- ❌ No custom vocabulary support

**Training/Fine-tuning:**
- ❌ No training capability
- ❌ No PyTorch Lightning integration
- ❌ No data augmentation
- ❌ Can't fine-tune on custom data

**Configuration:**
- ❌ Limited runtime configuration
- ❌ Can't change task conditioning
- ❌ Fixed model dimensions
- ❌ No experiment management

### 📦 What Gets Vendored (Minimal Set)

**Required Files (~20-30 files):**
```
mt3_infer/vendor/yourmt3/
├── model/
│   ├── ymt3.py              # Main model class
│   ├── perceiver_tf.py      # Perceiver encoder
│   ├── multi_t5.py          # Multi-T5 decoder
│   └── moe.py               # Mixture of Experts
├── utils/
│   ├── task_manager.py      # Tokenizer/detokenizer
│   ├── audio.py             # Audio preprocessing
│   ├── midi.py              # MIDI utilities
│   ├── event_codec.py       # Event encoding/decoding
│   ├── event2note.py        # Event → Note conversion
│   ├── note2event.py        # Note → Event conversion
│   └── tokenizer.py         # Token management
└── config/
    ├── config.py            # Model configuration
    ├── task.py              # Task definitions
    └── vocabulary.py        # Vocabulary presets
```

**Simplified Adapter:**
```python
# mt3_infer/adapters/yourmt3.py (~200-300 lines)
from mt3_infer.base import MT3Base
from mt3_infer.vendor.yourmt3 import YourMT3, TaskManager

class YourMT3Adapter(MT3Base):
    def load_model(self, checkpoint_path, device="auto"):
        # Load default YPTF.MoE+Multi (noPS) model
        # Fixed configuration, no runtime changes

    def preprocess(self, audio, sr):
        # Spectrogram extraction, 300-frame hop

    def forward(self, features):
        # Run inference with bsz=8

    def decode(self, outputs):
        # Multi-channel detokenization → MIDI
```

**Dependencies:**
- ✅ pytorch-lightning (only for model class, not trainer)
- ✅ No extra dependencies beyond existing mt3-infer

---

## Feature Comparison Table

| Feature | Option 1 (Full) | Option 2 (Simplified) | Lost in Option 2 |
|---------|----------------|----------------------|------------------|
| **Models** | 5 variants | 1 default | 4 other variants |
| **Encoder types** | 3 (T5, Perceiver-TF, Conformer) | 1 (Perceiver-TF) | T5, Conformer |
| **Decoder types** | 2 (T5, Multi-T5) | 1 (Multi-T5) | T5 |
| **Positional encoding** | 8+ types | 1 (RoPE) | 7 other types |
| **Audio codecs** | 2 (spec, melspec) | 1 (spec) | melspec |
| **Hop lengths** | Configurable | Fixed (300) | Runtime config |
| **Multi-task** | ✅ Full | ✅ Full | None |
| **8-stem separation** | ✅ | ✅ | None |
| **Pitch shift aug** | ✅ | ❌ | Training feature |
| **Training support** | ✅ | ❌ | Training entirely |
| **Runtime config** | ✅ Full | ⚠️ Limited | Most config |
| **Package size** | 94 files, 2.6GB | ~25 files, 536MB | 69 files, 2.1GB |
| **Complexity** | Very high | Medium | N/A |
| **Maintenance** | High effort | Medium effort | N/A |

---

## Practical Impact for Users

### What Users CAN Do with Option 2:

✅ **Transcribe any music audio** with YourMT3's best model
✅ **Separate 8 instrument stems** and get per-stem MIDI
✅ **Get high-quality transcriptions** with MoE model
✅ **Use same API** as MR-MT3 and MT3-PyTorch:
```python
from mt3_infer import load_model

model = load_model('yourmt3')  # Auto-downloads 536MB
midi = model.transcribe(audio, sr=16000)
```

### What Users CANNOT Do with Option 2:

❌ Switch to a different YourMT3 checkpoint variant
❌ Use pitch shift augmentation
❌ Change encoder/decoder architecture at runtime
❌ Fine-tune the model on their own data
❌ Use T5-only encoder instead of Perceiver-TF
❌ Change positional encoding type
❌ Use mel-spectrogram instead of spectrogram

### Are These Limitations Critical?

**For 95% of users:** ❌ **No, not critical**
- The default YPTF.MoE+Multi (noPS) is the **best model** in the collection
- Most users just want: audio in → MIDI out
- Advanced features are mainly for researchers/developers

**For power users:** ⚠️ **Maybe, but workarounds exist**
- Can use full YourMT3 repo separately if needed
- Can request Option 1 (full vendor) in v0.2.0
- Can manually load checkpoints via refs/yourmt3/

---

## Recommendation: Option 2 (Simplified)

### Why Option 2 is Best for v0.1.0:

1. **✅ Keeps core features:** Multi-task, 8-stem separation, high quality
2. **✅ Reasonable size:** 536MB vs 2.6GB (5x smaller)
3. **✅ Manageable complexity:** ~25 files vs 94 files
4. **✅ Faster to implement:** Can be done in v0.1.0
5. **✅ Easy to upgrade:** Can add Option 1 features in v0.2.0

### What Gets Lost is Acceptable:

- **Other model variants:** Default is the best one anyway
- **Architecture flexibility:** Users want results, not config
- **Training features:** mt3-infer is inference-only by design
- **Advanced config:** Simplicity > flexibility for most users

### Path Forward:

**v0.1.0 (Now):**
- ✅ MR-MT3: Speed champion (22.7x RT)
- ✅ MT3-PyTorch: Accuracy champion (767 notes)
- ✅ YourMT3 (simplified): Multi-task champion (8-stem separation)

**v0.2.0 (Future, if needed):**
- ⏳ YourMT3 full vendor (all 5 models, full features)
- ⏳ Magenta MT3 (JAX/Flax, original implementation)
- ⏳ Training/fine-tuning support

---

## Implementation Estimate

### Option 1 (Full Vendor):
- **Time:** 2-3 days
- **Risk:** High (dependency conflicts, complex debugging)
- **Code:** ~94 files vendored + adapter
- **Testing:** Extensive (multiple models, configurations)

### Option 2 (Simplified):
- **Time:** 4-6 hours
- **Risk:** Medium (need to extract essential code correctly)
- **Code:** ~25 files vendored + adapter
- **Testing:** Moderate (single model, fixed config)

---

**Recommendation: Go with Option 2 for v0.1.0** ✅

Users get 95% of the value with 25% of the complexity.
