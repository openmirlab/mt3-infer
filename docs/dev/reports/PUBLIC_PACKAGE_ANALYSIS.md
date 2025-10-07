# Public Package Distribution Analysis - YourMT3 Adapter

**Date:** 2025-10-06
**Issue:** Current `sys.path` injection approach is not suitable for public PyPI distribution

---

## 🚨 Problems with Current Approach

### 1. **Missing refs/ Directory**
```bash
# When user installs from PyPI:
uv add mt3-infer
# OR: uv pip install mt3-infer

# What they get:
mt3_infer/
├── adapters/
│   ├── yourmt3.py  ✓ (included)
│   └── mr_mt3.py   ✓ (included)
└── ...

# What they DON'T get:
refs/yourmt3/  ✗ (NOT included in package)
```

**Result:** `YourMT3Adapter` will fail on import because `refs/yourmt3/` doesn't exist.

### 2. **User Has to Manually Clone**
```bash
# User would need to:
uv add mt3-infer
cd /some/path
git clone https://huggingface.co/spaces/mimbres/YourMT3 refs/yourmt3
git lfs pull  # Download 2.6GB of checkpoints

# Then somehow tell mt3-infer where refs/ is located
```

**Result:** Bad user experience, error-prone, non-standard.

### 3. **Version Control Issues**
- No way to specify which version of YourMT3 to use
- User might clone a different commit than tested
- Reproducibility problems

### 4. **Checkpoint Distribution**
- 2.6GB of model weights not suitable for PyPI
- Git LFS required (adds complexity)
- Bandwidth costs for distribution

---

## ✅ Recommended Solutions

### **Option 1: Extract Inference Code (Best for Public Package)**

**Approach:** Do what we did for MR-MT3, but for YourMT3.

**Implementation:**
```python
# mt3_infer/adapters/yourmt3_extracted.py (~1500-2000 lines)

# Extract only inference code:
# - model/ymt3.py (inference methods only)
# - utils/task_manager.py (detokenization)
# - utils/event2note.py (note conversion)
# - utils/audio.py (segmentation)
# - Model architecture definitions

class YourMT3Adapter(MT3Base):
    def load_model(self, checkpoint_path, device):
        # Use extracted YourMT3 model class
        self.model = ExtractedYourMT3Model(...)
        # No sys.path needed!
```

**Pros:**
- ✅ Self-contained, no refs/ dependency
- ✅ Works with `uv add mt3-infer`
- ✅ Version controlled in your package
- ✅ Clean user experience

**Cons:**
- ⚠️ More code to maintain (~1500-2000 lines)
- ⚠️ Need to sync with upstream manually
- ⚠️ Initial extraction work (2-4 hours)

**Checkpoints:**
Users download separately (standard practice):
```python
# User downloads checkpoint once
from mt3_infer.utils import download_checkpoint
download_checkpoint("ymt3plus", cache_dir="~/.cache/mt3_infer")

# Then use it
adapter = YourMT3Adapter()
adapter.load_model("ymt3plus")  # Auto-finds in cache
```

---

### **Option 2: Optional Dependency with Git Source**

**Approach:** Make YourMT3 an optional git dependency.

**Implementation:**
```toml
# pyproject.toml
[project.optional-dependencies]
yourmt3 = [
    "mt3-yourmt3 @ git+https://huggingface.co/spaces/mimbres/YourMT3@main#subdirectory=amt/src"
]
```

```bash
# User installs
uv add "mt3-infer[yourmt3]"
# OR: uv pip install "mt3-infer[yourmt3]"
```

**Pros:**
- ✅ Automatic dependency management
- ✅ Version pinned to git commit
- ✅ Minimal code in your package

**Cons:**
- ✗ YourMT3 is NOT a proper Python package (it's a Gradio Space)
- ✗ No setup.py or pyproject.toml in upstream
- ✗ Won't work without restructuring upstream
- ✗ Still need refs/ directory structure

**Verdict:** ❌ Not feasible (upstream not packaged)

---

### **Option 3: Separate Package (mt3-infer-yourmt3)**

**Approach:** Create a separate package for YourMT3 support.

**Structure:**
```
mt3-infer/           (core package, ~200KB)
  ├── base.py
  ├── adapters/
  │   └── mr_mt3.py  ✓ Included

mt3-infer-yourmt3/   (separate package, ~500KB)
  ├── extracted YourMT3 code
  └── adapters/
      └── yourmt3.py

# User installs
uv add mt3-infer              # Core only
uv add mt3-infer-yourmt3      # Optional YourMT3 support
```

**Pros:**
- ✅ Clean separation of concerns
- ✅ Users only install what they need
- ✅ Each package can version independently

**Cons:**
- ⚠️ More complex project structure
- ⚠️ Still need to extract YourMT3 code
- ⚠️ Two packages to maintain

---

### **Option 4: Keep Current Approach (Not Recommended for Public)**

**Approach:** Document the manual setup process.

**README.md:**
```markdown
## YourMT3 Adapter (Advanced Setup Required)

The YourMT3 adapter requires manual setup:

1. Clone the reference repository:
   ```bash
   git clone https://huggingface.co/spaces/mimbres/YourMT3 refs/yourmt3
   cd refs/yourmt3
   git lfs pull
   ```

2. Ensure refs/ is in your project root:
   ```
   your-project/
   ├── refs/
   │   └── yourmt3/  ← Must be here
   └── venv/
   ```

3. Import and use:
   ```python
   from mt3_infer.adapters.yourmt3 import YourMT3Adapter
   ```
```

**Pros:**
- ✅ No code changes needed
- ✅ Works for advanced users

**Cons:**
- ✗ Terrible user experience
- ✗ Error-prone manual setup
- ✗ Not suitable for PyPI/uv package
- ✗ Violates Python packaging standards

---

## 📊 Comparison Matrix

| Approach | UX | Maintenance | Distribution | Recommended |
|----------|-----|-------------|--------------|-------------|
| **Extract Code** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ✅ **Best** |
| Git Dependency | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ❌ Not feasible | ❌ |
| Separate Package | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⚠️ OK |
| Current (sys.path) | ⭐ | ⭐⭐⭐⭐⭐ | ❌ Not suitable | ❌ |

---

## 🎯 Recommended Action Plan

### For Public PyPI Package: **Extract YourMT3 Code**

**Phase 1: Extraction (2-4 hours)**
1. Extract inference-only code from `refs/yourmt3/`:
   - `amt/src/model/ymt3.py` → `mt3_infer/adapters/_yourmt3_model.py`
   - `amt/src/utils/task_manager.py` → `mt3_infer/adapters/_yourmt3_tokenizer.py`
   - `amt/src/utils/event2note.py` → `mt3_infer/adapters/_yourmt3_decoder.py`
   - Remove all training code, keep only inference methods

2. Refactor extracted code:
   - Remove PyTorch Lightning dependencies
   - Convert to plain `nn.Module`
   - Strip out wandb, training callbacks, etc.

**Phase 2: Checkpoint Management**
```python
# mt3_infer/utils/checkpoints.py
from huggingface_hub import hf_hub_download

def download_yourmt3_checkpoint(model_key, cache_dir=None):
    """Download checkpoint from Hugging Face."""
    return hf_hub_download(
        repo_id="mimbres/YourMT3",
        filename=f"checkpoints/{model_key}.ckpt",
        cache_dir=cache_dir
    )
```

**Phase 3: Clean API**
```bash
# Installation
uv add mt3-infer
```

```python
# User experience
from mt3_infer.adapters.yourmt3 import YourMT3Adapter

adapter = YourMT3Adapter(model_key="ymt3plus")
# First time: auto-downloads checkpoint (518MB)
adapter.load_model()
midi = adapter.transcribe(audio, sr)
```

---

## 💡 Quick Decision Guide

**Ask yourself:**

1. **Will this be on PyPI?**
   - Yes → Must extract code ✅
   - No (local only) → Current approach OK ⚠️

2. **Do users need easy installation?**
   - Yes → Must extract code ✅
   - No (advanced users only) → Current approach OK ⚠️

3. **How much maintenance can you handle?**
   - Low → Keep current approach ⚠️ (but not suitable for public)
   - Medium/High → Extract code ✅

---

## 🏁 Final Recommendation

**For a public package on PyPI:**
→ **Extract YourMT3 inference code** (Option 1)

**For internal/local use only:**
→ **Keep current sys.path approach** (Option 4)

**If you want modular architecture:**
→ **Separate package** (Option 3)

---

## Next Steps if Extracting Code

1. Create extraction plan (identify files needed)
2. Extract model architecture (inference only)
3. Extract tokenization utilities
4. Extract decoder utilities
5. Test extracted code matches upstream output
6. Add checkpoint download utilities
7. Update documentation
8. Package for PyPI

Estimated effort: **2-4 hours** (similar to what we did for MR-MT3, but larger)

---

**Current Status:** YourMT3 adapter works perfectly for local development.
**For public package:** Code extraction recommended.
