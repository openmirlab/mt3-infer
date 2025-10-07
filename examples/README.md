# MT3-Infer Examples

Example scripts demonstrating MT3-Infer usage.

---

## Available Examples

### Model Comparison (`compare_models.py`)
Compare MR-MT3 and YourMT3 performance on real audio.

**Usage:**
```bash
uv run python examples/compare_models.py
```

**Output:**
- `comparison_mr_mt3.mid` - MR-MT3 transcription
- `comparison_yourmt3.mid` - YourMT3 transcription
- Performance comparison table

---

### GPU Testing (`test_gpu.py`)
Test both adapters on GPU with automatic device detection.

**Usage:**
```bash
uv run python examples/test_gpu.py
```

**Requirements:**
- CUDA-capable GPU
- CUDA toolkit installed

**Output:**
- GPU performance metrics
- `gpu_test_mr_mt3.mid` - MR-MT3 GPU output
- `gpu_test_yourmt3.mid` - YourMT3 GPU output

---

### YourMT3 Quick Test (`test_yourmt3_quick.py`)
Quick functional test for YourMT3 adapter.

**Usage:**
```bash
uv run python examples/test_yourmt3_quick.py
```

**Output:**
- MIDI transcription
- Note count and basic validation

---

### YourMT3 Verification (`verify_yourmt3.py`)
Systematic verification of YourMT3 adapter implementation.

**Usage:**
```bash
uv run python examples/verify_yourmt3.py
```

**Checks:**
- Import validation
- MT3Base interface compliance
- Model registry
- Checkpoint verification
- Dependency checks
- Functional test
- MIDI validation

---

## Running Examples

All examples use `uv run` to ensure correct environment:

```bash
# Basic syntax
uv run python examples/<script_name>.py

# With test audio (adjust path as needed)
uv run python examples/compare_models.py
```

---

## Test Outputs

Example scripts generate MIDI files in `test_outputs/` directory:
- `comparison_*.mid` - Model comparison outputs
- `gpu_test_*.mid` - GPU test outputs
- `yourmt3_*.mid` - YourMT3 test outputs

**Note:** Test outputs are gitignored and not included in package distribution.

---

## Creating New Examples

When creating new example scripts:

1. Add to this directory: `examples/`
2. Use `uv run python` for execution
3. Document in this README
4. Use clear, descriptive names
5. Include docstrings and comments
6. Output to `test_outputs/` directory

**Template:**
```python
#!/usr/bin/env python3
"""
Brief description of what this example demonstrates.
"""

from mt3_infer.adapters.mr_mt3 import MRMT3Adapter
from mt3_infer.utils.audio import load_audio

# Your example code here
adapter = MRMT3Adapter()
adapter.load_model('checkpoint.pth')
# ...
```
