# MT3-Infer

**Production-ready, unified inference toolkit for the MT3 music transcription model family**

MT3-Infer provides a clean, framework-neutral API for running music transcription inference across multiple MT3 implementations with a single consistent interface.

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7.1-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Features

- ✅ **Unified API**: One interface for all MT3 variants
- ✅ **Production Ready**: Clean, tested, ~8MB package size
- ✅ **Auto-Download**: Automatic checkpoint downloads on first use
- ✅ **4 Download Methods**: Auto, Python API, CLI, standalone script
- ✅ **3 Models**: MR-MT3 (fast), MT3-PyTorch (accurate), YourMT3 (multitask)
- ✅ **Framework Isolated**: Clean PyTorch/TensorFlow/JAX separation
- ✅ **CLI Tool**: `mt3-infer` command-line interface
- ✅ **Reproducible**: Pinned dependencies, verified checkpoints

---

## Quick Start

### Installation

```bash
# Using pip
pip install mt3-infer

# Using UV (recommended for development)
uv pip install mt3-infer
```

### Simple Transcription (One Line!)

```python
from mt3_infer import transcribe

# Transcribe audio to MIDI (auto-downloads checkpoint on first use)
midi = transcribe(audio, sr=16000)
midi.save("output.mid")
```

### Model Selection

```python
# Use fastest model (57x real-time)
midi = transcribe(audio, model="fast")

# Use most accurate model (147 notes detected)
midi = transcribe(audio, model="accurate")

# Use multitask model
midi = transcribe(audio, model="multitask")
```

### Download Checkpoints

```bash
# Download all models at once (874MB total)
mt3-infer download --all

# Download specific models
mt3-infer download mr_mt3 mt3_pytorch

# List available models
mt3-infer list

# Transcribe audio via CLI
mt3-infer transcribe input.wav -o output.mid -m fast
```

> **Heads up:** The downloader now pulls MR-MT3 weights directly from
> [`gudgud1014/MR-MT3`](https://huggingface.co/gudgud1014/MR-MT3), so you no
> longer need Git LFS for that model. Checkpoints are stored under
> `.mt3_checkpoints/<model>` and will be re-created automatically if you delete
> the directory.

---

## Supported Models

| Model | Alias | Framework | Speed | Accuracy | Size | Best For |
|-------|-------|-----------|-------|----------|------|----------|
| **MR-MT3** | `fast` | PyTorch | **57x real-time** | 116 notes | 176 MB | Speed-critical apps |
| **MT3-PyTorch** | `accurate`, `default` | PyTorch | 12x real-time | **147 notes** | 176 MB | General use, accuracy |
| **YourMT3** | `multitask` | PyTorch + Lightning | ~15x real-time | 118 notes | 536 MB | Multi-stem separation |

*Tested on NVIDIA RTX 4090 with PyTorch 2.7.1 + CUDA 12.6*

> Default `yourmt3` downloads the `YPTF.MoE+Multi (noPS)` checkpoint, matching the original YourMT3 Space output.

---

## Advanced Usage

### Explicit Model Loading

```python
from mt3_infer import load_model

# Load model explicitly (cached for reuse)
model = load_model("mt3_pytorch", device="cuda")
midi = model.transcribe(audio, sr=16000)
```

### Explore Available Models

```python
from mt3_infer import list_models, get_model_info

# List all models
models = list_models()
for name, info in models.items():
    print(f"{name}: {info['description']}")

# Get model details
info = get_model_info("fast")
print(f"Speed: {info['metadata']['performance']['speed_x_realtime']}x real-time")
```

### Disable Auto-Download

```python
from mt3_infer import load_model

# Raise error if checkpoint not found (don't auto-download)
model = load_model("mr_mt3", auto_download=False)
```

### Download Programmatically

```python
from mt3_infer import download_model

# Pre-download checkpoints before inference
download_model("mr_mt3")
download_model("mt3_pytorch")
download_model("yourmt3")
```

---

## Installation Options

### Basic Installation

```bash
pip install mt3-infer
```

### Development Installation

```bash
# Clone repository
git clone https://github.com/worzpro/mt3-infer.git
cd mt3-infer

# Install with UV (recommended)
uv sync --extra torch --extra dev

# Or with pip
pip install -e ".[torch,dev]"
```

### Optional Dependencies

```bash
# PyTorch backend (default)
pip install mt3-infer[torch]

# TensorFlow backend
pip install mt3-infer[tensorflow]

# All backends
pip install mt3-infer[all]

# Development tools
pip install mt3-infer[dev]

# MIDI synthesis (optional)
pip install mt3-infer[synthesis]
```

---

## CLI Tool

The `mt3-infer` CLI provides convenient access to all functionality:

```bash
# Download checkpoints
mt3-infer download --all                    # Download all models
mt3-infer download mr_mt3 mt3_pytorch       # Download specific models

# List available models
mt3-infer list

# Transcribe audio
mt3-infer transcribe input.wav -o output.mid
mt3-infer transcribe input.wav -m fast      # Use fast model
mt3-infer transcribe input.wav --device cuda # Use GPU

# Show help
mt3-infer --help
mt3-infer download --help
```

---

## Download Methods

MT3-Infer supports **4 flexible download methods**:

### 1. **Automatic Download** (Default)
Checkpoints download automatically on first use:
```python
midi = transcribe(audio)  # Auto-downloads if needed
```

### 2. **Python API**
Pre-download programmatically:
```python
from mt3_infer import download_model
download_model("mr_mt3")
```

### 3. **CLI**
Download via command line:
```bash
mt3-infer download --all
```

### 4. **Standalone Script**
Batch download without installing package:
```bash
python tools/download_all_checkpoints.py
```

See [docs/DOWNLOAD.md](docs/DOWNLOAD.md) for detailed download guide.

---

## Project Status

**Current Version:** 0.1.0 (Production Ready!)

### ✅ Completed Features
- ✅ Core infrastructure (MT3Base interface, utilities)
- ✅ 3 production adapters (MR-MT3, MT3-PyTorch, YourMT3)
- ✅ Public API (`transcribe()`, `load_model()`)
- ✅ Model registry with aliases
- ✅ Checkpoint download system (4 methods)
- ✅ CLI tool (`mt3-infer`)
- ✅ Production cleanup (~8MB package)
- ✅ Comprehensive documentation

### 📦 Package Statistics
- **Source code:** ~5 MB
- **Vendor dependencies:** ~3 MB
- **Documentation:** 284 KB
- **Total (source only):** ~8 MB
- **With downloaded models:** ~882 MB

### 🚧 Roadmap
- **v0.2.0** (Planned): Batch processing, additional optimizations
- **v0.3.0** (Planned): ONNX export, streaming inference
- **v1.0.0** (Planned): Full test coverage, PyPI release

**Note:** Magenta MT3 (JAX/Flax) has been excluded due to dependency conflicts with the PyTorch ecosystem. The current 3 models (MR-MT3, MT3-PyTorch, YourMT3) provide comprehensive coverage for speed, accuracy, and multi-stem use cases.

---

## Architecture

```
mt3_infer/
├── __init__.py          # Public API
├── api.py               # High-level functions (transcribe, load_model)
├── base.py              # MT3Base abstract interface
├── cli.py               # CLI tool
├── exceptions.py        # Custom exceptions
├── adapters/            # Model-specific implementations
│   ├── mr_mt3.py        # MR-MT3 (PyTorch)
│   ├── mt3_pytorch.py   # MT3-PyTorch (Magenta weights)
│   ├── yourmt3.py       # YourMT3 (PyTorch + Lightning)
│   └── vocab_utils.py   # Shared MIDI decoding
├── config/
│   └── checkpoints.yaml # Model registry & download config
├── utils/
│   ├── audio.py         # Audio preprocessing
│   ├── midi.py          # MIDI postprocessing
│   ├── download.py      # Checkpoint download system
│   └── framework.py     # Version checks
├── vendor/              # Vendored dependencies (self-contained)
│   ├── kunato_mt3/      # MT3-PyTorch model code
│   └── yourmt3/         # YourMT3 model code
└── tests/               # Test suite
```

---

## Documentation

### For Users
- **[Main README](README.md)** - This file
- **[Download Guide](docs/DOWNLOAD.md)** - Checkpoint download methods
- **[Examples](examples/)** - Usage examples

### For Developers
- **[Documentation Index](docs/README.md)** - Complete docs navigation
- **[Claude Code Guide](CLAUDE.md)** - Development with Claude Code
- **[API Specification](docs/dev/SPEC.md)** - Formal API spec
- **[Development Plan](docs/dev/PLAN.md)** - Implementation roadmap
- **[Design Principles](docs/dev/PRINCIPLES.md)** - Development guidelines

### Technical Reports
- **[Model Comparison](docs/dev/reports/MODEL_COMPARISON.md)** - Performance comparison
- **[GPU Performance](docs/dev/reports/GPU_PERFORMANCE.md)** - GPU benchmarks
- **[CPU Analysis](docs/dev/reports/CPU_SPEED_ANALYSIS.md)** - CPU performance

---

## Development

### Setup

```bash
# Install dependencies
uv sync --extra torch --extra dev

# Run tests
uv run pytest

# Run with coverage
uv run pytest --cov=mt3_infer --cov-report=html

# Linting
uv run ruff check .
uv run ruff check --fix .

# Type checking
uv run mypy mt3_infer/
```

### Using UV

This project uses [UV](https://github.com/astral-sh/uv) for dependency management. Always use `uv run`:

```bash
# ✅ Correct
uv run python script.py
uv run pytest

# ❌ Incorrect
python script.py
pytest
```

See [docs/dev/PRINCIPLES.md](docs/dev/PRINCIPLES.md) for details.

---

## Integration with worzpro-demo

To use mt3-infer in the worzpro-demo project:

```toml
# In worzpro-demo/pyproject.toml
[tool.uv.sources]
mt3-infer = { path = "../patched_modules/mt3-infer", extras = ["torch"] }
```

Then in Python:
```python
from mt3_infer import transcribe
midi = transcribe(audio, sr=16000)
```

---

## Examples

See the [examples/](examples/) directory for complete examples:

- **[public_api_demo.py](examples/public_api_demo.py)** - Main usage example
- **[synthesize_all_models.py](examples/synthesize_all_models.py)** - Compare all models
- **[demo_midi_synthesis.py](examples/demo_midi_synthesis.py)** - MIDI synthesis demo
- **[test_download.py](examples/test_download.py)** - Download validation
- **[compare_models.py](examples/compare_models.py)** - Model comparison

---

## License

MIT License - see [LICENSE](LICENSE) for details.

This project includes code adapted from:
- **Magenta MT3** (Apache-2.0) - Google Research
- **MR-MT3** (MIT) - Hao Hao Tan et al.
- **MT3-PyTorch** - Kunato's PyTorch port
- **YourMT3** (Apache-2.0) - Minz Won et al.

See [mt3_infer/config/checkpoints.yaml](mt3_infer/config/checkpoints.yaml) for full provenance.

---

## Contributing

We welcome contributions! Please:

1. Read [docs/dev/SPEC.md](docs/dev/SPEC.md) for API specifications
2. Follow [docs/dev/PRINCIPLES.md](docs/dev/PRINCIPLES.md) for development guidelines
3. Submit PRs with tests and documentation

---

## Citation

If you use MT3-Infer in your research, please cite the original MT3 papers:

```bibtex
@inproceedings{hawthorne2022mt3,
  title={Multi-Task Multitrack Music Transcription},
  author={Hawthorne, Curtis and others},
  booktitle={ISMIR},
  year={2022}
}
```

---

## Support

For issues and questions:
- **GitHub Issues**: [github.com/worzpro/mt3-infer/issues](https://github.com/worzpro/mt3-infer/issues)
- **Documentation**: [docs/](docs/)
- **Examples**: [examples/](examples/)

---

**Built for the worzpro-demo ecosystem** | **Powered by PyTorch**
