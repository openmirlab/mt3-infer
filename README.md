# MT3-Infer

**Production-ready, unified inference toolkit for the MT3 music transcription model family**

MT3-Infer provides a clean, framework-neutral API for running music transcription inference across multiple MT3 implementations with a single consistent interface.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/mt3-infer)](https://pypi.org/project/mt3-infer/)

---

## Why this exists

[MT3](https://github.com/magenta/mt3) ("Multi-Task Multitrack Music
Transcription") is Google Magenta's JAX/Flax model for transcribing
polyphonic, multi-instrument audio into MIDI. The reference implementation
is research code: pinned to an old JAX/Flax/t5x/seqio stack that's
increasingly painful to install alongside anything PyTorch-based, and it
ships as a training-and-eval codebase rather than a drop-in inference
library.

In the years since, the community produced several independent PyTorch
ports and variants — each with its own environment, its own quirks, and no
shared interface. **MT3-Infer** wraps three of them (MR-MT3, MT3-PyTorch,
YourMT3) behind one `MT3Base` interface and one public API
(`transcribe()`, `load_model()`), with automatic checkpoint management, so
"run MT3-family transcription" doesn't require reproducing three separate
research environments. The original JAX/Flax Magenta implementation itself
is **not** vendored here (see [Scope](#scope) below) — this repo only
reprovides the PyTorch-based descendants.

---

## Acknowledgments

MT3-Infer wraps and vendors code from the following projects. None of this
would exist without their original authors' work:

- **[Magenta MT3](https://github.com/magenta/mt3)** (Apache-2.0) — Google
  Research / Magenta team. The original model and task this whole family
  descends from (see [Citation](#citation)). Not currently vendored in this
  repo — see [Scope](#scope).
- **[MR-MT3](https://github.com/gudgud96/MR-MT3)** (MIT) — Hao Hao Tan, Kin
  Wai Cheuk, Taemin Cho, Wei-Hsiang Liao, Yuki Mitsufuji. Vendored in
  `mt3_infer/models/mr_mt3/`; pretrained weights re-hosted at
  [huggingface.co/gudgud1014/MR-MT3](https://huggingface.co/gudgud1014/MR-MT3).
- **[MT3-PyTorch](https://github.com/kunato/mt3-pytorch)** — kunato. Vendored
  unmodified in `mt3_infer/models/mt3_pytorch/`; no license is declared
  upstream, so the vendored tree is kept byte-for-byte as extracted (see
  [LICENSE](LICENSE) and `CLAUDE.md` for the full honesty note).
- **[YourMT3](https://huggingface.co/spaces/mimbres/YourMT3)** (Apache-2.0) —
  Sungkyun Chang, Emmanouil Benetos, Holger Kirchhoff, Simon Dixon (see the
  [YourMT3+ paper](https://arxiv.org/abs/2407.04822), MLSP 2024). Vendored
  in `mt3_infer/models/yourmt3/`.

Full commit-level provenance for each vendored tree is in
[`mt3_infer/config/external_integrations.yaml`](mt3_infer/config/external_integrations.yaml).

---

## Citation

If you use MT3-Infer in your research, please cite the original MT3 paper:

```bibtex
@inproceedings{gardner2022mt3,
  title     = {{MT3}: Multi-Task Multitrack Music Transcription},
  author    = {Gardner, Josh and Simon, Ian and Manilow, Ethan and Hawthorne, Curtis and Engel, Jesse},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2022},
  url       = {https://openreview.net/forum?id=iMSjopcOn0p}
}
```

(Verified against the paper's [arXiv listing](https://arxiv.org/abs/2111.03017)
and [DBLP](https://dblp.org/rec/conf/iclr/GardnerSMHE22.html) — the paper
was published at ICLR 2022, not ISMIR, and Josh Gardner is the first
author, not Curtis Hawthorne.)

If you use a specific backend, consider also citing its own paper — e.g.
YourMT3's [YourMT3+ paper](https://arxiv.org/abs/2407.04822) (MLSP 2024).

---

## Features

- **Unified API**: One interface for all MT3 variants
- **Production Ready**: Clean, tested, ~8MB package size
- **Auto-Download**: Automatic checkpoint downloads on first use
- **4 Download Methods**: Auto, Python API, CLI, standalone script
- **3 Models**: MR-MT3, MT3-PyTorch, YourMT3
- **Framework Isolated**: Clean PyTorch/TensorFlow/JAX separation
- **CLI Tool**: `mt3-infer` command-line interface
- **Reproducible**: Pinned dependencies, verified checkpoints

**Current status:** all three PyTorch-based backends (MR-MT3, MT3-PyTorch,
YourMT3) are implemented, tested, and installable from PyPI today. What's
not yet supported: the original Magenta MT3 (JAX/Flax) backend is not
wrapped (see Scope), and there's no batch-processing API, ONNX export, or
streaming inference — transcription is single-file, in-process, one model
at a time.

---

## Scope

**In scope:** a unified inference API over PyTorch-based MT3-family
transcription models, with checkpoint management (download, cache,
override) handled consistently across backends.

**Out of scope:**
- **Training or fine-tuning** any of the wrapped models — this is an
  inference-only toolkit.
- **Magenta MT3 (JAX/Flax)** — excluded due to dependency conflicts between
  the JAX/Flax/t5x/seqio stack and the PyTorch ecosystem the other three
  backends share. The three currently-wrapped PyTorch models provide
  comprehensive coverage for transcription scenarios in the meantime.
- **Editing/rendering vendored model code** — `mt3_infer/models/mt3_pytorch/`
  is license-frozen (no license declared upstream); it's kept exactly as
  extracted. See `CLAUDE.md` for the full constraint.

---

## Install

### Basic Installation

MT3-Infer is available on [PyPI](https://pypi.org/project/mt3-infer/).

```bash
# Using pip
pip install mt3-infer

# Using UV (recommended for development)
uv pip install mt3-infer
```

### Development Installation

```bash
# Clone repository
git clone https://github.com/openmirlab/mt3-infer.git
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

## Quick Start

### Simple Transcription (One Line!)

```python
from mt3_infer import transcribe

# Transcribe audio to MIDI (auto-downloads checkpoint on first use)
midi = transcribe(audio, sr=16000)
midi.save("output.mid")
```

### Model Selection

```python
# Use MR-MT3 model (57x real-time)
midi = transcribe(audio, model="mr_mt3")

# Use MT3-PyTorch model (147 notes detected)
midi = transcribe(audio, model="mt3_pytorch")

# Use YourMT3 model (multi-stem separation)
midi = transcribe(audio, model="yourmt3")
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
mt3-infer transcribe input.wav -o output.mid -m mr_mt3
```

> **Heads up:** The downloader now pulls MR-MT3 weights directly from
> [`gudgud1014/MR-MT3`](https://huggingface.co/gudgud1014/MR-MT3), so you no
> longer need Git LFS for that model. Checkpoints are stored under
> `.mt3_checkpoints/<model>` and will be re-created automatically if you delete
> the directory.

Set `MT3_CHECKPOINT_DIR` to store checkpoints somewhere else (e.g., shared storage) before running downloads or inference:

```bash
export MT3_CHECKPOINT_DIR=/data/models/mt3
```

Or use `.env` files (requires `python-dotenv`):

```bash
MT3_CHECKPOINT_DIR=/data/models/mt3
```

When the variable is set, both the Python API and CLI (including `mt3-infer download`) will read/write checkpoints inside that directory, preserving the same per-model layout as `.mt3_checkpoints/`.

---

## Supported Models

| Model | Framework | Speed | Notes Detected | Size | Features |
|-------|-----------|-------|----------------|------|----------|
| **MR-MT3** | PyTorch | 57x real-time | 116 notes | 176 MB | Optimized for speed |
| **MT3-PyTorch** | PyTorch | 12x real-time | 147 notes | 176 MB | Official architecture with auto-filtering* |
| **YourMT3** | PyTorch | ~15x real-time | 118 notes | 536 MB | 8-stem separation, Perceiver-TF + MoE |

*MT3-PyTorch includes automatic instrument leakage filtering (configurable via `auto_filter` parameter)

*Performance benchmarks from NVIDIA RTX 4090 with PyTorch 2.7.1 + CUDA 12.6*

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
info = get_model_info("mr_mt3")
print(f"Speed: {info['metadata']['performance']['speed_x_realtime']}x real-time")
```

### Disable Auto-Download

```python
from mt3_infer import load_model

# Raise error if checkpoint not found (don't auto-download)
model = load_model("mr_mt3", auto_download=False)
```

### Control MT3-PyTorch Instrument Filtering

MT3-PyTorch has automatic filtering to fix instrument leakage in drum tracks:

```python
# Default: filtering enabled (recommended)
model = load_model("mt3_pytorch")

# Disable filtering to see raw model output
model = load_model("mt3_pytorch", auto_filter=False)
```

### Override Checkpoint Directory

Use a shared storage location (e.g., NAS, cache volume) without changing your code:

```bash
export MT3_CHECKPOINT_DIR=/mnt/shared/mt3
uv run python -c "from mt3_infer import download_model; download_model('yourmt3')"
uv run mt3-infer download --all
```

To confirm the resolved location programmatically:

```python
from mt3_infer import download_model
path = download_model('mt3_pytorch')
print(path)
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
mt3-infer transcribe input.wav -m mr_mt3    # Use MR-MT3 model
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

See the CLI section above for detailed download instructions.

---

## Diagnostics & Troubleshooting

Extra smoke tests and tooling live in `examples/diagnostics/`:

- `download_mt3_pytorch.py` – manual vs. automatic checkpoint download walkthrough
- `test_all_models.py` – Loads all registered models and runs a short transcription
- `test_checkpoint_download.py` – Verifies checkpoints land in `MT3_CHECKPOINT_DIR`
- `test_yourmt3.py` – Full audio-to-MIDI flow for the YourMT3 MoE model

Run them via `uv run python examples/diagnostics/<script>.py` after setting any needed environment variables.

See also [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) for common issues.

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
│   ├── mr_mt3.py        # MR-MT3 adapter
│   ├── mt3_pytorch.py   # MT3-PyTorch adapter
│   ├── yourmt3.py       # YourMT3 adapter
│   └── vocab_utils.py   # Shared MIDI decoding
├── config/
│   └── checkpoints.yaml # Model registry & download config
├── utils/
│   ├── audio.py         # Audio preprocessing
│   ├── midi.py          # MIDI postprocessing
│   ├── download.py      # Checkpoint download system
│   └── framework.py     # Version checks
└── models/              # Model implementations
    ├── mr_mt3/          # MR-MT3 model code
    ├── mt3_pytorch/     # MT3-PyTorch model code
    └── yourmt3/         # YourMT3 model code
```

---

## Documentation

### For Users
- **[Main README](README.md)** - This file
- **[Examples](examples/)** - Usage examples
- **[Troubleshooting](docs/TROUBLESHOOTING.md)** - Common issues and solutions
- **[Benchmarks](docs/BENCHMARKS.md)** - Performance benchmarks

### For Developers
- **[Documentation Index](docs/README.md)** - Complete docs navigation
- **[API Specification](docs/dev/SPEC.md)** - Formal API spec
- **[Design Principles](docs/dev/PRINCIPLES.md)** - Development guidelines
- **[Download Guide](docs/dev/DOWNLOAD.md)** - Internal download documentation

---

## Integration with worzpro-demo

To use mt3-infer in the worzpro-demo project:

```toml
# In worzpro-demo/pyproject.toml
[tool.uv.sources]
mt3-infer = { git = "https://github.com/openmirlab/mt3-infer", extras = ["torch"] }
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

## What this project will NEVER bundle

MT3-Infer downloads pretrained checkpoints at runtime — it does **not**,
and will never, ship model weights inside the pip package or the git
repository itself:

- All three checkpoints (MR-MT3 ~176MB, MT3-PyTorch ~176MB, YourMT3 ~536MB)
  are fetched on first use into `.mt3_checkpoints/<model>/` (or
  `$MT3_CHECKPOINT_DIR`), never committed to this repo or bundled into the
  wheel — the published package is ~8MB of source only.
- The MR-MT3 checkpoint download is sha256-verified against a recorded hash
  before use. The MT3-PyTorch and YourMT3 checkpoints are currently
  downloaded via `git lfs` and are **not yet** checksum-verified at
  download time — their hashes are recorded for provenance/audit in
  [`mt3_infer/config/checkpoints.yaml`](mt3_infer/config/checkpoints.yaml),
  but enforcement at download time is a known gap, not yet wired in.
- Checkpoints are re-hosted or re-fetched from each backend's own upstream
  (HuggingFace for MR-MT3, git-lfs from the original repos for
  MT3-PyTorch and YourMT3) — none of these are currently mirrored under
  openmirlab control.

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
# Correct
uv run python script.py
uv run pytest

# Incorrect
python script.py
pytest
```

See docs/dev/PRINCIPLES.md for development guidelines.

### Contributing

We welcome contributions! Please:

1. Read docs/dev/SPEC.md for API specifications
2. Follow docs/dev/PRINCIPLES.md for development guidelines
3. Submit PRs with tests and documentation

---

## License

MIT License - see [LICENSE](LICENSE) for details.

See [Acknowledgments](#acknowledgments) above for the vendored projects this
repo builds on, and [LICENSE](LICENSE) /
[mt3_infer/config/external_integrations.yaml](mt3_infer/config/external_integrations.yaml)
for full per-model provenance and license status.

---

## Support

### Explicit lifecycle sessions

For reusable inference, use the composite session facade. A session owns one
selected profile (MR-MT3, MT3-PyTorch, or YourMT3), while other profiles may
remain in the disk checkpoint cache and are not loaded into memory.

```python
from mt3_infer import MT3Session

with MT3Session(model="accurate", device="cuda") as session:
    midi = session.infer(audio, sr=16000)
```

`load()` is idempotent and session-owned (it deliberately bypasses the legacy
one-shot global model cache); `release()` permits a later reload, and `close()`
is terminal and idempotent. `cache_info()` resolves the same custom/default
checkpoint path without downloading or creating directories. Devices accept
legacy `auto` plus explicit `cpu`, `cuda`, `cuda:N`, and `mps`; unavailable
explicit accelerators raise. Existing `load_model()` and `transcribe()`
one-shot APIs retain their opt-in global cache for compatibility. Profile
checkpoint metadata is package-owned in `mt3_infer/config/checkpoints.toml`.

For issues and questions:
- **GitHub Issues**: [github.com/openmirlab/mt3-infer/issues](https://github.com/openmirlab/mt3-infer/issues)
- **Documentation**: docs/
- **Examples**: examples/

---
