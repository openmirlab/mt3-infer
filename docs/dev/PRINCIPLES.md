# Development Principles

## Package Management

### Use UV for Environment Management

This project uses [uv](https://github.com/astral-sh/uv) as the package and virtual environment manager.

**Key Rules:**

1. **Virtual Environment Management**: Always use `uv` to manage the virtual environment
   ```bash
   # Install dependencies
   uv sync

   # Add new dependencies
   uv add <package-name>
   ```

2. **Running Python Scripts**: Always use `uv run` instead of `python` directly
   ```bash
   # ✅ Correct
   uv run python main.py
   uv run python scripts/train.py

   # ❌ Incorrect
   python main.py
   ```

## Version Compatibility

### Align with worzpro-demo

This package is designed to be installed as a dependency in the [worzpro-demo](../../worzpro-demo) project. To avoid version conflicts, **always reference the worzpro-demo project** when specifying versions for packages that commonly cause conflicts.

### Critical Packages Reference

The following packages should always align with versions specified in `../../worzpro-demo/pyproject.toml`:

- **PyTorch Ecosystem**:
  - `torch==2.7.1`
  - `torchvision==0.22.1`
  - `torchaudio==2.7.1`

- **TensorFlow Ecosystem**:
  - `tensorflow>=2.13.0`

- **CUDA Dependencies**:
  - `nvidia-cuda-runtime-cu12>=12.6.77`
  - `nvidia-cudnn-cu12>=9.5.1.17`

- **Other Critical Libraries**:
  - `lightning>=2.3.0`
  - `protobuf>=3.20.3,<4.0` (override for TensorFlow compatibility)

### Before Adding Dependencies

When adding new dependencies, especially those related to ML/DL frameworks:

1. **Check worzpro-demo first**: Look at `../../worzpro-demo/pyproject.toml` to see if the package is already specified
2. **Match the version**: Use the same version constraint to avoid conflicts
3. **Test integration**: After adding, verify the package can be installed alongside worzpro-demo dependencies

Example workflow:
```bash
# 1. Check worzpro-demo for existing version
grep "torch" ../../worzpro-demo/pyproject.toml

# 2. Add with matching version
uv add "torch==2.7.1"

# 3. Verify no conflicts
uv sync
```

## Rationale

### Why UV?

- **Fast**: UV is significantly faster than pip for dependency resolution
- **Consistent**: Lockfile ensures reproducible builds across environments
- **Modern**: Better dependency resolution algorithm

### Why Version Alignment?

The worzpro-demo project integrates multiple ML/DL modules (including this one). PyTorch, TensorFlow, and CUDA libraries are notorious for version incompatibilities. By aligning versions across all modules:

- **Prevents runtime errors** from incompatible binary dependencies
- **Reduces installation time** by avoiding duplicate package versions
- **Ensures consistent behavior** across the entire demo application
- **Simplifies debugging** when issues arise

### Common Conflict Scenarios to Avoid

- Different PyTorch versions requiring different CUDA toolkit versions
- TensorFlow and PyTorch competing for GPU memory due to different CUDA runtime expectations
- Protobuf version conflicts between TensorFlow and other packages
- Audio processing libraries (torchaudio, librosa) with conflicting backend requirements

## Quick Reference

```bash
# Setup project
uv sync

# Run scripts
uv run python main.py

# Add dependency (check worzpro-demo first!)
uv add <package-name>

# Update dependencies
uv lock --upgrade

# Check for version in worzpro-demo
grep "<package-name>" ../../worzpro-demo/pyproject.toml
```
