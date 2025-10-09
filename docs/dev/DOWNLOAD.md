# Checkpoint Download Guide

MT3-Infer provides flexible checkpoint download options to suit different workflows.

## Quick Start

### Option 1: Automatic Download (Recommended)

Checkpoints are **automatically downloaded** when you first use a model:

```python
from mt3_infer import transcribe

# First time: downloads mt3_pytorch checkpoint automatically
midi = transcribe(audio)

# Subsequent uses: uses cached checkpoint
midi = transcribe(audio)
```

### Option 2: Manual Pre-Download

Download checkpoints before using them:

```python
from mt3_infer import download_model

# Download specific model
download_model("mr_mt3")

# Download all models
for model in ["mr_mt3", "mt3_pytorch", "yourmt3"]:
    download_model(model)
```

### Option 3: CLI Download

Use the command-line interface:

```bash
# Download all models
mt3-infer download --all

# Download specific models
mt3-infer download mr_mt3 mt3_pytorch

# Download using aliases
mt3-infer download fast accurate

# List available models
mt3-infer list
```

### Option 4: Standalone Script

Run the standalone download script:

```bash
# Download all models
uv run python tools/download_all_checkpoints.py

# Download specific models
uv run python tools/download_all_checkpoints.py --models mr_mt3 mt3_pytorch

# List available models
uv run python tools/download_all_checkpoints.py --list
```

## Download Methods

MT3-Infer supports multiple download sources:

### 1. Git LFS (Default)

All three models use **Git LFS** for efficient large file handling:

- **MR-MT3**: https://github.com/gudgud96/MR-MT3
- **MT3-PyTorch**: https://github.com/kunato/mt3-pytorch (Magenta's official weights)
- **YourMT3**: https://github.com/mimbres/YourMT3

**Requirements:**
```bash
# Ubuntu/Debian
sudo apt-get install git git-lfs

# macOS
brew install git git-lfs

# Windows
# Download from https://git-scm.com/ and https://git-lfs.github.com/
```

### 2. Direct URL Downloads

For custom checkpoints, you can use direct URLs:

```python
from mt3_infer.utils.download import download_file

download_file(
    url="https://example.com/checkpoint.pth",
    output_path="checkpoints/custom.pth",
    expected_sha256="abc123..."  # Optional verification
)
```

### 3. Hugging Face Hub

For models hosted on Hugging Face:

```python
from mt3_infer.utils.download import download_from_huggingface

download_from_huggingface(
    repo_id="username/repo-name",
    filename="model.pth",
    output_path="checkpoints/model.pth"
)
```

## Checkpoint Sizes

| Model | Size | Download Source |
|-------|------|----------------|
| MR-MT3 | 176 MB | GitHub (Git LFS) |
| MT3-PyTorch | 176 MB | GitHub (Git LFS) |
| YourMT3 | 522 MB | GitHub (Git LFS) |

**Total:** ~874 MB for all models

## Storage Locations

Checkpoints are stored in the package directory:

```
mt3-infer/
├── refs/
│   ├── MR-MT3/               # Cloned repository with checkpoint
│   ├── mt3-pytorch/          # Cloned repository with checkpoint
│   └── YourMT3/              # Cloned repository with checkpoint
```

## Advanced Usage

### Disable Auto-Download

If you want to ensure models don't download automatically:

```python
from mt3_infer import load_model

# Raises error if checkpoint not found
model = load_model("mr_mt3", auto_download=False)
```

### Custom Checkpoint Paths

Override the default checkpoint location:

```python
from mt3_infer import load_model

model = load_model(
    "mr_mt3",
    checkpoint_path="/path/to/custom/checkpoint.pth"
)
```

### Progress Monitoring

The download utilities show progress bars by default:

```
Downloading from: https://github.com/gudgud96/MR-MT3
Saving to: /path/to/mt3-infer/refs/MR-MT3
[==================================================] 100.0% (176.0/176.0 MB)
✓ Downloaded successfully
```

### Re-downloading Checkpoints

To re-download a checkpoint:

```bash
# Delete existing checkpoint
rm -rf refs/MR-MT3

# Re-download
mt3-infer download mr_mt3
```

## Troubleshooting

### Git LFS Not Installed

**Error:** `Git or Git LFS not found`

**Solution:**
```bash
# Ubuntu/Debian
sudo apt-get install git-lfs
git lfs install

# macOS
brew install git-lfs
git lfs install
```

### Slow Downloads

Git LFS downloads can be slow depending on GitHub's servers. Consider:

1. Using a stable network connection
2. Downloading during off-peak hours
3. Using a VPN if GitHub is throttled in your region

### Partial Downloads

If a download is interrupted:

```bash
# The package automatically cleans up partial downloads
# Simply re-run the download command
mt3-infer download mr_mt3
```

### Disk Space

Ensure you have enough disk space:
- **Single model:** ~200-600 MB
- **All models:** ~1 GB

Check available space:
```bash
df -h .
```

## Offline Usage

To prepare for offline usage:

1. Download all models on a machine with internet:
   ```bash
   mt3-infer download --all
   ```

2. Copy the entire `mt3-infer` package directory to the offline machine

3. Use models normally (auto-download will be skipped since checkpoints exist)

## CI/CD Integration

For automated environments (Docker, CI/CD):

```dockerfile
# Dockerfile example
FROM python:3.10

# Install Git LFS
RUN apt-get update && apt-get install -y git-lfs

# Install package
COPY . /app
WORKDIR /app
RUN pip install -e .

# Pre-download all models
RUN mt3-infer download --all

# Now ready for inference
CMD ["python", "your_script.py"]
```

## API Reference

### Python API

```python
from mt3_infer import download_model

# Download a model
checkpoint_path = download_model("mr_mt3")

# Returns:
#   - Path object pointing to checkpoint
#   - None if model uses built-in resolution
```

### CLI

```bash
# Show help
mt3-infer download --help

# Download all models
mt3-infer download --all

# Download specific models
mt3-infer download MODEL1 MODEL2 ...

# List available models
mt3-infer list
```

### Standalone Script

```bash
# Show help
python tools/download_all_checkpoints.py --help

# Download all models
python tools/download_all_checkpoints.py

# Download specific models
python tools/download_all_checkpoints.py --models MODEL1 MODEL2

# List available models
python tools/download_all_checkpoints.py --list
```

## See Also

- [README.md](../README.md) - Main documentation
- [CLAUDE.md](../CLAUDE.md) - Development guide
- [checkpoints.yaml](../mt3_infer/config/checkpoints.yaml) - Model registry
