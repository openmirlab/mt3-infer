"""
Checkpoint download utilities for mt3_infer.

Supports downloading model checkpoints from various sources:
- GitHub Releases (via direct URL)
- Git LFS (via git clone + lfs pull)
- Hugging Face Hub
"""

import hashlib
import shutil
import subprocess
import urllib.request
from pathlib import Path
from typing import Optional, Callable
from urllib.parse import urlparse

from mt3_infer.exceptions import CheckpointDownloadError


def _show_progress(block_num: int, block_size: int, total_size: int) -> None:
    """Default progress callback for urllib downloads."""
    downloaded = block_num * block_size
    percent = min(100, (downloaded / total_size) * 100)
    mb_downloaded = downloaded / (1024 * 1024)
    mb_total = total_size / (1024 * 1024)

    # Simple progress bar (50 chars wide)
    bar_len = 50
    filled = int(bar_len * percent / 100)
    bar = "=" * filled + "-" * (bar_len - filled)

    print(f"\r[{bar}] {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end="", flush=True)

    if percent >= 100:
        print()  # New line when complete


def download_file(
    url: str,
    output_path: Path,
    expected_sha256: Optional[str] = None,
    progress_callback: Optional[Callable[[int, int, int], None]] = None
) -> None:
    """
    Download a file from URL with progress tracking and verification.

    Args:
        url: Download URL.
        output_path: Local file path to save downloaded file.
        expected_sha256: Expected SHA-256 checksum (optional).
        progress_callback: Progress callback (block_num, block_size, total_size).
                          Default shows a progress bar.

    Raises:
        CheckpointDownloadError: Download or verification failed.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Use default progress callback if none provided
    if progress_callback is None:
        progress_callback = _show_progress

    try:
        print(f"Downloading from: {url}")
        print(f"Saving to: {output_path}")

        urllib.request.urlretrieve(url, str(output_path), reporthook=progress_callback)

        print(f"✓ Downloaded successfully")

        # Verify checksum if provided
        if expected_sha256:
            print(f"Verifying checksum...")
            actual_sha256 = _compute_sha256(output_path)

            if actual_sha256 != expected_sha256:
                output_path.unlink()  # Delete corrupted file
                raise CheckpointDownloadError(
                    f"Checksum verification failed!\n"
                    f"Expected: {expected_sha256}\n"
                    f"Got:      {actual_sha256}\n"
                    "File may be corrupted. Please try again."
                )

            print(f"✓ Checksum verified")

    except Exception as e:
        if output_path.exists():
            output_path.unlink()  # Clean up partial download
        raise CheckpointDownloadError(f"Failed to download {url}: {e}")


def _compute_sha256(file_path: Path) -> str:
    """Compute SHA-256 checksum of a file."""
    sha256 = hashlib.sha256()

    with open(file_path, "rb") as f:
        # Read in chunks to handle large files
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def clone_git_lfs_repo(
    repo_url: str,
    output_dir: Path,
    branch: Optional[str] = None,
    depth: int = 1
) -> None:
    """
    Clone a Git repository with LFS files.

    Args:
        repo_url: Git repository URL.
        output_dir: Local directory to clone into.
        branch: Specific branch to clone (optional).
        depth: Clone depth (default: 1 for shallow clone).

    Raises:
        CheckpointDownloadError: Clone or LFS pull failed.
    """
    output_dir = Path(output_dir)

    if output_dir.exists():
        print(f"✓ Repository already exists: {output_dir}")
        return

    output_dir.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Build clone command
        cmd = ["git", "clone"]

        if depth > 0:
            cmd.extend(["--depth", str(depth)])

        if branch:
            cmd.extend(["--branch", branch])

        cmd.extend([repo_url, str(output_dir)])

        print(f"Cloning repository: {repo_url}")
        subprocess.run(cmd, check=True, capture_output=True, text=True)

        # Pull LFS files
        print(f"Pulling LFS files...")
        subprocess.run(
            ["git", "lfs", "pull"],
            cwd=str(output_dir),
            check=True,
            capture_output=True,
            text=True
        )

        print(f"✓ Repository cloned successfully")

    except subprocess.CalledProcessError as e:
        # Clean up failed clone
        if output_dir.exists():
            shutil.rmtree(output_dir)

        raise CheckpointDownloadError(
            f"Failed to clone repository: {repo_url}\n"
            f"Error: {e.stderr}"
        )
    except FileNotFoundError:
        raise CheckpointDownloadError(
            "Git or Git LFS not found. Please install:\n"
            "  Ubuntu/Debian: sudo apt-get install git git-lfs\n"
            "  macOS: brew install git git-lfs\n"
            "  Windows: Download from https://git-scm.com/"
        )


def download_from_huggingface(
    repo_id: str,
    filename: str,
    output_path: Path,
    revision: str = "main"
) -> None:
    """
    Download a file from Hugging Face Hub.

    Args:
        repo_id: Hugging Face repository ID (e.g., "username/repo-name").
        filename: File name within the repository.
        output_path: Local path to save the file.
        revision: Git revision (branch/tag/commit). Default is "main".

    Raises:
        CheckpointDownloadError: Download failed.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise CheckpointDownloadError(
            "Hugging Face Hub downloads require huggingface_hub package.\n"
            "Install with: uv add huggingface-hub"
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        print(f"Downloading from Hugging Face: {repo_id}/{filename}")

        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
            cache_dir=None,  # Use default cache
            local_dir=str(output_path.parent),
            local_dir_use_symlinks=False
        )

        # Move to final location if needed
        if Path(downloaded_path) != output_path:
            shutil.move(downloaded_path, output_path)

        print(f"✓ Downloaded successfully")

    except Exception as e:
        raise CheckpointDownloadError(f"Failed to download from Hugging Face: {e}")


def download_checkpoint(
    source_type: str,
    source_url: str,
    output_path: Path,
    sha256: Optional[str] = None,
    **kwargs
) -> None:
    """
    Download a checkpoint from various sources.

    Args:
        source_type: Download source type ("url", "git_lfs", "huggingface").
        source_url: Source URL or repository identifier.
        output_path: Local path to save the checkpoint.
        sha256: Expected SHA-256 checksum (optional, for URL downloads).
        **kwargs: Additional arguments passed to specific download functions.

    Raises:
        CheckpointDownloadError: Download failed.
    """
    output_path = Path(output_path)

    # Skip if already exists
    if output_path.exists():
        if output_path.is_file():
            print(f"✓ Checkpoint already exists: {output_path}")
            return
        elif output_path.is_dir() and any(output_path.iterdir()):
            print(f"✓ Checkpoint directory already exists: {output_path}")
            return

    print(f"\n{'=' * 70}")
    print(f"Downloading Checkpoint")
    print(f"{'=' * 70}")

    if source_type == "url":
        download_file(source_url, output_path, expected_sha256=sha256)

    elif source_type == "git_lfs":
        clone_git_lfs_repo(source_url, output_path, **kwargs)

    elif source_type == "huggingface":
        # Parse repo_id and filename from source_url
        # Expected format: "repo_id:filename" or just repo_id (downloads whole repo)
        if ":" in source_url:
            repo_id, filename = source_url.split(":", 1)
            download_from_huggingface(repo_id, filename, output_path, **kwargs)
        else:
            raise CheckpointDownloadError(
                "Hugging Face downloads require format 'repo_id:filename'\n"
                f"Got: {source_url}"
            )

    else:
        raise CheckpointDownloadError(
            f"Unsupported source type: {source_type}\n"
            "Supported types: url, git_lfs, huggingface"
        )

    print(f"{'=' * 70}\n")
