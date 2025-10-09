"""
Public API for mt3_infer package.

This module provides high-level functions for model loading and transcription.
"""

import importlib
import os
from pathlib import Path
from typing import Any, Dict, Optional

import mido
import numpy as np
import yaml

from mt3_infer.base import MT3Base
from mt3_infer.exceptions import ModelNotFoundError, CheckpointDownloadError
from mt3_infer.utils.download import download_checkpoint

# Global model cache
_MODEL_CACHE: Dict[str, MT3Base] = {}
_REGISTRY: Optional[Dict[str, Any]] = None


def _load_registry() -> Dict[str, Any]:
    """Load model registry from checkpoints.yaml."""
    global _REGISTRY
    
    if _REGISTRY is not None:
        return _REGISTRY
    
    config_path = Path(__file__).parent / "config" / "checkpoints.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(
            f"Model registry not found: {config_path}\n"
            "Package may be corrupted. Please reinstall mt3-infer."
        )
    
    with open(config_path, "r") as f:
        _REGISTRY = yaml.safe_load(f)
    
    return _REGISTRY


def _resolve_model_name(model: str) -> str:
    """Resolve model aliases to canonical names."""
    registry = _load_registry()
    
    # Check if it's an alias
    if "aliases" in registry and model in registry["aliases"]:
        return registry["aliases"][model]
    
    # Check if it's a valid model name
    if model in registry["models"]:
        return model
    
    # Check if it's 'default'
    if model == "default" and "default" in registry:
        return registry["default"]
    
    # Model not found
    available = list(registry["models"].keys())
    aliases = list(registry.get("aliases", {}).keys())
    raise ModelNotFoundError(
        f"Model '{model}' not found in registry.\n"
        f"Available models: {', '.join(available)}\n"
        f"Available aliases: {', '.join(aliases)}"
    )


def load_model(
    model: str = "default",
    checkpoint_path: Optional[str] = None,
    device: str = "auto",
    cache: bool = True,
    auto_download: bool = True,
    **kwargs
) -> MT3Base:
    """
    Load an MT3 model from the registry.

    Args:
        model: Model identifier (e.g., "mr_mt3", "mt3_pytorch", "yourmt3")
               or alias (e.g., "default", "official", "baseline").
        checkpoint_path: Override default checkpoint path (optional).
        device: Device placement ("cuda", "cpu", "auto"). Default is "auto".
        cache: Cache loaded models for reuse. Default is True.
        auto_download: Automatically download checkpoint if missing. Default is True.
        **kwargs: Additional model-specific parameters:
                 - auto_filter (bool): For MT3-PyTorch, enable automatic instrument
                   leakage filtering (default: True)

    Returns:
        Initialized MT3 model adapter.

    Raises:
        ModelNotFoundError: Unknown model name or alias.
        FileNotFoundError: Checkpoint not found and auto_download is False.
        CheckpointDownloadError: Download failed.
        RuntimeError: Model initialization failure.
        ImportError: Required framework not installed.

    Examples:
        >>> # Load default model (mt3_pytorch) - auto-downloads if needed
        >>> model = load_model()

        >>> # Load by model identifier
        >>> mr_mt3 = load_model("mr_mt3")  # 57x realtime
        >>> mt3_pytorch = load_model("mt3_pytorch")  # 12x realtime, official
        >>> yourmt3 = load_model("yourmt3")  # Multi-task

        >>> # Load specific model with device
        >>> model = load_model("mt3_pytorch", device="cuda")
        >>> midi = model.transcribe(audio, sr=16000)

        >>> # Disable auto-download (fail if checkpoint missing)
        >>> model = load_model("mr_mt3", auto_download=False)

        >>> # Disable automatic filtering for MT3-PyTorch
        >>> model = load_model("mt3_pytorch", auto_filter=False)
    """
    # Resolve model name/alias
    model_name = _resolve_model_name(model)

    # Check cache
    cache_key = f"{model_name}:{checkpoint_path or 'default'}:{device}"
    if cache and cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    # Load registry
    registry = _load_registry()
    model_config = registry["models"][model_name]

    # Determine checkpoint path
    if checkpoint_path is None:
        checkpoint_path = model_config["checkpoint"]["path"]
        if checkpoint_path is not None:
            checkpoint_path_obj = Path(checkpoint_path)
            env_dir = os.getenv("MT3_CHECKPOINT_DIR")
            if env_dir and not checkpoint_path_obj.is_absolute():
                base_dir = Path(env_dir).expanduser()
                try:
                    checkpoint_path_obj = base_dir / checkpoint_path_obj.relative_to(".mt3_checkpoints")
                except ValueError:
                    checkpoint_path_obj = base_dir / checkpoint_path_obj
            elif not checkpoint_path_obj.is_absolute():
                checkpoint_path_obj = Path.cwd() / checkpoint_path_obj
            checkpoint_path = str(checkpoint_path_obj)

    # Auto-download if checkpoint is missing
    if checkpoint_path is not None and auto_download:
        checkpoint_path_obj = Path(checkpoint_path)

        # Check if checkpoint exists (file or non-empty directory)
        checkpoint_missing = (
            not checkpoint_path_obj.exists() or
            (checkpoint_path_obj.is_dir() and not any(checkpoint_path_obj.iterdir()))
        )

        if checkpoint_missing and "download" in model_config["checkpoint"]:
            download_info = model_config["checkpoint"]["download"]

            # Determine output path based on source type
            if download_info["source_type"] == "git_lfs":
                # For git repos, clone to temp location then copy checkpoint
                import shutil
                import tempfile

                repo_name = download_info["source_url"].rstrip("/").split("/")[-1]

                with tempfile.TemporaryDirectory() as tmpdir:
                    temp_clone_path = Path(tmpdir) / repo_name

                    # Clone to temporary location
                    download_checkpoint(
                        source_type=download_info["source_type"],
                        source_url=download_info["source_url"],
                        output_path=temp_clone_path,
                        **{k: v for k, v in download_info.items()
                           if k not in ["source_type", "source_url", "target_path"]}
                    )

                    # Copy checkpoint file/directory to permanent location
                    target_path = download_info.get("target_path", "")
                    source_checkpoint = temp_clone_path / target_path

                    if not source_checkpoint.exists():
                        raise CheckpointDownloadError(
                            f"Checkpoint not found in repository: {target_path}\n"
                            f"Repository cloned to: {temp_clone_path}"
                        )

                    checkpoint_path_obj.parent.mkdir(parents=True, exist_ok=True)

                    # Handle both files and directories
                    if source_checkpoint.is_file():
                        shutil.copy2(source_checkpoint, checkpoint_path_obj)
                    elif source_checkpoint.is_dir():
                        if checkpoint_path_obj.exists():
                            shutil.rmtree(checkpoint_path_obj)
                        shutil.copytree(source_checkpoint, checkpoint_path_obj)

                    print(f"✓ Checkpoint copied to: {checkpoint_path_obj}")
            else:
                # For direct downloads, use the checkpoint path
                try:
                    download_checkpoint(
                        source_type=download_info["source_type"],
                        source_url=download_info["source_url"],
                        output_path=checkpoint_path_obj,
                        **{k: v for k, v in download_info.items()
                           if k not in ["source_type", "source_url"]}
                    )
                except CheckpointDownloadError as e:
                    raise CheckpointDownloadError(
                        f"Failed to download checkpoint for model '{model_name}':\n{e}\n\n"
                        f"You can disable auto-download with: load_model('{model}', auto_download=False)"
                    )
    
    # Dynamically import adapter class
    adapter_class_path = model_config["adapter_class"]
    module_path, class_name = adapter_class_path.rsplit(".", 1)
    
    try:
        module = importlib.import_module(module_path)
        AdapterClass = getattr(module, class_name)
    except ImportError as e:
        raise ImportError(
            f"Failed to import adapter: {adapter_class_path}\n"
            f"Error: {e}\n"
            "Ensure required framework is installed."
        )
    
    # Instantiate and load model
    # Handle adapter-specific initialization parameters
    # Pass kwargs to adapter (e.g., auto_filter for MT3PyTorchAdapter)
    try:
        adapter = AdapterClass(**kwargs)
    except TypeError:
        # If adapter doesn't accept kwargs, instantiate without them
        adapter = AdapterClass()

    adapter.load_model(checkpoint_path, device=device)
    
    # Cache if enabled
    if cache:
        _MODEL_CACHE[cache_key] = adapter
    
    return adapter


def transcribe(
    audio: np.ndarray,
    model: str = "default",
    sr: int = 16000,
    checkpoint_path: Optional[str] = None,
    device: str = "auto",
    auto_download: bool = True,
    **kwargs
) -> mido.MidiFile:
    """
    Transcribe audio to MIDI using specified MT3 model.

    This is the simplest way to use mt3_infer. The model is automatically loaded
    from the registry, cached, and used for transcription. Checkpoints are
    automatically downloaded if missing.

    Args:
        audio: Audio waveform with shape (n_samples,) or (n_samples, n_channels).
               Values should be in range [-1.0, 1.0], dtype float32 or float64.
        model: Model identifier or alias (e.g., "mr_mt3", "mt3_pytorch", "yourmt3").
               Default is "default" (mt3_pytorch).
        sr: Sample rate in Hz. Default is 16000.
        checkpoint_path: Override default checkpoint path (optional).
        device: Device placement ("cuda", "cpu", "auto"). Default is "auto".
        auto_download: Automatically download checkpoint if missing. Default is True.
        **kwargs: Model-specific parameters:

                 For MT3-PyTorch:
                 - auto_filter (bool): Enable automatic instrument leakage filtering
                   (default: True)

                 For YourMT3 (adaptive preprocessing):
                 - adaptive (bool): Enable adaptive preprocessing for dense drum patterns
                   (default: False)
                 - stretch_factors (List[float]): Time-stretch factors to try
                 - stretch_method (str): "librosa", "pyrubberband", or "auto"
                 - num_attempts (int): Number of attempts per configuration
                 - return_all (bool): Return all results for manual selection

    Returns:
        MIDI file object with transcribed notes.

    Raises:
        ValueError: Invalid audio format, sample rate, or model name.
        RuntimeError: Model loading or inference failure.
        ImportError: Required framework not installed.

    Examples:
        >>> import numpy as np
        >>> from mt3_infer import transcribe
        >>>
        >>> # Simple transcription with default model
        >>> audio = np.random.randn(16000 * 5).astype(np.float32)
        >>> midi = transcribe(audio)
        >>> midi.save("output.mid")
        >>>
        >>> # Use MR-MT3 (57x realtime)
        >>> midi = transcribe(audio, model="mr_mt3")
        >>>
        >>> # Use MT3-PyTorch (12x realtime, official)
        >>> midi = transcribe(audio, model="mt3_pytorch")
        >>>
        >>> # Use YourMT3 with adaptive preprocessing for dense drums
        >>> midi = transcribe(
        ...     audio, model="yourmt3",
        ...     adaptive=True,
        ...     num_attempts=3
        ... )
        >>>
        >>> # Specify device
        >>> midi = transcribe(audio, model="mt3_pytorch", device="cuda")
    """
    # Separate model initialization kwargs from transcribe kwargs
    model_init_kwargs = {k: v for k, v in kwargs.items()
                         if k in ['auto_filter', 'verbose']}
    transcribe_kwargs = {k: v for k, v in kwargs.items()
                        if k in ['adaptive', 'stretch_factors', 'stretch_method',
                                 'num_attempts', 'return_all']}

    # Load model (uses cache by default, auto-downloads if needed)
    mt3_model = load_model(
        model=model,
        checkpoint_path=checkpoint_path,
        device=device,
        auto_download=auto_download,
        **model_init_kwargs
    )

    # Transcribe with model-specific parameters
    midi = mt3_model.transcribe(audio, sr=sr, **transcribe_kwargs)

    return midi


def list_models() -> Dict[str, Dict[str, Any]]:
    """
    List all available models in the registry.

    Returns:
        Dictionary mapping model names to their metadata.

    Examples:
        >>> from mt3_infer import list_models
        >>> 
        >>> models = list_models()
        >>> for name, info in models.items():
        ...     print(f"{name}: {info['description']}")
        mr_mt3: Multi-instrument MT3 variant optimized for speed (57x real-time)
        mt3_pytorch: Official MT3 architecture in PyTorch (best accuracy, 12x real-time)
        yourmt3: Extended multi-task MT3 with 8-stem separation (~15x real-time)
    """
    registry = _load_registry()
    return registry["models"]


def get_model_info(model: str) -> Dict[str, Any]:
    """
    Get detailed information about a specific model.

    Args:
        model: Model identifier or alias.

    Returns:
        Dictionary containing model metadata.

    Raises:
        ModelNotFoundError: Unknown model name or alias.

    Examples:
        >>> from mt3_infer import get_model_info
        >>>
        >>> info = get_model_info("mr_mt3")
        >>> print(info["name"])
        MR-MT3
        >>> print(info["metadata"]["performance"]["speed_x_realtime"])
        57.0
    """
    model_name = _resolve_model_name(model)
    registry = _load_registry()
    return registry["models"][model_name]


def clear_cache() -> None:
    """
    Clear the model cache, freeing up memory.

    Useful when switching between many models or when memory is limited.

    Examples:
        >>> from mt3_infer import load_model, clear_cache
        >>> 
        >>> # Load and use model
        >>> model = load_model("mt3_pytorch")
        >>> # ... use model ...
        >>> 
        >>> # Free memory
        >>> clear_cache()
    """
    global _MODEL_CACHE
    _MODEL_CACHE.clear()


def download_model(model: str = "default") -> Optional[Path]:
    """
    Manually download a model checkpoint before using it.

    This allows users to pre-download checkpoints separately from model loading,
    which is useful for:
    - Downloading all models upfront
    - Verifying downloads before inference
    - Preparing environments offline

    Args:
        model: Model identifier or alias (e.g., "mr_mt3", "mt3_pytorch", "yourmt3").
               Default is "default" (mt3_pytorch).

    Returns:
        Path to the downloaded checkpoint, or None if no download needed.

    Raises:
        ModelNotFoundError: Unknown model name or alias.
        CheckpointDownloadError: Download failed.

    Examples:
        >>> from mt3_infer import download_model
        >>>
        >>> # Download default model
        >>> checkpoint_path = download_model()
        >>> print(f"Downloaded to: {checkpoint_path}")
        >>>
        >>> # Download all models
        >>> for model_id in ["mr_mt3", "mt3_pytorch", "yourmt3"]:
        ...     download_model(model_id)
        >>>
        >>> # Download specific model
        >>> download_model("mr_mt3")  # Downloads MR-MT3
    """
    # Resolve model name/alias
    model_name = _resolve_model_name(model)

    # Load registry
    registry = _load_registry()
    model_config = registry["models"][model_name]

    # Get checkpoint path
    checkpoint_path = model_config["checkpoint"]["path"]
    if checkpoint_path is None:
        print(f"✓ Model '{model_name}' uses built-in checkpoint resolution (no download needed)")
        return None

    checkpoint_path_obj = Path(checkpoint_path)

    if not checkpoint_path_obj.is_absolute():
        env_dir = os.getenv("MT3_CHECKPOINT_DIR")
        if env_dir:
            base_dir = Path(env_dir).expanduser()
            try:
                checkpoint_path_obj = base_dir / checkpoint_path_obj.relative_to(".mt3_checkpoints")
            except ValueError:
                checkpoint_path_obj = base_dir / checkpoint_path_obj
        else:
            checkpoint_path_obj = Path.cwd() / checkpoint_path_obj

    checkpoint_path = str(checkpoint_path_obj)

    # Check if already downloaded
    if checkpoint_path_obj.exists():
        if checkpoint_path_obj.is_file():
            print(f"✓ Checkpoint already exists: {checkpoint_path}")
            return checkpoint_path_obj
        elif checkpoint_path_obj.is_dir() and any(checkpoint_path_obj.iterdir()):
            print(f"✓ Checkpoint directory already exists: {checkpoint_path}")
            return checkpoint_path_obj

    # Download
    if "download" not in model_config["checkpoint"]:
        raise CheckpointDownloadError(
            f"No download information available for model '{model_name}'\n"
            f"Expected checkpoint at: {checkpoint_path}"
        )

    download_info = model_config["checkpoint"]["download"]

    # Determine output path based on source type
    if download_info["source_type"] == "git_lfs":
        # For git repos, clone to temp location then copy checkpoint
        import shutil
        import tempfile

        repo_name = download_info["source_url"].rstrip("/").split("/")[-1]

        with tempfile.TemporaryDirectory() as tmpdir:
            temp_clone_path = Path(tmpdir) / repo_name

            # Clone to temporary location
            download_checkpoint(
                source_type=download_info["source_type"],
                source_url=download_info["source_url"],
                output_path=temp_clone_path,
                **{k: v for k, v in download_info.items()
                   if k not in ["source_type", "source_url", "target_path"]}
            )

            # Copy checkpoint file/directory to permanent location
            target_path = download_info.get("target_path", "")
            source_checkpoint = temp_clone_path / target_path

            if not source_checkpoint.exists():
                raise CheckpointDownloadError(
                    f"Checkpoint not found in repository: {target_path}\n"
                    f"Repository cloned to: {temp_clone_path}"
                )

            checkpoint_path_obj.parent.mkdir(parents=True, exist_ok=True)

            # Handle both files and directories
            if source_checkpoint.is_file():
                shutil.copy2(source_checkpoint, checkpoint_path_obj)
            elif source_checkpoint.is_dir():
                if checkpoint_path_obj.exists():
                    shutil.rmtree(checkpoint_path_obj)
                shutil.copytree(source_checkpoint, checkpoint_path_obj)

            print(f"\n✓ Model '{model_name}' checkpoint downloaded successfully")
            print(f"  Location: {checkpoint_path}")

            return checkpoint_path_obj
    else:
        # For direct downloads, use the checkpoint path
        try:
            download_checkpoint(
                source_type=download_info["source_type"],
                source_url=download_info["source_url"],
                output_path=checkpoint_path_obj,
                **{k: v for k, v in download_info.items()
                   if k not in ["source_type", "source_url"]}
            )

            print(f"\n✓ Model '{model_name}' checkpoint downloaded successfully")
            print(f"  Location: {checkpoint_path}")

            return checkpoint_path_obj

        except CheckpointDownloadError as e:
            raise CheckpointDownloadError(
                f"Failed to download checkpoint for model '{model_name}':\n{e}"
            )
