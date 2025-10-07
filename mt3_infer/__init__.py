"""
MT3-Infer: Unified inference-only toolkit for the MT3 model family.

This package provides a clean, framework-neutral API for running music transcription
inference across multiple MT3 implementations (Magenta MT3, MR-MT3, MT3-PyTorch, YourMT3).
"""

__version__ = "0.1.0"

from mt3_infer.base import MT3Base
from mt3_infer.exceptions import (
    AudioError,
    CheckpointError,
    CheckpointDownloadError,
    FrameworkError,
    InferenceError,
    ModelNotFoundError,
    MT3InferError,
)

# Public API (Phase 5 - implemented!)
from mt3_infer.api import (
    clear_cache,
    download_model,
    get_model_info,
    list_models,
    load_model,
    transcribe,
)

__all__ = [
    "__version__",
    "MT3Base",
    # Exceptions
    "MT3InferError",
    "ModelNotFoundError",
    "CheckpointError",
    "CheckpointDownloadError",
    "FrameworkError",
    "AudioError",
    "InferenceError",
    # Public API
    "transcribe",
    "load_model",
    "download_model",
    "list_models",
    "get_model_info",
    "clear_cache",
]
