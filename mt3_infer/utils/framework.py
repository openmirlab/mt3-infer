"""
Framework version checking and device utilities.

Ensures framework versions match requirements and provides device detection helpers.
"""

from typing import Optional

from mt3_infer.exceptions import FrameworkError


def check_torch_version() -> None:
    """
    Verify PyTorch version matches requirements.

    Raises:
        FrameworkError: Version mismatch or torch not installed.

    Note:
        Required version: torch==2.7.1 (aligned with worzpro-demo)
    """
    try:
        import torch
    except ImportError as e:
        raise FrameworkError(
            "PyTorch is not installed.\n"
            "Install with: uv sync --extra torch\n"
            "Or: uv add 'mt3-infer[torch]'"
        ) from e

    required_version = "2.7.1"
    actual_version = torch.__version__.split("+")[0]  # Remove CUDA suffix

    if not actual_version.startswith(required_version):
        raise FrameworkError(
            f"torch=={required_version} required, found torch=={actual_version}\n"
            "Version mismatch detected to prevent conflicts with worzpro-demo.\n"
            "Fix with: uv sync --reinstall-package torch\n"
            "See docs/dev/PRINCIPLES.md for version alignment strategy."
        )


def check_tensorflow_version() -> None:
    """
    Verify TensorFlow version matches requirements.

    Raises:
        FrameworkError: Version mismatch or tensorflow not installed.

    Note:
        Required version: tensorflow>=2.13.0 (aligned with worzpro-demo)
    """
    try:
        import tensorflow as tf
    except ImportError as e:
        raise FrameworkError(
            "TensorFlow is not installed.\n"
            "Install with: uv sync --extra tensorflow\n"
            "Or: uv add 'mt3-infer[tensorflow]'"
        ) from e

    required_major = 2
    required_minor = 13
    actual_version = tf.__version__

    try:
        major, minor, *_ = map(int, actual_version.split(".")[:2])
    except (ValueError, IndexError) as e:
        raise FrameworkError(
            f"Cannot parse TensorFlow version: {actual_version}"
        ) from e

    if major < required_major or (major == required_major and minor < required_minor):
        raise FrameworkError(
            f"tensorflow>={required_major}.{required_minor}.0 required, "
            f"found tensorflow=={actual_version}\n"
            "Upgrade with: uv sync --upgrade-package tensorflow"
        )


def check_jax_version() -> None:
    """
    Verify JAX version matches requirements.

    Raises:
        FrameworkError: Version mismatch or jax not installed.

    Note:
        Required version: jax==0.4.28 (mt3-infer specific)
    """
    try:
        import jax
    except ImportError as e:
        raise FrameworkError(
            "JAX is not installed.\n"
            "Install with: uv sync --extra jax\n"
            "Or: uv add 'mt3-infer[jax]'"
        ) from e

    required_version = "0.4.28"
    actual_version = jax.__version__

    if not actual_version.startswith(required_version):
        raise FrameworkError(
            f"jax=={required_version} required, found jax=={actual_version}\n"
            "Fix with: uv sync --reinstall-package jax"
        )


def get_device(device_hint: Optional[str] = None) -> str:
    """
    Determine the target device for model inference.

    Args:
        device_hint: User-specified device ("cuda", "cpu", "auto", None).
                    "auto" or None will auto-detect CUDA availability.

    Returns:
        Normalized device string: "cuda" or "cpu".

    Raises:
        ValueError: Invalid device hint.

    Note:
        This function does NOT check framework availability.
        Framework-specific device placement should be handled by adapters.

    Example:
        >>> device = get_device("auto")  # Returns "cuda" if available, else "cpu"
        >>> device = get_device("cpu")   # Forces CPU
    """
    if device_hint is None:
        device_hint = "auto"

    device_hint = device_hint.lower()

    if device_hint not in ("cuda", "cpu", "auto"):
        raise ValueError(
            f"Invalid device: {device_hint}. Must be 'cuda', 'cpu', or 'auto'."
        )

    if device_hint == "auto":
        # Try to detect CUDA availability
        # This is framework-agnostic detection
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass

        try:
            import tensorflow as tf
            if tf.config.list_physical_devices("GPU"):
                return "cuda"  # Return "cuda" for consistency
        except ImportError:
            pass

        try:
            import jax
            if jax.devices("gpu"):
                return "cuda"  # Return "cuda" for consistency
        except ImportError:
            pass

        # No GPU detected, default to CPU
        import warnings
        warnings.warn(
            "No GPU detected. Using CPU for inference. "
            "This may be significantly slower than GPU inference.",
            UserWarning
        )
        return "cpu"

    return device_hint
