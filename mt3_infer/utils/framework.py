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
        Required version: torch>=2.4.0 (matches pyproject.toml floor)
    """
    try:
        import torch
    except ImportError as e:
        raise FrameworkError(
            "PyTorch is not installed.\n"
            "Install with: uv sync --extra torch\n"
            "Or: uv add 'mt3-infer[torch]'"
        ) from e

    required_major = 2
    required_minor = 4
    actual_version = torch.__version__.split("+")[0]  # Remove CUDA suffix

    try:
        major, minor, *_ = map(int, actual_version.split(".")[:2])
    except (ValueError, IndexError) as e:
        raise FrameworkError(
            f"Cannot parse PyTorch version: {actual_version}"
        ) from e

    if major < required_major or (major == required_major and minor < required_minor):
        raise FrameworkError(
            f"torch>={required_major}.{required_minor}.0 required, "
            f"found torch=={actual_version}\n"
            "Upgrade with: pip install --upgrade torch"
        )


def check_tensorflow_version() -> None:
    """
    Verify TensorFlow version matches requirements.

    Raises:
        FrameworkError: Version mismatch or tensorflow not installed.

    Note:
        Required version: tensorflow>=2.16.0 (matches pyproject.toml floor)
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
    required_minor = 16
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
        device_hint: User-specified device ("cuda", "cpu", "auto", "cuda:N", None).
                    "auto" or None will auto-detect CUDA availability. An
                    explicit "cuda"/"cuda:N" request is honored as-is -- see
                    Raises below -- matching the mt3_pytorch and yourmt3
                    adapters, which pass an explicit device straight to
                    torch and let it fail hard rather than silently
                    substituting CPU.

    Returns:
        Normalized device string: "cuda", "cuda:N", or "cpu".

    Raises:
        ValueError: Invalid device hint.
        RuntimeError: An explicitly-requested "cuda"/"cuda:N" device is not
            usable (CUDA unavailable, or GPU initialization failed). Only
            "auto" falls back to CPU; an explicit request is never silently
            downgraded.

    Note:
        "auto" still falls back to CPU with a warning when no working GPU
        is found -- that behavior is unchanged. Only the explicit-request
        path was changed to raise instead of falling back.

    Example:
        >>> device = get_device("auto")  # Returns "cuda" if available, else "cpu"
        >>> device = get_device("cpu")   # Forces CPU
        >>> device = get_device("cuda")  # Returns "cuda", or raises RuntimeError
    """
    import warnings

    if device_hint is None:
        device_hint = "auto"

    device_hint = device_hint.lower()

    is_cuda_request = device_hint == "cuda" or device_hint.startswith("cuda:")

    if device_hint not in ("cpu", "auto") and not is_cuda_request:
        raise ValueError(
            f"Invalid device: {device_hint}. Must be 'cuda', 'cuda:N', 'cpu', or 'auto'."
        )

    if device_hint == "auto":
        # Try PyTorch CUDA with actual tensor test (not just is_available)
        try:
            import torch
            if torch.cuda.is_available():
                # Actually try to use CUDA - this catches runtime errors
                try:
                    test_tensor = torch.zeros(1, device="cuda")
                    del test_tensor
                    return "cuda"
                except Exception as e:
                    warnings.warn(
                        f"CUDA available but failed to initialize: {e}. "
                        "Falling back to CPU.",
                        UserWarning
                    )
        except ImportError:
            pass

        # No working GPU, default to CPU
        warnings.warn(
            "No GPU detected or GPU initialization failed. Using CPU for inference. "
            "This may be significantly slower than GPU inference.",
            UserWarning
        )
        return "cpu"

    # User explicitly requested cuda (or cuda:N) - honor it or raise.
    # Unlike "auto", an explicit request is never silently downgraded to
    # CPU: that would override the caller's stated choice. This mirrors
    # mt3_pytorch.py and yourmt3.py, which pass an explicit device straight
    # to torch and let `.to(device)` raise naturally when it's unavailable.
    if is_cuda_request:
        try:
            import torch
        except ImportError as e:
            raise RuntimeError(
                "CUDA requested but PyTorch is not installed."
            ) from e

        if not torch.cuda.is_available():
            raise RuntimeError(
                f"Device '{device_hint}' was explicitly requested but CUDA is "
                "not available. Pass device='cpu' or device='auto' to allow "
                "CPU fallback."
            )

        try:
            test_tensor = torch.zeros(1, device=device_hint)
            del test_tensor
            return device_hint
        except Exception as e:
            raise RuntimeError(
                f"Device '{device_hint}' was explicitly requested but failed "
                f"to initialize: {e}"
            ) from e

    return device_hint
