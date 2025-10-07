"""
Exception hierarchy for mt3_infer package.

All custom exceptions inherit from MT3InferError for easy catching.
"""


class MT3InferError(Exception):
    """Base exception for all mt3_infer errors."""
    pass


class ModelNotFoundError(MT3InferError):
    """Model identifier not found in registry."""
    pass


class CheckpointError(MT3InferError):
    """Checkpoint verification or loading failed."""
    pass


class CheckpointDownloadError(CheckpointError):
    """Checkpoint download failed."""
    pass


class FrameworkError(MT3InferError):
    """Framework version mismatch or import failure."""
    pass


class AudioError(MT3InferError):
    """Invalid audio input format or processing error."""
    pass


class InferenceError(MT3InferError):
    """Model inference failed."""
    pass
