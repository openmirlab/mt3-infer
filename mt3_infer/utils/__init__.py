"""Utility modules for MT3-Infer."""

from mt3_infer.utils.audio import (
    load_audio,
    normalize_audio,
    resample_audio,
    validate_audio,
)
from mt3_infer.utils.framework import (
    check_jax_version,
    check_tensorflow_version,
    check_torch_version,
    get_device,
)
from mt3_infer.utils.midi import (
    count_notes,
    get_midi_duration,
    midi_to_hash,
    validate_midi,
)

__all__ = [
    # Audio utilities
    "load_audio",
    "validate_audio",
    "normalize_audio",
    "resample_audio",
    # Framework utilities
    "check_torch_version",
    "check_tensorflow_version",
    "check_jax_version",
    "get_device",
    # MIDI utilities
    "midi_to_hash",
    "validate_midi",
    "count_notes",
    "get_midi_duration",
]
