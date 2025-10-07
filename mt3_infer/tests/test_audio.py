"""
Smoke tests for audio utilities.
"""

import numpy as np
import pytest

from mt3_infer.exceptions import AudioError
from mt3_infer.utils.audio import normalize_audio, validate_audio


def test_validate_audio_success():
    """Test validate_audio with valid input."""
    audio = np.random.randn(16000).astype(np.float32)
    is_valid, msg = validate_audio(audio)

    assert is_valid is True
    assert msg == ""


def test_validate_audio_rejects_non_numpy():
    """Test validate_audio rejects non-numpy arrays."""
    is_valid, msg = validate_audio([1, 2, 3])

    assert is_valid is False
    assert "numpy.ndarray" in msg


def test_validate_audio_rejects_invalid_dimensions():
    """Test validate_audio rejects 3D arrays."""
    audio_3d = np.random.randn(10, 10, 10).astype(np.float32)
    is_valid, msg = validate_audio(audio_3d)

    assert is_valid is False
    assert "1D or 2D" in msg


def test_validate_audio_rejects_invalid_dtype():
    """Test validate_audio rejects int arrays."""
    audio_int = np.array([1, 2, 3], dtype=np.int16)
    is_valid, msg = validate_audio(audio_int)

    assert is_valid is False
    assert "float32 or float64" in msg


def test_validate_audio_rejects_empty():
    """Test validate_audio rejects empty arrays."""
    audio_empty = np.array([], dtype=np.float32)
    is_valid, msg = validate_audio(audio_empty)

    assert is_valid is False
    assert "empty" in msg


def test_validate_audio_warns_on_large_values():
    """Test validate_audio rejects suspiciously large values."""
    audio_huge = np.array([100.0, 200.0], dtype=np.float32)
    is_valid, msg = validate_audio(audio_huge)

    assert is_valid is False
    assert "suspiciously large" in msg


def test_normalize_audio_success():
    """Test normalize_audio normalizes to [-1, 1]."""
    audio = np.array([0.5, 1.0, -2.0], dtype=np.float32)
    normalized = normalize_audio(audio)

    assert np.allclose(normalized, [0.25, 0.5, -1.0])
    assert np.abs(normalized).max() <= 1.0


def test_normalize_audio_handles_zero():
    """Test normalize_audio with all-zero audio."""
    audio_zero = np.zeros(1000, dtype=np.float32)
    normalized = normalize_audio(audio_zero)

    assert np.allclose(normalized, audio_zero)


def test_normalize_audio_handles_empty():
    """Test normalize_audio with empty array."""
    audio_empty = np.array([], dtype=np.float32)
    normalized = normalize_audio(audio_empty)

    assert normalized.size == 0


def test_normalize_audio_preserves_dtype():
    """Test normalize_audio preserves dtype."""
    audio_f64 = np.array([0.5, 1.0, -2.0], dtype=np.float64)
    normalized = normalize_audio(audio_f64)

    assert normalized.dtype == np.float64
