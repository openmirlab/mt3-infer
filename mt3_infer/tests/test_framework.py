"""
Smoke tests for framework utilities.
"""

import pytest

from mt3_infer.utils.framework import get_device


def test_get_device_cpu():
    """Test get_device with explicit CPU."""
    device = get_device("cpu")
    assert device == "cpu"


def test_get_device_auto():
    """Test get_device with auto detection."""
    device = get_device("auto")
    assert device in ("cuda", "cpu")


def test_get_device_none_defaults_to_auto():
    """Test get_device with None defaults to auto."""
    device = get_device(None)
    assert device in ("cuda", "cpu")


def test_get_device_invalid_raises():
    """Test get_device raises on invalid device."""
    with pytest.raises(ValueError, match="Invalid device"):
        get_device("tpu")


def test_get_device_case_insensitive():
    """Test get_device is case insensitive."""
    device_upper = get_device("CPU")
    device_lower = get_device("cpu")
    assert device_upper == device_lower == "cpu"
