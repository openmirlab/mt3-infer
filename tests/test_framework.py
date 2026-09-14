"""
Smoke tests for framework utilities.
"""

import pytest
import torch

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


def test_get_device_explicit_cuda_raises_when_unavailable(monkeypatch):
    """An explicit device="cuda" request must RAISE when CUDA is unavailable,
    not silently fall back to CPU.

    This is the regression test for the bug found in the mr_mt3 backend
    audit: get_device() used to warn-and-return "cpu" here, silently
    overriding the caller's explicit choice. The other two adapters
    (mt3_pytorch.py, yourmt3.py) already fail hard in this situation because
    they pass the device straight to torch's `.to(device)`, which raises
    naturally. get_device() must now match that behavior.
    """
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="explicitly requested"):
        get_device("cuda")


def test_get_device_explicit_cuda_indexed_raises_when_unavailable(monkeypatch):
    """Same as above, for an indexed device string like "cuda:0"."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="explicitly requested"):
        get_device("cuda:0")


def test_get_device_auto_still_falls_back_to_cpu_when_unavailable(monkeypatch):
    """"auto" (unset device) must keep falling back to CPU with a warning --
    only the explicit-request path should raise. This guards against
    over-correcting the fix above.
    """
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.warns(UserWarning):
        device = get_device("auto")

    assert device == "cpu"


def test_get_device_none_still_falls_back_to_cpu_when_unavailable(monkeypatch):
    """device_hint=None resolves to "auto" and must not raise either."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.warns(UserWarning):
        device = get_device(None)

    assert device == "cpu"


def test_get_device_explicit_cuda_indexed_accepted_when_available():
    """mt3_pytorch.py and yourmt3.py accept arbitrary device strings (e.g.
    "cuda:0") by forwarding them straight to torch without their own
    whitelist. get_device() must widen its literal-string validation to
    match instead of rejecting "cuda:0" with a ValueError.
    """
    if not torch.cuda.is_available():
        pytest.skip("Requires a CUDA-enabled environment to verify the happy path.")

    device = get_device("cuda:0")
    assert device == "cuda:0"


def test_get_device_preserves_cuda_index_without_allocating(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    assert get_device("cuda:1") == "cuda:1"


def test_get_device_rejects_unavailable_mps(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="MPS"):
        get_device("mps")


def test_get_device_rejects_invalid_cuda_index_before_availability(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="index"):
        get_device("cuda:bad")


def test_get_device_rejects_unknown_device_string():
    """Non-cuda, non-cpu, non-auto strings are still rejected outright --
    the widening is scoped to the cuda:N family only.
    """
    with pytest.raises(ValueError, match="Invalid device"):
        get_device("tpu")
