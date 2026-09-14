"""Lifecycle and profile-cache contract tests for the public MT3 session."""
import numpy as np
import pytest

import mt3_infer.api as api
from mt3_infer import MT3Session


class _Adapter:
    def __init__(self):
        self.model = self
        self.cpu_calls = 0

    def transcribe(self, audio, sr=16000, **kwargs):
        return audio, sr, kwargs

    def cpu(self):
        self.cpu_calls += 1


def test_session_requires_explicit_load(monkeypatch):
    built, calls = [], []

    def load(*args, **kwargs):
        calls.append((args, kwargs))
        adapter = _Adapter()
        built.append(adapter)
        return adapter

    monkeypatch.setattr(api, "load_model", load)
    session = MT3Session("fast", auto_download=False, device="cpu")
    with pytest.raises(RuntimeError):
        session.infer(np.zeros(8, dtype=np.float32))
    session.load()
    session.load()
    assert session.status == "ready"
    assert len(built) == 1
    assert calls[0][1]["cache"] is False
    assert calls[0][1]["device"] == "cpu"
    assert session.infer(np.zeros(8, dtype=np.float32))[1] == 16000
    assert session.release() is session
    assert session.status == "released"
    assert built[0].cpu_calls == 1
    session.load()
    assert len(built) == 2
    assert session.close() is session
    assert session.close() is session
    assert session.status == "closed"
    with pytest.raises(RuntimeError, match="closed"):
        session.load()
    with pytest.raises(RuntimeError, match="ready"):
        session.infer(np.zeros(8, dtype=np.float32))


def test_profiles_have_independent_deterministic_cache_keys():
    small = MT3Session("fast")
    large = MT3Session("accurate")
    assert small.cache_info()["cache_key"] != large.cache_info()["cache_key"]
    assert small.cache_info()["model_loaded"] is False


def test_context_manager_closes(monkeypatch):
    monkeypatch.setattr(api, "load_model", lambda *a, **k: _Adapter())
    with MT3Session("fast", auto_download=False, device="cpu") as session:
        assert session.status == "ready"
    assert session.status == "closed"


def test_cache_info_uses_read_only_toml_resolver(monkeypatch, tmp_path):
    monkeypatch.setenv("MT3_CHECKPOINT_DIR", str(tmp_path / "cache"))
    session = MT3Session("fast", auto_download=False, device="cpu")
    info = session.cache_info()
    assert info["checkpoint_path"] == str(tmp_path / "cache" / "mr_mt3" / "mt3.pth")
    assert info["model"] == "mr_mt3"
    assert info["model_loaded"] is False
    assert not (tmp_path / "cache").exists()

    custom = MT3Session("fast", checkpoint_path=tmp_path / "private.pt", device="cpu")
    assert custom.cache_info()["checkpoint_path"] == str(tmp_path / "private.pt")
