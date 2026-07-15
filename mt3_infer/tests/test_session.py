"""Lifecycle and profile-cache contract tests for the public MT3 session."""
import numpy as np
import pytest

import mt3_infer.api as api
from mt3_infer import MT3Session


class _Adapter:
    def transcribe(self, audio, sr=16000, **kwargs):
        return audio, sr, kwargs

    def cpu(self):
        pass


def test_session_requires_explicit_load(monkeypatch):
    monkeypatch.setattr(api, "load_model", lambda *a, **k: _Adapter())
    session = MT3Session("fast", auto_download=False)
    with pytest.raises(RuntimeError):
        session.infer(np.zeros(8, dtype=np.float32))
    session.load()
    assert session.status == "ready"
    assert session.infer(np.zeros(8, dtype=np.float32))[1] == 16000
    session.release()
    assert session.status == "released"


def test_profiles_have_independent_deterministic_cache_keys():
    small = MT3Session("fast")
    large = MT3Session("accurate")
    assert small.cache_info()["cache_key"] != large.cache_info()["cache_key"]
    assert small.cache_info()["model_loaded"] is False


def test_context_manager_closes(monkeypatch):
    monkeypatch.setattr(api, "load_model", lambda *a, **k: _Adapter())
    with MT3Session("fast", auto_download=False) as session:
        assert session.status == "ready"
    assert session.status == "released"
