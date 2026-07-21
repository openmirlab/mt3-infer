"""Legacy one-shot API cache contract, isolated from session-owned runtimes."""

from types import SimpleNamespace

import mt3_infer.api as api
from mt3_infer import MT3Session


class _Adapter:
    constructed = 0

    def __init__(self):
        type(self).constructed += 1

    def load_model(self, checkpoint_path, device="auto"):
        self.checkpoint_path = checkpoint_path
        self.device = device


def test_legacy_load_model_cache_remains_opt_in(monkeypatch, tmp_path):
    _Adapter.constructed = 0
    api._MODEL_CACHE.clear()
    monkeypatch.setattr(api.importlib, "import_module", lambda name: SimpleNamespace(MRMT3Adapter=_Adapter))

    checkpoint = tmp_path / "weights.pth"
    first = api.load_model("fast", checkpoint_path=str(checkpoint), device="cpu", auto_download=False)
    second = api.load_model("fast", checkpoint_path=str(checkpoint), device="cpu", auto_download=False)

    assert first is second
    assert _Adapter.constructed == 1
    api._MODEL_CACHE.clear()


def test_session_load_never_mutates_legacy_global_cache(monkeypatch, tmp_path):
    _Adapter.constructed = 0
    api._MODEL_CACHE.clear()
    monkeypatch.setattr(api.importlib, "import_module", lambda name: SimpleNamespace(MRMT3Adapter=_Adapter))

    session = MT3Session(
        "fast",
        checkpoint_path=tmp_path / "weights.pth",
        device="cpu",
        auto_download=False,
    )
    session.load()

    assert _Adapter.constructed == 1
    assert api._MODEL_CACHE == {}
    session.release()
