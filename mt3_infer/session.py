"""Composite lifecycle session for MT3 model profiles.

``MT3Session`` owns orchestration and state while existing adapters own model
details.  Only the selected profile is resident in memory; other profiles may
remain in the disk checkpoint cache.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from .checkpoint_catalog import cache_key, get_profile


class MT3Session:
    """Explicit load/infer/release lifecycle for one MT3 profile."""

    def __init__(self, model: str = "default", *, backend: Optional[str] = None,
                 device: str = "auto", checkpoint_path: Optional[str] = None,
                 auto_download: bool = True, **model_kwargs: Any) -> None:
        profile = get_profile(model)
        self.model = profile["id"]
        self.backend = backend or profile.get("backend", "pytorch")
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.auto_download = auto_download
        self.model_kwargs = model_kwargs
        self._adapter = None
        self._status = "new"
        self._component_status: Dict[str, str] = {self.model: "new"}

    @property
    def status(self) -> str:
        return self._status

    @property
    def component_status(self) -> Dict[str, str]:
        return dict(self._component_status)

    def load(self) -> "MT3Session":
        if self._status == "ready":
            return self
        self._status = "loading"
        self._component_status[self.model] = "loading"
        try:
            from .api import load_model
            self._adapter = load_model(
                self.model, checkpoint_path=self.checkpoint_path,
                device=self.device, auto_download=self.auto_download,
                **self.model_kwargs,
            )
            self._status = "ready"
            self._component_status[self.model] = "ready"
            return self
        except Exception:
            self._status = "failed"
            self._component_status[self.model] = "failed"
            raise

    def infer(self, audio, *, sr: int = 16000, **kwargs):
        if self._status != "ready" or self._adapter is None:
            raise RuntimeError("MT3Session must be ready; call load() before infer()")
        return self._adapter.transcribe(audio, sr=sr, **kwargs)

    def release(self) -> None:
        adapter = self._adapter
        if adapter is not None:
            model_obj = getattr(adapter, "model", None)
            if model_obj is not None and hasattr(model_obj, "cpu"):
                model_obj.cpu()
            # Adapters differ in their private storage; dropping the reference
            # is the common release boundary and keeps disk cache untouched.
            self._adapter = None
        self._status = "released"
        self._component_status[self.model] = "released"

    def close(self) -> None:
        self.release()

    def cache_info(self) -> Dict[str, Any]:
        profile = get_profile(self.model)
        return {
            "package": "mt3-infer", "model": self.model, "backend": self.backend,
            "cache_key": cache_key(self.model, backend=self.backend),
            "checkpoint_path": self.checkpoint_path or profile.get("path"),
            "checkpoint_url": profile.get("url"), "sha256": profile.get("sha256"),
            "status": self._status, "model_loaded": self._adapter is not None,
            "components": self.component_status,
        }

    def __enter__(self) -> "MT3Session":
        return self.load()

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.close()
        return False


__all__ = ["MT3Session"]
