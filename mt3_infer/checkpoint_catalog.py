"""Package-owned MT3 checkpoint and profile metadata.

The catalog describes artifacts owned by mt3-infer.  Backend packages remain
free to own and resolve their own URLs; this module only exposes this
package's profile metadata and deterministic cache identity.
"""
from __future__ import annotations

from hashlib import sha256
import os
from pathlib import Path
from typing import Any, Dict, Optional

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.9/3.10
    import tomli as tomllib  # type: ignore


def checkpoint_config_path() -> Path:
    return Path(__file__).with_name("config") / "checkpoints.toml"


def load_checkpoint_config() -> Dict[str, Any]:
    with checkpoint_config_path().open("rb") as handle:
        return tomllib.load(handle)


def get_profile(model: str) -> Dict[str, Any]:
    config = load_checkpoint_config()
    profiles = config.get("profiles", {})
    aliases = config.get("aliases", {})
    canonical = aliases.get(model, model)
    if canonical not in profiles:
        raise KeyError(f"unknown MT3 profile {model!r}; available: {sorted(profiles)}")
    return {"id": canonical, **profiles[canonical]}


def cache_key(model: str, *, backend: Optional[str] = None,
              revision: Optional[str] = None, checksum: Optional[str] = None) -> str:
    profile = get_profile(model)
    material = "|".join(("mt3-infer", profile["id"], backend or profile.get("backend", ""),
                         revision or profile.get("revision", ""), checksum or profile.get("sha256", "")))
    return sha256(material.encode()).hexdigest()[:16]


def resolve_checkpoint_path(model: str, checkpoint_path: Optional[str] = None) -> Optional[Path]:
    """Resolve the session loader path without downloading or creating it."""
    raw_path = checkpoint_path or get_profile(model).get("path")
    if raw_path is None:
        return None
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    env_dir = os.environ.get("MT3_CHECKPOINT_DIR")
    if env_dir:
        base = Path(env_dir).expanduser()
        try:
            return base / path.relative_to(".mt3_checkpoints")
        except ValueError:
            return base / path
    return Path.cwd() / path


__all__ = ["checkpoint_config_path", "load_checkpoint_config", "get_profile", "cache_key", "resolve_checkpoint_path"]
