"""
Tests for mt3_infer/config/checkpoints.yaml's structure and download provenance.

Split into two groups:
- Structural tests (no network) that always run: every registered model has
  a coherent download block, and any `sha256` recorded for a `url`-type
  download is a well-formed hex digest.
- Liveness tests (marked `network`, excluded by default -- see
  `-m "not network"` in pyproject.toml's pytest addopts) that actually hit
  the source URLs/repos to make sure they still resolve. Run explicitly with:
      pytest mt3_infer/tests/test_checkpoint_registry.py -m network
"""

import re
import urllib.request

import pytest

from mt3_infer.api import _load_registry

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _download_blocks():
    registry = _load_registry()
    return {
        name: cfg["checkpoint"]["download"]
        for name, cfg in registry["models"].items()
        if "download" in cfg["checkpoint"]
    }


def test_every_model_has_a_download_block():
    registry = _load_registry()
    for name, cfg in registry["models"].items():
        assert "download" in cfg["checkpoint"], f"{name} has no download info"


def test_url_downloads_have_well_formed_sha256_when_present():
    """If a `url`-type download records a sha256, it must be a valid hex digest.

    Not every download type carries one yet (git_lfs clones aren't verified
    this way currently -- see the ADOPT campaign notes), but any sha256 that
    *is* present must at least be well-formed so a typo doesn't silently
    disable verification (download_file() only checks it if truthy).
    """
    for name, download in _download_blocks().items():
        if download.get("source_type") == "url" and "sha256" in download:
            sha256 = download["sha256"]
            assert SHA256_RE.match(sha256), f"{name}: malformed sha256 {sha256!r}"


def test_mr_mt3_sha256_recorded():
    """mr_mt3 is the one `url`-type download -- it should carry a verified
    sha256 (recorded 2026-07-11 by downloading fresh from source_url)."""
    download = _download_blocks()["mr_mt3"]
    assert download["source_type"] == "url"
    assert SHA256_RE.match(download["sha256"])


@pytest.mark.network
@pytest.mark.parametrize("model_name", ["mr_mt3", "mt3_pytorch", "yourmt3"])
def test_checkpoint_source_is_reachable(model_name):
    """Live check that each model's download source still resolves.

    git_lfs sources are checked as the repo's HTML page (a plain HTTP HEAD
    against a raw git:// URL isn't meaningful); url sources are checked
    directly.
    """
    download = _download_blocks()[model_name]
    source_type = download["source_type"]
    url = download["source_url"]

    if source_type == "git_lfs" and url.startswith("https://github.com/"):
        check_url = url.rstrip("/")
    elif source_type == "git_lfs" and "huggingface.co" in url:
        check_url = url.rstrip("/")
    else:
        check_url = url

    req = urllib.request.Request(check_url, method="HEAD")
    with urllib.request.urlopen(req, timeout=15) as resp:
        assert resp.status < 400, f"{model_name}: {check_url} returned {resp.status}"
