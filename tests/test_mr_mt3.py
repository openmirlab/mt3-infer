"""
Smoke tests for the MR-MT3 adapter and registry wiring.

These do not load checkpoints or run inference (that is covered by the
manual baseline scripts used during the ADOPT campaign) -- they verify the
adapter is importable, registered correctly, and satisfies the MT3Base
interface, so a broken import or registry entry fails fast in CI.
"""

import importlib

from mt3_infer.adapters import MRMT3Adapter
from mt3_infer.adapters.mr_mt3 import MRMT3Adapter as MRMT3AdapterDirect
from mt3_infer.api import _load_registry
from mt3_infer.base import MT3Base


def test_mr_mt3_adapter_exported_from_adapters_package():
    """MRMT3Adapter must be importable from mt3_infer.adapters (public surface)."""
    assert MRMT3Adapter is MRMT3AdapterDirect


def test_mr_mt3_adapter_is_mt3base_subclass():
    assert issubclass(MRMT3Adapter, MT3Base)


def test_mr_mt3_adapter_instantiates_without_checkpoint():
    adapter = MRMT3Adapter()
    assert isinstance(adapter, MT3Base)
    assert adapter.model is None
    assert "not loaded" in repr(adapter)


def test_mr_mt3_registered_in_checkpoints_yaml():
    registry = _load_registry()
    assert "mr_mt3" in registry["models"]
    entry = registry["models"]["mr_mt3"]
    assert entry["adapter_class"] == "mt3_infer.adapters.mr_mt3.MRMT3Adapter"


def test_mr_mt3_registry_adapter_class_resolves():
    """The adapter_class dotted path in the registry must import to MRMT3Adapter."""
    registry = _load_registry()
    adapter_class_path = registry["models"]["mr_mt3"]["adapter_class"]
    module_path, class_name = adapter_class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    resolved = getattr(module, class_name)

    assert resolved is MRMT3Adapter


def test_mr_mt3_adapter_has_required_interface():
    adapter = MRMT3Adapter()
    for method in ("load_model", "preprocess", "forward", "decode", "transcribe"):
        assert hasattr(adapter, method), f"MRMT3Adapter missing {method}"


def test_mr_mt3_alias_resolves_to_fast():
    registry = _load_registry()
    assert registry["aliases"]["fast"] == "mr_mt3"
