"""
Smoke tests for the MT3-PyTorch adapter and registry wiring.

These do not load checkpoints or run inference (that is covered by the
manual baseline scripts used during the ADOPT campaign) -- they verify the
adapter is importable, registered correctly, and satisfies the MT3Base
interface, so a broken import or registry entry fails fast in CI.
"""

import importlib

from mt3_infer.adapters import MT3PyTorchAdapter
from mt3_infer.adapters.mt3_pytorch import MT3PyTorchAdapter as MT3PyTorchAdapterDirect
from mt3_infer.api import _load_registry
from mt3_infer.base import MT3Base


def test_mt3_pytorch_adapter_exported_from_adapters_package():
    """MT3PyTorchAdapter must be importable from mt3_infer.adapters (public surface).

    It was previously only importable from mt3_infer.adapters.mt3_pytorch --
    the only one of the three adapters missing from the package's __all__.
    """
    assert MT3PyTorchAdapter is MT3PyTorchAdapterDirect


def test_mt3_pytorch_adapter_is_mt3base_subclass():
    assert issubclass(MT3PyTorchAdapter, MT3Base)


def test_mt3_pytorch_adapter_instantiates_without_checkpoint():
    adapter = MT3PyTorchAdapter()
    assert isinstance(adapter, MT3Base)
    assert "not loaded" in repr(adapter)


def test_mt3_pytorch_adapter_auto_filter_default_true():
    adapter = MT3PyTorchAdapter()
    assert adapter.auto_filter is True

    adapter_no_filter = MT3PyTorchAdapter(auto_filter=False)
    assert adapter_no_filter.auto_filter is False


def test_mt3_pytorch_registered_in_checkpoints_yaml():
    registry = _load_registry()
    assert "mt3_pytorch" in registry["models"]
    entry = registry["models"]["mt3_pytorch"]
    assert entry["adapter_class"] == "mt3_infer.adapters.mt3_pytorch.MT3PyTorchAdapter"


def test_mt3_pytorch_registry_adapter_class_resolves():
    """The adapter_class dotted path in the registry must import to MT3PyTorchAdapter."""
    registry = _load_registry()
    adapter_class_path = registry["models"]["mt3_pytorch"]["adapter_class"]
    module_path, class_name = adapter_class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    resolved = getattr(module, class_name)

    assert resolved is MT3PyTorchAdapter


def test_mt3_pytorch_adapter_has_required_interface():
    adapter = MT3PyTorchAdapter()
    for method in ("load_model", "preprocess", "forward", "decode", "transcribe"):
        assert hasattr(adapter, method), f"MT3PyTorchAdapter missing {method}"


def test_mt3_pytorch_is_default_model():
    registry = _load_registry()
    assert registry["default"] == "mt3_pytorch"
    assert registry["aliases"]["default"] == "mt3_pytorch"
