"""
Smoke tests for the YourMT3 adapter and registry wiring.

YourMT3 no longer depends on pytorch_lightning/lightning at all (see
model/lightning_shim.py) -- these tests run in the base dev environment
without the `full` extra. End-to-end inference is covered by the manual
baseline scripts used during the ADOPT campaign, since it needs the ~536MB
checkpoint.
"""

import importlib
import sys

import torch

from mt3_infer.adapters import YourMT3Adapter
from mt3_infer.adapters.yourmt3 import YourMT3Adapter as YourMT3AdapterDirect
from mt3_infer.api import _load_registry
from mt3_infer.base import MT3Base
from mt3_infer.exceptions import ModelNotFoundError
from mt3_infer.models.yourmt3.model.lightning_shim import LightningModuleShim
from mt3_infer.models.yourmt3.inference_loader import _checkpoint_unpickling_module_alias


def test_yourmt3_adapter_exported_from_adapters_package():
    assert YourMT3Adapter is YourMT3AdapterDirect


def test_yourmt3_adapter_is_mt3base_subclass():
    assert issubclass(YourMT3Adapter, MT3Base)


def test_yourmt3_adapter_instantiates_without_checkpoint():
    adapter = YourMT3Adapter()
    assert isinstance(adapter, MT3Base)
    assert adapter.model is None
    assert "not loaded" in repr(adapter)


def test_yourmt3_default_model_key_is_moe_nops():
    adapter = YourMT3Adapter()
    assert adapter.model_key == "yptf_moe_nops"


def test_yourmt3_rejects_unknown_model_key():
    try:
        YourMT3Adapter(model_key="not-a-real-key")
        assert False, "expected ModelNotFoundError"
    except ModelNotFoundError:
        pass


def test_yourmt3_lists_all_checkpoint_variants():
    available = YourMT3Adapter.list_available_models()
    for key in ("ymt3plus", "yptf_single", "yptf_multi", "yptf_moe_nops", "yptf_moe_ps"):
        assert key in available


def test_yourmt3_registered_in_checkpoints_yaml():
    registry = _load_registry()
    assert "yourmt3" in registry["models"]
    entry = registry["models"]["yourmt3"]
    assert entry["adapter_class"] == "mt3_infer.adapters.yourmt3.YourMT3Adapter"


def test_yourmt3_registry_adapter_class_resolves():
    registry = _load_registry()
    adapter_class_path = registry["models"]["yourmt3"]["adapter_class"]
    module_path, class_name = adapter_class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    resolved = getattr(module, class_name)

    assert resolved is YourMT3Adapter


def test_yourmt3_adapter_has_required_interface():
    adapter = YourMT3Adapter()
    for method in ("load_model", "preprocess", "forward", "decode", "transcribe"):
        assert hasattr(adapter, method), f"YourMT3Adapter missing {method}"


def test_yourmt3_alias_resolves_to_multitask():
    registry = _load_registry()
    assert registry["aliases"]["multitask"] == "yourmt3"


def test_yourmt3_does_not_import_pytorch_lightning():
    """YourMT3 must not need pytorch_lightning/lightning at all anymore."""
    assert "pytorch_lightning" not in sys.modules
    assert "lightning" not in sys.modules


class TestLightningModuleShim:
    """LightningModuleShim replaces pl.LightningModule as YourMT3's base class.

    Device semantics intentionally mirror
    lightning_fabric.utilities.device_dtype_mixin._DeviceDtypeModuleMixin:
    `.device` starts at CPU and only changes via `.to()`/`.cuda()`/`.cpu()`.
    """

    def test_device_defaults_to_cpu(self):
        shim = LightningModuleShim()
        assert shim.device == torch.device("cpu")

    def test_device_updates_after_to(self):
        shim = LightningModuleShim()
        shim.to("cpu")  # no-op move, but exercises the .to() override
        assert shim.device == torch.device("cpu")

    def test_is_nn_module(self):
        import torch.nn as nn

        assert isinstance(LightningModuleShim(), nn.Module)


def test_checkpoint_module_alias_is_scoped_and_restored():
    """The 'utils' sys.modules alias used for unpickling old checkpoints must
    not leak into the rest of the process -- verifies the P3 sys.modules
    surgery (previously a permanent, process-wide hijack) actually scopes
    cleanly and restores whatever was there before (including nothing)."""
    assert "utils" not in sys.modules

    with _checkpoint_unpickling_module_alias():
        assert "utils" in sys.modules
        import mt3_infer.models.yourmt3.utils as expected

        assert sys.modules["utils"] is expected

    assert "utils" not in sys.modules


def test_checkpoint_module_alias_restores_previous_module():
    """If some unrelated top-level `utils` module already existed, it must be
    restored afterward rather than deleted."""
    sentinel = object()
    sys.modules["utils"] = sentinel
    try:
        with _checkpoint_unpickling_module_alias():
            assert sys.modules["utils"] is not sentinel
        assert sys.modules["utils"] is sentinel
    finally:
        sys.modules.pop("utils", None)
