"""
Smoke tests for the YourMT3 adapter and registry wiring.

Importing YourMT3Adapter itself does not require pytorch_lightning (that
import is deferred until load_model() actually loads a checkpoint via
mt3_infer.models.yourmt3.inference_loader), so these tests run in the base
dev environment without the `full` extra. End-to-end inference is covered
by the manual baseline scripts used during the ADOPT campaign, since it
needs the ~536MB checkpoint and (currently) the `full` extra installed.
"""

import importlib

from mt3_infer.adapters import YourMT3Adapter
from mt3_infer.adapters.yourmt3 import YourMT3Adapter as YourMT3AdapterDirect
from mt3_infer.api import _load_registry
from mt3_infer.base import MT3Base
from mt3_infer.exceptions import ModelNotFoundError


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
