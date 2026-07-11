"""lightning_shim.py -- inference-only replacement for pytorch_lightning.LightningModule.

YourMT3 (see ymt3.py) was originally trained as a pytorch_lightning.LightningModule.
All training hooks (training_step, validation_step, configure_optimizers, ...) were
already stripped for this inference-only build; the only two things ymt3.py still
relies on from Lightning are (1) the nn.Module contract and (2) a `.device` property
that stays correct across `.to()`/`.cuda()`/`.cpu()` calls (used ~10x in
inference_file() and inference_loader.py's `model = model.to(device)`).

Vendoring that here removes the `lightning`/`pytorch_lightning` runtime dependency
(and its torchmetrics/multiprocessing transitive baggage) entirely for inference.
The `.device` semantics below intentionally mirror
`lightning_fabric.utilities.device_dtype_mixin._DeviceDtypeModuleMixin` (Apache-2.0)
so behavior is unchanged: `_device` starts at CPU and is only updated by `.to()`,
`.cuda()`, or `.cpu()` -- it does not introspect actual parameter/buffer devices.

Read by: model/ymt3.py (YourMT3's base class).
"""
from typing import Any, Optional, Union

import torch
import torch.nn as nn


class LightningModuleShim(nn.Module):
    """Minimal drop-in base class replacing pytorch_lightning.LightningModule.

    Provides only what YourMT3 actually uses for inference: the nn.Module
    contract plus a `.device` property. Not a general-purpose Lightning
    replacement -- no `self.log`, `self.hparams`, hooks, or Trainer integration.
    """

    def __init__(self) -> None:
        super().__init__()
        self._device = torch.device("cpu")

    @property
    def device(self) -> torch.device:
        device = self._device
        if device.type == "cuda" and device.index is None:
            return torch.device(f"cuda:{torch.cuda.current_device()}")
        return device

    def to(self, *args: Any, **kwargs: Any):
        device, _dtype = torch._C._nn._parse_to(*args, **kwargs)[:2]
        if device is not None:
            self._device = device
        return super().to(*args, **kwargs)

    def cuda(self, device: Optional[Union[torch.device, int]] = None):
        if device is None:
            device = torch.device("cuda", torch.cuda.current_device())
        elif isinstance(device, int):
            device = torch.device("cuda", index=device)
        self._device = device
        return super().cuda(device=device)

    def cpu(self):
        self._device = torch.device("cpu")
        return super().cpu()
