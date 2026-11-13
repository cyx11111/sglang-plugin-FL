# Copyright (c) 2026 BAAI. All rights reserved.
"""Patch SGLang FLA ``layernorm_gated`` for PTPU / Sunrise.

Upstream ``_get_sm_count`` calls ``torch.cuda.get_device_properties``, which
raises on a CUDA-free PTPU build after GDN ``chunk_gated_delta_rule`` returns.

Prefill-sized ``norm(x) * silu(z)`` uses the fused Triton row pass
(``impl/fla_rms_norm_gated.py``; disable with
``SGLANG_FL_SUNRISE_GDN_NORM_FUSED=0``).
"""

from __future__ import annotations

import logging
import os
import types
from typing import Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_patched = False


def patch_layernorm_gated_for_sunrise() -> None:
    """Replace ``rms_norm_gated`` with a PTPU-safe path. Idempotent."""
    global _patched
    if _patched:
        return

    try:
        if not (
            hasattr(torch, "ptpu")
            and torch.ptpu.is_available()
            and torch.ptpu.device_count() > 0
        ):
            return

        import sglang.srt.layers.attention.fla.layernorm_gated as ln_mod
        from sglang_fl.dispatch import call_op

        from ...impl.fla import fla_rms_norm_gated as rng

        _fused_ok = os.environ.get("SGLANG_FL_SUNRISE_GDN_NORM_FUSED", "1") != "0"

        def _rms_norm_gated_sunrise(
            *,
            x: torch.Tensor,
            weight: torch.Tensor,
            bias: Optional[torch.Tensor],
            z: Optional[torch.Tensor] = None,
            eps: float = 1e-6,
            group_size=None,
            norm_before_gate: bool = True,
            is_rms_norm: bool = False,
            activation: str = "swish",
        ):
            # Qwen3.5 GDN path: norm(x) * silu(z)
            fast = (
                z is not None
                and norm_before_gate
                and group_size is None
                and bias is None
                and is_rms_norm
                and activation in ("silu", "swish")
            )
            if fast and _fused_ok and rng.supported(x, z, weight):
                return rng.rms_norm_gated_fused(x, z, weight, eps)
            if fast:
                proxy = types.SimpleNamespace(
                    weight=weight,
                    variance_epsilon=eps,
                )
                out = call_op("rms_norm", proxy, x, None)
                return out * F.silu(z)

            # Other configs: pure PyTorch reference (no CUDA SM query).
            return ln_mod.rms_norm_ref(
                x,
                weight,
                bias,
                z=z,
                eps=eps,
                group_size=group_size,
                norm_before_gate=norm_before_gate,
                upcast=True,
            )

        ln_mod.rms_norm_gated = _rms_norm_gated_sunrise

        # Belt-and-suspenders: any leftover Triton path that still queries SMs.
        def _get_sm_count_ptpu(device: torch.device) -> int:
            if getattr(device, "type", None) in ("ptpu", "PrivateUse1"):
                return 64
            if device.type == "xpu" and torch.xpu.is_available():
                return torch.xpu.get_device_properties(device).gpu_subslice_count
            if torch.cuda.is_available():
                return torch.cuda.get_device_properties(device).multi_processor_count
            return 64

        ln_mod._get_sm_count = _get_sm_count_ptpu

        _patched = True
        logger.info(
            "sunrise: patched fla.layernorm_gated.rms_norm_gated for PTPU "
            "(avoid torch.cuda.get_device_properties)"
        )
    except Exception as exc:
        logger.warning("sunrise: failed to patch layernorm_gated for PTPU: %s", exc)
