# Copyright (c) 2026 BAAI. All rights reserved.
"""Skip NVIDIA ``sgl_kernel`` CUDA ops on PTPU.

SGLang's HTTP serve path imports ``sglang.srt.mem_cache.memory_pool_host``,
which does::

    if not (_is_npu or _is_xpu or _is_mps):
        from sgl_kernel.kvcacheio import ...

``sgl_kernel`` then loads architecture-specific ``common_ops*.so`` (sm90/sm100).
That library is CUDA-only and fails on Sunrise / PTPU::

    ImportError: [sgl_kernel] CRITICAL: Could not load any common_ops library!

This patch runs at plugin load (before ``http_server`` / ``Engine`` import) and
pre-registers a stub ``sgl_kernel`` package so the NVIDIA extension is never
dlopen'd. Real fused ops on PTPU come from ``torch_ptpu.sgl_kernel``.
"""

from __future__ import annotations

import logging
import sys
import types
from typing import Any, Callable

import torch

logger = logging.getLogger(__name__)

_patched = False

_KVCACHEIO_SYMBOLS = (
    "copy_all_layer_kv_cache_cpu",
    "transfer_kv_all_layer",
    "transfer_kv_all_layer_direct_lf_pf",
    "transfer_kv_all_layer_lf_pf",
    "transfer_kv_all_layer_lf_ph",
    "transfer_kv_all_layer_mla",
    "transfer_kv_all_layer_mla_lf_pf",
    "transfer_kv_direct",
    "transfer_kv_per_layer",
    "transfer_kv_per_layer_direct_pf_lf",
    "transfer_kv_per_layer_mla",
    "transfer_kv_per_layer_mla_pf_lf",
    "transfer_kv_per_layer_pf_lf",
    "transfer_kv_per_layer_ph_lf",
)


def _on_ptpu() -> bool:
    try:
        return hasattr(torch, "ptpu") and bool(torch.ptpu.is_available())
    except Exception:
        return False


def _unavailable(qualname: str) -> Callable[..., Any]:
    def _fn(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError(
            f"{qualname} is NVIDIA sgl_kernel-only and is not available on PTPU. "
            "Sunrise fused ops live in torch_ptpu.sgl_kernel."
        )

    _fn.__name__ = qualname.rsplit(".", 1)[-1]
    _fn.__qualname__ = qualname
    return _fn


class _StubModule(types.ModuleType):
    """Package/module stub that yields dummy callables for missing attrs."""

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        dummy = _unavailable(f"{self.__name__}.{name}")
        setattr(self, name, dummy)
        return dummy


def patch_sgl_kernel_stub_for_sunrise() -> None:
    """Pre-register a PTPU stub for ``sgl_kernel``. Idempotent."""
    global _patched
    if _patched:
        return
    if not _on_ptpu():
        return

    existing = sys.modules.get("sgl_kernel")
    if existing is not None and getattr(existing, "__file__", None) not in (
        None,
        "<sglang_fl ptpu stub>",
    ):
        # Real NVIDIA sgl_kernel already imported; do not overwrite.
        logger.warning(
            "sgl_kernel already imported from %s; PTPU stub not applied",
            getattr(existing, "__file__", "?"),
        )
        _patched = True
        return

    pkg = _StubModule("sgl_kernel")
    pkg.__file__ = "<sglang_fl ptpu stub>"
    pkg.__package__ = "sgl_kernel"
    pkg.__path__ = []  # mark as package; no real submodule search
    pkg.__version__ = "0.0.0+ptpu-stub"

    kv = _StubModule("sgl_kernel.kvcacheio")
    kv.__file__ = "<sglang_fl ptpu stub>"
    kv.__package__ = "sgl_kernel"
    for name in _KVCACHEIO_SYMBOLS:
        setattr(kv, name, _unavailable(f"sgl_kernel.kvcacheio.{name}"))

    sys.modules["sgl_kernel"] = pkg
    sys.modules["sgl_kernel.kvcacheio"] = kv
    pkg.kvcacheio = kv

    logger.info(
        "PTPU: stubbed NVIDIA sgl_kernel (skip sm90/sm100 common_ops load)"
    )
    _patched = True
