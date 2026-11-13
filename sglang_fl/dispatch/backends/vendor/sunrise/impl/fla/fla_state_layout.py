# Copyright (c) 2026 BAAI. All rights reserved.
"""Keep the GDN SSM state pool in PTPU ``[slot, HV, K, V]`` layout.

SGLang allocates the pool as ``[slot, HV, V, K]``; the PTPU kernels want
``[slot, HV, K, V]``. With ``K == V`` the buffer shape is identical, so the
pool can simply be *interpreted* in PTPU layout: the PTPU decode and chunk
kernels then update slots in place and the per-step gather / transpose /
scatter disappears. Anything that still expects SGLang layout (Triton
fallbacks) goes through :func:`run_in_sglang_layout`.

Disable with ``SGLANG_FL_SUNRISE_GDN_NATIVE_STATE=0``. Under speculative
decoding the PTPU verify path (``gdn_verify``) also uses this layout; with
``SGLANG_FL_SUNRISE_SPEC=0`` SGLang's Triton verify kernels need SGLang layout,
so the native layout is off.
"""

from __future__ import annotations

import logging
import os
from typing import Callable, Optional

import torch

from .fla_state_transpose import gather_transpose_to_scratch, transpose_scatter_to_pool

logger = logging.getLogger(__name__)

_native: Optional[bool] = None


def _decide() -> bool:
    if os.environ.get("SGLANG_FL_SUNRISE_GDN_NATIVE_STATE", "1") == "0":
        return False
    try:
        from sglang.srt.server_args import get_global_server_args

        if (
            get_global_server_args().speculative_algorithm is not None
            and os.environ.get("SGLANG_FL_SUNRISE_SPEC", "1") == "0"
        ):
            return False
    except Exception:
        pass
    logger.info("sunrise: GDN state pool kept in PTPU [slot, HV, K, V] layout")
    return True


def native_state_layout() -> bool:
    global _native
    if _native is None:
        _native = _decide()
    return _native


def set_native_state_layout(value: Optional[bool]) -> None:
    """Force the layout decision (tests); ``None`` re-reads the environment."""
    global _native
    _native = value


def check_native_pool(pool: torch.Tensor) -> None:
    assert pool.ndim == 4 and pool.shape[-1] == pool.shape[-2], (
        f"native GDN state layout needs K == V, got pool {tuple(pool.shape)}"
    )
    assert pool.dtype == torch.float32 and pool.is_contiguous(), (
        f"native GDN state layout needs a contiguous fp32 pool, got {pool.dtype}"
    )


def run_in_sglang_layout(
    pool: torch.Tensor,
    indices: Optional[torch.Tensor],
    fn: Callable[[torch.Tensor, Optional[torch.Tensor]], object],
):
    """Run ``fn(state, local_indices)`` on the selected slots in SGLang layout.

    Negative indices (padding) stay negative in ``local_indices``; their rows
    are zero on the way in and are not written back.
    """
    if indices is None:
        idx = torch.arange(pool.shape[0], dtype=torch.int32, device=pool.device)
    else:
        idx = indices.to(dtype=torch.int32).contiguous()
    scratch = torch.empty((idx.shape[0],) + tuple(pool.shape[1:]), dtype=pool.dtype, device=pool.device)
    gather_transpose_to_scratch(pool, idx, scratch)
    local = torch.arange(idx.shape[0], dtype=torch.int32, device=pool.device)
    local = torch.where(idx >= 0, local, idx)
    result = fn(scratch, local if indices is not None else None)
    transpose_scatter_to_pool(scratch, idx, pool)
    return result
