# Copyright (c) 2026 BAAI. All rights reserved.
"""Clearing mamba pool slots on PTPU.

``MambaPool.alloc`` zeroes a new slot with ``state[:, idx] = 0`` (index_put
along dim 1). On PTPU that runs at ~0.5 GB/s: 134 ms of device time per
request for the Qwen3.6 TP1 conv + SSM state (61 MB), i.e. ~540 ms per
4-request prefill batch, which the next schedule-stream copy waits for.
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

_BLOCK = int(os.environ.get("SGLANG_FL_SUNRISE_MAMBA_ZERO_BLOCK", "2048"))
_ITERS = int(os.environ.get("SGLANG_FL_SUNRISE_MAMBA_ZERO_ITERS", "16"))
_WARPS = int(os.environ.get("SGLANG_FL_SUNRISE_MAMBA_ZERO_WARPS", "4"))


@triton.jit
def _zero_rows_kernel(
    dst, rows, n_rows, row_elems, layer_stride, BLOCK: tl.constexpr, ITERS: tl.constexpr,
):
    i = tl.program_id(0)
    layer = tl.program_id(1).to(tl.int64)
    base = tl.program_id(2).to(tl.int64) * BLOCK * ITERS
    r = tl.load(rows + i).to(tl.int64)
    if (r >= 0) & (r < n_rows):
        d = dst + layer * layer_stride + r * row_elems + base
        offs = tl.arange(0, BLOCK)
        zero = tl.zeros([BLOCK], dtype=dst.dtype.element_ty)
        for it in tl.static_range(ITERS):
            o = it * BLOCK + offs
            tl.store(d + o, zero, mask=base + o < row_elems)


def zero_slots_supported(t: torch.Tensor, idx: torch.Tensor) -> bool:
    return (
        t.device.type == "ptpu"
        and t.dim() >= 2
        and t.is_contiguous()
        and isinstance(idx, torch.Tensor)
        and idx.device == t.device
        and idx.dim() == 1
        and idx.dtype in (torch.int32, torch.int64)
    )


def zero_slots(t: torch.Tensor, idx: torch.Tensor) -> None:
    """``t[:, idx] = 0`` for a contiguous ``[layers, slots, ...]`` state buffer."""
    n = idx.shape[0]
    if n == 0 or t.numel() == 0:
        return
    layers, slots = t.shape[0], t.shape[1]
    row_elems = t.numel() // (layers * slots)
    _zero_rows_kernel[(n, layers, triton.cdiv(row_elems, _BLOCK * _ITERS))](
        t, idx, slots, row_elems, slots * row_elems,
        BLOCK=_BLOCK, ITERS=_ITERS, num_warps=_WARPS,
    )
