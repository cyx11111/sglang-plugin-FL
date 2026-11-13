# Copyright (c) 2026 BAAI. All rights reserved.
"""``sigmoid(x @ w.T)`` for a ``[1, H]`` gate (Qwen2 / Qwen3.5 MoE ``shared_expert_gate``).

The bf16 N=1 GEMM plus sigmoid takes ~0.11 ms at 8192 x 2048 on PTPU, while
reading ``x`` takes ~0.02 ms. A program reduces ``ROWS`` rows (4 warps per
row, as in ``_rmsnorm_triton``), rounds each logit to the output dtype like
the GEMM output, applies sigmoid in fp32 and stores the ``ROWS`` gates as one
vector. Used for prefill-sized inputs only.
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

_ROWS = int(os.environ.get("SGLANG_FL_SUNRISE_SHARED_GATE_ROWS", "16"))
MIN_ROWS = int(os.environ.get("SGLANG_FL_SUNRISE_SHARED_GATE_MIN_ROWS", "1024"))


@triton.jit
def _sigmoid_row_dot_kernel(x_ptr, w_ptr, o_ptr, n_rows, stride_x, H: tl.constexpr, ROWS: tl.constexpr):
    offs = tl.arange(0, H)
    w = tl.load(w_ptr + offs).to(tl.float32)
    r0 = tl.program_id(0) * ROWS
    ridx = tl.arange(0, ROWS)
    res = tl.zeros([ROWS], dtype=tl.float32)
    x_row = x_ptr + r0.to(tl.int64) * stride_x + offs
    for i in tl.static_range(ROWS):
        x = tl.load(x_row, mask=(offs < H) & (r0 + i < n_rows), other=0.0).to(tl.float32)
        res = tl.where(ridx == i, tl.sum(x * w, axis=0), res)
        x_row += stride_x
    logit = res.to(o_ptr.dtype.element_ty).to(tl.float32)
    tl.store(o_ptr + r0 + ridx, tl.sigmoid(logit).to(o_ptr.dtype.element_ty), mask=r0 + ridx < n_rows)


def supported(x: torch.Tensor, weight: torch.Tensor) -> bool:
    if x.dim() != 2 or weight.dim() != 2:
        return False
    h = x.shape[1]
    return (
        x.device.type == "ptpu"
        and x.shape[0] >= MIN_ROWS and x.stride(1) == 1
        and weight.shape == (1, h) and weight.is_contiguous() and weight.dtype == x.dtype
        and h & (h - 1) == 0 and 1024 <= h <= 4096
    )


def sigmoid_row_dot(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """``torch.sigmoid(F.linear(x, weight))`` for ``weight`` of shape ``[1, H]``; returns ``[M, 1]``."""
    m, h = x.shape
    out = torch.empty((m, 1), dtype=x.dtype, device=x.device)
    if m:
        _sigmoid_row_dot_kernel[(triton.cdiv(m, _ROWS),)](
            x, weight, out, m, x.stride(0), H=h, ROWS=_ROWS, num_warps=4,
        )
    return out
