# Copyright (c) 2026 BAAI. All rights reserved.
"""GDN output norm ``rms_norm(x) * w * silu(z)`` in one row-wise pass.

The previous sunrise path ran the vendor ``rms_norm``, ``F.silu(z)`` and a
multiply: three passes over ``[T * HV, 128]``. Here each 128-element row 
is normalised and gated in fp32 with a single bf16 rounding, as in FLA's 
``rms_norm_gated`` kernel (closer to an fp32 reference than the 3-op path). 
A program walks ``rows_per_prog`` rows  (1-D vectors, 1 warp). 
Used for prefill-sized inputs only

``rms_norm_gated_heads`` takes ``z`` as the ``[T, heads * D]`` column slice of
the qkvz projection (row stride = projection width): row ``t * heads + h``
of ``x`` is gated by ``z[t, h * D:(h + 1) * D]``, so the ``z`` copy before the
norm is not needed (copy + norm 0.42 -> 0.16 ms per layer at TP1 / 8192 tokens,
0.13 -> 0.045 ms at TP4).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# 8 rows per program: 1 row is ~3x slower, 64 rows ~1.6x slower (PTPU).
_ROWS_PER_PROG = 8
# Below this (decode) the 3-op vendor path is as fast in eager and graphs.
MIN_ROWS = 8192


@triton.jit
def _rms_norm_gated_kernel(
    x_ptr, z_ptr, w_ptr, o_ptr, n_rows, rows_per_prog,
    stride_x, stride_z, stride_o, eps,
    D: tl.constexpr,
):
    offs = tl.arange(0, D)
    w = tl.load(w_ptr + offs).to(tl.float32)
    r0 = tl.program_id(0) * rows_per_prog
    r_end = tl.minimum(r0 + rows_per_prog, n_rows)
    x_row = x_ptr + r0.to(tl.int64) * stride_x + offs
    z_row = z_ptr + r0.to(tl.int64) * stride_z + offs
    o_row = o_ptr + r0.to(tl.int64) * stride_o + offs
    for _ in range(r0, r_end):
        x = tl.load(x_row).to(tl.float32)
        z = tl.load(z_row).to(tl.float32)
        rstd = 1 / tl.sqrt(tl.sum(x * x, axis=0) / D + eps)
        y = x * rstd * w
        y = y * z * tl.sigmoid(z)
        tl.store(o_row, y.to(o_ptr.dtype.element_ty))
        x_row += stride_x
        z_row += stride_z
        o_row += stride_o


@triton.jit
def _rms_norm_gated_heads_kernel(
    x_ptr, z_ptr, w_ptr, o_ptr, stride_zt, eps,
    HEADS: tl.constexpr, ROWS: tl.constexpr, D: tl.constexpr,
):
    # Consecutive programs walk the head blocks of one token, so x / z / out are
    # accessed in address order (a token-major 2-D grid: 0.377 vs 0.163 ms at TP1).
    pid = tl.program_id(0)
    t = pid // (HEADS // ROWS)
    h0 = (pid % (HEADS // ROWS)) * ROWS
    offs = tl.arange(0, D)
    w = tl.load(w_ptr + offs).to(tl.float32)
    r0 = t.to(tl.int64) * HEADS + h0
    x_row = x_ptr + r0 * D + offs
    o_row = o_ptr + r0 * D + offs
    z_row = z_ptr + t.to(tl.int64) * stride_zt + h0 * D + offs
    for _ in range(ROWS):
        x = tl.load(x_row).to(tl.float32)
        z = tl.load(z_row).to(tl.float32)
        rstd = 1 / tl.sqrt(tl.sum(x * x, axis=0) / D + eps)
        y = x * rstd * w
        y = y * z * tl.sigmoid(z)
        tl.store(o_row, y.to(o_ptr.dtype.element_ty))
        x_row += D
        z_row += D
        o_row += D


def supported(x: torch.Tensor, z: torch.Tensor, weight: torch.Tensor) -> bool:
    d = x.shape[-1]
    return (
        x.device.type == "ptpu"
        and x.dim() == 2 and z.shape == x.shape and x.shape[0] >= MIN_ROWS
        and x.stride(1) == 1 and z.stride(1) == 1
        and weight.shape == (d,) and weight.is_contiguous()
        and d & (d - 1) == 0 and d <= 1024
    )


def rms_norm_gated_fused(x: torch.Tensor, z: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    n, d = x.shape
    out = torch.empty((n, d), dtype=x.dtype, device=x.device)
    if n == 0:
        return out
    rows = _ROWS_PER_PROG
    _rms_norm_gated_kernel[(triton.cdiv(n, rows),)](
        x, z, weight, out, n, rows, x.stride(0), z.stride(0), out.stride(0), eps,
        D=d, num_warps=1,
    )
    return out


def heads_supported(x: torch.Tensor, z: torch.Tensor, weight: torch.Tensor, heads: int) -> bool:
    """``x``: contiguous ``[T * heads, D]``; ``z``: ``[T, heads * D]`` with unit inner stride."""
    if x.dim() != 2 or z.dim() != 2 or heads <= 0:
        return False
    d = x.shape[1]
    return (
        x.device.type == "ptpu"
        and x.shape[0] >= MIN_ROWS and x.is_contiguous()
        and x.shape[0] == z.shape[0] * heads and z.shape[1] == heads * d
        and z.stride(1) == 1 and z.dtype == x.dtype
        and (heads % _ROWS_PER_PROG == 0 or _ROWS_PER_PROG % heads == 0)
        and weight.shape == (d,) and weight.is_contiguous()
        and d & (d - 1) == 0 and d <= 1024
    )


def rms_norm_gated_heads(
    x: torch.Tensor, z: torch.Tensor, weight: torch.Tensor, eps: float, heads: int
) -> torch.Tensor:
    out = torch.empty_like(x)
    t = z.shape[0]
    if t == 0:
        return out
    rows = min(_ROWS_PER_PROG, heads)
    _rms_norm_gated_heads_kernel[(t * (heads // rows),)](
        x, z, weight, out, z.stride(0), eps,
        HEADS=heads, ROWS=rows, D=x.shape[1], num_warps=1,
    )
    return out
