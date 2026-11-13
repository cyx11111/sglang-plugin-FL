# Copyright (c) 2026 BAAI. All rights reserved.

"""Triton MoE glue for the PTPU block-INT8 MoE.

* ``gather_rows``: after ``moe_sort``, one pass that gathers each
  expert-sorted entry's INT8 row + scales and writes the inverse permutation.
  Replaces ``moe_expand_permute`` + a separate scale ``index_select``.
* ``weighted_reduce``: ``out[t] = sum_k w[t, k] * y[inv[t * top_k + k]]``.
  Replaces ``moe_reduce``.

Row indices are clamped to their buffers. ``weighted_reduce`` also serves
expert-parallel MoE for any hidden size with a power-of-two factor >= 32.
"""

from __future__ import annotations

import triton
import triton.language as tl
import torch


@triton.jit
def _gather_rows_kernel(
    permuted_rows, xq, xs, oq, os_, inv, N, M,
    TOPK: tl.constexpr, H: tl.constexpr, BH: tl.constexpr, G: tl.constexpr, GP: tl.constexpr,
):
    j = tl.program_id(0)
    flat = tl.load(permuted_rows + j)
    flat = tl.minimum(tl.maximum(flat, 0), N - 1)
    tok = tl.minimum(flat // TOPK, M - 1)
    for b in tl.static_range(H // BH):
        offs = b * BH + tl.arange(0, BH)
        tl.store(oq + j.to(tl.int64) * H + offs, tl.load(xq + tok.to(tl.int64) * H + offs))
    g = tl.arange(0, GP)
    gm = g < G
    tl.store(os_ + j * G + g, tl.load(xs + tok * G + g, mask=gm), mask=gm)
    tl.store(inv + flat, j)


@triton.jit
def _weighted_reduce_kernel(
    y, y2, w, inv, out, N,
    TOPK: tl.constexpr, H: tl.constexpr, BH: tl.constexpr, HAS_Y2: tl.constexpr,
    SKIP_ZERO: tl.constexpr,
):
    t = tl.program_id(0)
    offs = tl.program_id(1) * BH + tl.arange(0, BH)
    acc = tl.zeros([BH], dtype=tl.float32)
    for k in tl.static_range(TOPK):
        r = tl.load(inv + t * TOPK + k)
        r = tl.minimum(tl.maximum(r, 0), N - 1)
        wk = tl.load(w + t * TOPK + k).to(tl.float32)
        row = r.to(tl.int64) * H + offs
        if SKIP_ZERO:
            # Zero-weight entries may point at rows no GEMM wrote (EP: remote experts).
            m = (offs < H) & (wk != 0.0)
            v = tl.load(y + row, mask=m, other=0.0).to(tl.float32)
            if HAS_Y2:
                v += tl.load(y2 + row, mask=m, other=0.0).to(tl.float32)
        else:
            v = tl.load(y + row).to(tl.float32)
            if HAS_Y2:
                v += tl.load(y2 + row).to(tl.float32)
        acc += v * wk
    tl.store(out + t.to(tl.int64) * H + offs, acc.to(out.dtype.element_ty))


def _pow2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def _reduce_block(hidden_size: int) -> int:
    """Largest power of two <= 2048 dividing ``hidden_size`` (0 if below 32)."""
    bh = min(hidden_size & -hidden_size, 2048)
    return bh if bh >= 32 else 0


def reduce_supported(hidden_size: int) -> bool:
    return _reduce_block(hidden_size) > 0


def _gather_block(hidden_size: int) -> int:
    """Row tile of ``gather_rows``: the whole row for powers of two, else the
    largest power of two <= 2048 dividing it (MiniMax-M2 H=3072: 3 x 1024)."""
    return hidden_size if _pow2(hidden_size) else _reduce_block(hidden_size)


def supported(hidden_size: int, group_size: int) -> bool:
    return (
        hidden_size % group_size == 0
        and _gather_block(hidden_size) >= 256
        and _reduce_block(hidden_size) >= 256
    )


def gather_rows(permuted_rows, x_q, x_s, top_k):
    """Expand quantized tokens into expert-sorted rows.

    ``permuted_rows[j]`` is the flat ``(token * top_k + k)`` entry placed at
    row ``j`` (``moe_sort`` output). Returns ``(expanded_q, expanded_s,
    invert_permutation)`` with ``invert_permutation[flat] = j``.
    """
    n = permuted_rows.numel()
    m, h = x_q.shape
    g = x_s.shape[1]
    dev = x_q.device
    expanded_q = torch.empty(n, h, device=dev, dtype=torch.int8)
    expanded_s = torch.empty(n, g, device=dev, dtype=torch.float32)
    inv = torch.empty(n, device=dev, dtype=torch.int32)
    if n:
        _gather_rows_kernel[(n,)](
            permuted_rows, x_q.contiguous(), x_s.contiguous(),
            expanded_q, expanded_s, inv, n, m, top_k, h, _gather_block(h), g, triton.next_power_of_2(g),
        )
    return expanded_q, expanded_s, inv


def weighted_reduce(y, topk_weights, invert_permutation, out, y2=None, skip_zero=False):
    """``out[t] = sum_k topk_weights[t, k] * y[invert_permutation[t * top_k + k]]``.

    ``y2`` (same shape as ``y``) is summed with ``y`` in fp32 before weighting.
    ``skip_zero``: rows of zero-weight entries are not read (may be unwritten).
    """
    m, top_k = topk_weights.shape
    h = y.shape[1]
    bh = _reduce_block(h)
    if m:
        _weighted_reduce_kernel[(m, h // bh)](
            y, y if y2 is None else y2, topk_weights.contiguous(), invert_permutation,
            out, y.shape[0], top_k, h, bh, y2 is not None, skip_zero,
        )
    return out
