# Copyright (c) 2026 BAAI. All rights reserved.
"""MiniMax-M2 QK RMSNorm (TP) on PTPU.

``MiniMaxM2QKRMSNorm._forward_naive`` computes per-row sums of squares of q
and k (all-reduced over the attention TP group as one fp32 buffer
``[q rows | k rows | pad]``), then ``x * rsqrt(sum / D / tp + eps) * w``. Its
wrappers assert ``is_cuda``; the two Triton kernels are 1-D row kernels and run
unchanged on PTPU. ``_forward_naive`` itself is replaced so q / k (column slices
of the qkv projection) are read with their row strides instead of being copied. ``SGLANG_FL_SUNRISE_MINIMAX_QKNORM_TRITON=0`` uses torch ops
(about 10 small kernels per layer instead of 2).
"""

import logging
import os

import torch

logger = logging.getLogger(__name__)

_patched = False
_TRITON = os.environ.get("SGLANG_FL_SUNRISE_MINIMAX_QKNORM_TRITON", "1") != "0"


def rms_sumsq_torch(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    b = x1.shape[0]
    assert x2.shape[0] == b
    sum_sq = torch.zeros((2 * b + 3) // 4 * 4, device=x1.device, dtype=torch.float32)
    sum_sq[:b] = x1.float().square().sum(dim=-1)
    sum_sq[b:2 * b] = x2.float().square().sum(dim=-1)
    return sum_sq


def rms_apply_torch(x1, x2, w1, w2, sum_sq, tp_world: int = 1, eps: float = 1e-5):
    b, d1 = x1.shape
    d2 = x2.shape[1]
    inv1 = torch.rsqrt(sum_sq[:b] / d1 / tp_world + eps).unsqueeze(1)
    inv2 = torch.rsqrt(sum_sq[b:2 * b] / d2 / tp_world + eps).unsqueeze(1)
    out1 = (x1.float() * inv1 * w1.float()).to(x1.dtype)
    out2 = (x2.float() * inv2 * w2.float()).to(x2.dtype)
    return out1, out2


def _apply_kernel_strided():
    import triton
    import triton.language as tl

    @triton.jit
    def _qk_rms_apply_kernel(
        x1, x2, w1, w2, sum_sq, o1, o2, B, D1, D2, stride_x1, stride_x2, tp_world, eps,
        BS1: tl.constexpr, BS2: tl.constexpr,
    ):
        """``rmsnorm_apply_kernel_serial`` reading row-strided q / k (column slices
        of the qkv projection) and writing contiguous outputs."""
        row = tl.program_id(0)
        inv1 = tl.rsqrt(tl.load(sum_sq + row) / D1 / tp_world + eps)
        inv2 = tl.rsqrt(tl.load(sum_sq + row + B) / D2 / tp_world + eps)
        c1 = tl.arange(0, BS1)
        m1 = c1 < D1
        c2 = tl.arange(0, BS2)
        m2 = c2 < D2
        a = tl.load(x1 + row.to(tl.int64) * stride_x1 + c1, mask=m1, other=0.0)
        b = tl.load(x2 + row.to(tl.int64) * stride_x2 + c2, mask=m2, other=0.0)
        wa = tl.load(w1 + c1, mask=m1, other=1.0).to(tl.float32)
        wb = tl.load(w2 + c2, mask=m2, other=1.0).to(tl.float32)
        tl.store(o1 + row.to(tl.int64) * D1 + c1, (a.to(tl.float32) * inv1 * wa).to(a.dtype), mask=m1)
        tl.store(o2 + row.to(tl.int64) * D2 + c2, (b.to(tl.float32) * inv2 * wb).to(b.dtype), mask=m2)

    return _qk_rms_apply_kernel


def _make_forward_naive(minimax_m2, rms_sumsq_serial):
    """``MiniMaxM2QKRMSNorm._forward_naive`` without the q / k ``.contiguous()`` copies."""
    import triton

    apply_kernel = _apply_kernel_strided()

    def _forward_naive(self, q, k):
        if q.stride(-1) != 1 or k.stride(-1) != 1:
            q, k = q.contiguous(), k.contiguous()
        sum_sq = rms_sumsq_serial(q, k)
        if self._world_size > 1:
            sum_sq = minimax_m2.attn_tp_all_reduce(sum_sq)
        b, d1 = q.shape
        d2 = k.shape[1]
        out1 = torch.empty(b, d1, device=q.device, dtype=q.dtype)
        out2 = torch.empty(b, d2, device=k.device, dtype=k.dtype)
        if b:
            apply_kernel[(b,)](
                q, k, self._q_norm.weight, self._k_norm.weight, sum_sq, out1, out2, b, d1, d2,
                q.stride(0), k.stride(0), self._world_size, self._eps,
                triton.next_power_of_2(d1), triton.next_power_of_2(d2),
            )
        return out1, out2

    return _forward_naive


def _make_triton(minimax_m2):
    import triton

    sumsq_kernel = minimax_m2.rmsnorm_sumsq_kernel_serial
    apply_kernel = minimax_m2.rmsnorm_apply_kernel_serial

    def rms_sumsq_serial(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        b, d1 = x1.shape
        d2 = x2.shape[1]
        assert x2.shape[0] == b
        # Padding (16-byte all-reduce alignment) is never read.
        sum_sq = torch.empty((2 * b + 3) // 4 * 4, device=x1.device, dtype=torch.float32)
        if b:
            sumsq_kernel[(b,)](
                x1, x2, x1.stride(0), x2.stride(0), sum_sq, b, d1, d2,
                triton.next_power_of_2(d1), triton.next_power_of_2(d2),
            )
        return sum_sq

    def rms_apply_serial(x1, x2, w1, w2, sum_sq, tp_world: int = 1, eps: float = 1e-5):
        # The kernel writes the outputs with the inputs' row strides.
        x1, x2 = x1.contiguous(), x2.contiguous()
        b, d1 = x1.shape
        d2 = x2.shape[1]
        out1 = torch.empty(b, d1, device=x1.device, dtype=x1.dtype)
        out2 = torch.empty(b, d2, device=x2.device, dtype=x2.dtype)
        if b:
            apply_kernel[(b,)](
                x1, x2, w1, w2, sum_sq, out1, out2, b, d1, d2, x1.stride(0), x2.stride(0),
                tp_world, eps, triton.next_power_of_2(d1), triton.next_power_of_2(d2),
            )
        return out1, out2

    return rms_sumsq_serial, rms_apply_serial


def patch_minimax_m2_for_sunrise() -> None:
    global _patched
    if _patched:
        return
    try:
        from sglang.srt.models import minimax_m2
    except Exception as exc:
        logger.info("sunrise: minimax_m2 unavailable, QK norm not patched (%s)", exc)
        return
    if _TRITON:
        minimax_m2.rms_sumsq_serial, minimax_m2.rms_apply_serial = _make_triton(minimax_m2)
        minimax_m2.MiniMaxM2QKRMSNorm._forward_naive = _make_forward_naive(
            minimax_m2, minimax_m2.rms_sumsq_serial
        )
    else:
        minimax_m2.rms_sumsq_serial = rms_sumsq_torch
        minimax_m2.rms_apply_serial = rms_apply_torch
    _patched = True
    logger.info("sunrise: MiniMax-M2 TP QK RMSNorm -> %s", "Triton row kernels" if _TRITON else "torch")
