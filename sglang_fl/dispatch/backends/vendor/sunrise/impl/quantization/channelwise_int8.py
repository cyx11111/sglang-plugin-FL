# Copyright (c) 2026 BAAI. All rights reserved.
"""PTPU dynamic-token / channel-weight W8A8 kernels.

Checkpoint contract:
* weight ``[N, K]`` INT8, scale ``[N, 1]`` FP32;
* activation is quantized per row to INT8 with scale ``[M, 1]`` FP32;
* output is BF16.

Both kernels use static launch configurations so they are safe in PTPU graphs.
"""

from __future__ import annotations

from typing import Optional, Tuple
import re
import os

import torch
import triton
import triton.language as tl

_DG_BLOCK = 128
_DG_MIN_N = 256
_DG_MIN_K = 1024
_M_PER_BLOCK = re.compile(r"m_per_block=(\d+)")
_DG_SUPPORT: dict[tuple[int, int, int], bool] = {}
_DG_ONES_A: dict[torch.device, torch.Tensor] = {}
_DG_ONES_B: dict[tuple[int, int, torch.device], torch.Tensor] = {}

@triton.jit
def _quant_rows_kernel(
    x,
    q,
    scale,
    K,
    stride_xm,
    stride_qm,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < K
    value = tl.load(x + row * stride_xm + cols, mask=mask, other=0.0).to(
        tl.float32
    )
    amax = tl.max(tl.abs(value))
    s = tl.maximum(amax, 1.0e-10) / 127.0
    scaled = value / s
    # Round half away from zero without CUDA libdevice calls.
    rounded = tl.where(
        scaled >= 0, tl.floor(scaled + 0.5), tl.ceil(scaled - 0.5)
    )
    rounded = tl.minimum(tl.maximum(rounded, -128), 127)
    tl.store(q + row * stride_qm + cols, rounded.to(tl.int8), mask=mask)
    tl.store(scale + row, s)


def per_token_quant_int8(
    x: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize ``[..., K]`` BF16/FP16 activations per row."""
    k = x.shape[-1]
    x2d = x.reshape(-1, k).contiguous()
    m = x2d.shape[0]
    q = torch.empty_like(x2d, dtype=torch.int8)
    scale = torch.empty((m, 1), dtype=torch.float32, device=x.device)
    if m:
        if m >= 256 and k <= 2048:
            # The vendor group quantizer accepts one group spanning the whole
            # row and wins once prefill supplies enough rows. It truncates like
            # Sunrise blockwise activation quantization.
            from torch_ptpu.sgl_kernel import (
                per_token_group_quant_int8 as vendor_quant,
            )

            vendor_quant(x2d, q, scale, k, 1.0e-10, -128.0, 127.0)
        else:
            # At decode batch sizes the vendor launch has a higher fixed cost;
            # the one-program-per-row kernel is faster and uses round-to-nearest
            # as required by the compressed-tensors reference path.
            block = triton.next_power_of_2(k)
            _quant_rows_kernel[(m,)](
                x2d,
                q,
                scale,
                k,
                x2d.stride(0),
                q.stride(0),
                BLOCK=block,
                num_warps=min(max(block // 256, 1), 8),
                num_stages=1,
            )
    return q.reshape(x.shape), scale


@triton.jit
def _native_epilogue(
    out,
    a_scale,
    w_scale,
    bias,
    N,
    HAS_BIAS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs < N
    value = tl.load(out + row * N + offs, mask=mask, other=0.0).to(tl.float32)
    value *= tl.load(a_scale + row)
    value *= tl.load(w_scale + offs, mask=mask, other=0.0)
    if HAS_BIAS:
        value += tl.load(bias + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(out + row * N + offs, value.to(tl.bfloat16), mask=mask)


def _capturing() -> bool:
    try:
        from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode

        return bool(get_is_capture_mode())
    except Exception:
        return False


def _deepgemm_profitable(dg, m: int, n: int, k: int) -> bool:
    key = (m, n, k)
    if key in _DG_SUPPORT:
        return _DG_SUPPORT[key]
    # The current vendor table can select a fallback config that faults for
    # large-M channelwise prefill (seen at TP1 N=2048,K=4096). Large eager
    # shapes use SD/Triton below; keep deepgemm for decode/mid-M only.
    if _capturing() or m >= 256 or n < _DG_MIN_N or k < _DG_MIN_K:
        return False
    ok = False
    try:
        if dg.is_support_int8_blockwise_gemm(
            m, n, k, _DG_BLOCK, _DG_BLOCK, 1
        ):
            widths = []
            for fn in (dg.get_best_config_log, dg.get_fallback_config_log):
                found = _M_PER_BLOCK.search(
                    str(fn(m, n, k, _DG_BLOCK, _DG_BLOCK, 1))
                )
                if found:
                    widths.append(int(found.group(1)))
            tile = min(widths) if widths else 0
            ok = bool(tile) and not (tile <= 16 and m > 4 * tile)
    except Exception:
        ok = False
    _DG_SUPPORT[key] = ok
    return ok


def _ones_a(m: int, kb: int, device: torch.device) -> torch.Tensor:
    need = m * kb
    buf = _DG_ONES_A.get(device)
    if buf is None or buf.numel() < need:
        if _capturing():
            out = torch.empty(m, kb, dtype=torch.float32, device=device)
            out.fill_(1)
            return out
        buf = torch.empty(
            1 << max(need - 1, 0).bit_length(),
            dtype=torch.float32,
            device=device,
        )
        buf.fill_(1)
        _DG_ONES_A[device] = buf
    return buf[:need].view(m, kb)


def _ones_b(n: int, k: int, device: torch.device) -> torch.Tensor:
    key = (n, k, device)
    out = _DG_ONES_B.get(key)
    if out is None:
        out = torch.empty(
            n // _DG_BLOCK,
            k // _DG_BLOCK,
            dtype=torch.float32,
            device=device,
        )
        out.fill_(1)
        _DG_ONES_B[key] = out
    return out


def _try_native_mm(q, weight, a_scale, w_scale, bias):
    m, k = q.shape
    n = weight.shape[0]
    try:
        import deepgemm_ptpu as dg
    except Exception:
        return None
    if n % _DG_BLOCK or k % _DG_BLOCK:
        return None
    if not _deepgemm_profitable(dg, m, n, k):
        return None
    out = dg.int8_blockwise_normal_gemm(
        q,
        weight,
        _ones_a(m, k // _DG_BLOCK, q.device),
        _ones_b(n, k, q.device),
        _DG_BLOCK,
        _DG_BLOCK,
    )
    block_n = min(1024, triton.next_power_of_2(n))
    _native_epilogue[(m, triton.cdiv(n, block_n))](
        out,
        a_scale,
        w_scale,
        bias,
        n,
        HAS_BIAS=bias is not None,
        BLOCK_N=block_n,
    )
    return out


def _try_sd_mm(q, weight, a_scale, w_scale, bias, out_dtype):
    """Tablas-backed channelwise GEMM for large eager-prefill shapes."""
    m, k = q.shape
    n = weight.shape[0]
    # Kept opt-in: isolated shapes are correct and fast, but the current
    # torch_ptpu implementation can leave the allocator in an error state
    # during multi-rank model prefill ("too many resources requested").
    if os.environ.get("SGLANG_FL_SUNRISE_CHANNEL_SD", "0") != "1":
        return None
    if _capturing() or m < 256 or n < 4096 or k < 2048:
        return None
    try:
        from torch_ptpu.sd_kernel import int8_scaled_mm
    except (ImportError, AttributeError):
        return None
    return int8_scaled_mm(
        q,
        weight,
        a_scale,
        w_scale,
        dtype=out_dtype,
        bias=bias,
    )


@triton.jit
def _channelwise_mm_kernel(
    a,
    weight,
    a_scale,
    w_scale,
    bias,
    out,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_wn,
    stride_wk,
    stride_om,
    stride_on,
    HAS_BIAS: tl.constexpr,
    EVEN_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    width = GROUP_M * grid_n
    group = pid // width
    first_m = group * GROUP_M
    group_m = min(grid_m - first_m, GROUP_M)
    pid_m = first_m + (pid % group_m)
    pid_n = (pid % width) // group_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    safe_m = tl.minimum(offs_m, M - 1)
    safe_n = tl.minimum(offs_n, N - 1)

    a_ptrs = a + safe_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    # Stored weight is [N, K]; present its tile to dot as [K, N].
    w_ptrs = (
        weight
        + safe_n[None, :] * stride_wn
        + offs_k[:, None] * stride_wk
    )
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for k_off in range(0, K, BLOCK_K):
        if EVEN_K:
            av = tl.load(a_ptrs)
            wv = tl.load(w_ptrs)
        else:
            remain = K - k_off
            av = tl.load(a_ptrs, mask=offs_k[None, :] < remain, other=0)
            wv = tl.load(w_ptrs, mask=offs_k[:, None] < remain, other=0)
        acc = tl.dot(av, wv, acc, out_dtype=tl.int32)
        a_ptrs += BLOCK_K * stride_ak
        w_ptrs += BLOCK_K * stride_wk

    value = acc.to(tl.float32)
    value *= tl.load(a_scale + safe_m)[:, None]
    value *= tl.load(w_scale + safe_n) [None, :]
    if HAS_BIAS:
        value += tl.load(bias + safe_n).to(tl.float32)[None, :]

    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(
        out + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
        value.to(out.dtype.element_ty),
        mask=mask,
    )


def _config(m: int, n: int, k: int) -> tuple[int, int, int, int]:
    """Conservative PTPU INT8-MMA tiles validated by the sibling vLLM path."""
    if m <= 16:
        if k <= 128:
            return 16, min(64, triton.next_power_of_2(n)), 128, 8
        block_n = 16 if n <= 1024 else (32 if n <= 5120 else 64)
        return 16, block_n, min(256, triton.next_power_of_2(k)), 8
    if m <= 128:
        block_m = min(64, triton.next_power_of_2(m))
        block_n = 32 if n <= 5120 and k >= 1024 else 64
        return block_m, block_n, min(256, triton.next_power_of_2(k)), 8
    block_m = 64
    block_n = 128 if m > 512 and n >= 5120 and k >= 512 else 64
    return block_m, block_n, min(128, triton.next_power_of_2(k)), 8


def channelwise_w8a8_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Apply a channelwise W8A8 Linear while preserving leading dimensions."""
    if weight.dtype != torch.int8 or weight.dim() != 2:
        raise ValueError("channelwise W8A8 weight must be 2-D INT8")
    if weight_scale.dtype != torch.float32 or tuple(weight_scale.shape) != (
        weight.shape[0],
        1,
    ):
        raise ValueError("channelwise W8A8 scale must be FP32 [N, 1]")
    k = x.shape[-1]
    n, wk = weight.shape
    if wk != k:
        raise ValueError(f"Linear K mismatch: activation {k}, weight {wk}")

    x2d = x.reshape(-1, k).contiguous()
    q, a_scale = per_token_quant_int8(x2d)
    m = x2d.shape[0]
    native = _try_native_mm(q, weight, a_scale, weight_scale, bias)
    if native is not None:
        return native.reshape(x.shape[:-1] + (n,))
    native = _try_sd_mm(
        q, weight, a_scale, weight_scale, bias, x.dtype
    )
    if native is not None:
        return native.reshape(x.shape[:-1] + (n,))
    out = torch.empty((m, n), dtype=x.dtype, device=x.device)
    if m:
        bm, bn, bk, warps = _config(m, n, k)
        grid = (triton.cdiv(m, bm) * triton.cdiv(n, bn),)
        _channelwise_mm_kernel[grid](
            q,
            weight,
            a_scale,
            weight_scale,
            bias,
            out,
            m,
            n,
            k,
            q.stride(0),
            q.stride(1),
            weight.stride(0),
            weight.stride(1),
            out.stride(0),
            out.stride(1),
            HAS_BIAS=bias is not None,
            EVEN_K=(k % bk == 0),
            BLOCK_M=bm,
            BLOCK_N=bn,
            BLOCK_K=bk,
            GROUP_M=8,
            num_warps=warps,
            num_stages=2,
        )
    return out.reshape(x.shape[:-1] + (n,))


__all__ = ["channelwise_w8a8_linear", "per_token_quant_int8"]
