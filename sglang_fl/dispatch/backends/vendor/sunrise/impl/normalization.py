# Copyright (c) 2026 BAAI. All rights reserved.

"""
Sunrise (PTPU) normalization operator implementations.

Routes ``rms_norm`` / ``gemma_rms_norm`` to PTPU native
``rmsnorm`` / ``fused_add_rmsnorm`` from ``torch_ptpu.sgl_kernel``.

Ported from vllm-plugin-fl sunrise ``impl/normalization.py`` +
``impl/gemma_rms_norm.py`` into the sglang_fl ``call_op`` registration path.
"""

from __future__ import annotations

import os
from typing import Optional, Union

import torch
import triton
import triton.language as tl

from torch_ptpu.sgl_kernel import fused_add_rmsnorm, rmsnorm

_TRITON_ROWS = os.environ.get("SGLANG_FL_SUNRISE_RMSNORM_TRITON", "1") != "0"
_ROWS_PER_PROGRAM = int(os.environ.get("SGLANG_FL_SUNRISE_RMSNORM_ROWS", "1"))
_MIN_ROWS = int(os.environ.get("SGLANG_FL_SUNRISE_RMSNORM_MIN_ROWS", "1024"))
_WARPS = int(os.environ.get("SGLANG_FL_SUNRISE_RMSNORM_WARPS", "4"))
_MIN_HIDDEN_SIZE = 1024


@triton.jit
def _rmsnorm_kernel(
    x_ptr,
    r_ptr,
    w_ptr,
    o_ptr,
    ro_ptr,
    n_rows,
    rows_per_program,
    stride_x,
    stride_r,
    eps,
    H: tl.constexpr,
    HAS_RES: tl.constexpr,
):
    offs = tl.arange(0, H)
    weight = tl.load(w_ptr + offs).to(tl.float32)
    row_start = tl.program_id(0) * rows_per_program
    row_end = tl.minimum(row_start + rows_per_program, n_rows)
    x_row = x_ptr + row_start.to(tl.int64) * stride_x + offs
    r_row = r_ptr + row_start.to(tl.int64) * stride_r + offs
    o_row = o_ptr + row_start.to(tl.int64) * H + offs
    ro_row = ro_ptr + row_start.to(tl.int64) * H + offs
    for _ in range(row_start, row_end):
        value = tl.load(x_row).to(tl.float32)
        if HAS_RES:
            value = value + tl.load(r_row).to(tl.float32)
            tl.store(ro_row, value.to(ro_ptr.dtype.element_ty))
        rstd = 1 / tl.sqrt(tl.sum(value * value, axis=0) / H + eps)
        tl.store(o_row, (value * rstd * weight).to(o_ptr.dtype.element_ty))
        x_row += stride_x
        r_row += stride_r
        o_row += H
        ro_row += H


def _triton_rmsnorm_supported(
    x: torch.Tensor, weight: torch.Tensor, residual=None
) -> bool:
    hidden_size = x.shape[-1]
    return (
        x.dim() == 2
        and x.shape[0] >= _MIN_ROWS
        and x.stride(1) == 1
        and hidden_size & (hidden_size - 1) == 0
        and _MIN_HIDDEN_SIZE <= hidden_size <= 8192
        and weight.shape == (hidden_size,)
        and weight.is_contiguous()
        and (
            residual is None
            or (
                residual.shape == x.shape
                and residual.stride(1) == 1
                and residual.dtype == x.dtype
            )
        )
    )


def _triton_rmsnorm_rows(
    x: torch.Tensor, weight: torch.Tensor, eps: float, residual=None
):
    """Return ``out`` or ``(out, new_residual)`` without modifying inputs."""
    num_rows, hidden_size = x.shape
    out = torch.empty(
        (num_rows, hidden_size), dtype=x.dtype, device=x.device
    )
    residual_out = (
        torch.empty_like(out) if residual is not None else out
    )
    _rmsnorm_kernel[(triton.cdiv(num_rows, _ROWS_PER_PROGRAM),)](
        x,
        residual if residual is not None else x,
        weight,
        out,
        residual_out,
        num_rows,
        _ROWS_PER_PROGRAM,
        x.stride(0),
        residual.stride(0) if residual is not None else 0,
        eps,
        H=hidden_size,
        HAS_RES=residual is not None,
        num_warps=_WARPS,
    )
    return (out, residual_out) if residual is not None else out


def _norm(x_flat, residual_flat, weight, epsilon):
    if _TRITON_ROWS and _triton_rmsnorm_supported(
        x_flat, weight, residual_flat
    ):
        return _triton_rmsnorm_rows(
            x_flat, weight, epsilon, residual=residual_flat
        )
    if residual_flat is not None:
        return fused_add_rmsnorm(x_flat, residual_flat, weight, epsilon)
    return rmsnorm(x_flat, weight, epsilon)


def _flat_last_dim(x: torch.Tensor) -> tuple[torch.Tensor, tuple]:
    """Flatten to ``[tokens, hidden]`` for PTPU kernels (``reshape`` not ``view``)."""
    orig_shape = x.shape
    return x.reshape(-1, orig_shape[-1]), orig_shape


def rms_norm_sunrise(
    obj,
    x: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """RMS normalization using PTPU sgl_kernel.

    Args:
        obj: RMSNorm-like layer (``obj.weight``, ``obj.variance_epsilon``).
        x: Input tensor; last dim must equal ``obj.weight.shape[-1]``.
        residual: Optional residual for fused add+rmsnorm.

    Returns:
        ``x_normalized``, or ``(x_normalized, new_residual)`` when residual given.
    """
    weight = obj.weight
    epsilon = obj.variance_epsilon

    x_flat, orig_shape = _flat_last_dim(x)

    if residual is not None:
        residual_flat = residual.reshape(-1, orig_shape[-1])
        x_out, residual_out = _norm(x_flat, residual_flat, weight, epsilon)
        return x_out.view(orig_shape), residual_out.view(orig_shape)

    x_out = _norm(x_flat, None, weight, epsilon)
    return x_out.view(orig_shape)


def _gemma_offset_weight(obj) -> torch.Tensor:
    """Lazy ``(weight + 1)`` cache — GemmaRMSNorm uses zero-centred weights."""
    cached = getattr(obj, "_fl_cached_gemma_offset_weight", None)
    weight_data = obj.weight.data
    if (
        cached is None
        or cached.device != weight_data.device
        or cached.dtype != weight_data.dtype
        or cached.shape != weight_data.shape
    ):
        cached = (weight_data.detach() + 1.0).to(weight_data.dtype)
        obj._fl_cached_gemma_offset_weight = cached
    return cached


def gemma_rms_norm_sunrise(
    obj,
    x: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """GemmaRMSNorm via PTPU rmsnorm with effective weight ``(1 + w)``.

    Same idea as vLLM sunrise ``GemmaRMSNormFL``: reuse stock ``rmsnorm`` /
    ``fused_add_rmsnorm`` after baking Gemma's +1 into the weight tensor.
    """
    weight = _gemma_offset_weight(obj)
    epsilon = obj.variance_epsilon

    x_flat, orig_shape = _flat_last_dim(x)

    if residual is not None:
        residual_flat = residual.reshape(-1, orig_shape[-1])
        x_out, residual_out = _norm(x_flat, residual_flat, weight, epsilon)
        return x_out.view(orig_shape), residual_out.view(orig_shape)

    x_out = _norm(x_flat, None, weight, epsilon)
    return x_out.view(orig_shape)
