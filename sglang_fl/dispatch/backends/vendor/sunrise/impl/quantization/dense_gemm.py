# Copyright (c) 2026 BAAI. All rights reserved.

"""Dense block-INT8 projections on ``torch_ptpu.sgl_kernel.w8a8_block_int8_matmul``.

Replacement for sglang's ``apply_w8a8_block_int8_linear``. Weights are
``[N, K]`` int8 with one FP32 dequant scale per ``[block_n, block_k]`` tile
(the checkpoint's ``weight_scale_inv``).
"""

from __future__ import annotations

import os
from typing import List, Optional

import torch
import torch_ptpu

from torch_ptpu.sgl_kernel import (
    get_w8a8_block_int8_matmul_workspace_bytes,
    w8a8_block_int8_matmul,
)

from . import gemm_plan
from .scaled_int8_quant import per_token_group_quant_int8_sunrise

# The kernel drops to ~120 TOPS (vs ~190) for M >= 4096 when N is a multiple
# of 4096 and >= 8192 (TP=1 in_proj_qkvz, N=12288: 3.42 ms at M=8192 vs
# 2.26 ms as 2048-row pieces). Such GEMMs run as row blocks of at most this
# many rows; 0 disables.
_M_SPLIT = int(os.environ.get("SGLANG_FL_SUNRISE_INT8_GEMM_M_SPLIT", "3072"))


def _m_split_rows(m: int, n: int) -> int:
    if _M_SPLIT <= 0 or m <= _M_SPLIT or n < 8192 or n % 4096:
        return 0
    return _M_SPLIT


# For 64 <= M < 128 the kernel picks a split-K config (MB-sized workspace)
# that is slower than its M=128 config when N >= 8192 and M=128 does not split
# K (K=5120: N=16384 0.26-0.35 ms vs 0.18 ms, N=34816 0.53-0.77 vs 0.32).
# Such GEMMs run on 128 rows, the extra rows uninitialised and dropped.
_M_PAD = 128 if os.environ.get("SGLANG_FL_SUNRISE_INT8_GEMM_M_PAD", "1") != "0" else 0
_NO_SPLIT_K_BYTES = 1024
_m_pad_cache: dict = {}


def _m_pad_rows(m: int, n: int, k: int, block_n: int, block_k: int) -> int:
    if not _M_PAD or m < 64 or m >= _M_PAD or n < 8192:
        return 0
    key = (m, n, k, block_n, block_k)
    rows = _m_pad_cache.get(key)
    if rows is None:
        ws = get_w8a8_block_int8_matmul_workspace_bytes
        split_k = ws(m, n, k, block_n, block_k) >= _NO_SPLIT_K_BYTES
        rows = _M_PAD if split_k and ws(_M_PAD, n, k, block_n, block_k) < _NO_SPLIT_K_BYTES else 0
        _m_pad_cache[key] = rows
    return rows


def _matmul_row_blocks(q_input, weight, x_scale, weight_scale, block_n, block_k, rows):
    m, k = q_input.shape
    n = weight.shape[0]
    output = torch.empty((m, n), device=q_input.device, dtype=torch.bfloat16)
    # Workspace size is not monotonic in M (N=8192, K=5120: 0.5 MB at 3072
    # rows, 3.7 MB at 57), so the tail block must be sized on its own.
    block_rows = {rows, m % rows} - {0}
    workspace = torch.empty(
        max(
            max(get_w8a8_block_int8_matmul_workspace_bytes(r, n, k, block_n, block_k) for r in block_rows),
            1,
        ),
        device=q_input.device,
        dtype=torch.int8,
    )
    for s in range(0, m, rows):
        e = min(s + rows, m)
        torch_ptpu.opmx.sgl.w8a8_block_int8_matmul(
            workspace, output[s:e], q_input[s:e], weight, x_scale[s:e], weight_scale,
            block_n, block_k, 1,
        )
    return output


def _matmul(q_input, x_scale, weight, weight_scale, block_n, block_k):
    workspace = torch.empty(
        max(get_w8a8_block_int8_matmul_workspace_bytes(
            q_input.shape[0], weight.shape[0], q_input.shape[1], block_n, block_k), 1),
        device=q_input.device, dtype=torch.int8,
    )
    # The kernel always emits bfloat16.
    return w8a8_block_int8_matmul(workspace, q_input, weight, x_scale, weight_scale, block_n, block_k)


def run_plan(input_2d, weight, weight_scale, block_n, block_k, plan) -> torch.Tensor:
    """``[M, N]`` bf16 output of the GEMM run as ``plan`` (``None``: at M rows)."""
    m = input_2d.shape[0]
    if plan is not None and plan[0] == "pad":
        q_input, x_scale = per_token_group_quant_int8_sunrise(input_2d, block_k, rows=plan[1])
        return _matmul(q_input, x_scale, weight, weight_scale, block_n, block_k)[:m]
    q_input, x_scale = per_token_group_quant_int8_sunrise(input_2d, block_k)
    if plan is not None and plan[0] == "split":
        s = plan[1]
        return torch.cat((
            _matmul(q_input, x_scale, weight[:s], weight_scale[: s // block_n], block_n, block_k),
            _matmul(q_input, x_scale, weight[s:], weight_scale[s // block_n:], block_n, block_k),
        ), dim=1)
    rows = _m_split_rows(m, weight.shape[0])
    if rows:
        return _matmul_row_blocks(q_input, weight, x_scale, weight_scale, block_n, block_k, rows)
    return _matmul(q_input, x_scale, weight, weight_scale, block_n, block_k)


def tune_linear(weight: torch.Tensor, weight_scale: torch.Tensor, block_size: List[int]) -> None:
    block_n, block_k = block_size[0], block_size[1]
    gemm_plan.tune(
        weight, weight_scale, block_n, block_k,
        lambda x, plan: run_plan(x, weight, weight_scale, block_n, block_k, plan),
    )


def apply_w8a8_block_int8_linear_sunrise(
    input: torch.Tensor,
    weight: torch.Tensor,
    block_size: List[int],
    weight_scale: torch.Tensor,
    input_scale: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """``y = dequant(quant(x)) @ dequant(W).T (+ bias)`` for block-INT8 weights.

    ``weight_scale`` is ``[ceil(N / block_n), ceil(K / block_k)]`` float32.
    """
    assert input_scale is None, "Sunrise block-INT8 only supports dynamic activation scales"
    block_n, block_k = block_size[0], block_size[1]

    input_2d = input.reshape(-1, input.shape[-1])
    m = input_2d.shape[0]
    n, k = weight.shape
    output_shape = [*input.shape[:-1], n]

    if m == 0:
        return torch.empty(output_shape, device=input.device, dtype=input.dtype)

    plan = gemm_plan.lookup(m, n, k, block_n, block_k)
    if plan is None:
        pad = _m_pad_rows(m, n, k, block_n, block_k)
        plan = ("pad", pad) if pad else None
    output = run_plan(input_2d, weight, weight_scale, block_n, block_k, plan)
    if bias is not None:
        output = output + bias
    return output.to(dtype=input.dtype).view(*output_shape)
