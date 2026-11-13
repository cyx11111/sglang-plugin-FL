# Copyright (c) 2026 BAAI. All rights reserved.
"""Channelwise W8A8 MoE on the vendor INT8 grouped GEMM."""

from __future__ import annotations

import logging
import re
from typing import Optional

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)
_N_PER_BLOCK = re.compile(r"n_per_block=(\d+)")
_BLOCK_CACHE: dict[tuple[int, int, int, int], Optional[int]] = {}
_ONES: dict[tuple[int, int, int, int, torch.device], torch.Tensor] = {}


def _capturing() -> bool:
    try:
        from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode

        return bool(get_is_capture_mode())
    except Exception:
        return False


def _block(dg, rows: int, n: int, k: int, experts: int) -> Optional[int]:
    key = (rows, n, k, experts)
    if key in _BLOCK_CACHE:
        return _BLOCK_CACHE[key]
    if _capturing():
        return None
    chosen, best_tile = None, 0
    for block in (128, 64):
        if n % block or k % block:
            continue
        try:
            if not dg.is_support_int8_blockwise_gemm(
                rows, n, k, block, block, experts
            ):
                continue
            tiles = []
            for fn in (dg.get_best_config_log, dg.get_fallback_config_log):
                found = _N_PER_BLOCK.search(
                    str(fn(rows, n, k, block, block, experts))
                )
                if found:
                    tiles.append(int(found.group(1)))
            tile = min(tiles) if tiles else 0
        except Exception:
            continue
        if tile and block <= 2 * tile and tile > best_tile:
            chosen, best_tile = block, tile
    _BLOCK_CACHE[key] = chosen
    return chosen


def _ones(experts, n_blocks, k_blocks, block, device):
    key = (experts, n_blocks, k_blocks, block, device)
    out = _ONES.get(key)
    if out is None:
        out = torch.empty(
            experts, n_blocks, k_blocks, dtype=torch.float32, device=device
        )
        out.fill_(1)
        out = out.transpose(1, 2)
        _ONES[key] = out
    return out


@triton.jit
def _quant_gather(
    src,
    dst,
    scales,
    rows,
    inverse,
    K,
    SFK,
    TOPK: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_SFK: tl.constexpr,
):
    i = tl.program_id(0)
    flat = tl.load(rows + i)
    token = flat // TOPK
    offs = tl.arange(0, BLOCK_K)
    mask = offs < K
    value = tl.load(src + token * K + offs, mask=mask, other=0.0).to(tl.float32)
    scale = tl.maximum(tl.max(tl.abs(value)), 1.0e-10) / 127.0
    q = value / scale
    q = tl.where(q >= 0, tl.floor(q + 0.5), tl.ceil(q - 0.5))
    q = tl.minimum(tl.maximum(q, -128), 127)
    tl.store(dst + i * K + offs, q.to(tl.int8), mask=mask)
    soffs = tl.arange(0, BLOCK_SFK)
    tl.store(scales + i * SFK + soffs, scale, mask=soffs < SFK)
    tl.store(inverse + flat, i)


@triton.jit
def _silu_quant(
    x,
    q,
    scales,
    wscale,
    row_expert,
    N,
    ROWS_TOTAL,
    SFK,
    ROWS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_SFK: tl.constexpr,
):
    row = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    col = tl.arange(0, BLOCK_N)
    rmask = row < ROWS_TOTAL
    mask = rmask[:, None] & (col[None, :] < N)
    base = x + row[:, None] * (2 * N) + col[None, :]
    gate = tl.load(base, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(base + N, mask=mask, other=0.0).to(tl.float32)
    expert = tl.load(row_expert + row, mask=rmask, other=0)
    ws = wscale + expert[:, None] * (2 * N) + col[None, :]
    gate *= tl.load(ws, mask=mask, other=0.0)
    up *= tl.load(ws + N, mask=mask, other=0.0)
    value = (gate * tl.sigmoid(gate)) * up
    scale = tl.maximum(tl.max(tl.abs(value), axis=1), 1.0e-10) / 127.0
    quant = value / scale[:, None]
    quant = tl.where(
        quant >= 0, tl.floor(quant + 0.5), tl.ceil(quant - 0.5)
    )
    quant = tl.minimum(tl.maximum(quant, -128), 127)
    tl.store(
        q + row[:, None] * N + col[None, :],
        quant.to(tl.int8),
        mask=mask,
    )
    soffs = tl.arange(0, BLOCK_SFK)
    tl.store(
        scales + row[:, None] * SFK + soffs[None, :],
        scale[:, None],
        mask=rmask[:, None] & (soffs[None, :] < SFK),
    )


@triton.jit
def _reduce(
    src,
    weights,
    inverse,
    out,
    wscale,
    row_expert,
    COLS,
    TOPK: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    token = tl.program_id(0)
    offs = tl.program_id(1) * BLOCK_C + tl.arange(0, BLOCK_C)
    mask = offs < COLS
    acc = tl.zeros([BLOCK_C], tl.float32)
    for slot in tl.static_range(TOPK):
        flat = token * TOPK + slot
        row = tl.load(inverse + flat)
        expert = tl.load(row_expert + row)
        value = tl.load(src + row * COLS + offs, mask=mask, other=0.0)
        scale = tl.load(
            wscale + expert * COLS + offs, mask=mask, other=0.0
        )
        acc += value.to(tl.float32) * scale * tl.load(weights + flat)
    tl.store(out + token * COLS + offs, acc.to(tl.bfloat16), mask=mask)


def try_fused(
    hidden_states,
    w1,
    w2,
    topk_weights,
    topk_ids,
    inplace,
    w1_scale,
    w2_scale,
) -> Optional[torch.Tensor]:
    try:
        import deepgemm_ptpu as dg
        import llm_kernel_ptpu as lk
    except Exception:
        return None
    if hidden_states.dtype != torch.bfloat16 or not hidden_states.is_contiguous():
        return None
    tokens, k = hidden_states.shape
    experts, n, wk = w1.shape
    width = w2.shape[2]
    rows = topk_ids.numel()
    # Qwen3.5 TP4 has a very narrow second stage (K=128): vendor grouped GEMM
    # improves eager decode but its large-prefill fallback is substantially
    # slower, and it gives no graph-replay win over the tuned Triton path.
    if rows > 1024:
        return None
    if wk != k or w2.shape != (experts, k, width):
        return None
    if w1_scale.shape != (experts, n, 1):
        return None
    if w2_scale.shape != (experts, k, 1):
        return None
    b1 = _block(dg, rows, n, k, experts)
    b2 = _block(dg, rows, k, width, experts)
    if b1 is None or b2 is None:
        return None

    topk = topk_ids.shape[1]
    sfk1, sfk2 = k // b1, width // b2
    route = torch.zeros(2 * experts + 1, dtype=torch.int32, device=hidden_states.device)
    histogram, boundaries = route[:experts], route[experts:]
    permuted = torch.empty(rows, dtype=torch.int32, device=hidden_states.device)
    row_expert = torch.empty(rows, dtype=torch.int32, device=hidden_states.device)
    ids = topk_ids if topk_ids.dtype == torch.int32 else topk_ids.to(torch.int32)
    lk.moe_exp_count(ids, histogram)
    lk.moe_sort(ids, histogram, boundaries[1:], permuted, row_expert)

    inverse = torch.empty(rows, dtype=torch.int32, device=hidden_states.device)
    a1 = torch.empty(rows, k, dtype=torch.int8, device=hidden_states.device)
    s1 = torch.empty(rows, sfk1, dtype=torch.float32, device=hidden_states.device)
    bk = triton.next_power_of_2(k)
    _quant_gather[(rows,)](
        hidden_states,
        a1,
        s1,
        permuted,
        inverse,
        k,
        sfk1,
        TOPK=topk,
        BLOCK_K=bk,
        BLOCK_SFK=triton.next_power_of_2(sfk1),
        num_warps=min(max(bk // 256, 1), 8),
    )

    c1 = torch.empty(rows, n, dtype=torch.bfloat16, device=hidden_states.device)
    dg.int8_blockwise_group_gemm(
        a1,
        w1.transpose(1, 2),
        s1,
        _ones(experts, n // b1, sfk1, b1, hidden_states.device),
        b1,
        b1,
        boundaries,
        out=c1,
    )

    a2 = torch.empty(rows, width, dtype=torch.int8, device=hidden_states.device)
    s2 = torch.empty(rows, sfk2, dtype=torch.float32, device=hidden_states.device)
    rows_per_program = 8 if width <= 128 and width == triton.next_power_of_2(width) else 1
    _silu_quant[(triton.cdiv(rows, rows_per_program),)](
        c1,
        a2,
        s2,
        w1_scale,
        row_expert,
        width,
        rows,
        sfk2,
        ROWS=rows_per_program,
        BLOCK_N=triton.next_power_of_2(width),
        BLOCK_SFK=triton.next_power_of_2(sfk2),
        num_warps=min(max(rows_per_program * width // 256, 1), 8),
    )

    c2 = torch.empty(rows, k, dtype=torch.bfloat16, device=hidden_states.device)
    dg.int8_blockwise_group_gemm(
        a2,
        w2.transpose(1, 2),
        s2,
        _ones(experts, k // b2, sfk2, b2, hidden_states.device),
        b2,
        b2,
        boundaries,
        out=c2,
    )
    out = hidden_states if inplace else torch.empty_like(hidden_states)
    block_c = min(256, triton.next_power_of_2(k))
    _reduce[(tokens, triton.cdiv(k, block_c))](
        c2,
        topk_weights.reshape(-1),
        inverse,
        out,
        w2_scale,
        row_expert,
        k,
        TOPK=topk,
        BLOCK_C=block_c,
        num_warps=8,
    )
    return out


__all__ = ["try_fused"]
