# Copyright (c) 2026 BAAI. All rights reserved.

"""PTPU block-INT8 W8A8 routed-MoE on ``int8_blockwise_grouped_gemm``.

Tokens are processed in chunks of ``CHUNK_SIZE``. Per chunk: moe_sort,
per-token-group INT8 quant of the un-expanded tokens, expand-permute of the
INT8 rows (+ scale gather), grouped GEMM (w13), fused silu_and_mul + quant,
grouped GEMM (w2), moe_reduce. The grouped GEMM wants K-major weights,
which ``transpose(1, 2)`` of the contiguous ``[E, N, K]`` parameters provides.

Quantizing before the top-k expansion is bit-identical (the quant is
row-wise) and touches 1/top_k of the rows. The row gather and the weighted
reduce run as Triton kernels (``_moe_route_triton``; disable with
``SGLANG_FL_SUNRISE_MOE_TRITON_GLUE=0``). The fused activation quant keeps
silu in fp32 (<=1 LSB / ~0.4% scale difference vs the two-step path); set
``SGLANG_FL_SUNRISE_MOE_FUSED_ACT_QUANT=0`` to use the two-step path.
"""

from __future__ import annotations

import os
from typing import List

import torch

from torch_ptpu.sgl_kernel import (
    get_int8_blockwise_grouped_gemm_workspace_bytes,
    int8_blockwise_grouped_gemm,
    moe_exp_count,
    moe_expand_permute,
    moe_reduce,
    moe_sort,
    silu_and_mul,
)

from . import moe_route as _moe_route_triton
from ..quantization.scaled_int8_quant import per_token_group_quant_int8_sunrise

silu_and_mul_per_token_group_quant_int8 = None
if os.environ.get("SGLANG_FL_SUNRISE_MOE_FUSED_ACT_QUANT", "1") != "0":
    try:
        from llm_kernel_ptpu import silu_and_mul_per_token_group_quant_int8
    except ImportError:  # older llm_kernel_ptpu
        pass

# The fused kernel quantizes in groups of 128 and needs 2*I % 256 == 0.
_FUSED_ACT_QUANT_GROUP = 128

_TRITON_GLUE = os.environ.get("SGLANG_FL_SUNRISE_MOE_TRITON_GLUE", "1") != "0"


def _capturing() -> bool:
    try:
        from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode

        return bool(get_is_capture_mode())
    except Exception:
        return False


def _use_triton_glue(hidden_size, block_k):
    return _TRITON_GLUE and _moe_route_triton.supported(hidden_size, block_k)


def _expand_quantized(x, permuted_rows, top_k, block_k):
    """INT8-quantize ``x`` then expand-permute rows; returns (q, scales, inv_perm)."""
    x_q, x_s = per_token_group_quant_int8_sunrise(x, block_k)
    if _use_triton_glue(x.shape[1], block_k):
        return _moe_route_triton.gather_rows(permuted_rows, x_q, x_s, top_k)
    m_total = permuted_rows.numel()
    expanded_q = torch.empty(m_total, x.shape[1], device=x.device, dtype=torch.int8)
    invert_permutation = torch.empty(m_total, device=x.device, dtype=torch.int32)
    # moe_expand_permute is a row copy; move the INT8 bytes as bf16 pairs.
    moe_expand_permute(
        x_q.view(torch.bfloat16),
        permuted_rows,
        expanded_q.view(torch.bfloat16),
        invert_permutation,
    )
    # permuted_rows holds flat (token * top_k + k) indices. PTPU index_select
    # misreads int32 indices (out-of-bounds asserts, wrong rows); use int64.
    expanded_s = x_s.index_select(0, (permuted_rows // top_k).long())
    return expanded_q, expanded_s, invert_permutation


def _act_quant(y1, block_k):
    n = y1.shape[1]
    if (
        silu_and_mul_per_token_group_quant_int8 is not None
        and block_k == _FUSED_ACT_QUANT_GROUP
        and n % (2 * _FUSED_ACT_QUANT_GROUP) == 0
    ):
        return silu_and_mul_per_token_group_quant_int8(y1)
    return per_token_group_quant_int8_sunrise(silu_and_mul(y1), block_k)


def _i_chunk_views(w13, w2, w13_scale, w2_scale, i_chunk):
    """Per-chunk ``[E, N, K]`` weights/scales of a pool reordered by
    :func:`reorder_moe_weights_for_i_chunks` (parameter shapes are unchanged)."""
    E, n2, H = w13.shape
    n_chunks = (n2 // 2) // i_chunk
    w13_v = w13.view(n_chunks, E, 2 * i_chunk, H)
    w2_v = w2.view(n_chunks, E, H, i_chunk)
    s13_v = w13_scale.view(n_chunks, E, w13_scale.shape[1] // n_chunks, w13_scale.shape[2])
    s2_v = w2_scale.view(n_chunks, E, w2_scale.shape[1], w2_scale.shape[2] // n_chunks)
    return [(w13_v[c], w2_v[c], s13_v[c], s2_v[c]) for c in range(n_chunks)]


def moe_i_chunk_supported(intermediate: int, block_n: int, block_k: int, i_chunk: int) -> bool:
    # Only for 128x128 blocks: with 64x64 blocks (MiniMax-M2.7 experts) the
    # chunked grouped GEMMs are 6-38x slower than one call (E=32, I=1536:
    # 3.76 / 16.7 ms vs 0.57 / 0.44 ms per layer at 8 / 64 tokens).
    return (
        block_n == block_k == 128
        and 0 < i_chunk < intermediate
        and intermediate % i_chunk == 0
        and i_chunk % block_n == 0
        and i_chunk % block_k == 0
    )


@torch.no_grad()
def reorder_moe_weights_for_i_chunks(w13, w2, w13_scale, w2_scale, i_chunk):
    """Reorder in place so each ``i_chunk`` slice of the intermediate dim is one
    contiguous ``[E, 2*i_chunk, H]`` / ``[E, H, i_chunk]`` block.

    ``int8_blockwise_grouped_gemm`` slows down ~10x once the intermediate dim
    exceeds 256 (TP=1: 4 ms/layer at M=24 vs ~0.3 ms for four 128-wide calls).
    w13 rows are ``[gate(I); up(I)]``; chunk ``c`` keeps its gate and up rows.
    """
    E, n2, H = w13.shape
    intermediate = n2 // 2
    n_chunks = intermediate // i_chunk
    gs13 = w13_scale.shape[1] // 2 // n_chunks
    gs2 = w2_scale.shape[2] // n_chunks

    def _reorder(t, shape, perm):
        src = t.view(shape).permute(perm).contiguous()
        t.view(-1).copy_(src.view(-1))
        del src

    _reorder(w13, (E, 2, n_chunks, i_chunk, H), (2, 0, 1, 3, 4))
    _reorder(w13_scale, (E, 2, n_chunks, gs13, w13_scale.shape[2]), (2, 0, 1, 3, 4))
    _reorder(w2, (E, H, n_chunks, i_chunk), (2, 0, 1, 3))
    _reorder(w2_scale, (E, w2_scale.shape[1], n_chunks, gs2), (2, 0, 1, 3))


def _grouped_gemm(out, x_q, x_s, w, w_s, cumsum_dev, dtype, block_n, block_k):
    w_t, w_s_t = w.transpose(1, 2), w_s.transpose(1, 2)
    workspace = torch.empty(
        get_int8_blockwise_grouped_gemm_workspace_bytes(
            x_q.shape[0], w_t.shape[2], x_q.shape[1], w_t.shape[0], block_n, block_k
        ),
        device=x_q.device,
        dtype=torch.int8,
    )
    int8_blockwise_grouped_gemm(
        out, x_q, w_t, x_s, w_s_t, cumsum_dev, workspace, dtype, block_n, block_k
    )


def fused_moe_int8_blockwise(
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    block_shape: List[int],
    inplace: bool,
    i_chunk: int = 0,
    expert_parallel: bool = False,
) -> torch.Tensor:
    """``i_chunk > 0``: weights were reordered by
    :func:`reorder_moe_weights_for_i_chunks` with that chunk size.

    ``expert_parallel``: ``topk_ids`` holds local expert ids, ``-1`` for experts
    on other ranks (SGLang's standard dispatcher). Those entries are sorted
    behind the local experts as an extra id and get zero weight; the weighted
    reduce does not read their rows.
    """
    y = hidden_states if inplace else torch.empty_like(hidden_states)
    chunks = _i_chunk_views(w13, w2, w13_scale, w2_scale, i_chunk) if i_chunk else None

    CHUNK_SIZE = 8 * 1024
    num_tokens, hidden_size = hidden_states.shape
    num_experts, _, _ = w13.shape
    top_k = topk_ids.shape[-1]
    # moe_sort supports power-of-two expert counts (8..256).
    sort_experts = max(8, 1 << num_experts.bit_length()) if expert_parallel else num_experts
    if expert_parallel and not _moe_route_triton.reduce_supported(hidden_size):
        raise NotImplementedError(f"sunrise EP MoE: hidden size {hidden_size} unsupported")

    block_n, block_k = block_shape
    i8_w1_transposed = w13.transpose(1, 2)
    w1_scales_transposed = w13_scale.transpose(1, 2)
    i8_w2_transposed = w2.transpose(1, 2)
    w2_scales_transposed = w2_scale.transpose(1, 2)

    for chunk in range((num_tokens // CHUNK_SIZE) + 1):
        begin_chunk_idx, end_chunk_idx = (
            chunk * CHUNK_SIZE,
            min((chunk + 1) * CHUNK_SIZE, num_tokens),
        )
        x_chunked = hidden_states[begin_chunk_idx:end_chunk_idx]
        tokens_in_chunk, _ = x_chunked.shape

        if tokens_in_chunk == 0:
            break

        topk_ids_chunked = topk_ids[begin_chunk_idx:end_chunk_idx]
        topk_weights_chunked = topk_weights[begin_chunk_idx:end_chunk_idx]
        remote = None
        if expert_parallel:
            remote = topk_ids_chunked < 0
            topk_ids_chunked = torch.where(remote, num_experts, topk_ids_chunked)

        histogram = torch.zeros(
            sort_experts, device=hidden_states.device, dtype=torch.int32
        )
        moe_exp_count(topk_ids_chunked, histogram)

        cumsum = torch.empty(
            sort_experts, device=hidden_states.device, dtype=torch.int32
        )
        # moe_sort writes every entry: each id is a sorted expert (EP: remote ids
        # are mapped to an extra sorted id above).
        sorted_expert_ids = torch.empty_like(
            topk_ids_chunked.flatten(), dtype=torch.int32
        )
        permuted_rows = torch.empty_like(
            topk_ids_chunked.flatten(), dtype=torch.int32
        )

        moe_sort(
            topk_ids_chunked,
            histogram,
            cumsum,
            permuted_rows,
            sorted_expert_ids,
        )

        i8_x1, x_scales1, invert_permutation = _expand_quantized(
            x_chunked, permuted_rows, top_k, block_k
        )

        cumsum_dev = torch.zeros(
            num_experts + 1, device=hidden_states.device, dtype=torch.int32
        )
        cumsum_dev[1:] = cumsum[:num_experts]

        m_total = tokens_in_chunk * top_k
        # Rows the grouped GEMMs run on. The vendor kernel's plan / workspace
        # assume every row belongs to one of the experts (rows past the last
        # offset overran the workspace), so under EP the remote entries,
        # sorted last, are either cut off (host count) or, while capturing a
        # graph, appended to the last local expert (zero weight in the reduce).
        m_gemm = m_total
        if remote is not None:
            topk_weights_chunked = torch.where(remote, 0.0, topk_weights_chunked)
            if _capturing():
                cumsum_dev[num_experts] = m_total
            else:
                m_gemm = int(cumsum[num_experts - 1].item())
        if m_gemm < m_total:
            i8_x1, x_scales1 = i8_x1[:m_gemm], x_scales1[:m_gemm]
        if chunks is not None:
            fuse_last = (expert_parallel or _use_triton_glue(hidden_size, block_k)) and y.is_contiguous()
            expanded_y = last_y = None
            for ci, (w13_c, w2_c, s13_c, s2_c) in enumerate(chunks):
                y2_c = torch.empty(
                    m_total, hidden_size, device=hidden_states.device, dtype=hidden_states.dtype
                )
                if m_gemm:
                    y1_c = torch.empty(
                        m_gemm, w13_c.shape[1], device=hidden_states.device, dtype=hidden_states.dtype
                    )
                    _grouped_gemm(y1_c, i8_x1, x_scales1, w13_c, s13_c, cumsum_dev,
                                  hidden_states.dtype, block_n, block_k)
                    q2_c, s2q_c = _act_quant(y1_c, block_k)
                    del y1_c
                    _grouped_gemm(y2_c[:m_gemm], q2_c, s2q_c, w2_c, s2_c, cumsum_dev,
                                  hidden_states.dtype, block_n, block_k)
                    del q2_c, s2q_c
                if expanded_y is None:
                    expanded_y = y2_c
                elif fuse_last and ci == len(chunks) - 1:
                    last_y = y2_c
                else:
                    expanded_y.add_(y2_c)
                    del y2_c
            del i8_x1
            if last_y is not None:
                _moe_route_triton.weighted_reduce(
                    expanded_y, topk_weights_chunked, invert_permutation,
                    y[begin_chunk_idx:end_chunk_idx], y2=last_y, skip_zero=expert_parallel,
                )
            else:
                _reduce_into(expanded_y, topk_weights_chunked, invert_permutation,
                             y[begin_chunk_idx:end_chunk_idx], hidden_size, block_k, expert_parallel)
            del expanded_y, last_y
            continue

        expanded_y = torch.empty(
            m_total,
            w2.shape[1],
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        if m_gemm == 0:
            _reduce_into(expanded_y, topk_weights_chunked, invert_permutation,
                         y[begin_chunk_idx:end_chunk_idx], hidden_size, block_k, expert_parallel)
            continue

        n_w1 = w13.shape[1]
        expanded_y1 = torch.empty(
            m_gemm,
            n_w1,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )

        workspace_bytes = get_int8_blockwise_grouped_gemm_workspace_bytes(
            i8_x1.shape[0],
            i8_w1_transposed.shape[2],
            i8_x1.shape[1],
            i8_w1_transposed.shape[0],
            block_n,
            block_k,
        )
        workspace = torch.empty(
            workspace_bytes, device=i8_x1.device, dtype=torch.int8
        )
        int8_blockwise_grouped_gemm(
            expanded_y1,
            i8_x1,
            i8_w1_transposed,
            x_scales1,
            w1_scales_transposed,
            cumsum_dev,
            workspace,
            hidden_states.dtype,
            block_n,
            block_k,
        )
        del workspace
        del i8_x1

        i8_x2, x_scales2 = _act_quant(expanded_y1, block_k)
        del expanded_y1

        workspace_bytes = get_int8_blockwise_grouped_gemm_workspace_bytes(
            i8_x2.shape[0],
            i8_w2_transposed.shape[2],
            i8_x2.shape[1],
            i8_w2_transposed.shape[0],
            block_n,
            block_k,
        )
        workspace = torch.empty(
            workspace_bytes, device=i8_x2.device, dtype=torch.int8
        )
        int8_blockwise_grouped_gemm(
            expanded_y[:m_gemm],
            i8_x2,
            i8_w2_transposed,
            x_scales2,
            w2_scales_transposed,
            cumsum_dev,
            workspace,
            hidden_states.dtype,
            block_n,
            block_k,
        )
        del workspace
        del i8_x2

        _reduce_into(expanded_y, topk_weights_chunked, invert_permutation,
                     y[begin_chunk_idx:end_chunk_idx], hidden_size, block_k, expert_parallel)
        del expanded_y

    return y


def _reduce_into(expanded_y, topk_weights, invert_permutation, y_chunk, hidden_size, block_k,
                 expert_parallel=False):
    if expert_parallel:
        out = y_chunk if y_chunk.is_contiguous() else torch.empty_like(y_chunk)
        _moe_route_triton.weighted_reduce(expanded_y, topk_weights, invert_permutation, out,
                                          skip_zero=True)
        if out is not y_chunk:
            y_chunk.copy_(out)
    elif _use_triton_glue(hidden_size, block_k) and y_chunk.is_contiguous():
        _moe_route_triton.weighted_reduce(expanded_y, topk_weights, invert_permutation, y_chunk)
    else:
        moe_reduce(expanded_y, topk_weights, invert_permutation, y_chunk)
