# Copyright (c) 2026 BAAI. All rights reserved.
"""PTPU INT8-MMA grouped GEMM for channelwise W8A8 MoE."""

from __future__ import annotations

from typing import Any, Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _kernel(
    A,
    B,
    C,
    AScale,
    BScale,
    TopkWeights,
    SortedIds,
    ExpertIds,
    NumPostPad,
    N,
    K,
    EM,
    num_valid_tokens,
    num_rows,
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_asm,
    stride_bse,
    stride_bsn,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    n_m = tl.cdiv(EM, BLOCK_M)
    n_n = tl.cdiv(N, BLOCK_N)
    width = GROUP_M * n_n
    group = pid // width
    first_m = group * GROUP_M
    group_m = min(n_m - first_m, GROUP_M)
    pid_m = first_m + (pid % group_m)
    pid_n = (pid % width) // group_m
    if pid_m * BLOCK_M >= tl.load(NumPostPad):
        return

    token = tl.load(SortedIds + pid_m * BLOCK_M + tl.arange(0, BLOCK_M))
    valid = token < num_valid_tokens
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    safe_out_row = tl.minimum(token, num_valid_tokens - 1)
    c_ptr = C + safe_out_row[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = valid[:, None] & (offs_n[None, :] < N)

    expert = tl.load(ExpertIds + pid_m)
    if expert == -1:
        tl.store(c_ptr, tl.zeros((BLOCK_M, BLOCK_N), C.dtype.element_ty), mask=c_mask)
        return

    a_row = tl.minimum(token // top_k, num_rows - 1)
    safe_n = tl.minimum(offs_n, N - 1)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptr = A + a_row[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptr = (
        B
        + expert * stride_be
        + offs_k[:, None] * stride_bk
        + safe_n[None, :] * stride_bn
    )
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.int32)
    for _ in range(0, K, BLOCK_K):
        acc = tl.dot(
            tl.load(a_ptr), tl.load(b_ptr), acc, out_dtype=tl.int32
        )
        a_ptr += BLOCK_K * stride_ak
        b_ptr += BLOCK_K * stride_bk

    value = acc.to(tl.float32)
    value *= tl.load(AScale + a_row * stride_asm)[:, None]
    value *= tl.load(
        BScale + expert * stride_bse + safe_n * stride_bsn
    )[None, :]
    if MUL_ROUTED_WEIGHT:
        value *= tl.load(TopkWeights + safe_out_row)[:, None]
    tl.store(c_ptr, value.to(C.dtype.element_ty), mask=c_mask)


def can_run(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    A_scale: Optional[torch.Tensor],
    B_scale: Optional[torch.Tensor],
    sorted_token_ids: Optional[torch.Tensor],
    top_k: int,
    config: dict[str, Any],
    use_fp8_w8a8: bool,
    use_int8_w8a8: bool,
    use_int8_w8a16: bool,
    use_int4_w4a16: bool,
    per_channel_quant: bool,
    block_shape,
    B_bias,
    a_use_tma: bool,
    b_use_tma: bool,
    fuse_sum_all_reduce: bool,
    fuse_add_to_output: bool,
) -> bool:
    if not use_int8_w8a8 or not per_channel_quant:
        return False
    if use_fp8_w8a8 or use_int8_w8a16 or use_int4_w4a16:
        return False
    if block_shape is not None or B_bias is not None:
        return False
    if a_use_tma or b_use_tma or fuse_sum_all_reduce or fuse_add_to_output:
        return False
    if sorted_token_ids is None or A_scale is None or B_scale is None:
        return False
    if A.dtype != torch.int8 or B.dtype != torch.int8 or B.ndim != 3:
        return False
    if A_scale.numel() != A.shape[0]:
        return False
    if B_scale.ndim != 3 or B_scale.shape[2] != 1:
        return False
    bk = int(config["BLOCK_SIZE_K"])
    return B.shape[2] % bk == 0 and bk >= 16


def invoke(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    A_scale: torch.Tensor,
    B_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool,
    top_k: int,
    config: dict[str, Any],
) -> None:
    bm = int(config["BLOCK_SIZE_M"])
    bn = int(config["BLOCK_SIZE_N"])
    bk = int(config["BLOCK_SIZE_K"])
    n, k = B.shape[1], B.shape[2]
    em = sorted_token_ids.numel()
    if A.shape[0] < bm:
        em = min(em, A.shape[0] * top_k * bm)
    grid = (triton.cdiv(em, bm) * triton.cdiv(n, bn),)

    if C.dim() == 2:
        stride_cm, stride_cn = C.stride(0), C.stride(1)
    else:
        stride_cm, stride_cn = C.stride(-2), C.stride(-1)
    _kernel[grid](
        A,
        B,
        C,
        A_scale,
        B_scale,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        n,
        k,
        em,
        A.shape[0] * top_k,
        A.shape[0],
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(2),
        B.stride(1),
        stride_cm,
        stride_cn,
        A_scale.stride(0),
        B_scale.stride(0),
        B_scale.stride(1),
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        top_k=top_k,
        BLOCK_M=bm,
        BLOCK_N=bn,
        BLOCK_K=bk,
        GROUP_M=int(config.get("GROUP_SIZE_M", 8)),
        num_warps=int(config.get("num_warps", 8)),
        num_stages=2,
    )


__all__ = ["can_run", "invoke"]
