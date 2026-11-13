# Copyright (c) 2026 BAAI. All rights reserved.
"""Block-align top-k assignments for SGLang's Triton MoE on PTPU."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

_BLOCK = 256


@triton.jit(do_not_specialize=["numel"])
def _count_and_init(
    topk_ids,
    sorted_ids,
    expert_ids,
    counts,
    ranks,
    numel,
    max_padded,
    max_blocks,
    NUM_EXPERTS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    tl.store(sorted_ids + offs, numel, mask=offs < max_padded)
    tl.store(expert_ids + offs, -1, mask=offs < max_blocks)
    valid = offs < numel
    expert = tl.load(topk_ids + offs, mask=valid, other=0).to(tl.int32)
    rank = tl.atomic_add(counts + expert, 1, mask=valid)
    tl.store(ranks + offs, rank, mask=valid)


@triton.jit
def _prefix(
    counts,
    starts_and_totals,
    num_tokens_post_pad,
    NUM_EXPERTS: tl.constexpr,
    BLOCK_EXPERT: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
):
    expert = tl.arange(0, BLOCK_EXPERT)
    valid = expert < NUM_EXPERTS
    total = tl.load(counts + expert, mask=valid, other=0)
    aligned = tl.where(valid, tl.cdiv(total, BLOCK_SIZE_M) * BLOCK_SIZE_M, 0)
    starts = tl.cumsum(aligned, axis=0) - aligned
    tl.store(starts_and_totals + expert, starts, mask=valid)
    tl.store(starts_and_totals + NUM_EXPERTS + expert, total, mask=valid)
    tl.store(num_tokens_post_pad, tl.sum(aligned, axis=0))


@triton.jit(do_not_specialize=["numel"])
def _scatter_and_label(
    topk_ids,
    ranks,
    starts_and_totals,
    sorted_ids,
    expert_ids,
    numel,
    NUM_EXPERTS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid < NUM_EXPERTS:
        start = tl.load(starts_and_totals + pid)
        total = tl.load(starts_and_totals + NUM_EXPERTS + pid)
        first = start // BLOCK_SIZE_M
        for i in range(tl.cdiv(total, BLOCK_SIZE_M)):
            tl.store(expert_ids + first + i, pid)

    offs = pid * BLOCK + tl.arange(0, BLOCK)
    valid = offs < numel
    expert = tl.load(topk_ids + offs, mask=valid, other=0).to(tl.int32)
    rank = tl.load(ranks + offs, mask=valid, other=0)
    start = tl.load(starts_and_totals + expert, mask=valid, other=0)
    tl.store(sorted_ids + start + rank, offs, mask=valid)


def moe_align_block_size_sunrise(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return padded sorted token ids, expert per block, and valid padded size."""
    topk_ids = topk_ids.contiguous()
    numel = topk_ids.numel()
    if not numel:
        empty = torch.empty(0, dtype=torch.int32, device=topk_ids.device)
        return empty, empty.clone(), torch.zeros(
            1, dtype=torch.int32, device=topk_ids.device
        )
    max_padded = (
        numel * block_size
        if numel < num_experts
        else numel + (num_experts + 1) * (block_size - 1)
    )
    max_blocks = triton.cdiv(max_padded, block_size)
    sorted_ids = torch.empty(max_padded, dtype=torch.int32, device=topk_ids.device)
    expert_ids = torch.empty(max_blocks, dtype=torch.int32, device=topk_ids.device)
    num_post = torch.empty(1, dtype=torch.int32, device=topk_ids.device)
    counts = torch.zeros(num_experts, dtype=torch.int32, device=topk_ids.device)
    ranks = torch.empty(numel, dtype=torch.int32, device=topk_ids.device)
    starts = torch.empty(2 * num_experts, dtype=torch.int32, device=topk_ids.device)

    _count_and_init[(triton.cdiv(max(max_padded, max_blocks), _BLOCK),)](
        topk_ids,
        sorted_ids,
        expert_ids,
        counts,
        ranks,
        numel,
        max_padded,
        max_blocks,
        NUM_EXPERTS=num_experts,
        BLOCK=_BLOCK,
    )
    _prefix[(1,)](
        counts,
        starts,
        num_post,
        NUM_EXPERTS=num_experts,
        BLOCK_EXPERT=triton.next_power_of_2(num_experts),
        BLOCK_SIZE_M=block_size,
    )
    _scatter_and_label[
        (max(triton.cdiv(numel, _BLOCK), num_experts),)
    ](
        topk_ids,
        ranks,
        starts,
        sorted_ids,
        expert_ids,
        numel,
        NUM_EXPERTS=num_experts,
        BLOCK_SIZE_M=block_size,
        BLOCK=_BLOCK,
    )
    return sorted_ids, expert_ids, num_post


__all__ = ["moe_align_block_size_sunrise"]
