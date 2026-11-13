# Copyright (c) 2026 BAAI. All rights reserved.
"""EAGLE chain (``topk == 1``) tree build and greedy verify, one program per request.

Replace the CUDA / NPU ``build_tree_kernel_efficient`` / ``verify_tree_greedy``
ops on PTPU. The torch-op versions took 1.7 / 2.4 ms per MTP step at 24
requests (a dozen eager kernels each).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _chain_tree_kernel(
    seq_lens, positions, retrieve_index, retrieve_next, retrieve_sibling, mask, bs,
    N: tl.constexpr, NP2: tl.constexpr, BS_UP: tl.constexpr, BLOCK: tl.constexpr, FULL: tl.constexpr,
):
    b = tl.program_id(0)
    seq_len = tl.load(seq_lens + b).to(tl.int64)
    i = tl.arange(0, NP2)
    mi = i < N
    base = b.to(tl.int64) * N
    tl.store(positions + base + i, seq_len + i, mask=mi)
    tl.store(retrieve_index + base + i, base + i, mask=mi)
    tl.store(retrieve_next + base + i, tl.where(i < N - 1, i + 1, -1), mask=mi)
    tl.store(retrieve_sibling + base + i, tl.full([NP2], -1, tl.int64), mask=mi)
    if FULL:
        # Request b's block starts after sum_{r < b} (seq_len_r + N) * N entries.
        r = tl.arange(0, BS_UP)
        prev = tl.load(seq_lens + r, mask=r < b, other=0).to(tl.int64)
        start = tl.sum(tl.where(r < b, (prev + N) * N, 0), axis=0)
        row_len = seq_len + N
        for row in tl.static_range(N):
            row_ptr = mask + start + row * row_len
            limit = seq_len + row + 1
            for c0 in range(0, row_len, BLOCK):
                c = c0 + tl.arange(0, BLOCK)
                tl.store(row_ptr + c, c < limit, mask=c < row_len)
    else:
        # QLEN_ONLY: per request an N x N causal block.
        rr = tl.arange(0, NP2)
        for row in tl.static_range(N):
            tl.store(mask + base * N + row * N + rr, rr <= row, mask=rr < N)


@triton.jit
def _chain_verify_kernel(
    candidates, target, predicts, accept_index, accept_num,
    N: tl.constexpr, NP2: tl.constexpr, S1: tl.constexpr,
):
    b = tl.program_id(0)
    i = tl.arange(0, NP2)
    m = i < N
    base = b.to(tl.int64) * N
    tgt = tl.load(target + base + i, mask=m, other=0)
    nxt = tl.load(candidates + base + i + 1, mask=i + 1 < N, other=0)
    match = (nxt == tgt) & (i + 1 < N)
    # Number of accepted drafts = index of the first mismatch.
    accepted = tl.min(tl.where(match, NP2, i), axis=0)
    on = i <= accepted
    old = tl.load(predicts + base + i, mask=m, other=0)
    tl.store(predicts + base + i, tl.where(on, tgt.to(old.dtype), old), mask=m)
    ai = tl.where(on, base + i, -1)
    tl.store(accept_index + b.to(tl.int64) * S1 + i, ai.to(accept_index.dtype.element_ty), mask=i < S1)
    tl.store(accept_num + b, accepted.to(accept_num.dtype.element_ty))


def chain_tree(seq_lens: torch.Tensor, seq_lens_sum: int, n: int, full_mask: bool,
               tree_mask_buf=None, position_buf=None):
    """Returns ``(tree_mask, positions, retrieve_index, retrieve_next_token, retrieve_next_sibling)``."""
    bs = seq_lens.numel()
    dev = seq_lens.device
    positions = position_buf if position_buf is not None else torch.empty(bs * n, dtype=torch.int64, device=dev)
    retrieve = torch.empty((3, bs, n), dtype=torch.int64, device=dev)
    total = seq_lens_sum * n + n * n * bs if full_mask else n * n * bs
    mask = tree_mask_buf if tree_mask_buf is not None else torch.empty(total, dtype=torch.bool, device=dev)
    if bs:
        _chain_tree_kernel[(bs,)](
            seq_lens, positions, retrieve[0], retrieve[1], retrieve[2], mask, bs,
            N=n, NP2=triton.next_power_of_2(n), BS_UP=triton.next_power_of_2(bs), BLOCK=1024,
            FULL=full_mask, num_warps=1,
        )
    return mask, positions, retrieve[0], retrieve[1], retrieve[2]


def chain_verify(predicts, accept_index, accept_token_num, candidates, target_predict):
    """Greedy verify of EAGLE chains; same outputs as ``verify_tree_greedy`` with ``topk == 1``."""
    bs, n = candidates.shape
    if bs:
        _chain_verify_kernel[(bs,)](
            candidates.contiguous(), target_predict.contiguous(), predicts, accept_index, accept_token_num,
            N=n, NP2=triton.next_power_of_2(n), S1=accept_index.shape[1], num_warps=1,
        )
    return predicts, accept_index, accept_token_num
