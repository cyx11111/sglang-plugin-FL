# Copyright (c) 2026 BAAI. All rights reserved.

"""PTPU BF16 routed-MoE: moe_sort / expand-permute / grouped GEMM / reduce.

Paths:
  * ``deepgemm_ptpu.m_grouped_bf16_gemm_nt_contiguous`` when the vendor GEMM
    supports the launch. Launch shape is ``M * top_k``, so PTPU graph capture
    works.
  * FlagGems Sunrise ``group_mm`` otherwise. It consumes the device-side
    expert end offsets directly and is graph-capture safe.
  * Otherwise a per-expert ``F.linear`` loop. It copies ``cumsum`` to the
    host; that ``tangStreamSynchronize`` is illegal inside graph capture.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F

from torch_ptpu.sgl_kernel import (
    moe_exp_count,
    moe_expand_permute,
    moe_reduce,
    moe_sort,
    silu_and_mul,
)

def _stream_capturing() -> bool:
    # torch_ptpu has no ``is_current_stream_capturing``; SGLang flags the
    # whole graph capture (warmup + capture) via ``model_capture_mode``.
    try:
        from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode

        return bool(get_is_capture_mode())
    except Exception:
        return False


def prepare_bf16_moe_weights(layer: torch.nn.Module) -> None:
    """Rebind BF16 expert weights once to contiguous ``[E, K, N]`` layout.

    FlagGems Sunrise ``group_mm`` requires each group's matrix to be contiguous
    in K-major layout. Rebinding after checkpoint loading avoids both a
    per-forward transpose/copy and retaining a second full expert-weight copy.
    """
    if getattr(layer, "_sunrise_bf16_group_mm_layout", False):
        return
    w13 = layer.w13_weight
    w2 = layer.w2_weight
    if w13.dtype != torch.bfloat16 or w2.dtype != torch.bfloat16:
        return
    if w13.ndim != 3 or w2.ndim != 3:
        return

    # Already [E, K, N] (e.g. a runner created Triton-layout weights).
    if (
        w13.shape[1] == w2.shape[2]
        and w13.shape[2] == 2 * w2.shape[1]
    ):
        layer._sunrise_bf16_group_mm_layout = True
        return
    # Standard SGLang layout is [E, N, K].
    if not (
        w13.shape[2] == w2.shape[1]
        and w13.shape[1] == 2 * w2.shape[2]
    ):
        return

    from sglang.srt.layers.utils.common import copy_or_rebind_param

    copy_or_rebind_param(
        layer, "w13_weight", w13.transpose(1, 2).contiguous()
    )
    copy_or_rebind_param(
        layer, "w2_weight", w2.transpose(1, 2).contiguous()
    )
    layer._sunrise_bf16_group_mm_layout = True


def _grouped_bf16_supported(
    hidden_states: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    top_k: int,
) -> bool:
    """Whether ``deepgemm_ptpu`` can run this launch entirely on device."""
    if hidden_states.dtype != torch.bfloat16:
        return False
    if w13.dtype != torch.bfloat16 or w2.dtype != torch.bfloat16:
        return False
    if w13.ndim != 3 or w2.ndim != 3:
        return False
    # Grouped GEMM wants K-major B. ``transpose(1, 2)`` of a contiguous
    # ``[E, N, K]`` weight keeps ``stride(1) == 1``.
    if w13.stride(-1) != 1 or w2.stride(-1) != 1:
        return False
    num_experts = w13.shape[0]
    hidden = w2.shape[1]
    inter = w2.shape[2]
    if w13.shape[0] != num_experts or w13.shape[2] != hidden:
        return False
    if w13.shape[1] != 2 * inter:
        return False
    total = hidden_states.shape[0] * top_k
    try:
        import deepgemm_ptpu as deepgemm
    except Exception:
        return False
    return bool(
        deepgemm.is_support_m_grouped_bf16_gemm(total, 2 * inter, hidden, num_experts)
        and deepgemm.is_support_m_grouped_bf16_gemm(total, hidden, inter, num_experts)
    )


def _run_bf16_grouped(
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    inplace: bool,
) -> torch.Tensor:
    """Device-only BF16 MoE. Launch shape is ``M * top_k``, so PTPU graph capture works."""
    import deepgemm_ptpu as deepgemm

    num_tokens, hidden_size = hidden_states.shape
    num_experts = w13.shape[0]
    top_k = topk_ids.shape[-1]
    total = num_tokens * top_k
    device = hidden_states.device

    histogram = torch.zeros(num_experts, device=device, dtype=torch.int32)
    moe_exp_count(topk_ids, histogram)

    cumsum = torch.empty(num_experts, device=device, dtype=torch.int32)
    sorted_expert_ids = torch.zeros(total, device=device, dtype=torch.int32)
    permuted_rows = torch.zeros(total, device=device, dtype=torch.int32)
    moe_sort(topk_ids, histogram, cumsum, permuted_rows, sorted_expert_ids)

    # moe_sort stores the end offset of each expert. Grouped GEMM wants a
    # leading 0 so group g is ``cumsum[g] .. cumsum[g + 1]``.
    dg_cumsum = torch.zeros(num_experts + 1, device=device, dtype=torch.int32)
    dg_cumsum[1:] = cumsum

    expanded_x = torch.empty(
        total, hidden_size, device=device, dtype=hidden_states.dtype
    )
    invert_permutation = torch.empty(total, device=device, dtype=torch.int32)
    moe_expand_permute(hidden_states, permuted_rows, expanded_x, invert_permutation)

    gate_up = deepgemm.m_grouped_bf16_gemm_nt_contiguous(
        expanded_x, w13.transpose(1, 2), dg_cumsum
    )
    mid = silu_and_mul(gate_up)
    expanded_y = deepgemm.m_grouped_bf16_gemm_nt_contiguous(
        mid, w2.transpose(1, 2), dg_cumsum
    )

    y = hidden_states if inplace else torch.empty_like(hidden_states)
    moe_reduce(expanded_y, topk_weights.reshape(-1), invert_permutation, y)
    return y


def _run_bf16_flaggems_grouped(
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    inplace: bool,
) -> torch.Tensor:
    """Device-only BF16 MoE using FlagGems Sunrise grouped matrix multiply."""
    from flag_gems.runtime.backend._sunrise.ops.group_gemm import group_mm

    num_tokens, hidden_size = hidden_states.shape
    num_experts = w13.shape[0]
    top_k = topk_ids.shape[-1]
    total = num_tokens * top_k
    device = hidden_states.device

    histogram = torch.zeros(num_experts, device=device, dtype=torch.int32)
    moe_exp_count(topk_ids, histogram)

    # group_mm expects the inclusive end offset of every expert group.
    cumsum = torch.empty(num_experts, device=device, dtype=torch.int32)
    sorted_expert_ids = torch.zeros(total, device=device, dtype=torch.int32)
    permuted_rows = torch.zeros(total, device=device, dtype=torch.int32)
    moe_sort(topk_ids, histogram, cumsum, permuted_rows, sorted_expert_ids)

    expanded_x = torch.empty(
        total, hidden_size, device=device, dtype=hidden_states.dtype
    )
    invert_permutation = torch.empty(total, device=device, dtype=torch.int32)
    moe_expand_permute(
        hidden_states, permuted_rows, expanded_x, invert_permutation
    )

    # Weights were rebound after loading to contiguous [E, K, N].
    gate_up = group_mm(expanded_x, w13, cumsum)
    mid = silu_and_mul(gate_up)
    expanded_y = group_mm(mid, w2, cumsum)

    y = hidden_states if inplace else torch.empty_like(hidden_states)
    moe_reduce(
        expanded_y,
        topk_weights.reshape(-1),
        invert_permutation,
        y,
    )
    return y


def _run_bf16_host_loop(
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    inplace: bool,
) -> torch.Tensor:
    num_tokens, hidden_size = hidden_states.shape
    num_experts, _, _ = w13.shape
    top_k = topk_ids.shape[-1]

    histogram = torch.zeros(
        num_experts, device=hidden_states.device, dtype=torch.int32
    )
    moe_exp_count(topk_ids, histogram)

    cumsum = torch.empty(
        num_experts, device=hidden_states.device, dtype=torch.int32
    )
    sorted_expert_ids = torch.zeros_like(topk_ids.flatten())
    permuted_rows = torch.zeros_like(topk_ids.flatten())
    moe_sort(topk_ids, histogram, cumsum, permuted_rows, sorted_expert_ids)

    expanded_x = torch.empty(
        num_tokens * top_k,
        hidden_size,
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    invert_permutation = torch.empty(
        num_tokens * top_k, device=hidden_states.device, dtype=torch.int32
    )
    moe_expand_permute(hidden_states, permuted_rows, expanded_x, invert_permutation)

    expanded_y = expanded_x  # inplace
    cumsum_cpu = torch.zeros(num_experts + 1, device="cpu", dtype=torch.int32)
    cumsum_cpu[1:] = cumsum.cpu()
    cumsum_cpu = cumsum_cpu.tolist()
    for e in range(num_experts):
        if cumsum_cpu[e + 1] - cumsum_cpu[e] <= 0:
            continue
        e_input = expanded_x[cumsum_cpu[e] : cumsum_cpu[e + 1]]
        e_input = F.linear(e_input, w13[e])
        e_output = silu_and_mul(e_input)
        e_output = F.linear(e_output, w2[e])
        expanded_y[cumsum_cpu[e] : cumsum_cpu[e + 1]] = e_output

    y = hidden_states if inplace else torch.empty_like(hidden_states)
    moe_reduce(expanded_y, topk_weights, invert_permutation, y)
    return y


def fused_moe_bf16(
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    inplace: bool,
    weights_k_major: bool = False,
) -> torch.Tensor:
    top_k = topk_ids.shape[-1]
    if not weights_k_major and _grouped_bf16_supported(
        hidden_states, w13, w2, top_k
    ):
        return _run_bf16_grouped(
            hidden_states, topk_weights, topk_ids, w13, w2, inplace
        )
    if weights_k_major:
        return _run_bf16_flaggems_grouped(
            hidden_states, topk_weights, topk_ids, w13, w2, inplace
        )
    if _stream_capturing():
        raise RuntimeError(
            "PTPU graph capture cannot run the host-sync BF16 MoE loop "
            "(cumsum.cpu() calls tangStreamSynchronize). "
            "deepgemm_ptpu.m_grouped_bf16_gemm_nt_contiguous does not support "
            f"this launch (M={hidden_states.shape[0]}, top_k={top_k}, "
            f"w13={tuple(w13.shape)}, w2={tuple(w2.shape)}, dtype={hidden_states.dtype})."
        )
    return _run_bf16_host_loop(
        hidden_states, topk_weights, topk_ids, w13, w2, inplace
    )
