# Copyright (c) 2026 BAAI. All rights reserved.

"""MoE router top-k on ``llm_kernel_ptpu.topk_softmax`` (fused softmax + top-k + renorm).

Covers plain softmax routing (Qwen2/Qwen3/Qwen3.5 MoE) and ungrouped sigmoid +
correction-bias routing (MiniMax-M2, ``_topk_triton``). Grouped routing,
custom routing functions, fused shared experts, EPLB
dispatch and padded-token masking raise ``NotImplementedError`` so OpManager
falls back to the reference implementation.
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _sigmoid_bias_topk_kernel(
    logits,
    bias,
    w_out,
    id_out,
    stride_l,
    E: tl.constexpr,
    K: tl.constexpr,
    RENORM: tl.constexpr,
):
    """Sigmoid + correction-bias top-k in one Triton row kernel."""
    row = tl.program_id(0)
    offs = tl.arange(0, E)
    x = tl.load(logits + row.to(tl.int64) * stride_l + offs).to(tl.float32)
    s = 1.0 / (1.0 + tl.exp(-x))
    c = s + tl.load(bias + offs).to(tl.float32)
    ks = tl.arange(0, K)
    wv = tl.zeros([K], dtype=tl.float32)
    iv = tl.zeros([K], dtype=tl.int32)
    for k in tl.static_range(K):
        max_value = tl.max(c, 0)
        idx = tl.min(tl.where(c == max_value, offs, E), 0)
        weight = tl.sum(tl.where(offs == idx, s, 0.0), 0)
        wv = tl.where(ks == k, weight, wv)
        iv = tl.where(ks == k, idx, iv)
        c = tl.where(offs == idx, float("-inf"), c)
    if RENORM:
        wv = wv / tl.sum(wv, 0)
    tl.store(w_out + row * K + ks, wv)
    tl.store(id_out + row * K + ks, iv)


def _sigmoid_bias_topk_supported(num_experts: int, top_k: int) -> bool:
    return (
        num_experts >= 32
        and num_experts & (num_experts - 1) == 0
        and 0 < top_k <= num_experts
        and top_k & (top_k - 1) == 0
    )


def _sigmoid_bias_topk(
    router_logits: torch.Tensor,
    correction_bias: torch.Tensor,
    top_k: int,
    renormalize: bool,
):
    m, num_experts = router_logits.shape
    weights = torch.empty(
        m, top_k, dtype=torch.float32, device=router_logits.device
    )
    ids = torch.empty(m, top_k, dtype=torch.int32, device=router_logits.device)
    if m:
        logits = (
            router_logits
            if router_logits.stride(1) == 1
            else router_logits.contiguous()
        )
        _sigmoid_bias_topk_kernel[(m,)](
            logits,
            correction_bias,
            weights,
            ids,
            logits.stride(0),
            E=num_experts,
            K=top_k,
            RENORM=bool(renormalize),
        )
    return weights, ids


def _fast_path_ok(cfg, num_token_non_padded, expert_location_dispatch_info) -> bool:
    return (
        cfg.scoring_func == "softmax"
        and not cfg.use_grouped_topk
        and cfg.correction_bias is None
        and cfg.custom_routing_function is None
        and not cfg.num_fused_shared_experts
        and not (cfg.routed_scaling_factor and cfg.apply_routed_scaling_factor_on_output)
        and num_token_non_padded is None
        and expert_location_dispatch_info is None
    )


def _sigmoid_bias_path_ok(cfg, router_logits, num_token_non_padded, expert_location_dispatch_info) -> bool:
    return (
        cfg.scoring_func == "sigmoid"
        and cfg.correction_bias is not None
        and not cfg.use_grouped_topk
        and cfg.custom_routing_function is None
        and not cfg.num_fused_shared_experts
        and not (cfg.routed_scaling_factor and cfg.apply_routed_scaling_factor_on_output)
        and num_token_non_padded is None
        and expert_location_dispatch_info is None
        and router_logits.dim() == 2
        and _sigmoid_bias_topk_supported(router_logits.shape[1], cfg.top_k)
    )


def topk_sunrise(
    obj,
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    *,
    num_token_non_padded: Optional[torch.Tensor] = None,
    expert_location_dispatch_info=None,
):
    from sglang.srt.layers.moe.topk import StandardTopKOutput

    cfg = obj.topk_config
    if _sigmoid_bias_path_ok(cfg, router_logits, num_token_non_padded, expert_location_dispatch_info):
        topk_weights, topk_ids = _sigmoid_bias_topk(
            router_logits, cfg.correction_bias, cfg.top_k, cfg.renormalize
        )
        return StandardTopKOutput(topk_weights, topk_ids, router_logits)
    if not _fast_path_ok(cfg, num_token_non_padded, expert_location_dispatch_info):
        raise NotImplementedError("sunrise topk: only plain softmax / sigmoid+bias routing")

    from llm_kernel_ptpu import topk_softmax

    m = router_logits.shape[0]
    topk_weights = torch.empty(m, cfg.top_k, dtype=torch.float32, device=router_logits.device)
    topk_ids = torch.empty(m, cfg.top_k, dtype=torch.int32, device=router_logits.device)
    if m > 0:
        topk_softmax(topk_weights, topk_ids, router_logits.contiguous(), cfg.renormalize)
    return StandardTopKOutput(topk_weights, topk_ids, router_logits)
