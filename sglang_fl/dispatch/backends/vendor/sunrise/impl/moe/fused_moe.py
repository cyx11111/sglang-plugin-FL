# Copyright (c) 2026 BAAI. All rights reserved.

"""Sunrise (PTPU) FusedMoE entry: ``call_op("fused_moe", obj, layer, dispatch_output)``.

Unpacks the sglang dispatch output and routes by the layer's weights:
  * BF16 / unquantized (no ``w2`` scale): ``fused_moe_bf16.py``
  * block-INT8 (``w2_weight_scale_inv`` set): ``fused_moe_int8.py``
"""

from __future__ import annotations

from typing import Tuple

import torch

from .fused_moe_bf16 import fused_moe_bf16
from .fused_moe_int8 import fused_moe_int8_blockwise


def _unpack_topk(dispatch_output) -> Tuple[torch.Tensor, torch.Tensor]:
    topk_output = dispatch_output.topk_output
    if hasattr(topk_output, "topk_weights") and hasattr(topk_output, "topk_ids"):
        return topk_output.topk_weights, topk_output.topk_ids
    # Legacy tuple: (topk_weights, topk_ids, ...)
    return topk_output[0], topk_output[1]


def _moe_config(obj):
    cfg = getattr(obj, "moe_runner_config", None)
    if cfg is not None:
        return cfg
    runner = getattr(obj, "runner", None)
    if runner is not None:
        return runner.config
    return None


def _first_attr(obj, names):
    for name in names:
        value = getattr(obj, name, None)
        if value is not None:
            return value
    return None


def _layer_quant_tensors(obj, layer: torch.nn.Module):
    """Pull optional INT8 scales / block_shape from the MoE layer when present.

    sglang's ``BlockInt8MoEMethod`` registers ``w13_weight_scale_inv`` /
    ``w2_weight_scale_inv`` and keeps the block size on its quant config.
    """
    w13_scale = _first_attr(
        layer, ("w13_weight_scale_inv", "w13_weight_scale", "w13_scale")
    )
    w2_scale = _first_attr(layer, ("w2_weight_scale_inv", "w2_weight_scale", "w2_scale"))
    a13_scale = _first_attr(layer, ("w13_input_scale", "a13_scale"))
    a2_scale = _first_attr(layer, ("w2_input_scale", "a2_scale"))
    block_shape = getattr(layer, "block_shape", None)
    if block_shape is None:
        quant_config = getattr(obj, "quant_config", None)
        block_shape = getattr(quant_config, "weight_block_size", None)
    return w13_scale, w2_scale, a13_scale, a2_scale, block_shape


def fused_moe_sunrise(
    obj,
    layer: torch.nn.Module,
    dispatch_output,
):
    """
    PTPU fused MoE expert path for UnquantizedFusedMoEMethod OOT dispatch
    and for SunriseBlockInt8MoEMethod.apply.

    Args:
        obj: UnquantizedFusedMoEMethod / SunriseBlockInt8MoEMethod instance
        layer: MoE layer (w13_weight / w2_weight, optional INT8 scales)
        dispatch_output: StandardDispatchOutput (hidden_states + topk_output)

    Returns:
        StandardCombineInput
    """
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput

    hidden_states = dispatch_output.hidden_states
    topk_weights, topk_ids = _unpack_topk(dispatch_output)

    # PTPU moe_* kernels require int32 ids (SGLang TopK often returns int64).
    topk_ids = topk_ids.to(torch.int32)

    cfg = _moe_config(obj)
    activation = getattr(cfg, "activation", "silu") if cfg is not None else "silu"
    no_combine = getattr(cfg, "no_combine", False) if cfg is not None else False
    inplace = getattr(cfg, "inplace", False) if cfg is not None else False
    apply_router_weight_on_input = (
        getattr(cfg, "apply_router_weight_on_input", False) if cfg is not None else False
    )

    assert activation == "silu", "only support silu"
    assert not no_combine, "unsupported"
    assert not apply_router_weight_on_input, "unsupported"

    w13 = layer.w13_weight
    w2 = layer.w2_weight
    w13_scale, w2_scale, _a13_scale, _a2_scale, block_shape = _layer_quant_tensors(
        obj, layer
    )

    expert_parallel = getattr(layer, "moe_ep_size", 1) > 1
    if w2_scale is None:
        if expert_parallel:
            raise NotImplementedError("sunrise BF16 MoE does not support expert parallelism (--ep-size)")
        y = fused_moe_bf16(
            hidden_states=hidden_states,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            w13=w13,
            w2=w2,
            inplace=inplace,
            weights_k_major=getattr(
                layer, "_sunrise_bf16_group_mm_layout", False
            ),
        )
    else:
        assert block_shape is not None, "INT8 MoE requires layer.block_shape"
        y = fused_moe_int8_blockwise(
            hidden_states=hidden_states,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            w13=w13,
            w2=w2,
            w13_scale=w13_scale,
            w2_scale=w2_scale,
            block_shape=block_shape,
            inplace=inplace,
            i_chunk=getattr(layer, "_sunrise_moe_i_chunk", 0),
            expert_parallel=expert_parallel,
        )

    return StandardCombineInput(hidden_states=y)
