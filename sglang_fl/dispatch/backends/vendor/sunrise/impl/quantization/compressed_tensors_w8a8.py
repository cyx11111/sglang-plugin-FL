# Copyright (c) 2026 BAAI. All rights reserved.
"""Sunrise schemes for compressed-tensors channelwise W8A8.

SGLang's CUDA scheme imports ``sgl_kernel.int8_scaled_mm`` and its fused-MoE
scheme is NPU-only. PTPU routes both Dense and MoE through its native Triton
INT8-MMA paths while retaining the checkpoint's ``[N, 1]`` FP32 scales.

The schemes are selected only by ``patches/compressed_tensors.py`` on Sunrise,
so every non-Sunrise platform is unaffected.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import torch
from compressed_tensors.quantization import QuantizationStrategy

from sglang.srt.layers.moe import MoeRunnerConfig
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsLinearScheme,
    CompressedTensorsMoEScheme,
)
from sglang.srt.layers.parameter import ChannelQuantScaleParameter, ModelWeightParameter
from sglang.srt.utils import set_weight_attrs

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher.standard import (
        StandardCombineInput,
        StandardDispatchOutput,
    )

logger = logging.getLogger(__name__)
_REQUANT_BLOCK = 128


def _requant_channel_to_block(
    weight: torch.Tensor, channel_scale: torch.Tensor, block: int = _REQUANT_BLOCK
) -> tuple[torch.Tensor, torch.Tensor]:
    """Requantize ``[..., N, K]`` channel INT8 to block INT8 without BF16 storage."""
    n, k = weight.shape[-2:]
    if n % block or k % block:
        raise ValueError(f"weight shape {(n, k)} is not divisible by block {block}")
    if channel_scale.shape != weight.shape[:-1] + (1,):
        raise ValueError(
            f"channel scale {tuple(channel_scale.shape)} mismatches "
            f"weight {tuple(weight.shape)}"
        )
    q = torch.empty_like(weight, dtype=torch.int8)
    scales = []
    kb = k // block
    lead = weight.shape[:-2]
    for start in range(0, n, block):
        value = (
            weight[..., start : start + block, :].to(torch.float32)
            * channel_scale[..., start : start + block, :].to(torch.float32)
        )
        tiles = value.reshape(*lead, block, kb, block)
        amax = tiles.abs().amax(dim=(-3, -1)).clamp_min_(1.0e-6)
        scale = amax / 127.0
        quant = torch.round(
            tiles / scale.unsqueeze(-2).unsqueeze(-1)
        ).clamp_(-127, 127)
        q[..., start : start + block, :].copy_(
            quant.to(torch.int8).reshape(*lead, block, k)
        )
        scales.append(scale)
    return q.contiguous(), torch.stack(scales, dim=-2).contiguous()


def _block_requant_enabled() -> bool:
    return os.environ.get(
        "SGLANG_FL_SUNRISE_CHANNEL_REQUANT_BLOCK", "1"
    ) != "0"


class SunriseCompressedTensorsW8A8Linear(CompressedTensorsLinearScheme):
    """Channel/token compressed-tensors W8A8 Linear on PTPU INT8 MMA."""

    def __init__(self, weight_quant, input_quant):
        if weight_quant.strategy != QuantizationStrategy.CHANNEL:
            raise ValueError("sunrise compressed W8A8 requires channel weights")
        if input_quant.strategy != QuantizationStrategy.TOKEN or not input_quant.dynamic:
            raise ValueError(
                "sunrise compressed W8A8 requires dynamic per-token activations"
            )
        if not weight_quant.symmetric or not input_quant.symmetric:
            raise ValueError("sunrise compressed W8A8 currently requires symmetric INT8")

    @classmethod
    def get_min_capability(cls) -> int:
        return 0

    def create_weights(
        self,
        layer: torch.nn.Module,
        output_partition_sizes: list[int],
        input_size_per_partition: int,
        params_dtype: torch.dtype,
        weight_loader,
        **kwargs,
    ) -> None:
        output_size_per_partition = sum(output_partition_sizes)
        layer.logical_widths = output_partition_sizes
        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition,
                dtype=torch.int8,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)
        weight_scale = ChannelQuantScaleParameter(
            data=torch.empty(
                output_size_per_partition, 1, dtype=torch.float32
            ),
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_scale", weight_scale)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.weight = torch.nn.Parameter(
            layer.weight.data.contiguous(), requires_grad=False
        )
        layer.weight_scale = torch.nn.Parameter(
            layer.weight_scale.data.contiguous(), requires_grad=False
        )
        n, k = layer.weight.shape
        if _block_requant_enabled() and n % _REQUANT_BLOCK == 0 and k % _REQUANT_BLOCK == 0:
            weight, scale = _requant_channel_to_block(
                layer.weight.data, layer.weight_scale.data
            )
            layer.weight = torch.nn.Parameter(weight, requires_grad=False)
            layer._sunrise_block_scale = torch.nn.Parameter(
                scale, requires_grad=False
            )
            from .dense_gemm import tune_linear

            tune_linear(
                layer.weight,
                layer._sunrise_block_scale,
                [_REQUANT_BLOCK, _REQUANT_BLOCK],
            )
            logger.info_once(
                "sunrise: aligned compressed-tensors weights requantized "
                "channel INT8 -> 128x128 block INT8 at load time"
            )

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        block_scale = getattr(layer, "_sunrise_block_scale", None)
        if block_scale is not None:
            from .dense_gemm import apply_w8a8_block_int8_linear_sunrise

            return apply_w8a8_block_int8_linear_sunrise(
                input=x,
                weight=layer.weight,
                block_size=[_REQUANT_BLOCK, _REQUANT_BLOCK],
                weight_scale=block_scale,
                input_scale=None,
                bias=bias,
            )
        from .channelwise_int8 import channelwise_w8a8_linear

        return channelwise_w8a8_linear(
            x, layer.weight, layer.weight_scale, bias
        )


class SunriseCompressedTensorsW8A8MoE(CompressedTensorsMoEScheme):
    """Channel/token compressed-tensors experts on SGLang's INT8 Triton MoE."""

    def __init__(self, weight_quant, input_quant):
        if weight_quant.strategy != QuantizationStrategy.CHANNEL:
            raise ValueError("sunrise compressed W8A8 MoE requires channel weights")
        if input_quant.strategy != QuantizationStrategy.TOKEN or not input_quant.dynamic:
            raise ValueError(
                "sunrise compressed W8A8 MoE requires dynamic per-token activations"
            )
        if not weight_quant.symmetric or not input_quant.symmetric:
            raise ValueError("sunrise compressed W8A8 MoE requires symmetric INT8")
        self.moe_runner_config = None

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        from sglang.srt.layers.moe.fused_moe_triton import (
            FusedMoeWeightScaleSupported,
        )

        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size,
                dtype=torch.int8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition,
                dtype=torch.int8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        w13_scale = torch.nn.Parameter(
            torch.ones(
                num_experts,
                2 * intermediate_size_per_partition,
                1,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_scale)
        w2_scale = torch.nn.Parameter(
            torch.ones(num_experts, hidden_size, 1, dtype=torch.float32),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale", w2_scale)

        scale_attrs = dict(extra_weight_attrs)
        scale_attrs["quant_method"] = FusedMoeWeightScaleSupported.CHANNEL.value
        set_weight_attrs(w13_scale, scale_attrs)
        set_weight_attrs(w2_scale, scale_attrs)
        layer.w13_input_scale = None
        layer.w2_input_scale = None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        for name in (
            "w13_weight",
            "w2_weight",
            "w13_weight_scale",
            "w2_weight_scale",
        ):
            value = getattr(layer, name)
            setattr(
                layer,
                name,
                torch.nn.Parameter(value.data.contiguous(), requires_grad=False),
            )
        w13 = layer.w13_weight
        w2 = layer.w2_weight
        moe_block = None
        if _block_requant_enabled():
            for candidate in (_REQUANT_BLOCK, 64):
                if all(
                    dim % candidate == 0
                    for dim in (
                        w13.shape[-2],
                        w13.shape[-1],
                        w2.shape[-2],
                        w2.shape[-1],
                    )
                ):
                    moe_block = candidate
                    break
        if moe_block is not None:
            q13, s13 = _requant_channel_to_block(
                w13.data, layer.w13_weight_scale.data, moe_block
            )
            q2, s2 = _requant_channel_to_block(
                w2.data, layer.w2_weight_scale.data, moe_block
            )
            layer.w13_weight = torch.nn.Parameter(q13, requires_grad=False)
            layer.w2_weight = torch.nn.Parameter(q2, requires_grad=False)
            layer.w13_weight_scale = torch.nn.Parameter(
                s13, requires_grad=False
            )
            layer.w2_weight_scale = torch.nn.Parameter(s2, requires_grad=False)
            layer.block_shape = [moe_block, moe_block]
            layer._sunrise_channel_requant_block = True
            from ..moe.fused_moe_int8 import (
                moe_i_chunk_supported,
                reorder_moe_weights_for_i_chunks,
            )

            i_chunk = int(
                os.environ.get("SGLANG_FL_SUNRISE_MOE_I_CHUNK", "256")
            )
            intermediate = layer.w2_weight.shape[2]
            if moe_i_chunk_supported(
                intermediate,
                moe_block,
                moe_block,
                i_chunk,
            ):
                reorder_moe_weights_for_i_chunks(
                    layer.w13_weight.data,
                    layer.w2_weight.data,
                    layer.w13_weight_scale.data,
                    layer.w2_weight_scale.data,
                    i_chunk,
                )
                layer._sunrise_moe_i_chunk = i_chunk
            logger.info_once(
                "sunrise: compressed-tensors W8A8 experts requantized "
                "to %dx%d block INT8",
                moe_block,
                moe_block,
            )
        if moe_block is None:
            logger.info_once(
                "sunrise: non-aligned compressed-tensors W8A8 experts "
                "use native channelwise INT8 Triton MoE"
            )

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ) -> None:
        self.moe_runner_config = moe_runner_config

    def apply_weights(
        self,
        layer: torch.nn.Module,
        dispatch_output: "StandardDispatchOutput",
    ) -> "StandardCombineInput":
        from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import (
            fused_experts_impl,
        )
        from sglang.srt.layers.moe.token_dispatcher.standard import (
            StandardCombineInput,
        )

        from ..moe.fused_moe import _unpack_topk

        if getattr(layer, "_sunrise_channel_requant_block", False):
            from ..moe.fused_moe import fused_moe_sunrise

            return fused_moe_sunrise(self, layer, dispatch_output)

        topk_weights, topk_ids = _unpack_topk(dispatch_output)
        cfg = self.moe_runner_config
        from ..moe.channelwise_deepgemm_moe import try_fused

        routed_scale = getattr(cfg, "routed_scaling_factor", None)
        deepgemm_safe = (
            getattr(cfg, "activation", "silu") == "silu"
            and getattr(cfg, "is_gated", True)
            and not getattr(cfg, "apply_router_weight_on_input", False)
            and not getattr(cfg, "no_combine", False)
            and routed_scale in (None, 1.0)
            and getattr(cfg, "gemm1_alpha", None) is None
            and getattr(cfg, "gemm1_clamp_limit", None) is None
            and getattr(layer, "moe_ep_size", 1) == 1
        )
        output = (
            try_fused(
                dispatch_output.hidden_states,
                layer.w13_weight,
                layer.w2_weight,
                topk_weights,
                topk_ids,
                getattr(cfg, "inplace", False),
                layer.w13_weight_scale,
                layer.w2_weight_scale,
            )
            if deepgemm_safe
            else None
        )
        if output is not None:
            return StandardCombineInput(hidden_states=output)

        output = fused_experts_impl(
            dispatch_output.hidden_states,
            layer.w13_weight,
            layer.w2_weight,
            topk_weights,
            topk_ids,
            inplace=getattr(cfg, "inplace", False),
            activation=getattr(cfg, "activation", "silu"),
            is_gated=getattr(cfg, "is_gated", True),
            apply_router_weight_on_input=getattr(
                cfg, "apply_router_weight_on_input", False
            ),
            use_int8_w8a8=True,
            per_channel_quant=True,
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            no_combine=getattr(cfg, "no_combine", False),
            routed_scaling_factor=getattr(cfg, "routed_scaling_factor", None),
            gemm1_alpha=getattr(cfg, "gemm1_alpha", None),
            gemm1_limit=getattr(cfg, "gemm1_clamp_limit", None),
            filter_expert=getattr(layer, "moe_ep_size", 1) > 1,
        )
        return StandardCombineInput(hidden_states=output)

