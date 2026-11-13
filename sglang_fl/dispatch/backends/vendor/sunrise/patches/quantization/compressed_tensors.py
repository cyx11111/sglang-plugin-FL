# Copyright (c) 2026 BAAI. All rights reserved.
"""Enable compressed-tensors channelwise W8A8 checkpoints on Sunrise.

Upstream checks CUDA compute capability even on an out-of-tree device, and
only exposes fused-MoE W8A8 on NPU.  Replace only the dynamic-token,
channel-weight INT8 schemes. Other compressed-tensors schemes keep their
upstream selection.
"""

from __future__ import annotations

import logging

from compressed_tensors.quantization import QuantizationStrategy

logger = logging.getLogger(__name__)
_compressed_tensors_patched = False


def patch_compressed_tensors_for_sunrise() -> None:
    global _compressed_tensors_patched
    if _compressed_tensors_patched:
        return

    from sglang.srt.layers.quantization.compressed_tensors import (
        compressed_tensors as ct,
    )
    from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_int8 import (
        CompressedTensorsW8A8Int8,
    )

    from ...impl.quantization.compressed_tensors_w8a8 import (
        SunriseCompressedTensorsW8A8Linear,
        SunriseCompressedTensorsW8A8MoE,
    )
    from ...impl.quantization.channelwise_int8 import per_token_quant_int8
    from ...impl.moe import channelwise_moe
    from ...impl.moe.moe_align import moe_align_block_size_sunrise

    # SGLang's stock quantizer calls CUDA libdevice.round, which cannot be
    # compiled by TANG. The MoE module imports it by value, so patch both the
    # owner and the already-bound module global.
    from sglang.srt.layers.quantization import int8_kernel
    from sglang.srt.layers.moe.moe_runner.triton_utils import (
        fused_moe,
        fused_moe_triton_kernels,
        fused_moe_triton_config,
        moe_align_block_size,
    )

    int8_kernel.per_token_quant_int8 = per_token_quant_int8
    fused_moe_triton_kernels.per_token_quant_int8 = per_token_quant_int8
    moe_align_block_size.moe_align_block_size = moe_align_block_size_sunrise
    fused_moe.moe_align_block_size = moe_align_block_size_sunrise
    # The generic Triton MoE config lookup assumes CUDA/ROCm and receives
    # ``None`` from SGLang's device-name helper on PTPU. A stable name makes it
    # fall back to the built-in shape heuristics when no tuned JSON exists.
    fused_moe_triton_config.get_device_name = lambda: "sunrise_ptpu"
    orig_default_moe_config = fused_moe_triton_config.get_default_config

    def get_default_config(*args, **kwargs):
        m = kwargs.get("M", args[0] if args else 0)
        experts = kwargs.get("E", args[1] if len(args) > 1 else 0)
        dtype = kwargs.get("dtype", args[5] if len(args) > 5 else None)
        block_shape = kwargs.get(
            "block_shape", args[7] if len(args) > 7 else None
        )
        if dtype == "int8_w8a8" and block_shape is None and m <= experts:
            # PTPU INT8 MMA reaches rate at K=128. The upstream generic INT8
            # default uses K=64 and N=32, doubling both K-loop work and launch
            # count for Qwen's 256-expert, I/TP=128 decode shapes.
            return {
                "BLOCK_SIZE_M": 16,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 1,
                "num_warps": 8,
                "num_stages": 2,
            }
        if dtype == "int8_w8a8" and block_shape is None:
            return {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": 64,
                "BLOCK_SIZE_K": 128,
                "GROUP_SIZE_M": 8,
                "num_warps": 8,
                "num_stages": 2,
            }
        return orig_default_moe_config(*args, **kwargs)

    fused_moe_triton_config.get_default_config = get_default_config

    orig_invoke_moe = fused_moe.invoke_fused_moe_kernel

    logged_moe_dispatch = False

    def invoke_fused_moe_kernel(*args, **kwargs):
        nonlocal logged_moe_dispatch
        (
            a,
            weight,
            bias,
            out,
            a_scale,
            weight_scale,
            _weight_zp,
            topk_weights,
            _topk_ids,
            sorted_ids,
            expert_ids,
            num_post_pad,
            mul_weight,
            top_k,
            config,
        ) = args[:15]
        try:
            from sglang.srt.model_executor.cuda_graph_runner import (
                get_is_capture_mode,
            )

            capturing = bool(get_is_capture_mode())
        except Exception:
            capturing = False
        custom_a, custom_a_scale = a, a_scale
        if (
            not capturing
            and
            kwargs.get("use_int8_w8a8", False)
            and kwargs.get("per_channel_quant", False)
            and a_scale is None
        ):
            custom_a, custom_a_scale = per_token_quant_int8(a)
        use_channelwise = channelwise_moe.can_run(
            custom_a,
            weight,
            out,
            custom_a_scale,
            weight_scale,
            sorted_ids,
            top_k,
            config,
            kwargs.get("use_fp8_w8a8", False),
            kwargs.get("use_int8_w8a8", False),
            kwargs.get("use_int8_w8a16", False),
            kwargs.get("use_int4_w4a16", False),
            kwargs.get("per_channel_quant", False),
            kwargs.get("block_shape"),
            bias,
            kwargs.get("a_use_tma", False),
            kwargs.get("b_use_tma", False),
            kwargs.get("fuse_sum_all_reduce", False),
            kwargs.get("fuse_add_to_output", False),
        )
        if kwargs.get("use_int8_w8a8", False) and not logged_moe_dispatch:
            logged_moe_dispatch = True
            logger.info(
                "sunrise channelwise MoE dispatch: custom=%s A=%s AS=%s "
                "B=%s BS=%s block_shape=%s config=%s",
                use_channelwise,
                tuple(custom_a.shape),
                None
                if custom_a_scale is None
                else tuple(custom_a_scale.shape),
                tuple(weight.shape),
                None if weight_scale is None else tuple(weight_scale.shape),
                kwargs.get("block_shape"),
                config,
            )
        if use_channelwise:
            channelwise_moe.invoke(
                custom_a,
                weight,
                out,
                custom_a_scale,
                weight_scale,
                topk_weights,
                sorted_ids,
                expert_ids,
                num_post_pad,
                mul_weight,
                top_k,
                config,
            )
            return None
        return orig_invoke_moe(*args, **kwargs)

    fused_moe.invoke_fused_moe_kernel = invoke_fused_moe_kernel

    config_cls = ct.CompressedTensorsConfig
    orig_check = config_cls._check_scheme_supported
    orig_linear = config_cls.get_linear_scheme
    orig_moe = config_cls.get_moe_scheme

    def _check_scheme_supported(self, min_capability: int, error: bool = True):
        # The upstream helper unconditionally calls torch.cuda even though
        # Sunrise is an out-of-tree device. Only bypass it when every configured
        # quantized target uses the channel/token W8A8 scheme implemented here;
        # do not accidentally admit FP8, NVFP4, WNA16, or a mixed checkpoint.
        supported = bool(self.target_scheme_map) and all(
            scheme.get("weights") is not None
            and scheme.get("input_activations") is not None
            and self._is_dynamic_token_w8a8(
                scheme["weights"], scheme["input_activations"]
            )
            for scheme in self.target_scheme_map.values()
        )
        if error and not supported:
            raise RuntimeError(
                "sunrise compressed-tensors currently supports only "
                "channel-weight, dynamic-token W8A8 checkpoints"
            )
        return supported

    def get_linear_scheme(self, layer, layer_name=None):
        scheme = orig_linear(self, layer, layer_name)
        if isinstance(scheme, CompressedTensorsW8A8Int8):
            scheme_dict = self.get_scheme_dict(layer, layer_name)
            weight_quant = scheme_dict.get("weights")
            input_quant = scheme_dict.get("input_activations")
            if self._is_dynamic_token_w8a8(weight_quant, input_quant):
                return SunriseCompressedTensorsW8A8Linear(
                    weight_quant, input_quant
                )
        return scheme

    def get_moe_scheme(self, layer, layer_name=None):
        # Mirror upstream's fused-name resolution only for the one scheme that
        # upstream rejects on non-NPU platforms.
        self._add_fused_moe_to_target_scheme_map()
        names = [
            layer_name + suffix
            for suffix in (".0.gate_proj", ".0.up_proj", ".0.down_proj")
        ]
        scheme_dicts = [self.get_scheme_dict(layer, name) for name in names]
        scheme_dict = scheme_dicts[0] if scheme_dicts else None
        if not all(item == scheme_dict for item in scheme_dicts):
            raise ValueError(
                "All MoE projections need the same quantization scheme"
            )
        if scheme_dict is not None:
            weight_quant = scheme_dict.get("weights")
            input_quant = scheme_dict.get("input_activations")
            if (
                weight_quant is not None
                and input_quant is not None
                and weight_quant.strategy == QuantizationStrategy.CHANNEL
                and self._is_dynamic_token_w8a8(weight_quant, input_quant)
            ):
                return SunriseCompressedTensorsW8A8MoE(
                    weight_quant, input_quant
                )
        return orig_moe(self, layer, layer_name)

    config_cls._check_scheme_supported = _check_scheme_supported
    config_cls.get_linear_scheme = get_linear_scheme
    config_cls.get_moe_scheme = get_moe_scheme
    config_cls._sunrise_original_check_scheme_supported = orig_check
    _compressed_tensors_patched = True
    logger.info(
        "sunrise: enabled native compressed-tensors channelwise W8A8"
    )

