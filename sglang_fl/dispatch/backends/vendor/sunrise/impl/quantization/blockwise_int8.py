# Copyright (c) 2026 BAAI. All rights reserved.
"""``blockwise_int8`` (W8A8, 128x128 weight blocks, dynamic per-token-group
activations) on Sunrise / PTPU.

Registered over sglang's ``BlockInt8Config`` by
``patches/blockwise_int8.py`` so checkpoints declaring
``quant_method: blockwise_int8`` pick it up without any CLI flag.

Differences from upstream ``BlockInt8Config``:
  * Honors ``modules_to_not_convert`` (used by HF-style block-INT8 exports)
    in addition to ``ignored_layers``, and resolves fused modules such as
    Qwen3.5's ``in_proj_ba`` through the model's ``packed_modules_mapping``.
  * Linear layers run ``impl/dense_gemm.py``.
  * MoE experts run ``impl/fused_moe.py`` (-> ``impl/fused_moe_int8.py``)
    instead of the Triton MoE runner.
  * Experts may use another block size than ``weight_block_size``: config key
    ``moe_weight_block_size``, env ``SGLANG_FL_SUNRISE_MOE_BLOCK_SIZE=64,64``,
    or read from the checkpoint's expert scale shapes.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import torch

from sglang.srt.layers.quantization.base_config import QuantizeMethodBase
from sglang.srt.layers.quantization.blockwise_int8 import (
    BlockInt8Config,
    BlockInt8LinearMethod,
    BlockInt8MoEMethod,
)
from sglang.srt.layers.quantization.unquant import (
    UnquantizedFusedMoEMethod,
    UnquantizedLinearMethod,
)
from sglang.srt.layers.quantization.utils import is_layer_skipped

logger = logging.getLogger(__name__)


def _normalize_ignored_layers(layers: Optional[List[str]]) -> List[str]:
    # Keep both "model." and bare variants so prefixes match regardless of
    # whether the model wraps its body in ``model.``.
    normalized: List[str] = []
    for layer in layers or []:
        base = layer.removeprefix("model.")
        normalized.append(base)
        normalized.append(f"model.{base}")
    return list(dict.fromkeys(normalized))


def _safetensors_shape(path: str, name: str) -> Optional[List[int]]:
    import json
    import struct

    with open(path, "rb") as f:
        header = json.loads(f.read(struct.unpack("<Q", f.read(8))[0]))
    entry = header.get(name)
    return entry["shape"] if entry else None


def _detect_expert_block(default: Optional[List[int]]) -> Optional[List[int]]:
    """Expert weight block size of the checkpoint being served, from the shape of
    one expert weight and its ``weight_scale_inv`` (safetensors headers only)."""
    import json
    import re

    try:
        from sglang.srt.server_args import get_global_server_args

        model_path = get_global_server_args().model_path
    except Exception:
        return default
    if not model_path or not os.path.isdir(model_path):
        return default
    index = os.path.join(model_path, "model.safetensors.index.json")
    try:
        if os.path.exists(index):
            weight_map = json.load(open(index))["weight_map"]
        else:
            single = os.path.join(model_path, "model.safetensors")
            if not os.path.exists(single):
                return default
            with open(single, "rb") as f:
                import struct

                names = json.loads(f.read(struct.unpack("<Q", f.read(8))[0]))
            weight_map = {k: "model.safetensors" for k in names if k != "__metadata__"}
        pattern = re.compile(r"\.experts\.\d+\.(w1|gate_proj)\.weight_scale_inv$")
        scale_name = next((k for k in weight_map if pattern.search(k)), None)
        if scale_name is None:
            return default
        weight_name = scale_name[: -len("_scale_inv")]
        w = _safetensors_shape(os.path.join(model_path, weight_map[weight_name]), weight_name)
        s = _safetensors_shape(os.path.join(model_path, weight_map[scale_name]), scale_name)
        if not w or not s or len(w) != 2 or len(s) != 2:
            return default
        block = []
        for size, blocks in zip(w, s):
            b = size // blocks
            if b <= 0 or -(-size // b) != blocks:
                return default
            block.append(b)
        return block
    except Exception as exc:
        logger.warning("sunrise: could not read the expert block size from %s (%s)", model_path, exc)
        return default


class SunriseBlockInt8Config(BlockInt8Config):
    def __init__(
        self,
        is_checkpoint_int8_serialized: bool = False,
        activation_scheme: str = "dynamic",
        ignored_layers: Optional[List[str]] = None,
        weight_block_size: List[int] = None,
        packed_modules_mapping: Optional[Dict[str, List[str]]] = None,
        moe_weight_block_size: Optional[List[int]] = None,
    ) -> None:
        super().__init__(
            is_checkpoint_int8_serialized=is_checkpoint_int8_serialized,
            activation_scheme=activation_scheme,
            ignored_layers=ignored_layers,
            weight_block_size=weight_block_size,
        )
        self.packed_modules_mapping = dict(packed_modules_mapping or {})
        self._moe_block = list(moe_weight_block_size) if moe_weight_block_size else None
        self._moe_config: Optional["SunriseBlockInt8Config"] = None

    @classmethod
    def get_min_capability(cls) -> int:
        return 0

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "SunriseBlockInt8Config":
        quant_method = cls.get_from_keys(config, ["quant_method"])
        activation_scheme = cls.get_from_keys(config, ["activation_scheme"])
        ignored_layers = cls.get_from_keys_or(
            config, ["ignored_layers", "modules_to_not_convert"], None
        )
        weight_block_size = cls.get_from_keys_or(config, ["weight_block_size"], None)
        packed_modules_mapping = (
            cls.get_from_keys_or(config, ["packed_modules_mapping"], {}) or {}
        )
        return cls(
            is_checkpoint_int8_serialized="int8" in quant_method,
            activation_scheme=activation_scheme,
            ignored_layers=_normalize_ignored_layers(ignored_layers),
            weight_block_size=weight_block_size,
            packed_modules_mapping=packed_modules_mapping,
            moe_weight_block_size=cls.get_from_keys_or(config, ["moe_weight_block_size"], None),
        )

    def moe_config(self) -> "SunriseBlockInt8Config":
        """Config for the MoE experts: same, except for the expert block size when the
        checkpoint quantizes experts with other blocks than ``weight_block_size``
        (MiniMax-M2.7 INT8: 64x64 experts, 128x128 attention, config says 128)."""
        if self._moe_config is None:
            block = self._moe_block
            if block is None:
                env = os.environ.get("SGLANG_FL_SUNRISE_MOE_BLOCK_SIZE", "").strip()
                block = [int(v) for v in env.split(",")] if env else _detect_expert_block(self.weight_block_size)
            if block and self.weight_block_size and list(block) != list(self.weight_block_size):
                import copy

                cfg = copy.copy(self)
                cfg.weight_block_size = list(block)
                cfg._moe_config = cfg
                logger.info("sunrise: block-INT8 MoE experts use %s weight blocks (dense layers %s)",
                            list(block), list(self.weight_block_size))
                self._moe_config = cfg
            else:
                self._moe_config = self
        return self._moe_config

    def _is_skipped(self, prefix: str) -> bool:
        return is_layer_skipped(
            prefix, self.ignored_layers, fused_mapping=self.packed_modules_mapping
        )

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional[QuantizeMethodBase]:
        from sglang.srt.layers.linear import LinearBase
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE

        if isinstance(layer, LinearBase):
            if self._is_skipped(prefix):
                return UnquantizedLinearMethod()
            return SunriseBlockInt8LinearMethod(self)
        if isinstance(layer, FusedMoE):
            if self._is_skipped(prefix):
                return UnquantizedFusedMoEMethod(
                    layer.use_triton_kernels, layer.use_flashinfer_trtllm_moe
                )
            return SunriseBlockInt8MoEMethod(self.moe_config())
        return None

    def apply_weight_name_mapper(self, hf_to_sglang_mapper) -> None:
        if self.ignored_layers:
            self.ignored_layers = list(
                dict.fromkeys(hf_to_sglang_mapper.apply_list(self.ignored_layers))
            )


class SunriseBlockInt8LinearMethod(BlockInt8LinearMethod):
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.weight = torch.nn.Parameter(
            layer.weight.data.contiguous(), requires_grad=False
        )
        layer.weight_scale_inv = torch.nn.Parameter(
            layer.weight_scale_inv.data.contiguous(), requires_grad=False
        )
        from .dense_gemm import tune_linear

        tune_linear(layer.weight, layer.weight_scale_inv, self.quant_config.weight_block_size)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        from .dense_gemm import apply_w8a8_block_int8_linear_sunrise

        return apply_w8a8_block_int8_linear_sunrise(
            input=x,
            weight=layer.weight,
            block_size=self.quant_config.weight_block_size,
            weight_scale=layer.weight_scale_inv,
            input_scale=None,
            bias=bias,
        )


class SunriseBlockInt8MoEMethod(BlockInt8MoEMethod):
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        from ..moe.fused_moe_int8 import (
            moe_i_chunk_supported,
            reorder_moe_weights_for_i_chunks,
        )

        super().process_weights_after_loading(layer)
        i_chunk = int(os.environ.get("SGLANG_FL_SUNRISE_MOE_I_CHUNK", "256"))
        intermediate = layer.w2_weight.shape[2]
        block_n, block_k = self.quant_config.weight_block_size
        if not moe_i_chunk_supported(intermediate, block_n, block_k, i_chunk):
            return
        reorder_moe_weights_for_i_chunks(
            layer.w13_weight.data,
            layer.w2_weight.data,
            layer.w13_weight_scale_inv.data,
            layer.w2_weight_scale_inv.data,
            i_chunk,
        )
        layer._sunrise_moe_i_chunk = i_chunk

    def create_moe_runner(self, layer: torch.nn.Module, moe_runner_config) -> None:
        # The PTPU path in impl/fused_moe.py replaces the Triton MoeRunner.
        self.moe_runner_config = moe_runner_config

    def apply(self, layer: torch.nn.Module, dispatch_output):
        from ..moe.fused_moe import fused_moe_sunrise

        return fused_moe_sunrise(self, layer, dispatch_output)
