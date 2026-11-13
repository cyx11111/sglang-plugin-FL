# Copyright (c) 2026 BAAI. All rights reserved.

"""Sunrise monkey-patches on sglang internals — entrypoint.

Auto-imported by ``sglang_fl.load_plugin()`` (see ``_apply_vendor_patches``)
only when FlagGems DeviceDetector vendor is ``sunrise``. One thin
``patch_xxx`` wrapper per concern; implementations live under ``patches/``.

The package ``__init__`` already applied the ``sgl_kernel`` stub and the
spawn preload at import time (scheduler subprocesses need them before
``load_plugins()`` runs).
"""

import logging

logger = logging.getLogger(__name__)
_patches_applied = False
_load_patches_applied = False


def apply_sunrise_patches():
    """Device registration, torch op fixes, quantization routing."""
    global _patches_applied
    if _patches_applied:
        return
    _patches_applied = True

    patch_supported_devices()
    patch_cumsum_empty()
    patch_bf16_moe_weight_layout()
    patch_vocab_embedding_eager()
    patch_compressed_tensors()
    patch_blockwise_int8()


def apply_sunrise_load_patches():
    """Patches on sglang layers / processors used during model load and run."""
    global _load_patches_applied
    if _load_patches_applied:
        return
    _load_patches_applied = True

    # Must run first: serve imports memory_pool_host -> NVIDIA sgl_kernel.
    patch_sgl_kernel_stub()
    patch_rotary_embedding_init()
    patch_layernorm_gated()
    patch_mm_image_processor_device()
    patch_vision_rotary_eager()
    patch_moe_shared_expert_gate()
    patch_causal_conv1d_update()
    patch_async_d2h()
    patch_mrope_decode_positions()
    patch_flagcx_allreduce_inplace()
    patch_gdn_qkvz_split()
    patch_gdn_prefill_fused()
    patch_spec_decode()
    patch_async_h2d()
    patch_tang_pool_trim()
    patch_mamba_slot_clear()
    patch_host_sync_hints()
    patch_minimax_m2()
    patch_moe_ep()


def patch_moe_ep() -> None:
    # --ep-size: the standard MoE dispatcher's device lookup does not know PTPU.
    from .patches.models.moe_ep import patch_moe_ep_for_sunrise

    patch_moe_ep_for_sunrise()


def patch_minimax_m2() -> None:
    # MiniMax-M2 TP QK RMSNorm Triton helpers assert is_cuda.
    from .patches.models.minimax_m2 import patch_minimax_m2_for_sunrise

    patch_minimax_m2_for_sunrise()


def patch_host_sync_hints() -> None:
    # M-RoPE row check / GDN chunk lengths from host batch info instead of device syncs.
    from .patches.runtime.host_sync_hints import patch_host_sync_hints_for_sunrise

    patch_host_sync_hints_for_sunrise()


def patch_tang_pool_trim() -> None:
    from .patches.runtime.allocator import patch_tang_pool_trim_for_sunrise

    patch_tang_pool_trim_for_sunrise()


def patch_supported_devices() -> None:
    from .patches.runtime.supported_devices import patch_supported_devices as _patch

    _patch()


def patch_cumsum_empty() -> None:
    from .patches.runtime.cumsum import patch_cumsum_empty_for_sunrise

    patch_cumsum_empty_for_sunrise()


def patch_bf16_moe_weight_layout() -> None:
    """Prepare unquantized expert weights for graph-safe FlagGems group_mm."""
    from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod

    method = UnquantizedFusedMoEMethod.process_weights_after_loading
    if getattr(method, "_sunrise_bf16_group_mm", False):
        return

    from .impl.moe.fused_moe_bf16 import prepare_bf16_moe_weights

    def process_weights_after_loading(self, layer):
        method(self, layer)
        prepare_bf16_moe_weights(layer)

    process_weights_after_loading._sunrise_bf16_group_mm = True
    UnquantizedFusedMoEMethod.process_weights_after_loading = (
        process_weights_after_loading
    )


def patch_vocab_embedding_eager() -> None:
    # Vocab mask helper is torch.compile'd (disable=_is_npu only).
    # TP>1 on PTPU hits Dynamo FakeTensor failure on bool |.
    from .patches.models.vocab_embedding import (
        patch_vocab_embedding_eager_for_sunrise,
    )

    patch_vocab_embedding_eager_for_sunrise()


def patch_blockwise_int8() -> None:
    # blockwise_int8 checkpoints run on PTPU INT8 kernels.
    from .patches.quantization.compressed_tensors import (
        patch_blockwise_int8_for_sunrise,
    )

    patch_blockwise_int8_for_sunrise()


def patch_compressed_tensors() -> None:
    # FlagOS-Compressor channelwise W8A8 checkpoints: PTPU-safe loader and
    # correctness bridge. Other compressed-tensors schemes remain upstream.
    from .patches.quantization.compressed_tensors import (
        patch_compressed_tensors_for_sunrise,
    )

    patch_compressed_tensors_for_sunrise()


def patch_sgl_kernel_stub() -> None:
    from .patches.runtime.sgl_kernel_stub import patch_sgl_kernel_stub_for_sunrise

    patch_sgl_kernel_stub_for_sunrise()


def patch_rotary_embedding_init() -> None:
    from .patches.models.rotary_embedding_init import (
        patch_rotary_embedding_init_sunrise,
    )

    patch_rotary_embedding_init_sunrise()


def patch_layernorm_gated() -> None:
    # FLA gated RMSNorm.
    from .patches.models.layernorm_gated import patch_layernorm_gated_for_sunrise

    patch_layernorm_gated_for_sunrise()


def patch_mm_image_processor_device() -> None:
    # Fast image processor: force cpu/ptpu.
    from .patches.models.mm_processor import (
        patch_mm_image_processor_device as _patch,
    )

    _patch()


def patch_vision_rotary_eager() -> None:
    # ViT apply_rotary_pos_emb is torch.compile'd; inductor has no PTPU.
    from .patches.models.vision_rotary import (
        patch_vision_rotary_eager_for_sunrise,
    )

    patch_vision_rotary_eager_for_sunrise()


def patch_moe_shared_expert_gate() -> None:
    from .patches.models.moe_shared_expert_gate import (
        patch_moe_shared_expert_gate_for_sunrise,
    )

    patch_moe_shared_expert_gate_for_sunrise()


def patch_causal_conv1d_update() -> None:
    from .patches.models.causal_conv1d import (
        patch_causal_conv1d_update_for_sunrise,
    )

    patch_causal_conv1d_update_for_sunrise()


def patch_async_d2h() -> None:
    # Overlap scheduler: non_blocking D2H is synchronous on PTPU (no pinned memory).
    from .patches.runtime.async_h2d import patch_async_d2h_for_sunrise

    patch_async_d2h_for_sunrise()


def patch_mrope_decode_positions() -> None:
    # Decode M-RoPE positions: skip the per-step H2D (blocks on PTPU).
    from .patches.models.mrope_decode_positions import (
        patch_mrope_decode_positions_for_sunrise,
    )

    patch_mrope_decode_positions_for_sunrise()


def patch_flagcx_allreduce_inplace() -> None:
    from .patches.runtime.flagcx_allreduce_inplace import (
        patch_flagcx_allreduce_inplace_for_sunrise,
    )

    patch_flagcx_allreduce_inplace_for_sunrise()


def patch_gdn_qkvz_split() -> None:
    # Triton split kernel is ~5x slower than torch slice copies on PTPU.
    from .patches.models.gdn_prefill_fused import (
        patch_gdn_qkvz_split_for_sunrise,
    )

    patch_gdn_qkvz_split_for_sunrise()


def patch_gdn_prefill_fused() -> None:
    from .patches.models.gdn_prefill_fused import (
        patch_gdn_prefill_fused_for_sunrise,
    )

    patch_gdn_prefill_fused_for_sunrise()


def patch_async_h2d() -> None:
    # ForwardBatch metadata H2D / mamba index writes queued behind the forward stream.
    from .patches.runtime.async_h2d import patch_async_h2d_for_sunrise

    patch_async_h2d_for_sunrise()


def patch_mamba_slot_clear() -> None:
    # MambaPool.alloc's index_put clear: ~134 ms of device time per request on PTPU.
    from .patches.models.mamba_slot_clear import patch_mamba_slot_clear_for_sunrise

    patch_mamba_slot_clear_for_sunrise()


def patch_spec_decode() -> None:
    # MTP / EAGLE chains: GDN verify, tree ops, mamba verify buffers, draft backend.
    from .patches.models.spec_decode import patch_spec_decode_for_sunrise

    patch_spec_decode_for_sunrise()


apply_sunrise_patches()
apply_sunrise_load_patches()
