# Copyright (c) 2026 BAAI. All rights reserved.
"""GDN decode ``causal_conv1d_update`` on the PTPU kernel.

sglang's Triton update kernel takes ~0.35 ms per GDN layer at bs=96 on PTPU;
``llm_kernel_ptpu.causal_conv1d_update`` takes ~0.012 ms with identical state
updates. MTP target verify (``intermediate_conv_window`` without a tree) runs
``impl/fla_gdn_verify.conv_verify``; other speculative-decode arguments fall
back to Triton. Prefill ``causal_conv1d_fn`` stays on Triton (the PTPU version
is ~60x slower for gdn_backend's transposed input).

Disable with ``SGLANG_FL_SUNRISE_CONV1D_UPDATE=0``.
"""

import logging
import os
import sys

logger = logging.getLogger(__name__)

_patched = False


def patch_causal_conv1d_update_for_sunrise() -> None:
    global _patched
    if _patched or os.environ.get("SGLANG_FL_SUNRISE_CONV1D_UPDATE", "1") == "0":
        return
    try:
        from llm_kernel_ptpu import causal_conv1d_update as ptpu_update
    except Exception as exc:
        logger.info("sunrise: PTPU causal_conv1d_update unavailable (%s)", exc)
        return

    import torch

    from sglang.srt.layers.attention.mamba import causal_conv1d_triton

    from ...impl.fla import fla_gdn_verify as gdn_verify

    triton_update = causal_conv1d_triton.causal_conv1d_update
    spec_verify = os.environ.get("SGLANG_FL_SUNRISE_SPEC", "1") != "0"

    def causal_conv1d_update(
        x,
        conv_state,
        weight,
        bias=None,
        activation=None,
        cache_seqlens=None,
        conv_state_indices=None,
        num_accepted_tokens=None,
        intermediate_conv_window=None,
        intermediate_state_indices=None,
        retrieve_next_token=None,
        retrieve_next_sibling=None,
        retrieve_parent_token=None,
        pad_slot_id=causal_conv1d_triton.PAD_SLOT_ID,
        metadata=None,
        validate_data=False,
    ):
        if (
            spec_verify
            and intermediate_conv_window is not None
            and num_accepted_tokens is None
            and retrieve_next_token is None
            and conv_state_indices is not None
            and intermediate_state_indices is not None
            and cache_seqlens is None
            and gdn_verify.conv_verify_supported(x, conv_state, weight, intermediate_conv_window)
        ):
            return gdn_verify.conv_verify(
                x, conv_state, weight, bias, activation, conv_state_indices,
                intermediate_conv_window, intermediate_state_indices,
            )
        if (
            num_accepted_tokens is not None
            or intermediate_conv_window is not None
            or retrieve_next_token is not None
            or x.device.type != "ptpu"
        ):
            return triton_update(
                x, conv_state, weight, bias, activation, cache_seqlens,
                conv_state_indices, num_accepted_tokens, intermediate_conv_window,
                intermediate_state_indices, retrieve_next_token, retrieve_next_sibling,
                retrieve_parent_token, pad_slot_id, metadata, validate_data,
            )
        if conv_state_indices is not None and conv_state_indices.dtype != torch.int32:
            conv_state_indices = conv_state_indices.to(torch.int32)
        return ptpu_update(
            x, conv_state, weight, bias, activation, cache_seqlens,
            conv_state_indices, pad_slot_id,
        )

    causal_conv1d_triton.causal_conv1d_update = causal_conv1d_update
    for mod_name in (
        "sglang.srt.layers.attention.linear.gdn_backend",
        "sglang.srt.layers.attention.mamba.causal_conv1d",
    ):
        mod = sys.modules.get(mod_name)
        if mod is None:
            continue
        for attr in ("causal_conv1d_update", "_causal_conv1d_update_triton"):
            if getattr(mod, attr, None) is triton_update:
                setattr(mod, attr, causal_conv1d_update)
    _patched = True
    logger.info("sunrise: GDN decode causal_conv1d_update -> llm_kernel_ptpu")
