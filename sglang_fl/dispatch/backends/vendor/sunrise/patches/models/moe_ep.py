# Copyright (c) 2026 BAAI. All rights reserved.
"""Expert parallelism (``--ep-size``) with SGLang's standard MoE dispatcher on PTPU.

``StandardDispatcher.dispatch`` builds the global -> local expert map on
``get_device()``, which only knows CUDA / XPU / HPU / NPU / MUSA / MPS and
raises on PTPU. The dispatcher module gets a ``get_device`` that falls back to
the current PTPU device.

With EP, graph replay also copies ``num_token_non_padded`` (0-dim) into its
``[1]`` buffer; PTPU's ``torch._foreach_copy_`` does not broadcast, so
same-sized sources are reshaped to the destination first.
"""

import logging

import torch

logger = logging.getLogger(__name__)

_patched = False


def patch_moe_ep_for_sunrise() -> None:
    global _patched
    if _patched:
        return
    try:
        from sglang.srt.layers.moe.token_dispatcher import standard
    except Exception as exc:
        logger.info("sunrise: MoE standard dispatcher unavailable, EP not patched (%s)", exc)
        return
    inner = standard.get_device

    def get_device(device_id=None):
        try:
            return inner(device_id)
        except RuntimeError:
            idx = torch.ptpu.current_device() if device_id is None else device_id
            return f"ptpu:{idx}"

    standard.get_device = get_device

    from sglang.srt.model_executor import cuda_graph_runner

    copy_inner = cuda_graph_runner._grouped_foreach_copy_

    def _grouped_foreach_copy_(dsts, srcs):
        srcs = [
            s.reshape(d.shape) if s.shape != d.shape and s.numel() == d.numel() else s
            for d, s in zip(dsts, srcs)
        ]
        return copy_inner(dsts, srcs)

    cuda_graph_runner._grouped_foreach_copy_ = _grouped_foreach_copy_
    _patched = True
    logger.info("sunrise: MoE dispatcher local expert map on the PTPU device (EP)")
