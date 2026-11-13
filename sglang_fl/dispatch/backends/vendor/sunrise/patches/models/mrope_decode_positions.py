# Copyright (c) 2026 BAAI. All rights reserved.
"""Build text-only decode M-RoPE positions on device.

``ForwardBatch._compute_mrope_positions`` builds one CPU tensor per request
and uploads the concatenation with ``.to(device, non_blocking=True)``. On
PTPU every H2D copy blocks until the current (forward) stream drains, so
under the overlap scheduler ``ForwardBatch.init_new`` waits for the whole
previous decode step. For text-only decode the result is ``seq_lens - 1`` on
all three sections, which ``ForwardBatch.positions`` already holds on device.
"""

import logging

logger = logging.getLogger(__name__)

_patched = False


def patch_mrope_decode_positions_for_sunrise() -> None:
    global _patched
    if _patched:
        return

    import torch

    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

    orig = ForwardBatch._compute_mrope_positions

    def _compute_mrope_positions(self, model_runner, batch):
        pos = self.positions
        if (
            self.forward_mode.is_decode()
            and pos is not None
            and pos.device.type == "ptpu"
            and pos.shape[0] == self.seq_lens_cpu.shape[0]
            and all(mm is None for mm in batch.multimodal_inputs)
        ):
            self.mrope_positions = pos.to(torch.int64).unsqueeze(0).expand(3, -1).contiguous()
            return
        return orig(self, model_runner, batch)

    ForwardBatch._compute_mrope_positions = _compute_mrope_positions
    _patched = True
    logger.info("sunrise: text-only decode M-RoPE positions built on device")
