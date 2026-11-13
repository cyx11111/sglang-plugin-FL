# Copyright (c) 2026 BAAI. All rights reserved.
"""Clear new mamba pool slots with a Triton kernel on PTPU.

``MambaPool.alloc`` zeroes the conv / SSM state of each new slot with
``state[:, idx] = 0``; PTPU's index_put takes 134 ms of device time per
request for the Qwen3.6 TP1 state (61 MB), 0.25 ms with
``impl.mamba_slots.zero_slots``. The clear runs on the schedule stream and the
next forward waits for it, so it added ~540 ms to every 4-request prefill
batch.

Disable with ``SGLANG_FL_SUNRISE_MAMBA_ZERO=0``.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_patched = False


def patch_mamba_slot_clear_for_sunrise() -> None:
    global _patched
    if _patched or os.environ.get("SGLANG_FL_SUNRISE_MAMBA_ZERO", "1") == "0":
        return

    from sglang.srt.mem_cache.memory_pool import MambaPool

    from ...impl.fla import mamba_slots

    orig_alloc = MambaPool.alloc

    def alloc(self, need_size: int):
        cache = getattr(self, "mamba_cache", None)
        free = self.free_slots
        if cache is None or need_size > len(free):
            return orig_alloc(self, need_size)
        states = list(cache.conv) + [cache.temporal]
        if not all(mamba_slots.zero_slots_supported(t, free) for t in states):
            return orig_alloc(self, need_size)
        select_index = free[:need_size]
        self.free_slots = free[need_size:]
        for t in states:
            mamba_slots.zero_slots(t, select_index)
        return select_index

    MambaPool.alloc = alloc
    _patched = True
    logger.info("sunrise: mamba pool slots cleared with a Triton kernel")
