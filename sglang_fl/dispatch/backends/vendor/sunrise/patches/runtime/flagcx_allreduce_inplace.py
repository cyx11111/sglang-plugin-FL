# Copyright (c) 2026 BAAI. All rights reserved.
"""In-place FlagCX all-reduce on PTPU.

``FlagCXCommunicator.all_reduce`` reduces into a fresh tensor and copies it
back into the input, and ``CommunicatorFL.all_reduce`` copies again for
in-place semantics. Sunrise FlagCX accepts sendbuff == recvbuff, so reduce in
place: one ``copy_`` less per all-reduce (~81 per forward on
Qwen3.6-35B-A3B TP=4).

Routing graph-captured all-reduces through torch PCCL instead was measured
slower end to end (decode 44.9 vs 43.6 ms/step at bs=96) although PCCL wins
in isolation (0.108 vs 0.134 ms per [96, 2048] call inside a graph). Rechecked
with ``PCCL_PROTO=simple`` on MiniMax-M2.7 TP8 (separate PCCL groups created
next to SGLang's): in a graph 0.128 vs 0.154 ms per [64, 3072] bf16 call, but
end to end decode bs=64 only 65.5 -> 64.7 ms/step and prefill 13.7k -> 12.0k
tok/s, so FlagCX stays.

FlagCX 0.13's PTPU adaptor is not safe for the large in-place collectives used
by the TP-sharded vision encoder (for example ``[4096, 1152]`` BF16). Keep
separate send/receive buffers by default; larger tensors additionally use the
chunked path. Set ``SGLANG_FL_SUNRISE_FLAGCX_INPLACE_MAX_BYTES`` to a positive
byte limit only when opting small tensors back into the measured fast path.
``SGLANG_FL_SUNRISE_FLAGCX_MAX_ALLREDUCE_BYTES`` controls the chunk size
(default 1 MiB, 0 disables chunking).
"""

import logging
import os

logger = logging.getLogger(__name__)

_patched = False
_MAX_INPLACE_BYTES = int(
    os.environ.get("SGLANG_FL_SUNRISE_FLAGCX_INPLACE_MAX_BYTES", "0")
)
_MAX_ALLREDUCE_BYTES = int(
    os.environ.get(
        "SGLANG_FL_SUNRISE_FLAGCX_MAX_ALLREDUCE_BYTES",
        str(1024 * 1024),
    )
)


def _install_sunrise_flagcx_communicator() -> None:
    """Register PTPU device/stream handling without changing common FlagCX code."""
    import torch

    from sglang_fl.distributed.device_communicators import flagcx as flagcx_mod

    base = flagcx_mod.FlagCXCommunicator
    if getattr(base, "_sunrise_ptpu", False):
        return

    class SunriseFlagCXCommunicator(base):
        _sunrise_ptpu = True

        def __init__(self, group, device):
            # The common initializer handles every other part of FlagCX setup,
            # but older versions route unknown devices through torch.cuda.device.
            cuda_device = torch.cuda.device

            def device_context(value):
                if torch.device(value).type == "ptpu":
                    return torch.ptpu.device(value)
                return cuda_device(value)

            torch.cuda.device = device_context
            try:
                super().__init__(group=group, device=device)
            finally:
                torch.cuda.device = cuda_device

        def _get_raw_stream_handle(self):
            stream = torch.ptpu.current_stream()
            for attr in ("ptpu_stream", "cuda_stream", "tang_stream"):
                ptr = getattr(stream, attr, None)
                if ptr is not None:
                    return ptr
            stream_fn = getattr(stream, "stream", None)
            if callable(stream_fn):
                ptr = stream_fn()
                if ptr is not None:
                    return ptr
            raise AttributeError(
                f"Unsupported PTPU stream object {stream!r}: expected "
                "`ptpu_stream`, `cuda_stream`, `tang_stream`, or callable stream()"
            )

        def _sync_current_stream(self):
            torch.ptpu.current_stream().synchronize()

    flagcx_mod.FlagCXCommunicator = SunriseFlagCXCommunicator


def patch_flagcx_allreduce_inplace_for_sunrise() -> None:
    global _patched
    if _patched:
        return

    import torch

    from sglang_fl.distributed.communicator import CommunicatorFL

    _install_sunrise_flagcx_communicator()
    orig = CommunicatorFL.all_reduce

    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        fc = self._flagcx_comm
        size_bytes = input_.numel() * input_.element_size()
        if (
            fc is not None
            and not fc.disabled
            and input_.is_contiguous()
            and _MAX_ALLREDUCE_BYTES > 0
            and size_bytes > _MAX_ALLREDUCE_BYTES
        ):
            flat = input_.view(-1)
            chunk_elems = max(
                1, _MAX_ALLREDUCE_BYTES // input_.element_size()
            )
            for start in range(0, flat.numel(), chunk_elems):
                chunk = flat[start : start + chunk_elems]
                chunk.copy_(fc._flagcx_all_reduce(chunk))
            return input_
        if (
            fc is None
            or fc.disabled
            or not input_.is_contiguous()
            or _MAX_INPLACE_BYTES <= 0
            or size_bytes > _MAX_INPLACE_BYTES
        ):
            return orig(self, input_)
        buf = fc._buffer_type(input_.data_ptr())
        fc.flagcx.flagcxAllReduce(
            buf,
            buf,
            input_.numel(),
            fc._dtype_enum.from_torch(input_.dtype),
            fc._redop_enum.from_torch(torch.distributed.ReduceOp.SUM),
            fc.comm,
            fc._get_stream(),
        )
        return input_

    CommunicatorFL.all_reduce = all_reduce
    _patched = True
    logger.info(
        "sunrise: FlagCX all-reduce chunk bytes=%d, in-place max bytes=%d "
        "(in-place 0 uses separate send/receive buffers)",
        _MAX_ALLREDUCE_BYTES,
        _MAX_INPLACE_BYTES,
    )
