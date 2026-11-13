# Copyright (c) 2026 BAAI. All rights reserved.
"""Keep scheduler-side host/device copies from stalling on PTPU.

H2D: two places waited for the device under the overlap scheduler (see
``impl/h2d.py`` for why PTPU H2D copies block):

* ``ForwardBatch.init_new`` converts metadata lists with ``.to(device)``.
  Inside ``run_batch`` that copy queues behind the previous forward on the
  forward stream: MTP's draft prefill waited ~475 ms for the target forward
  per 8192-token chunk, the draft extend ~43 ms for the verify forward per
  decode step. The copies go through ``impl.h2d.async_h2d`` on the schedule
  stream (registered in ``Scheduler.run_event_loop``). TP1 / PP1 only:
  with TP>1 serve traffic this ended in PCCL all-reduce hardware exceptions
  (``SGLANG_FL_SUNRISE_ASYNC_H2D_INIT_NEW=0`` turns it off at TP1 too).
* ``HybridReqToTokenPool.alloc`` writes ``mapping[select_index]`` with a
  Python list index, which waits for the queued work (~0.15-0.5 s per
  prefill batch); with the index copied to the device first the write takes
  0.1 ms. New mamba slots are taken in one ``MambaPool.alloc`` call (one
  slot clear per batch, see ``patch_mamba_slot_clear``).

Disable with ``SGLANG_FL_SUNRISE_ASYNC_H2D=0``.

D2H: ``GenerationBatchResult.copy_to_cpu`` is moved to a worker thread because
torch_ptpu has no pinned host memory. Disable with
``SGLANG_FL_SUNRISE_ASYNC_D2H=0``.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor

logger = logging.getLogger(__name__)

_h2d_patched = False
_d2h_patched = False
_D2H_POLL_SECONDS = 0.0002
_d2h_executor = None
_d2h_executor_lock = threading.Lock()
_d2h_tls = threading.local()


def async_d2h_enabled() -> bool:
    return os.environ.get("SGLANG_FL_SUNRISE_ASYNC_D2H", "1") != "0"


class _PendingCopy:
    """Stand-in for a device event backed by the asynchronous copy future."""

    __slots__ = ("_future",)

    def __init__(self, future: Future):
        self._future = future

    def record(self, *args, **kwargs):
        pass

    def query(self) -> bool:
        return self._future.done()

    def synchronize(self):
        self._future.result()

    def wait(self, stream=None):
        self._future.result()


def _get_d2h_executor() -> ThreadPoolExecutor:
    global _d2h_executor
    if _d2h_executor is None:
        with _d2h_executor_lock:
            if _d2h_executor is None:
                _d2h_executor = ThreadPoolExecutor(
                    1, thread_name_prefix="sunrise-d2h"
                )
    return _d2h_executor


def _d2h_side_stream(torch, device: int):
    streams = getattr(_d2h_tls, "streams", None)
    if streams is None:
        streams = _d2h_tls.streams = {}
    stream = streams.get(device)
    if stream is None:
        torch.ptpu.set_device(device)
        stream = streams[device] = torch.ptpu.Stream()
        if os.environ.get("SGLANG_FL_SUNRISE_STREAM_DEBUG", "0") == "1":
            logger.warning(
                "sunrise streams: async D2H side stream %s",
                getattr(stream, "ptpu_stream", None),
            )
    return stream


def patch_async_d2h_for_sunrise() -> None:
    global _d2h_patched
    if _d2h_patched or not async_d2h_enabled():
        return

    import torch

    from sglang.srt.managers.utils import GenerationBatchResult

    original = GenerationBatchResult.copy_to_cpu

    def _copy_job(result, return_logprob, ready, device, future):
        try:
            while not ready.query():
                time.sleep(_D2H_POLL_SECONDS)
            with torch.ptpu.stream(_d2h_side_stream(torch, device)):
                original(result, return_logprob)
            future.set_result(None)
        except BaseException as exc:
            future.set_exception(exc)

    def copy_to_cpu(self, return_logprob: bool):
        next_token_ids = self.next_token_ids
        if (
            not isinstance(next_token_ids, torch.Tensor)
            or next_token_ids.device.type != "ptpu"
        ):
            return original(self, return_logprob)
        logits_output = self.logits_output
        if logits_output is not None and logits_output.hidden_states is not None:
            logits_output.hidden_states = logits_output.hidden_states.clone()
        ready = torch.ptpu.Event()
        ready.record()
        future = Future()
        self.copy_done = _PendingCopy(future)
        device = next_token_ids.device.index
        if device is None:
            device = torch.ptpu.current_device()
        _get_d2h_executor().submit(
            _copy_job, self, return_logprob, ready, device, future
        )

    GenerationBatchResult.copy_to_cpu = copy_to_cpu
    _d2h_patched = True
    logger.info("sunrise: overlap-scheduler result D2H moved to a worker thread")


def patch_async_h2d_for_sunrise() -> None:
    global _h2d_patched
    if (
        _h2d_patched
        or os.environ.get("SGLANG_FL_SUNRISE_ASYNC_H2D", "1") == "0"
    ):
        return

    import torch

    from sglang.srt.managers import scheduler as sched_mod
    from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool, ReqToTokenPool
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

    from ...impl.runtime import h2d

    h2d.install_tensor_to_hook()

    init_new = ForwardBatch.init_new  # bound classmethod (or a profiler's wrapper)

    init_new_on = os.environ.get("SGLANG_FL_SUNRISE_ASYNC_H2D_INIT_NEW", "1") != "0"

    def init_new_async_h2d(batch, model_runner):
        if (
            not init_new_on
            or str(getattr(model_runner, "device", "")).split(":")[0] != "ptpu"
            # With collectives in the forward, schedule-stream copies overlapping it
            # end in PCCL all-reduce CU / UMM exceptions or index-out-of-bounds
            # scatters (TP4 / TP2 serve); TP1 only.
            or getattr(model_runner, "tp_size", 1) > 1
            or getattr(model_runner, "pp_size", 1) > 1
        ):
            return init_new(batch, model_runner)
        with h2d.async_h2d_scope():
            return init_new(batch, model_runner)

    ForwardBatch.init_new = staticmethod(init_new_async_h2d)

    orig_loop = sched_mod.Scheduler.run_event_loop

    def run_event_loop(self):
        if str(self.device).split(":")[0] != "ptpu" or sched_mod.use_mlx():
            return orig_loop(self)
        # Same as upstream, plus registering the schedule stream as the H2D copy stream.
        self.schedule_stream = self.device_module.Stream(priority=0)
        h2d.set_copy_stream(self.schedule_stream, [getattr(self, "forward_stream", None)])
        if os.environ.get("SGLANG_FL_SUNRISE_STREAM_DEBUG", "0") == "1":
            ids = {n: getattr(getattr(self, n, None), "ptpu_stream", None)
                   for n in ("forward_stream", "copy_stream", "schedule_stream")}
            logger.warning("sunrise streams: %s", ids)
        with self.device_module.StreamContext(self.schedule_stream):
            sched_mod.dispatch_event_loop(self)

    sched_mod.Scheduler.run_event_loop = run_event_loop

    orig_alloc = HybridReqToTokenPool.alloc

    def alloc(self, reqs):
        mapping = self.req_index_to_mamba_index_mapping
        if self.enable_mamba_extra_buffer or mapping.device.type != "ptpu":
            return orig_alloc(self, reqs)
        select_index = ReqToTokenPool.alloc(self, reqs)
        if select_index is None:
            return None
        new_reqs = [req for req in reqs if req.mamba_pool_idx is None]
        if new_reqs:
            mids = self.mamba_pool.alloc(len(new_reqs))
            assert mids is not None, (
                "Not enough space for mamba cache, try to increase --mamba-full-memory-ratio "
                f"or --max-mamba-cache-size. {self.mamba_pool.available_size()=}, {len(reqs)=}"
            )
            for j, req in enumerate(new_reqs):
                req.mamba_pool_idx = mids[j]
        mamba_indices = [req.mamba_pool_idx for req in reqs]
        assert len(select_index) == len(mamba_indices)
        idx = torch.tensor(select_index, dtype=torch.int64).to(mapping.device, non_blocking=True)
        mapping[idx] = torch.stack(mamba_indices).to(dtype=torch.int32)
        return select_index

    HybridReqToTokenPool.alloc = alloc
    _h2d_patched = True
    logger.info("sunrise: ForwardBatch metadata H2D and mamba index writes no longer wait for the forward stream")
