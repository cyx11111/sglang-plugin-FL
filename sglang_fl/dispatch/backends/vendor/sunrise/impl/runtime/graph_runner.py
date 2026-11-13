# Copyright (c) 2026 BAAI. All rights reserved.
"""PTPU graph runner — ``torch.cuda.CUDAGraph`` is a dummy on this device.

SGLang's default ``CudaGraphRunner._create_device_graph`` instantiates
``torch.cuda.CUDAGraph()``. On a CUDA-free PTPU build that class is the
PyTorch dummy base and raises::

    RuntimeError: Tried to instantiate dummy base class CUDAGraph

The real API (same as vllm-plugin-FL) is ``torch.ptpu.PTPUGraph`` plus the
``torch.ptpu.graph(...)`` capture context.

``torch.ptpu.graph`` / ``torch.ptpu.stream`` compare ``stream.device`` on
enter. Some torch_ptpu builds and SGLang-created streams only expose
``device_index``, which yields::

    AttributeError: 'Stream' object has no attribute 'device'
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Callable, Optional

import torch

from sglang.srt.model_executor.cuda_graph_runner import CudaGraphRunner

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner

_stream_device_patched = False

def _ptpu_graph_cls():
    ptpu = torch.ptpu
    for name in ("PTPUGraph", "CUDAGraph"):
        cls = getattr(ptpu, name, None)
        if cls is not None:
            return cls
    raise RuntimeError(
        "torch.ptpu has neither PTPUGraph nor CUDAGraph. "
        "Upgrade torch_ptpu, or start with --disable-cuda-graph."
    )


def _stream_device_index(stream) -> int:
    idx = getattr(stream, "device_index", None)
    if callable(idx):
        try:
            idx = idx()
        except Exception:
            idx = None
    if idx is None:
        idx = torch.ptpu.current_device()
    return int(idx)


def _ensure_ptpu_stream_device_attr() -> None:
    """Make ``Stream.device`` available for torch.ptpu.graph StreamContext."""
    global _stream_device_patched
    if _stream_device_patched:
        return
    stream_cls = getattr(torch.ptpu, "Stream", None)
    if stream_cls is not None and not hasattr(stream_cls, "device"):

        def _device(self):
            return torch.device("ptpu", _stream_device_index(self))

        try:
            stream_cls.device = property(_device)  # type: ignore[attr-defined]
            logger.info(
                "PtpuGraphRunner: patched torch.ptpu.Stream.device "
                "(torch_ptpu graph requires it)"
            )
        except Exception as exc:
            logger.warning(
                "PtpuGraphRunner: could not patch Stream.device: %s", exc
            )
    _stream_device_patched = True


def _ensure_stream_has_device(stream) -> None:
    """Instance-level fallback when class patch is unavailable."""
    if stream is None or hasattr(stream, "device"):
        return
    device = torch.device("ptpu", _stream_device_index(stream))
    try:
        object.__setattr__(stream, "device", device)
    except Exception:
        try:
            stream.__dict__["device"] = device
        except Exception:
            pass


_capture_side_stream = None


def _ptpu_capture_stream(preferred) -> Optional[object]:
    """Dedicated side stream for ``torch.ptpu.graph`` (must expose ``.device``).

    Never the current stream: torch_ptpu then hands the graph's private-pool
    blocks back to later eager allocations, and every replay overwrites those
    tensors (e.g. ``ScheduleBatch.seq_lens``). SGLang's stream is used only if
    it is a real PTPU stream distinct from the current one.
    """
    global _capture_side_stream
    _ensure_ptpu_stream_device_attr()

    cur = torch.ptpu.current_stream()
    stream_cls = getattr(torch.ptpu, "Stream", None)
    if (
        preferred is not None
        and stream_cls is not None
        and isinstance(preferred, stream_cls)
        and preferred != cur
    ):
        _ensure_stream_has_device(preferred)
        if hasattr(preferred, "device"):
            return preferred

    if _capture_side_stream is None:
        _capture_side_stream = torch.ptpu.Stream()
        _ensure_stream_has_device(_capture_side_stream)
    return _capture_side_stream


class PtpuGraphRunner(CudaGraphRunner):
    """CudaGraphRunner that captures with ``torch.ptpu.PTPUGraph``."""

    def __init__(self, model_runner: ModelRunner):
        # Parent binds ``self.device_module = torch.get_device_module(device)``
        # and later calls ``graph_pool_handle()`` during capture. On PTPU that
        # lookup can still return ``torch.cuda`` (dummy pool / graph). Force
        # the real module for the duration of parent ``__init__``.
        saved = torch.get_device_module

        def _ptpu_module(*_args, **_kwargs):
            return torch.ptpu

        torch.get_device_module = _ptpu_module  # type: ignore[assignment]
        try:
            _ensure_ptpu_stream_device_attr()
            super().__init__(model_runner)
        finally:
            torch.get_device_module = saved
        self.device_module = torch.ptpu
        # TP>1 FlagCX collectives are recorded into the PTPU graph. vLLM's
        # Sunrise cudagraph patch syncs the device after every replay so
        # FlagCX progress is pumped; without any host wait on the step every
        # rank blocks on the first decode step. With the async result D2H
        # (patch_async_d2h) the copy worker's event polling is that wait and
        # the sync would serialize host and device again, so it defaults off.
        from ...patches.runtime.async_h2d import async_d2h_enabled

        tp = int(getattr(model_runner, "tp_size", 1) or 1)
        pp = int(getattr(model_runner, "pp_size", 1) or 1)
        dist = os.environ.get("SGLANG_FL_DIST_BACKEND", "").strip().lower()
        if not dist and os.environ.get("FLAGCX_PATH", "").strip():
            dist = "flagcx"
        sync_default = "0" if async_d2h_enabled() else "1"
        self._flagcx_replay_sync = (
            (tp * pp) > 1
            and dist == "flagcx"
            and os.environ.get("SGLANG_FL_SUNRISE_GRAPH_REPLAY_SYNC", sync_default)
            != "0"
        )
        if self._flagcx_replay_sync:
            logger.info(
                "PtpuGraphRunner: device sync after each graph replay "
                "(FlagCX TP=%d PP=%d)",
                tp,
                pp,
            )

    def replay(self, *args, **kwargs):
        output = super().replay(*args, **kwargs)
        if self._flagcx_replay_sync:
            torch.ptpu.synchronize()
        return output

    def can_run(self, forward_batch):
        """Skip graph replay when a FlagGems gather cap would truncate KV.

        Paged FlagGems (vLLM-style ``block_table`` + ``seqused_k``) does not
        gather, so ``flaggems_graph_kv_budget`` is 0 and this is a no-op.
        """
        if not super().can_run(forward_batch):
            return False
        budget = self._flaggems_graph_kv_budget(int(forward_batch.batch_size))
        if budget <= 0:
            return True
        n_kv = getattr(forward_batch, "seq_lens_sum", None)
        if n_kv is None:
            cpu = getattr(forward_batch, "seq_lens_cpu", None)
            if cpu is None:
                return True
            n_kv = int(cpu.sum().item()) if hasattr(cpu, "sum") else int(sum(cpu))
        if int(n_kv) > budget:
            return False
        return True

    def _flaggems_graph_kv_budget(self, bs: int) -> int:
        attn = getattr(self.model_runner, "attn_backend", None)
        seen = []
        while attn is not None and attn not in seen:
            seen.append(attn)
            fn = getattr(attn, "flaggems_graph_kv_budget", None)
            if callable(fn):
                return int(fn(bs))
            attn = getattr(attn, "full_attn_backend", None) or getattr(
                attn, "attn_backend", None
            )
        return 0

    def _create_device_graph(self):
        return _ptpu_graph_cls()()

    def _capture_graph(self, graph, pool, stream, run_once_fn: Callable):
        return _ptpu_capture(graph, pool, stream, run_once_fn)


def _ptpu_capture(graph, pool, stream, run_once_fn: Callable):
    # Positional graph object — torch.ptpu.graph does not take cuda_graph=.
    # Do not forward SGLang's raw stream: StreamContext compares
    # ``_src_prev_stream.device`` and crashes if either side lacks it.
    capture_stream = _ptpu_capture_stream(stream)
    prev = torch.ptpu.current_stream()
    _ensure_stream_has_device(prev)
    with torch.ptpu.graph(graph, pool=pool, stream=capture_stream):
        return run_once_fn()


def ptpu_draft_extend_graph_runner_cls():
    """EAGLE v2 draft-extend (MTP layer) graph runner on ``torch.ptpu`` graphs."""
    from sglang.srt.speculative.eagle_draft_extend_cuda_graph_runner import (
        EAGLEDraftExtendCudaGraphRunner,
    )

    return _ptpu_eagle_runner_cls(EAGLEDraftExtendCudaGraphRunner, "PtpuEAGLEDraftExtendGraphRunner")


def ptpu_draft_graph_runner_cls():
    """EAGLE multi-step draft decode graph runner on ``torch.ptpu`` graphs."""
    from sglang.srt.speculative.eagle_draft_cuda_graph_runner import EAGLEDraftCudaGraphRunner

    return _ptpu_eagle_runner_cls(EAGLEDraftCudaGraphRunner, "PtpuEAGLEDraftGraphRunner")


def _ptpu_eagle_runner_cls(base, name):
    class Runner(base):
        def __init__(self, eagle_worker, **kwargs):
            # Capture runs inside the parent __init__ (see PtpuGraphRunner).
            saved = torch.get_device_module
            torch.get_device_module = lambda *_a, **_k: torch.ptpu  # type: ignore[assignment]
            try:
                _ensure_ptpu_stream_device_attr()
                super().__init__(eagle_worker, **kwargs)
            finally:
                torch.get_device_module = saved

        def _create_graph(self):
            return _ptpu_graph_cls()()

        def _capture_init(self, run_once_fn):
            for _ in range(2):
                torch.ptpu.synchronize()
                self.model_runner.tp_group.barrier()
                run_once_fn()

        def _capture_graph(self, graph, pool, stream, run_once_fn):
            return _ptpu_capture(graph, pool, stream, run_once_fn)

    Runner.__name__ = Runner.__qualname__ = name
    return Runner
