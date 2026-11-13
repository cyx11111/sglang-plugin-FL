# Copyright (c) 2026 BAAI. All rights reserved.
"""Host-to-device copies that do not wait for the forward stream (PTPU).

torch_ptpu has no pinned host memory: an H2D copy runs synchronously on the
issuing stream, so issued behind queued kernels (``ForwardBatch.init_new``
inside the overlap scheduler's ``run_batch`` runs on the forward stream) it
blocks the host until they finish: ~0.4 s behind a prefill forward, ~43 ms
behind an MTP verify forward. Issued on the idle schedule stream it returns
in a few ms, and the current stream waits for it on the device.

The copy stream is the scheduler's schedule stream (registered by
``patches/async_h2d``). A fresh ``torch.ptpu.Stream()`` is not used:
PTPU hands out streams round-robin from a pool of eight, so a late-created
stream can alias the busy forward stream.

Destinations come from the schedule stream's pool but are read on other
streams, and ``record_stream`` has no PTPU kernel: freed too early, a block is
handed to the next schedule-stream allocation and overwritten while the
forward still reads it (index-out-of-bounds scatters at TP4 serve, hangs at
TP2). Copies are therefore grouped per outermost :func:`async_h2d_scope`
(one ``init_new``); every later scope entry records a fence on the current
and the forward stream, and a group is released once two later fences have
completed, i.e. after the forward that used it. A count-based window is not
enough: M-RoPE decode copies one delta per request. The copy events are kept
in the same group: destroying a PTPU event while a stream still has a queued
wait on it can leave that stream stuck (TP ranks then hang in all-reduce).
"""

from __future__ import annotations

import threading
from collections import deque
from contextlib import contextmanager

import torch

_FENCES_TO_RELEASE = 2
_MAX_PENDING = 512
_gen: list = []
# [tensors, fences seen, latest fence events]
_pending: deque = deque()
_copy_stream = None
_fence_streams: list = []
_tls = threading.local()
_orig_to = None


def set_copy_stream(stream, fence_streams=()) -> None:
    """``fence_streams``: streams that consume the copies besides the stream that
    issued ``init_new`` (the overlap scheduler's forward stream)."""
    global _copy_stream, _fence_streams
    _copy_stream = stream
    _fence_streams = list(fence_streams)


def _sid(stream):
    return getattr(stream, "ptpu_stream", id(stream))


def _close_generation() -> None:
    global _gen
    streams, seen = [], set()
    for s in [torch.ptpu.current_stream()] + _fence_streams:
        if s is not None and _sid(s) not in seen:
            seen.add(_sid(s))
            streams.append(s)
    fence = []
    for s in streams:
        ev = torch.ptpu.Event()
        ev.record(s)
        fence.append(ev)
    if _gen:
        _pending.append([_gen, 0, None])
        _gen = []
    for item in _pending:
        if item[1] < _FENCES_TO_RELEASE:
            item[1] += 1
            item[2] = fence
    while _pending:
        tensors, n, evs = _pending[0]
        if n >= _FENCES_TO_RELEASE and all(ev.query() for ev in evs):
            _pending.popleft()
        elif len(_pending) > _MAX_PENDING:
            for ev in evs:
                ev.synchronize()
            _pending.popleft()
        else:
            break


def async_h2d(t: torch.Tensor, device) -> torch.Tensor:
    """``t.to(device, non_blocking=True)`` issued on the copy stream; the current stream
    waits for it on the device. Falls back to a plain copy without a copy stream."""
    device = torch.device(device)
    to = _orig_to or torch.Tensor.to
    stream = _copy_stream
    cur = torch.ptpu.current_stream()
    if stream is None or _sid(stream) == _sid(cur):
        return to(t, device, non_blocking=True)
    with torch.ptpu.stream(stream):
        out = to(t, device, non_blocking=True)
        ev = torch.ptpu.Event()
        ev.record(stream)
    cur.wait_event(ev)
    # The event must outlive the queued wait as well.
    _gen.append((out, ev))
    return out


def _ptpu_target(args, kwargs):
    tgt = args[0] if args else kwargs.get("device")
    if isinstance(tgt, torch.device):
        return tgt if tgt.type == "ptpu" else None
    if isinstance(tgt, str) and tgt.split(":")[0] == "ptpu":
        return torch.device(tgt)
    return None


def install_tensor_to_hook() -> None:
    """Route CPU -> PTPU ``.to(device[, dtype])`` calls inside :func:`async_h2d_scope`.

    Blocking copies are routed too: a pageable H2D source is consumed before the
    call returns either way, and later work on the current stream waits for
    the copy event."""
    global _orig_to
    if _orig_to is not None:
        return
    _orig_to = torch.Tensor.to

    def to(self, *args, **kwargs):
        if getattr(_tls, "on", False) and self.device.type == "cpu":
            dev = _ptpu_target(args, kwargs)
            if dev is not None and set(kwargs) <= {"device", "dtype", "non_blocking"}:
                dtype = kwargs.get("dtype")
                if len(args) == 2 and isinstance(args[1], torch.dtype) and dtype is None:
                    dtype = args[1]
                elif len(args) > 1:
                    return _orig_to(self, *args, **kwargs)
                return async_h2d(self if dtype is None else _orig_to(self, dtype), dev)
        return _orig_to(self, *args, **kwargs)

    torch.Tensor.to = to


@contextmanager
def async_h2d_scope():
    prev = getattr(_tls, "on", False)
    if not prev and _copy_stream is not None:
        _close_generation()
    _tls.on = True
    try:
        yield
    finally:
        _tls.on = prev
