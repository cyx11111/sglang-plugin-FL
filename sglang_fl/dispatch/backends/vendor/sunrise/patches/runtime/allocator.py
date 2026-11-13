# Copyright (c) 2026 BAAI. All rights reserved.

"""Release inactive TangRT 0.25 stream-pool segments after prefill.

TangRT 0.25 keeps every size returned by ``tangFreeAsync`` in a per-stream
cache.  Its exported release-threshold environment variables do not make the
cache trim, and ``torch.ptpu.empty_cache()`` is a no-op in the corresponding
torch_ptpu build.  Variable prefill token counts therefore retain one large
segment per observed shape.

The driver already contains the missing release sequence:

1. synchronize the stream (complete its allocation checkpoint);
2. ``TABlockPool::forceReleaseAsyncCache(stream)``;
3. ``TAdevice_s::trimTo(0)``.

The first method is not exported by driver 0.25, so this compatibility patch
calls its version-specific implementation after checking both the runtime
version and function byte signatures.  It is intentionally disabled on every
other driver version.
"""

from __future__ import annotations

import ctypes
import functools
import logging
import os
import threading
from typing import Iterable

logger = logging.getLogger(__name__)

_patched = False
_tuner_patched = False

_TANG_VERSION = 250
_CTX_GET_CURRENT_DEVICE_OFFSET = 0x7B8F0
_FORCE_RELEASE_ASYNC_CACHE_OFFSET = 0x6EAA0
_TRIM_TO_OFFSET = 0x99570
_CONTEXT_FROM_STREAM_OFFSET = 0x18
_DEVICE_FROM_CONTEXT_OFFSET = 0x18
_POOL_FROM_DEVICE_OFFSET = 0x88

_FORCE_RELEASE_SIGNATURE = bytes.fromhex("41 57 48 8d 87 50 01 00 00 41 56")
_TRIM_TO_SIGNATURE = bytes.fromhex("48 8b bf 88 00 00 00 e9")


class _TangPoolTrimmer:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._available = False
        self._force_release = None
        self._trim_to = None
        self._initialize()

    @staticmethod
    def _runtime_version(runtime, name: str) -> int:
        value = ctypes.c_int()
        fn = getattr(runtime, name)
        fn.argtypes = [ctypes.POINTER(ctypes.c_int)]
        fn.restype = ctypes.c_int
        if fn(ctypes.byref(value)) != 0:
            return -1
        return value.value

    def _initialize(self) -> None:
        try:
            runtime = ctypes.CDLL("libtangrt_shared.so")
            runtime_version = self._runtime_version(
                runtime, "tangRuntimeGetVersion"
            )
            driver_version = self._runtime_version(
                runtime, "tangDriverGetVersion"
            )
            if (runtime_version, driver_version) != (
                _TANG_VERSION,
                _TANG_VERSION,
            ):
                logger.info(
                    "sunrise: TangRT pool trim disabled for runtime/driver "
                    "version %s/%s",
                    runtime_version,
                    driver_version,
                )
                return

            driver = ctypes.CDLL("libtang.so.0")
            anchor = ctypes.cast(
                driver.taCtxGetCurrentDevice, ctypes.c_void_p
            ).value
            if not anchor:
                return
            base = anchor - _CTX_GET_CURRENT_DEVICE_OFFSET
            force_addr = base + _FORCE_RELEASE_ASYNC_CACHE_OFFSET
            trim_addr = base + _TRIM_TO_OFFSET
            if (
                ctypes.string_at(
                    force_addr, len(_FORCE_RELEASE_SIGNATURE)
                )
                != _FORCE_RELEASE_SIGNATURE
                or ctypes.string_at(trim_addr, len(_TRIM_TO_SIGNATURE))
                != _TRIM_TO_SIGNATURE
            ):
                logger.warning(
                    "sunrise: TangRT 0.25 pool trim disabled because the "
                    "driver function signatures do not match"
                )
                return

            self._force_release = ctypes.CFUNCTYPE(
                None, ctypes.c_void_p, ctypes.c_void_p
            )(force_addr)
            self._trim_to = ctypes.CFUNCTYPE(
                ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t
            )(trim_addr)
            self._available = True
        except (AttributeError, OSError, ValueError):
            logger.exception("sunrise: failed to initialize TangRT pool trim")

    @staticmethod
    def _stream_handle(stream) -> int:
        return int(getattr(stream, "ptpu_stream", 0) or 0)

    def trim(self, streams: Iterable[object]) -> int:
        if not self._available:
            return 0

        unique_streams = {}
        for stream in streams:
            handle = self._stream_handle(stream)
            if handle:
                unique_streams.setdefault(handle, stream)
        if not unique_streams:
            return 0

        with self._lock:
            for stream in unique_streams.values():
                stream.synchronize()

            first_handle = next(iter(unique_streams))
            context = ctypes.c_void_p.from_address(
                first_handle + _CONTEXT_FROM_STREAM_OFFSET
            ).value
            if not context:
                return 0
            device = ctypes.c_void_p.from_address(
                context + _DEVICE_FROM_CONTEXT_OFFSET
            ).value
            if not device:
                return 0
            pool = ctypes.c_void_p.from_address(
                device + _POOL_FROM_DEVICE_OFFSET
            ).value
            if not pool:
                return 0

            for handle in unique_streams:
                self._force_release(pool, handle)
            return int(self._trim_to(device, 0))


def _patch_flaggems_tuner_tensor_refs() -> None:
    global _tuner_patched
    if _tuner_patched:
        return

    from flag_gems.utils.libentry import LibTuner

    original = LibTuner.run

    @functools.wraps(original)
    def run_without_retained_tensors(self, *args, **kwargs):
        try:
            return original(self, *args, **kwargs)
        finally:
            # These fields exist for the offline benchmark_config() helper.
            # A serving process never consumes them after run(), while keeping
            # them pins the last shape's PTPU tensors in every tuner instance.
            self._last_benchmark_args = None
            self._last_benchmark_meta = None
            self.nargs = {}

    LibTuner.run = run_without_retained_tensors
    _tuner_patched = True


def patch_tang_pool_trim_for_sunrise() -> None:
    global _patched
    if (
        _patched
        or os.environ.get("SGLANG_FL_SUNRISE_TRIM_POOL", "1") == "0"
    ):
        return

    import torch

    from sglang.srt.managers import scheduler as scheduler_mod

    _patch_flaggems_tuner_tensor_refs()
    trimmer = _TangPoolTrimmer()
    if not trimmer._available:
        return

    original = scheduler_mod.Scheduler.process_batch_result

    def scheduler_streams(scheduler):
        streams = [
            getattr(scheduler, name, None)
            for name in ("schedule_stream", "copy_stream")
        ]
        for worker_name in ("tp_worker", "model_worker"):
            worker = getattr(scheduler, worker_name, None)
            if worker is None:
                continue
            runners = list(getattr(worker, "model_runner_list", ()) or ())
            runner = getattr(worker, "model_runner", None)
            if runner is not None:
                runners.append(runner)
            streams.extend(
                getattr(item, "forward_stream", None) for item in runners
            )
        streams.append(torch.ptpu.current_stream())
        return (stream for stream in streams if stream is not None)

    def process_batch_result(self, batch, result):
        output = original(self, batch, result)
        if batch.forward_mode.is_extend():
            released = trimmer.trim(scheduler_streams(self))
            if released:
                logger.debug(
                    "sunrise: released %.2f MiB from TangRT stream pool",
                    released / (1 << 20),
                )
        return output

    scheduler_mod.Scheduler.process_batch_result = process_batch_result
    _patched = True
    logger.info(
        "sunrise: enabled TangRT 0.25 stream-pool trim after prefill"
    )
