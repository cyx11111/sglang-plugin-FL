# Copyright (c) 2026 BAAI. All rights reserved.
"""Patch SGLang multimodal image-processor device for PTPU / Sunrise.

Upstream ``BaseMultimodalProcessor.process_mm_data`` sets
``kwargs["device"] = "cuda"`` for any non-CPU / non-XPU / non-NPU platform.
On a CUDA-free PTPU build that becomes ``image.to("cuda")`` →
``AssertionError: Torch not compiled with CUDA enabled``.

Default: force the fast image processor onto **CPU** (torchvision preprocess
does not need the accelerator). Override with
``SGLANG_FL_MM_IMAGE_PROCESSOR_DEVICE=ptpu`` if desired.
"""

from __future__ import annotations

import logging
import os

import torch

logger = logging.getLogger(__name__)

_patched = False


def _resolve_mm_device() -> str:
    override = os.environ.get("SGLANG_FL_MM_IMAGE_PROCESSOR_DEVICE", "").strip()
    if override:
        return override
    # CPU is the safe default: HF BaseImageProcessorFast uses torchvision ops.
    return "cpu"


def patch_mm_image_processor_device() -> None:
    """Rewrite fast image-processor device away from ``cuda`` on PTPU. Idempotent."""
    global _patched
    if _patched:
        return

    try:
        if not (
            hasattr(torch, "ptpu")
            and torch.ptpu.is_available()
            and torch.ptpu.device_count() > 0
        ):
            return

        from sglang.srt.multimodal.processors.base_processor import (
            BaseMultimodalProcessor,
        )

        _orig = BaseMultimodalProcessor.process_mm_data
        mm_device = _resolve_mm_device()

        def _process_mm_data(self, input_text, images=None, videos=None, audios=None, **kwargs):
            # Upstream may inject device="cuda" inside; we also accept an
            # explicit override from the caller and normalize after return
            # is not enough — must fix kwargs before the HF processor runs.
            # Wrap by calling original after forcing device via a thin shim:
            # monkey-patch the branch by pre-setting and post-fixing kwargs
            # through a local wrapper of the HF call path.
            #
            # Implementation: run original but temporarily patch the module-
            # level device assignment by intercepting kwargs in a nested
            # wrapper around processor.__call__ if needed.
            #
            # Simplest reliable approach: set kwargs["device"] up-front; the
            # upstream block only assigns when Fast processor is active and
            # overwrites our value — so we must wrap the method body.
            return _process_mm_data_impl(
                self,
                _orig,
                mm_device,
                input_text,
                images=images,
                videos=videos,
                audios=audios,
                **kwargs,
            )

        BaseMultimodalProcessor.process_mm_data = _process_mm_data
        _patched = True
        logger.info(
            "sunrise: patched BaseMultimodalProcessor.process_mm_data "
            "image-processor device → %s (was cuda)",
            mm_device,
        )
    except Exception as exc:
        logger.warning(
            "sunrise: failed to patch multimodal image-processor device: %s",
            exc,
        )


def _process_mm_data_impl(self, orig, mm_device, input_text, images=None, videos=None, audios=None, **kwargs):
    """Call original ``process_mm_data`` but rewrite ``device=cuda`` → ``mm_device``."""
    # Upstream overwrites kwargs["device"] right before processor.__call__.
    # Intercept by wrapping self._processor.__call__ for this invocation.
    processor = getattr(self, "_processor", None)
    if processor is None:
        return orig(self, input_text, images=images, videos=videos, audios=audios, **kwargs)

    _orig_call = processor.__call__

    def _call_with_device_fix(*args, **kw):
        if kw.get("device") in (None, "cuda", "cuda:0"):
            kw["device"] = mm_device
        # Also handle nested images_kwargs used by some processors.
        img_kw = kw.get("images_kwargs")
        if isinstance(img_kw, dict) and img_kw.get("device") in (
            None,
            "cuda",
            "cuda:0",
        ):
            img_kw = dict(img_kw)
            img_kw["device"] = mm_device
            kw["images_kwargs"] = img_kw
        return _orig_call(*args, **kw)

    try:
        processor.__call__ = _call_with_device_fix  # type: ignore[method-assign]
        return orig(self, input_text, images=images, videos=videos, audios=audios, **kwargs)
    finally:
        processor.__call__ = _orig_call  # type: ignore[method-assign]
