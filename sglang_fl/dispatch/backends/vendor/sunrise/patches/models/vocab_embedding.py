# Copyright (c) 2026 BAAI. All rights reserved.
"""Force eager ``get_masked_input_and_mask`` on PTPU (needed when TP>1).

Why TP=1 works but TP>1 fails
-----------------------------
Installed SGLang variants differ slightly, but the idea is the same:

* ``tp_size == 1``: embed directly — never calls the mask helper.
* ``tp_size > 1``: calls ``get_masked_input_and_mask``, which is decorated::

      @torch.compile(dynamic=True, backend=..., disable=_is_npu)

  NPU is opted out; PTPU is not. Dynamo then fake-tensor-traces
  ``org_vocab_mask | added_vocab_mask`` and dies on custom kernels
  (FlagGems / PTPU), typically::

      Cannot access data pointer of Tensor (e.g. FakeTensor...)

``disable_cuda_graph`` does **not** disable this function-level compile.

Some SGLang versions call the helper from ``forward`` (line ~474); newer
ones go through ``_embed_local_shard``. We patch the module symbol and wrap
**both** entry points so every TP worker rebinds before the upstream body runs.
"""

from __future__ import annotations

import logging
import os
from typing import Tuple

import torch

logger = logging.getLogger(__name__)

_patched = False


def _should_patch() -> bool:
    force = os.environ.get("SGLANG_FL_FORCE_EAGER_VOCAB_MASK", "").strip().lower()
    if force in ("1", "true", "yes", "on"):
        return True
    try:
        return hasattr(torch, "ptpu") and torch.ptpu.is_available()
    except Exception:
        return hasattr(torch, "ptpu")


def get_masked_input_and_mask_eager(
    input_: torch.Tensor,
    org_vocab_start_index: int,
    org_vocab_end_index: int,
    num_org_vocab_padding: int,
    added_vocab_start_index: int,
    added_vocab_end_index: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Same math as SGLang's helper, without ``torch.compile``."""
    org_vocab_mask = (input_ >= org_vocab_start_index) & (input_ < org_vocab_end_index)
    added_vocab_mask = (input_ >= added_vocab_start_index) & (
        input_ < added_vocab_end_index
    )
    added_offset = (
        added_vocab_start_index
        - (org_vocab_end_index - org_vocab_start_index)
        - num_org_vocab_padding
    )
    valid_offset = (org_vocab_start_index * org_vocab_mask) + (
        added_offset * added_vocab_mask
    )
    vocab_mask = org_vocab_mask | added_vocab_mask
    input_ = vocab_mask * (input_ - valid_offset)
    return input_, ~vocab_mask


def _rebind(vpe) -> None:
    vpe.get_masked_input_and_mask = get_masked_input_and_mask_eager


def patch_vocab_embedding_eager_for_sunrise() -> None:
    """Replace compiled mask helper; wrap forward / _embed_local_shard."""
    global _patched
    if _patched:
        return
    if not _should_patch():
        logger.debug("sunrise: skip vocab mask eager patch (not PTPU / not forced)")
        return

    try:
        import sglang.srt.layers.vocab_parallel_embedding as vpe
    except ImportError as exc:
        logger.warning(
            "sunrise: vocab_parallel_embedding import failed — skip: %s", exc
        )
        return

    if not hasattr(vpe, "get_masked_input_and_mask"):
        logger.warning("sunrise: get_masked_input_and_mask missing — skip")
        return

    _rebind(vpe)

    cls = getattr(vpe, "VocabParallelEmbedding", None)
    if cls is not None:
        # Older SGLang: forward() calls get_masked_input_and_mask directly (~L474).
        if hasattr(cls, "forward"):
            _orig_forward = cls.forward

            def _forward_ptpu(self, input_):
                _rebind(vpe)
                return _orig_forward(self, input_)

            cls.forward = _forward_ptpu

        # Newer SGLang: mask lives under _embed_local_shard.
        if hasattr(cls, "_embed_local_shard"):
            _orig_embed = cls._embed_local_shard

            def _embed_local_shard_ptpu(self, input_):
                _rebind(vpe)
                return _orig_embed(self, input_)

            cls._embed_local_shard = _embed_local_shard_ptpu

    # Confirm we no longer point at Dynamo's compile_wrapper.
    bound = vpe.get_masked_input_and_mask
    still_compiled = "compile" in type(bound).__name__.lower() or getattr(
        bound, "_torchdynamo_orig_callable", None
    )
    if still_compiled:
        logger.warning(
            "sunrise: get_masked_input_and_mask still looks compiled after rebind "
            "(type=%s) — set TORCHDYNAMO_DISABLE=1 as fallback",
            type(bound).__name__,
        )
    else:
        # warning: TP scheduler procs set logger to WARNING in load_plugin.
        logger.warning(
            "sunrise: patched get_masked_input_and_mask → local eager "
            "(TP>1 PTPU; wrapped VocabParallelEmbedding.forward)"
        )

    _patched = True
