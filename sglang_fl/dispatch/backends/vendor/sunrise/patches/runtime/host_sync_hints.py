# Copyright (c) 2026 BAAI. All rights reserved.
"""Hand host-side batch facts to the kernels that would otherwise sync for them.

Two device syncs ran in every prefill forward (17-53 ms each behind the
previous forward; at TP>1 they hold back the launch of the next batch):

* M-RoPE ``_needs_native_interleaved`` checks with ``.item()`` whether the
  t/h/w position rows differ. They cannot for a batch without multimodal
  inputs, which ``ForwardBatch`` knows when it builds the positions.
* GDN chunk tables read the per-sequence lengths with ``.tolist()``; the
  scheduler already has them as ``extend_seq_lens_cpu``.

Hints are bound to the exact tensor object and version, so any other tensor
falls back to the device sync. Disable with ``SGLANG_FL_SUNRISE_HOST_HINTS=0``.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_patched = False


def patch_host_sync_hints_for_sunrise() -> None:
    global _patched
    if _patched or os.environ.get("SGLANG_FL_SUNRISE_HOST_HINTS", "1") == "0":
        return

    from sglang.srt.layers.attention.hybrid_linear_attn_backend import MambaAttnBackendBase
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

    from ...impl.attention import mrotary_embedding
    from ...impl.fla import fla_chunk_meta as _chunk_meta

    def _text_only(batch) -> bool:
        mm = getattr(batch, "multimodal_inputs", None)
        return mm is None or all(m is None for m in mm)

    def _wrap_mrope(name):
        orig = getattr(ForwardBatch, name, None)
        if orig is None:
            return

        def wrapped(self, model_runner, batch):
            out = orig(self, model_runner, batch)
            pos = getattr(self, "mrope_positions", None)
            if pos is not None and pos.device.type == "ptpu" and _text_only(batch):
                mrotary_embedding.mark_rows_equal(pos)
            return out

        setattr(ForwardBatch, name, wrapped)

    _wrap_mrope("_compute_mrope_positions")
    _wrap_mrope("_compute_spec_mrope_positions")

    orig_meta = MambaAttnBackendBase._forward_metadata

    def _forward_metadata(self, forward_batch):
        md = orig_meta(self, forward_batch)
        qsl = getattr(md, "query_start_loc", None)
        mode = forward_batch.forward_mode
        lens = getattr(forward_batch, "extend_seq_lens_cpu", None)
        if (
            qsl is not None
            and qsl.device.type == "ptpu"
            and lens is not None
            and mode.is_extend()
            and not mode.is_target_verify()
            and not mode.is_draft_extend(include_v2=True)
        ):
            _chunk_meta.register_host_lens(qsl, lens)
        return md

    MambaAttnBackendBase._forward_metadata = _forward_metadata
    _patched = True
    logger.info("sunrise: host hints for M-RoPE row check and GDN chunk lengths")
