# Copyright (c) 2026 BAAI. All rights reserved.
"""Sunrise / PTPU AttentionBackend for SGLang full-attention layers.

Priority for full-attn compute:
  1. PTPU ``decode_attention_fwd`` / ``prefix_caching_fwd`` when
     ``qk_head_dim == v_head_dim == 128`` (paged: ``kv_indptr`` + ``kv_indices``)
  1b. Extend with ``qk_head_dim == v_head_dim == 256``: PTPU
     ``paged_prefill_attention_fwd`` (same paged metadata; handles prefixes).
  2. FlagGems ``flash_attn_varlen_func`` with paged KV, only for layers the
     PTPU kernels above do not cover: pass the token pool as ``k``/``v``,
     ``block_table`` from ``req_to_token``, and ``seqused_k`` from
     ``seq_lens``. SGLang KV is token-addressed, so each token is a page
     (``block_size=1``): ``block_table[b, t]`` is the pool index. KV is viewed
     as ``(num_tokens, 1, nheads_k, headdim)`` so FlagGems paged FA sees
     ``page_size=1``. The FlagGems ``k_page_stride`` patch and tile clamp are
     applied only when such a layer exists.
  3. Dense ``k_cache[idx]`` gather only if paged FlagGems rejects the kwargs
  4. ``TorchNativeAttnBackend`` (SDPA) last resort

KV write stays on SGLang ``token_to_kv_pool.set_kv_buffer`` (token slots).
That is the analog of vLLM ``reshape_and_cache_flash`` for this pool layout.

Not yet: sliding-window KV, speculative trees, MLA / NSA, encoder-decoder.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.torch_native_backend import TorchNativeAttnBackend
from sglang.srt.layers.dp_attention import get_attention_tp_size
from sglang.srt.layers.radix_attention import AttentionType

from .paged_decode import HEAD_DIMS as PAGED_DECODE_HEAD_DIMS
from .paged_decode import PagedDecode
from .paged_decode import load_kernel as load_paged_decode_kernel
from .paged_decode import measure_capacity as measure_paged_decode_capacity

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.speculative.spec_info import SpecInput

logger = logging.getLogger(__name__)

# PTPU sgl MHA kernels are specialized for this head size.
_PTPU_QK_HEAD_DIM = 128
# Head sizes served by PTPU ``paged_prefill_attention_fwd`` for extend.
_PTPU_PAGED_PREFILL_HEAD_DIMS = (256,)
_fallback_logged = False
_flaggems_paged_logged = False


def _try_import_flash_attn_varlen():
    try:
        from flag_gems import flash_attn_varlen_func
    except Exception:
        return None
    return flash_attn_varlen_func


def _prepare_flaggems_flash() -> None:
    """Global FlagGems flash tweaks; applied only when a layer falls back to it.

    Both mutate FlagGems module state process-wide: the ``k_page_stride``
    kernel patch and the BLOCK=16/16 tile clamp.
    """
    try:
        from sglang_fl.dispatch.backends.vendor.sunrise.patches.models.flaggems_paged_flash import (
            patch_flaggems_paged_flash,
        )

        patch_flaggems_paged_flash()
    except Exception as exc:
        logger.debug("FlagGems paged-flash k_page_stride patch skipped: %s", exc)
    _maybe_clamp_flaggems_flash_tile()


def _maybe_clamp_flaggems_flash_tile() -> None:
    """Best-effort: pin FlagGems sunrise flash tiles to a known-safe config."""
    try:
        from flag_gems import runtime as gems_runtime
        from flag_gems.runtime.backend._sunrise.heuristics_config_utils import (
            HEURISTICS_CONFIGS,
        )
    except Exception:
        return
    overrides = {
        "BLOCK_M": lambda args: 16,
        "BLOCK_N": lambda args: 16,
        "num_warps": lambda args: 8,
        "num_stages": lambda args: 1,
    }
    for name in (
        "mha_block_128",
        "mha_block_64",
        "mha_block_32",
        "mha_block_16",
        "mha_varlen_decode",
    ):
        cfg = HEURISTICS_CONFIGS.get(name)
        if isinstance(cfg, dict):
            cfg.update(overrides)
    if not getattr(gems_runtime, "_sglang_fl_sunrise_safe_flash_tile", False):
        gems_runtime._sglang_fl_sunrise_safe_flash_tile = True
        logger.info(
            "SunriseAttnBackend: FlagGems flash tile clamped to "
            "BLOCK=16/16 warps=8 stages=1"
        )


@dataclass
class ForwardMetadata:
    attn_logits: Optional[torch.Tensor]
    attn_lse: Optional[torch.Tensor]
    max_extend_len: Optional[int]
    num_kv_splits: Optional[torch.Tensor]
    kv_indptr: torch.Tensor
    kv_indices: torch.Tensor
    qo_indptr: Optional[torch.Tensor]
    custom_mask: Optional[torch.Tensor]
    mask_indptr: Optional[torch.Tensor]
    kv_seqlen_max: int
    # Valid prefix of ``kv_indices`` (PTPU paged + dense-gather fallback).
    kv_len: int = 0
    # FlagGems paged flash (vLLM-style). ``block_table`` is [bs, max_blocks]
    # of token-pool indices; ``seqused_k`` is live KV length per request.
    block_table: Optional[torch.Tensor] = None
    seqused_k: Optional[torch.Tensor] = None
    # Speculative extend as paged decode: ``kv_indptr`` / ``kv_indices`` have one
    # row per (request, new token); row j sees the prefix plus new tokens 0..j.
    spec_rows: bool = False


def _create_kv_indices(
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
) -> None:
    """Fill ``kv_indices`` from ``req_to_token`` (torch fallback; no Triton)."""
    bs = int(req_pool_indices.shape[0])
    # Prefer SGLang's Triton helper when it can run on this device.
    try:
        from sglang.srt.layers.attention.utils import (
            create_flashinfer_kv_indices_triton,
        )

        create_flashinfer_kv_indices_triton[(bs,)](
            req_to_token,
            req_pool_indices,
            seq_lens,
            kv_indptr,
            None,
            kv_indices,
            req_to_token.stride(0),
        )
        return
    except Exception:
        pass

    # Small ``bs`` / ``S``: one sync then indexed copies (metadata-sized).
    lens = seq_lens.tolist()
    starts = kv_indptr[:bs].tolist()
    for i, (start, length) in enumerate(zip(starts, lens)):
        length = int(length)
        if length <= 0:
            continue
        start = int(start)
        kv_indices[start : start + length].copy_(
            req_to_token[req_pool_indices[i], :length]
        )


class SunriseAttnBackend(AttentionBackend):
    """PTPU attention backend registered as ``attention_backend=sunrise``."""

    def __init__(self, model_runner: ModelRunner):
        super().__init__()

        from torch_ptpu.sgl_kernel import (
            decode_attention_fwd,
            prefix_caching_fwd,
        )

        self._decode_attention_fwd = decode_attention_fwd
        self._prefix_caching_fwd = prefix_caching_fwd
        self._flash_attn_varlen = _try_import_flash_attn_varlen()
        # Last-resort SDPA when FlagGems flash is unavailable.
        self._native = TorchNativeAttnBackend(model_runner)
        self._get_kv_splits = None
        try:
            from torch_ptpu.sgl_kernel import get_mha_decoding_num_kv_splits

            self._get_kv_splits = get_mha_decoding_num_kv_splits
        except Exception:
            try:
                from torch_ptpu.sgl_kernel import get_kv_splits_num

                self._get_kv_splits = get_kv_splits_num
            except Exception:
                self._get_kv_splits = None

        max_bs = model_runner.req_to_token_pool.size
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self.device = model_runner.device
        self.max_context_len = model_runner.model_config.context_len
        self.num_head = (
            model_runner.model_config.num_attention_heads // get_attention_tp_size()
        )
        self.num_kv_head = model_runner.model_config.get_num_kv_heads(
            get_attention_tp_size()
        )
        model_head_dim = getattr(model_runner.model_config, "head_dim", None)
        try:
            self.v_head_dim = model_runner.token_to_kv_pool.get_v_head_dim()
        except Exception:
            self.v_head_dim = model_runner.token_to_kv_pool.get_value_buffer(0).shape[-1]

        # Hybrid GDN: pool v_head_dim is often the linear-attn dim (128).
        # Detect FlagGems from RadixAttention, never from the pool alone.
        self._flaggems_full_attn = self._flash_attn_varlen is not None and (
            self._probe_flaggems_full_attn(model_runner)
            or (
                model_head_dim is not None
                and int(model_head_dim) != _PTPU_QK_HEAD_DIM
            )
        )

        self._need_ptpu_kv_meta = self._probe_has_ptpu_full_attn(model_runner) or (
            not self._flaggems_full_attn
        )

        self._paged_prefill_attention_fwd = None
        if os.environ.get("SGLANG_FL_SUNRISE_PAGED_PREFILL", "1") != "0":
            try:
                from torch_ptpu.sgl_kernel import paged_prefill_attention_fwd

                self._paged_prefill_attention_fwd = paged_prefill_attention_fwd
            except Exception:
                pass
        self._need_ptpu_extend_meta = self._need_ptpu_kv_meta or (
            self._paged_prefill_attention_fwd is not None
            and self._probe_has_paged_prefill_attn(model_runner)
        )
        if self._need_ptpu_extend_meta and not self._need_ptpu_kv_meta:
            logger.info(
                "SunriseAttnBackend: extend for head_dim in %s uses PTPU "
                "paged_prefill_attention_fwd",
                _PTPU_PAGED_PREFILL_HEAD_DIMS,
            )

        self._paged_decode: Optional[PagedDecode] = None
        decode_dim = self._probe_head_dim(model_runner, PAGED_DECODE_HEAD_DIMS)
        # With head_dim 128 the PTPU KV metadata (kv_indptr / kv_indices) is built
        # anyway; the paged decode kernel reads the same two tensors.
        if decode_dim is not None:
            fn = load_paged_decode_kernel()
            if fn is not None:
                cap = measure_paged_decode_capacity(
                    fn, self.num_head, self.num_kv_head, decode_dim, self.device
                )
                if cap > 0:
                    self._paged_decode = PagedDecode(
                        fn, self.num_head, self.num_kv_head, decode_dim, cap, self.device
                    )
                logger.info(
                    "SunriseAttnBackend: decode head_dim=%d heads=%d kv_heads=%d "
                    "-> %s (exact batch*kv_heads*splits <= %d)",
                    decode_dim, self.num_head, self.num_kv_head,
                    "PTPU paged_decode_attention_fwd" if cap > 0 else "FlagGems",
                    cap,
                )
        self._need_paged_decode_meta = self._paged_decode is not None
        # MTP verify / draft extend (topk=1 chains) through the paged decode kernel:
        # 1.29 -> 0.36 ms per layer for 24 x 2 tokens at 2k context (TP1).
        self._spec_rows = self._paged_decode is not None and not self._need_ptpu_kv_meta and (
            os.environ.get("SGLANG_FL_SUNRISE_SPEC_DECODE_ROWS", "1") != "0"
        )
        self._spec_rows_kv_indptr: Optional[torch.Tensor] = None
        self._spec_row_steps: dict = {}
        self.cuda_graph_rows_kv_indptr: Optional[torch.Tensor] = None
        self.cuda_graph_rows_kv_indices: Optional[torch.Tensor] = None

        # FlagGems paged metadata (block_table / seqused_k, graph buffer of
        # graph_bs x context_len) is only built for modes where some layer
        # actually falls back to FlagGems flash.
        self._flaggems_decode, self._flaggems_extend = (
            self._probe_flaggems_fallback(model_runner)
        )
        self._use_flaggems_paged = self._flaggems_decode or self._flaggems_extend
        if self._use_flaggems_paged:
            _prepare_flaggems_flash()
            logger.info(
                "SunriseAttnBackend: FlagGems flash_attn_varlen fallback "
                "(decode=%s, extend=%s)",
                self._flaggems_decode,
                self._flaggems_extend,
            )
        self._kv_pool_tokens = int(
            getattr(model_runner, "max_total_num_tokens", 0)
            or model_runner.token_to_kv_pool.size
        )
        self.cuda_graph_paged_kv_indices: Optional[torch.Tensor] = None
        self.cuda_graph_extend_kv_indices: Optional[torch.Tensor] = None
        self._graph_spec_tokens_per_req = 0

        # Same knobs Triton uses; PTPU decode also consumes split metadata.
        self.max_kv_splits = getattr(
            model_runner.server_args, "triton_attention_num_kv_splits", 8
        )
        try:
            from sglang.srt.utils import get_device_core_count

            self.device_core_count = get_device_core_count(model_runner.gpu_id)
        except Exception:
            self.device_core_count = 0

        self.kv_indptr = torch.zeros(
            (max_bs + 1,), dtype=torch.int32, device=model_runner.device
        )
        self.qo_indptr = torch.zeros(
            (max_bs + 1,), dtype=torch.int32, device=model_runner.device
        )
        self.mask_indptr = torch.zeros(
            (max_bs + 1,), dtype=torch.int32, device=model_runner.device
        )
        self.max_bs = max_bs
        # Eager decode workspace is grown on demand. Do NOT preallocate at
        # req-pool size (often thousands) × heads × splits × dim — that alone
        # can be gigabytes and is unrelated to a 4B forward.
        self.decode_attn_logits: Optional[torch.Tensor] = None
        self.decode_attn_lse: Optional[torch.Tensor] = None
        self.decode_num_kv_splits: Optional[torch.Tensor] = None
        self.decode_cu_seqlens_q = torch.arange(
            0, max_bs + 1, dtype=torch.int32, device=self.device
        )
        self.empty_custom_mask = torch.empty(
            0, dtype=torch.uint8, device=self.device
        )
        self._kv_indices_buf: Optional[torch.Tensor] = None
        self._block_table_buf: Optional[torch.Tensor] = None
        self.cuda_graph_attn_logits: Optional[torch.Tensor] = None
        self.cuda_graph_attn_lse: Optional[torch.Tensor] = None
        self.cuda_graph_num_kv_splits: Optional[torch.Tensor] = None
        self.cuda_graph_kv_indices: Optional[torch.Tensor] = None
        self.cuda_graph_block_table: Optional[torch.Tensor] = None
        self.forward_metadata: Optional[ForwardMetadata] = None

    def support_triton(self) -> bool:
        return False

    def flaggems_graph_kv_budget(self, bs: int) -> int:
        """Max live KV tokens a FlagGems decode graph may gather.

        Paged FlagGems does not gather; 0 means no cap (graph always OK).
        """
        del bs
        return 0

    @staticmethod
    def _probe_flaggems_full_attn(model_runner: "ModelRunner") -> bool:
        """True if any RadixAttention layer is not PTPU head_dim 128."""
        has_fg, _has_ptpu = SunriseAttnBackend._probe_full_attn_kinds(model_runner)
        return has_fg

    @staticmethod
    def _probe_has_ptpu_full_attn(model_runner: "ModelRunner") -> bool:
        """True if any RadixAttention layer can use PTPU sgl kernels."""
        _has_fg, has_ptpu = SunriseAttnBackend._probe_full_attn_kinds(model_runner)
        return has_ptpu

    def _probe_flaggems_fallback(self, model_runner: "ModelRunner") -> tuple[bool, bool]:
        """(decode, extend): whether any layer routes to FlagGems flash.

        Mirrors the dispatch in ``forward_decode`` / ``forward_extend``.
        Unknown models conservatively report True.
        """
        if self._flash_attn_varlen is None:
            return False, False
        model = getattr(model_runner, "model", None)
        if model is None:
            return True, True
        from sglang.srt.layers.radix_attention import RadixAttention

        decode = extend = False
        for layer in model.modules():
            if not isinstance(layer, RadixAttention) or self._ptpu_attn_supported(layer):
                continue
            decode = decode or not self._use_paged_decode(layer)
            extend = extend or not (
                self._use_paged_prefill(layer)
                or layer.is_cross_attention
                or layer.attn_type == AttentionType.ENCODER_ONLY
            )
        return decode, extend

    @staticmethod
    def _probe_has_paged_prefill_attn(model_runner: "ModelRunner") -> bool:
        """True if any RadixAttention layer can use PTPU paged prefill."""
        model = getattr(model_runner, "model", None)
        if model is None:
            return False
        from sglang.srt.layers.radix_attention import RadixAttention

        return any(
            isinstance(mod, RadixAttention)
            and SunriseAttnBackend._paged_prefill_head_ok(mod)
            for mod in model.modules()
        )

    @staticmethod
    def _probe_head_dim(model_runner: "ModelRunner", dims: tuple) -> Optional[int]:
        """Common ``qk == v`` head dim of all RadixAttention layers if in ``dims``."""
        model = getattr(model_runner, "model", None)
        if model is None:
            return None
        from sglang.srt.layers.radix_attention import RadixAttention

        found = {
            (int(getattr(m, "qk_head_dim", 0) or 0), int(getattr(m, "v_head_dim", 0) or 0))
            for m in model.modules()
            if isinstance(m, RadixAttention)
        }
        if len(found) != 1:
            return None
        qk, vh = found.pop()
        return qk if qk == vh and qk in dims else None

    def _use_paged_decode(self, layer: RadixAttention) -> bool:
        return (
            self._paged_decode is not None
            and int(layer.qk_head_dim) == self._paged_decode.dim
            and int(layer.v_head_dim) == self._paged_decode.dim
            and int(layer.tp_q_head_num) == self._paged_decode.heads
            and int(layer.tp_k_head_num) == self._paged_decode.kv_heads
            and not layer.is_cross_attention
            and layer.attn_type != AttentionType.ENCODER_ONLY
        )

    @staticmethod
    def _paged_prefill_head_ok(layer: RadixAttention) -> bool:
        qk = int(getattr(layer, "qk_head_dim", 0) or 0)
        vh = int(getattr(layer, "v_head_dim", 0) or 0)
        return qk == vh and qk in _PTPU_PAGED_PREFILL_HEAD_DIMS

    def _use_paged_prefill(self, layer: RadixAttention) -> bool:
        return (
            self._paged_prefill_attention_fwd is not None
            and self._need_ptpu_extend_meta
            and self._paged_prefill_head_ok(layer)
            and not layer.is_cross_attention
            and layer.attn_type != AttentionType.ENCODER_ONLY
        )

    @staticmethod
    def _probe_full_attn_kinds(model_runner: "ModelRunner") -> tuple[bool, bool]:
        """Return (has_non_128_full_attn, has_128_full_attn)."""
        model = getattr(model_runner, "model", None)
        if model is None:
            return False, False
        try:
            from sglang.srt.layers.radix_attention import RadixAttention
        except Exception:
            return False, False
        has_fg = False
        has_ptpu = False
        for mod in model.modules():
            if not isinstance(mod, RadixAttention):
                continue
            qk = int(getattr(mod, "qk_head_dim", 0) or 0)
            vh = int(getattr(mod, "v_head_dim", 0) or 0)
            if qk <= 0:
                continue
            if qk != _PTPU_QK_HEAD_DIM or vh != _PTPU_QK_HEAD_DIM:
                has_fg = True
            else:
                has_ptpu = True
        return has_fg, has_ptpu

    @staticmethod
    def _ptpu_attn_supported(layer: RadixAttention) -> bool:
        return (
            int(layer.qk_head_dim) == _PTPU_QK_HEAD_DIM
            and int(layer.v_head_dim) == _PTPU_QK_HEAD_DIM
        )

    def _maybe_log_fallback(self, layer: RadixAttention, path: str) -> None:
        global _fallback_logged
        if _fallback_logged:
            return
        _fallback_logged = True
        logger.warning(
            "SunriseAttnBackend: qk_head_dim=%s v_head_dim=%s — PTPU sgl "
            "attn needs %s; using %s for decode+extend",
            layer.qk_head_dim,
            layer.v_head_dim,
            _PTPU_QK_HEAD_DIM,
            path,
        )

    def _flash_attn_varlen_call(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        out: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        sm_scale: float,
        causal: bool,
        softcap: float,
        *,
        cu_seqlens_k: Optional[torch.Tensor] = None,
        seqused_k: Optional[torch.Tensor] = None,
        block_table: Optional[torch.Tensor] = None,
    ) -> None:
        """Call FlagGems flash. Paged path matches vLLM sunrise (no dense gather)."""
        fn = self._flash_attn_varlen
        assert fn is not None
        cu_q = (
            cu_seqlens_q
            if cu_seqlens_q.dtype == torch.int32
            else cu_seqlens_q.to(dtype=torch.int32)
        )
        kwargs = dict(
            q=q,
            k=k,
            v=v,
            out=out,
            cu_seqlens_q=cu_q,
            max_seqlen_q=int(max_seqlen_q),
            max_seqlen_k=int(max_seqlen_k),
            softmax_scale=float(sm_scale),
            causal=bool(causal),
        )
        if seqused_k is not None:
            kwargs["seqused_k"] = (
                seqused_k
                if seqused_k.dtype == torch.int32
                else seqused_k.to(dtype=torch.int32)
            )
        if block_table is not None:
            kwargs["block_table"] = block_table
        if cu_seqlens_k is not None:
            kwargs["cu_seqlens_k"] = (
                cu_seqlens_k
                if cu_seqlens_k.dtype == torch.int32
                else cu_seqlens_k.to(dtype=torch.int32)
            )
        if softcap and softcap > 0:
            kwargs["softcap"] = float(softcap)
        try:
            fn(**kwargs)
            return
        except TypeError:
            if block_table is not None:
                raise
        assert cu_seqlens_k is not None
        cu_k = (
            cu_seqlens_k
            if cu_seqlens_k.dtype == torch.int32
            else cu_seqlens_k.to(dtype=torch.int32)
        )
        ret = fn(
            q,
            k,
            v,
            cu_q,
            cu_k,
            int(max_seqlen_q),
            int(max_seqlen_k),
            softmax_scale=float(sm_scale),
            causal=bool(causal),
        )
        if ret is not None and ret is not out:
            out.copy_(ret)

    def _reshape_kv_cache(
        self,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        layer: RadixAttention,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if k_cache.dim() == 2:
            k_cache = k_cache.view(-1, layer.tp_k_head_num, layer.qk_head_dim)
            v_cache = v_cache.view(-1, layer.tp_v_head_num, layer.v_head_dim)
        return k_cache, v_cache

    @staticmethod
    def _as_flaggems_paged_kv(
        k_cache: torch.Tensor, v_cache: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """FlagGems paged FA expects ``(num_pages, page_size, nheads_k, headdim)``.

        SGLang's token pool is ``(num_tokens, nheads_k, headdim)``. Insert
        ``page_size=1`` so ``shape[-2]`` is KV heads. Passing 3D makes FlagGems
        treat ``head_dim`` as ``num_heads_k`` and assert
        ``nheads_q % nheads_k == 0``.
        """
        if k_cache.dim() == 3:
            k_cache = k_cache.unsqueeze(1)
        if v_cache.dim() == 3:
            v_cache = v_cache.unsqueeze(1)
        return k_cache, v_cache

    def _forward_flaggems_dense(
        self,
        q_: torch.Tensor,
        o_: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        cu_seqlens_q: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        causal: bool,
    ) -> None:
        """Legacy FlagGems path: materialize ``k_cache[idx]`` then dense varlen."""
        meta = self.forward_metadata
        assert meta is not None
        n_kv = int(meta.kv_len) if meta.kv_len > 0 else 0
        if n_kv <= 0 or int(meta.kv_indices.shape[0]) < max(n_kv, 1):
            bs = int(forward_batch.batch_size)
            n_kv = self._host_seq_lens_sum(forward_batch, bs)
            kv_indices = self._ensure_kv_indices(n_kv)
            kv_indptr = self._fill_decode_kv(
                bs,
                forward_batch.seq_lens,
                forward_batch.req_pool_indices,
                kv_indices,
            )
            meta.kv_indices = kv_indices
            meta.kv_indptr = kv_indptr
            meta.kv_len = int(n_kv)
        n_kv = min(max(n_kv, 1), int(meta.kv_indices.shape[0]))
        idx = meta.kv_indices[:n_kv].to(dtype=torch.long)
        k_g = k_cache[idx]
        v_g = v_cache[idx]
        k_g, v_g = self._reshape_kv_cache(k_g, v_g, layer)
        self._flash_attn_varlen_call(
            q_,
            k_g.contiguous(),
            v_g.contiguous(),
            o_,
            cu_seqlens_q,
            max_seqlen_q,
            max_seqlen_k,
            float(layer.scaling),
            causal,
            self._logit_cap(layer),
            cu_seqlens_k=meta.kv_indptr,
        )

    def _forward_flaggems(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool,
        *,
        is_decode: bool,
    ) -> torch.Tensor:
        """FlagGems flash_attn_varlen — paged KV like vLLM sunrise."""
        global _flaggems_paged_logged
        if is_decode:
            q = q.reshape(-1, layer.tp_q_head_num * layer.qk_head_dim)
        if layer.qk_head_dim != layer.v_head_dim:
            o = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
        else:
            o = torch.empty_like(q)

        self._save_kv(layer, forward_batch, k, v, save_kv_cache)

        meta = self.forward_metadata
        assert meta is not None
        k_cache = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
        v_cache = forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id)
        k_cache, v_cache = self._reshape_kv_cache(k_cache, v_cache, layer)

        q_ = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
        o_ = o.view(-1, layer.tp_q_head_num, layer.v_head_dim)

        graph_paged = (
            meta.block_table is not None
            and self.cuda_graph_block_table is not None
            and meta.block_table.data_ptr() == self.cuda_graph_block_table.data_ptr()
        )
        if is_decode:
            bs = forward_batch.batch_size
            cu_seqlens_q = self.decode_cu_seqlens_q[: bs + 1]
            max_seqlen_q = 1
            causal = False
        else:
            assert meta.qo_indptr is not None and meta.max_extend_len is not None
            cu_seqlens_q = meta.qo_indptr
            max_seqlen_q = int(meta.max_extend_len)
            causal = True
        max_seqlen_k = (
            int(meta.kv_seqlen_max)
            if graph_paged
            else self._host_max_seq_len(forward_batch)
        )

        if meta.block_table is not None:
            seqused_k = meta.seqused_k
            if seqused_k is None:
                seqused_k = forward_batch.seq_lens[: forward_batch.batch_size]
            k_paged, v_paged = self._as_flaggems_paged_kv(k_cache, v_cache)
            try:
                self._flash_attn_varlen_call(
                    q_,
                    k_paged,
                    v_paged,
                    o_,
                    cu_seqlens_q,
                    max_seqlen_q,
                    max_seqlen_k,
                    float(layer.scaling),
                    causal,
                    self._logit_cap(layer),
                    seqused_k=seqused_k,
                    block_table=meta.block_table,
                )
                if not _flaggems_paged_logged:
                    _flaggems_paged_logged = True
                    logger.info(
                        "SunriseAttnBackend: FlagGems paged flash "
                        "(block_table + seqused_k, no k_cache[idx] gather)"
                    )
                return o
            except TypeError:
                logger.warning(
                    "SunriseAttnBackend: FlagGems flash_attn_varlen rejected "
                    "paged kwargs; falling back to dense k_cache[idx] gather"
                )

        self._forward_flaggems_dense(
            q_,
            o_,
            k_cache,
            v_cache,
            layer,
            forward_batch,
            cu_seqlens_q,
            max_seqlen_q,
            max_seqlen_k,
            causal,
        )
        return o

    # ---- metadata ------------------------------------------------------------

    def _fill_num_kv_splits(
        self, num_kv_splits: torch.Tensor, seq_lens: torch.Tensor
    ) -> None:
        if self._get_kv_splits is None or self.device_core_count <= 0:
            num_kv_splits.fill_(self.max_kv_splits)
            return
        try:
            # get_mha_decoding_num_kv_splits(num_kv_splits, seq_lens, num_group,
            #   num_head, num_kv_head, max_kv_splits, device_core_count)
            import inspect

            nparams = len(inspect.signature(self._get_kv_splits).parameters)
            if nparams >= 7:
                self._get_kv_splits(
                    num_kv_splits,
                    seq_lens,
                    1,
                    self.num_head,
                    self.num_kv_head,
                    self.max_kv_splits,
                    self.device_core_count,
                )
            else:
                self._get_kv_splits(
                    num_kv_splits,
                    seq_lens,
                    self.num_head,
                    self.max_kv_splits,
                    self.device_core_count,
                )
        except Exception:
            num_kv_splits.fill_(self.max_kv_splits)

    @staticmethod
    def _host_seq_lens_sum(forward_batch: ForwardBatch, bs: int) -> int:
        """Host-side token count; never calls device ``.item()``."""
        s = getattr(forward_batch, "seq_lens_sum", None)
        if s is not None:
            return max(int(s), 1)
        cpu = getattr(forward_batch, "seq_lens_cpu", None)
        if cpu is not None:
            if isinstance(cpu, torch.Tensor):
                return max(int(cpu[:bs].sum().item()), 1)
            return max(int(sum(cpu[:bs])), 1)
        return max(int(bs), 1)

    def _host_max_seq_len(self, forward_batch: ForwardBatch) -> int:
        """Host-side max seq len; never D2H from device seq_lens."""
        cpu = getattr(forward_batch, "seq_lens_cpu", None)
        if cpu is not None:
            if isinstance(cpu, torch.Tensor) and cpu.numel() > 0:
                return int(cpu.max().item())
            if not isinstance(cpu, torch.Tensor) and len(cpu) > 0:
                return int(max(cpu))
        return int(self.max_context_len)

    @staticmethod
    def _host_max_extend_len(forward_batch: ForwardBatch, cap: int) -> int:
        cpu = getattr(forward_batch, "extend_seq_lens_cpu", None)
        if cpu:
            return int(max(cpu))
        return int(cap)

    def _ensure_kv_indices(self, n: int) -> torch.Tensor:
        n = max(int(n), 1)
        buf = self._kv_indices_buf
        if buf is None or buf.numel() < n:
            cap = n if buf is None else max(n, int(buf.numel()) * 2)
            self._kv_indices_buf = torch.empty(
                cap, dtype=torch.int32, device=self.device
            )
        return self._kv_indices_buf[:n]

    def _ensure_block_table(self, bs: int, max_blocks: int) -> torch.Tensor:
        """Growable eager ``[bs, max_blocks]`` int32 block table."""
        bs = max(int(bs), 1)
        max_blocks = max(int(max_blocks), 1)
        buf = self._block_table_buf
        if buf is None or buf.shape[0] < bs or buf.shape[1] < max_blocks:
            cap_bs = bs if buf is None else max(bs, int(buf.shape[0]) * 2)
            cap_bs = min(max(cap_bs, bs), int(self.max_bs))
            cap_blk = (
                max_blocks if buf is None else max(max_blocks, int(buf.shape[1]))
            )
            cap_blk = min(max(cap_blk, max_blocks), int(self.max_context_len))
            self._block_table_buf = torch.zeros(
                (cap_bs, cap_blk), dtype=torch.int32, device=self.device
            )
            buf = self._block_table_buf
        return buf[:bs, :max_blocks]

    def _fill_block_table(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        block_table: torch.Tensor,
        n_cols: int,
    ) -> None:
        """Copy token-pool indices into FlagGems ``block_table`` (block_size=1)."""
        n_cols = min(
            max(int(n_cols), 0),
            int(block_table.shape[1]),
            int(self.req_to_token.shape[1]),
        )
        if n_cols <= 0 or bs <= 0:
            return
        src = self.req_to_token[req_pool_indices[:bs], :n_cols]
        dst = block_table[:bs, :n_cols]
        if src.dtype != dst.dtype:
            src = src.to(dtype=dst.dtype)
        dst.copy_(src)

    def _flaggems_block_table_meta(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        n_cols: int,
        block_table: Optional[torch.Tensor] = None,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not self._use_flaggems_paged:
            return None, None
        if block_table is None:
            block_table = self._ensure_block_table(bs, n_cols)
        self._fill_block_table(bs, req_pool_indices, block_table, n_cols)
        return block_table, seq_lens[:bs]

    def _ensure_decode_workspace(self, bs: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bs = max(int(bs), 1)
        logits = self.decode_attn_logits
        if logits is None or logits.shape[0] < bs:
            cap = bs if logits is None else max(bs, int(logits.shape[0]) * 2)
            cap = min(cap, int(self.max_bs))
            cap = max(cap, bs)
            self.decode_attn_logits = torch.empty(
                (cap, self.num_head, self.max_kv_splits, self.v_head_dim),
                dtype=torch.float32,
                device=self.device,
            )
            self.decode_attn_lse = torch.empty(
                (cap, self.num_head, self.max_kv_splits),
                dtype=torch.float32,
                device=self.device,
            )
            self.decode_num_kv_splits = torch.empty(
                (cap,), dtype=torch.int32, device=self.device
            )
        assert self.decode_attn_logits is not None
        assert self.decode_attn_lse is not None
        assert self.decode_num_kv_splits is not None
        return (
            self.decode_attn_logits[:bs],
            self.decode_attn_lse[:bs],
            self.decode_num_kv_splits[:bs],
        )

    def _fill_decode_kv(
        self,
        bs: int,
        seq_lens: torch.Tensor,
        req_pool_indices: torch.Tensor,
        kv_indices: torch.Tensor,
    ) -> torch.Tensor:
        kv_indptr = self.kv_indptr
        kv_indptr[1 : bs + 1] = torch.cumsum(
            seq_lens.to(dtype=torch.int32)[:bs], dim=0
        )
        kv_indptr_view = kv_indptr[: bs + 1]
        _create_kv_indices(
            self.req_to_token,
            req_pool_indices[:bs],
            seq_lens[:bs],
            kv_indptr_view,
            kv_indices,
        )
        return kv_indptr_view

    @staticmethod
    def _spec_rows_kv_bound(seq_lens_sum: int, bs: int, n: int) -> int:
        """Upper bound on the row KV total (rows see at most ``seq_len + n`` slots)."""
        return n * int(seq_lens_sum) + bs * n * n + 1

    def _fill_spec_rows(
        self,
        bs: int,
        n: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        forward_mode: ForwardMode,
        kv_indptr: torch.Tensor,
        kv_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Paged-decode rows for a fixed-length speculative extend of ``n`` tokens per
        request: row ``b * n + j`` reads the first ``L_b + j + 1`` slots of request b,
        where ``L_b`` is the cached prefix (target verify: ``seq_lens`` excludes the
        new tokens; draft extend v2: it includes them). ``kv_indptr[0]`` stays 0."""
        steps = self._spec_row_steps.get(n)
        if steps is None:
            steps = torch.arange(1, n + 1, dtype=torch.int32, device=self.device)
            self._spec_row_steps[n] = steps
        base = seq_lens[:bs].to(torch.int32)
        if not forward_mode.is_target_verify():
            base = base - n
        # Padded graph rows (seq_len 1) would get empty rows otherwise.
        row_lens = (base.unsqueeze(1) + steps).clamp_min(1).reshape(-1)
        rows = bs * n
        kv_indptr[1 : rows + 1] = torch.cumsum(row_lens, dim=0)
        # The index kernel assumes contiguous inputs (expand + reshape is a stride-0
        # view when bs == 1).
        row_req = req_pool_indices[:bs].repeat_interleave(n)
        indptr = kv_indptr[: rows + 1]
        _create_kv_indices(self.req_to_token, row_req, row_lens, indptr, kv_indices)
        return indptr

    def init_draft_decode_metadata(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        extra: int,
        seq_lens_sum: Optional[int] = None,
        kv_indices: Optional[torch.Tensor] = None,
    ) -> None:
        """Paged-decode metadata of an EAGLE chain draft step (``topk == 1``): request b
        reads the first ``seq_lens[b] + extra`` slots, the prefix plus the step's draft
        tokens (same layout as SGLang's ``generate_draft_decode_kv_indices``).
        ``kv_indices`` defaults to the growable eager buffer."""
        if not self._need_paged_decode_meta:
            raise NotImplementedError("sunrise multi-step draft needs the paged decode kernel")
        lens = seq_lens[:bs].to(torch.int32) + extra
        self.kv_indptr[1 : bs + 1] = torch.cumsum(lens, dim=0)
        indptr = self.kv_indptr[: bs + 1]
        if kv_indices is None:
            total = (int(seq_lens_sum) if seq_lens_sum is not None else bs * int(self.max_context_len))
            kv_indices = self._ensure_kv_indices(total + bs * extra)
        _create_kv_indices(self.req_to_token, req_pool_indices[:bs], lens, indptr, kv_indices)
        self.forward_metadata = ForwardMetadata(
            attn_logits=None,
            attn_lse=None,
            max_extend_len=None,
            num_kv_splits=None,
            kv_indptr=indptr,
            kv_indices=kv_indices,
            qo_indptr=None,
            custom_mask=None,
            mask_indptr=None,
            kv_seqlen_max=int(self.max_context_len),
            kv_len=int(kv_indices.numel()),
        )

    def _spec_rows_metadata(
        self, kv_indptr: torch.Tensor, kv_indices: torch.Tensor, n: int, kv_len: int
    ) -> ForwardMetadata:
        return ForwardMetadata(
            attn_logits=None,
            attn_lse=None,
            max_extend_len=n,
            num_kv_splits=None,
            kv_indptr=kv_indptr,
            kv_indices=kv_indices,
            qo_indptr=None,
            custom_mask=None,
            mask_indptr=None,
            kv_seqlen_max=int(self.max_context_len),
            kv_len=int(kv_len),
            spec_rows=True,
        )

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        # Always refresh native metadata (used when head_dim is unsupported).
        self._native.init_forward_metadata(forward_batch)

        bs = forward_batch.batch_size
        kv_indptr = self.kv_indptr
        mode = forward_batch.forward_mode
        block_table = seqused_k = None
        need_block_table = (
            self._flaggems_decode if mode.is_decode_or_idle() else self._flaggems_extend
        )
        if need_block_table:
            block_table, seqused_k = self._flaggems_block_table_meta(
                bs,
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                self._host_max_seq_len(forward_batch),
            )

        if mode.is_decode_or_idle():
            if forward_batch.spec_info is not None:
                raise NotImplementedError(
                    "SunriseAttnBackend: speculative decode metadata is not "
                    "implemented yet"
                )
            n_idx = 0
            kv_indices = self.kv_indptr[:1]
            if self._need_ptpu_kv_meta or self._need_paged_decode_meta:
                n_idx = self._host_seq_lens_sum(forward_batch, bs)
                kv_indices = self._ensure_kv_indices(n_idx)
                kv_indptr = self._fill_decode_kv(
                    bs,
                    forward_batch.seq_lens,
                    forward_batch.req_pool_indices,
                    kv_indices,
                )
            else:
                kv_indptr = kv_indptr[: bs + 1]
            attn_logits, attn_lse, num_kv_splits = self._ensure_decode_workspace(bs)
            self._fill_num_kv_splits(num_kv_splits, forward_batch.seq_lens[:bs])
            self.forward_metadata = ForwardMetadata(
                attn_logits=attn_logits,
                attn_lse=attn_lse,
                max_extend_len=None,
                num_kv_splits=num_kv_splits,
                kv_indptr=kv_indptr,
                kv_indices=kv_indices,
                qo_indptr=None,
                custom_mask=None,
                mask_indptr=None,
                kv_seqlen_max=int(self.max_context_len),
                kv_len=int(n_idx),
                block_table=block_table,
                seqused_k=seqused_k,
            )
            return

        if mode.is_target_verify():
            self.forward_metadata = self._target_verify_metadata(forward_batch, bs)
            return

        if mode.is_extend(include_draft_extend_v2=True) or mode.is_mixed():
            # PTPU prefix_caching reads *all* KV from the token pool (after
            # set_kv_buffer). Build indices over full seq_lens, not prefix-only.
            n_idx = 0
            kv_indices = self.kv_indptr[:1]
            kv_indptr = kv_indptr[: bs + 1]
            if self._need_ptpu_extend_meta:
                kv_indptr[1 : bs + 1] = torch.cumsum(
                    forward_batch.seq_lens.to(dtype=torch.int32), dim=0
                )
                n_idx = self._host_seq_lens_sum(forward_batch, bs)
                kv_indices = self._ensure_kv_indices(n_idx)
                _create_kv_indices(
                    self.req_to_token,
                    forward_batch.req_pool_indices,
                    forward_batch.seq_lens,
                    kv_indptr,
                    kv_indices,
                )
            qo_indptr = self.qo_indptr
            qo_indptr[1 : bs + 1] = torch.cumsum(
                forward_batch.extend_seq_lens.to(dtype=torch.int32), dim=0
            )
            qo_indptr = qo_indptr[: bs + 1]
            max_extend_len = self._host_max_extend_len(
                forward_batch, self.max_context_len
            )
            mask_indptr = self.mask_indptr[: bs + 1].zero_()
            self.forward_metadata = ForwardMetadata(
                attn_logits=None,
                attn_lse=None,
                max_extend_len=int(max_extend_len),
                num_kv_splits=None,
                kv_indptr=kv_indptr,
                kv_indices=kv_indices,
                qo_indptr=qo_indptr,
                custom_mask=self.empty_custom_mask,
                mask_indptr=mask_indptr,
                kv_seqlen_max=int(self.max_context_len),
                kv_len=int(n_idx),
                block_table=block_table,
                seqused_k=seqused_k,
            )
            return

        raise NotImplementedError(
            f"SunriseAttnBackend: unsupported forward_mode={mode}"
        )

    def _target_verify_metadata(self, forward_batch: ForwardBatch, bs: int) -> ForwardMetadata:
        """MTP / EAGLE chain verify: each request extends its ``seq_lens`` prefix by
        ``draft_token_num`` tokens whose KV slots are already in ``req_to_token``.
        With ``topk == 1`` the draft mask is causal, so the paged prefill kernel's
        causal mode replaces the custom tree mask."""
        spec = forward_batch.spec_info
        n_draft = int(spec.draft_token_num)
        if int(getattr(spec, "topk", 1) or 1) > 1:
            raise NotImplementedError("SunriseAttnBackend: EAGLE trees (topk > 1) are not supported")
        if self._flaggems_extend:
            raise NotImplementedError("SunriseAttnBackend: target verify with FlagGems fallback layers")
        if self._spec_rows:
            rows = bs * n_draft
            buf = self._spec_rows_kv_indptr
            if buf is None or buf.numel() < rows + 1:
                buf = torch.zeros(max(rows + 1, 2 * (0 if buf is None else buf.numel())),
                                  dtype=torch.int32, device=self.device)
                self._spec_rows_kv_indptr = buf
            n_idx = self._spec_rows_kv_bound(self._host_seq_lens_sum(forward_batch, bs), bs, n_draft)
            kv_indices = self._ensure_kv_indices(n_idx)
            kv_indptr = self._fill_spec_rows(
                bs, n_draft, forward_batch.req_pool_indices, forward_batch.seq_lens,
                forward_batch.forward_mode, buf, kv_indices,
            )
            return self._spec_rows_metadata(kv_indptr, kv_indices, n_draft, n_idx)
        kv_indptr = self.kv_indptr[: bs + 1]
        n_idx = 0
        kv_indices = self.kv_indptr[:1]
        if self._need_ptpu_extend_meta:
            kv_lens = forward_batch.seq_lens[:bs].to(torch.int32) + n_draft
            kv_indptr[1 : bs + 1] = torch.cumsum(kv_lens, dim=0)
            n_idx = self._host_seq_lens_sum(forward_batch, bs) + bs * n_draft
            kv_indices = self._ensure_kv_indices(n_idx)
            _create_kv_indices(
                self.req_to_token, forward_batch.req_pool_indices, kv_lens, kv_indptr, kv_indices
            )
        qo_indptr = self.qo_indptr[: bs + 1]
        torch.mul(self.decode_cu_seqlens_q[: bs + 1], n_draft, out=qo_indptr)
        return ForwardMetadata(
            attn_logits=None,
            attn_lse=None,
            max_extend_len=n_draft,
            num_kv_splits=None,
            kv_indptr=kv_indptr,
            kv_indices=kv_indices,
            qo_indptr=qo_indptr,
            custom_mask=self.empty_custom_mask,
            mask_indptr=self.mask_indptr[: bs + 1].zero_(),
            kv_seqlen_max=int(self.max_context_len),
            kv_len=int(n_idx),
        )

    # ---- CUDA / PTPU graph (SGLang 0.5.11 capture/replay) --------------------

    def _graph_kv_seq_cap(self) -> int:
        """Per-request KV slots in the CUDA-graph ``kv_indices`` buffer."""
        if not self._need_ptpu_kv_meta:
            return 1
        return int(self.max_context_len)

    def _graph_kv_indices_view(self, bs: int) -> torch.Tensor:
        """View of size ``bs * seq_cap`` into the persistent graph buffer."""
        assert self.cuda_graph_kv_indices is not None
        n = max(int(bs), 1) * self._graph_kv_seq_cap()
        n = min(n, int(self.cuda_graph_kv_indices.numel()))
        return self.cuda_graph_kv_indices[:n]

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        """Persistent buffers whose addresses are captured into the graph."""
        ntok = max(int(max_num_tokens), 1)
        graph_bs = max(int(max_bs), ntok)
        self.cuda_graph_attn_logits = torch.zeros(
            (ntok, self.num_head, self.max_kv_splits, self.v_head_dim),
            dtype=torch.float32,
            device=self.device,
        )
        self.cuda_graph_attn_lse = torch.zeros(
            (ntok, self.num_head, self.max_kv_splits),
            dtype=torch.float32,
            device=self.device,
        )
        self.cuda_graph_num_kv_splits = torch.full(
            (ntok,),
            self.max_kv_splits,
            dtype=torch.int32,
            device=self.device,
        )
        self.cuda_graph_kv_indices = torch.zeros(
            (ntok * self._graph_kv_seq_cap(),),
            dtype=torch.int32,
            device=self.device,
        )
        if self._need_paged_decode_meta:
            # Live KV of a decode batch never exceeds the token pool; padded
            # rows add one slot each.
            self.cuda_graph_paged_kv_indices = torch.zeros(
                (self._kv_pool_tokens + ntok + 1,), dtype=torch.int32, device=self.device
            )
        if self._need_ptpu_extend_meta:
            # Target-verify graphs: prefix + draft tokens of every (padded) request.
            self.cuda_graph_extend_kv_indices = torch.zeros(
                (self._kv_pool_tokens + 2 * ntok + 1,), dtype=torch.int32, device=self.device
            )
        if self._spec_rows and ntok > int(max_bs):
            # Speculative graphs (several tokens per request): every row lists its
            # request's live KV, so the pool is covered once per token.
            n = -(-ntok // max(int(max_bs), 1))
            self.cuda_graph_rows_kv_indptr = torch.zeros(
                (ntok + 1,), dtype=torch.int32, device=self.device
            )
            self.cuda_graph_rows_kv_indices = torch.zeros(
                (self._spec_rows_kv_bound(self._kv_pool_tokens, ntok, n),),
                dtype=torch.int32, device=self.device,
            )
        if self._flaggems_decode:
            self.cuda_graph_block_table = torch.zeros(
                (graph_bs, int(self.max_context_len)),
                dtype=torch.int32,
                device=self.device,
            )

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
    ):
        """Bind decode kernels to the persistent graph buffers (once)."""
        if forward_mode.is_target_verify() or forward_mode.is_draft_extend_v2():
            self._graph_spec_tokens_per_req = num_tokens // bs
            self.forward_metadata = self._graph_spec_extend_metadata(
                bs, num_tokens // bs, req_pool_indices, seq_lens, spec_info, forward_mode
            )
            return
        del num_tokens, encoder_lens
        if spec_info is not None:
            raise NotImplementedError(
                "SunriseAttnBackend: speculative draft cuda graph is not "
                "implemented yet"
            )
        if not forward_mode.is_decode_or_idle():
            raise NotImplementedError(
                f"SunriseAttnBackend: cuda graph capture only supports "
                f"decode/idle, got {forward_mode}"
            )
        assert self.cuda_graph_kv_indices is not None
        assert self.cuda_graph_attn_logits is not None
        assert self.cuda_graph_attn_lse is not None
        assert self.cuda_graph_num_kv_splits is not None

        kv_len = 0
        kv_indices = self.cuda_graph_kv_indices[:1]
        kv_indptr = self.kv_indptr[: bs + 1]
        if self._need_ptpu_kv_meta:
            kv_indices = self._graph_kv_indices_view(bs)
            kv_indptr = self._fill_decode_kv(
                bs, seq_lens, req_pool_indices, kv_indices
            )
            kv_len = int(kv_indices.numel())
        elif self._need_paged_decode_meta:
            assert self.cuda_graph_paged_kv_indices is not None
            kv_indices = self.cuda_graph_paged_kv_indices
            kv_indptr = self._fill_decode_kv(
                bs, seq_lens, req_pool_indices, kv_indices
            )
        self._fill_num_kv_splits(
            self.cuda_graph_num_kv_splits[:bs], seq_lens[:bs]
        )
        block_table = None
        seqused_k = None
        if self._flaggems_decode:
            assert self.cuda_graph_block_table is not None
            block_table = self.cuda_graph_block_table[:bs]
            block_table, seqused_k = self._flaggems_block_table_meta(
                bs,
                req_pool_indices,
                seq_lens,
                n_cols=1,
                block_table=block_table,
            )
        self.forward_metadata = ForwardMetadata(
            attn_logits=self.cuda_graph_attn_logits[:bs],
            attn_lse=self.cuda_graph_attn_lse[:bs],
            max_extend_len=None,
            num_kv_splits=self.cuda_graph_num_kv_splits[:bs],
            kv_indptr=kv_indptr,
            kv_indices=kv_indices,
            qo_indptr=None,
            custom_mask=None,
            mask_indptr=None,
            kv_seqlen_max=int(self.max_context_len),
            kv_len=kv_len,
            block_table=block_table,
            seqused_k=seqused_k,
        )

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
        seq_lens_cpu: Optional[torch.Tensor],
    ):
        """Refill captured buffers in place; do not replace tensor objects."""
        del seq_lens_sum, encoder_lens
        if forward_mode.is_target_verify() or forward_mode.is_draft_extend_v2():
            self._graph_spec_extend_metadata(
                bs, self._graph_spec_tokens_per_req, req_pool_indices, seq_lens, spec_info, forward_mode
            )
            return
        if spec_info is not None:
            raise NotImplementedError(
                "SunriseAttnBackend: speculative draft cuda graph is not "
                "implemented yet"
            )
        if not forward_mode.is_decode_or_idle():
            raise NotImplementedError(
                f"SunriseAttnBackend: cuda graph replay only supports "
                f"decode/idle, got {forward_mode}"
            )
        assert self.cuda_graph_kv_indices is not None
        assert self.cuda_graph_num_kv_splits is not None
        if self._need_ptpu_kv_meta:
            kv_indices = self._graph_kv_indices_view(bs)
            self._fill_decode_kv(
                bs, seq_lens, req_pool_indices, kv_indices
            )
        elif self._need_paged_decode_meta:
            assert self.cuda_graph_paged_kv_indices is not None
            self._fill_decode_kv(
                bs, seq_lens, req_pool_indices, self.cuda_graph_paged_kv_indices
            )
        self._fill_num_kv_splits(
            self.cuda_graph_num_kv_splits[:bs], seq_lens[:bs]
        )
        if self._flaggems_decode:
            assert self.cuda_graph_block_table is not None
            n_cols = int(self.max_context_len)
            if seq_lens_cpu is not None:
                if isinstance(seq_lens_cpu, torch.Tensor) and seq_lens_cpu.numel() > 0:
                    n_cols = int(seq_lens_cpu.max().item())
                elif not isinstance(seq_lens_cpu, torch.Tensor) and len(seq_lens_cpu) > 0:
                    n_cols = int(max(seq_lens_cpu))
            self._fill_block_table(
                bs,
                req_pool_indices,
                self.cuda_graph_block_table[:bs],
                n_cols,
            )

    def _graph_spec_extend_metadata(
        self, bs, n_draft, req_pool_indices, seq_lens, spec_info, forward_mode
    ) -> ForwardMetadata:
        """Graph metadata (capture and replay) in the persistent buffers for the
        fixed-length speculative extends: target verify (``seq_lens`` excludes
        the ``n_draft`` new tokens) and draft extend v2 (``seq_lens`` includes them).

        Same layout as :meth:`_target_verify_metadata`; ``qo_indptr`` is the
        same for every captured batch size and ``kv_indices`` is sized for the
        whole token pool plus every padded draft token."""
        if int(getattr(spec_info, "topk", 1) or 1) > 1:
            raise NotImplementedError("SunriseAttnBackend: EAGLE trees (topk > 1) are not supported")
        if self._flaggems_extend or not self._need_ptpu_extend_meta:
            raise NotImplementedError("SunriseAttnBackend: speculative graphs need the PTPU extend kernels")
        if self.cuda_graph_rows_kv_indices is not None:
            kv_indices = self.cuda_graph_rows_kv_indices
            kv_indptr = self._fill_spec_rows(
                bs, n_draft, req_pool_indices, seq_lens, forward_mode,
                self.cuda_graph_rows_kv_indptr, kv_indices,
            )
            return self._spec_rows_metadata(kv_indptr, kv_indices, n_draft, kv_indices.numel())
        kv_indptr = self.kv_indptr[: bs + 1]
        kv_lens = seq_lens[:bs].to(torch.int32)
        if forward_mode.is_target_verify():
            kv_lens = kv_lens + n_draft
        else:
            # Padded rows carry the fill seq_len (1); keep kv_len >= q_len.
            kv_lens = kv_lens.clamp_min(n_draft)
        kv_indptr[1 : bs + 1] = torch.cumsum(kv_lens, dim=0)
        kv_indices = self.cuda_graph_extend_kv_indices
        _create_kv_indices(self.req_to_token, req_pool_indices[:bs], kv_lens, kv_indptr, kv_indices)
        qo_indptr = self.qo_indptr[: bs + 1]
        torch.mul(self.decode_cu_seqlens_q[: bs + 1], n_draft, out=qo_indptr)
        return ForwardMetadata(
            attn_logits=None,
            attn_lse=None,
            max_extend_len=n_draft,
            num_kv_splits=None,
            kv_indptr=kv_indptr,
            kv_indices=kv_indices,
            qo_indptr=qo_indptr,
            custom_mask=self.empty_custom_mask,
            mask_indptr=self.mask_indptr[: bs + 1].zero_(),
            kv_seqlen_max=int(self.max_context_len),
            kv_len=int(kv_indices.numel()),
        )

    def get_cuda_graph_seq_len_fill_value(self):
        return 1

    # ---- kernels -------------------------------------------------------------

    @staticmethod
    def _logit_cap(layer: RadixAttention) -> float:
        cap = getattr(layer, "logit_cap", 0.0) or 0.0
        return float(cap)

    def _save_kv(
        self,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        k: torch.Tensor,
        v: torch.Tensor,
        save_kv_cache: bool,
    ) -> None:
        if not save_kv_cache or k is None or v is None:
            return
        # k/v are slices of the fused qkv output. PTPU index_put from a
        # strided source is ~200x slower than from a contiguous one.
        k = k.contiguous()
        v = v.contiguous()
        # Match TritonAttnBackend: pass optional KV scales when present.
        k_scale = getattr(layer, "k_scale", None)
        v_scale = getattr(layer, "v_scale", None)
        if k_scale is not None and v_scale is not None:
            forward_batch.token_to_kv_pool.set_kv_buffer(
                layer,
                forward_batch.out_cache_loc,
                k,
                v,
                k_scale,
                v_scale,
            )
        else:
            forward_batch.token_to_kv_pool.set_kv_buffer(
                layer,
                forward_batch.out_cache_loc,
                k,
                v,
            )

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
    ):
        if self._use_paged_decode(layer):
            return self._forward_decode_paged(
                q, k, v, layer, forward_batch, save_kv_cache
            )
        if not self._ptpu_attn_supported(layer):
            if self._flash_attn_varlen is not None:
                self._maybe_log_fallback(layer, "FlagGems flash_attn_varlen")
                return self._forward_flaggems(
                    q,
                    k,
                    v,
                    layer,
                    forward_batch,
                    save_kv_cache,
                    is_decode=True,
                )
            self._maybe_log_fallback(layer, "TorchNative SDPA")
            return self._native.forward_decode(
                q, k, v, layer, forward_batch, save_kv_cache=save_kv_cache
            )

        q = q.reshape(-1, layer.tp_q_head_num * layer.qk_head_dim)
        if layer.qk_head_dim != layer.v_head_dim:
            o = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
        else:
            o = torch.empty_like(q)

        self._save_kv(layer, forward_batch, k, v, save_kv_cache)

        meta = self.forward_metadata
        assert meta is not None and meta.attn_logits is not None
        assert meta.attn_lse is not None and meta.num_kv_splits is not None

        self._decode_attention_fwd(
            q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
            forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id),
            forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id),
            o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
            meta.kv_indptr,
            meta.kv_indices,
            meta.attn_logits,
            meta.attn_lse,
            meta.num_kv_splits,
            # The kernel takes the split count here (grid dim and split-sum shared
            # memory), whatever the wrapper calls it; the context length (204800 for
            # MiniMax-M2.7) fails to launch or raises a hardware exception.
            int(meta.attn_logits.shape[2]),
            float(layer.scaling),
            self._logit_cap(layer),
        )
        return o

    def _forward_decode_paged(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool,
    ) -> torch.Tensor:
        heads, dim = layer.tp_q_head_num, layer.qk_head_dim
        q = q.reshape(-1, heads, dim).contiguous()
        o = q.new_empty((q.shape[0], heads * dim))
        self._save_kv(layer, forward_batch, k, v, save_kv_cache)
        if q.shape[0] == 0:
            return o
        meta = self.forward_metadata
        assert meta is not None
        self._paged_decode(
            q,
            forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id),
            forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id),
            o.view(-1, heads, dim),
            meta.kv_indptr,
            meta.kv_indices,
            float(layer.scaling),
            self._logit_cap(layer),
        )
        return o

    def _forward_extend_paged_prefill(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool,
    ) -> torch.Tensor:
        o = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
        # Kernel reads the whole sequence (prefix + new tokens) from the pool.
        self._save_kv(layer, forward_batch, k, v, save_kv_cache)

        meta = self.forward_metadata
        assert meta is not None and meta.qo_indptr is not None
        assert meta.max_extend_len is not None
        self._paged_prefill_attention_fwd(
            q.contiguous().view(-1, layer.tp_q_head_num, layer.qk_head_dim),
            forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id),
            forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id),
            o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
            meta.qo_indptr,
            meta.kv_indptr,
            meta.kv_indices,
            meta.custom_mask,
            meta.mask_indptr,
            meta.max_extend_len,
            float(layer.scaling),
            self._logit_cap(layer),
            True,
        )
        return o

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
    ):
        meta = self.forward_metadata
        if meta is not None and meta.spec_rows:
            if not self._use_paged_decode(layer):
                raise NotImplementedError(
                    "SunriseAttnBackend: speculative decode rows need the paged decode kernel "
                    "for every attention layer (set SGLANG_FL_SUNRISE_SPEC_DECODE_ROWS=0)"
                )
            return self._forward_decode_paged(q, k, v, layer, forward_batch, save_kv_cache)
        if self._use_paged_prefill(layer):
            return self._forward_extend_paged_prefill(
                q, k, v, layer, forward_batch, save_kv_cache
            )
        if not self._ptpu_attn_supported(layer):
            if self._flash_attn_varlen is not None:
                self._maybe_log_fallback(layer, "FlagGems flash_attn_varlen")
                if (
                    layer.is_cross_attention
                    or layer.attn_type == AttentionType.ENCODER_ONLY
                ):
                    return self._native.forward_extend(
                        q, k, v, layer, forward_batch, save_kv_cache=save_kv_cache
                    )
                return self._forward_flaggems(
                    q,
                    k,
                    v,
                    layer,
                    forward_batch,
                    save_kv_cache,
                    is_decode=False,
                )
            self._maybe_log_fallback(layer, "TorchNative SDPA")
            return self._native.forward_extend(
                q, k, v, layer, forward_batch, save_kv_cache=save_kv_cache
            )

        if layer.qk_head_dim != layer.v_head_dim:
            o = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
        else:
            o = torch.empty_like(q)

        # Must write new K/V before prefix_caching (kernel only reads buffers).
        self._save_kv(layer, forward_batch, k, v, save_kv_cache)

        if (
            layer.is_cross_attention
            or layer.attn_type == AttentionType.ENCODER_ONLY
        ):
            raise NotImplementedError(
                "SunriseAttnBackend: cross / encoder-only attention not supported"
            )

        meta = self.forward_metadata
        assert meta is not None and meta.qo_indptr is not None
        assert meta.max_extend_len is not None
        assert meta.custom_mask is not None and meta.mask_indptr is not None

        self._prefix_caching_fwd(
            q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
            forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id),
            forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id),
            o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
            meta.qo_indptr,
            meta.kv_indptr,
            meta.kv_indices,
            meta.custom_mask,
            meta.mask_indptr,
            meta.max_extend_len,
            float(layer.scaling),
            self._logit_cap(layer),
        )
        return o


class SunriseMultiStepDraftBackend:
    """EAGLE chain draft decode for ``speculative_num_steps > 1`` (``topk == 1``): one
    :class:`SunriseAttnBackend` per draft forward; step i reads ``seq_lens + i + 1``
    slots through the paged decode kernel (counterpart of SGLang's
    ``TritonMultiStepDraftBackend``)."""

    def __init__(self, model_runner: ModelRunner, topk: int, speculative_num_steps: int):
        if topk != 1:
            raise NotImplementedError("sunrise multi-step draft supports EAGLE chains (topk == 1) only")
        self.topk = topk
        self.speculative_num_steps = speculative_num_steps
        self.device = model_runner.device
        self.attn_backends: list = [
            SunriseAttnBackend(model_runner) for _ in range(speculative_num_steps - 1)
        ]
        self.cuda_graph_kv_indices: list = []

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        bs = forward_batch.batch_size
        total = SunriseAttnBackend._host_seq_lens_sum(forward_batch, bs)
        for i, backend in enumerate(self.attn_backends):
            backend.init_draft_decode_metadata(
                bs, forward_batch.req_pool_indices, forward_batch.seq_lens, i + 1, seq_lens_sum=total
            )

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        backend = self.attn_backends[0]
        n = backend._kv_pool_tokens + max(int(max_bs), int(max_num_tokens)) * self.speculative_num_steps + 1
        self.cuda_graph_kv_indices = [
            torch.zeros((n,), dtype=torch.int32, device=self.device) for _ in self.attn_backends
        ]

    def _graph_metadata(self, forward_batch: ForwardBatch, bs: int):
        for i, backend in enumerate(self.attn_backends):
            backend.init_draft_decode_metadata(
                bs, forward_batch.req_pool_indices, forward_batch.seq_lens, i + 1,
                kv_indices=self.cuda_graph_kv_indices[i],
            )

    def init_forward_metadata_capture_cuda_graph(self, forward_batch: ForwardBatch):
        self._graph_metadata(forward_batch, forward_batch.batch_size)

    def init_forward_metadata_replay_cuda_graph(self, forward_batch: ForwardBatch, bs: int):
        self._graph_metadata(forward_batch, bs)
