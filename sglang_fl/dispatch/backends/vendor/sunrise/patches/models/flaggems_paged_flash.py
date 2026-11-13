# Copyright (c) 2026 BAAI. All rights reserved.

"""
Runtime monkey-patch for FlagGems' Sunrise paged FlashAttention kernel.

Why this exists
---------------
``flag_gems.runtime.backend._sunrise.ops.flash_kernel.flash_varlen_fwd_kernel``
walks the paged KV cache assuming a row-major contiguous layout, i.e. it
computes the physical address of cached token ``t`` as

    page_addr = block_table[t // block_size] * (block_size * k_row_stride)
              + (t % block_size) * k_row_stride

That formula assumes the K/V cache has ``stride(0) == block_size * k_row_stride``.

For *hybrid* models (attention + mamba / linear_attn — e.g. Qwen3.5-4B), vLLM
allocates a single shared block pool and reshapes the attention KV cache via
``_update_hybrid_attention_mamba_layout`` to the interleaved layout

    (num_pages, 2, block_size, num_kv_heads, head_dim)

(K and V slabs alternate per physical page). The resulting K view is *not*
contiguous: ``stride(0) == 2 * block_size * k_row_stride``. The kernel's
hardcoded page stride then points at the wrong page (page ``N`` reads from
slot ``N/2``), producing NaNs on the first ``full_attention`` decode step
and the "!!!!" output we observed for Qwen3.5-4B.

The upstream generic FlagGems kernel already takes a ``k_page_stride``
argument (PR #3410, merged in 5.3.0-rc2.post1); the Sunrise-vendored copy
just never got that update. This patch ports the same fix without touching
the FlagGems repo on disk: we monkey-patch the in-memory Sunrise kernel and
its helpers to accept a ``k_page_stride`` argument that is the *real*
``cache.stride(0)``.

How the patch is structured
---------------------------
1. ``fwd_params`` gains a trailing ``k_page_stride`` slot. The patched
   ``__init__`` accepts either the original positional signature OR the
   extended one, and auto-derives ``k_page_stride = k_ptr.stride(0) if
   is_paged else 0`` when the caller does not supply it. This lets the
   *unmodified* upstream ``mha_varlan_fwd`` / ``mha_fwd`` keep working —
   they construct ``fwd_params`` positionally with the legacy arg count.
2. ``virtual_to_cache`` / ``load_from_kvcache`` (Triton ``@jit`` helpers)
   gain ``k_page_stride`` / ``k_row_stride`` arguments and use them in the
   address calculation.
3. ``flash_varlen_fwd_kernel`` (Triton ``@jit`` kernel) gains a trailing
   ``k_page_stride`` parameter and forwards it to ``load_from_kvcache``.
4. ``apply_patch()`` rebinds the patched objects on
   ``flag_gems.runtime.backend._sunrise.ops.flash_kernel`` and
   ``flag_gems.runtime.backend._sunrise.ops.flash_api``. Because the upstream
   ``mha_*`` functions resolve ``fwd_params`` and ``flash_varlen_fwd_kernel``
   from their module globals at call time, rebinding the module attributes
   is sufficient — we do not need to replace the long ``mha_*`` functions.

When to remove this patch
-------------------------
Once the same ``k_page_stride`` plumbing lands in the vendor-internal
``tritonops`` Sunrise backend (currently it does not — the fix is only in
upstream FlagGems generic ``ops/flash_kernel.py``), this entire module can
be deleted and the import in ``__init__.py`` removed.
"""

from __future__ import annotations

import logging

import triton
import triton.language as tl


logger = logging.getLogger(__name__)

_PATCH_APPLIED = False


def apply_patch() -> bool:
    """Idempotently install the Sunrise paged-attention fix.

    Returns ``True`` on first successful patch, ``False`` if already applied
    or if the Sunrise FlagGems backend is unavailable (non-PTPU environment,
    different vendor, etc. — in which case the patch is a no-op).
    """
    global _PATCH_APPLIED
    if _PATCH_APPLIED:
        return False

    try:
        from flag_gems.runtime.backend._sunrise.ops import (
            attention as _gems_attention,
            flash_api as _gems_flash_api,
            flash_kernel as _gems_flash_kernel,
        )
        from flag_gems.utils import libentry
    except Exception as exc:  # pragma: no cover - non-Sunrise / no FlagGems
        logger.debug(
            "Skipping Sunrise paged-attention patch (FlagGems Sunrise backend "
            "not importable): %s",
            exc,
        )
        return False

    # ---- Pull the helpers the patched kernel needs into our namespace ----
    # The Triton compiler resolves free names via the function's __globals__,
    # so every name the patched kernel touches must exist in *this* module
    # at compile time. We re-export the originals unchanged.
    apply_softcap = _gems_flash_kernel.apply_softcap
    apply_alibi = _gems_flash_kernel.apply_alibi
    apply_mask = _gems_flash_kernel.apply_mask
    softmax_rescale = _gems_flash_kernel.softmax_rescale
    apply_dropout = _gems_flash_kernel.apply_dropout

    # ---------------------------------------------------------------------
    # Patched JIT helpers
    # ---------------------------------------------------------------------
    @triton.jit
    def virtual_to_cache(
        virtual_index,
        max_virtual_index,
        page_table_ptr,
        block_size,
        k_page_stride,
        k_row_stride,
        boundary_check: tl.constexpr = False,
    ):
        # virtual_index is the kv sequence index in the current batch element.
        # page_table_ptr is already pointed at the batch element's block table.
        # block_size is the number of tokens per physical block.
        #
        # k_page_stride is the *real* element stride between two consecutive
        # physical pages in the kv cache. It is NOT always
        # ``block_size * k_row_stride``: vLLM may allocate the cache as
        # (num_pages, 2, block_size, num_kv_heads, head_dim) and pass K/V as
        # views on dim 1, so each physical page slot contains both K and V
        # slabs and the page stride is twice the naive value. The python
        # wrapper passes the actual ``cache.stride(0)`` to avoid this layout
        # assumption.
        virtual_page_index = virtual_index // block_size
        page_offset = virtual_index % block_size
        if boundary_check:
            page_block_index = tl.load(
                page_table_ptr + virtual_page_index,
                mask=virtual_index < max_virtual_index,
                other=0,
            ).to(tl.int32)
        else:
            page_block_index = tl.load(
                page_table_ptr + virtual_page_index
            ).to(tl.int32)
        return page_block_index * k_page_stride + page_offset * k_row_stride

    @triton.jit
    def load_from_kvcache(
        virtual_index,
        max_virtual_index,
        page_table_ptr,
        k_ptr_base,
        v_ptr_base,
        block_size,
        d: tl.constexpr,
        k_row_stride,
        k_page_stride,
        BLOCK_K: tl.constexpr,
        boundary_check: tl.constexpr = False,
    ):
        kvcache_idx = virtual_to_cache(
            virtual_index,
            max_virtual_index,
            page_table_ptr,
            block_size,
            k_page_stride,
            k_row_stride,
            boundary_check,
        )
        # ``virtual_to_cache`` already produces an element offset relative to
        # ``k_ptr_base`` / ``v_ptr_base`` (page index scaled by
        # ``k_page_stride`` plus in-page row scaled by ``k_row_stride``), so
        # we only add the head-dim arange.
        k_offset = tl.arange(0, BLOCK_K)[:, None] + kvcache_idx[None, :]
        v_offset = tl.arange(0, BLOCK_K)[None, :] + kvcache_idx[:, None]
        if d == BLOCK_K:
            bK_mask = virtual_index[None, :] < max_virtual_index[None, :]
            bV_mask = virtual_index[:, None] < max_virtual_index[:, None]
            bK = tl.load(k_ptr_base + k_offset, mask=bK_mask, other=0.0)
            bV = tl.load(v_ptr_base + v_offset, mask=bV_mask, other=0.0)
        else:
            bK_mask = (tl.arange(0, BLOCK_K)[:, None] < d) & (
                virtual_index[None, :] < max_virtual_index[None, :]
            )
            bV_mask = (tl.arange(0, BLOCK_K)[None, :] < d) & (
                virtual_index[:, None] < max_virtual_index[:, None]
            )
            bK = tl.load(k_ptr_base + k_offset, mask=bK_mask, other=0.0)
            bV = tl.load(v_ptr_base + v_offset, mask=bV_mask, other=0.0)
        return bK, bV

    # ---------------------------------------------------------------------
    # Patched kernel
    # ---------------------------------------------------------------------
    @libentry()
    @triton.jit(
        do_not_specialize=[
            "q_batch_stride",
            "k_batch_stride",
            "v_batch_stride",
            "o_batch_stride",
            "b",
            "bk",
            "seqlen_q",
            "seqlen_k",
            "seqlen_q_rounded",
            "seqlen_k_rounded",
            "total_q",
            "k_page_stride",
        ]
    )
    def flash_varlen_fwd_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        o_ptr,
        p_ptr,
        softmax_lse_ptr,
        q_row_stride,
        k_row_stride,
        v_row_stride,
        q_head_stride,
        k_head_stride,
        v_head_stride,
        o_row_stride,
        o_head_stride,
        q_batch_stride,
        k_batch_stride,
        v_batch_stride,
        o_batch_stride,
        is_cu_seqlens_q: tl.constexpr,
        cu_seqlens_q_ptr,
        is_cu_seqlens_k: tl.constexpr,
        cu_seqlens_k_ptr,
        is_seqused_k: tl.constexpr,
        seqused_k_ptr,
        # sizes
        b,
        bk,
        h: tl.constexpr,
        hk: tl.constexpr,
        h_hk_ratio: tl.constexpr,
        seqlen_q,
        seqlen_k,
        seqlen_q_rounded,
        seqlen_k_rounded,
        d: tl.constexpr,
        d_rounded: tl.constexpr,
        # scaling factors
        is_softcap: tl.constexpr,
        softcap: tl.constexpr,
        scale_softmax: tl.constexpr,
        scale_softmax_log2: tl.constexpr,
        # dropout
        is_dropout: tl.constexpr,
        p_dropout: tl.constexpr,
        rp_dropout: tl.constexpr,
        p_dropout_in_uint8_t: tl.constexpr,
        philox_args,
        return_softmax: tl.constexpr,
        # causal and swa
        is_causal: tl.constexpr,
        is_local: tl.constexpr,
        window_size_left: tl.constexpr,
        window_size_right: tl.constexpr,
        seqlenq_ngroups_swapped: tl.constexpr,
        is_paged: tl.constexpr,
        # alibi
        is_alibi: tl.constexpr,
        alibi_slopes_ptr,
        alibi_slopes_batch_stride: tl.constexpr,
        # block table
        total_q,
        page_table_ptr,
        page_table_batch_stride: tl.constexpr,
        block_size: tl.constexpr,
        k_page_stride,
        # kernel params
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        num_warps: tl.constexpr,
        num_stages: tl.constexpr,
    ):
        m_block = tl.program_id(0)
        bid = tl.program_id(1)
        hid = tl.program_id(2)

        if is_cu_seqlens_q:
            q_eos = tl.load(cu_seqlens_q_ptr + bid + 1).to(tl.int32)
            q_bos = tl.load(cu_seqlens_q_ptr + bid).to(tl.int32)
            q_len = q_eos - q_bos
            q_offset = q_bos * q_row_stride
            o_offset = q_bos * o_row_stride
            lse_offset = q_bos * 1
        else:
            q_len = seqlen_q
            q_offset = bid * q_batch_stride
            o_offset = bid * o_batch_stride
            lse_offset = bid * seqlen_q

        if is_cu_seqlens_k:
            k_eos = tl.load(cu_seqlens_k_ptr + bid + 1).to(tl.int32)
            k_bos = tl.load(cu_seqlens_k_ptr + bid).to(tl.int32)
            k_len_cache = k_eos - k_bos
        else:
            k_len_cache = seqlen_k

        if is_seqused_k:
            k_len = tl.load(seqused_k_ptr + bid).to(tl.int32)
        else:
            k_len = k_len_cache

        if m_block * BLOCK_M > q_len:
            return

        is_even_mn: tl.constexpr = False

        if is_local:
            n_block_min = max(
                0,
                (m_block * BLOCK_M + k_len - q_len - window_size_left) // BLOCK_N,
            )
        else:
            n_block_min = 0

        n_block_max = tl.cdiv(k_len, BLOCK_N)
        if is_causal or is_local:
            n_block_max = min(
                n_block_max,
                tl.cdiv(
                    (m_block + 1) * BLOCK_M
                    + k_len
                    - q_len
                    + window_size_right,
                    BLOCK_N,
                ),
            )

        if is_dropout:
            philox_seed = tl.load(philox_args).to(tl.uint64)
            philox_offset = tl.load(philox_args + 1).to(tl.uint64)

        # Locate the page table entry for the current batch element
        if is_paged:
            page_table_ptr += bid * page_table_batch_stride
        q_row_offset = hid * q_head_stride
        k_row_offset = (hid // h_hk_ratio) * k_head_stride
        k_ptr_base = k_ptr + k_row_offset
        v_ptr_base = v_ptr + k_row_offset

        gQ = tl.make_block_ptr(
            base=q_ptr + q_offset + q_row_offset,
            shape=(q_len, d),
            strides=(q_row_stride, 1),
            offsets=(0, 0),
            block_shape=(BLOCK_M, BLOCK_K),
            order=(1, 0),
        )
        bQ = tl.load(gQ.advance([m_block * BLOCK_M, 0]), boundary_check=(0, 1))

        acc_ = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
        rowmax_ = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
        rowsum_ = tl.zeros([BLOCK_M], dtype=tl.float32)

        if is_alibi:
            alibi_offset = bid * alibi_slopes_batch_stride + hid
            alibi_slope = tl.load(alibi_slopes_ptr + alibi_offset)
            alibi_slope /= scale_softmax
        else:
            alibi_slope = 0.0

        if not is_causal and not is_local:
            n_masking_steps = 1
        elif is_even_mn:
            n_masking_steps = tl.cdiv(BLOCK_M, BLOCK_N)
        else:
            n_masking_steps = tl.cdiv(BLOCK_M, BLOCK_N) + 1

        n_masking_steps = min(n_block_max - n_block_min, n_masking_steps)

        row_idx = m_block * BLOCK_M + tl.arange(0, BLOCK_M)
        n_block = n_block_max - 1
        for step in tl.range(0, n_masking_steps):
            col_idx = n_block * BLOCK_N + tl.arange(0, BLOCK_N)
            if is_paged:
                bK, bV = load_from_kvcache(
                    col_idx,
                    k_len,
                    page_table_ptr,
                    k_ptr_base,
                    v_ptr_base,
                    block_size,
                    d,
                    k_row_stride,
                    k_page_stride,
                    BLOCK_K=BLOCK_K,
                    boundary_check=True,
                )
            else:
                start_n = n_block * BLOCK_N
                k_ptr_seq = k_ptr_base + k_bos * k_row_stride
                v_ptr_seq = v_ptr_base + k_bos * k_row_stride
                gK = tl.make_block_ptr(
                    base=k_ptr_seq,
                    shape=(k_len, d),
                    strides=(k_row_stride, 1),
                    offsets=(start_n, 0),
                    block_shape=(BLOCK_N, BLOCK_K),
                    order=(0, 1),
                )
                gV = tl.make_block_ptr(
                    base=v_ptr_seq,
                    shape=(k_len, d),
                    strides=(k_row_stride, 1),
                    offsets=(start_n, 0),
                    block_shape=(BLOCK_N, BLOCK_K),
                    order=(0, 1),
                )
                bK = tl.load(gK, boundary_check=(0, 1))
                bK = tl.trans(bK)
                bV = tl.load(gV, boundary_check=(0, 1))
            S = tl.dot(bQ, bK, out_dtype=tl.float32)
            S = apply_softcap(S, softcap, is_softcap)
            S = apply_alibi(
                S,
                col_idx,
                row_idx,
                q_len,
                k_len,
                is_causal=is_causal,
                is_alibi=is_alibi,
                alibi_slope=alibi_slope,
            )
            S = apply_mask(
                S,
                col_idx,
                row_idx,
                q_len,
                k_len,
                window_size_left,
                window_size_right,
                is_even_mn=is_even_mn,
                is_causal=is_causal,
                is_local=is_local,
            )

            acc_, P, rowmax_, rowsum_ = softmax_rescale(
                acc_,
                S,
                rowmax_,
                rowsum_,
                softmax_scale_log2e=scale_softmax_log2,
                is_border=True,
            )
            P = P.to(v_ptr.type.element_ty)

            if is_dropout:
                P = apply_dropout(
                    P,
                    n_block * BLOCK_N,
                    m_block * BLOCK_M,
                    k_len,
                    bid,
                    hid,
                    philox_seed,
                    philox_offset,
                    p_dropout_in_uint8_t,
                    is_dropout,
                    encode_dropout_in_sign_bit=False,
                    NUM_HEADS=h,
                    BLOCK_M=BLOCK_M,
                    BLOCK_N=BLOCK_N,
                )

            acc_ = tl.dot(P, bV, acc_)
            n_block -= 1

        for n_block in tl.range(
            n_block_max - n_masking_steps - 1, n_block_min - 1, step=-1
        ):
            col_idx = n_block * BLOCK_N + tl.arange(0, BLOCK_N)
            if is_paged:
                bK, bV = load_from_kvcache(
                    col_idx,
                    k_len,
                    page_table_ptr,
                    k_ptr_base,
                    v_ptr_base,
                    block_size,
                    d,
                    k_row_stride,
                    k_page_stride,
                    BLOCK_K=BLOCK_K,
                )
            else:
                start_n = n_block * BLOCK_N
                k_ptr_seq = k_ptr_base + k_bos * k_row_stride
                v_ptr_seq = v_ptr_base + k_bos * k_row_stride
                gK = tl.make_block_ptr(
                    base=k_ptr_seq,
                    shape=(k_len, d),
                    strides=(k_row_stride, 1),
                    offsets=(start_n, 0),
                    block_shape=(BLOCK_N, BLOCK_K),
                    order=(0, 1),
                )
                gV = tl.make_block_ptr(
                    base=v_ptr_seq,
                    shape=(k_len, d),
                    strides=(k_row_stride, 1),
                    offsets=(start_n, 0),
                    block_shape=(BLOCK_N, BLOCK_K),
                    order=(0, 1),
                )
                bK = tl.load(gK)
                bK = tl.trans(bK)
                bV = tl.load(gV)
            S = tl.dot(bQ, bK, out_dtype=tl.float32)
            S = apply_softcap(S, softcap, is_softcap)
            S = apply_alibi(
                S,
                col_idx,
                row_idx,
                q_len,
                k_len,
                is_causal=is_causal,
                is_alibi=is_alibi,
                alibi_slope=alibi_slope,
            )
            S = apply_mask(
                S,
                col_idx,
                row_idx,
                q_len,
                k_len,
                window_size_left,
                window_size_right,
                is_even_mn=True,
                is_causal=False,
                is_local=is_local,
            )

            acc_, P, rowmax_, rowsum_ = softmax_rescale(
                acc_,
                S,
                rowmax_,
                rowsum_,
                softmax_scale_log2e=scale_softmax_log2,
                is_border=is_local,
            )
            P = P.to(v_ptr.type.element_ty)

            if is_dropout:
                P = apply_dropout(
                    P,
                    m_block * BLOCK_M,
                    n_block * BLOCK_N,
                    k_len,
                    bid,
                    hid,
                    philox_seed,
                    philox_offset,
                    p_dropout_in_uint8_t,
                    is_dropout,
                    encode_dropout_in_sign_bit=False,
                    NUM_HEADS=h,
                    BLOCK_M=BLOCK_M,
                    BLOCK_N=BLOCK_N,
                )
            acc_ = tl.dot(P, bV, acc_)

        # LSE
        lse = tl.where(
            rowsum_ == 0 | (rowsum_ != rowsum_),
            float("inf"),
            rowmax_ * scale_softmax + tl.log(rowsum_),
        )
        inv_sum = tl.where(
            rowsum_ == 0 | (rowsum_ != rowsum_), 1.0, 1.0 / rowsum_
        )

        acc_ *= inv_sum[:, None]

        out = acc_.to(o_ptr.type.element_ty)  # noqa

        # Write back output
        o_row_offset = hid * o_head_stride
        gO = tl.make_block_ptr(
            base=o_ptr + o_offset + o_row_offset,
            shape=(q_len, d),
            strides=(o_row_stride, 1),
            offsets=(0, 0),
            block_shape=(BLOCK_M, BLOCK_K),
            order=(1, 0),
        )
        tl.store(
            gO.advance([m_block * BLOCK_M, 0]), out, boundary_check=(0, 1)
        )

        # Write back lse
        softmax_lse_ptr += hid * total_q
        lse_row_offset = lse_offset + m_block * BLOCK_M + tl.arange(0, BLOCK_M)
        tl.store(
            softmax_lse_ptr + lse_row_offset,
            lse,
            mask=lse_row_offset < (lse_offset + q_len),
        )

    # ---------------------------------------------------------------------
    # Patched fwd_params
    # ---------------------------------------------------------------------
    # Append ``k_page_stride`` to the slot tuple in the order matching the
    # patched kernel's parameter list. The args tuple built by
    # ``mha_*_fwd`` via ``getattr(params, k) for k in params.__slots__`` is
    # passed positionally to the kernel, so order is critical here.
    _OriginalParams = _gems_flash_api.fwd_params
    _PATCHED_SLOTS = tuple(_OriginalParams.__slots__) + ("k_page_stride",)
    if "k_page_stride" in _OriginalParams.__slots__:
        # Already includes the new slot — assume someone (or a previous patch)
        # already provides it. Avoid double-extending.
        _PATCHED_SLOTS = tuple(_OriginalParams.__slots__)

    class fwd_params:
        """Drop-in replacement that auto-derives ``k_page_stride``.

        Accepts either the original positional signature (legacy callers that
        don't know about ``k_page_stride``) or the extended one with the new
        slot appended. When the extra arg is missing, it is derived as
        ``k_ptr.stride(0) if is_paged else 0`` so the patched kernel sees
        the real physical page stride.
        """

        __slots__ = _PATCHED_SLOTS

        def __init__(self, *args):
            slots = self.__slots__
            n_slots = len(slots)
            n_args = len(args)
            if n_args == n_slots:
                # Caller already supplied k_page_stride explicitly.
                for slot, val in zip(slots, args):
                    object.__setattr__(self, slot, val)
            elif n_args == n_slots - 1:
                # Legacy caller — fill in everything else, then derive
                # k_page_stride from the just-stored k_ptr / is_paged.
                for slot, val in zip(slots[:-1], args):
                    object.__setattr__(self, slot, val)
                is_paged = getattr(self, "is_paged", False)
                k_ptr = getattr(self, "k_ptr", None)
                if is_paged and k_ptr is not None and hasattr(k_ptr, "stride"):
                    derived = k_ptr.stride(0)
                else:
                    derived = 0
                object.__setattr__(self, "k_page_stride", derived)
            else:
                raise TypeError(
                    f"fwd_params expected {n_slots - 1} or {n_slots} "
                    f"positional args, got {n_args}"
                )

        def args(self):
            return tuple(getattr(self, k) for k in self.__slots__)

    # ---------------------------------------------------------------------
    # Install the patch on the FlagGems Sunrise modules
    # ---------------------------------------------------------------------
    # Helpers — only really needed for completeness, since the patched kernel
    # already binds them via its own __globals__. We rebind anyway in case
    # any other code in the FlagGems Sunrise backend reuses them.
    _gems_flash_kernel.virtual_to_cache = virtual_to_cache
    _gems_flash_kernel.load_from_kvcache = load_from_kvcache

    # Kernel & fwd_params — these MUST be rebound on flash_api too because
    # mha_varlan_fwd / mha_fwd resolve them via their own module globals at
    # call time:
    _gems_flash_kernel.flash_varlen_fwd_kernel = flash_varlen_fwd_kernel
    _gems_flash_api.flash_varlen_fwd_kernel = flash_varlen_fwd_kernel
    _gems_flash_api.fwd_params = fwd_params

    # ``attention.py`` does ``from flash_api import mha_fwd, mha_varlan_fwd``
    # so it has direct references. Those functions still work unchanged
    # because they look up ``fwd_params`` / ``flash_varlen_fwd_kernel`` in
    # ``flash_api``'s globals, which we just rebound above. No need to touch
    # ``_gems_attention``.
    _ = _gems_attention  # kept for explicitness; no rebinding required.

    _PATCH_APPLIED = True
    logger.info(
        "Applied FlagGems Sunrise paged-attention k_page_stride patch "
        "(in-process; no on-disk FlagGems edits)."
    )
    return True


def patch_flaggems_paged_flash() -> bool:
    """SGLang entry point used by ``patch.py`` and ``attention_backend``."""
    return apply_patch()


__all__ = ["apply_patch", "patch_flaggems_paged_flash"]
