# Copyright (c) 2026 BAAI. All rights reserved.
"""Sunrise / PTPU FLA adapter for ``chunk_gated_delta_rule``.

Drives the fused 6-stage PTPU macro from ``torch_ptpu.sgl_kernel``::

    chunk_gated_delta_rule(
        o, h, q, k, v, g, beta, scale,
        initial_state, initial_state_indices,
        cu_seqlens, chunk_indices, chunk_offsets, chunk_indices_phase1,
    )

See device binding notes: ``o`` / ``h`` pre-allocated; ``cu_seqlens`` and all
``chunk_*`` tables int32 on the same device as ``q``.
"""

from __future__ import annotations

from typing import Optional

import torch

from .fla_chunk_meta import prepare_chunk_tables_cached
from .fla_state_layout import check_native_pool, native_state_layout, run_in_sglang_layout
from .fla_l2norm import l2norm_fwd

# Must match PTPU fused macro / FLA ``CHUNK_SIZE``.
_BT = 64
# ``solve_tril`` phase-1 table inside the macro (PTPU linear_attn bindings).
_BT_PHASE1 = 16


def _original(fn_name: str):
    from sglang_fl.dispatch.fla_patch import get_original

    fn = get_original(fn_name)
    if fn is None:
        raise RuntimeError(
            f"FLA original '{fn_name}' not available — fla_patch not applied yet"
        )
    return fn


def _to_ptpu_state_layout(state: torch.Tensor) -> torch.Tensor:
    """SGLang ``[*, H, V, K]`` → PTPU fused macro ``[*, H, K, V]`` (float32)."""
    return state.transpose(-1, -2).contiguous().to(dtype=torch.float32)


def _from_ptpu_state_layout(
    state: torch.Tensor, dtype: torch.dtype
) -> torch.Tensor:
    """PTPU ``[*, H, K, V]`` → SGLang ``[*, H, V, K]``.

    Contiguous on purpose: the result is scattered with ``index_put``, which
    is ~50x slower on PTPU from a strided (transposed) source.
    """
    return state.transpose(-1, -2).to(dtype=dtype).contiguous()


def chunk_gated_delta_rule_sunrise(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: Optional[torch.Tensor] = None,
    initial_state_indices: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    head_first: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
):
    """SGLang/FLA API → PTPU fused ``chunk_gated_delta_rule`` macro.

    Returns ``(o, None, h)`` to match the CUDA vendor 3-tuple contract
    (``o, _, _ = chunk_gated_delta_rule(...)``).

    When ``cu_seqlens`` is missing the fused kernel cannot run; fall back to
    SGLang's original Triton/FLA implementation.
    """
    if cu_seqlens is None:
        original = _original("chunk_gated_delta_rule")

        def call(state, indices):
            return original(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                scale=scale,
                initial_state=state,
                initial_state_indices=indices,
                cu_seqlens=cu_seqlens,
                head_first=head_first,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            )

        if initial_state is not None and native_state_layout():
            return run_in_sglang_layout(initial_state, initial_state_indices, call)
        return call(initial_state, initial_state_indices)

    if head_first:
        from einops import rearrange

        q, k, v, beta, g = [
            rearrange(x, "b h t ... -> b t h ...") for x in (q, k, v, beta, g)
        ]

    # Prefer PTPU ``ptpu_l2_normalize`` (via local l2norm wrapper).
    if use_qk_l2norm_in_kernel:
        q = l2norm_fwd(q)
        k = l2norm_fwd(k)

    if scale is None:
        scale = k.shape[-1] ** -0.5

    # Fused macro expects contiguous activations (packing / rearrange / l2norm
    # often leave non-contiguous views).
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    g = g.contiguous()
    beta = beta.contiguous()

    # Metadata: int32 on q.device (kernel contract). One host sync builds all
    # three tables, shared by every GDN layer of the forward; NT comes from
    # the host sum (no ``offsets[-1].item()``).
    (
        cu_seqlens_i32,
        chunk_indices,
        chunk_offsets,
        chunk_indices_phase1,
        NT,
    ) = prepare_chunk_tables_cached(cu_seqlens, q.device, _BT, _BT_PHASE1)

    B = q.shape[0]
    # GDN may use different head counts for q/k vs v; PTPU checks h vs **v** heads.
    H = v.shape[2]
    K = k.shape[-1]
    V = v.shape[-1]

    # Caller-owned outputs for the fused macro.
    o = torch.empty_like(v)
    # Per-chunk state ``(B, NT, H, V, K)`` — same layout as FLA / chunk_fwd_o.
    h = k.new_empty(B, NT, H, V, K)

    # State pool: SGLang is ``[*, H, V, K]``; PTPU fused macro is ``[*, H, K, V]``.
    # Gather → transpose → in-place update → transpose → scatter back.
    ptpu_state: Optional[torch.Tensor] = None
    ptpu_state_indices: Optional[torch.Tensor] = None
    native = initial_state is not None and native_state_layout()
    if native:
        # Pool already in PTPU layout: the macro updates the slots in place.
        check_native_pool(initial_state)
        ptpu_state = initial_state
        if initial_state_indices is not None:
            ptpu_state_indices = initial_state_indices.to(dtype=torch.int32).contiguous()
        else:
            ptpu_state_indices = torch.arange(
                initial_state.shape[0], dtype=torch.int32, device=q.device
            )
    elif initial_state is not None:
        selected = (
            initial_state[initial_state_indices]
            if initial_state_indices is not None
            else initial_state
        )
        ptpu_state = _to_ptpu_state_layout(selected)
        ptpu_state_indices = torch.arange(
            ptpu_state.shape[0], dtype=torch.int32, device=q.device
        ).contiguous()

    from torch_ptpu.sgl_kernel import chunk_gated_delta_rule as _ptpu_fused

    _ptpu_fused(
        o,
        h,
        q,
        k,
        v,
        g,
        beta,
        float(scale),
        ptpu_state,
        ptpu_state_indices,
        cu_seqlens_i32,
        chunk_indices,
        chunk_offsets,
        chunk_indices_phase1,
    )

    if not native and initial_state is not None and ptpu_state is not None:
        final_vk = _from_ptpu_state_layout(ptpu_state, initial_state.dtype)
        if initial_state_indices is not None:
            initial_state[initial_state_indices] = final_vk
        else:
            initial_state.copy_(final_vk)

    return o, None, h
