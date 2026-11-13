# Copyright (c) 2026 BAAI. All rights reserved.
"""MTP target verify for GDN layers on PTPU (EAGLE chain, ``topk == 1``).

SGLang's Triton ``fused_sigmoid_gating_delta_rule_update`` (verify mode) keeps
a ``[K, 32]`` fp32 state tile per warp and reduces it along K, the layout that
hangs PTPU. Here the vendor kernel
``torch_ptpu.sgl_kernel.fused_sigmoid_gating_delta_rule_update`` runs once per
draft position and updates the pool slot in place; after every position but
the last the slot is copied to ``intermediate_states_buffer[row, step]``. The
pool therefore ends up holding the state after the last draft token, and
:func:`scatter_accepted` only rewrites requests that rejected a draft.

The pool and the intermediate buffer share the PTPU ``[HV, K, V]`` state
layout (``_state_layout``). The vendor kernel reads ``a`` / ``b`` as dense
``[tokens, HV]`` rows (strided views give wrong results); q / k / v may be
row-strided.
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

from .fla_packed_decode import decode_chunk_rows

_ARANGE: dict = {}
# 2048-element blocks x 16 per program: 0.098 ms for 24 x 2 MB states (clone 0.085 ms);
# one block per program is ~2.7x slower, int64-typed loads ~35x slower.
_COPY_BLOCK = int(os.environ.get("SGLANG_FL_SUNRISE_GDN_VERIFY_COPY_BLOCK", "2048"))
_COPY_ITERS = int(os.environ.get("SGLANG_FL_SUNRISE_GDN_VERIFY_COPY_ITERS", "16"))
_COPY_WARPS = int(os.environ.get("SGLANG_FL_SUNRISE_GDN_VERIFY_COPY_WARPS", "4"))


@triton.jit
def _copy_rows_kernel(
    src, dst, src_rows, dst_rows, n_src_rows, n_dst_rows, row_elems,
    src_layer_stride, dst_layer_stride, BLOCK: tl.constexpr, ITERS: tl.constexpr,
):
    i = tl.program_id(0)
    layer = tl.program_id(1).to(tl.int64)
    base = tl.program_id(2).to(tl.int64) * BLOCK * ITERS
    sr = tl.load(src_rows + i).to(tl.int64)
    dr = tl.load(dst_rows + i).to(tl.int64)
    if (sr >= 0) & (sr < n_src_rows) & (dr >= 0) & (dr < n_dst_rows):
        s = src + layer * src_layer_stride + sr * row_elems + base
        d = dst + layer * dst_layer_stride + dr * row_elems + base
        offs = tl.arange(0, BLOCK)
        for it in tl.static_range(ITERS):
            o = it * BLOCK + offs
            m = base + o < row_elems
            tl.store(d + o, tl.load(s + o, mask=m), mask=m)


def copy_rows(
    src: torch.Tensor, dst: torch.Tensor, src_rows: torch.Tensor, dst_rows: torch.Tensor,
    row_elems: int, layers: int = 1,
) -> None:
    """``dst_l[dst_rows[i]] = src_l[src_rows[i]]`` for each layer ``l`` and each ``i`` with
    both rows ``>= 0``; ``src`` / ``dst`` are contiguous ``[layers, rows, row_elems]``
    buffers (any trailing shape). Replaces ``index_select`` + strided ``copy_``
    (6.6 ms for 24 GDN states on PTPU)."""
    n = src_rows.shape[0]
    if n == 0:
        return
    assert src.is_contiguous() and dst.is_contiguous()
    n_src = src.numel() // (layers * row_elems)
    n_dst = dst.numel() // (layers * row_elems)
    _copy_rows_kernel[(n, layers, triton.cdiv(row_elems, _COPY_BLOCK * _COPY_ITERS))](
        src, dst, src_rows, dst_rows, n_src, n_dst, row_elems,
        n_src * row_elems, n_dst * row_elems,
        BLOCK=_COPY_BLOCK, ITERS=_COPY_ITERS, num_warps=_COPY_WARPS,
    )


def _arange(n: int, device) -> torch.Tensor:
    key = (str(device), torch.int32)
    buf = _ARANGE.get(key)
    if buf is None or buf.numel() < n:
        buf = torch.arange(max(n, 512), dtype=torch.int32, device=device)
        _ARANGE[key] = buf
    return buf[:n]


@triton.jit
def _conv_verify_kernel(
    x, state, w, bias, out, slots, n_slots, D,
    stride_xb, stride_xt, stride_sb, stride_sd, stride_wd, stride_ob, stride_ot,
    T: tl.constexpr, HAS_BIAS: tl.constexpr, SILU: tl.constexpr, BLOCK: tl.constexpr,
):
    b = tl.program_id(0)
    d = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    m = d < D
    slot = tl.load(slots + b).to(tl.int64)
    has_state = (slot >= 0) & (slot < n_slots)
    s = state + slot * stride_sb + d * stride_sd
    c0 = tl.load(s, mask=m & has_state, other=0.0).to(tl.float32)
    c1 = tl.load(s + 1, mask=m & has_state, other=0.0).to(tl.float32)
    c2 = tl.load(s + 2, mask=m & has_state, other=0.0).to(tl.float32)
    w0 = tl.load(w + d * stride_wd, mask=m, other=0.0).to(tl.float32)
    w1 = tl.load(w + d * stride_wd + 1, mask=m, other=0.0).to(tl.float32)
    w2 = tl.load(w + d * stride_wd + 2, mask=m, other=0.0).to(tl.float32)
    w3 = tl.load(w + d * stride_wd + 3, mask=m, other=0.0).to(tl.float32)
    if HAS_BIAS:
        acc0 = tl.load(bias + d, mask=m, other=0.0).to(tl.float32)
    else:
        acc0 = tl.zeros([BLOCK], dtype=tl.float32)
    for t in tl.static_range(T):
        xt = tl.load(x + b * stride_xb + t * stride_xt + d, mask=m, other=0.0).to(tl.float32)
        acc = acc0 + c0 * w0 + c1 * w1 + c2 * w2 + xt * w3
        if SILU:
            acc = acc / (1.0 + tl.exp(-acc))
        tl.store(out + b * stride_ob + t * stride_ot + d, acc.to(out.dtype.element_ty), mask=m)
        c0 = c1
        c1 = c2
        c2 = xt


@triton.jit
def _conv_windows_kernel(
    x, state, win, slots, rows, n_slots, D,
    stride_xb, stride_xt, stride_sb, stride_win_r, stride_win_t,
    T: tl.constexpr, BLOCK: tl.constexpr,
):
    # Flat [D, 3] window row: element e is channel e // 3, column e % 3. The window
    # after token t is [state | x_0 .. x_t][t + 1 + col]: a shifted contiguous load
    # from the state row, or token t + col - 2 of x. Stores are contiguous.
    b = tl.program_id(0)
    e = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    m = e < 3 * D
    d = e // 3
    col = e - d * 3
    slot = tl.load(slots + b).to(tl.int64)
    row = tl.load(rows + b).to(tl.int64)
    has_state = (slot >= 0) & (slot < n_slots)
    s = state + slot * stride_sb
    xb = x + b * stride_xb + d
    for t in tl.static_range(T):
        src = col + t + 1
        from_state = src < 3
        sv = tl.load(s + e + t + 1, mask=m & from_state & has_state, other=0.0)
        xv = tl.load(xb + (src - 3) * stride_xt, mask=m & (src >= 3), other=0.0)
        v = tl.where(from_state, sv, xv)
        tl.store(win + row * stride_win_r + t * stride_win_t + e, v, mask=m & (row >= 0))


def conv_verify_supported(x, conv_state, weight, intermediate_conv_window) -> bool:
    """``x``: ``[B, D, T]`` view of token-major ``[B * T, D]`` rows (as ``gdn_backend`` passes it)."""
    return (
        x.device.type == "ptpu" and x.dim() == 3 and x.stride(1) == 1
        and weight.dim() == 2 and weight.shape[1] == 4 and weight.stride(1) == 1
        and conv_state.dim() == 3 and conv_state.shape[2] == 3
        and conv_state.stride(2) == 1 and conv_state.stride(1) == 3
        and intermediate_conv_window.dim() == 4 and intermediate_conv_window.shape[1] == x.shape[2]
        and intermediate_conv_window.shape[3] == 3 and intermediate_conv_window.stride(3) == 1
        and intermediate_conv_window.stride(2) == 3
        and conv_state.dtype == intermediate_conv_window.dtype == x.dtype
    )


def conv_verify(x, conv_state, weight, bias, activation, conv_state_indices,
                intermediate_conv_window, intermediate_state_indices) -> torch.Tensor:
    """Target-verify ``causal_conv1d_update`` (width 4): outputs for the ``T`` draft tokens
    and the conv window after each of them in ``intermediate_conv_window[row, t]``.

    0.056 ms per layer at TP1 / 24 requests / T=2 (SGLang's Triton update: 0.63 ms).
    Unlike SGLang's update, ``conv_state`` is not rolled: :func:`scatter_accepted`
    rewrites every request's state from the windows. Returns ``[B, D, T]`` with
    token-major memory, like ``x``.
    """
    bsz, dim, t = x.shape
    out = torch.empty((bsz, t, dim), dtype=x.dtype, device=x.device)
    if bsz == 0:
        return out.transpose(1, 2)
    slots = conv_state_indices.to(torch.int32)
    block = 256
    _conv_verify_kernel[(bsz, triton.cdiv(dim, block))](
        x, conv_state, weight, bias if bias is not None else weight, out, slots,
        conv_state.shape[0], dim,
        x.stride(0), x.stride(2), conv_state.stride(0), conv_state.stride(1), weight.stride(0),
        out.stride(0), out.stride(1),
        T=t, HAS_BIAS=bias is not None, SILU=activation in ("silu", "swish", True), BLOCK=block,
        num_warps=2,
    )
    wblock = 1024
    _conv_windows_kernel[(bsz, triton.cdiv(3 * dim, wblock))](
        x, conv_state, intermediate_conv_window, slots, intermediate_state_indices.to(torch.int32),
        conv_state.shape[0], dim,
        x.stride(0), x.stride(2), conv_state.stride(0),
        intermediate_conv_window.stride(0), intermediate_conv_window.stride(1),
        T=t, BLOCK=wblock, num_warps=4,
    )
    return out.transpose(1, 2)


def target_verify(
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    ssm_states: torch.Tensor,
    cache_indices: torch.Tensor,
    intermediate_states_buffer: torch.Tensor,
    cache_steps: int,
    retrieve_parent_token=None,
    **_,
) -> torch.Tensor:
    """``q`` / ``k``: ``[1, B * T, H, K]``, ``v``: ``[1, B * T, HV, V]``, ``a`` / ``b``:
    ``[B * T, HV]`` with the ``T = cache_steps`` draft tokens of a request adjacent.
    Returns ``[1, B * T, HV, V]``."""
    if retrieve_parent_token is not None:
        raise NotImplementedError("sunrise GDN verify supports EAGLE chains (topk == 1) only")
    from torch_ptpu.sgl_kernel import fused_sigmoid_gating_delta_rule_update as vendor

    t = int(cache_steps)
    n = q.shape[1]
    bsz = n // t
    hv, dv = v.shape[2], v.shape[3]
    # Padding rows carry a negative slot; slot 0 is MambaPool's dummy slot.
    idx = cache_indices[:bsz].to(torch.int32).clamp_min(0)
    if A_log.dtype != torch.bfloat16:
        A_log = A_log.to(torch.bfloat16)
    ar = _arange(bsz + 1, q.device)
    row_elems = ssm_states[0].numel()
    out = q.new_empty((1, n, hv, dv))
    rows = decode_chunk_rows(bsz, hv) or bsz
    for j in range(t):
        a_j, b_j = a[j::t].contiguous(), b[j::t].contiguous()
        q_j, k_j, v_j, out_j = q[:, j::t], k[:, j::t], v[:, j::t], out[:, j::t]
        for s in range(0, bsz, rows):
            e = min(s + rows, bsz)
            out_j[:, s:e] = vendor(
                A_log, a_j[s:e], dt_bias, 1.0, 20.0,
                q_j[:, s:e], k_j[:, s:e], v_j[:, s:e], b_j[s:e],
                ssm_states, idx[s:e],
                scale=q.shape[3] ** -0.5, use_qk_l2norm_in_kernel=True,
                cu_seqlens=ar[: e - s + 1], is_kda=False,
            )
        if j + 1 < t:
            copy_rows(ssm_states, intermediate_states_buffer, idx, ar[:bsz] * t + j, row_elems)
    return out


def scatter_with_mask(dst: torch.Tensor, src: torch.Tensor, dst_indices: torch.Tensor, steps: torch.Tensor) -> None:
    """``dst[l, dst_indices[i]] = src[l, i, steps[i]]`` for ``steps[i] >= 0`` (all layers).

    ``dst``: ``[L, slots, *state]``, ``src``: ``[L, rows, T, *state]``, both contiguous.
    Same contract as SGLang's ``fused_mamba_state_scatter_with_mask``, which
    rejects non-CUDA tensors.
    """
    n = steps.shape[0]
    if n == 0:
        return
    assert dst.shape[0] == src.shape[0] and dst.shape[2:] == src.shape[3:]
    t = src.shape[2]
    steps = steps.to(torch.int32)
    # A program per (request, layer, chunk); skipped requests exit after two index
    # loads (0.31 ms for 2 of 24 requests over 30 layers; looping over requests
    # inside a program instead: 3.3 ms).
    src_rows = torch.where((steps >= 0) & (steps < t), _arange(n, steps.device) * t + steps, -1)
    copy_rows(src, dst, src_rows, dst_indices.to(torch.int32), dst[0, 0].numel(), layers=dst.shape[0])


def scatter_accepted(caches, state_indices: torch.Tensor, accepted_steps: torch.Tensor) -> None:
    """Commit the accepted draft position of each request (``MambaPool.SpeculativeState``).

    SSM: the pool already holds the last position (see :func:`target_verify`),
    so only requests with ``accepted_steps < T - 1`` are copied. Conv windows:
    SGLang's verify conv update rolls the pool by all draft tokens, so every
    request is rewritten from ``intermediate_conv_window``.
    """
    t = caches.intermediate_ssm.shape[2]
    ssm_steps = torch.where(accepted_steps < t - 1, accepted_steps, torch.full_like(accepted_steps, -1))
    scatter_with_mask(caches.temporal, caches.intermediate_ssm, state_indices, ssm_steps)
    for conv, window in zip(caches.conv, caches.intermediate_conv_window):
        scatter_with_mask(conv, window, state_indices, accepted_steps)
