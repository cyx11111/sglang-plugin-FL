# Copyright (c) 2026 BAAI. All rights reserved.
"""GDN prefill: causal conv1d + SiLU + q/k/v split + q/k L2 norm in one pass.

Replaces, for the channel-last ``mixed_qkv [T, dim]`` of ``GDNAttnBackend``:
SGLang's Triton ``causal_conv1d_fn`` (incl. the conv-state update), the
``torch.split`` views, ``l2norm_fwd`` of q and k and the ``v.contiguous()``
copy that the PTPU chunk macro needs. At TP1 / 8192 tokens that chain costs
~2.5 ms per GDN layer.

One program handles one 128-channel head of one ``BT``-token chunk of one
sequence and walks its tokens in order, keeping the previous three inputs in
registers; every value is a 1-D ``[D]`` vector. The conv arithmetic (bf16
products accumulated in fp32, SiLU, bf16 rounding of the conv output) matches
SGLang's kernel; the L2 norm is FLA's ``x / sqrt(sum(x^2) + eps)`` in fp32.
As in SGLang, only the program of a sequence's first chunk reads the initial
conv state and writes the final one, so there is no read/write race.
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

_BT = int(os.environ.get("SGLANG_FL_SUNRISE_GDN_PREFILL_FUSED_BT", "128"))
# 4 warps (cross-warp tl.sum every token) is ~70x slower on PTPU.
_NUM_WARPS = int(os.environ.get("SGLANG_FL_SUNRISE_GDN_PREFILL_FUSED_WARPS", "1"))


@triton.jit
def _conv_split_l2norm_kernel(
    x_ptr,  # [T, dim] bf16, channel-last
    w_ptr,  # [dim, 4]
    bias_ptr,
    conv_state_ptr,  # [slots, dim, >=3]
    cache_idx_ptr,  # [S]
    has_init_ptr,  # [S] bool
    qsl_ptr,  # [S + 1] int32 query_start_loc
    out_ptr,  # q [T, HK, D] ++ k [T, HK, D] ++ v [T, HV, D]
    n_tok,
    stride_x_tok,
    stride_w_dim,
    stride_w_width,
    stride_cs_seq,
    stride_cs_dim,
    stride_cs_tok,
    eps,
    HK: tl.constexpr,
    HV: tl.constexpr,
    D: tl.constexpr,
    BT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_INIT: tl.constexpr,
    SILU: tl.constexpr,
):
    i_s = tl.program_id(0)
    i_c = tl.program_id(1)
    i_h = tl.program_id(2)

    start = tl.load(qsl_ptr + i_s).to(tl.int64)
    seqlen = (tl.load(qsl_ptr + i_s + 1) - tl.load(qsl_ptr + i_s)).to(tl.int32)
    t0 = i_c * BT
    if t0 >= seqlen:
        return
    slot = tl.load(cache_idx_ptr + i_s).to(tl.int64)
    if slot < 0:
        return

    feats = i_h * D + tl.arange(0, D)
    x_base = x_ptr + start * stride_x_tok + feats
    cs_base = conv_state_ptr + slot * stride_cs_seq + feats * stride_cs_dim

    w0 = tl.load(w_ptr + feats * stride_w_dim + 0 * stride_w_width)
    w1 = tl.load(w_ptr + feats * stride_w_dim + 1 * stride_w_width)
    w2 = tl.load(w_ptr + feats * stride_w_dim + 2 * stride_w_width)
    w3 = tl.load(w_ptr + feats * stride_w_dim + 3 * stride_w_width)
    if HAS_BIAS:
        acc0 = tl.load(bias_ptr + feats).to(tl.float32)
    else:
        acc0 = tl.zeros([D], dtype=tl.float32)

    if i_c == 0:
        load_init = False
        if HAS_INIT:
            load_init = tl.load(has_init_ptr + i_s).to(tl.int1)
        if load_init:
            c0 = tl.load(cs_base + 0 * stride_cs_tok)
            c1 = tl.load(cs_base + 1 * stride_cs_tok)
            c2 = tl.load(cs_base + 2 * stride_cs_tok)
        else:
            c0 = tl.zeros([D], dtype=x_ptr.dtype.element_ty)
            c1 = tl.zeros([D], dtype=x_ptr.dtype.element_ty)
            c2 = tl.zeros([D], dtype=x_ptr.dtype.element_ty)
        # New state = last 3 inputs of (old state ++ x); old slots are zero
        # when there is no initial state.
        for j in tl.static_range(3):
            src = seqlen - 3 + j
            xv = tl.load(x_base + tl.maximum(src, 0).to(tl.int64) * stride_x_tok, mask=(src >= 0) & (feats >= 0),
                         other=0.0)
            old_i = j + seqlen
            old = tl.where(old_i == 0, c0, tl.where(old_i == 1, c1, c2))
            tl.debug_barrier()
            tl.store(cs_base + j * stride_cs_tok, tl.where(src >= 0, xv, old))
    else:
        c0 = tl.load(x_base + (t0 - 3).to(tl.int64) * stride_x_tok)
        c1 = tl.load(x_base + (t0 - 2).to(tl.int64) * stride_x_tok)
        c2 = tl.load(x_base + (t0 - 1).to(tl.int64) * stride_x_tok)

    # q / k / v regions are consecutive in one buffer; an offset (not a
    # branch-selected pointer) keeps the stores vectorised (~35x on PTPU).
    region = tl.where(i_h < HK, 0, tl.where(i_h < 2 * HK, 1, 2))
    row = tl.where(i_h < 2 * HK, HK * D, HV * D)
    o_off = region.to(tl.int64) * n_tok * (HK * D) + (i_h - region * HK) * D
    do_norm = i_h < 2 * HK
    o_tok = out_ptr + o_off + (start + t0) * row + tl.arange(0, D)
    x_tok = x_base + t0.to(tl.int64) * stride_x_tok

    t_end = tl.minimum(t0 + BT, seqlen)
    for _ in range(t0, t_end):
        xt = tl.load(x_tok)
        acc = acc0 + c0 * w0
        acc = acc + c1 * w1
        acc = acc + c2 * w2
        acc = acc + xt * w3
        if SILU:
            acc = acc / (1 + tl.exp(-acc))
        y = acc.to(x_ptr.dtype.element_ty).to(tl.float32)
        if do_norm:
            y = y / tl.sqrt(tl.sum(y * y, axis=0) + eps)
        tl.store(o_tok, y.to(out_ptr.dtype.element_ty))
        o_tok += row
        x_tok += stride_x_tok
        c0 = c1
        c1 = c2
        c2 = xt


def conv_split_l2norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias,
    conv_states: torch.Tensor,
    cache_indices: torch.Tensor,
    has_initial_state,
    query_start_loc: torch.Tensor,
    max_seqlen: int,
    num_k_heads: int,
    num_v_heads: int,
    head_dim: int,
    silu: bool = True,
    eps: float = 1e-6,
):
    """Returns ``q, k, v`` of shapes ``(1, T, HK, D)``, ``(1, T, HK, D)``,
    ``(1, T, HV, D)``, contiguous, q / k L2-normalised; updates ``conv_states``."""
    T, dim = x.shape
    HK, HV, D = num_k_heads, num_v_heads, head_dim
    out = x.new_empty(T * (2 * HK + HV) * D)
    q = out[: T * HK * D].view(1, T, HK, D)
    k = out[T * HK * D: 2 * T * HK * D].view(1, T, HK, D)
    v = out[2 * T * HK * D:].view(1, T, HV, D)
    S = query_start_loc.shape[0] - 1
    if S == 0 or max_seqlen == 0:
        return q, k, v
    grid = (S, triton.cdiv(max_seqlen, _BT), dim // D)
    _conv_split_l2norm_kernel[grid](
        x, weight, bias if bias is not None else weight,
        conv_states, cache_indices,
        has_initial_state if has_initial_state is not None else cache_indices,
        query_start_loc, out, T,
        x.stride(0), weight.stride(0), weight.stride(1),
        conv_states.stride(0), conv_states.stride(1), conv_states.stride(2),
        eps,
        HK=HK, HV=HV, D=D, BT=_BT,
        HAS_BIAS=bias is not None, HAS_INIT=has_initial_state is not None, SILU=silu,
        num_warps=_NUM_WARPS,
    )
    return q, k, v
