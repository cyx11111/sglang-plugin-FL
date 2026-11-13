# Copyright (c) 2026 BAAI. All rights reserved.
"""Fused gather-transpose / transpose-scatter Triton kernels for the PTPU GDN
decode wrappers.

Why this exists
---------------
PTPU's C++ kernels ``fused_sigmoid_gating_delta_rule_update`` and
``chunk_delta_h_fwd`` process the recurrent state in ``(HV, K, V)`` layout with
**V innermost**, while SGLang allocates its GDN state pool as
``(num_slots, HV, V, K)`` with **K innermost** (see
``fla/fused_sigmoid_gating_recurrent.py``'s ``o_v[None, :] * K + o_k[:, None]``
addressing and ``fla/chunk_delta_h.py``'s ``make_block_ptr(h0, (V, K), (K, 1))``).

For Qwen3.5 (``head_k_dim == head_v_dim == 128``) the two buffers share byte
size and outer strides, so handing PTPU the pool directly does **not** crash —
it silently writes a transposed state that accumulates across GDN layers into
garbage output. The decode wrappers therefore must:

1. Gather the per-sequence slots out of the pool, transpose ``V`` and ``K``, and
   materialise a contiguous ``(B, HV, K, V)`` scratch in PTPU layout.
2. After the kernel returns, transpose ``K`` and ``V`` back and scatter into the
   pool so SGLang's storage stays in its native ``(V, K)`` layout.

Expressing steps 1 and 2 as a chain of PyTorch primitives (``index_select``,
``transpose``, ``copy_``, ``index_put_``) costs ~25 µs of launch overhead per op
on PTPU regardless of compute, which dominates the decode step once TP sharding
shrinks the kernel work. This module collapses each chain into a single Triton
launch that reads the int32 slot ids natively and does the transpose
in-register.

Padding rows
------------
SGLang marks cudagraph / batch padding rows with a **negative** state index and
its own kernels guard every state access with ``if idx >= 0`` (the packed-decode
kernel additionally writes zeros into ``out`` for those rows). We mirror that:

* ``gather_transpose_to_scratch``: zero-fills the scratch row (a negative slot
  must never be used for pointer arithmetic — it would read outside the pool).
* ``transpose_scatter_to_pool``: skips the writeback entirely, which also avoids
  a data race between concurrent programs targeting the same padding slot.
* ``copy_out_masked``: zero-fills the corresponding ``out`` rows.
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

# Default tile size. K = V = 128 on Qwen3.5-A3B; one program handles
# (BLOCK_K, BLOCK_V) = (64, 64) = 4096 elements, which fits comfortably in
# PTPU's SRAM. For each (b, hv) pair we launch ⌈K/64⌉ × ⌈V/64⌉ = 4 programs.
_DEFAULT_BLOCK_K: int = 64
_DEFAULT_BLOCK_V: int = 64
_DEFAULT_BLOCK_OUT_V: int = 64
_ROWS_COPY = os.environ.get("SGLANG_FL_SUNRISE_GDN_ROWS_COPY", "1") == "1"
_ROWS_BLOCK = int(os.environ.get("SGLANG_FL_SUNRISE_GDN_ROWS_BLOCK", "1024"))


@triton.jit
def _fused_gather_transpose_kernel(
    # ---- tensor pointers ----
    src_ptr,  # pool: (num_slots, HV, V, K), contiguous
    idx_ptr,  # ssm_state_indices: (B,), int32
    dst_ptr,  # scratch: (>=B, HV, K, V), contiguous (same dtype as src)
    # ---- strides ----
    stride_src_n,
    stride_src_hv,
    stride_src_v,
    stride_src_k,
    stride_dst_b,
    stride_dst_hv,
    stride_dst_k,
    stride_dst_v,
    # ---- runtime shape ----
    K,
    V,
    # ---- compile-time tile shape ----
    BLOCK_K: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    """One program handles one ``(b, hv, k_block, v_block)`` tile.

    Reads ``pool[ssm_state_indices[b], hv, v_block, k_block]`` of shape
    ``(BLOCK_V, BLOCK_K)`` (V outer, K inner), transposes to
    ``(BLOCK_K, BLOCK_V)``, and writes ``scratch[b, hv, k_block, v_block]``.
    Padding rows (negative slot) get zeros.
    """
    pid_b = tl.program_id(0)
    pid_hv = tl.program_id(1)
    pid_kv = tl.program_id(2)

    # Row-major decomposition keeps adjacent ``pid_kv`` programs working on
    # adjacent ``v_block`` tiles, which favours coalesced reads of the
    # K-innermost source layout (each ``v_block`` reads a full K row stripe).
    num_v_blocks = tl.cdiv(V, BLOCK_V)
    pid_k = pid_kv // num_v_blocks
    pid_v = pid_kv % num_v_blocks

    k_offs = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    v_offs = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
    k_mask = k_offs < K
    v_mask = v_offs < V

    # int32 source → int64 for stride arithmetic (state pools can exceed 2^31
    # elements for high-throughput configs).
    slot = tl.load(idx_ptr + pid_b).to(tl.int64)

    dst_offsets = (
        pid_b * stride_dst_b
        + pid_hv * stride_dst_hv
        + k_offs[:, None] * stride_dst_k
        + v_offs[None, :] * stride_dst_v
    )
    dst_mask = k_mask[:, None] & v_mask[None, :]

    if slot < 0:
        # Padding row: SGLang's kernels start from a zero state here. Writing
        # zeros keeps the scratch deterministic and avoids an out-of-pool read.
        zero = tl.zeros([BLOCK_K, BLOCK_V], dtype=tl.float32)
        tl.store(dst_ptr + dst_offsets, zero.to(dst_ptr.dtype.element_ty), mask=dst_mask)
        return

    # Load source tile in (V, K) layout: shape (BLOCK_V, BLOCK_K).
    src_offsets = (
        slot * stride_src_n
        + pid_hv * stride_src_hv
        + v_offs[:, None] * stride_src_v
        + k_offs[None, :] * stride_src_k
    )
    src_mask = v_mask[:, None] & k_mask[None, :]
    src_tile = tl.load(src_ptr + src_offsets, mask=src_mask, other=0.0)

    # Transpose in-register: (BLOCK_V, BLOCK_K) → (BLOCK_K, BLOCK_V).
    # ``tl.trans`` is metadata-only on the in-SRAM tile; no extra memory traffic.
    tl.store(dst_ptr + dst_offsets, tl.trans(src_tile, 1, 0), mask=dst_mask)


@triton.jit
def _fused_transpose_scatter_kernel(
    # ---- tensor pointers ----
    src_ptr,  # scratch: (>=B, HV, K, V), contiguous (PTPU layout)
    idx_ptr,  # ssm_state_indices: (B,), int32
    dst_ptr,  # pool: (num_slots, HV, V, K), contiguous (SGLang layout)
    # ---- strides ----
    stride_src_b,
    stride_src_hv,
    stride_src_k,
    stride_src_v,
    stride_dst_n,
    stride_dst_hv,
    stride_dst_v,
    stride_dst_k,
    # ---- runtime shape ----
    K,
    V,
    # ---- compile-time tile shape ----
    BLOCK_K: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    """One program handles one ``(b, hv, k_block, v_block)`` tile.

    Reads ``scratch[b, hv, k_block, v_block]`` of shape ``(BLOCK_K, BLOCK_V)``
    (K outer, V inner), transposes to ``(BLOCK_V, BLOCK_K)``, and writes
    ``pool[ssm_state_indices[b], hv, v_block, k_block]``. Padding rows
    (negative slot) are skipped, matching SGLang's ``if idx >= 0`` guard.
    """
    pid_b = tl.program_id(0)
    pid_hv = tl.program_id(1)
    pid_kv = tl.program_id(2)

    slot = tl.load(idx_ptr + pid_b).to(tl.int64)
    if slot < 0:
        return

    num_v_blocks = tl.cdiv(V, BLOCK_V)
    pid_k = pid_kv // num_v_blocks
    pid_v = pid_kv % num_v_blocks

    k_offs = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    v_offs = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
    k_mask = k_offs < K
    v_mask = v_offs < V

    src_offsets = (
        pid_b * stride_src_b
        + pid_hv * stride_src_hv
        + k_offs[:, None] * stride_src_k
        + v_offs[None, :] * stride_src_v
    )
    src_mask = k_mask[:, None] & v_mask[None, :]
    src_tile = tl.load(src_ptr + src_offsets, mask=src_mask, other=0.0)

    dst_offsets = (
        slot * stride_dst_n
        + pid_hv * stride_dst_hv
        + v_offs[:, None] * stride_dst_v
        + k_offs[None, :] * stride_dst_k
    )
    dst_mask = v_mask[:, None] & k_mask[None, :]
    tl.store(dst_ptr + dst_offsets, tl.trans(src_tile, 1, 0), mask=dst_mask)


@triton.jit
def _copy_out_masked_kernel(
    src_ptr,  # PTPU output: (B, HV, V) view
    idx_ptr,  # ssm_state_indices: (B,), int32
    dst_ptr,  # caller's out: (B, HV, V) view
    stride_src_b,
    stride_src_hv,
    stride_src_v,
    stride_dst_b,
    stride_dst_hv,
    stride_dst_v,
    V,
    BLOCK_V: tl.constexpr,
):
    """Copy PTPU's ``o`` into the caller's ``out``, zeroing padding rows.

    SGLang's packed-decode kernel stores zeros for rows whose state index is
    negative; PTPU has no such notion and computes a value from the zero state
    we handed it, so mask those rows here rather than paying a separate launch.
    """
    pid_b = tl.program_id(0)
    pid_hv = tl.program_id(1)
    pid_v = tl.program_id(2)

    v_offs = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
    v_mask = v_offs < V

    dst = dst_ptr + pid_b * stride_dst_b + pid_hv * stride_dst_hv + v_offs * stride_dst_v

    slot = tl.load(idx_ptr + pid_b).to(tl.int64)
    if slot < 0:
        zero = tl.zeros([BLOCK_V], dtype=tl.float32)
        tl.store(dst, zero.to(dst_ptr.dtype.element_ty), mask=v_mask)
        return

    src = src_ptr + pid_b * stride_src_b + pid_hv * stride_src_hv + v_offs * stride_src_v
    tl.store(dst, tl.load(src, mask=v_mask, other=0.0), mask=v_mask)


@triton.jit
def _copy_out_masked_rows_kernel(
    src_ptr,  # (B, ROW) with ROW = HV*V contiguous elements per row
    idx_ptr,  # ssm_state_indices: (B,), int32
    dst_ptr,  # (B, ROW), same per-row layout
    stride_src_b,
    stride_dst_b,
    ROW,
    BLOCK: tl.constexpr,
):
    """``copy_out_masked`` when ``(HV, V)`` is contiguous: one flat block per program."""
    pid_b = tl.program_id(0)
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < ROW
    dst = dst_ptr + pid_b.to(tl.int64) * stride_dst_b + offs
    slot = tl.load(idx_ptr + pid_b)
    if slot < 0:
        tl.store(dst, tl.zeros([BLOCK], dtype=tl.float32).to(dst_ptr.dtype.element_ty), mask=mask)
        return
    src = src_ptr + pid_b.to(tl.int64) * stride_src_b + offs
    tl.store(dst, tl.load(src, mask=mask, other=0.0), mask=mask)


def gather_transpose_to_scratch(
    initial_state: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    scratch: torch.Tensor,
    block_k: int = _DEFAULT_BLOCK_K,
    block_v: int = _DEFAULT_BLOCK_V,
) -> None:
    """Fused gather+transpose: copy slots out of SGLang's state pool into a
    PTPU-layout scratch in one Triton launch.

    Equivalent to (but ~3 PyTorch launches cheaper than):

    .. code-block:: python

        slot_idx = ssm_state_indices.to(torch.int64)
        scratch.copy_(initial_state[slot_idx].transpose(-1, -2))

    Args:
        initial_state: SGLang state pool, shape ``(num_slots, HV, V, K)``,
            contiguous, any floating dtype.
        ssm_state_indices: per-sequence slot ids, shape ``(B,)``, ``int32``.
            Negative entries mark padding rows and yield a zero scratch row.
        scratch: pre-allocated scratch in PTPU layout, shape
            ``(>=B, HV, K, V)``, contiguous, same dtype as ``initial_state``.
            Only the first ``B`` rows are written.
        block_k, block_v: Triton compile-time tile shape. The ``(64, 64)``
            default gives 4 tiles per ``(b, hv)`` for ``K == V == 128``.
    """
    B = ssm_state_indices.shape[0]
    if B == 0:
        return

    HV, V, K = initial_state.shape[-3:]
    assert scratch.shape[-3] == HV, (
        f"scratch HV={scratch.shape[-3]} != initial_state HV={HV}"
    )
    assert scratch.shape[-2] == K, (
        f"scratch K dim ({scratch.shape[-2]}) must equal initial_state K ({K})"
    )
    assert scratch.shape[-1] == V, (
        f"scratch V dim ({scratch.shape[-1]}) must equal initial_state V ({V})"
    )
    assert scratch.shape[0] >= B, f"scratch capacity {scratch.shape[0]} < B={B}"
    assert initial_state.dtype == scratch.dtype, (
        f"dtype mismatch: initial_state={initial_state.dtype}, "
        f"scratch={scratch.dtype}"
    )

    num_kv_blocks = triton.cdiv(K, block_k) * triton.cdiv(V, block_v)
    grid = (B, HV, num_kv_blocks)

    _fused_gather_transpose_kernel[grid](
        initial_state,
        ssm_state_indices,
        scratch,
        initial_state.stride(-4),
        initial_state.stride(-3),
        initial_state.stride(-2),
        initial_state.stride(-1),
        scratch.stride(-4),
        scratch.stride(-3),
        scratch.stride(-2),
        scratch.stride(-1),
        K,
        V,
        BLOCK_K=block_k,
        BLOCK_V=block_v,
    )


def transpose_scatter_to_pool(
    scratch: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    initial_state: torch.Tensor,
    block_k: int = _DEFAULT_BLOCK_K,
    block_v: int = _DEFAULT_BLOCK_V,
) -> None:
    """Fused transpose+scatter: write PTPU-layout scratch back into SGLang's
    state pool in one Triton launch.

    Semantically equivalent to:

    .. code-block:: python

        for b in range(B):
            if ssm_state_indices[b] >= 0:
                initial_state[ssm_state_indices[b]] = scratch[b].transpose(-1, -2)

    Args:
        scratch: PTPU-layout state, shape ``(>=B, HV, K, V)``, contiguous.
        ssm_state_indices: per-sequence slot ids, shape ``(B,)``, ``int32``.
            Rows with a negative slot are **skipped**, matching SGLang's
            ``idx >= 0`` guard and avoiding races on a shared padding slot.
        initial_state: SGLang state pool, shape ``(num_slots, HV, V, K)``,
            contiguous.
        block_k, block_v: Triton compile-time tile shape.
    """
    B = ssm_state_indices.shape[0]
    if B == 0:
        return

    HV, V, K = initial_state.shape[-3:]
    assert scratch.shape[-3] == HV
    assert scratch.shape[-2] == K
    assert scratch.shape[-1] == V
    assert scratch.shape[0] >= B
    assert initial_state.dtype == scratch.dtype

    num_kv_blocks = triton.cdiv(K, block_k) * triton.cdiv(V, block_v)
    grid = (B, HV, num_kv_blocks)

    _fused_transpose_scatter_kernel[grid](
        scratch,
        ssm_state_indices,
        initial_state,
        scratch.stride(-4),
        scratch.stride(-3),
        scratch.stride(-2),
        scratch.stride(-1),
        initial_state.stride(-4),
        initial_state.stride(-3),
        initial_state.stride(-2),
        initial_state.stride(-1),
        K,
        V,
        BLOCK_K=block_k,
        BLOCK_V=block_v,
    )


def copy_out_masked(
    o: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    out: torch.Tensor,
    block_v: int = _DEFAULT_BLOCK_OUT_V,
) -> None:
    """Copy ``o`` into ``out`` in one launch, zeroing padding rows.

    Args:
        o: PTPU kernel output, viewable as ``(B, HV, V)``.
        ssm_state_indices: per-sequence slot ids, shape ``(B,)``, ``int32``.
        out: caller's destination buffer, viewable as ``(B, HV, V)``.
        block_v: Triton compile-time tile width along ``V``.
    """
    B = ssm_state_indices.shape[0]
    if B == 0:
        return

    assert o.ndim == 3 and out.ndim == 3, (
        f"copy_out_masked expects 3-D (B, HV, V) views, got {o.shape} / {out.shape}"
    )
    assert o.shape == out.shape, f"shape mismatch: o={o.shape}, out={out.shape}"
    assert o.shape[0] == B, f"o batch {o.shape[0]} != indices {B}"

    HV, V = o.shape[1], o.shape[2]
    if _ROWS_COPY and all(t.stride(2) == 1 and t.stride(1) == V for t in (o, out)):
        row = HV * V
        _copy_out_masked_rows_kernel[(B, triton.cdiv(row, _ROWS_BLOCK))](
            o, ssm_state_indices, out, o.stride(0), out.stride(0), row, BLOCK=_ROWS_BLOCK,
        )
        return
    grid = (B, HV, triton.cdiv(V, block_v))

    _copy_out_masked_kernel[grid](
        o,
        ssm_state_indices,
        out,
        o.stride(0),
        o.stride(1),
        o.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        V,
        BLOCK_V=block_v,
    )
