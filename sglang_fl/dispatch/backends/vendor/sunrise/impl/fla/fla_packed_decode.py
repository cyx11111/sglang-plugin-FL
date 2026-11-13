# Copyright (c) 2026 BAAI. All rights reserved.
"""PTPU drop-in for ``fla.fused_recurrent.fused_recurrent_gated_delta_rule_packed_decode``.

Redirects SGLang's packed-decode GDN fast path to PTPU's
``torch_ptpu.sgl_kernel.fused_sigmoid_gating_delta_rule_update``, with a
layout transpose between SGLang's ``[slot, HV, V, K]`` state pool and PTPU's
``[slot, HV, K, V]`` convention.

SGLang-specific differences vs the vLLM port
--------------------------------------------
* Padding sentinel is a **negative** state index (``idx < 0``). The gather
  kernel zero-fills those scratch rows; the scatter kernel skips them; and
  ``copy_out_masked`` zero-fills the corresponding ``out`` rows so the
  contract matches SGLang Triton.
* Softplus knobs are hardcoded to ``(1.0, 20.0)`` (same as SGLang's packed
  decode path).
"""

from __future__ import annotations

import importlib
import os
from typing import Optional

import torch

from .fla_state_transpose import (
    copy_out_masked,
    gather_transpose_to_scratch,
    transpose_scatter_to_pool,
)
from .fla_state_layout import check_native_pool, native_state_layout, run_in_sglang_layout

_FLA_SOURCES = {
    "fused_recurrent_gated_delta_rule_packed_decode": (
        "sglang.srt.layers.attention.fla.fused_recurrent",
        "fused_recurrent_gated_delta_rule_packed_decode",
    ),
}


def _fla_original(name: str):
    """Return the unpatched SGLang FLA callable registered under ``name``."""
    try:
        from sglang_fl.dispatch.fla_patch import get_original

        fn = get_original(name)
        if fn is not None:
            return fn
    except Exception:
        pass

    try:
        module, attr = _FLA_SOURCES[name]
    except KeyError:
        raise RuntimeError(f"No SGLang source recorded for FLA function '{name}'")
    return getattr(importlib.import_module(module), attr)


# Starting capacity: covers common ``max_num_seqs=256`` on first allocation.
_DEFAULT_HEADROOM: int = 256

# The vendor kernel slows down past ~48 rows (HV=48: 295 us at 32 rows, 665
# us at 48, 1340 us at 96 vs 878 us as 32-row calls). Batches with
# B >= 48 and B * HV >= 768 run as calls of this many rows; 0 disables.
_CHUNK_ROWS = int(os.environ.get("SGLANG_FL_SUNRISE_GDN_DECODE_CHUNK", "32"))


def decode_chunk_rows(batch: int, num_v_heads: int) -> int:
    if _CHUNK_ROWS <= 0 or batch < 48 or batch * num_v_heads < 768 or batch <= _CHUNK_ROWS:
        return 0
    return _CHUNK_ROWS

_PTPU_STATE_SCRATCH: dict[tuple, torch.Tensor] = {}
_ARANGE_BUF: dict[tuple, torch.Tensor] = {}


def _next_capacity(needed: int, current: int) -> int:
    target = max(needed, _DEFAULT_HEADROOM)
    if current > 0:
        target = max(target, current * 2)
    return target


def _ensure_state_scratch(
    B: int,
    HV: int,
    K: int,
    V: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    key = (device, dtype, HV, K, V)
    buf = _PTPU_STATE_SCRATCH.get(key)
    cur = 0 if buf is None else buf.shape[0]
    if buf is None or cur < B:
        new_capacity = _next_capacity(B, cur)
        buf = torch.empty(new_capacity, HV, K, V, device=device, dtype=dtype)
        _PTPU_STATE_SCRATCH[key] = buf
    return buf


def _ensure_arange_buf(
    min_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    key = (device, dtype)
    buf = _ARANGE_BUF.get(key)
    cur = 0 if buf is None else buf.numel()
    if buf is None or cur < min_size:
        new_capacity = _next_capacity(min_size, cur)
        buf = torch.arange(new_capacity, device=device, dtype=dtype)
        _ARANGE_BUF[key] = buf
    return buf


def fused_recurrent_gated_delta_rule_packed_decode(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    out: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """PTPU-native re-implementation of SGLang's packed-decode GDN fast path.

    Writes into ``out`` and ``initial_state`` (via ``ssm_state_indices``), then
    returns ``(out, initial_state)``.
    """
    if initial_state is None or ssm_state_indices is None:
        return _fallback(
            mixed_qkv=mixed_qkv,
            a=a,
            b=b,
            A_log=A_log,
            dt_bias=dt_bias,
            scale=scale,
            initial_state=initial_state,
            out=out,
            ssm_state_indices=ssm_state_indices,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )

    if mixed_qkv.ndim != 2 or ssm_state_indices.ndim != 1 or initial_state.ndim != 4:
        return _fallback(
            mixed_qkv=mixed_qkv,
            a=a,
            b=b,
            A_log=A_log,
            dt_bias=dt_bias,
            scale=scale,
            initial_state=initial_state,
            out=out,
            ssm_state_indices=ssm_state_indices,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )

    B = mixed_qkv.shape[0]
    HV, V, K = initial_state.shape[-3:]
    qkv_dim = mixed_qkv.shape[1]
    qk_dim_total = qkv_dim - HV * V
    if qk_dim_total <= 0 or qk_dim_total % 2 != 0:
        return _fallback(
            mixed_qkv=mixed_qkv,
            a=a,
            b=b,
            A_log=A_log,
            dt_bias=dt_bias,
            scale=scale,
            initial_state=initial_state,
            out=out,
            ssm_state_indices=ssm_state_indices,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )
    q_dim = qk_dim_total // 2
    if q_dim % K != 0:
        return _fallback(
            mixed_qkv=mixed_qkv,
            a=a,
            b=b,
            A_log=A_log,
            dt_bias=dt_bias,
            scale=scale,
            initial_state=initial_state,
            out=out,
            ssm_state_indices=ssm_state_indices,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )
    H = q_dim // K

    # mixed_qkv layout per token: [Q (H*K) | K (H*K) | V (HV*V)].
    # Reshape to (1, B, h, d) to match PTPU / sglang steady-decode contract.
    q = mixed_qkv[:, 0 : H * K].view(1, B, H, K)
    k = mixed_qkv[:, H * K : 2 * H * K].view(1, B, H, K)
    v = mixed_qkv[:, 2 * H * K :].view(1, B, HV, V)

    device = mixed_qkv.device
    state_dtype = initial_state.dtype
    # The PTPU kernel reads ``A_log`` as bf16 whatever its dtype; SGLang keeps
    # it fp32 (bf16-exact values from the checkpoint).
    if A_log.dtype != torch.bfloat16:
        A_log = A_log.to(torch.bfloat16)

    from torch_ptpu.sgl_kernel import (
        fused_sigmoid_gating_delta_rule_update as _ptpu_fused_sigmoid_gating,
    )

    if native_state_layout():
        check_native_pool(initial_state)
        indices_i32 = ssm_state_indices.to(dtype=torch.int32).contiguous()
        # Padding rows carry a negative slot; route them to slot 0, which
        # SGLang's MambaPool reserves and never hands out (free slots start
        # at 1). Their output rows are zeroed by ``copy_out_masked``.
        safe_indices = indices_i32.clamp_min(0)
        cu_seqlens = _ensure_arange_buf(B + 1, device, torch.int32)[: B + 1]
        rows = decode_chunk_rows(B, HV)
        if rows:
            out_rows = out.squeeze(1)
            for s in range(0, B, rows):
                e = min(s + rows, B)
                o = _ptpu_fused_sigmoid_gating(
                    A_log, a[s:e], dt_bias, 1.0, 20.0,
                    q[:, s:e], k[:, s:e], v[:, s:e], b[s:e],
                    initial_state, safe_indices[s:e],
                    scale=scale, use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
                    cu_seqlens=cu_seqlens[: e - s + 1], is_kda=False,
                )
                copy_out_masked(o.squeeze(0), indices_i32[s:e], out_rows[s:e])
            return out, initial_state
        o = _ptpu_fused_sigmoid_gating(
            A_log,
            a,
            dt_bias,
            1.0,
            20.0,
            q,
            k,
            v,
            b,
            initial_state,
            safe_indices,
            scale=scale,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            cu_seqlens=cu_seqlens,
            is_kda=False,
        )
        copy_out_masked(o.squeeze(0), indices_i32, out.squeeze(1))
        return out, initial_state

    state_scratch_full = _ensure_state_scratch(B, HV, K, V, device, state_dtype)
    ptpu_state_buf = state_scratch_full[:B]

    arange_full = _ensure_arange_buf(B + 1, device, torch.int32)
    cu_seqlens = arange_full[: B + 1]
    local_indices = arange_full[:B]

    indices_i32 = ssm_state_indices.to(dtype=torch.int32).contiguous()
    gather_transpose_to_scratch(initial_state, indices_i32, ptpu_state_buf)

    o = _ptpu_fused_sigmoid_gating(
        A_log,
        a,
        dt_bias,
        1.0,
        20.0,
        q,
        k,
        v,
        b,
        ptpu_state_buf,
        local_indices,
        scale=scale,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        cu_seqlens=cu_seqlens,
        is_kda=False,
    )
    # o.shape == (1, B, HV, V) after the kernel's internal squeeze(0).

    # Write into caller's ``out`` (B, 1, HV, V), zeroing padding rows.
    copy_out_masked(o.squeeze(0), indices_i32, out.squeeze(1))

    transpose_scatter_to_pool(ptpu_state_buf, indices_i32, initial_state)
    return out, initial_state


def _fallback(
    *,
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    scale: float,
    initial_state: Optional[torch.Tensor],
    out: torch.Tensor,
    ssm_state_indices: Optional[torch.Tensor],
    use_qk_l2norm_in_kernel: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    original = _fla_original("fused_recurrent_gated_delta_rule_packed_decode")

    def call(state, indices):
        return original(
            mixed_qkv=mixed_qkv,
            a=a,
            b=b,
            A_log=A_log,
            dt_bias=dt_bias,
            scale=scale,
            initial_state=state,
            out=out,
            ssm_state_indices=indices,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )

    if initial_state is not None and native_state_layout():
        run_in_sglang_layout(initial_state, ssm_state_indices, call)
        return out, initial_state
    return call(initial_state, ssm_state_indices)
