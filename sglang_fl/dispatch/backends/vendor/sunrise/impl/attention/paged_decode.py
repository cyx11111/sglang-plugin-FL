# Copyright (c) 2026 BAAI. All rights reserved.
"""PTPU ``paged_decode_attention_fwd`` for full-attention decode (head_dim 128 / 256).

The kernel is only exact while ``batch * kv_heads * splits`` stays within a
device capacity; above it results stop being reproducible (~1e-2 error).
Measured capacity is 96 or 128 depending on the card, so 96 is used
(``SGLANG_FL_SUNRISE_PAGED_DECODE_CAP`` overrides) after one low-memory
exactness check at startup, when the KV pool already fills the device. Small
batches get up to 16 KV splits, large batches are processed in groups of
``capacity // kv_heads`` requests with a single split. The plan depends on the
batch size only, so it is graph-capture safe.

Disable with ``SGLANG_FL_SUNRISE_PAGED_DECODE=0``.
"""

from __future__ import annotations

import logging
import os
from typing import Callable, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

# head_dim 128 (MiniMax-M2.7 TP8, 6 q / 1 kv heads): 2-7x faster than
# ``decode_attention_fwd`` for bs >= 8 (one block per q head re-reads the KV).
HEAD_DIMS = (128, 256)
_MAX_SPLITS = 16
_DEFAULT_CAP = 96
_PROBE_KV_LEN = 256
_CAPACITY_CACHE: dict[tuple[int, int, int, str], int] = {}


def load_kernel() -> Optional[Callable]:
    if os.environ.get("SGLANG_FL_SUNRISE_PAGED_DECODE", "1") == "0":
        return None
    try:
        from llm_kernel_ptpu import paged_decode_attention_fwd

        return paged_decode_attention_fwd
    except Exception:
        # Older torch_ptpu releases bundle the same wrapper in sgl_kernel
        # instead of exposing the separate llm_kernel_ptpu package.
        try:
            from torch_ptpu.sgl_kernel import paged_decode_attention_fwd

            return paged_decode_attention_fwd
        except Exception:
            return None


def _workspace_floats(cap: int, heads: int, kv_heads: int, dim: int) -> int:
    return (cap // kv_heads) * heads * (dim + 1)


def _reproducible(fn, heads, kv_heads, dim, batch, splits, device) -> bool:
    torch.manual_seed(0)
    n = batch * _PROBE_KV_LEN
    k = torch.randn(n, kv_heads, dim, device=device, dtype=torch.bfloat16)
    v = torch.randn(n, kv_heads, dim, device=device, dtype=torch.bfloat16)
    q = torch.randn(batch, heads, dim, device=device, dtype=torch.bfloat16)
    indptr = torch.arange(0, n + 1, _PROBE_KV_LEN, device=device, dtype=torch.int32)
    indices = torch.randperm(n, device="cpu").to(torch.int32).to(device)
    ns = torch.full((batch,), splits, device=device, dtype=torch.int32)
    ws = torch.empty(batch * heads * splits * (dim + 1), device=device, dtype=torch.float32)
    scale = dim ** -0.5
    outs = []
    for _ in range(2):
        o = torch.zeros_like(q)
        fn(q, k, v, ws, o, indptr, indices, ns, splits, scale, 0.0)
        outs.append(o)
    group = heads // kv_heads
    kk = k[indices.long()].view(batch, _PROBE_KV_LEN, kv_heads, dim).float()
    vv = v[indices.long()].view(batch, _PROBE_KV_LEN, kv_heads, dim).float()
    qg = q.float().view(batch, kv_heads, group, dim)
    att = torch.softmax(torch.einsum("bkgd,blkd->bkgl", qg, kk) * scale, -1)
    ref = torch.einsum("bkgl,blkd->bkgd", att, vv).reshape(batch, heads, dim)
    err = ((outs[0].float() - ref).norm() / ref.norm()).item()
    return torch.equal(outs[0], outs[1]) and err < 5e-3


def measure_capacity(fn, heads: int, kv_heads: int, dim: int, device) -> int:
    """Checked ``batch * kv_heads * splits`` capacity, 0 if the check fails."""
    key = (heads, kv_heads, dim, str(torch.device(device)))
    cached = _CAPACITY_CACHE.get(key)
    if cached is not None:
        return cached

    cap = int(os.environ.get("SGLANG_FL_SUNRISE_PAGED_DECODE_CAP", _DEFAULT_CAP))
    cap -= cap % kv_heads
    try:
        ok = cap > 0 and _reproducible(fn, heads, kv_heads, dim, cap // kv_heads, 1, device)
        ok = ok and _reproducible(
            fn, heads, kv_heads, dim, 1, min(_MAX_SPLITS, cap // kv_heads), device
        )
    except Exception as exc:
        logger.warning("sunrise paged decode check failed: %s", exc)
        ok = False
    torch.ptpu.synchronize()
    result = cap if ok else 0
    # The target and every EAGLE draft step use identical head shapes, but each
    # SunriseAttnBackend used to repeat this allocation-heavy stochastic probe.
    # Under large MTP pools a later probe could fail after the target had already
    # established that the kernel is exact, disabling metadata only for that
    # draft step. Cache successful checks process-wide; failures remain
    # retryable in case they were caused by transient memory pressure.
    if result > 0:
        _CAPACITY_CACHE[key] = result
    return result


class PagedDecode:
    def __init__(self, fn, heads: int, kv_heads: int, dim: int, cap: int, device):
        self.fn = fn
        self.heads = heads
        self.kv_heads = kv_heads
        self.dim = dim
        self.cap = cap
        self.group = cap // kv_heads
        self.workspace = torch.empty(
            _workspace_floats(cap, heads, kv_heads, dim), device=device, dtype=torch.float32
        )
        # Allocated up front so graph capture never allocates metadata.
        self._split_bufs = {
            s: torch.full((max(self.group, 1),), s, device=device, dtype=torch.int32)
            for s in range(1, _MAX_SPLITS + 1)
        }

    def plan(self, batch: int) -> Tuple[int, List[Tuple[int, int]]]:
        if batch * self.kv_heads <= self.cap:
            splits = max(1, min(_MAX_SPLITS, self.cap // (batch * self.kv_heads)))
            return splits, [(0, batch)]
        return 1, [(lo, min(lo + self.group, batch)) for lo in range(0, batch, self.group)]

    def __call__(self, q, k_buf, v_buf, out, kv_indptr, kv_indices, scale, logit_cap):
        batch = q.shape[0]
        splits, groups = self.plan(batch)
        for lo, hi in groups:
            self.fn(
                q[lo:hi],
                k_buf,
                v_buf,
                self.workspace,
                out[lo:hi],
                kv_indptr[lo : hi + 1],
                kv_indices,
                self._split_bufs[splits][: hi - lo],
                splits,
                scale,
                logit_cap,
            )
