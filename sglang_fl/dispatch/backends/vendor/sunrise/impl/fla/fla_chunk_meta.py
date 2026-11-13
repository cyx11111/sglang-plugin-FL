# Copyright (c) 2026 BAAI. All rights reserved.
"""Chunk index/offset tables for PTPU GDN prefill (FLA-compatible layout).

PTPU launch overhead makes a pure on-device path (``repeat_interleave`` /
``cumsum`` / ``.item()``) *slower* than one D2H of the tiny length vector
plus CPU list construction. Typical GDN prefill has small ``S`` (seq count)
and modest ``NT``; host build + one H2D wins.
"""

from __future__ import annotations

import weakref
from typing import List, Sequence, Tuple

import torch


def prepare_lens(cu_seqlens: torch.Tensor) -> torch.Tensor:
    return cu_seqlens[1:] - cu_seqlens[:-1]


def _lens_host(cu_seqlens: torch.Tensor) -> List[int]:
    """One device→host sync of per-sequence lengths (small ``S``)."""
    return [int(n) for n in (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()]


def _nchunks_host(lens: Sequence[int], chunk_size: int) -> List[int]:
    cs = int(chunk_size)
    return [(int(n) + cs - 1) // cs for n in lens]


def _indices_from_nchunks(
    nchunks: Sequence[int], device: torch.device
) -> torch.Tensor:
    """Build FLA ``[NT, 2]`` (seq_id, local_chunk) on CPU, copy once."""
    seq_ids: List[int] = []
    locals_: List[int] = []
    for i, nc in enumerate(nchunks):
        if nc <= 0:
            continue
        seq_ids.extend([i] * nc)
        locals_.extend(range(nc))
    if not locals_:
        return torch.empty((0, 2), dtype=torch.int32, device=device)
    # CPU tensor → single H2D (avoids per-seq device ``arange`` launches).
    out = torch.tensor([seq_ids, locals_], dtype=torch.int32).t().contiguous()
    return out.to(device=device, non_blocking=True)


def _offsets_from_nchunks(
    nchunks: Sequence[int], device: torch.device
) -> torch.Tensor:
    """Exclusive cumsum of chunk counts → ``[S+1]`` int32 on ``device``."""
    offsets = [0]
    total = 0
    for nc in nchunks:
        total += int(nc)
        offsets.append(total)
    return torch.tensor(offsets, dtype=torch.int32, device=device)


def prepare_chunk_offsets(cu_seqlens: torch.Tensor, chunk_size: int) -> torch.Tensor:
    """FLA-compatible exclusive cumsum of per-sequence chunk counts → ``[S+1]``."""
    return _offsets_from_nchunks(
        _nchunks_host(_lens_host(cu_seqlens), chunk_size), cu_seqlens.device
    )


def prepare_chunk_indices(cu_seqlens: torch.Tensor, chunk_size: int) -> torch.Tensor:
    """FLA-compatible ``chunk_indices`` of shape ``[NT, 2]`` (seq_id, local_chunk)."""
    return _indices_from_nchunks(
        _nchunks_host(_lens_host(cu_seqlens), chunk_size), cu_seqlens.device
    )


_cache: dict = {}
# (weakref(cu_seqlens), version, host lens) registered when the metadata is built.
_lens_hints: list = []
_MAX_HINTS = 4


def register_host_lens(cu_seqlens: torch.Tensor, lens: Sequence[int]) -> None:
    """Host-side per-sequence lengths of ``cu_seqlens`` (the scheduler already has them),
    so the chunk tables need no device sync."""
    _lens_hints.append((weakref.ref(cu_seqlens), cu_seqlens._version, [int(n) for n in lens]))
    del _lens_hints[:-_MAX_HINTS]


def _hinted_lens(cu_seqlens: torch.Tensor):
    for ref, version, lens in reversed(_lens_hints):
        if ref() is cu_seqlens and version == cu_seqlens._version and len(lens) + 1 == cu_seqlens.numel():
            return lens
    return None


def prepare_chunk_tables_cached(
    cu_seqlens: torch.Tensor,
    device: torch.device,
    chunk_size: int,
    phase1_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """int32 ``cu_seqlens`` + ``prepare_chunk_tables``, memoized on the
    caller's ``cu_seqlens`` tensor object.

    Every GDN layer of a forward passes the same metadata tensor; without the
    cache each layer pays a D2H sync (the host then waits for all queued
    kernels) plus three H2D copies.
    """
    key = (chunk_size, phase1_size, str(device))
    hit = _cache.get(key)
    if (
        hit is not None
        and hit[0]() is cu_seqlens
        and hit[1] == cu_seqlens._version
    ):
        return hit[2]
    lens = _hinted_lens(cu_seqlens)
    cu_i32 = cu_seqlens.to(device=device, dtype=torch.int32).contiguous()
    tables = (cu_i32, *prepare_chunk_tables(cu_i32, chunk_size, phase1_size, lens=lens))
    _cache[key] = (weakref.ref(cu_seqlens), cu_seqlens._version, tables)
    return tables


def prepare_chunk_tables(
    cu_seqlens: torch.Tensor,
    chunk_size: int,
    phase1_size: int,
    lens: Sequence[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Build indices / offsets / phase1 indices in **one** length sync (none when the
    host ``lens`` are given).

    Returns ``(chunk_indices, chunk_offsets, chunk_indices_phase1, nt)`` where
    ``nt`` is the host-side total chunk count (avoids ``offsets[-1].item()``).
    """
    device = cu_seqlens.device
    if lens is None:
        lens = _lens_host(cu_seqlens)
    nchunks = _nchunks_host(lens, chunk_size)
    nchunks_p1 = _nchunks_host(lens, phase1_size)
    indices = _indices_from_nchunks(nchunks, device)
    offsets = _offsets_from_nchunks(nchunks, device)
    indices_p1 = _indices_from_nchunks(nchunks_p1, device)
    return indices, offsets, indices_p1, sum(nchunks)
