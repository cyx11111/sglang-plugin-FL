# Copyright (c) 2026 BAAI. All rights reserved.

"""
Sunrise (PTPU) MRotaryEmbedding operator.

Routes to ``torch_ptpu.sgl_kernel.apply_rope_with_cos_sin_cache``.

Kernel contract (``apply_rope_with_cos_sin_cache``):
  - ``cos_sin_cache`` must be float32
  - ``positions`` must be int64 (1D ``[T]`` or 2D ``[3, T]`` for mrope)
  - ``query`` / ``key``: ``[T, ...]``, internally viewed as ``[T, -1, head_size]``
  - ``mrope_section``: ``None`` or length-3 list/tuple with all entries ``> 0``
    (maps to t/h/w; ``None`` → kernel uses ``-1,-1,-1``)
  - ``is_neox``: True → NeoX (half) layout; False → interleaved
"""

from __future__ import annotations

import weakref
from typing import Optional, Sequence, Tuple, Union

import torch

from torch_ptpu.sgl_kernel import apply_rope_with_cos_sin_cache

from .rotary import cos_sin_cache_f32


def _normalize_mrope_section(
    mrope_section: Optional[Union[Sequence[int], torch.Tensor]],
) -> Optional[Tuple[int, int, int]]:
    if mrope_section is None:
        return None
    section = tuple(int(x) for x in mrope_section)
    if len(section) != 3 or any(x <= 0 for x in section):
        raise ValueError(
            f"mrope_section must be 3 positive ints, got {mrope_section!r}"
        )
    return section  # type: ignore[return-value]


def _stream_capturing() -> bool:
    # torch_ptpu has no ``is_current_stream_capturing``; SGLang flags the
    # whole graph capture (warmup + capture) via ``model_capture_mode``.
    try:
        from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode

        return bool(get_is_capture_mode())
    except Exception:
        return False


_distinct_rows_cache = None


def mark_rows_equal(positions: torch.Tensor) -> None:
    """Host-side hint: ``positions`` (``[3, T]``) has equal t/h/w rows (text-only batch),
    so :func:`_needs_native_interleaved` can skip its device sync."""
    global _distinct_rows_cache
    _distinct_rows_cache = (weakref.ref(positions), positions._version, False)


def _needs_native_interleaved(obj, positions: torch.Tensor) -> bool:
    """Interleaved M-RoPE with distinct t/h/w rows (multimodal tokens).

    The PTPU kernel only implements sectioned M-RoPE; when all three rows are
    equal (text tokens, decode) both layouts give identical results. Decode
    graphs only ever see equal rows, so the host-syncing check is skipped
    while capturing.
    """
    if positions.ndim != 2 or positions.shape[-1] == 0:
        return False
    if not getattr(obj, "mrope_interleaved", False):
        return False
    if _stream_capturing():
        return False
    # Every attention layer of a forward sees the same positions tensor; one
    # host sync per forward instead of one per layer.
    global _distinct_rows_cache
    hit = _distinct_rows_cache
    if hit is not None and hit[0]() is positions and hit[1] == positions._version:
        return hit[2]
    distinct = bool((positions[1:] != positions[:1]).any())
    _distinct_rows_cache = (weakref.ref(positions), positions._version, distinct)
    return distinct


def mrotary_embedding_sunrise(
    obj,
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Multimodal / standard RoPE on PTPU.

    Dispatch signature matches ``call_op("mrotary_embedding", obj, positions,
    query, key)`` used by ``mrotary_embedding_bridge``.
    """
    if _needs_native_interleaved(obj, positions):
        return obj.forward_native(positions, query, key)

    head_size = obj.head_size
    is_neox_style = getattr(obj, "is_neox_style", True)
    mrope_section = _normalize_mrope_section(getattr(obj, "mrope_section", None))

    # Kernel: 2D positions + valid mrope_section → multimodal path.
    # Otherwise flatten to 1D positions and leave section as None (-1,-1,-1).
    if positions.ndim == 2 and mrope_section is not None:
        rope_positions = positions
    else:
        rope_positions = positions[0] if positions.ndim == 2 else positions
        mrope_section = None

    # Kernel requires int64 positions.
    if rope_positions.dtype != torch.int64:
        rope_positions = rope_positions.to(dtype=torch.int64)

    query_shape = query.shape
    key_shape = key.shape
    num_tokens = query.shape[0]

    # Kernel does query.view(T, -1, head_size); pass contiguous [T, H*D].
    query_flat = query.contiguous().view(num_tokens, -1)
    key_flat = key.contiguous().view(num_tokens, -1)

    q_out, k_out = apply_rope_with_cos_sin_cache(
        positions=rope_positions,
        query=query_flat,
        key=key_flat,
        head_size=head_size,
        cos_sin_cache=cos_sin_cache_f32(obj, obj.cos_sin_cache),
        is_neox=is_neox_style,
        mrope_section=mrope_section,
    )

    return q_out.view(query_shape), k_out.view(key_shape)
