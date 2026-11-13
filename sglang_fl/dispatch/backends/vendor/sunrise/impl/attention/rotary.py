# Copyright (c) 2026 BAAI. All rights reserved.

"""
Sunrise (PTPU) rotary embedding operator implementations.

Routes RoPE to ``torch_ptpu.sgl_kernel.apply_rope_with_cos_sin_cache``.
"""

from __future__ import annotations

import torch

from torch_ptpu.sgl_kernel import apply_rope_with_cos_sin_cache


_CACHED_F32_CACHE_ATTR = "_sglang_fl_sunrise_cos_sin_cache_f32"


def cos_sin_cache_f32(obj, raw: torch.Tensor) -> torch.Tensor:
    """float32 copy of ``raw`` memoized on ``obj``, rebuilt when ``raw`` changes.

    SGLang replaces ``obj.cos_sin_cache`` with a longer tensor in
    ``_ensure_cos_sin_cache_length`` (context reserve, speculative decoding),
    so the memo is keyed on the source tensor's storage and length.
    """
    key = (raw.data_ptr(), tuple(raw.shape))
    cached = getattr(obj, _CACHED_F32_CACHE_ATTR, None)
    if cached is not None and cached[0] == key:
        return cached[1]
    cache_f32 = raw.to(dtype=torch.float32)
    try:
        setattr(obj, _CACHED_F32_CACHE_ATTR, (key, cache_f32))
    except (AttributeError, TypeError):
        pass
    return cache_f32


def _get_cos_sin_cache_f32(obj, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """float32 ``[max_pos, rotary_dim]`` cos_sin_cache for PTPU sgl.

    Prefers ``obj.cos_sin_cache``; otherwise rebuilds it from the dispatched
    ``cos`` / ``sin`` halves (slices of one contiguous buffer).
    """
    raw = getattr(obj, "cos_sin_cache", None)
    if raw is None:
        return torch.cat([cos, sin], dim=-1).to(dtype=torch.float32)
    return cos_sin_cache_f32(obj, raw)


def _full_heads(x: torch.Tensor, head_size: int):
    """Contiguous ``[tokens, heads, head_size]`` tensor that ``x = full[..., :d]`` slices,
    or ``None`` if ``x`` is not such a slice."""
    if x.dim() != 3 or x.stride(-1) != 1:
        return None
    t, h, _ = x.shape
    if x.stride(1) != head_size or x.stride(0) != h * head_size:
        return None
    needed = (x.storage_offset() + t * h * head_size) * x.element_size()
    if x.untyped_storage().nbytes() < needed:
        return None
    return torch.as_strided(x, (t, h, head_size), (h * head_size, head_size, 1), x.storage_offset())


def _rotate_torch(query, key, cos, sin, positions, interleaved):
    """Reference RoPE on ``[tokens, heads, rotary_dim]`` (cos / sin ``[max_pos, rotary_dim // 2]``)."""
    c = cos[positions].unsqueeze(1).float()
    s = sin[positions].unsqueeze(1).float()

    def rot(x):
        xf = x.float()
        if interleaved:
            x1, x2 = xf[..., ::2], xf[..., 1::2]
            return torch.stack((x1 * c - x2 * s, x2 * c + x1 * s), dim=-1).flatten(-2).to(x.dtype)
        x1, x2 = xf.chunk(2, dim=-1)
        return torch.cat((x1 * c - x2 * s, x2 * c + x1 * s), dim=-1).to(x.dtype)

    return rot(query), rot(key)


def rotary_embedding_sunrise(
    obj,
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
    rotary_interleaved: bool = False,
    inplace: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE on PTPU.

    Args:
        obj: The calling RotaryEmbedding layer (used to fetch the original
            ``cos_sin_cache`` and to memoize the float32 version).
        query: ``[num_tokens, num_heads, rotary_dim]`` (3D, as produced by
            ``rotary_embedding_bridge``).
        key: ``[num_tokens, num_kv_heads, rotary_dim]`` (3D).
        cos, sin: ``[max_seq_len, rotary_dim // 2]`` each; redundant when
            ``obj`` exposes ``cos_sin_cache`` but kept for fallback.
        position_ids: ``[num_tokens]`` flat tensor of absolute positions.
        rotary_interleaved: ``False`` for neox-style (Qwen3); ``True`` for
            GPT-J style interleaved layout.
        inplace: Honoured implicitly -- PTPU sgl operates in-place on the
            flat 2D view and returns the same tensors.

    Returns:
        Tuple ``(q_embed, k_embed)`` with the original 3D shapes restored.
    """
    num_tokens = query.shape[0]
    rotary_dim = query.shape[-1]
    head_size = getattr(obj, "head_size", rotary_dim)

    if rotary_dim != head_size:
        # Partial rotary (MiniMax-M2: 64 of 128): the bridge passes
        # ``q[..., :rotary_dim]``. PTPU's kernel takes the flat full-head buffer
        # and rotates only the leading ``rotary_dim`` channels (the width of
        # ``cos_sin_cache``) into new full-head outputs, so when the slices are
        # views of contiguous ``[tokens, heads, head_size]`` buffers the full
        # heads go in and the rotated slices of the outputs come back.
        q_full = _full_heads(query, head_size)
        k_full = _full_heads(key, head_size)
        if q_full is None or k_full is None:
            return _rotate_torch(query, key, cos, sin, position_ids, rotary_interleaved)
        q_out, k_out = apply_rope_with_cos_sin_cache(
            positions=position_ids,
            query=q_full.view(num_tokens, -1),
            key=k_full.view(num_tokens, -1),
            head_size=head_size,
            cos_sin_cache=_get_cos_sin_cache_f32(obj, cos, sin),
            is_neox=not rotary_interleaved,
        )
        return (q_out.view(q_full.shape)[..., :rotary_dim],
                k_out.view(k_full.shape)[..., :rotary_dim])

    query_shape = query.shape
    key_shape = key.shape

    # PTPU sgl wants 2D [tokens, num_heads * head_size]. ``.contiguous()`` is
    # required because the upstream wrapper produces a strided slice via
    # ``query[..., :rotary_dim]``.
    query_flat = query.contiguous().view(num_tokens, -1)
    key_flat = key.contiguous().view(num_tokens, -1)

    cos_sin_cache_f32 = _get_cos_sin_cache_f32(obj, cos, sin)

    q_out, k_out = apply_rope_with_cos_sin_cache(
        positions=position_ids,
        query=query_flat,
        key=key_flat,
        head_size=head_size,
        cos_sin_cache=cos_sin_cache_f32,
        is_neox=not rotary_interleaved,
    )

    return q_out.view(query_shape), k_out.view(key_shape)
