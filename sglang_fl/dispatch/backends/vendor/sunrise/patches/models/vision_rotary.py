# Copyright (c) 2026 BAAI. All rights reserved.
"""Force eager ViT / rotary RoPE on PTPU.

SGLang's ``apply_rotary_pos_emb`` is often ``torch.compile``-wrapped.
Inductor does not support PTPU (``Device ptpu not supported``).

Installed SGLang versions differ:
  - newer: ``apply_rotary_pos_emb_native_eager`` in utils
  - older: only compiled ``apply_rotary_pos_emb_native`` / ``apply_rotary_pos_emb``

This patch resolves an eager callable across versions (unwrap compile, or
ship a local fallback), then rebinds vision / rotary entry points.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

_patched = False


def _is_ptpu() -> bool:
    try:
        return (
            hasattr(torch, "ptpu")
            and torch.ptpu.is_available()
            and torch.ptpu.device_count() > 0
        )
    except Exception:
        return False


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _local_apply_rotary_pos_emb_eager(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Version-agnostic eager RoPE (same math as upstream native_eager)."""
    orig_q_dtype = q.dtype
    orig_k_dtype = k.dtype
    q, k = q.float(), k.float()
    cos = cos.unsqueeze(unsqueeze_dim).float()
    sin = sin.unsqueeze(unsqueeze_dim).float()
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed.to(orig_q_dtype), k_embed.to(orig_k_dtype)


def _unwrap_compiled(fn: Callable) -> Optional[Callable]:
    """Best-effort unwrap of a ``torch.compile`` wrapper."""
    for attr in (
        "_torchdynamo_orig_callable",
        "_orig_mod",
        "__wrapped__",
        "_torch_compile_wrapper",
    ):
        inner = getattr(fn, attr, None)
        if callable(inner):
            return inner
    # OptimizedModule-style
    inner = getattr(fn, "_orig_mod", None)
    if callable(inner):
        return inner
    return None


def _resolve_eager_rope() -> Tuple[Callable, str]:
    """Return (eager_fn, source_description)."""
    # Prefer explicit eager symbol (newer SGLang).
    try:
        from sglang.srt.layers.rotary_embedding.utils import (
            apply_rotary_pos_emb_native_eager as eager,
        )

        return eager, "utils.apply_rotary_pos_emb_native_eager"
    except ImportError:
        pass

    # Older trees: only compiled native — unwrap if possible.
    try:
        import sglang.srt.layers.rotary_embedding.utils as rope_utils

        for name in (
            "apply_rotary_pos_emb_native",
            "apply_rotary_pos_emb",
        ):
            fn = getattr(rope_utils, name, None)
            if not callable(fn):
                continue
            unwrapped = _unwrap_compiled(fn)
            if unwrapped is not None:
                return unwrapped, f"unwrap({name})"
    except Exception:
        pass

    try:
        import sglang.srt.layers.rotary_embedding as rope_pkg

        fn = getattr(rope_pkg, "apply_rotary_pos_emb", None)
        if callable(fn):
            unwrapped = _unwrap_compiled(fn)
            if unwrapped is not None:
                return unwrapped, "unwrap(rotary_embedding.apply_rotary_pos_emb)"
    except Exception:
        pass

    return _local_apply_rotary_pos_emb_eager, "local_fallback"


def patch_vision_rotary_eager_for_sunrise() -> None:
    """Replace compile-backed ``apply_rotary_pos_emb`` with eager on PTPU."""
    global _patched
    if _patched:
        return

    if not _is_ptpu():
        return

    try:
        eager, src = _resolve_eager_rope()

        # 1) Package / utils aliases
        try:
            import sglang.srt.layers.rotary_embedding as rope_pkg
            import sglang.srt.layers.rotary_embedding.utils as rope_utils

            rope_pkg.apply_rotary_pos_emb = eager
            for name in (
                "apply_rotary_pos_emb",
                "apply_rotary_pos_emb_native",
                "apply_rotary_pos_emb_native_eager",
            ):
                if hasattr(rope_utils, name) or name == "apply_rotary_pos_emb_native":
                    setattr(rope_utils, name, eager)
        except Exception as exc:
            logger.warning("sunrise: rotary package rebind failed: %s", exc)

        # 2) Vision module + VisionAttention.forward.__globals__
        import sglang.srt.layers.attention.vision as vision_mod

        vision_mod.apply_rotary_pos_emb = eager
        if hasattr(vision_mod, "ROTARY_EMBED_CLASSES"):
            vision_mod.ROTARY_EMBED_CLASSES["normal"] = eager

        VisionAttention = getattr(vision_mod, "VisionAttention", None)
        if VisionAttention is not None and hasattr(VisionAttention, "forward"):
            fwd = VisionAttention.forward
            g = getattr(fwd, "__globals__", None)
            if g is not None:
                g["apply_rotary_pos_emb"] = eager
                if "apply_rotary_pos_emb_native_eager" in g:
                    g["apply_rotary_pos_emb_native_eager"] = eager

            if not getattr(VisionAttention, "_sunrise_rope_eager_wrapped", False):
                _orig_forward = VisionAttention.forward

                def _forward_eager_rope(self, *args, **kwargs):
                    gg = _orig_forward.__globals__
                    gg["apply_rotary_pos_emb"] = eager
                    if "apply_rotary_pos_emb_native_eager" in gg:
                        gg["apply_rotary_pos_emb_native_eager"] = eager
                    return _orig_forward(self, *args, **kwargs)

                VisionAttention.forward = _forward_eager_rope
                VisionAttention._sunrise_rope_eager_wrapped = True

        _patched = True
        logger.info(
            "sunrise: forced ViT apply_rotary_pos_emb → eager "
            "(source=%s; inductor has no PTPU)",
            src,
        )
    except Exception as exc:
        logger.warning(
            "sunrise: failed to patch vision rotary for PTPU: %s",
            exc,
        )
