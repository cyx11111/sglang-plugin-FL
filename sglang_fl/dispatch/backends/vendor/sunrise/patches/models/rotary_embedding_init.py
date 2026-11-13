

import logging
logger = logging.getLogger(__name__)

_patched = False

def patch_rotary_embedding_init_sunrise() -> None:
    """Let ``RotaryEmbedding.__init__`` finish on PTPU. Idempotent.

    On non-CUDA devices SGLang imports ``vllm._custom_ops.rotary_embedding``
    as a fallback kernel, which is absent here (RoPE is served by the
    dispatch bridge). When that import fails, complete the remainder of the
    original ``__init__``; attributes set before the import (incl.
    ``self.dtype``) are already in place.
    """
    global _patched
    if _patched:
        return

    try:
        import torch
        from sglang.srt.layers.rotary_embedding import base as rope_base
        from sglang.srt.layers.rotary_embedding.base import RotaryEmbedding

        if not (
            hasattr(torch, "ptpu")
            and torch.ptpu.is_available()
            and torch.ptpu.device_count() > 0
        ):
            logger.debug(
                "rotary_patch: not on Sunrise – RotaryEmbedding init patch skipped"
            )
            return

        _orig_rope_init = RotaryEmbedding.__init__

        def _sunrise_patched_rope_init(self, *args, **kwargs):
            try:
                _orig_rope_init(self, *args, **kwargs)
                return
            except ModuleNotFoundError as e:
                if "vllm" not in str(e):
                    raise

            cache = self._compute_cos_sin_cache().to(self.dtype)
            self.use_fallback_kernel = False
            self.cos_sin_cache: torch.Tensor
            self.register_buffer("cos_sin_cache", cache, persistent=False)
            self._apply_rotary_emb_wrapped = rope_base.apply_rotary_emb
            if rope_base.get_global_server_args().rl_on_policy_target is not None:
                self._forward_method = self.forward_native
                self._apply_rotary_emb_wrapped = torch.compile(dynamic=True)(
                    rope_base.apply_rotary_emb
                )
            self.position_cos, self.position_sin = None, None

        RotaryEmbedding.__init__ = _sunrise_patched_rope_init

        _patched = True
        logger.info(
            "rotary_patch: RotaryEmbedding __init__ patched – "
            "rotary_embedding / mrotary_embedding bridges active on Sunrise"
        )

    except Exception as exc:
        logger.error(
            "rotary_patch: failed to patch rotary embedding __init__: %s", exc
        )
