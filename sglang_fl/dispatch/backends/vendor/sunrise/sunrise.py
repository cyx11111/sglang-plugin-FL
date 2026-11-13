# Sunrise backend implementation.

from __future__ import annotations

from typing import Optional

from sglang_fl.dispatch.backends import Backend


class SunriseBackend(Backend):
    """
    Sunrise backend for operator implementations.

    Uses Sunrise libraries (torch_ptpu).
    Ops that don't yet have a Sunrise-native kernel raise NotImplementedError,
    which lets OpManager fall back to flaggems or reference automatically.
    """

    _available: Optional[bool] = None

    @property
    def name(self) -> str:
        return "sunrise"

    @property
    def vendor(self) -> Optional[str]:
        return "sunrise"

    def is_available(self) -> bool:
        """
        Check if the sunrise hardware and libraries are available.

        Implement this method to detect if your vendor's hardware/software
        is present and functional.
        """
        import torch
        if SunriseBackend._available is None:
            # Check if Sunrise device is available
            try:
                if torch.ptpu.is_available() and torch.ptpu.device_count() > 0:
                    SunriseBackend._available = True
                else:
                    SunriseBackend._available = False
            except Exception:
                SunriseBackend._available = False
        return SunriseBackend._available

    def rotary_embedding(
        self,
        obj,
        query: torch.Tensor,
        key: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        position_ids: torch.Tensor,
        rotary_interleaved: bool = False,
        inplace: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply RoPE using ``torch_ptpu.sgl_kernel.apply_rope_with_cos_sin_cache``."""
        from .impl.attention.rotary import rotary_embedding_sunrise

        return rotary_embedding_sunrise(
            obj,
            query,
            key,
            cos,
            sin,
            position_ids,
            rotary_interleaved=rotary_interleaved,
            inplace=inplace,
        )
    def mrotary_embedding(
        self,
        obj,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """M-RoPE / text RoPE via PTPU ``apply_rope_with_cos_sin_cache`` (+ ``mrope_section``)."""
        from .impl.attention.mrotary_embedding import mrotary_embedding_sunrise

        return mrotary_embedding_sunrise(obj, positions, query, key)

    def rms_norm(
        self,
        obj,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """RMS normalization (and optional fused residual) on PTPU sgl_kernel."""
        from .impl.normalization import rms_norm_sunrise

        return rms_norm_sunrise(obj, x, residual)
        
    def gemma_rms_norm(
        self,
        obj,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """GemmaRMSNorm: PTPU rmsnorm with weight ``(1 + w)``."""
        from .impl.normalization import gemma_rms_norm_sunrise

        return gemma_rms_norm_sunrise(obj, x, residual)

    def chunk_gated_delta_rule(
        self,
        q,
        k,
        v,
        g,
        beta,
        scale,
        initial_state=None,
        initial_state_indices=None,
        cu_seqlens=None,
        head_first=False,
        use_qk_l2norm_in_kernel=False,
    ):
        from .impl.fla.fla_chunk import chunk_gated_delta_rule_sunrise

        return chunk_gated_delta_rule_sunrise(
            q,
            k,
            v,
            g,
            beta,
            scale,
            initial_state,
            initial_state_indices,
            cu_seqlens,
            head_first,
            use_qk_l2norm_in_kernel,
        )

    def fused_recurrent_gated_delta_rule_packed_decode(
        self,
        mixed_qkv,
        a,
        b,
        A_log,
        dt_bias,
        scale,
        initial_state,
        out,
        ssm_state_indices,
        use_qk_l2norm_in_kernel=False,
    ):
        from .impl.fla.fla_packed_decode import (
            fused_recurrent_gated_delta_rule_packed_decode,
        )

        return fused_recurrent_gated_delta_rule_packed_decode(
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
        
    def topk(
        self,
        obj,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        *,
        num_token_non_padded: Optional[torch.Tensor] = None,
        expert_location_dispatch_info=None,
    ):
        """MoE router via ``llm_kernel_ptpu.topk_softmax``; non-softmax routing falls back."""
        from .impl.moe.topk import topk_sunrise

        return topk_sunrise(
            obj,
            hidden_states,
            router_logits,
            num_token_non_padded=num_token_non_padded,
            expert_location_dispatch_info=expert_location_dispatch_info,
        )

    def fused_moe(self, obj, layer, dispatch_output):

        from .impl.moe.fused_moe import fused_moe_sunrise

        return fused_moe_sunrise(obj, layer, dispatch_output)

    def silu_and_mul(self, obj, x: torch.Tensor) -> torch.Tensor:
        """SiLU-and-mul (SwiGLU) via ``torch_ptpu.sgl_kernel.silu_and_mul``."""
        from .impl.activation import silu_and_mul_sunrise

        return silu_and_mul_sunrise(obj, x)