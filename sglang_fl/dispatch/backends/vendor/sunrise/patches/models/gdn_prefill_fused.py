# Copyright (c) 2026 BAAI. All rights reserved.
"""GDN prefill: fused conv1d + SiLU + q/k/v split + q/k L2 norm on PTPU.

``GDNAttnBackend.forward_extend`` runs SGLang's Triton ``causal_conv1d_fn``,
then the chunk macro L2-normalises q / k and copies v contiguous (~4.4 ms per
layer at TP1 / 4x2048 tokens). One Triton pass (``_conv_l2norm_prefill``)
produces the normalised contiguous q / k / v and the conv-state update in
~0.5-0.9 ms; the chunk rule is then called with
``use_qk_l2norm_in_kernel=False``. Target-verify and layouts the kernel does
not cover use the original method.

Disable with ``SGLANG_FL_SUNRISE_GDN_PREFILL_FUSED=0``.

Prefill-sized extends also skip the ``z`` copy: ``Qwen3_5GatedDeltaNet.forward``
runs here and the gated norm reads ``z`` as the column slice of the qkvz
projection (``SGLANG_FL_SUNRISE_GDN_Z_VIEW=0`` keeps the copy).
"""

import contextvars
import logging
import os

import torch

logger = logging.getLogger(__name__)

_prefill_patched = False
_split_patched = False
_qkv_view = contextvars.ContextVar("sunrise_gdn_qkv_view", default=False)


def fused_qkvzba_split_reshape_cat_contiguous_torch(
    mixed_qkvz, mixed_ba, num_heads_qk, num_heads_v, head_qk, head_v
):
    """Split contiguous Q/K/V/Z and B/A projections with torch slicing."""
    qkv_dim = num_heads_qk * head_qk * 2 + num_heads_v * head_v
    mixed_qkv = mixed_qkvz[:, :qkv_dim].contiguous()
    z = mixed_qkvz[:, qkv_dim:].contiguous().view(-1, num_heads_v, head_v)
    b = mixed_ba[:, :num_heads_v].contiguous()
    a = mixed_ba[:, num_heads_v:].contiguous()
    return mixed_qkv, z, b, a


def patch_gdn_qkvz_split_for_sunrise() -> None:
    global _split_patched
    if (
        _split_patched
        or os.environ.get("SGLANG_FL_SUNRISE_GDN_TORCH_SPLIT", "1") == "0"
    ):
        return
    try:
        from sglang.srt.models import qwen3_5
    except Exception as exc:
        logger.info(
            "sunrise: qwen3_5 unavailable, GDN split not patched (%s)", exc
        )
        return
    if not hasattr(qwen3_5, "fused_qkvzba_split_reshape_cat_contiguous"):
        return
    qwen3_5.fused_qkvzba_split_reshape_cat_contiguous = (
        fused_qkvzba_split_reshape_cat_contiguous_torch
    )
    _split_patched = True
    logger.info("sunrise: Qwen3.5 GDN qkvz/ba split -> torch slicing")


def _fits(self, layer, forward_batch, mixed_qkv, conv_states) -> bool:
    from sglang.srt.layers.attention.linear.kernels.gdn_triton import TritonGDNKernel

    if forward_batch.forward_mode.is_target_verify():
        return False
    if not isinstance(self.kernel_dispatcher.extend_kernel, TritonGDNKernel):
        return False
    if forward_batch.extend_seq_lens_cpu is None or forward_batch.extend_prefix_lens is None:
        return False
    if not (mixed_qkv.dim() == 2 and mixed_qkv.device.type == "ptpu" and mixed_qkv.stride(1) == 1):
        return False
    if layer.activation not in (None, "silu", "swish"):
        return False
    w = layer.conv_weights
    d = layer.head_q_dim
    dim = mixed_qkv.shape[1]
    return (
        w.dim() == 2 and w.shape == (dim, 4)
        and d == layer.head_k_dim == layer.head_v_dim and d & (d - 1) == 0
        and layer.num_q_heads == layer.num_k_heads
        and layer.q_dim == layer.k_dim == layer.num_k_heads * d
        and layer.v_dim == layer.num_v_heads * d
        and dim == layer.q_dim + layer.k_dim + layer.v_dim
        and conv_states.dim() == 3 and conv_states.shape[1] == dim and conv_states.shape[2] >= 3
        and conv_states.dtype == mixed_qkv.dtype
    )


def _patch_qkv_views() -> None:
    """In non-verify extend, hand ``mixed_qkv`` / ``b`` / ``a`` to the backend as
    column views of the projections (the fused kernels take row strides),
    skipping the ``[T, qkv_dim]`` copy. Decode / verify keep the copies."""
    if os.environ.get("SGLANG_FL_SUNRISE_GDN_QKV_VIEW", "1") == "0":
        return
    try:
        from sglang.srt.models import qwen3_5
    except Exception:
        return
    name = "fused_qkvzba_split_reshape_cat_contiguous"
    if not hasattr(qwen3_5, name):
        return
    inner_split = getattr(qwen3_5, name)

    def split(mixed_qkvz, mixed_ba, num_heads_qk, num_heads_v, head_qk, head_v):
        if not _qkv_view.get():
            return inner_split(mixed_qkvz, mixed_ba, num_heads_qk, num_heads_v, head_qk, head_v)
        qkv_dim = num_heads_qk * head_qk * 2 + num_heads_v * head_v
        z = mixed_qkvz[:, qkv_dim:].contiguous().view(-1, num_heads_v, head_v)
        return mixed_qkvz[:, :qkv_dim], z, mixed_ba[:, :num_heads_v], mixed_ba[:, num_heads_v:]

    setattr(qwen3_5, name, split)

    from ...impl.fla import fla_rms_norm_gated as rng
    # SGLang only takes the fused split for GVA ratios 1 / 2 / 4 (a Triton
    # kernel limit) and otherwise copies q / k / v out and concatenates them
    # again. The torch split covers the same ``[q|k|v|z]`` layout for any ratio.
    any_ratio = inner_split is fused_qkvzba_split_reshape_cat_contiguous_torch

    def plain_split(self):
        return self.num_v_heads // self.num_k_heads in (1, 2, 4) or any_ratio

    def forward_split(self, hidden_states, forward_batch):
        qkvz, ba = self._forward_input_proj(hidden_states)
        mixed_qkv, z, b, a = split(
            qkvz, ba,
            -(-self.num_k_heads // self.attn_tp_size), -(-self.num_v_heads // self.attn_tp_size),
            self.head_k_dim, self.head_v_dim,
        )
        core = self.attn(forward_batch, mixed_qkv=mixed_qkv, a=a, b=b)
        z_shape_og = z.shape
        core = core.reshape(-1, core.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        if core.shape != z.shape:
            pad = torch.zeros_like(z)
            pad[: core.shape[0], :] = core
            core = pad
        out = self.norm(core, z).reshape(z_shape_og)
        output, _ = self.out_proj(out.reshape(*out.shape[:-2], -1))
        return output

    gdn_cls = qwen3_5.Qwen3_5GatedDeltaNet
    inner_forward = gdn_cls.forward
    z_view = (
        os.environ.get("SGLANG_FL_SUNRISE_GDN_Z_VIEW", "1") != "0"
        and os.environ.get("SGLANG_FL_SUNRISE_GDN_NORM_FUSED", "1") != "0"
    )

    def z_view_fits(self, hidden_states):
        norm = self.norm
        nv = -(-self.num_v_heads // self.attn_tp_size)
        return (
            plain_split(self)
            and hidden_states.dim() == 2
            and hidden_states.shape[0] * nv >= rng.MIN_ROWS
            and norm.group_size in (None, self.head_v_dim)
            and norm.norm_before_gate
            and getattr(norm, "bias", None) is None
            and norm.activation in ("silu", "swish")
        )

    def forward_z_view(self, hidden_states, forward_batch):
        qkvz, ba = self._forward_input_proj(hidden_states)
        nk = -(-self.num_k_heads // self.attn_tp_size)
        nv = -(-self.num_v_heads // self.attn_tp_size)
        dv = self.head_v_dim
        qkv_dim = 2 * nk * self.head_k_dim + nv * dv
        z = qkvz[:, qkv_dim:]
        core = self.attn(forward_batch, mixed_qkv=qkvz[:, :qkv_dim], a=ba[:, nv:], b=ba[:, :nv])
        x = core.reshape(-1, dv)
        if rng.heads_supported(x, z, self.norm.weight, nv):
            out = rng.rms_norm_gated_heads(x, z, self.norm.weight, self.norm.eps, nv)
        else:
            z = z.reshape(-1, dv)
            if x.shape != z.shape:
                pad = torch.zeros_like(z)
                pad[: x.shape[0], :] = x
                x = pad
            out = self.norm(x, z)
        output, _ = self.out_proj(out.reshape(qkvz.shape[0], nv * dv))
        return output

    def forward(self, hidden_states, forward_batch, *args, **kwargs):
        mode = forward_batch.forward_mode
        view = mode.is_extend() and not mode.is_target_verify()
        if view and z_view and not args and not kwargs and z_view_fits(self, hidden_states):
            return forward_z_view(self, hidden_states, forward_batch)
        token = _qkv_view.set(view)
        try:
            if (
                any_ratio and not args and not kwargs
                and self.num_v_heads // self.num_k_heads not in (1, 2, 4)
            ):
                return forward_split(self, hidden_states, forward_batch)
            return inner_forward(self, hidden_states, forward_batch, *args, **kwargs)
        finally:
            _qkv_view.reset(token)

    gdn_cls.forward = forward


def patch_gdn_prefill_fused_for_sunrise() -> None:
    global _prefill_patched
    if (
        _prefill_patched
        or os.environ.get("SGLANG_FL_SUNRISE_GDN_PREFILL_FUSED", "1") == "0"
    ):
        return

    from sglang.srt.layers.attention.linear import gdn_backend
    from sglang.srt.layers.attention.linear.kernels import gdn_triton

    from ...impl.fla.fla_conv_l2norm import conv_split_l2norm
    from ...impl.fla.fla_gating import fused_gdn_gating_flat, gating_supported

    if os.environ.get("SGLANG_FL_SUNRISE_GDN_GATING_FLAT", "1") != "0":
        gating_orig = gdn_backend.fused_gdn_gating

        def fused_gdn_gating(A_log, a, b, dt_bias, beta=1.0, threshold=20.0):
            if gating_supported(a, b):
                return fused_gdn_gating_flat(A_log, a, b, dt_bias, beta, threshold)
            return gating_orig(A_log, a, b, dt_bias, beta, threshold)

        gdn_backend.fused_gdn_gating = fused_gdn_gating

    _patch_qkv_views()

    original = gdn_backend.GDNAttnBackend.forward_extend

    def forward_extend(self, layer, forward_batch, mixed_qkv, a, b, **kwargs):
        if not isinstance(mixed_qkv, gdn_backend.torch.Tensor):
            return original(self, layer, forward_batch, mixed_qkv, a, b, **kwargs)
        cache = self.req_to_token_pool.mamba2_layer_cache(layer.layer_id)
        conv_states = cache.conv[0]
        if not _fits(self, layer, forward_batch, mixed_qkv, conv_states):
            return original(self, layer, forward_batch, mixed_qkv.contiguous(), a, b, **kwargs)

        fm = self.forward_metadata
        query_start_loc = fm.query_start_loc
        cache_indices = fm.mamba_cache_indices
        ssm_states = cache.temporal

        if fm.has_mamba_track_mask:
            mixed_t = mixed_qkv.transpose(0, 1)
            conv_states[fm.conv_states_mask_indices] = mixed_t[:, fm.track_conv_indices].transpose(0, 1)

        q, k, v = conv_split_l2norm(
            mixed_qkv, layer.conv_weights, layer.bias, conv_states, cache_indices,
            forward_batch.extend_prefix_lens > 0, query_start_loc,
            max(forward_batch.extend_seq_lens_cpu, default=0),
            layer.num_k_heads, layer.num_v_heads, layer.head_k_dim,
            silu=layer.activation is not None,
        )
        g, beta = gdn_backend.fused_gdn_gating(layer.A_log, a, b, layer.dt_bias)
        core_attn_out, _, h = gdn_triton.chunk_gated_delta_rule(
            q=q, k=k, v=v, g=g, beta=beta,
            initial_state=ssm_states,
            initial_state_indices=cache_indices,
            cu_seqlens=query_start_loc,
            head_first=False,
            use_qk_l2norm_in_kernel=False,
        )
        if h is not None:
            self._track_mamba_state_extend(forward_batch, h, ssm_states, fm)
        return core_attn_out

    gdn_backend.GDNAttnBackend.forward_extend = forward_extend
    _prefill_patched = True
    logger.info("sunrise: GDN prefill conv1d+split+l2norm fused (Triton)")
