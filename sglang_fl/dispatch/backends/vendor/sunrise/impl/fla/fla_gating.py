# Copyright (c) 2026 BAAI. All rights reserved.
"""GDN ``fused_gdn_gating`` with a flat grid.

SGLang's kernel launches one 1-warp program per (token, 8 heads): 32768
programs for 8192 tokens at TP1, ~0.49 ms per layer on PTPU for a few hundred
KB of data. This version handles ``BLOCK`` flattened (token, head) elements
per program with the same arithmetic and output dtypes:

    g    = -exp(A_log) * softplus(a + dt_bias)          (fp32)
    beta = sigmoid(b), rounded to b.dtype, stored fp32

``a`` / ``b`` may be row-strided views (unit stride along heads).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

_BLOCK = 1024


@triton.jit
def _gdn_gating_flat_kernel(
    g_ptr, beta_ptr, A_log_ptr, a_ptr, b_ptr, dt_bias_ptr,
    n, stride_a, stride_b,
    HV: tl.constexpr, softplus_beta: tl.constexpr, threshold: tl.constexpr, BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tok = offs // HV
    head = offs % HV
    A_log = tl.load(A_log_ptr + head, mask=mask, other=0.0)
    a = tl.load(a_ptr + tok * stride_a + head, mask=mask, other=0.0)
    b = tl.load(b_ptr + tok * stride_b + head, mask=mask, other=0.0)
    bias = tl.load(dt_bias_ptr + head, mask=mask, other=0.0)
    x = a.to(tl.float32) + bias.to(tl.float32)
    softplus_x = tl.where(
        softplus_beta * x <= threshold, (1 / softplus_beta) * tl.log(1 + tl.exp(softplus_beta * x)), x
    )
    g = -tl.exp(A_log.to(tl.float32)) * softplus_x
    tl.store(g_ptr + offs, g.to(g_ptr.dtype.element_ty), mask=mask)
    beta = tl.sigmoid(b.to(tl.float32))
    tl.store(beta_ptr + offs, beta.to(b_ptr.dtype.element_ty), mask=mask)


def fused_gdn_gating_flat(A_log, a, b, dt_bias, beta: float = 1.0, threshold: float = 20.0):
    batch, num_heads = a.shape
    g = torch.empty(1, batch, num_heads, dtype=torch.float32, device=a.device)
    beta_out = torch.empty(1, batch, num_heads, dtype=torch.float32, device=b.device)
    n = batch * num_heads
    if n:
        _gdn_gating_flat_kernel[(triton.cdiv(n, _BLOCK),)](
            g, beta_out, A_log, a, b, dt_bias, n, a.stride(0), b.stride(0),
            HV=num_heads, softplus_beta=beta, threshold=threshold, BLOCK=_BLOCK, num_warps=4,
        )
    return g, beta_out


def gating_supported(a, b) -> bool:
    return a.dim() == 2 and b.dim() == 2 and a.stride(1) == 1 and b.stride(1) == 1 and a.shape == b.shape
