# Copyright (c) 2026 BAAI. All rights reserved.
"""PTPU drop-in for ``fla.l2norm.l2norm_fwd``."""

from __future__ import annotations

from typing import Optional

import torch


def l2norm_fwd(
    x: torch.Tensor,
    eps: float = 1e-6,
    output_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Last-axis L2 normalization on PTPU.

    Mirrors SGLang's contract (same shape out; only the last dim is
    normalized). SGLang's helper collapses the leading dims to a 2-D matrix
    before its Triton kernel — not needed here since PTPU's normalize works on
    the last dim regardless of rank.
    """
    from torch_ptpu.sgl_kernel import ptpu_l2_normalize as _ptpu_l2_normalize

    out = _ptpu_l2_normalize(x, p=2.0, dim=-1, eps=eps)
    if output_dtype is not None and out.dtype != output_dtype:
        out = out.to(output_dtype)
    return out
