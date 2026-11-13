# Copyright (c) 2026 BAAI. All rights reserved.

"""PTPU INT8 activation quantization for the block-INT8 W8A8 path.

Replacement for sglang's Triton ``per_token_group_quant_int8``. One FP32 scale
per ``group_size`` elements of each token (``amax / 127``); values are
truncated toward zero, matching the upstream Triton kernel.
"""

from __future__ import annotations

from typing import Tuple

import torch

from torch_ptpu.sgl_kernel import per_token_group_quant_int8

_INT8_MIN = -128.0
_INT8_MAX = 127.0


def per_token_group_quant_int8_sunrise(
    x: torch.Tensor,
    group_size: int,
    eps: float = 1e-10,
    rows: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 2D ``[M, K]`` activation to INT8 with ``[M, K // group_size]`` FP32 scales.

    ``rows > M`` allocates ``rows`` output rows; rows past ``M`` are left uninitialised.
    """
    assert x.dim() == 2, f"expected 2D activation, got {tuple(x.shape)}"
    assert x.shape[-1] % group_size == 0, (
        f"K={x.shape[-1]} is not divisible by group_size={group_size}"
    )
    x = x.contiguous()
    m, k = x.shape
    n_rows = max(rows, m)
    x_q = torch.empty((n_rows, k), device=x.device, dtype=torch.int8)
    x_s = torch.empty((n_rows, k // group_size), device=x.device, dtype=torch.float32)
    if m > 0:
        per_token_group_quant_int8(x, x_q[:m], x_s[:m], group_size, eps, _INT8_MIN, _INT8_MAX)
    return x_q, x_s
