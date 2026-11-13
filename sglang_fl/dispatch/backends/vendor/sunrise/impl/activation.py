# Copyright (c) 2026 BAAI. All rights reserved.

"""
Sunrise (PTPU) activation operator implementations.

Routes ``silu_and_mul`` to ``torch_ptpu.sgl_kernel.silu_and_mul``.
"""

from __future__ import annotations

import torch

from torch_ptpu.sgl_kernel import silu_and_mul as _ptpu_silu_and_mul


def silu_and_mul_sunrise(obj, x: torch.Tensor) -> torch.Tensor:
    """SiLU(gate) * up on PTPU (SwiGLU half-split).

    Args:
        obj: Calling ``SiluAndMul`` module (unused; dispatch ABI parity).
        x: ``[..., 2 * d]`` — ``[..., :d]`` gate, ``[..., d:]`` up.

    Returns:
        ``[..., d]``.
    """
    return _ptpu_silu_and_mul(x)
