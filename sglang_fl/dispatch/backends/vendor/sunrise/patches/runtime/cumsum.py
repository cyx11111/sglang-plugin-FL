# Copyright (c) 2026 BAAI. All rights reserved.
"""Make ``torch.cumsum`` on empty tensors skip the PTPU kernel.

Returns the (dtype-cast) empty input instead of launching the op.
"""

import torch

_patched = False


def patch_cumsum_empty_for_sunrise() -> None:
    global _patched
    if _patched:
        return
    _orig = torch.cumsum

    def cumsum(input, dim=0, *, dtype=None, out=None):
        if input.numel() == 0:
            r = input.to(dtype) if dtype is not None else input.clone()
            if out is not None:
                out.copy_(r)
                return out
            return r
        return _orig(input, dim, dtype=dtype, out=out)

    torch.cumsum = cumsum
    _patched = True
