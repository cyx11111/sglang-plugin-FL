# Copyright (c) 2026 BAAI. All rights reserved.

"""Sunrise vendor backend for sglang-plugin-FL dispatch.

The public Sunrise runtime uses the operators bundled with ``torch_ptpu``.
It does not depend on the newer, separately packaged ``ptpu-opmux``.
"""

from __future__ import annotations

def _on_ptpu() -> bool:
    try:
        import torch

        return hasattr(torch, "ptpu") and bool(torch.ptpu.is_available())
    except Exception:
        return False

# Dispatch discovery imports every vendor package; only run PTPU work on Sunrise.
if _on_ptpu():
    from .patches.runtime.mp_spawn_preload import (  # noqa: E402
        patch_mp_spawn_preload_for_sunrise,
    )
    from .patches.runtime.sgl_kernel_stub import (  # noqa: E402
        patch_sgl_kernel_stub_for_sunrise,
    )

    patch_sgl_kernel_stub_for_sunrise()
    # Spawned schedulers import memory_pool_host before load_plugins() runs.
    patch_mp_spawn_preload_for_sunrise()
