# Copyright (c) 2026 BAAI. All rights reserved.

"""Sunrise patch implementations.

Classification
--------------
* **Package import** (``sunrise/__init__.py``, on PTPU only):
  ``patch_sgl_kernel_stub``, ``patch_mp_spawn_preload``. Scheduler subprocesses
  import ``memory_pool_host`` -> ``sgl_kernel`` before ``load_plugins()``.
* **Plugin load** (``patch.py``, via ``sglang_fl.load_plugin``):
  - ``apply_sunrise_patches``: supported devices, ``torch.cumsum`` on empty
    tensors, eager vocab mask, compressed-tensors W8A8 quantization routing.
  - ``apply_sunrise_load_patches``: rotary-embedding init, FLA gated RMSNorm,
    multimodal image processor device, eager ViT rotary, FlagGems paged-flash,
    in-place MoE shared-expert gate.
"""
