# Copyright (c) 2026 BAAI. All rights reserved.
"""Load the Sunrise vendor package inside spawned SGLang subprocesses.

``Engine._launch_subprocesses`` forces the ``spawn`` start method, so scheduler
and detokenizer children are fresh interpreters. ``multiprocessing.spawn._main``
unpickles the process object — which imports ``sglang.srt.managers.scheduler``
and therefore ``memory_pool_host`` / NVIDIA ``sgl_kernel`` — *before* the child
ever reaches ``load_plugins()``. Anything the parent installed in ``sys.modules``
(the PTPU ``sgl_kernel`` stub) is gone by then.

The child command line is built in the parent by
``multiprocessing.spawn.get_command_line()``, which returns
``[python, ..., "-c", "from multiprocessing.spawn import spawn_main; ..."]``.
Prepending an import of the Sunrise vendor package to that ``-c`` program gives
the child the same ``sgl_kernel`` stub state as the parent, before any sglang
module is imported.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)

_patched = False

# Runs in the child interpreter ahead of ``spawn_main``. Keep it dependency-free:
# the vendor package only needs ``torch``; it must not import sglang here.
_PRELOAD_PROG = (
    "import sys as _sys\n"
    "try:\n"
    "    import sglang_fl.dispatch.backends.vendor.sunrise as _sglang_fl_sunrise\n"
    "except Exception as _exc:\n"
    "    print(f'[sglang_fl] sunrise spawn preload failed: {_exc}', file=_sys.stderr)\n"
)


def _on_ptpu() -> bool:
    try:
        return hasattr(torch, "ptpu") and bool(torch.ptpu.is_available())
    except Exception:
        return False


def patch_mp_spawn_preload_for_sunrise() -> None:
    """Inject the Sunrise preload into spawned children. Idempotent."""
    global _patched
    if _patched:
        return
    if not _on_ptpu():
        return

    from multiprocessing import spawn as mp_spawn

    original = mp_spawn.get_command_line
    if getattr(original, "_sglang_fl_sunrise_preload", False):
        _patched = True
        return

    def get_command_line(**kwds):
        cmd = original(**kwds)
        try:
            prog_idx = cmd.index("-c") + 1
        except ValueError:
            # Frozen interpreter: no ``-c`` program to extend.
            return cmd
        cmd[prog_idx] = _PRELOAD_PROG + cmd[prog_idx]
        return cmd

    get_command_line._sglang_fl_sunrise_preload = True
    mp_spawn.get_command_line = get_command_line
    _patched = True
    logger.info(
        "PTPU: spawned subprocesses will preload the Sunrise vendor package "
        "(sgl_kernel stub before sglang imports)"
    )
