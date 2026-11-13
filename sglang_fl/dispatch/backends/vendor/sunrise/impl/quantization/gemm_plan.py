# Copyright (c) 2026 BAAI. All rights reserved.

"""Per-(N, K, M) choice of how to run a dense block-INT8 GEMM.

The vendor dispatcher has tuned configs for a fixed list of (N, K); other
shapes use a generic heuristic whose speed jumps between neighbouring M
(27B gate_up, N=34816 K=5120: 0.332 ms at M=40, 0.210 ms at M=48). For each
decode-size M the weight-loading step times, inside a PTPU graph:

* ``raw``   the GEMM at M;
* ``pad``   the GEMM on M' > M rows (at most 2M), extra rows dropped;
* ``split`` two GEMMs on a row split of the weight, outputs concatenated
  (pieces may hit tuned shapes, e.g. 34816 = 2 x 17408).

and keeps the fastest when it beats ``raw`` by ``_MIN_GAIN``. M without a
plan keep the static rules in ``dense_gemm``.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Callable, Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

_ENABLED = os.environ.get("SGLANG_FL_SUNRISE_INT8_GEMM_TUNE", "1") != "0"
_M_MIN = int(os.environ.get("SGLANG_FL_SUNRISE_INT8_GEMM_TUNE_M_MIN", "32"))
_M_MAX = int(os.environ.get("SGLANG_FL_SUNRISE_INT8_GEMM_TUNE_M_MAX", "256"))
_MIN_GAIN = 1.05
_PAD_TARGETS = (48, 64, 96, 128, 192, 256)
_SPLIT_FRACTIONS = (1 / 2, 5 / 8, 3 / 4)
_REPS = 10
_REPLAYS = 3

Plan = Tuple[str, int]  # ("raw", 0) | ("pad", rows) | ("split", n_first)
_plans: Dict[tuple, Dict[int, Plan]] = {}


def lookup(m: int, n: int, k: int, block_n: int, block_k: int) -> Optional[Plan]:
    """The timed plan for this M, or ``None`` if M was not timed."""
    table = _plans.get((n, k, block_n, block_k))
    return None if table is None else table.get(m)


def _decode_ms() -> List[int]:
    try:
        from sglang.srt.server_args import get_global_server_args

        sa = get_global_server_args()
        caps = [c for c in (sa.cuda_graph_max_bs, sa.max_running_requests) if c]
        cap = min(caps) if caps else _M_MAX
        bss = [b for b in (sa.cuda_graph_bs or []) if b <= cap]
        ms = set(bss)
        if sa.speculative_algorithm is not None and sa.speculative_num_draft_tokens:
            ms |= {b * sa.speculative_num_draft_tokens for b in bss}
    except Exception:
        ms = set(range(32, 129, 8))
    return sorted(m for m in ms if _M_MIN <= m <= _M_MAX)


def _graph_ms(fn: Callable[[], object]) -> float:
    from ..runtime.graph_runner import _ptpu_capture_stream, _ptpu_graph_cls

    fn()
    torch.ptpu.synchronize()
    graph = _ptpu_graph_cls()()
    with torch.ptpu.graph(graph, stream=_ptpu_capture_stream(None)):
        for _ in range(_REPS):
            fn()
    graph.replay()
    torch.ptpu.synchronize()
    t = time.perf_counter()
    for _ in range(_REPLAYS):
        graph.replay()
    torch.ptpu.synchronize()
    ms = (time.perf_counter() - t) / (_REPLAYS * _REPS) * 1e3
    del graph
    return ms


def tune(weight: torch.Tensor, weight_scale: torch.Tensor, block_n: int, block_k: int,
         run: Callable[[torch.Tensor, Optional[Plan]], torch.Tensor]) -> None:
    """Fill the plan table for ``weight``'s (N, K); ``run(x, plan)`` runs the GEMM
    with an explicit plan (``None`` = raw)."""
    n, k = weight.shape
    key = (n, k, block_n, block_k)
    if not _ENABLED or key in _plans or weight.device.type != "ptpu":
        return
    try:
        from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode

        if get_is_capture_mode():
            return
    except Exception:
        pass
    ms = _decode_ms()
    _plans[key] = {}
    if not ms:
        return
    try:
        table = _tune(n, k, block_n, block_k, ms, weight, run)
    except Exception as exc:  # timing problems must not break loading
        logger.warning("sunrise: INT8 GEMM tuning N=%d K=%d skipped (%s)", n, k, exc)
        table = {}
    _plans[key] = table
    torch.ptpu.empty_cache()
    changed = {m: p for m, p in table.items() if p[0] != "raw"}
    if changed:
        logger.info("sunrise: INT8 GEMM N=%d K=%d plans %s", n, k,
                    " ".join(f"{m}:{p[0]}{p[1]}" for m, p in sorted(changed.items())))


def _tune(n, k, block_n, block_k, ms, weight, run) -> Dict[int, Plan]:
    rows = sorted(set(ms) | {t for t in _PAD_TARGETS if t <= max(2 * max(ms), 1)})
    x = torch.randn(max(rows), k, device=weight.device, dtype=torch.bfloat16)
    raw = {m: _graph_ms(lambda m=m: run(x[:m], None)) for m in rows}

    splits = sorted({int(round(n * f / block_n)) * block_n for f in _SPLIT_FRACTIONS})
    splits = [s for s in splits if 0 < s < n] if n >= 4096 else []
    split = None
    if splits:
        probe = min(ms, key=lambda m: abs(m - 64))
        times = {s: _graph_ms(lambda s=s: run(x[:probe], ("split", s))) for s in splits}
        split = min(times, key=times.get)

    table: Dict[int, Plan] = {}
    for m in ms:
        best_t, best = raw[m], None
        pads = [r for r in rows if m < r <= 2 * m and raw[r] < raw[m]]
        if pads:
            r = min(pads, key=raw.get)
            t = _graph_ms(lambda m=m, r=r: run(x[:m], ("pad", r)))
            if t < best_t:
                best_t, best = t, ("pad", r)
        if split is not None:
            t = _graph_ms(lambda m=m: run(x[:m], ("split", split)))
            if t < best_t:
                best_t, best = t, ("split", split)
        table[m] = best if best is not None and raw[m] >= _MIN_GAIN * best_t else ("raw", 0)
    return table
