# Copyright (c) 2026 BAAI. All rights reserved.
"""Apply the Qwen2/Qwen3.5 MoE shared-expert gate on PTPU without the ``[M, H]`` broadcast.

Upstream computes ``F.sigmoid(shared_expert_gate(x)) * shared_output``. The
``[M, 1] * [M, H]`` broadcast ``mul`` runs on the native torch_ptpu kernel
(``mul`` is on the FlagGems blacklist), which is ~15x slower than FlagGems'
in-place ``mul_`` for this pattern (4.5 ms vs 0.3 ms at M=8192, H=2048).

The gate is a per-token scalar, so ``down(act) * s == down(act * s)``: it is
applied in place to the ``[M, I/tp]`` activation before ``down_proj`` (16x
smaller than the output for Qwen3.5 TP=4; 0.34 ms -> ~0.03 ms at M=8192).
Other shared-expert module types keep the in-place ``mul_`` on the output.

Prefill-sized gates (``M >= 1024``) are computed as ``sigmoid(x @ w.T)`` in one
Triton row pass (``impl/row_dot.py``; 0.114 -> 0.031 ms at 8192 x 2048);
disable with ``SGLANG_FL_SUNRISE_SHARED_GATE_TRITON=0``.
"""

import logging
import os

logger = logging.getLogger(__name__)

_patched = False


def patch_moe_shared_expert_gate_for_sunrise() -> None:
    global _patched
    if _patched:
        return

    import torch
    import torch.nn.functional as F

    from sglang.srt.models import qwen2_moe

    from ...impl.moe import row_dot as _row_dot_triton

    block = qwen2_moe.Qwen2MoeSparseMoeBlock
    mlp_cls = qwen2_moe.Qwen2MoeMLP
    orig = block._forward_shared_experts
    gate_triton = os.environ.get("SGLANG_FL_SUNRISE_SHARED_GATE_TRITON", "1") != "0"

    def _gate(sg, hidden_states):
        if (
            gate_triton
            and type(sg) is torch.nn.Linear
            and sg.bias is None
            and _row_dot_triton.supported(hidden_states, sg.weight)
        ):
            return _row_dot_triton.sigmoid_row_dot(hidden_states, sg.weight)
        return F.sigmoid(sg(hidden_states))

    def _forward_shared_experts(self, hidden_states):
        if self.shared_expert is None or self.shared_expert_gate is None:
            return orig(self, hidden_states)
        if qwen2_moe.use_intel_amx_backend(self.shared_expert_gate):
            return orig(self, hidden_states)
        gate = _gate(self.shared_expert_gate, hidden_states)
        se = self.shared_expert
        if type(se) is mlp_cls:
            gate_up, _ = se.gate_up_proj(hidden_states)
            act = se.act_fn(gate_up)
            act.mul_(gate)
            shared_output, _ = se.down_proj(act)
            return shared_output
        return se(hidden_states).mul_(gate)

    block._forward_shared_experts = _forward_shared_experts
    _patched = True
    logger.info("sunrise: Qwen2MoeSparseMoeBlock shared-expert gate -> pre-down_proj mul_")
