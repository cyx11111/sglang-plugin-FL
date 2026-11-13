# Copyright (c) 2026 BAAI. All rights reserved.
"""MTP / EAGLE speculative decoding (spec v2, ``topk == 1``) on PTPU.

SGLang's speculative path assumes CUDA / NPU in a few places:

* ``MambaPool`` allocates the verify buffers (``intermediate_ssm``,
  ``intermediate_conv_window``) with ``device="cuda"``; here they go on the
  pool device.
* GDN target verify: SGLang's Triton recurrent kernel uses the 2-D state-tile
  reduction that hangs PTPU, and its conv update takes 0.6 ms per layer. Both go
  to ``impl/fla_gdn_verify`` (vendor recurrent kernel per draft position);
  ``update_mamba_state_after_mtp_verify`` uses the same module because SGLang's
  scatter rejects non-CUDA tensors.
* ``build_tree_kernel_efficient`` / ``verify_tree_greedy`` are CUDA / NPU ops:
  ``impl/spec_kernels`` covers EAGLE chains (``topk == 1``).
* ``eagle_info_v2``: non-greedy verify (``tree_speculative_sampling_target_only``)
  and ``assign_extend_cache_locs_func`` have no PTPU branch; PTPU takes the HIP
  branches (greedy verify for every request, Triton cache-location kernel).
* ``DraftBackendFactory`` has no ``sunrise`` entry (draft-extend backend), and the
  draft-extend graph is only captured on CUDA / MUSA / NPU; PTPU uses
  ``impl/graph_runner.ptpu_draft_extend_graph_runner_cls``
  (``SGLANG_FL_SUNRISE_SPEC_DRAFT_EXTEND_GRAPH=0`` keeps it eager).
* ``Qwen3_5ForCausalLMMTP.set_embed_and_head`` calls ``torch.cuda.synchronize``.
* ``spec_utils`` helpers are ``torch.compile``'d (disabled on NPU only); Inductor
  has no PTPU backend, so they run eagerly.
* ``aten::record_stream`` has no PTPU kernel; the spec v2 overlap path calls it
  on small index / length tensors read by the forward stream. PTPU tensors are
  instead kept alive in a bounded queue (hundreds of steps, each ending in a
  device sync).

Multi-step drafts (``speculative_num_steps > 1``) are not supported yet.
Disable with ``SGLANG_FL_SUNRISE_SPEC=0``.
"""

from __future__ import annotations

import logging
import os
import sys

import torch

logger = logging.getLogger(__name__)

_patched = False


def _on_ptpu(t: torch.Tensor) -> bool:
    return t.device.type == "ptpu"


def _patch_mamba_pool() -> None:
    from sglang.srt.mem_cache import memory_pool as mp

    pool_cls = mp.MambaPool
    orig_init = pool_cls.__init__

    def __init__(self, *args, **kwargs):
        n_draft = kwargs.get("speculative_num_draft_tokens")
        device = kwargs.get("device")
        if n_draft is None or device is None or torch.device(device).type != "ptpu":
            return orig_init(self, *args, **kwargs)
        orig_init(self, *args, **dict(kwargs, speculative_num_draft_tokens=None))
        st = self.mamba_cache
        lead = (st.temporal.shape[0], kwargs["spec_state_size"] + 1, n_draft)
        self.mamba_cache = pool_cls.SpeculativeState(
            conv=st.conv,
            temporal=st.temporal,
            intermediate_ssm=torch.zeros(
                lead + tuple(st.temporal.shape[2:]), dtype=st.temporal.dtype, device=device
            ),
            intermediate_conv_window=[
                torch.zeros(lead + tuple(c.shape[2:]), dtype=c.dtype, device=device) for c in st.conv
            ],
        )
        self.mem_usage = self.mamba_cache.mem_usage_bytes() / (1 << 30)
        logger.info(
            "sunrise: MambaPool verify buffers on %s: intermediate_ssm %.2f GB",
            device, self.mamba_cache.intermediate_ssm.numel()
            * self.mamba_cache.intermediate_ssm.element_size() / (1 << 30),
        )

    pool_cls.__init__ = __init__


def _patch_gdn_verify() -> None:
    from sglang.srt.layers.attention import hybrid_linear_attn_backend as hlab
    from sglang.srt.layers.attention.linear.kernels import gdn_triton

    from ...impl.fla import (
        fla_gdn_verify as gdn_verify,
        fla_state_layout as _state_layout,
    )

    kernel_cls = gdn_triton.TritonGDNKernel
    orig_verify = kernel_cls.target_verify

    def target_verify(self, *args, **kwargs):
        q = kwargs.get("q", args[2] if len(args) > 2 else None)
        if q is None or not _on_ptpu(q):
            return orig_verify(self, *args, **kwargs)
        if not _state_layout.native_state_layout():
            raise NotImplementedError(
                "sunrise GDN verify needs the PTPU state layout (SGLANG_FL_SUNRISE_GDN_NATIVE_STATE=1)"
            )
        names = ("A_log", "dt_bias", "q", "k", "v", "a", "b")
        kw = dict(zip(names, args), **kwargs)
        return gdn_verify.target_verify(**kw)

    kernel_cls.target_verify = target_verify

    backend_cls = hlab.HybridLinearAttnBackend
    orig_update = backend_cls.update_mamba_state_after_mtp_verify

    def update_mamba_state_after_mtp_verify(
        self, accepted_steps, mamba_track_indices, mamba_steps_to_track, model
    ):
        if not _on_ptpu(accepted_steps):
            return orig_update(self, accepted_steps, mamba_track_indices, mamba_steps_to_track, model)
        if mamba_track_indices is not None:
            raise NotImplementedError("sunrise MTP: mamba prefix-cache tracking (radix cache) is not supported")
        n = accepted_steps.shape[0]
        lin = self.linear_attn_backend
        gdn_verify.scatter_accepted(
            lin.req_to_token_pool.get_speculative_mamba2_params_all_layers(),
            lin.forward_metadata.mamba_cache_indices[:n],
            accepted_steps,
        )

    backend_cls.update_mamba_state_after_mtp_verify = update_mamba_state_after_mtp_verify


def _patch_tree_ops() -> None:
    from sglang.srt.speculative import eagle_utils

    from ...impl.runtime import spec_kernels

    orig_build = eagle_utils.build_tree_kernel_efficient
    orig_verify = eagle_utils.verify_tree_greedy_func

    def build_tree_kernel_efficient(
        verified_id, parent_list, top_scores_index, draft_tokens, seq_lens, seq_lens_sum,
        topk, spec_steps, num_verify_tokens,
        tree_mask_mode=eagle_utils.TreeMaskMode.FULL_MASK, tree_mask_buf=None, position_buf=None,
    ):
        if not _on_ptpu(seq_lens):
            return orig_build(
                verified_id, parent_list, top_scores_index, draft_tokens, seq_lens, seq_lens_sum,
                topk, spec_steps, num_verify_tokens, tree_mask_mode, tree_mask_buf, position_buf,
            )
        if topk != 1:
            raise NotImplementedError("sunrise EAGLE: only chains (topk == 1) are supported")
        if tree_mask_mode not in (eagle_utils.TreeMaskMode.FULL_MASK, eagle_utils.TreeMaskMode.QLEN_ONLY):
            raise NotImplementedError(f"sunrise EAGLE chain: tree mask mode {tree_mask_mode}")
        draft_tokens = torch.cat((verified_id.unsqueeze(1), draft_tokens), dim=1).flatten()
        mask, positions, r_index, r_next, r_sibling = spec_kernels.chain_tree(
            seq_lens, seq_lens_sum, num_verify_tokens, tree_mask_mode == eagle_utils.TreeMaskMode.FULL_MASK,
            tree_mask_buf, position_buf,
        )
        return mask, positions, r_index, r_next, r_sibling, draft_tokens

    def verify_tree_greedy_func(
        predicts, accept_index, accept_token_num, candidates, retrieve_index,
        retrieve_next_token, retrieve_next_sibling, target_predict, topk=-1,
    ):
        if not _on_ptpu(predicts):
            return orig_verify(
                predicts, accept_index, accept_token_num, candidates, retrieve_index,
                retrieve_next_token, retrieve_next_sibling, target_predict, topk,
            )
        if topk != 1:
            raise NotImplementedError("sunrise EAGLE: only chains (topk == 1) are supported")
        return spec_kernels.chain_verify(predicts, accept_index, accept_token_num, candidates, target_predict)

    # Callers import these by name; rebind every loaded speculative module.
    import sglang.srt.speculative.eagle_info  # noqa: F401
    import sglang.srt.speculative.eagle_info_v2  # noqa: F401
    import sglang.srt.speculative.eagle_worker_v2  # noqa: F401

    for name, mod in list(sys.modules.items()):
        if not name.startswith("sglang.srt.speculative") or mod is None:
            continue
        if getattr(mod, "build_tree_kernel_efficient", None) is orig_build:
            mod.build_tree_kernel_efficient = build_tree_kernel_efficient
        if getattr(mod, "verify_tree_greedy_func", None) is orig_verify:
            mod.verify_tree_greedy_func = verify_tree_greedy_func


def _patch_eagle_info_v2() -> None:
    from sglang.srt.speculative import eagle_info_v2

    # Only read at call time by ``sample`` (greedy verify for every request,
    # instead of the CUDA-only tree_speculative_sampling_target_only) and by
    # ``assign_extend_cache_locs_func`` (Triton kernel instead of returning None).
    eagle_info_v2._is_hip = True
    logger.info("sunrise: EAGLE v2 verify is greedy for all requests on PTPU (as on HIP / NPU)")


def _patch_draft_backend_factory() -> None:
    from sglang.srt.speculative import draft_utils

    fac = draft_utils.DraftBackendFactory
    orig_decode = fac.create_decode_backend
    orig_extend = fac.create_draft_extend_backend

    def _backend(self, name):
        return self.draft_attn_backend or getattr(self.server_args, name) or self.server_args.attention_backend

    def create_decode_backend(self):
        if self.speculative_num_steps > 1 and _backend(self, "decode_attention_backend") == "sunrise":
            from ...impl.attention.attention_backend import (
                SunriseMultiStepDraftBackend,
            )

            return SunriseMultiStepDraftBackend(
                self.draft_model_runner, self.topk, self.speculative_num_steps
            )
        return orig_decode(self)

    def create_draft_extend_backend(self):
        name = (
            "decode_attention_backend"
            if self.server_args.speculative_attention_mode == "decode"
            else "prefill_attention_backend"
        )
        if _backend(self, name) == "sunrise":
            from ...impl.attention.attention_backend import SunriseAttnBackend

            return SunriseAttnBackend(self.draft_model_runner)
        return orig_extend(self)

    fac.create_decode_backend = create_decode_backend
    fac.create_draft_extend_backend = create_draft_extend_backend

    from sglang.srt.speculative import eagle_worker_v2

    worker_cls = eagle_worker_v2.EagleDraftWorker
    orig_graphs = worker_cls.init_cuda_graphs

    def init_cuda_graphs(self):
        # Upstream only knows CUDA / MUSA / NPU runners (KeyError for ptpu when
        # speculative_num_steps > 1); on PTPU both graphs are captured here.
        if torch.device(self.device).type != "ptpu":
            return orig_graphs(self)
        self.cuda_graph_runner = None
        self.cuda_graph_runner_for_draft_extend = None
        if self.server_args.disable_cuda_graph:
            return
        from ...impl.attention.attention_backend import (
            SunriseAttnBackend,
            SunriseMultiStepDraftBackend,
        )
        from ...impl.runtime.graph_runner import (
            ptpu_draft_extend_graph_runner_cls,
            ptpu_draft_graph_runner_cls,
        )

        if (
            self.speculative_num_steps > 1
            and isinstance(self.draft_attn_backend, SunriseMultiStepDraftBackend)
            and os.environ.get("SGLANG_FL_SUNRISE_SPEC_DRAFT_GRAPH", "1") != "0"
        ):
            self.cuda_graph_runner = ptpu_draft_graph_runner_cls()(self)
            logger.info("sunrise: captured the EAGLE multi-step draft graph")
        if (
            isinstance(self.draft_extend_attn_backend, SunriseAttnBackend)
            and os.environ.get("SGLANG_FL_SUNRISE_SPEC_DRAFT_EXTEND_GRAPH", "1") != "0"
        ):
            self.cuda_graph_runner_for_draft_extend = ptpu_draft_extend_graph_runner_cls()(self)
            logger.info("sunrise: captured the EAGLE draft-extend graph")

    worker_cls.init_cuda_graphs = init_cuda_graphs


def _patch_compiled_helpers() -> None:
    import sglang.srt.speculative.spec_utils  # noqa: F401  (later importers see the eager versions)

    names = ("select_top_k_tokens", "get_src_tgt_cache_loc", "create_num_accepted_drafts_filter",
             "get_last_loc_large_page_size_top_k_1")
    mods = [m for n, m in list(sys.modules.items()) if n.startswith("sglang.srt.speculative") and m is not None]
    for mod in mods:
        for name in names:
            orig = getattr(getattr(mod, name, None), "_torchdynamo_orig_callable", None)
            if orig is not None:
                setattr(mod, name, orig)


def _patch_draft_extend_post() -> None:
    """Draft extend after verify: the draft-extend graph already computes the top-k of
    every token; gather the accepted rows instead of re-gathering the full-vocab logits
    and recomputing softmax + top-k eagerly (~2 ms per step at 24 requests)."""
    from sglang.srt.speculative import eagle_worker_v2
    from sglang.srt.speculative.eagle_info import EagleDraftInput
    from sglang.srt.utils.common import fast_topk

    cls = eagle_worker_v2.EagleDraftWorker
    orig = cls._draft_extend_for_decode

    def _draft_extend_for_decode(self, batch, batch_result):
        runner = self.cuda_graph_runner_for_draft_extend
        if torch.device(self.device).type != "ptpu" or runner is None or batch.forward_mode.is_idle():
            return orig(self, batch, batch_result)
        draft_input = EagleDraftInput(
            hidden_states=batch_result.logits_output.hidden_states,
            num_tokens_per_req=self.speculative_num_steps + 1,
            num_tokens_for_logprob_per_req=self.speculative_num_steps + 1,
        )
        select_index = (
            torch.arange(len(batch.seq_lens), device=self.device) * self.speculative_num_draft_tokens
            + batch_result.accept_lens - 1
        )
        with self.plan_stream_ctx:
            forward_batch = draft_input.prepare_for_extend_to_fill_draft_kvcache(
                batch, batch_result.next_token_ids, self.speculative_num_draft_tokens, self.draft_runner, runner,
            )
        if self.plan_stream:
            torch.get_device_module(self.device).current_stream().wait_stream(self.plan_stream)
        if forward_batch.spec_info.num_accepted_drafts is None:
            forward_batch.spec_info.num_accepted_drafts = batch_result.accept_lens - 1
            forward_batch.spec_info.num_accepted_tokens = batch_result.accept_lens
        if runner.can_run(forward_batch):
            out = runner.replay(forward_batch)
            topk_p, topk_index = out.topk_p[select_index], out.topk_index[select_index]
        else:  # metadata was initialized by prepare_for_extend_to_fill_draft_kvcache
            out = self.draft_runner.forward(forward_batch, skip_attn_backend_init=True).logits_output
            probs = torch.softmax(out.next_token_logits[select_index], dim=-1)
            topk_p, topk_index = fast_topk(probs, self.topk, dim=-1)
        nxt = batch_result.next_draft_input
        nxt.topk_p, nxt.topk_index, nxt.hidden_states = topk_p, topk_index, out.hidden_states[select_index]

    cls._draft_extend_for_decode = _draft_extend_for_decode


def _patch_record_stream() -> None:
    from collections import deque
    from threading import Lock

    # PTPU has no allocator-level record_stream support. Hold a tensor only
    # until an event recorded on the consumer stream completes; retaining the
    # last N tensors indefinitely leaks large eager-prefill activations.
    pending = deque()
    lock = Lock()
    max_pending = 512
    orig = torch.Tensor.record_stream

    def record_stream(self, stream):
        if self.device.type == "ptpu":
            event = torch.ptpu.Event()
            event.record(stream)
            with lock:
                if pending:
                    live = deque()
                    while pending:
                        tensor, old_event = pending.popleft()
                        if not old_event.query():
                            live.append((tensor, old_event))
                    pending.extend(live)
                pending.append((self, event))
                if len(pending) > max_pending:
                    _, oldest_event = pending.popleft()
                    oldest_event.synchronize()
            return
        return orig(self, stream)

    torch.Tensor.record_stream = record_stream


def _patch_mtp_model() -> None:
    from sglang.srt.models import qwen3_5_mtp
    from sglang.srt.models.qwen3_5 import Qwen3_5ForCausalLM

    cls = qwen3_5_mtp.Qwen3_5ForCausalLMMTP
    # The compressed-tensors config targets the checkpoint's unfused
    # q/k/v_proj and gate/up_proj names, while the MTP runtime constructs
    # qkv_proj and gate_up_proj. The target model publishes this mapping, but
    # the draft wrapper does not, so model-loader config construction cannot
    # resolve the fused MTP modules.
    cls.packed_modules_mapping = Qwen3_5ForCausalLM.packed_modules_mapping
    orig = cls.set_embed_and_head

    def set_embed_and_head(self, embed, head):
        if not _on_ptpu(embed):
            return orig(self, embed, head)
        del self.model.embed_tokens.weight
        if not self.config.tie_word_embeddings:
            del self.lm_head.weight
        self.model.embed_tokens.weight = embed
        self.lm_head.weight = head
        torch.ptpu.synchronize()

    cls.set_embed_and_head = set_embed_and_head


def patch_spec_decode_for_sunrise() -> None:
    global _patched
    if _patched or os.environ.get("SGLANG_FL_SUNRISE_SPEC", "1") == "0":
        return
    _patched = True
    for fn in (_patch_mamba_pool, _patch_gdn_verify, _patch_tree_ops, _patch_eagle_info_v2,
               _patch_draft_backend_factory, _patch_mtp_model, _patch_record_stream,
               _patch_compiled_helpers, _patch_draft_extend_post):
        try:
            fn()
        except Exception as exc:  # keeps non-speculative serving working
            logger.warning("sunrise: speculative-decoding patch %s failed: %s", fn.__name__, exc)
    logger.info("sunrise: MTP / EAGLE chain speculative decoding patches applied")
