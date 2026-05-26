# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Spec-V2 (overlap-scheduling) worker for Frozen-KV MTP.

Inheritance-based wrapper around ``FrozenKVMTPWorker`` that exposes the
``BaseSpecWorker`` interface so the scheduler's ``event_loop_overlap``
can run with the existing V1 correctness-tested logic.

The dominant overlap win comes from letting iter N+1's CPU work
(``resolve_seq_lens_cpu``, sampling-info prep, request-pool updates,
``get_next_batch_to_run`` for iter N+2) run on the schedule stream
concurrently with iter N's ``draft + verify + seed`` chain on the
forward stream.

Trade-off vs the full ``EAGLEWorkerV2`` rewrite: this MVP keeps the
V1 worker body intact and adds only the V2 interface.  The per-step
draft loop still runs as one block; we do not split it into a separate
``BaseDraftWorker``.  Trades some additional overlap potential for a
much smaller code surface and zero risk of breaking the frozen-KV
invariants V1 already gets right.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.base_spec_worker import BaseSpecWorker
from sglang.srt.speculative.frozen_kv_mtp_worker import FrozenKVMTPWorker

logger = logging.getLogger(__name__)


class FrozenKVMTPWorkerV2(FrozenKVMTPWorker, BaseSpecWorker):
    """Spec-V2 worker for Frozen-KV MTP -- inherits V1, adds the V2 hooks.

    Adds:
      * ``BaseSpecWorker`` interface (``target_worker`` / ``draft_worker`` /
        ``clear_cache_pool``).
      * ``forward_batch_generation(batch, on_publish=None)`` -- fires
        ``on_publish(seq_lens_after_verify)`` at the verify-end fence
        so the future map publishes new ``seq_lens`` before the seed
        step blocks the forward stream.
    """

    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: Optional[int],
        moe_ep_rank: int,
        attn_cp_rank: int,
        moe_dp_rank: int,
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        # The V1 ctor binds the frozen-KV context onto the draft model,
        # shares embed/head, builds the draft attn backend, captures
        # cuda graphs, and plumbs the shared KV pool.  We add only the
        # ``_target_worker`` ref for the BaseSpecWorker property.
        # V1 sets ``self.target_worker`` as an instance attribute at line
        # ``frozen_kv_mtp_worker.py:105``.  Cache the reference under a
        # private name first so ``__init__`` can complete even though
        # ``BaseSpecWorker`` declares ``target_worker`` as an abstract
        # property (it is overridden below to read ``_target_worker``).
        # The V1 ctor assigns ``self.target_worker = target_worker``
        # which the property override below routes back into our private
        # backing field.
        FrozenKVMTPWorker.__init__(
            self,
            server_args=server_args,
            gpu_id=gpu_id,
            tp_rank=tp_rank,
            dp_rank=dp_rank,
            moe_ep_rank=moe_ep_rank,
            attn_cp_rank=attn_cp_rank,
            moe_dp_rank=moe_dp_rank,
            nccl_port=nccl_port,
            target_worker=target_worker,
        )
        logger.info(
            "FrozenKVMTPWorkerV2 initialized (spec-V2 overlap-scheduling "
            "wrapper around FrozenKVMTPWorker)."
        )

    # ----- BaseSpecWorker interface ---------------------------------

    @property
    def target_worker(self) -> TpModelWorker:
        return self._target_worker

    @target_worker.setter
    def target_worker(self, value: TpModelWorker) -> None:
        # V1 ctor does ``self.target_worker = target_worker``; route that
        # into ``_target_worker`` so the abstract property contract is
        # satisfied for ``BaseSpecWorker``.
        self._target_worker = value

    @property
    def draft_worker(self):
        # V1 worker IS the draft worker (inherits TpModelWorker and holds
        # ``self.draft_model_runner`` directly).  Return ``self`` so the
        # scheduler's draft-side RPCs route to V1's TpModelWorker methods.
        return self

    @property
    def draft_runner(self):
        # Alias for the EAGLE-V2-style accessor ``draft_worker.draft_runner``
        # expected by ``mem_cache/kv_cache_builder.py:62``.  V1 calls the
        # same attribute ``draft_model_runner``.
        return self.draft_model_runner

    def clear_cache_pool(self):
        # Frozen-KV MTP shares the target's KV pool; the assistant's
        # own pool is a 64-slot dummy.  Match V1 behavior: no-op.
        pass

    # ----- spec-V2 forward entry ------------------------------------

    def forward_batch_generation(
        self,
        batch: ScheduleBatch,
        on_publish=None,
        **kwargs,
    ):
        """V2-compatible forward; fires ``on_publish`` at verify-end.

        For prefill batches (EXTEND / is_extend_in_batch), publishes
        before the draft seed runs.  For decode batches, publishes
        between ``verify`` and ``forward_draft_extend_after_decode``.
        """
        # Lazy imports to avoid heavy modules at file load.
        from sglang.srt.layers.moe.utils import (
            speculative_moe_a2a_backend_context,
            speculative_moe_backend_context,
        )
        from sglang.srt.managers.scheduler import GenerationBatchResult
        from sglang.srt.observability.req_time_stats import set_time_batch
        from sglang.srt.observability.trace import get_global_tracing_enabled

        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            (
                logits_output,
                next_token_ids,
                seq_lens_cpu,
                can_run_cuda_graph,
            ) = self.forward_target_extend(batch)

            # vLLM-style fence point: publish new seq_lens AFTER target
            # prefill writes its KV but BEFORE the draft seed runs.
            if on_publish is not None:
                on_publish(batch.seq_lens)

            from sglang.srt.model_executor.forward_batch_info import (
                CaptureHiddenMode,
            )
            from sglang.srt.speculative.frozen_kv_mtp_info import (
                FrozenKVMTPDraftInput,
            )

            with (
                self.draft_tp_context(self.draft_model_runner.tp_group),
                speculative_moe_backend_context(),
                speculative_moe_a2a_backend_context(),
            ):
                self.forward_draft_extend(
                    batch,
                    logits_output.hidden_states,
                    next_token_ids,
                    seq_lens_cpu,
                    logits_output.mm_input_embeds,
                )

            # Ferry the seed-step output as next_draft_input for the
            # spec-V2 future_map.stash.  Must size by batch.req_pool_indices
            # (= future_indices used by scheduler) so the stash kernel
            # doesn't shape-mismatch.
            next_draft_input_pf = batch.spec_info
            if not isinstance(next_draft_input_pf, FrozenKVMTPDraftInput):
                next_draft_input_pf = FrozenKVMTPDraftInput.create_idle_input(
                    device=self.device,
                    hidden_size=self._recurrent_hidden_size,
                    dtype=self.model_config.dtype,
                    topk=self.topk,
                    capture_hidden_mode=CaptureHiddenMode.LAST,
                )

            return GenerationBatchResult(
                logits_output=logits_output,
                next_token_ids=next_token_ids,
                num_correct_drafts=0,
                can_run_cuda_graph=can_run_cuda_graph,
                next_draft_input=next_draft_input_pf,
            )

        # Decode branch.  V1 ``draft()`` was patched to be None-safe on
        # ``batch.sampling_info.penalizer_orchestrator`` so the call works
        # under both V1 (orchestrator present) and V2 (cumulate already
        # done schedule-side, orchestrator may be None or stub).
        set_time_batch(batch.reqs, "set_spec_draft_start_time", trace_only=True)
        with (
            self.draft_tp_context(self.draft_model_runner.tp_group),
            speculative_moe_backend_context(),
            speculative_moe_a2a_backend_context(),
        ):
            verify_input = self.draft(batch)
        set_time_batch(batch.reqs, "set_spec_draft_end_time", trace_only=True)
        set_time_batch(batch.reqs, "set_spec_verify_start_time", trace_only=True)

        batch.spec_info = verify_input
        verify_output = self.verify(batch)

        # vLLM-style fence point: publish new seq_lens AFTER verify
        # writes accepted KV but BEFORE the seed (draft-extend) blocks
        # the forward stream.  This lets the scheduler stage iter N+1's
        # CPU prep against the now-known new seq_lens while iter N's
        # seed runs.
        if on_publish is not None:
            on_publish(batch.seq_lens)

        if get_global_tracing_enabled():
            for idx, req in enumerate(batch.reqs):
                num_correct_drafts = verify_output.num_correct_drafts_per_req_cpu[idx]
                req.time_stats.set_spec_verify_end_time(
                    num_correct_drafts=num_correct_drafts
                )

        from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
        from sglang.srt.speculative.frozen_kv_mtp_info import FrozenKVMTPDraftInput

        set_time_batch(batch.reqs, "set_spec_draft_extend_start_time", trace_only=True)
        with (
            self.draft_tp_context(self.draft_model_runner.tp_group),
            speculative_moe_backend_context(),
            speculative_moe_a2a_backend_context(),
        ):
            draft_extend_input = verify_output.draft_extend_input
            if (
                self.server_args.enable_dp_attention
                or draft_extend_input.input_ids.shape[0] > 0
            ):
                batch.spec_info = draft_extend_input
                self.forward_draft_extend_after_decode(batch)
        set_time_batch(batch.reqs, "set_spec_draft_extend_end_time", trace_only=True)

        # Resolve next_draft_input for the spec-V2 future_map.stash.
        # The future map indexes by ``batch.req_pool_indices`` (size B).
        # All payload fields must be sized [B, ...] -- not [total_accepted]
        # (which is the flat shape of ``verify_output.accept_tokens``).
        spec_info_after_seed = batch.spec_info
        bs = batch.req_pool_indices.shape[0]

        if (
            isinstance(spec_info_after_seed, FrozenKVMTPDraftInput)
            and spec_info_after_seed.bonus_tokens is not None
            and spec_info_after_seed.bonus_tokens.shape[0] == bs
        ):
            # Seed step ran for all reqs (no req finished mid-verify).
            # batch.spec_info already has full-batch-sized fields.
            next_draft_input = spec_info_after_seed
        else:
            # Either no seed ran (all reqs finished) or the seed
            # populated only unfinished rows.  Build a full-B
            # FrozenKVMTPDraftInput, copy seed values into the
            # unfinished rows, zero-pad the rest.
            unfinished_idx = None
            if (
                isinstance(spec_info_after_seed, FrozenKVMTPDraftInput)
                and spec_info_after_seed.bonus_tokens is not None
                and spec_info_after_seed.bonus_tokens.shape[0] > 0
                and draft_extend_input.req_pool_indices is not None
            ):
                # Recover the indices of the unfinished reqs by matching
                # ``draft_extend_input.req_pool_indices`` (the subset that
                # ran the seed) to ``batch.req_pool_indices`` (the full
                # batch).  This is the same mapping the verify path uses
                # to slice ``unfinished_index_device``.
                full_idx = batch.req_pool_indices
                unfinished_pool = draft_extend_input.req_pool_indices
                unfinished_idx = torch.where(
                    (full_idx.unsqueeze(1) == unfinished_pool.unsqueeze(0)).any(dim=1)
                )[0]

            hidden_size = self._recurrent_hidden_size
            dtype = self.model_config.dtype
            topk = self.topk

            # Build [B] bonus_tokens by picking the LAST accepted token
            # per req from the flat ``verify_output.accept_tokens`` (which
            # is sized [total_accepted_across_all_reqs]).  The number
            # accepted per req is in ``num_correct_drafts_per_req_cpu``
            # plus 1 for the always-accepted bonus.
            num_accept_per_req = torch.tensor(
                [n + 1 for n in verify_output.num_correct_drafts_per_req_cpu],
                device=self.device,
                dtype=torch.long,
            )
            per_req_last_idx = torch.cumsum(num_accept_per_req, dim=0) - 1
            full_bonus = verify_output.accept_tokens[per_req_last_idx].to(torch.int32)
            full_topk_p = torch.zeros(
                (bs, topk), device=self.device, dtype=torch.float32
            )
            full_topk_index = torch.zeros(
                (bs, topk), device=self.device, dtype=torch.int64
            )
            full_hidden = (
                torch.zeros((bs, hidden_size), device=self.device, dtype=dtype)
                if hidden_size is not None
                else None
            )

            if (
                unfinished_idx is not None
                and unfinished_idx.shape[0] > 0
                and isinstance(spec_info_after_seed, FrozenKVMTPDraftInput)
            ):
                if (
                    spec_info_after_seed.topk_p is not None
                    and spec_info_after_seed.topk_p.shape[0] == unfinished_idx.shape[0]
                ):
                    full_topk_p[unfinished_idx] = spec_info_after_seed.topk_p
                if (
                    spec_info_after_seed.topk_index is not None
                    and spec_info_after_seed.topk_index.shape[0]
                    == unfinished_idx.shape[0]
                ):
                    full_topk_index[unfinished_idx] = spec_info_after_seed.topk_index
                if (
                    full_hidden is not None
                    and spec_info_after_seed.hidden_states is not None
                    and spec_info_after_seed.hidden_states.shape[0]
                    == unfinished_idx.shape[0]
                ):
                    full_hidden[unfinished_idx] = spec_info_after_seed.hidden_states.to(
                        dtype
                    )

            next_draft_input = FrozenKVMTPDraftInput(
                bonus_tokens=full_bonus,
                hidden_states=full_hidden,
                topk_p=full_topk_p,
                topk_index=full_topk_index,
                capture_hidden_mode=CaptureHiddenMode.LAST,
                new_seq_lens=batch.seq_lens.to(torch.int32),
            )

        # accept_lens is required by the spec-V2 batch result processor's
        # _resolve_spec_overlap_tokens path.  It is per-req: total accepted
        # tokens including the bonus.  V1 returns num_correct_drafts_per_req
        # (without bonus); add 1 for the bonus.
        accept_lens_cpu = torch.tensor(
            [n + 1 for n in verify_output.num_correct_drafts_per_req_cpu],
            dtype=torch.int32,
            device="cpu",
        )

        return GenerationBatchResult(
            logits_output=verify_output.logits_output,
            next_token_ids=verify_output.accept_tokens,
            num_correct_drafts=sum(verify_output.num_correct_drafts_per_req_cpu),
            num_correct_drafts_per_req_cpu=verify_output.num_correct_drafts_per_req_cpu,
            can_run_cuda_graph=False,
            next_draft_input=next_draft_input,
            accept_lens=accept_lens_cpu,
            speculative_num_draft_tokens=self.speculative_num_draft_tokens,
        )
