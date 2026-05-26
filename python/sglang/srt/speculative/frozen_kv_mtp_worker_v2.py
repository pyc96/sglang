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
"""
FrozenKVMTPWorkerV2 — overlap-scheduling (spec v2) worker for FROZEN_KV_MTP.

Sibling of ``EagleWorkerV2``, ported to FROZEN_KV_MTP semantics.

Critical FROZEN_KV_MTP differences vs vanilla EAGLE:
  1. Draft reads target's frozen KV cache via
     ``draft_attn_backend.token_to_kv_pool`` swap. Every draft forward
     is wrapped in ``_target_kv_pool_view``.
  2. Assistant draft hidden_size differs from target (1024 vs 2816 on
     Gemma-4-26B-A4B-IT). ``hidden_states`` on EagleDraftInput is sized
     by ``backbone_hidden_size``.
  3. Draft has no KV pool. ``_draft_extend_for_decode`` does NOT
     pre-allocate slots from the draft side.
  4. RoPE positions clamped to ``seq_lens - 1`` via
     ``set_frozen_kv_positions`` because the draft reads target KV at
     the last written slot.

Why sibling (not wrap) — recap of PR #24:
  v1 ``EagleVerifyInput.verify`` performs per-req scheduler bookkeeping
  (``req.kv_committed_len``, ``req.output_ids.append``,
  ``req.update_finish_state``, ``req.spec_verify_ct``). The spec-v2
  ``ScheduleBatchResultProcessor._resolve_spec_overlap_tokens`` performs
  THE SAME bookkeeping. Running both -> double-counted ->
  KV pool leak within two decodes. EAGLE V2 sidesteps this by calling
  ``EagleVerifyInput.sample`` (no per-req loop) + ``fill_bonus_tokens``
  Triton kernel. We do the same.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

from sglang.srt.layers.moe.utils import (
    speculative_moe_a2a_backend_context,
    speculative_moe_backend_context,
)
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
)
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.base_spec_worker import BaseDraftWorker, BaseSpecWorker
from sglang.srt.speculative.eagle_info import EagleDraftInput
from sglang.srt.speculative.eagle_info_v2 import fill_bonus_tokens
from sglang.srt.speculative.frozen_kv_mtp_info import (
    FrozenKVMTPDraftInput,
    FrozenKVMTPVerifyInput,
)
from sglang.srt.speculative.frozen_kv_mtp_worker import FrozenKVMTPWorker
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

logger = logging.getLogger(__name__)


class FrozenKVMTPDraftWorker(BaseDraftWorker):
    """Adapter exposing the BaseDraftWorker contract over the v1 worker.

    ``kv_cache_builder.get_draft_kv_pool`` reads
    ``draft_worker.draft_worker.draft_runner.{token_to_kv_pool,
    model_config}`` for spec-v2. For FROZEN_KV_MTP the draft borrows
    the target's allocator, so these fields point at the target's KV
    pool — which is what we want (the cache builder treats the
    draft and target pools uniformly).
    """

    def __init__(self, v1_worker: FrozenKVMTPWorker):
        object.__setattr__(self, "_v1", v1_worker)

    @property
    def draft_runner(self):
        return self._v1.draft_model_runner

    @property
    def draft_attn_backend(self):
        return self._v1.draft_attn_backend

    @property
    def device(self):
        return self._v1.device

    def draft(self, batch: ScheduleBatch):
        return self._v1.draft(batch)

    def draft_extend(self):
        # FROZEN_KV_MTP has no separate draft_extend (the seed step is
        # folded into _draft_extend_for_decode on the spec worker).
        # Satisfy the abstract method as a no-op.
        return None

    def __getattr__(self, name):
        v1 = self.__dict__.get("_v1")
        if v1 is None:
            raise AttributeError(name)
        return getattr(v1, name)


class FrozenKVMTPWorkerV2(BaseSpecWorker):
    """Spec-v2 worker for FROZEN_KV_MTP.

    Forward contract (mirrors EagleWorkerV2):

        forward_batch_generation(batch, on_publish=None)
        -> GenerationBatchResult

    Extend (prefill) path:
        1. Target prefill (target_worker.forward_batch_generation).
        2. on_publish(seq_lens) — fence; next-iter schedule starts.
        3. Draft seed via v1's forward_draft_extend; install fresh
           FrozenKVMTPDraftInput on batch.spec_info.

    Decode path:
        1. Draft (v1.draft) — builds FrozenKVMTPVerifyInput.
        2. v2 verify — our own path. Does NOT call v1's per-req
           bookkeeping loop.
        3. on_publish(batch.seq_lens + accept_lens) — fence after verify.
        4. _draft_extend_for_decode — seed step for next iter.
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
        # Reuse v1 init — load draft, bind kv_context, build draft attn
        # backend + cuda graph runner. We override the forward path only.
        self._v1 = FrozenKVMTPWorker(
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
        self._target_worker = target_worker
        self._draft_worker = FrozenKVMTPDraftWorker(self._v1)

        self.server_args = server_args
        self.device = server_args.device
        self.speculative_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )
        self.topk = server_args.speculative_eagle_topk
        self.speculative_num_steps = server_args.speculative_num_steps
        self.speculative_num_draft_tokens = server_args.speculative_num_draft_tokens
        self.req_to_token_pool = self._v1.req_to_token_pool

        logger.info(
            "FrozenKVMTPWorkerV2 initialized (spec-v2 worker over the v1 "
            "FrozenKVMTPWorker; draft reads target's frozen KV pool)."
        )

    # ---- BaseSpecWorker contract ----------------------------------------- #

    @property
    def target_worker(self) -> TpModelWorker:
        return self._target_worker

    @property
    def draft_worker(self) -> BaseDraftWorker:
        return self._draft_worker

    def clear_cache_pool(self):
        return self._v1.clear_cache_pool()

    def __getattr__(self, name):
        v1 = self.__dict__.get("_v1")
        if v1 is None:
            raise AttributeError(name)
        return getattr(v1, name)

    # ---- forward_batch_generation ---------------------------------------- #

    def forward_batch_generation(
        self, batch: ScheduleBatch, on_publish=None
    ) -> GenerationBatchResult:
        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            return self._forward_extend(batch, on_publish=on_publish)
        return self._forward_decode(batch, on_publish=on_publish)

    def _forward_extend(self, batch: ScheduleBatch, on_publish=None):
        (
            logits_output,
            next_token_ids,
            seq_lens_cpu,
            can_run_cuda_graph,
        ) = self._v1.forward_target_extend(batch)

        if on_publish is not None:
            on_publish(batch.seq_lens)

        with (
            self._v1.draft_tp_context(self._v1.draft_model_runner.tp_group),
            speculative_moe_backend_context(),
            speculative_moe_a2a_backend_context(),
        ):
            self._v1.forward_draft_extend(
                batch,
                logits_output.hidden_states,
                next_token_ids,
                seq_lens_cpu,
                logits_output.mm_input_embeds,
            )

        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=next_token_ids,
            num_correct_drafts=0,
            can_run_cuda_graph=can_run_cuda_graph,
            next_draft_input=self._coerce_draft_input(batch.spec_info),
        )

    def _forward_decode(self, batch: ScheduleBatch, on_publish=None):
        # 1) Draft — build FrozenKVMTPVerifyInput.
        with (
            self._v1.draft_tp_context(self._v1.draft_model_runner.tp_group),
            speculative_moe_backend_context(),
            speculative_moe_a2a_backend_context(),
        ):
            verify_input = self._v1.draft(batch)
        batch.spec_info = verify_input

        # 2-4) Verify (no v1 bookkeeping), publish fence, seed.
        return self._verify_v2(batch, verify_input, on_publish=on_publish)

    # ---- verify (custom, no per-req bookkeeping) ------------------------- #

    def _verify_v2(
        self,
        batch: ScheduleBatch,
        verify_input: FrozenKVMTPVerifyInput,
        on_publish=None,
    ) -> GenerationBatchResult:
        device = self.device
        bs = batch.batch_size()

        verify_input.num_tokens_per_req = self.speculative_num_steps + 1
        verify_forward_batch, can_run_cuda_graph = verify_input.prepare_for_v2_verify(
            self.req_to_token_pool, batch, self._target_worker
        )

        forward_batch_output = self._target_worker.forward_batch_generation(
            batch=None,
            forward_batch=verify_forward_batch,
            is_verify=True,
            skip_attn_backend_init=True,
        )
        logits_output = forward_batch_output.logits_output

        # Grammar mask (rare). EAGLE V2's lines 1047-1063.
        vocab_mask = None
        if batch.has_grammar:
            from sglang.srt.constrained.grammar_utils import generate_token_bitmask

            retrieve_next_token_cpu = verify_input.retrieve_next_token.cpu()
            retrieve_next_sibling_cpu = verify_input.retrieve_next_sibling.cpu()
            draft_tokens_cpu = verify_input.draft_token.view(
                verify_input.retrieve_next_token.shape
            ).cpu()
            vocab_mask = generate_token_bitmask(
                batch.reqs,
                verify_input,
                retrieve_next_token_cpu,
                retrieve_next_sibling_cpu,
                draft_tokens_cpu,
                batch.sampling_info.vocab_size,
            )
            if vocab_mask is not None:
                assert verify_input.grammar is not None
                vocab_mask = vocab_mask.to(verify_input.retrieve_next_token.device)
                batch.sampling_info.vocab_mask = None

        # Sample (v2 sample(); no per-req bookkeeping).
        predict, accept_lens, accept_index = verify_input.sample(
            batch, logits_output, vocab_mask
        )

        # EOS truncation. v1's EagleVerifyInput.verify per-req loop walks
        # accept_index_row and, on the first token that matches an EOS /
        # stop-token / max-new-tokens cap, sets all subsequent positions
        # of accept_index[i, :] to -1 and shrinks the per-req accept count
        # accordingly. v2's sample() does NOT do this — without the
        # truncation, finished requests commit post-EOS tokens, and the
        # MM color test drops from 30/30 to 28-29/30.
        #
        # We do the same scan here purely-functionally (no req state
        # mutation): the scheduler's `process_batch_result_decode` still
        # owns `update_finish_state`, `output_ids.extend`, grammar
        # tracking, etc. — it just sees the right (truncated)
        # `accept_lens[i]` slice.
        if not batch.forward_mode.is_idle():
            self._truncate_at_eos_inplace(batch, predict, accept_lens, accept_index)

        new_seq_lens = batch.seq_lens + accept_lens

        # Publish fence — after verify, before seed.
        if on_publish is not None:
            on_publish(new_seq_lens)

        # Per-bs bonus_tokens for the scheduler's stash.
        if not batch.forward_mode.is_idle():
            accept_tokens = predict[accept_index]
            bonus_tokens = torch.empty_like(accept_lens, dtype=torch.int32)
            fill_bonus_tokens[(bs,)](
                accept_tokens,
                accept_lens,
                bonus_tokens,
                self.speculative_num_draft_tokens,
            )
        else:
            bonus_tokens = torch.empty((0,), device=device, dtype=torch.int32)

        # Skeleton next_draft_input (populated by _draft_extend_for_decode).
        next_draft_input = FrozenKVMTPDraftInput(
            bonus_tokens=bonus_tokens,
            new_seq_lens=new_seq_lens,
            num_tokens_per_req=self.speculative_num_steps + 1,
            num_tokens_for_logprob_per_req=self.speculative_num_steps + 1,
            capture_hidden_mode=CaptureHiddenMode.LAST,
        )

        # Seed step — produces next-iter topk_p / topk_index / hidden_states.
        self._draft_extend_for_decode(
            batch=batch,
            predict=predict,
            accept_lens=accept_lens,
            accept_index=accept_index,
            next_draft_input=next_draft_input,
            target_hidden_states=logits_output.hidden_states,
        )

        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=predict,
            can_run_cuda_graph=can_run_cuda_graph,
            speculative_num_draft_tokens=self.speculative_num_draft_tokens,
            next_draft_input=next_draft_input,
            accept_lens=accept_lens,
            extra_keep_alive_refs=[verify_forward_batch],
        )

    # ---- seed step (next-iter draft prep) -------------------------------- #

    def _draft_extend_for_decode(
        self,
        batch: ScheduleBatch,
        predict: torch.Tensor,
        accept_lens: torch.Tensor,
        accept_index: torch.Tensor,
        next_draft_input: FrozenKVMTPDraftInput,
        target_hidden_states: torch.Tensor,
    ) -> None:
        """Run the assistant on the verified last-accept token to produce
        next-iter topk_p / topk_index / hidden_states.

        FROZEN_KV_MTP-flavored sibling of
        ``EagleDraftWorker._draft_extend_for_decode`` — two key
        differences:
          * Wrapped (transitively, via v1's helper) in
            ``_target_kv_pool_view`` so the assistant reads target KV.
          * Does NOT call ``prepare_for_extend_to_fill_draft_kvcache``
            (EAGLE V2's per-draft KV slot allocator). The draft has no
            KV pool — the target's KV already covers it.
        """
        bs = batch.batch_size()
        if bs == 0 or batch.forward_mode.is_idle():
            self._fill_next_draft_input_with_zeros(next_draft_input, bs=bs)
            return

        # Extract per-req last accepted token + last hidden state.
        # `predict` is shape [bs * speculative_num_draft_tokens] flat;
        # `accept_index` is shape [bs, spec_steps + 1] with -1 padding.
        # For each req i, `accept_lens[i]` is the count incl. bonus, so
        # the last accept's column index is `accept_lens[i] - 1`.
        col_idx = (accept_lens.to(torch.int64) - 1).clamp(min=0)
        row_idx = torch.arange(bs, device=self.device, dtype=torch.int64)
        ai = accept_index.to(torch.int64)
        last_accept_flat_idx = ai[row_idx, col_idx].clamp(min=0)
        last_token_ids = predict[last_accept_flat_idx]
        last_hidden = (
            target_hidden_states[last_accept_flat_idx]
            if target_hidden_states is not None and target_hidden_states.shape[0] > 0
            else torch.zeros(
                (bs, self._v1._recurrent_hidden_size),
                device=self.device,
                dtype=self._v1.model_config.dtype,
            )
        )

        # Run the assistant seed step (the v1 helper does the kv_context
        # swap, attn metadata init, and installs a fresh
        # FrozenKVMTPDraftInput on batch.spec_info with the new
        # topk_p / topk_index / hidden_states fields populated).
        with (
            self._v1.draft_tp_context(self._v1.draft_model_runner.tp_group),
            speculative_moe_backend_context(),
            speculative_moe_a2a_backend_context(),
        ):
            self._v1._run_assistant_seed_step(
                batch=batch,
                last_token_ids=last_token_ids,
                last_hidden_states=last_hidden,
                seq_lens_cpu=batch.seq_lens_cpu,
                mm_input_embeds=None,
                draft_input=None,
            )

        # Mirror the v1 seed's outputs onto the v2 next_draft_input.
        seeded: FrozenKVMTPDraftInput = batch.spec_info
        self._copy_draft_fields_to_next_input(seeded, next_draft_input, bs)

    # ---- helpers --------------------------------------------------------- #

    def _copy_draft_fields_to_next_input(
        self,
        src: FrozenKVMTPDraftInput,
        dst: FrozenKVMTPDraftInput,
        bs: int,
    ) -> None:
        topk = self.topk
        recurrent_hidden = self._v1._recurrent_hidden_size
        target_dtype = self._v1.model_config.dtype

        def _pad(t, shape_tail, dtype):
            if t is None or t.shape[0] == 0:
                return torch.zeros((bs,) + shape_tail, device=self.device, dtype=dtype)
            if t.shape[0] == bs:
                return t
            pad = torch.zeros(
                (bs - t.shape[0],) + shape_tail, device=t.device, dtype=t.dtype
            )
            return torch.cat([t, pad], dim=0)

        dst.topk_p = _pad(getattr(src, "topk_p", None), (topk,), torch.float32)
        dst.topk_index = _pad(getattr(src, "topk_index", None), (topk,), torch.int64)
        dst.hidden_states = _pad(
            getattr(src, "hidden_states", None),
            (recurrent_hidden,),
            target_dtype,
        )

    def _fill_next_draft_input_with_zeros(
        self, dst: FrozenKVMTPDraftInput, bs: int
    ) -> None:
        topk = self.topk
        recurrent_hidden = self._v1._recurrent_hidden_size
        target_dtype = self._v1.model_config.dtype
        dst.topk_p = torch.zeros((bs, topk), device=self.device, dtype=torch.float32)
        dst.topk_index = torch.zeros((bs, topk), device=self.device, dtype=torch.int64)
        dst.hidden_states = torch.zeros(
            (bs, recurrent_hidden), device=self.device, dtype=target_dtype
        )

    def _truncate_at_eos_inplace(
        self,
        batch: ScheduleBatch,
        predict: torch.Tensor,
        accept_lens: torch.Tensor,
        accept_index: torch.Tensor,
    ) -> None:
        """Mirror v1 ``EagleVerifyInput.verify``'s per-req EOS truncation
        in pure-functional form. Mutates ``accept_lens`` and
        ``accept_index`` in place so:

          * ``accept_index[i, j+1:] = -1`` past the first EOS-matching
            (or max-new-tokens-overflowing) accepted token,
          * ``accept_lens[i]`` = truncated count incl. the EOS token
            itself,

        without touching any ``req`` state (``output_ids``,
        ``finished_reason``, ``grammar``, ``reasoning_tokens``, ...).
        The scheduler's ``process_batch_result_decode`` performs those
        mutations once it sees the corrected per-req slice.

        Why purely functional:
          v1 mutates ``req`` state inside the verify loop because the v1
          batch_result_processor doesn't repeat that work. v2's
          batch_result_processor DOES repeat it (lines 607-623 of
          batch_result_processor.py). Calling ``update_finish_state`` etc.
          twice causes:
            * grammar.accept_token() double-accepts (corrupts grammar).
            * Reasoning token counter doubled.
            * Two FINISH_MATCHED_TOKEN with different ``finished_len``.

        Grammar limitation (documented):
          v1's loop checks grammar termination after each accepted token
          and stops the spec if the grammar terminates mid-accept. v2's
          purely-functional truncation does NOT do this — grammar
          termination is detected downstream by the scheduler after the
          full ``accept_lens[i]`` slice has been ``output_ids.extend``-ed.
          For grammar-using requests this can cause one extra committed
          token past the grammar terminator. Acceptable for now (the MM
          color test uses no grammar). Tracked as a follow-up.
        """
        # Per-req CPU walk. ``predict`` and ``accept_index`` live on the
        # device; we copy small per-row slices on demand. The cost is one
        # GPU->CPU sync (the ``.tolist()``) per call, which is what v1
        # already pays.
        bs = len(batch.reqs)
        if bs == 0:
            return

        accept_index_cpu = accept_index.tolist()  # [[ints]] of shape [bs, spec_steps+1]
        predict_cpu = predict.tolist()
        new_accept_lens_cpu = accept_lens.tolist()
        mutated_index = False

        for i, (req, accept_index_row) in enumerate(zip(batch.reqs, accept_index_cpu)):
            if req.sampling_params.ignore_eos:
                continue
            # Walk row, count tokens up to (and including) the first EOS.
            new_count = 0
            for j, idx in enumerate(accept_index_row):
                if idx == -1:
                    break
                new_count += 1
                tok_id = predict_cpu[idx]
                # max_new_tokens cap: if appending this token would push
                # ``len(output_ids)`` past the cap, the cap-finish fires
                # AT this token. Mirror v1's `update_finish_state` cap
                # check.
                cur_out_len = len(req.output_ids) + new_count
                cap = req.sampling_params.max_new_tokens
                if cap is not None and cur_out_len >= cap:
                    accept_index_row[j + 1 :] = [-1] * (len(accept_index_row) - j - 1)
                    mutated_index = True
                    break
                # EOS / stop-token check (pure; no req mutation).
                if _is_finish_token(req, tok_id):
                    accept_index_row[j + 1 :] = [-1] * (len(accept_index_row) - j - 1)
                    mutated_index = True
                    break
            new_accept_lens_cpu[i] = new_count

        # Push corrected accept_lens back to GPU (small per-bs copy).
        # Avoid the round trip when nothing changed.
        if mutated_index or new_accept_lens_cpu != accept_lens.tolist():
            # Build new accept_index from the mutated CPU rows.
            new_accept_index_cpu = accept_index_cpu  # mutated in place above
            accept_index.copy_(
                torch.tensor(
                    new_accept_index_cpu,
                    dtype=accept_index.dtype,
                    device=accept_index.device,
                )
            )
            accept_lens.copy_(
                torch.tensor(
                    new_accept_lens_cpu,
                    dtype=accept_lens.dtype,
                    device=accept_lens.device,
                )
            )

    def _coerce_draft_input(self, spec_info) -> Optional[EagleDraftInput]:
        if spec_info is None:
            return None
        if isinstance(spec_info, EagleDraftInput):
            return spec_info
        logger.warning(
            "FrozenKVMTPWorkerV2: unexpected spec_info type %s; returning None.",
            type(spec_info).__name__,
        )
        return None


def _is_finish_token(req, token_id: int) -> bool:
    """Pure-functional EOS / stop-token check. Mirrors the predicate
    inside ``Req._check_token_based_finish`` (schedule_batch.py:1167)
    WITHOUT mutating ``req.finished_reason`` / ``finished_len`` — the
    scheduler's ``process_batch_result_decode`` does that downstream
    after the corrected accept_lens slice reaches ``update_finish_state``.
    """
    if (
        req.sampling_params.stop_token_ids
        and token_id in req.sampling_params.stop_token_ids
    ):
        return True
    if req.eos_token_ids and token_id in req.eos_token_ids:
        return True
    tok = getattr(req, "tokenizer", None)
    if tok is not None:
        if token_id == tok.eos_token_id:
            return True
        extra = getattr(tok, "additional_stop_token_ids", None)
        if extra and token_id in extra:
            return True
    return False
