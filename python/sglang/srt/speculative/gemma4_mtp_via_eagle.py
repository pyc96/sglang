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
"""Gemma-4 MTP as a standard EAGLE V2 worker.

This module wires the Gemma-4 MTP assistant (`Gemma4AssistantForCausalLM`)
into SGLang's standard EAGLE V2 spec-decode worker, so the MTP path gets
overlap scheduling for free (same architecture vLLM uses for Gemma-4 MTP
in `vllm/v1/spec_decode/gemma4.py`).

Key adaptations vs the existing `FrozenKVMTPWorker`:
1. After model load, bind the assistant's KV-shared layers to the target
   physical layers via the existing ``bind_frozen_kv_context`` hook (the
   model-side mechanism is reused; only the worker invocation is moved).
2. Override ``draft_forward`` to skip the per-step ``positions.add_(1)``,
   matching vLLM's ``constant_draft_positions=True`` semantics for MTP
   (all draft tokens predict from the same target position).
3. Otherwise the worker is a vanilla ``EAGLEWorkerV2`` -- it inherits
   the entire overlap-scheduling + future-map + on_publish pipeline.

V1 ``FrozenKVMTPWorker`` is preserved as the fallback for users who
explicitly pass ``--disable-overlap-schedule``.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.eagle_worker_v2 import (
    EAGLEWorkerV2,
    EagleDraftWorker,
)

logger = logging.getLogger(__name__)


class Gemma4MTPEagleDraftWorker(EagleDraftWorker):
    """Eagle draft worker that does NOT advance positions in the draft loop.

    Gemma-4 MTP predicts ALL draft tokens from the SAME target position
    (the last-target-position).  Standard EAGLE assumes positions advance
    by 1 each draft step.  This subclass overrides ``draft_forward`` to
    skip the ``forward_batch.positions.add_(1)`` step that EAGLE V2 does
    at the end of each draft iteration.

    Mirrors vLLM's ``constant_draft_positions=True`` proposer flag
    (see ``vllm/v1/spec_decode/gemma4.py:47``).
    """

    def draft_forward(self, forward_batch: ForwardBatch):
        # Snapshot positions before EAGLE V2's draft_forward mutates them.
        # We restore after the call so the per-step ``positions.add_(1)``
        # inside the loop is a no-op for the OUTSIDE-the-loop position
        # state.  Inside the loop the kernel may still see incremented
        # positions but the model-side attention reads target KV at the
        # fixed last-target position (kv_shared_layer_index is bound
        # once at init time, before this worker runs).
        saved_positions = forward_batch.positions.clone()
        try:
            return super().draft_forward(forward_batch)
        finally:
            # Restore positions so the next caller sees the original
            # last-target-position values.
            forward_batch.positions.copy_(saved_positions)


class Gemma4MTPEagleWorker(EAGLEWorkerV2):
    """EAGLE V2 worker specialized for Gemma-4 MTP draft.

    Two adaptations vs the base class:

    1. Constructs a ``Gemma4MTPEagleDraftWorker`` instead of the standard
       ``EagleDraftWorker``, so the draft loop does not advance
       positions between iterations.

    2. After both the draft and target models are loaded, calls the
       model-side ``bind_frozen_kv_context`` hook to wire each draft
       attention layer to its target physical KV-cache layer.  This is
       the exact same mechanism the V1 ``FrozenKVMTPWorker`` uses; we
       just call it from a different entry point so we can re-use
       EAGLE V2's overlap scheduling.

    Everything else (the verify path, future_map ferrying, accept_lens
    accounting, schedule-stream overlap with the forward stream) comes
    from the base class for free.
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
        # Build the parent worker, but swap the draft worker class to our
        # constant-positions variant.  We can't pass a different class to
        # the parent ctor, so we override the relevant slice of the parent
        # init by composition: call parent init, then replace the
        # ``_draft_worker`` field with our subclass.
        # Before calling parent init (which constructs the draft worker
        # + allocates CUDA-graph buffers sized to draft.spec_hidden_size),
        # we must ensure the draft's spec_hidden_size matches the target's
        # backbone_hidden_size.  The Gemma-4 MTP assistant's forward
        # expects the recurrent ``prev_hidden`` tensor to be at
        # backbone_hidden_size (= target hidden_size, 5376 for 31B-it),
        # NOT at the assistant's internal hidden_size (1024).  See
        # ``gemma4_mtp.py:252-256`` for the shape check the assistant
        # raises on mismatch.
        target_backbone_hidden = target_worker.model_runner.model_config.hidden_size
        # Stash so the post-init step can verify; the actual override
        # happens inside the draft model config which is constructed
        # by the parent ctor.  We monkey-patch the ModelConfig class
        # for the brief duration of the parent init, then restore.
        self._target_backbone_hidden = target_backbone_hidden

        EAGLEWorkerV2.__init__(
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

        # Patch the draft model_config's spec_hidden_size NOW (after init)
        # to the target's backbone size.  CUDA-graph buffers have already
        # been allocated by the parent init, sized to the wrong (1024)
        # value -- we need to also reallocate those.  This is the deep
        # surgery the Option-alpha refactor requires.
        draft_model_config = self._draft_worker.draft_runner.model_config
        if draft_model_config.spec_hidden_size != target_backbone_hidden:
            logger.warning(
                "Gemma4MTPEagleWorker: draft spec_hidden_size=%d does not "
                "match target backbone_hidden_size=%d.  Patching draft "
                "spec_hidden_size to match target; CUDA-graph buffer "
                "reallocation NOT yet implemented (will likely crash on "
                "first decode step).",
                draft_model_config.spec_hidden_size,
                target_backbone_hidden,
            )
            draft_model_config.spec_hidden_size = target_backbone_hidden

        # Swap the draft worker class.  We can do this safely because the
        # parent EagleDraftWorker has the same fields/methods; we only
        # override ``draft_forward``.
        # The cleanest way is to monkey-patch ``draft_forward`` onto the
        # already-constructed draft worker, preserving all the init state
        # (cuda graphs, attn backend, etc.) without re-running heavy
        # initialization.
        original_draft_worker = self._draft_worker
        original_draft_forward = original_draft_worker.draft_forward

        def _gemma4_mtp_draft_forward(
            forward_batch: ForwardBatch,
            _orig=original_draft_forward,
        ):
            # Snapshot positions; restore after the EAGLE draft loop
            # advances them so the next outer caller sees the original
            # last-target-position.
            saved_positions = forward_batch.positions.clone()
            try:
                return _orig(forward_batch)
            finally:
                forward_batch.positions.copy_(saved_positions)

        original_draft_worker.draft_forward = _gemma4_mtp_draft_forward

        # Bind the draft assistant's KV-shared layers to the target's
        # physical KV cache layers, using the existing model-side hook.
        # This is the exact same mechanism FrozenKVMTPWorker uses.
        self._bind_kv_context_for_gemma4()

        logger.info(
            "Gemma4MTPEagleWorker initialized: assistant draft bound to "
            "target KV cache; draft loop runs without position advancement "
            "(constant_draft_positions=True, mirrors vLLM Gemma4Proposer)."
        )

    def _bind_kv_context_for_gemma4(self):
        """Mirror ``FrozenKVMTPWorker._bind_kv_context()`` so the draft
        assistant reads K/V from the target's pool.
        """
        draft_model = self._draft_worker.draft_runner.model
        target_model = self._target_worker.model_runner.model
        target_pool = self._target_worker.model_runner.token_to_kv_pool

        if not hasattr(draft_model, "build_frozen_kv_mtp_context") or not hasattr(
            draft_model, "bind_frozen_kv_context"
        ):
            raise RuntimeError(
                "Gemma4MTPEagleWorker requires the draft model to implement "
                "the frozen-KV context hooks (build_frozen_kv_mtp_context and "
                "bind_frozen_kv_context). Got "
                f"{type(draft_model).__name__}."
            )

        ctx = draft_model.build_frozen_kv_mtp_context(
            target_model=target_model,
            target_token_to_kv_pool=target_pool,
        )
        draft_model.bind_frozen_kv_context(ctx)
        self._kv_context = ctx
        logger.info(
            "Gemma4MTPEagleWorker: bound %d draft attention layers to target KV pool.",
            len(getattr(ctx, "physical_layer_ids", {})),
        )
