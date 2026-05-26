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
FrozenKVMTPWorkerV2 — overlap-scheduling (spec v2) entrypoint for the
FROZEN_KV_MTP code path.

STATUS: scaffolding only. Phase 1 of the plan in
``runs/20260525_frozen_kv_mtp_v2_plan/PLAN.md`` was attempted using a
``FrozenKVMTPWorker``-wrapping subclass; the result boots and runs
forward through verify, but the v1 ``EagleVerifyInput.verify`` does
scheduler-level bookkeeping (``req.kv_committed_len`` /
``req.output_ids.append`` / ``req.update_finish_state`` /
``req.spec_verify_ct``) that the spec-v2
``ScheduleBatchResultProcessor._resolve_spec_overlap_tokens`` ALSO
performs after the fact, leading to:

  - Double-incremented ``kv_committed_len`` → KV pool leak
    (reproduced: "pool memory leak detected" after a single decode).
  - Wrong-shape ``next_token_ids`` returned to ``output_tokens_buf``
    indexing (v1 returns flat-concat ``accept_tokens[total_accepted]``;
    v2 expects ``[bs * speculative_num_draft_tokens]``).

Both fixes are surgical (and ports of well-understood EAGLE V2 helpers),
but the cleanest design is NOT a wrap — it's a sibling worker that runs
its own ``verify`` path (matching ``EagleWorkerV2.verify``) wrapped in
the frozen-KV ``target_kv_pool_view`` contextmanager. That requires:

  1. Lift ``EagleVerifyInput.sample`` + ``fill_bonus_tokens`` to be
     reusable from a FROZEN_KV_MTP-aware ``verify`` (already module-level).
  2. Skip v1's ``EagleVerifyInput.verify`` per-req bookkeeping loop;
     defer to the scheduler's batch_result_processor.
  3. Replace the assistant "seed" step with an in-graph next-iter draft
     prep matching ``EagleDraftWorker._draft_extend_for_decode``,
     wrapped in ``target_kv_pool_view``.
  4. Implement ``EagleDraftInputV2Mixin.prepare_for_decode`` overrides
     for FROZEN_KV_MTP so the draft doesn't double-allocate KV slots
     (the target's allocator IS the draft's allocator).

The composition approach this file originally took (wrapping the v1
worker) hits the bookkeeping conflict at every iteration. Sibling
re-implementation is the right design and is left as a separate PR
(estimated 1-2 weeks of work, blocking on the EAGLE V2 abstractions
above being refactored into reusable helpers).

When SGLANG_FROZEN_KV_MTP_V2=1 is set, this module is loaded and the
scaffolding below raises a clear NotImplementedError pointing at the
plan doc. Default behavior (env unset) is unchanged — the v1
FrozenKVMTPWorker is dispatched.
"""

from __future__ import annotations

from typing import Optional

from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.base_spec_worker import BaseDraftWorker, BaseSpecWorker


class FrozenKVMTPWorkerV2(BaseSpecWorker):
    """Spec-v2 entrypoint for FROZEN_KV_MTP. NOT yet implemented.

    The class exists so:
      1. ``SpeculativeAlgorithm.create_worker`` can dispatch to it when
         ``SGLANG_FROZEN_KV_MTP_V2=1``.
      2. The plan and architectural blockers are discoverable in-source
         (this module's docstring + this class's docstring).
      3. A future PR that lands the sibling-worker implementation has a
         well-defined entrypoint and BaseSpecWorker contract to fill in.
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
        raise NotImplementedError(
            "FrozenKVMTPWorkerV2 (overlap scheduling for Frozen-KV MTP) "
            "is not yet implemented end-to-end. See module docstring + "
            "runs/20260525_frozen_kv_mtp_v2_plan/PLAN.md for the design "
            "and the architectural blockers (v1 verify double-bookkeeping "
            "+ wrong-shape next_token_ids). Unset SGLANG_FROZEN_KV_MTP_V2 "
            "to use the v1 FrozenKVMTPWorker (no overlap scheduling)."
        )

    # The BaseSpecWorker abstract contract — properties added so static
    # analyzers don't complain. Never called because __init__ raises.

    @property
    def target_worker(self) -> TpModelWorker:  # pragma: no cover
        raise NotImplementedError

    @property
    def draft_worker(self) -> BaseDraftWorker:  # pragma: no cover
        raise NotImplementedError

    def clear_cache_pool(self):  # pragma: no cover
        raise NotImplementedError
