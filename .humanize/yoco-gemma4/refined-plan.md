# Port vLLM YOCO Fast-Prefill to SGLang for Gemma-4

## Goal Description

Add an opt-in fast-prefill path to SGLang's Gemma-4 model that, when a
served Gemma-4 checkpoint has `num_kv_shared_layers > 0` and the new
server arg `--kv-sharing-fast-prefill` is set, runs the last
`num_kv_shared_layers` decoder layers (the "cross-decoder") only on the
per-request last-extend-token rows during EXTEND-mode forwards, instead
of on every input token. This matches the optimization vLLM ships for
`Gemma4ForCausalLM` in `vllm/model_executor/models/gemma4.py:1190-1273`
and is expected to materially close the SGLang-vs-vLLM throughput and
TTFT gap on long-input prefill workloads (target: summarization 8000/1000
n=80 throughput gap to vLLM narrows by ≥ 30 %).

Quality must be preserved: per-prompt token output must be byte-identical
to the non-fast-prefill path for greedy sampling, and MMLU N=500 must
stay within ±1 pp of the pre-patch SGLang result (0.780).

The implementation is gated to Gemma-4 only, runs only with the triton
attention backend, runs only in the eager EXTEND path (PCG bypass for v0),
and is split into three stacked draft PRs on `pyc96/sglang`.

## Acceptance Criteria

- AC-1: A new server arg `--kv-sharing-fast-prefill` (default `False`)
  is plumbed end-to-end from the CLI through `ServerArgs` and
  `ModelConfig` to a `bool` attribute the model can read at forward time.
  - Positive Tests (expected to PASS):
    - `python -m sglang.launch_server --help` lists
      `--kv-sharing-fast-prefill` in its help text.
    - Server starts with `--kv-sharing-fast-prefill` and the server log
      includes a line confirming the flag is enabled (e.g.
      `"KV-sharing fast prefill enabled for <model_arch>"`).
    - Server starts without the flag and logs no such confirmation
      (default-off behavior preserved).
  - Negative Tests (expected to FAIL):
    - Server startup is rejected with a clear error if the flag is set
      and the served model is not Gemma-4 (or has
      `num_kv_shared_layers == 0`).
    - Server startup is rejected with a clear error if the flag is set
      and the attention backend is not `triton`.

- AC-2: `Gemma4TextModel.forward` contains a predicate-gated YOCO branch
  that runs only when all of the following are true: the flag is on, the
  model has `num_kv_shared_layers > 0`, `forward_mode.is_extend()` is
  true, the mode is not `TARGET_VERIFY`, `extend_seq_lens` is populated,
  the batch is not under `is_in_piecewise_cuda_graph()`, the model has
  no per-layer embedding (`hidden_size_per_layer_input == 0`), and no
  request in the batch requests input logprobs.
  - Positive Tests (expected to PASS):
    - With flag on, EXTEND-mode forward of a Gemma-4 checkpoint
      executes the YOCO branch (verifiable by a debug log or counter
      that the test reads).
    - With flag off, the same forward executes the existing layer loop
      verbatim (no YOCO).
    - With flag on but `forward_mode == DECODE`, the YOCO branch is
      bypassed.
    - With flag on but the batch is `TARGET_VERIFY`, the YOCO branch
      is bypassed.
  - Negative Tests (expected to FAIL):
    - With flag on and a request that has input-logprob start <
      extend_seq_len, the YOCO branch must be bypassed (taking the
      branch would produce wrong input-logprob results).
    - With flag on and the model has no KV-shared layers, the YOCO
      branch must be bypassed.

- AC-3: When the YOCO branch runs, it produces a final
  `hidden_states: [T, H]` tensor whose per-request last-extend-token row
  (`hidden_states[cumsum(extend_seq_lens) - 1]`) is bit-identical (or
  within bf16 numerical noise) to the same row computed by the
  non-YOCO path on the same input.
  - Positive Tests (expected to PASS):
    - A unit harness that runs `Gemma4TextModel.forward` twice on the
      same input (once with YOCO on, once with YOCO off) and asserts
      `torch.allclose(yoco_out[last_index], baseline_out[last_index],
      atol=5e-3, rtol=5e-3)` for bf16.
    - A 20-prompt end-to-end parity test that runs greedy sampling
      (temperature=0, max_tokens=64) twice (YOCO on, off) and asserts
      the generated token IDs are identical for each prompt.
  - Negative Tests (expected to FAIL):
    - A parity test where intentionally the wrong `last_index`
      construction is used (e.g. off-by-one) must fail.
    - A parity test where the attention metadata for the cross-decoder
      phase incorrectly uses the original `qo_indptr` (instead of the
      rebuilt decode-shaped one) must fail.

- AC-4: The attention metadata used by the cross-decoder phase is a
  decode-shaped triton `ForwardMetadata`: `qo_indptr = [0, 1, ..., B]`,
  `max_extend_len = 1`, `kv_indptr` and `kv_indices` cover the full
  prefix-plus-extend KV span per request. The original metadata is
  restored on exit from the YOCO branch.
  - Positive Tests (expected to PASS):
    - A unit test that inspects the attention metadata in effect during
      the cross-decoder phase and verifies the decode-shape invariants
      above.
    - A unit test that verifies the original metadata is restored after
      the YOCO branch returns (so subsequent code paths see the same
      `ForwardMetadata` instance / fields as if YOCO had not run).
  - Negative Tests (expected to FAIL):
    - A version that forgets to restore the metadata triggers a failing
      assertion when the next forward pass tries to use stale state.
    - A version that uses `qo_indptr = cumsum(extend_seq_lens)` (the
      front-half plan) for the cross-decoder phase fails the cross-
      decoder attention-output shape check.

- AC-5: KV cache integrity is preserved: front-half layers write their
  K/V into the pool exactly as today; back-half layers do not write
  (`save_kv_cache=False` continues to hold because
  `is_kv_shared_layer=True` for them).
  - Positive Tests (expected to PASS):
    - A unit test that snapshots `token_to_kv_pool` slot counts before
      and after a YOCO-enabled forward and verifies the write count
      equals (front-half layer count) × (token count) — i.e. zero
      writes from the back half.
    - A multi-step test that runs YOCO-prefill then a DECODE step and
      verifies the decoded tokens match the non-YOCO baseline (proves
      the KV pool state after YOCO is identical to baseline).
  - Negative Tests (expected to FAIL):
    - A version that mistakenly sets `save_kv_cache=True` for the
      back-half layers must fail the snapshot test (write count would
      include back-half writes).

- AC-6: Quality bar — MMLU N=500 (seed 0, temp 0) on
  `google/gemma-4-31B-it` with YOCO enabled is within ±1 pp of the
  same configuration with YOCO disabled.
  - Positive Tests (expected to PASS):
    - Two MMLU runs: SGLang with YOCO on, SGLang with YOCO off. Both
      complete N=500 with zero errors. The accuracy delta is ≤ 0.01
      (within ±1 pp).
  - Negative Tests (expected to FAIL):
    - If accuracy drops by > 1 pp, the patch is rejected.

- AC-7: Performance bar — on the fixed campaign workload
  (`google/gemma-4-31B-it`, TP=2, H100, triton, NEXTN spec config 3/4/1,
  warmup 2, seed 1), the summ 8000/1000 n=80 scenario achieves
  `output_throughput ≥ 492 tok/s` with YOCO enabled (closes ≥ 30 % of
  the gap between the current SGLang best 331 tok/s and vLLM's
  868 tok/s).
  - Positive Tests (expected to PASS):
    - Bench result file `result_*_yoco_on_summ_8000_1000_n80_*.jsonl`
      has `output_throughput >= 492.0`.
    - Bench result on chat 1000/1000 n=80 with YOCO enabled does not
      regress more than 1 % vs the current SGLang best 1499 tok/s
      (i.e. `output_throughput >= 1484`).
  - Negative Tests (expected to FAIL):
    - If summ throughput < 492 tok/s after the final implementation,
      the SOTA loop continues with another round of tuning before
      stopping.
    - If chat throughput regresses > 1 %, the patch is rejected and
      either reverted or fixed.

- AC-8: Three stacked draft PRs land in `pyc96/sglang` (never
  `sgl-project/sglang`, never submitted), each with its own benchmark
  comparison and quality-score table in the PR body.
  - Positive Tests (expected to PASS):
    - `gh pr list --repo pyc96/sglang --state open --search "yoco"`
      returns three drafts.
    - Each PR body contains a benchmark delta table and an MMLU score
      table.
    - Each PR is built on top of the prior one (PR-B base = PR-A head,
      PR-C base = PR-B head).
  - Negative Tests (expected to FAIL):
    - Any PR submitted (non-draft) is a contract violation.
    - Any PR opened against `sgl-project/sglang` is a contract violation.
    - PR with empty body or missing the bench/MMLU table is rejected
      from "done".

## Path Boundaries

### Upper Bound (Maximum Acceptable Scope)

The implementation:
- Adds the `--kv-sharing-fast-prefill` server arg with all eligibility
  checks (Gemma-4 architecture allow-list, triton attention backend
  required, mutual-exclusion with EAGLE/EAGLE3 speculative algorithms,
  rejection of incompatible combos at startup).
- Implements the YOCO branch inside `Gemma4TextModel.forward` with full
  predicate gating, triton-only fast-path attention metadata builder, and
  always-scatter-to-`[T, H]` output contract.
- Adds a unit test for the model branch (predicate, gather/back/scatter
  semantics, attention-metadata invariants) plus a per-prompt parity
  test (20 prompts × greedy sampling × YOCO on/off, assert byte-identical
  token IDs).
- Runs the full fixed-benchmark suite (chat 1000/1000 n=80, summ
  8000/1000 n=80, MMLU N=500) on SGLang YOCO-on, SGLang YOCO-off, and
  vLLM nightly MTP, and embeds the result table in each draft PR body.
- Documents the new arg in the user-facing help text and in
  `python/sglang/srt/server_args.py` docstrings.

### Lower Bound (Minimum Acceptable Scope)

The implementation:
- Adds the `--kv-sharing-fast-prefill` server arg as a plain bool flag
  on `ServerArgs` and `ModelConfig`, default False, no startup-time
  validation beyond a single guard at the YOCO branch site.
- Implements the YOCO branch inside `Gemma4TextModel.forward` with the
  predicate at AC-2, the always-scatter output contract, and a minimal
  triton metadata rebuild.
- Adds one parity test (20-prompt greedy, YOCO on vs off) and one
  benchmark comparison (summ 8000/1000 n=80 with YOCO on vs off vs
  vLLM).
- Stages one draft PR (instead of three stacked) on `pyc96/sglang`
  containing all the changes.

### Allowed Choices

- Can use:
  - The existing `LogitsProcessor._get_pruned_states` index construction
    pattern (`cumsum(extend_seq_lens) - 1`).
  - Direct construction of a triton `ForwardMetadata` dataclass inside
    the model for the cross-decoder phase (the user is pinned to triton,
    so backend-agnosticism is not required).
  - A small helper module/file under `python/sglang/srt/models/` or
    `python/sglang/srt/layers/` if the YOCO logic is large enough to
    warrant separation from `gemma4_causal.py`.
  - Python-level branching inside `Gemma4TextModel.forward` (no Triton
    kernel rewrite is required).
  - Stacked PRs (PR-A: flag plumbing; PR-B: model + metadata + tests;
    PR-C: bench results + any tuning) per the user's "multiple PRs per
    task" allowance.
- Cannot use:
  - Any change to the existing non-YOCO layer loop's semantics
    (default-off must be a no-op).
  - Any change to the KV cache layout or to `req_to_token_pool` /
    `token_to_kv_pool` data structures.
  - Any change to attention backends other than triton (flashinfer,
    fa3, trtllm_mha are out of scope for v0).
  - Any change to multimodal towers (vision/audio); YOCO must coexist
    with the existing `mm_disabled_models` treatment without changes.
  - Any direct dependency on vLLM source code or vLLM-specific
    abstractions (we mirror the technique, not the implementation).
  - Any PR submitted to `sgl-project/sglang` upstream; all PRs draft on
    `pyc96/sglang` only.
  - Any PR that lacks a benchmark+MMLU table in its body.

## Feasibility Hints and Suggestions

### Conceptual Approach

```text
ServerArgs (CLI: --kv-sharing-fast-prefill, default False)
    └─→ ModelConfig.kv_sharing_fast_prefill
            └─→ Gemma4TextModel.__init__ reads it, stores
                self.kv_sharing_fast_prefill_enabled

Gemma4TextModel.forward(input_ids, positions, forward_batch, ...):
    if can_run_yoco(self, forward_batch):
        # YOCO branch
        hidden_states = self.embed_tokens(input_ids)
        # Front half: layers [0, first_kv_shared_layer_idx)
        for layer_idx in range(0, self.first_kv_shared_layer_idx):
            hidden_states = self.layers[layer_idx](positions, hidden_states,
                                                   forward_batch)[0]

        # Gather index
        last_index = torch.cumsum(forward_batch.extend_seq_lens, dim=0) - 1

        # Gather + cross-decoder + scatter
        front_out = hidden_states.clone()  # save full-shape for scatter
        gathered_h = front_out[last_index]
        gathered_pos = positions[last_index]

        # Build decode-shaped attn metadata for back half, restore on exit
        with self._cross_decoder_attn_metadata_scope(forward_batch):
            cross_h = gathered_h
            for layer_idx in range(self.first_kv_shared_layer_idx, len(self.layers)):
                cross_h = self.layers[layer_idx](gathered_pos, cross_h,
                                                 forward_batch)[0]

        # Scatter back
        front_out.index_copy_(0, last_index, cross_h)
        hidden_states = front_out

        # Final norm + LM head proceeds as usual on hidden_states
    else:
        # Existing path verbatim
        ...
    hidden_states = self.norm(hidden_states)
    return hidden_states
```

The `_cross_decoder_attn_metadata_scope` is a `contextmanager` that:
1. Saves the current `forward_batch.attn_backend.forward_metadata`.
2. Builds a new `ForwardMetadata` instance with `qo_indptr=[0..B]`,
   `max_extend_len=1`, `kv_indptr=cumsum(seq_lens)`, `kv_indices`
   pointing at the full per-request KV slots from the pool.
3. Yields.
4. Restores the saved metadata.

For triton specifically, look at `triton_backend.py:298-363` (DECODE
branch of `init_forward_metadata`) for the exact field shapes and the
SWA buffer setup.

### Relevant References

- `python/sglang/srt/models/gemma4_causal.py:922-1003` — the existing
  `Gemma4TextModel.forward` layer loop that this patch modifies.
- `python/sglang/srt/models/gemma4_causal.py:360-403` — KV-shared
  layer detection logic (`is_kv_shared_layer`, `kv_shared_layer_index`)
  that the new branch piggybacks on.
- `python/sglang/srt/layers/logits_processor.py:432-447` — reference
  construction of `last_index = cumsum(extend_seq_lens) - 1`.
- `python/sglang/srt/layers/attention/triton_backend.py:298-363, 437-482` —
  DECODE and EXTEND branches of `init_forward_metadata` (the templates
  for the metadata builder).
- `python/sglang/srt/model_executor/forward_batch_info.py:273-432` —
  `ForwardBatch` field list (especially `extend_seq_lens`, `seq_lens`,
  `out_cache_loc`, `req_pool_indices`, `forward_mode`,
  `capture_hidden_mode`).
- `python/sglang/srt/server_args.py:2205-2277` — `_handle_model_specific_adjustments`
  for Gemma-4 (existing site for any Gemma-4-specific arg validation).
- `python/sglang/srt/configs/model_config.py:215-244` — `mm_disabled_models`
  and the existing `Gemma4ForConditionalGeneration` handling that the new
  flag must coexist with.
- `python/sglang/srt/speculative/frozen_kv_mtp_worker.py:493-503` — the
  worker's `forward_target_extend` entry, which is the call site that
  benefits from YOCO when MTP is on.
- `vllm/model_executor/models/gemma4.py:759-952, 1190-1273` — vLLM
  reference impl for `Gemma4SelfDecoderLayers`, `Gemma4CrossDecoderLayers`,
  and `fast_prefill_forward`. **For inspiration only; do not copy code
  directly — the SGLang port must use SGLang's own framework abstractions.**
- `vllm/v1/attention/backends/utils.py:367-433, 728-776` — vLLM
  reference for `make_kv_sharing_fast_prefill_common_attn_metadata` and
  `create_fast_prefill_custom_backend`. **For inspiration only.**
- `agent-pod/runs/20260522_gemma4_26b_a4b_it_h100_sota_humanize/analysis/framework-gap-investigation.md` —
  prior D1 attempt that documents the PCG/Inductor incompatibility we
  must work around (`is_in_piecewise_cuda_graph()` guard).
- `agent-pod/runs/20260523_gemma4_31b_it_h100_sota_humanize/yoco/draft.md` —
  the draft document this plan refines.

## Dependencies and Sequence

### Milestones

1. Milestone 1 — Flag plumbing (PR-A on `pyc96/sglang`)
   - Phase A: Add `kv_sharing_fast_prefill: bool = False` to
     `ServerArgs` with `argparse` registration and help text.
   - Phase B: Plumb the flag through `ModelConfig` (or wherever the
     model can read it at construction time).
   - Phase C: Add a startup-time validator in
     `_handle_model_specific_adjustments` (or equivalent) that rejects
     incompatible combinations (non-Gemma-4 arch, non-triton attn
     backend, EAGLE/EAGLE3 spec algo).
   - Phase D: Add a unit/integration test that asserts the flag appears
     in `--help`, parses correctly, and the validator rejects bad combos.

2. Milestone 2 — Model branch + attention metadata (PR-B, stacked on PR-A)
   - Phase A: Add an `is_in_yoco_scope()` predicate helper.
   - Phase B: Add a `_cross_decoder_attn_metadata_scope` context manager
     (triton-only) that saves, rebuilds, and restores the attention
     metadata for the back-half phase.
   - Phase C: Add the YOCO branch inside `Gemma4TextModel.forward`
     (gather + back-half loop + scatter).
   - Phase D: Wire the branch behind the flag-on predicate.
   - Phase E: Add a unit test for `_cross_decoder_attn_metadata_scope`
     (asserts shapes, restoration).
   - Phase F: Add the per-prompt parity test (20 prompts, greedy,
     YOCO on vs off, assert identical token IDs).
   - Phase G: Add a `Gemma4TextModel.forward` unit test that runs both
     paths on the same input and asserts allclose on the last-token row.

3. Milestone 3 — Benchmark + landing (PR-C, stacked on PR-B)
   - Phase A: Run the fixed benchmark suite (chat + summ + MMLU) with
     YOCO on, YOCO off, and vLLM nightly MTP, all on the campaign
     hardware (H100 TP=2).
   - Phase B: Embed the result table in the PR-C body.
   - Phase C: If summ throughput < 492 tok/s, do one round of tuning
     (e.g. micro-optimize the gather, try `index_select` vs fancy
     indexing, profile and address any hot spot) and re-bench.
   - Phase D: Update the campaign's `model-loop-checkpoint.md` and
     `final_report.md` with the new "best SGLang" row.

Dependencies between components:
- PR-B depends on PR-A's flag being readable from the model.
- PR-C depends on PR-B's correctness tests passing.
- All three PRs share the same base branch `pyc/sota-gemma4-31b-mm-disabled`
  (stacked).

## Task Breakdown

| Task ID | Description | Target AC | Tag | Depends On |
|---------|-------------|-----------|------|------------|
| task1  | Add `kv_sharing_fast_prefill` to `ServerArgs` + argparse + help text | AC-1 | coding | - |
| task2  | Plumb flag through `ModelConfig` so models can read it at `__init__` | AC-1 | coding | task1 |
| task3  | Startup validators (Gemma-4 arch, triton backend, no EAGLE/EAGLE3 with the flag) | AC-1 | coding | task2 |
| task4  | Unit test for flag plumbing + validator rejection of bad combos | AC-1 | coding | task3 |
| task5  | Open PR-A (flag plumbing only) on `pyc96/sglang` (draft) | AC-8 | coding | task4 |
| task6  | Implement `_cross_decoder_attn_metadata_scope` context manager (triton-only) | AC-4 | coding | task5 |
| task7  | Add `can_run_yoco` predicate function | AC-2 | coding | task5 |
| task8  | Implement YOCO branch in `Gemma4TextModel.forward` (gather + back-half + scatter) | AC-2, AC-3, AC-5 | coding | task6, task7 |
| task9  | Unit test for `_cross_decoder_attn_metadata_scope` (shape + restoration) | AC-4 | coding | task6 |
| task10 | Unit test for `Gemma4TextModel.forward` allclose on last-token row, YOCO on vs off | AC-3 | coding | task8 |
| task11 | Per-prompt parity test: 20 prompts greedy, YOCO on vs off, identical token IDs | AC-3 | coding | task8 |
| task12 | Unit test for KV cache write-count invariant | AC-5 | coding | task8 |
| task13 | Open PR-B (model + metadata + tests) stacked on PR-A (draft) | AC-8 | coding | task9, task10, task11, task12 |
| task14 | Run MMLU N=500 with YOCO on and YOCO off, capture results | AC-6 | coding | task13 |
| task15 | Run the fixed benchmark (chat + summ) on YOCO on, YOCO off, and vLLM | AC-7 | coding | task13 |
| task16 | If summ throughput < 492 tok/s, do one tuning round and rebench | AC-7 | coding | task15 |
| task17 | Open PR-C (bench + tuning) stacked on PR-B (draft) with results table | AC-7, AC-8 | coding | task16 |
| task18 | Update campaign `model-loop-checkpoint.md` and `final_report.md` | AC-7 | coding | task17 |
| task19 | Final review pass: confirm all three PRs are draft, all contain bench/MMLU tables, none submitted | AC-8 | analyze | task17 |

## Claude-Codex Deliberation

### Agreements
- The YOCO branch must be gated behind a predicate that excludes
  DECODE, TARGET_VERIFY, PCG-active forwards, input-logprob requests,
  and models with PLE.
- Default-off is the safer first cut; can be flipped to default-on for
  Gemma-4 after correctness is established.
- Triton-only metadata rebuild is acceptable because the user is pinned
  to triton.

### Resolved Disagreements
- None at plan-generation time. Codex review will surface any during the
  RLCR loop.

### Convergence Status
- Final Status: `converged` (user-confirmed all open questions; one-shot
  draft → user-approved → plan generated).

## Pending User Decisions

(All decisions resolved before plan generation; none pending.)

- DEC-1: Default value of `--kv-sharing-fast-prefill` — RESOLVED:
  `False` (opt-in).
- DEC-2: Hidden-mode handling — RESOLVED: always scatter back to `[T, H]`.
- DEC-3: Attention metadata strategy — RESOLVED: triton-only fast path.
- DEC-4: Validation scope — RESOLVED: MMLU + per-prompt parity test.
- DEC-5: Port scope — RESOLVED: Gemma-4 only.
- DEC-6: PR strategy — RESOLVED: three stacked draft PRs.
- DEC-7: SOTA-loop stop criterion — RESOLVED: summ tok/s gap closes
  by ≥ 30 % (≥ 492 tok/s).

## Implementation Notes

### Code Style Requirements
- Implementation code and comments must NOT contain plan-specific
  terminology such as "AC-", "Milestone", "Step", "Phase", or similar
  workflow markers. These terms are for plan documentation only, not
  for the resulting codebase.
- Use descriptive, domain-appropriate naming: e.g.
  `enable_kv_sharing_fast_prefill`, `cross_decoder_attn_metadata_scope`,
  `gather_last_token_index`.
- Follow existing SGLang code style: lower_snake_case for functions,
  CapWords for classes, ALL_CAPS for module-level constants.
- Add a brief docstring on the YOCO branch citing the inspiration source
  (vLLM Gemma-4 fast-prefill) and noting the predicate semantics.
- Do not import from `vllm.*` anywhere in the SGLang code.
- All new code paths must have a corresponding unit test, with at least
  one positive and one negative case per acceptance criterion.

### Branching and PR Conventions
- All branches branch from `pyc/sota-gemma4-31b-mm-disabled` (the current
  best base for the 31b-it campaign).
- PR-A branch name: `pyc/yoco-fast-prefill-config`.
- PR-B branch name: `pyc/yoco-fast-prefill-impl`, stacked on PR-A.
- PR-C branch name: `pyc/yoco-fast-prefill-bench`, stacked on PR-B.
- All three opened as draft via `gh pr create --repo pyc96/sglang --draft`.
- No `git push` to `sgl-project/sglang` from any branch.
