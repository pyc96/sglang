# Draft: Port vLLM's YOCO Fast-Prefill to SGLang for Gemma-4

## Background

vLLM's Gemma-4 (and Gemma-3n) implementation contains a "fast prefill" path
that exploits the model's KV-cache-sharing structure. The last
`num_kv_shared_layers` decoder layers reuse K/V from earlier layers of the
same type (sliding vs full) and write nothing of their own to the KV cache.
Because the only function of those layers' hidden_states is to feed the LM
head, **their compute is only needed at the per-request "last extend
token" positions** — i.e. `num_reqs` rows instead of `num_extend_tokens`
rows.

For Gemma-4 31B-IT on H100 + TP=2 + an 80-prompt × 8000-token-input
summarization workload, this means the last `num_kv_shared_layers` of the
60-layer dense model can process ~80 rows in the cross-decoder phase
instead of ~640,000 rows. Per the campaign benchmark, the SGLang summ TPOT
already beats vLLM (23.4 vs 31.8 ms), but SGLang's summ throughput is
−62% vs vLLM (331 vs 868 tok/s) and median TTFT is +96% (78s vs 40s).
**Closing that gap is the YOCO opportunity.**

This draft proposes porting the YOCO fast-prefill technique to SGLang's
Gemma-4 path, with full benchmark + MMLU validation per the campaign's
acceptance bar.

## Problem statement

When `forward_mode == EXTEND` and the served model has
`num_kv_shared_layers > 0`:
- The first `K = num_hidden_layers − num_kv_shared_layers` decoder layers
  must run on the full `[T = sum(extend_seq_lens), H]` hidden_states
  tensor — they write into the KV cache that the back half will read.
- The last `num_kv_shared_layers` layers do **not** write KV; they only
  produce hidden_states. Only the per-request last-extend-token rows of
  those hidden_states are sampled (via `LogitsProcessor._get_pruned_states`
  at `logits_processor.py:432-447`, which builds
  `last_index = cumsum(extend_seq_lens) - 1` and gathers
  `hidden_states[last_index]`).

So all per-token compute in the cross-decoder for the non-last positions
is **wasted work**. YOCO eliminates it by gathering hidden_states to the
last-token positions before the cross-decoder runs, running the
cross-decoder only on the gathered rows, then scattering the cross-decoder
output back into a full-shape hidden_states tensor (so that downstream
consumers that expect `CaptureHiddenMode == FULL` still get the right
contract for the rows they actually read).

## What the optimization touches

Per the SGLang+vLLM audit:

1. **Model code** — split `Gemma4TextModel.forward` (the layer loop in
   `python/sglang/srt/models/gemma4_causal.py:922-1003`) into a self-decoder
   half (layers `[0, K)`) and a cross-decoder half (layers `[K, N)`) with
   a gather/scatter between them.

2. **Attention metadata** — the cross-decoder runs Q at decode-shape
   (one query token per request) but K/V at full prefix length. The
   triton extend kernel (`triton_backend.py:894+`) is built from
   `qo_indptr = cumsum(extend_seq_lens)`, so calling it directly on the
   gathered Q would walk past the end of the q buffer. The cleanest fix
   is to temporarily build a DECODE-shaped `ForwardMetadata` for the
   cross-decoder phase (qo_indptr = `[0..B]`, max_extend_len = 1,
   `kv_indptr` and `kv_indices` rebuilt from `seq_lens` covering the
   full prefix-plus-extend KV span).

3. **Per-Layer Embeddings (PLE)** — Gemma-4 31B-IT and E4B-IT have
   `hidden_size_per_layer_input == 0` (no PLE), so the
   `per_layer_inputs` argument that gets sliced per layer is `None`.
   For the MoE 26B-A4B-IT variant, PLE is also disabled
   (`enable_moe_block=True` does not imply PLE). **PLE is therefore
   out of scope for the v0 patch**: a small guard `if has_ple:
   fall_back_to_eager_full_path` covers all Gemma-4 sizes we care about.
   Note: Gemma-3n E2B/E4B *does* use PLE; if we ever extend YOCO to
   Gemma-3n, the PLE per_layer_inputs tensor needs to be sliced by
   `last_index` before being passed into the cross-decoder.

4. **Speculative decoding (frozen-KV MTP)** — YOCO applies only to
   target-model EXTEND (`forward_target_extend` at
   `frozen_kv_mtp_worker.py:493-503`). The assistant has
   `num_kv_shared_layers=0` (`gemma4_mtp.py:71`), so YOCO degrades to
   a no-op on the assistant forward. TARGET_VERIFY is decode-shaped
   (Q-per-req=1+spec_tokens) and also a no-op. The worker calls
   `target_worker.forward_batch_generation(batch)` with
   `capture_hidden_mode == FULL`, then reads only the last-token row via
   `_select_last_extend_hidden`. So **YOCO must still produce
   full-shape hidden_states on output** (scatter back), but the
   downstream consumer is happy as long as the per-request last-token row
   is the post-cross-decoder value.

5. **Piecewise CUDA graph (PCG)** — SGLang's `PiecewiseCudaGraphRunner`
   currently captures the full `Gemma4TextModel.forward` end-to-end at
   each `capture_num_tokens` bucket. Splitting the loop into two halves
   that run at different token counts breaks the assumption of one fixed
   shape per captured graph. **v0 strategy:** keep YOCO opt-in via a
   server arg and **disable PCG when YOCO is enabled** (or vice versa).
   The eager EXTEND path is what 31B-IT currently uses anyway (the
   benchmark logs show `disable_piecewise_cuda_graph=False` is set but
   piecewise is empirically not engaged for the spec-decode hot path —
   see D1 attempt-ledger finding). This means YOCO's v0 lives in the
   eager EXTEND path and does not coexist with PCG.

6. **Multimodal** — already disabled in the current campaign branch via
   `mm_disabled_models` for `Gemma4ForConditionalGeneration`. YOCO has
   no interaction with multimodal towers; the bidi image-attention mask
   is applied only when `forward_mode == EXTEND AND contains_image_inputs()`
   (`gemma4_mm.py:614-622`). For the text-only benchmark workload, no
   image tokens → no mask → no interaction.

## Proposed mechanism

### Step 1: configuration flag

Add `kv_sharing_fast_prefill: bool = False` to `ServerArgs`. Default
False to preserve current behavior. Plumb through to `ModelConfig` and
read inside the model.

### Step 2: predicate

Inside `Gemma4TextModel.forward`, gate YOCO behind:

```python
can_run_yoco = (
    self.config.num_kv_shared_layers > 0
    and forward_batch.forward_mode.is_extend()
    and not forward_batch.forward_mode.is_target_verify()
    and forward_batch.extend_seq_lens is not None
    and forward_batch.batch_size > 0
    and not is_in_piecewise_cuda_graph()         # PCG bypass
    and not self._has_ple                        # PLE bypass (v0)
    and not _has_input_logprobs(forward_batch)   # input-logprob bypass
    and self.fast_prefill_enabled                # opt-in flag
)
```

If predicate fails, fall back to the existing layer loop unchanged.

### Step 3: gather index

```python
# Built on GPU; one int64 tensor of shape [B].
last_index = torch.cumsum(forward_batch.extend_seq_lens, dim=0) - 1
```

This is identical to the construction at `logits_processor.py:432-447`
that the LM head gather already uses, so we know it's semantically
correct.

### Step 4: front half

Run layers `[0, K)` on full `[T, H]` `hidden_states` as today. Each layer
writes its own K/V into the KV cache (because `is_kv_shared_layer=False`
for these layers).

### Step 5: build decode-shaped attention metadata for back half

The cross-decoder needs a `ForwardMetadata` that says "Q has B rows
(one per request), K/V is the full prefix+extend per request". Approach:

- Temporarily build a "shadow" `ForwardBatch` (or just temporarily mutate
  `forward_batch.forward_mode = DECODE`, `forward_batch.seq_lens` and
  `forward_batch.req_pool_indices` unchanged, and rebuild attention metadata
  via `attn_backend.init_forward_metadata(shadow_batch)`).
- Restore the original metadata after the back half finishes.

The triton backend's DECODE path (`triton_backend.py:298-363`) builds
`kv_indptr = cumsum(seq_lens)` and `kv_indices` that point at the full
KV pool slots for each request — which is exactly what we want.

### Step 6: gather → cross-decoder → scatter

```python
gathered_h        = front_half_out[last_index]               # [B, H]
gathered_pos      = positions[last_index]                    # [B]
# (skip per_layer_inputs gather — guarded out by has_ple)

cross_h = run_cross_decoder(gathered_pos, gathered_h, forward_batch)
# cross_h shape: [B, H]

if forward_batch.capture_hidden_mode in (CaptureHiddenMode.FULL,
                                          CaptureHiddenMode.LAST):
    # Scatter back into the full hidden_states tensor.
    full_h = front_half_out.clone()
    full_h.index_copy_(0, last_index, cross_h)
    hidden_states = full_h
else:
    # NULL mode: downstream LogitsProcessor will gather by last_index
    # again; we can return the gathered tensor directly and let
    # LogitsProcessor's NULL/short-circuit branch pass through.
    hidden_states = cross_h  # caller must respect this is [B, H], not [T, H]
```

For the v0 we choose the **always-scatter** path — it preserves the
existing `[T, H]` contract everywhere downstream, including frozen-KV MTP
worker's `_select_last_extend_hidden`. The clone + scatter cost is
`O(T * H)` = tiny compared to the saved cross-decoder compute.

### Step 7: invariants

- KV cache write semantics unchanged: front-half layers write their K/V
  with `save_kv_cache=True`; back-half layers have
  `is_kv_shared_layer=True` and `save_kv_cache=False`, so they only
  read. No KV-pool corruption possible.
- `req_to_token_pool` and `out_cache_loc` are not touched.
- `seq_lens`, `extend_seq_lens`, `extend_prefix_lens` are not modified
  in the persistent `ForwardBatch`; the back-half metadata is built from
  a copy or via a context manager that restores on exit.
- Final norm + LM head receives a `[T, H]` tensor with correct values at
  the `last_index` rows (post-cross-decoder) and pre-cross-decoder values
  at all other rows. Since the LM head only reads `last_index` rows, the
  per-token logits are bit-identical to the non-YOCO path.

## Quality bar

- MMLU N=500, seed 0, temp 0: must stay within ±1 pp of the current
  patched SGLang's 0.780 → i.e. ≥ 0.770.
- The fixed benchmark scenarios (chat 1000/1000 n=80, summ 8000/1000
  n=80) must complete all 80 prompts in both stacks.

## Performance bar

- summ scenario output tok/s: ≥ 331 (current SGLang no-MTP best on the
  patched branch). Target: meaningful improvement, e.g. ≥ 500 tok/s
  closing ~30 % of the gap to vLLM's 868.
- chat scenario must not regress more than 1 % vs the current SGLang
  best (1499 tok/s on the patched MTP cap-80 server).
- Median TTFT on summ must improve (current 78023 ms is the bound).

## Out of scope for v0

- Gemma-3n E2B / E4B PLE-aware YOCO.
- Coexistence with PCG (will be added after v0 ships and SGLang's
  piecewise capture is extended to support dynamic mid-graph reshapes).
- `--kv-sharing-fast-prefill` interaction with EAGLE / EAGLE3 (matches
  vLLM behavior — banned at startup).
- Attention backends other than triton (`flashinfer`, `fa3`, `trtllm_mha`
  not in scope; `triton` is the user-pinned backend for this campaign).

## Open design questions for the user

1. **Default value of `--kv-sharing-fast-prefill`**: should it default
   to `False` (opt-in, matches vLLM) or default to `True` for Gemma-4
   (auto-enabled in `_handle_model_specific_adjustments`, like the
   `swa_full_tokens_ratio` and `attention-backend triton` defaults)?

2. **NULL vs FULL hidden mode handling**: scatter-back always (simpler,
   preserves contract everywhere), or return the gathered tensor in
   NULL mode (saves the `clone()` + `index_copy_` cost when the caller
   doesn't need the full tensor)?

3. **Backend metadata rebuild strategy**: full `init_forward_metadata`
   call inside the model (works for any backend but adds invocation
   overhead — measured at ~0.1-0.3 ms on H100 for `bs=80`); or a
   triton-only fast-path that constructs a minimal `ForwardMetadata`
   directly in the model (faster but couples the model to the backend's
   metadata dataclass)?

4. **Validation scope**: in addition to MMLU + bench, do we want a
   per-prompt parity test (run the same N prompts with YOCO on/off,
   diff the generated tokens) to prove zero quality regression beyond
   MMLU's aggregate signal?

5. **Scope of the port**: only `Gemma4ForCausalLM` and
   `Gemma4ForConditionalGeneration` (i.e. Gemma-4 only), or also wire
   the same flag for any other SGLang model that already has
   `num_kv_shared_layers > 0`? Look at SGLang's model registry: are
   there other models with that attribute non-zero today?

6. **PR strategy**: one big PR (config + model + attn metadata + tests)
   or split into a stack (config + attn-builder wrapper as PR 1; model
   integration as PR 2; tests as PR 3)? Per the user's "one task can
   have multiple PRs" constraint, a stack is allowed.

7. **Acceptance criterion for SOTA loop**: should the loop continue past
   v0 (to add PLE support, PCG coexistence, etc.) until the summ
   throughput ties vLLM, or stop at v0 once the summ gap is materially
   closed (e.g. ≥ 30 %)?
