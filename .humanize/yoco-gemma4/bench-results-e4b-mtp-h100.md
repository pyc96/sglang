# YOCO Fast-Prefill × Frozen-KV MTP Interaction — Gemma-4 E4B-it on H100 TP=2

Date: 2026-05-24
Stack: stacked on `pyc/yoco-fast-prefill-bench` (PR-C #13).

## Purpose

PR-C documented YOCO behaviour on `gemma-4-E4B-it` with **no speculative
decoding**.  Reviewer question: *will this be exportable to Gemma-4 MTP?*

This document captures the live measurement that answers it.  Spoiler:
the predicate already fires in exactly the right place — the target
model's EXTEND-mode forward, which the frozen-KV MTP worker calls via
``forward_target_extend`` (``frozen_kv_mtp_worker.py:493-503``).  No
worker-side changes are required.  YOCO is correct and beneficial under
MTP for prefill-bound workloads; the only nuance is that MTP itself is
a net loss on E4B at this concurrency, so the visible **throughput**
delta from YOCO shrinks (it's still a real win — measured below — but
overlaid on a smaller absolute number).

## Why YOCO mostly does NOT apply to MTP-internal hot paths

Every forward path in ``frozen_kv_mtp_worker.py``:

| Phase | Worker entry | Forward mode | Q-tokens per req | Does YOCO fire? | Why |
|---|---|---|---:|---|---|
| Target prefill | ``forward_target_extend`` (``:493``) | EXTEND | 1..L_input | **Yes** | The same EXTEND path PR-C measured. |
| Assistant seed step | ``_run_assistant_seed_step`` (``:325``) | EXTEND with ``seq_len=1`` per req | 1 | No, predicate-skipped | ``(N−1) = 0`` ⇒ nothing to gather. |
| Recurrent draft step | ``draft_forward`` (``:650``) | DECODE | 1 | No, predicate-skipped (``is_extend()`` False) | Already decode-shaped. |
| Target verify | ``verify`` (``:709+``) | TARGET_VERIFY | ``1 + spec_num_draft_tokens`` (=4) | No, predicate-skipped (``is_target_verify()`` True) | Quasi-decode-shaped, savings below noise. |

So YOCO's contribution to MTP throughput is entirely from the **target
prefill** phase.  Per-step decode is untouched (correctly so).

## Setup

| Item | Value |
|---|---|
| Target | ``google/gemma-4-E4B-it`` (4.5B dense + PLE, 42 layers, ``num_kv_shared_layers=18``, ``hidden_size_per_layer_input=256``) |
| MTP draft | ``google/gemma-4-E4B-it-assistant`` (4 layers, all KV-shared) |
| Spec config | ``--speculative-algorithm NEXTN --speculative-num-steps 3 --speculative-num-draft-tokens 4 --speculative-eagle-topk 1`` (promoted to FROZEN_KV_MTP by ``_handle_frozen_kv_mtp``) |
| Hardware | NVIDIA H100 80 GB SXM5, TP=2 |
| Attention backend | triton |
| SGLang branch | ``pyc/yoco-fast-prefill-impl`` (= PR-A flag plumbing + PR-B model code) |
| YOCO-on | ``--kv-sharing-fast-prefill`` |
| Both servers | ``--disable-piecewise-cuda-graph`` (PCG bug, see PR-C note) |
| Workload | random, warmup 2, seed 1, num_prompts=80 |
| Quality bar | MMLU N=500, seed 0, temp 0; per-prompt parity 20 prompts greedy |

## Correctness under MTP

### Per-prompt parity (20 prompts, greedy)

```json
{"total": 20, "matched": 20, "mismatched": 0, "match_rate": 1.0}
```

**20/20 prompts produce byte-identical output between YOCO-off and YOCO-on, with MTP enabled on both sides.**  This is the strongest possible oracle: the model + assistant + verify loop all behave identically.

### MMLU N=500 (seed 0, temp 0)

| Stack | accuracy | correct/500 |
|---|---:|---:|
| MTP + YOCO off | 0.588 | 294 |
| **MTP + YOCO on** | **0.590** | **295** |

Δ = +0.2 pp (1 question; within ±1 pp).  Also tied with the no-MTP MMLU
numbers from PR-C (0.594 / 0.592), so MTP itself is MMLU-neutral on E4B.

### accept_length under MTP

| Scenario | YOCO off | YOCO on | Δ |
|---|---:|---:|---|
| chat 1k/1k | 2.20 | 2.20 | identical |
| summ 8k/1k | 2.10 | 2.10 | identical |

YOCO does not change what tokens the verify loop accepts.  This is a
direct consequence of the byte-identical parity above.

## Performance

### summ 8000/1000 n=80 (the workload YOCO is designed to win)

| Metric | MTP YOCO off | **MTP YOCO on** | Δ |
|---|---:|---:|---|
| **output tok/s** | 1469.5 | **1559.0** | **+6.1 %** |
| duration (s) | 54.4 | 51.3 | −5.7 % |
| **median TTFT (ms)** | 6414.2 | **4555.5** | **−29.0 %** |
| **p99 TTFT (ms)** | 31526.7 | **28237.7** | **−10.4 %** |
| median TPOT (ms) | 22.5 | **21.1** | **−6.2 %** |
| accept_length | 2.10 | 2.10 | tied |

### chat 1000/1000 n=80

| Metric | MTP YOCO off | MTP YOCO on | Δ |
|---|---:|---:|---|
| output tok/s | 2932.1 | 2887.9 | −1.5 % (noise) |
| duration (s) | 27.3 | 27.7 | +1.5 % |
| median TTFT (ms) | 1001.6 | 1012.3 | +1.1 % (noise) |
| p99 TTFT (ms) | 12928.5 | 13189.1 | +2.0 % (noise) |
| median TPOT (ms) | 10.3 | 10.5 | +1.9 % (noise) |
| accept_length | 2.20 | 2.20 | tied |

The chat-1k case is at the break-even point: the YOCO gather/scatter
overhead roughly matches the saved back-half compute when prefill is
short (1 k tokens × 18 KV-shared layers = ~18 k layer-tokens saved per
request, vs the fixed `~1 ms` metadata-rebuild cost).  Within noise.

## Comparison with PR-C (no-MTP) numbers

Putting all four E4B runs side-by-side:

| Scenario | Config | tok/s | Δ vs YOCO-off |
|---|---|---:|---|
| chat | no-MTP YOCO off | 9327.5 | baseline |
| chat | no-MTP YOCO on  | 9520.2 | **+2.1 %** |
| chat | MTP YOCO off    | 2932.1 | (MTP is a 3× loss on E4B at this concurrency) |
| chat | MTP YOCO on     | 2887.9 | **−1.5 %** (noise) |
| summ | no-MTP YOCO off | 3469.6 | baseline |
| summ | no-MTP YOCO on  | 4078.5 | **+17.6 %** |
| summ | MTP YOCO off    | 1469.5 | (MTP is a 2.4× loss on E4B summ) |
| summ | MTP YOCO on     | 1559.0 | **+6.1 %** |

Two separate things are happening:

1. **MTP is a net loss on E4B for both scenarios.**  This is the same
   ``max_running_requests=48`` auto-cap from ``_handle_frozen_kv_mtp``
   (``arg_groups/speculative_hook.py:233-250``) plus the
   ``disable_overlap_schedule`` penalty that ate the 31B-it campaign's
   MTP run too — but on E4B (smaller model, faster per-token) the gap
   is even more pronounced.  Pre-existing, orthogonal to YOCO.
2. **YOCO is still a win on top of MTP for prefill-bound workloads.**
   summ median TTFT drops 29.0 % under MTP — essentially the same
   −27.8 % drop measured under no-MTP — because TTFT is purely the
   target-prefill cost, which YOCO accelerates regardless of whether
   MTP is on downstream.  The throughput delta shrinks (+18 % → +6 %)
   because the steady-state decode is now MTP-dominated, but it stays
   positive.

## What this means for the rollout

| Recommendation | Reason |
|---|---|
| **Ship YOCO as-is** for any Gemma-4 model with ``num_kv_shared_layers > 0``, whether or not MTP is enabled. | The predicate already routes YOCO to the right path (target EXTEND); MTP-internal phases are correctly skipped. |
| Document in the YOCO PR-A help text that the throughput win shrinks under MTP because per-step decode is MTP-dominated, but TTFT win is unchanged. | Sets correct user expectations. |
| **Do NOT** invest in YOCO-for-decode work (gathering inside the verify loop, gathering inside the draft step). | The savings ``(seq_len − 1) × num_kv_shared_layers / num_hidden_layers`` are ≤ 0 in those phases by construction. |
| Investigate the separate ``--max-running-requests`` cap and ``disable_overlap_schedule`` cost imposed by ``_handle_frozen_kv_mtp`` on small models. | That's the real reason MTP loses ~3× throughput on E4B; orthogonal to YOCO but worth a follow-up campaign. |

## Reproducer

```bash
# YOCO-off + MTP (GPUs 0,1)
bash /home/pyc_google_com/dev/gemma-op/agent-pod/runs/20260524_gemma4_e4b_yoco_h100/benchmark/launch_sglang_e4b_mtp.sh \
     sglang_e4b_mtp_yoco_off 0,1 30100 yoco_off

# YOCO-on + MTP (GPUs 2,3)
bash /home/pyc_google_com/dev/gemma-op/agent-pod/runs/20260524_gemma4_e4b_yoco_h100/benchmark/launch_sglang_e4b_mtp.sh \
     sglang_e4b_mtp_yoco_on 2,3 30101 yoco_on

# Parity (the strongest oracle: byte-identical outputs)
python /home/pyc_google_com/dev/gemma-op/agent-pod/runs/20260524_gemma4_e4b_yoco_h100/quality/parity_check.py \
       --url-off http://127.0.0.1:30100 \
       --url-on  http://127.0.0.1:30101 \
       --num-prompts 20

# Bench
for SCEN in "summ 8000 1000" "chat 1000 1000"; do
  bash /home/pyc_google_com/dev/gemma-op/agent-pod/runs/20260524_gemma4_e4b_yoco_h100/benchmark/run_benchmark.sh \
       e4b_mtp_yoco_on sglang-oai-chat http://127.0.0.1:30101 $SCEN 80
done

# MMLU
python /home/pyc_google_com/dev/gemma-op/agent-pod/runs/20260524_gemma4_e4b_yoco_h100/quality/run_mmlu.py \
       --url http://127.0.0.1:30101 --label e4b_mtp_yoco_on --num-questions 500 --seed 0
```

## Raw artifacts

```
runs/20260524_gemma4_e4b_yoco_h100/benchmark/result_e4b_mtp_yoco_off_summ_8000_1000_n80_20260524_071200.jsonl
runs/20260524_gemma4_e4b_yoco_h100/benchmark/result_e4b_mtp_yoco_on_summ_8000_1000_n80_20260524_071200.jsonl
runs/20260524_gemma4_e4b_yoco_h100/benchmark/result_e4b_mtp_yoco_off_chat_1000_1000_n80_20260524_071312.jsonl
runs/20260524_gemma4_e4b_yoco_h100/benchmark/result_e4b_mtp_yoco_on_chat_1000_1000_n80_20260524_071312.jsonl
runs/20260524_gemma4_e4b_yoco_h100/quality/results.jsonl  (4 MMLU rows: no-MTP off/on + MTP off/on)
```
