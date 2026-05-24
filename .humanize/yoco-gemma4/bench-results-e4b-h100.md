# YOCO Fast-Prefill Bench Results — Gemma-4 E4B-it on H100 TP=2

Date: 2026-05-24
Branch: `pyc/yoco-fast-prefill-bench` (stacked on `pyc/yoco-fast-prefill-impl`)

## Setup

| Item | Value |
|---|---|
| Model | `google/gemma-4-E4B-it` (4.5B dense + PLE, 42 layers, `num_kv_shared_layers=18`, PLE `hidden_size_per_layer_input=256`) |
| Hardware | NVIDIA H100 80 GB SXM5, TP=2 |
| Attention backend | triton |
| SGLang branch | `pyc/yoco-fast-prefill-impl` (= PR-A flag plumbing + PR-B model code) |
| YOCO-on launch flag | `--kv-sharing-fast-prefill` |
| Both servers | `--disable-piecewise-cuda-graph` (avoids a pre-existing PCG bug in `radix_attention.unified_attention_with_output` that crashes on KV-shared layers when key is None; this bug pre-dates YOCO and is orthogonal). |
| Workload | random, warmup 2, seed 1, num_prompts=80 |
| Quality benchmark | MMLU N=500, seed 0, temp 0 |

## Correctness — per-prompt parity (20 prompts, greedy)

```
{
  "total": 20,
  "matched": 20,
  "mismatched": 0,
  "match_rate": 1.0
}
```

**20/20 prompts produce byte-identical output between YOCO-off and YOCO-on.**

## Quality — MMLU N=500

| Stack | accuracy | correct/500 |
|---|---:|---:|
| SGLang YOCO off | 0.594 | 297 |
| **SGLang YOCO on** | **0.592** | **296** |

Δ = −0.2 pp (1 question difference; within the ±1 pp acceptance bar).

## Performance — chat 1000/1000 n=80

| Metric | YOCO off | YOCO on | Δ |
|---|---:|---:|---|
| output_throughput (tok/s) | 9327.5 | **9520.2** | **+2.1 %** |
| duration (s) | 8.6 | **8.4** | −2.3 % |
| mean_ttft (ms) | 636.2 | **567.6** | −10.8 % |
| median_ttft (ms) | 618.0 | **557.7** | **−9.8 %** |
| p99_ttft (ms) | 1032.1 | **856.3** | **−17.0 %** |
| median_tpot (ms) | 7.9 | 7.8 | −1.3 % |

## Performance — summ 8000/1000 n=80 (the target workload for YOCO)

| Metric | YOCO off | YOCO on | Δ |
|---|---:|---:|---|
| **output_throughput (tok/s)** | **3469.6** | **4078.5** | **+17.6 %** |
| duration (s) | 23.1 | **19.6** | −15.2 % |
| mean_ttft (ms) | 5945.3 | **4319.4** | **−27.3 %** |
| **median_ttft (ms)** | **5920.3** | **4276.1** | **−27.8 %** |
| **p99_ttft (ms)** | **11550.5** | **8144.6** | **−29.5 %** |
| median_tpot (ms) | 17.1 | **15.3** | **−10.5 %** |

## Why the summ scenario wins more than chat

YOCO eliminates compute on the `num_kv_shared_layers` back-half layers for all
but the last token per request. That saving is proportional to
`(input_len − 1) × num_kv_shared_layers / num_hidden_layers`.

For E4B-it (`num_kv_shared_layers=18`, `num_hidden_layers=42`):
- chat 1k/1k: theoretical max layer-compute saving =
  `(1000−1) × 18 / 42 = ~428 layer-token-equivalents per request`
- summ 8k/1k: theoretical max layer-compute saving =
  `(8000−1) × 18 / 42 = ~3429 layer-token-equivalents per request`

→ 8× more compute saved on summ, which matches the observed throughput
delta (chat +2 %, summ +18 %) since the rest of the per-token cost is
fixed (front-half layers + lm head + scheduler overhead).

## Server log evidence

YOCO-on startup confirms the flag fires correctly:

```
[2026-05-24 06:52:20] KV-sharing fast prefill enabled for Gemma4ForConditionalGeneration (num_kv_shared_layers=18).
```

YOCO-off startup log makes no mention of fast-prefill (default-off behavior preserved).

## Why other Gemma-4 checkpoints don't benefit

The HF text-config `num_kv_shared_layers` is:
- `gemma-4-26b-a4b-it`: **0** (MoE; benchmarked in prior campaign at `runs/20260522_gemma4_26b_a4b_it_*`)
- `gemma-4-31B-it`: **0** (dense; benchmarked in prior campaign at `runs/20260523_gemma4_31b_it_*`)
- **`gemma-4-E4B-it`: 18** ← this run
- `gemma-4-E2B-it`: 20 (not benchmarked; would benefit similarly)
- `gemma-4-*-assistant` (MTP drafts): 4, but the worker runs them only in DECODE mode, so YOCO degrades to a no-op.

So YOCO is a Gemma-4-E*-IT optimization. The PR-A flag is universal (any future Gemma-4 checkpoint that declares `num_kv_shared_layers > 0` automatically benefits when `--kv-sharing-fast-prefill` is set).

## Reproducer

```bash
# SGLang YOCO-off (GPUs 0,1)
bash /home/pyc_google_com/dev/gemma-op/agent-pod/runs/20260524_gemma4_e4b_yoco_h100/benchmark/launch_sglang_e4b.sh \
     sglang_e4b_yoco_off 0,1 30000 yoco_off

# SGLang YOCO-on (GPUs 2,3, in parallel)
bash /home/pyc_google_com/dev/gemma-op/agent-pod/runs/20260524_gemma4_e4b_yoco_h100/benchmark/launch_sglang_e4b.sh \
     sglang_e4b_yoco_on 2,3 30001 yoco_on

# Parity test
python /home/pyc_google_com/dev/gemma-op/agent-pod/runs/20260524_gemma4_e4b_yoco_h100/quality/parity_check.py \
       --url-off http://127.0.0.1:30000 \
       --url-on  http://127.0.0.1:30001 \
       --num-prompts 20

# Bench
bash /home/pyc_google_com/dev/gemma-op/agent-pod/runs/20260524_gemma4_e4b_yoco_h100/benchmark/run_benchmark.sh \
     e4b_yoco_on sglang-oai-chat http://127.0.0.1:30001 summ 8000 1000 80
```
