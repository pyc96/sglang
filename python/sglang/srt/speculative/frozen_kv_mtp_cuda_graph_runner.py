from __future__ import annotations

import bisect
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional

import torch

from sglang.srt.layers.dp_attention import DpPaddingMode, set_dp_buffer_len
from sglang.srt.model_executor.cuda_graph_runner import (
    CUDA_GRAPH_CAPTURE_FAILED_MSG,
    CudaGraphRunner,
    DeepEPCudaGraphRunnerAdapter,
    get_batch_sizes_to_capture,
    get_global_graph_memory_pool,
    model_capture_mode,
    set_global_graph_memory_pool,
    set_is_extend_in_batch,
    set_torch_compile_config,
)
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.model_executor.input_buffers import ForwardInputBuffers
from sglang.srt.speculative.frozen_kv_mtp_info import FrozenKVMTPDraftInput
from sglang.srt.utils import (
    require_attn_tp_gather,
    require_gathered_buffer,
    require_mlp_sync,
    require_mlp_tp_gather,
)

if TYPE_CHECKING:
    from sglang.srt.speculative.frozen_kv_mtp_worker import FrozenKVMTPWorker


@dataclass
class FrozenKVMTPInputBuffers(ForwardInputBuffers):
    req_pool_indices: torch.Tensor
    positions: torch.Tensor
    mrope_positions: torch.Tensor
    seq_lens: torch.Tensor
    seq_lens_cpu: torch.Tensor
    topk_p: torch.Tensor
    topk_index: torch.Tensor
    hidden_states: torch.Tensor
    global_num_tokens_gpu: Optional[torch.Tensor]
    global_num_tokens_for_logprob_gpu: Optional[torch.Tensor]


@dataclass
class FrozenKVMTPSeedInputBuffers(ForwardInputBuffers):
    """Static input/output buffers for the assistant *seed* step.

    The seed step is a single decode-shape forward of the assistant
    model (one token per request).  Inputs differ from the recurrent
    loop's: there are no ``topk_p`` / ``topk_index`` (these are
    OUTPUTS of the seed); instead we feed ``input_ids`` (the bonus
    token from prefill or verify) and ``hidden_states`` (the last
    target hidden state, sized at the assistant's recurrent
    ``hidden_size``).
    """

    req_pool_indices: torch.Tensor
    positions: torch.Tensor
    mrope_positions: torch.Tensor
    seq_lens: torch.Tensor
    seq_lens_cpu: torch.Tensor
    input_ids: torch.Tensor
    hidden_states: torch.Tensor
    # Outputs that the captured graph writes into; the worker reads
    # them after replay and stitches them onto the new
    # FrozenKVMTPDraftInput for the next iter.
    out_topk_p: torch.Tensor
    out_topk_index: torch.Tensor
    out_hidden_states: torch.Tensor
    global_num_tokens_gpu: Optional[torch.Tensor]
    global_num_tokens_for_logprob_gpu: Optional[torch.Tensor]


class FrozenKVMTPCudaGraphRunner:
    """CUDA graph runner for the Frozen-KV MTP recurrent draft-loop step
    and the assistant seed step.

    The recurrent loop runs ``speculative_num_steps - 1`` forwards of
    the draft model and produces a tree of candidate tokens.  Captured
    via ``_capture_loop_graph`` + ``replay`` (legacy, unchanged).

    The seed step runs ONE forward of the draft model and seeds the
    next iter's ``FrozenKVMTPDraftInput`` with ``topk_p`` / ``topk_index``
    / ``hidden_states``.  It runs after the target's prefill and after
    every verify -- twice per scheduler iter in steady state.  Captured
    via ``_capture_seed_graph`` + ``replay_seed`` (new in this PR).

    Before this PR the seed step ran EAGER, costing ~1 forward worth of
    GPU latency per scheduler iter (~20-25 % of decode wall time at
    ``speculative_num_steps=3``).
    """

    def __init__(self, frozen_kv_mtp_worker: FrozenKVMTPWorker):
        self.frozen_kv_mtp_worker = frozen_kv_mtp_worker
        self.model_runner = model_runner = frozen_kv_mtp_worker.draft_model_runner
        self.graphs = {}
        self.output_buffers = {}
        self.enable_torch_compile = model_runner.server_args.enable_torch_compile
        self.disable_padding = model_runner.server_args.disable_cuda_graph_padding
        self.require_gathered_buffer = require_gathered_buffer(model_runner.server_args)
        self.require_mlp_tp_gather = require_mlp_tp_gather(model_runner.server_args)
        self.require_mlp_sync = require_mlp_sync(model_runner.server_args)
        self.require_attn_tp_gather = require_attn_tp_gather(model_runner.server_args)
        self.tp_size = self.model_runner.tp_size
        self.dp_size = self.model_runner.dp_size
        self.speculative_num_steps = model_runner.server_args.speculative_num_steps
        self.topk = model_runner.server_args.speculative_eagle_topk
        self.draft_attn_backend = frozen_kv_mtp_worker.draft_attn_backend
        self.enable_profile_cuda_graph = (
            model_runner.server_args.enable_profile_cuda_graph
        )
        self.enable_pdmux = False
        self.deepep_adapter = DeepEPCudaGraphRunnerAdapter()

        self.num_tokens_per_bs = self.topk
        self.capture_bs, self.compile_bs = get_batch_sizes_to_capture(
            model_runner, self.num_tokens_per_bs
        )
        self.max_bs = max(self.capture_bs)
        self.max_num_token = self.max_bs * self.num_tokens_per_bs

        self.draft_attn_backend.init_cuda_graph_state(self.max_bs, self.max_num_token)
        self.seq_len_fill_value = (
            self.draft_attn_backend.get_cuda_graph_seq_len_fill_value()
        )
        seq_lens_cpu = torch.full(
            (self.max_num_token,), self.seq_len_fill_value, dtype=torch.int32
        )

        if self.enable_torch_compile:
            set_torch_compile_config()

        with torch.device(model_runner.device):
            req_pool_indices = torch.zeros((self.max_num_token,), dtype=torch.int64)
            positions = torch.zeros((self.max_num_token,), dtype=torch.int64)
            mrope_positions = torch.zeros((3, self.max_num_token), dtype=torch.int64)
            seq_lens = torch.full(
                (self.max_num_token,), self.seq_len_fill_value, dtype=torch.int32
            )
            topk_p = torch.zeros((self.max_bs, self.topk), dtype=torch.float32)
            topk_index = torch.zeros((self.max_bs, self.topk), dtype=torch.int64)
            hidden_states = torch.zeros(
                (self.max_bs, frozen_kv_mtp_worker._recurrent_hidden_size),
                dtype=self.model_runner.dtype,
            )

            if self.require_gathered_buffer:
                if self.require_mlp_tp_gather:
                    global_num_tokens_gpu = torch.zeros(
                        (self.dp_size,), dtype=torch.int32
                    )
                    global_num_tokens_for_logprob_gpu = torch.zeros(
                        (self.dp_size,), dtype=torch.int32
                    )
                else:
                    assert self.require_attn_tp_gather
                    global_num_tokens_gpu = torch.zeros((1,), dtype=torch.int32)
                    global_num_tokens_for_logprob_gpu = torch.zeros(
                        (1,), dtype=torch.int32
                    )
            else:
                global_num_tokens_gpu = None
                global_num_tokens_for_logprob_gpu = None

        self.buffers = FrozenKVMTPInputBuffers(
            req_pool_indices=req_pool_indices,
            positions=positions,
            mrope_positions=mrope_positions,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            topk_p=topk_p,
            topk_index=topk_index,
            hidden_states=hidden_states,
            global_num_tokens_gpu=global_num_tokens_gpu,
            global_num_tokens_for_logprob_gpu=global_num_tokens_for_logprob_gpu,
        )
        self.buffers.share_buffers()

        try:
            with model_capture_mode():
                self.capture()
        except RuntimeError as e:
            raise Exception(
                f"Capture frozen-KV MTP cuda graph failed: {e}\n"
                f"{CUDA_GRAPH_CAPTURE_FAILED_MSG}"
            )

        # ---- Seed-step graphs ---------------------------------------
        # The seed step has different inputs (no topk_p/topk_index; has
        # input_ids + last_hidden_states) and writes its outputs to a
        # second set of static buffers.  Captured at the SAME batch
        # sizes as the recurrent loop, so users get cuda graph coverage
        # whenever the recurrent loop has it too.
        #
        # Set ``SGLANG_FROZEN_KV_MTP_DISABLE_SEED_CG=1`` to skip seed
        # capture and fall back to the pre-PR eager seed path
        # (for A/B benchmarking).
        import logging as _logging
        import os as _os

        _seed_logger = _logging.getLogger(__name__)
        self.seed_graphs: dict = {}
        self.seed_output_buffers: dict = {}
        if _os.environ.get("SGLANG_FROZEN_KV_MTP_DISABLE_SEED_CG", "0") == "1":
            _seed_logger.warning(
                "SGLANG_FROZEN_KV_MTP_DISABLE_SEED_CG=1: skipping seed "
                "cuda graph capture (eager seed path)."
            )
            return
        try:
            _seed_logger.info("Capture Frozen-KV MTP seed cuda graph begin.")
            self._init_seed_buffers()
            with model_capture_mode():
                self._capture_seed()
            _seed_logger.info(
                "Capture Frozen-KV MTP seed cuda graph end (captured %d batch sizes).",
                len(self.seed_graphs),
            )
        except Exception as e:
            # Seed capture failure is recoverable: fall back to eager
            # seed (the pre-PR behavior).  Log loudly so users notice.
            _seed_logger.warning(
                "Capture Frozen-KV MTP seed cuda graph failed: %s\n"
                "Falling back to eager seed step (the recurrent loop graphs are unaffected).",
                e,
            )
            self.seed_graphs = {}
            self.seed_output_buffers = {}

    def _init_seed_buffers(self) -> None:
        """Allocate static input/output buffers for the seed step.

        Seed step is one decode-shape forward of the assistant: one input
        token + last-hidden per request.  The captured graph writes
        ``topk_p``, ``topk_index`` and ``hidden_states`` into the seed
        output buffers; the worker's ``replay_seed`` copies them out
        into a fresh ``FrozenKVMTPDraftInput`` for the next iter.
        """
        worker = self.frozen_kv_mtp_worker
        hidden = worker._recurrent_hidden_size
        # The seed step's expanded_bs is just bs (no topk fanout).
        max_seed_bs = max(self.capture_bs)

        with torch.device(self.model_runner.device):
            req_pool_indices = torch.zeros((max_seed_bs,), dtype=torch.int64)
            positions = torch.zeros((max_seed_bs,), dtype=torch.int64)
            mrope_positions = torch.zeros((3, max_seed_bs), dtype=torch.int64)
            seq_lens = torch.full(
                (max_seed_bs,), self.seq_len_fill_value, dtype=torch.int32
            )
            input_ids = torch.zeros((max_seed_bs,), dtype=torch.int64)
            hidden_states = torch.zeros(
                (max_seed_bs, hidden), dtype=self.model_runner.dtype
            )
            out_topk_p = torch.zeros((max_seed_bs, self.topk), dtype=torch.float32)
            out_topk_index = torch.zeros((max_seed_bs, self.topk), dtype=torch.int64)
            out_hidden_states = torch.zeros(
                (max_seed_bs, hidden), dtype=self.model_runner.dtype
            )

            if self.require_gathered_buffer:
                if self.require_mlp_tp_gather:
                    g_num_tokens = torch.zeros((self.dp_size,), dtype=torch.int32)
                    g_num_tokens_logp = torch.zeros((self.dp_size,), dtype=torch.int32)
                else:
                    g_num_tokens = torch.zeros((1,), dtype=torch.int32)
                    g_num_tokens_logp = torch.zeros((1,), dtype=torch.int32)
            else:
                g_num_tokens = None
                g_num_tokens_logp = None

        seq_lens_cpu = torch.full(
            (max_seed_bs,), self.seq_len_fill_value, dtype=torch.int32
        )

        self.seed_buffers = FrozenKVMTPSeedInputBuffers(
            req_pool_indices=req_pool_indices,
            positions=positions,
            mrope_positions=mrope_positions,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            input_ids=input_ids,
            hidden_states=hidden_states,
            out_topk_p=out_topk_p,
            out_topk_index=out_topk_index,
            out_hidden_states=out_hidden_states,
            global_num_tokens_gpu=g_num_tokens,
            global_num_tokens_for_logprob_gpu=g_num_tokens_logp,
        )
        self.seed_buffers.share_buffers()

    def _capture_seed(self) -> None:
        """Capture the seed-step graph for each batch size in ``capture_bs``."""
        for bs in self.capture_bs:
            graph, out = self._capture_one_seed_graph(bs)
            self.seed_graphs[bs] = graph
            self.seed_output_buffers[bs] = out

    def _capture_one_seed_graph(self, bs: int):
        """Build one captured seed-step graph for batch size ``bs``.

        The captured graph:
          1. Calls ``draft_model_runner.forward()`` on a decode-shape
             ``ForwardBatch`` whose inputs come from
             ``self.seed_buffers``.
          2. Runs ``_capture_for_decode`` to extract topk_p / topk_index
             / hidden_states into the seed output buffers.
        """
        worker = self.frozen_kv_mtp_worker
        buffers = self.seed_buffers
        graph = self._create_graph()
        stream = self.stream

        req_pool_indices = buffers.req_pool_indices[:bs]
        positions = buffers.positions[:bs]
        mrope_positions = buffers.mrope_positions[:, :bs]
        seq_lens = buffers.seq_lens[:bs]
        seq_lens_cpu = buffers.seq_lens_cpu[:bs]
        input_ids = buffers.input_ids[:bs]
        hidden_states = buffers.hidden_states[:bs]
        out_topk_p = buffers.out_topk_p[:bs]
        out_topk_index = buffers.out_topk_index[:bs]
        out_hidden_states = buffers.out_hidden_states[:bs]

        if self.require_mlp_tp_gather:
            buffers.global_num_tokens_gpu.copy_(
                torch.tensor(
                    [bs] * self.dp_size,
                    dtype=torch.int32,
                    device=buffers.positions.device,
                )
            )
            buffers.global_num_tokens_for_logprob_gpu.copy_(
                torch.tensor(
                    [bs] * self.dp_size,
                    dtype=torch.int32,
                    device=buffers.positions.device,
                )
            )
            global_num_tokens = buffers.global_num_tokens_gpu
            global_num_tokens_for_logprob = buffers.global_num_tokens_for_logprob_gpu
            global_dp_buffer_len = bs * self.dp_size
        elif self.require_attn_tp_gather:
            buffers.global_num_tokens_gpu.copy_(
                torch.tensor([bs], dtype=torch.int32, device=buffers.positions.device)
            )
            buffers.global_num_tokens_for_logprob_gpu.copy_(
                torch.tensor([bs], dtype=torch.int32, device=buffers.positions.device)
            )
            global_num_tokens = buffers.global_num_tokens_gpu
            global_num_tokens_for_logprob = buffers.global_num_tokens_for_logprob_gpu
            global_dp_buffer_len = bs
        else:
            global_num_tokens = None
            global_num_tokens_for_logprob = None
            global_dp_buffer_len = None

        # Seed input on the FrozenKVMTPDraftInput is the recurrent
        # ``hidden_states`` (= last target hidden); we install it now
        # and the graph will read it from the static buffer on replay.
        spec_info = FrozenKVMTPDraftInput()
        spec_info.bonus_tokens = input_ids
        spec_info.hidden_states = hidden_states
        spec_info.capture_hidden_mode = CaptureHiddenMode.LAST
        spec_info.num_tokens_per_req = 1
        spec_info.num_tokens_for_logprob_per_req = 1
        spec_info.positions = positions

        forward_batch = ForwardBatch(
            forward_mode=ForwardMode.DECODE,
            batch_size=bs,
            input_ids=input_ids,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            out_cache_loc=None,
            seq_lens_sum=seq_lens.sum().item(),
            return_logprob=False,
            positions=positions,
            mrope_positions=mrope_positions,
            global_num_tokens_gpu=global_num_tokens,
            global_num_tokens_for_logprob_gpu=global_num_tokens_for_logprob,
            dp_padding_mode=DpPaddingMode.get_default_mode_in_cuda_graph(),
            global_dp_buffer_len=global_dp_buffer_len,
            spec_algorithm=self.model_runner.spec_algorithm,
            spec_info=spec_info,
            capture_hidden_mode=CaptureHiddenMode.LAST,
        )

        def run_once_seed():
            if self.model_runner.is_hybrid_swa:
                self.model_runner.token_to_kv_pool.invalidate_loc_cache()
            forward_batch.dp_local_start_pos = forward_batch.dp_local_num_tokens = None
            set_dp_buffer_len(
                global_dp_buffer_len,
                bs,
                forward_batch.dp_padding_mode.is_max_len(),
            )
            set_is_extend_in_batch(False)
            # Run the draft model forward (one decode step).
            logits_output = self.model_runner.forward(
                forward_batch, skip_attn_backend_init=True
            ).logits_output
            # Compute seed topk_p / topk_index / hidden_states and
            # write into our static output buffers (in-place copy).
            probs = torch.softmax(logits_output.next_token_logits, dim=-1)
            from sglang.srt.speculative.frozen_kv_mtp_utils import fast_topk

            seed_topk_p, seed_topk_index = fast_topk(probs, self.topk, dim=-1)
            out_topk_p.copy_(seed_topk_p)
            out_topk_index.copy_(seed_topk_index)
            out_hidden_states.copy_(logits_output.hidden_states)
            return out_topk_p, out_topk_index, out_hidden_states

        # Same target-KV-pool swap protocol as the recurrent capture.
        from sglang.srt.speculative.frozen_kv_mtp_utils import (
            _maybe_swap_swa_state,
            _restore_swa_state,
        )

        target_pool = self.frozen_kv_mtp_worker.kv_context.target_token_to_kv_pool
        saved_backend_pool = self.draft_attn_backend.token_to_kv_pool
        self.draft_attn_backend.token_to_kv_pool = target_pool
        saved_swa_state = _maybe_swap_swa_state(self.draft_attn_backend, target_pool)
        try:
            with forward_context(ForwardContext(attn_backend=self.draft_attn_backend)):
                self.frozen_kv_mtp_worker._init_frozen_kv_metadata_capture_cuda_graph(
                    forward_batch
                )
                self.deepep_adapter.capture(is_extend_in_batch=False)
                self._capture_init(run_once_seed)
                out = self._capture_graph(
                    graph, get_global_graph_memory_pool(), stream, run_once_seed
                )
        finally:
            self.draft_attn_backend.token_to_kv_pool = saved_backend_pool
            _restore_swa_state(self.draft_attn_backend, saved_swa_state)
        set_global_graph_memory_pool(graph.pool())
        return graph, out

    def can_run_seed(self, bs: int) -> bool:
        """True iff a captured seed-step graph exists for ``bs``."""
        if not self.seed_graphs:
            return False
        if self.disable_padding:
            return bs in self.seed_graphs
        return bs <= self.max_bs

    def replay_seed(
        self,
        bs: int,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens_sum: int,
    ):
        """Replay the captured seed-step graph and return the populated
        ``(topk_p, topk_index, hidden_states)`` tensors sized ``[bs, ...]``.

        Caller must ensure ``bs`` is in ``self.capture_bs`` (or padded to
        the nearest captured size up to ``max_bs``) and that the input
        tensors match the shape contract of the captured graph.
        """
        buffers = self.seed_buffers
        # Pick the smallest captured bs >= raw_bs (mirrors recurrent replay).
        index = bisect.bisect_left(self.capture_bs, bs)
        graph_bs = self.capture_bs[index]
        if graph_bs != bs:
            buffers.seq_lens.fill_(self.seq_len_fill_value)
            buffers.positions.zero_()
            buffers.input_ids.zero_()

        buffers.req_pool_indices[:bs].copy_(req_pool_indices)
        buffers.positions[:bs].copy_(positions)
        buffers.seq_lens[:bs].copy_(seq_lens)
        buffers.input_ids[:bs].copy_(input_ids)
        buffers.hidden_states[:bs].copy_(hidden_states)
        if seq_lens_cpu is not None:
            if graph_bs != bs:
                buffers.seq_lens_cpu.fill_(self.seq_len_fill_value)
            buffers.seq_lens_cpu[:bs].copy_(seq_lens_cpu)

        if self.require_gathered_buffer:
            buffers.global_num_tokens_gpu.fill_(graph_bs)
            buffers.global_num_tokens_for_logprob_gpu.fill_(graph_bs)

        # Build a transient ForwardBatch ONLY for the metadata replay
        # call (the graph itself uses the static buffers).
        from sglang.srt.model_executor.forward_batch_info import (
            ForwardBatch as _FB,
            ForwardMode as _FM,
        )

        meta_fb = _FB(
            forward_mode=_FM.DECODE,
            batch_size=graph_bs,
            input_ids=buffers.input_ids[:graph_bs],
            req_pool_indices=buffers.req_pool_indices[:graph_bs],
            seq_lens=buffers.seq_lens[:graph_bs],
            seq_lens_cpu=buffers.seq_lens_cpu[:graph_bs],
            out_cache_loc=None,
            seq_lens_sum=seq_lens_sum + (graph_bs - bs) * self.seq_len_fill_value,
            return_logprob=False,
            positions=buffers.positions[:graph_bs],
            spec_algorithm=self.model_runner.spec_algorithm,
            capture_hidden_mode=CaptureHiddenMode.LAST,
        )
        self.frozen_kv_mtp_worker._init_frozen_kv_metadata_replay_cuda_graph(
            meta_fb, graph_bs, meta_fb.seq_lens_sum
        )

        self.seed_graphs[graph_bs].replay()
        out_topk_p, out_topk_index, out_hidden_states = self.seed_output_buffers[
            graph_bs
        ]
        # Slice back to raw bs for the caller.
        return (
            out_topk_p[:bs].clone(),
            out_topk_index[:bs].clone(),
            out_hidden_states[:bs].clone(),
        )

    def can_run(self, forward_batch: ForwardBatch):
        if self.require_mlp_tp_gather:
            cuda_graph_bs = max(forward_batch.global_num_tokens_cpu) // (
                self.topk * self.topk
            )
        else:
            cuda_graph_bs = (
                forward_batch.batch_size // self.topk
                if self.topk > 1
                else forward_batch.batch_size
            )

        is_bs_supported = (
            cuda_graph_bs in self.graphs
            if self.disable_padding
            else cuda_graph_bs <= self.max_bs
        )
        if self.require_mlp_sync:
            is_bs_supported = is_bs_supported and forward_batch.can_run_dp_cuda_graph
        return is_bs_supported

    def _create_graph(self):
        return torch.cuda.CUDAGraph()

    def _capture_init(self, run_once_fn):
        for _ in range(2):
            torch.cuda.synchronize()
            self.model_runner.tp_group.barrier()
            run_once_fn()

    def _capture_graph(self, graph, pool, stream, run_once_fn):
        with torch.cuda.graph(graph, pool=pool, stream=stream):
            out = run_once_fn()
        return out

    def _replay(self):
        self.graphs[self.bs].replay()

    def capture(self):
        CudaGraphRunner.capture(self)

    def capture_one_batch_size(
        self, num_seqs: int, forward: Callable, stream_idx: int = 0
    ):
        del forward, stream_idx
        buffers = self.buffers
        graph = self._create_graph()
        stream = self.stream
        request_bs = num_seqs
        expanded_bs = request_bs * self.num_tokens_per_bs

        req_pool_indices = buffers.req_pool_indices[:expanded_bs]
        positions = buffers.positions[:expanded_bs]
        mrope_positions = buffers.mrope_positions[:, :expanded_bs]
        seq_lens = buffers.seq_lens[:expanded_bs]
        seq_lens_cpu = buffers.seq_lens_cpu[:expanded_bs]
        topk_p = buffers.topk_p[:request_bs]
        topk_index = buffers.topk_index[:request_bs]
        hidden_states = buffers.hidden_states[:request_bs]

        if self.require_mlp_tp_gather:
            buffers.global_num_tokens_gpu.copy_(
                torch.tensor(
                    [expanded_bs] * self.dp_size,
                    dtype=torch.int32,
                    device=buffers.positions.device,
                )
            )
            buffers.global_num_tokens_for_logprob_gpu.copy_(
                torch.tensor(
                    [expanded_bs] * self.dp_size,
                    dtype=torch.int32,
                    device=buffers.positions.device,
                )
            )
            global_num_tokens = buffers.global_num_tokens_gpu
            global_num_tokens_for_logprob = buffers.global_num_tokens_for_logprob_gpu
            global_dp_buffer_len = expanded_bs * self.dp_size
        elif self.require_attn_tp_gather:
            buffers.global_num_tokens_gpu.copy_(
                torch.tensor(
                    [expanded_bs],
                    dtype=torch.int32,
                    device=buffers.positions.device,
                )
            )
            buffers.global_num_tokens_for_logprob_gpu.copy_(
                torch.tensor(
                    [expanded_bs],
                    dtype=torch.int32,
                    device=buffers.positions.device,
                )
            )
            global_num_tokens = buffers.global_num_tokens_gpu
            global_num_tokens_for_logprob = buffers.global_num_tokens_for_logprob_gpu
            global_dp_buffer_len = expanded_bs
        else:
            global_num_tokens = None
            global_num_tokens_for_logprob = None
            global_dp_buffer_len = None

        spec_info = FrozenKVMTPDraftInput(
            topk_p=topk_p,
            topk_index=topk_index,
            hidden_states=hidden_states,
            capture_hidden_mode=CaptureHiddenMode.LAST,
        )
        spec_info.num_tokens_per_req = self.topk
        spec_info.num_tokens_for_logprob_per_req = self.topk
        spec_info.positions = positions

        forward_batch = ForwardBatch(
            forward_mode=ForwardMode.DECODE,
            batch_size=expanded_bs,
            input_ids=None,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            out_cache_loc=None,
            seq_lens_sum=seq_lens.sum().item(),
            return_logprob=False,
            positions=positions,
            mrope_positions=mrope_positions,
            global_num_tokens_gpu=global_num_tokens,
            global_num_tokens_for_logprob_gpu=global_num_tokens_for_logprob,
            dp_padding_mode=DpPaddingMode.get_default_mode_in_cuda_graph(),
            global_dp_buffer_len=global_dp_buffer_len,
            spec_algorithm=self.model_runner.spec_algorithm,
            spec_info=spec_info,
            capture_hidden_mode=CaptureHiddenMode.LAST,
        )

        def run_once():
            if self.model_runner.is_hybrid_swa:
                self.model_runner.token_to_kv_pool.invalidate_loc_cache()

            forward_batch.dp_local_start_pos = forward_batch.dp_local_num_tokens = None
            set_dp_buffer_len(
                global_dp_buffer_len,
                expanded_bs,
                forward_batch.dp_padding_mode.is_max_len(),
            )
            set_is_extend_in_batch(False)

            hidden_states_backup = forward_batch.spec_info.hidden_states
            ret = self.frozen_kv_mtp_worker.draft_forward(
                forward_batch, skip_attn_backend_init=True
            )
            forward_batch.spec_info.hidden_states = hidden_states_backup
            return ret

        # Swap the draft backend's token_to_kv_pool to the frozen target pool
        # for the capture; the single backend-attr swap is seen by both
        # ``get_token_to_kv_pool()`` (via ``get_attn_backend()``) and the
        # backend's own reads.  Also swap SWA-aware backend state so
        # SWA-aware backends (notably trtllm_mha) build SWA-aware metadata
        # against the target's SWA pool.  See
        # ``frozen_kv_mtp_utils._maybe_swap_swa_state``.
        from sglang.srt.speculative.frozen_kv_mtp_utils import (
            _maybe_swap_swa_state,
            _restore_swa_state,
        )

        target_pool = self.frozen_kv_mtp_worker.kv_context.target_token_to_kv_pool
        saved_backend_pool = self.draft_attn_backend.token_to_kv_pool
        self.draft_attn_backend.token_to_kv_pool = target_pool
        saved_swa_state = _maybe_swap_swa_state(self.draft_attn_backend, target_pool)
        try:
            with forward_context(ForwardContext(attn_backend=self.draft_attn_backend)):
                self.frozen_kv_mtp_worker._init_frozen_kv_metadata_capture_cuda_graph(
                    forward_batch
                )
                self.deepep_adapter.capture(is_extend_in_batch=False)
                self._capture_init(run_once)
                out = self._capture_graph(
                    graph, get_global_graph_memory_pool(), stream, run_once
                )
        finally:
            self.draft_attn_backend.token_to_kv_pool = saved_backend_pool
            _restore_swa_state(self.draft_attn_backend, saved_swa_state)
        set_global_graph_memory_pool(graph.pool())
        return graph, out

    def _postprocess_output_to_raw_bs(self, out, raw_bs):
        parent_list, top_scores_index, draft_tokens = (t[:raw_bs] for t in out)
        return parent_list, top_scores_index, draft_tokens

    def replay(self, forward_batch: ForwardBatch):
        self.deepep_adapter.replay()
        buffers = self.buffers

        raw_expanded_bs = forward_batch.batch_size
        raw_bs = (
            raw_expanded_bs // self.num_tokens_per_bs
            if self.topk > 1
            else raw_expanded_bs
        )
        raw_num_token = raw_expanded_bs

        if self.require_mlp_tp_gather:
            max_num_tokens = max(forward_batch.global_num_tokens_cpu)
            max_batch_size = max_num_tokens // (
                self.num_tokens_per_bs * self.num_tokens_per_bs
            )
            index = bisect.bisect_left(self.capture_bs, max_batch_size)
        else:
            index = bisect.bisect_left(self.capture_bs, raw_bs)

        bs = self.capture_bs[index]
        expanded_bs = bs * self.num_tokens_per_bs
        if bs != raw_bs:
            buffers.seq_lens.fill_(self.seq_len_fill_value)
            buffers.positions.zero_()

        num_tokens = expanded_bs
        buffers.seq_lens[:raw_expanded_bs].copy_(forward_batch.seq_lens)
        buffers.positions[:raw_num_token].copy_(forward_batch.positions)
        if forward_batch.mrope_positions is not None:
            buffers.mrope_positions[:, :raw_num_token].copy_(
                forward_batch.mrope_positions
            )
        buffers.topk_p[:raw_bs].copy_(forward_batch.spec_info.topk_p)
        buffers.topk_index[:raw_bs].copy_(forward_batch.spec_info.topk_index)
        buffers.hidden_states[:raw_bs].copy_(forward_batch.spec_info.hidden_states)
        buffers.req_pool_indices[:raw_expanded_bs].copy_(forward_batch.req_pool_indices)

        if self.require_gathered_buffer:
            buffers.global_num_tokens_gpu.fill_(expanded_bs)
            buffers.global_num_tokens_for_logprob_gpu.fill_(expanded_bs)

        if bs != raw_bs:
            forward_batch.batch_size = expanded_bs
            forward_batch.seq_lens = buffers.seq_lens[:expanded_bs]
            forward_batch.req_pool_indices = buffers.req_pool_indices[:expanded_bs]
            forward_batch.positions = buffers.positions[:num_tokens]
            if forward_batch.mrope_positions is not None:
                forward_batch.mrope_positions = buffers.mrope_positions[:, :num_tokens]

        if forward_batch.seq_lens_cpu is not None:
            if bs != raw_bs:
                buffers.seq_lens_cpu.fill_(self.seq_len_fill_value)
            buffers.seq_lens_cpu[:raw_expanded_bs].copy_(forward_batch.seq_lens_cpu)
            forward_batch.seq_lens_cpu = buffers.seq_lens_cpu[:expanded_bs]

        self.frozen_kv_mtp_worker._init_frozen_kv_metadata_replay_cuda_graph(
            forward_batch,
            expanded_bs,
            forward_batch.seq_lens_sum
            + (expanded_bs - raw_expanded_bs) * self.seq_len_fill_value,
        )

        self.raw_bs = raw_bs
        self.bs = bs
        self._replay()
        out = self.output_buffers[bs]

        if bs != raw_bs:
            out = self._postprocess_output_to_raw_bs(out, raw_bs)
            forward_batch.batch_size = raw_expanded_bs
            forward_batch.positions = buffers.positions[:raw_num_token]
            forward_batch.seq_lens = buffers.seq_lens[:raw_expanded_bs]
            forward_batch.req_pool_indices = buffers.req_pool_indices[:raw_expanded_bs]
            if forward_batch.mrope_positions is not None:
                forward_batch.mrope_positions = buffers.mrope_positions[
                    :, :raw_num_token
                ]
            if forward_batch.seq_lens_cpu is not None:
                forward_batch.seq_lens_cpu = buffers.seq_lens_cpu[:raw_expanded_bs]

        return out
