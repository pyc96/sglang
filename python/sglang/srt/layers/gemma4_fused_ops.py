"""Fused triton kernels for Gemma4 decoder layer operations.

Fuses standard RMSNorm + residual-add (+ optional scalar multiply) into
a single kernel pass to reduce kernel launch overhead.

Also provides a single-launch fused router for Gemma4 MoE (PR #26120 in
pyc96/sglang fork): replaces the per-layer ``torch.topk`` ->
``softmax`` -> ``per_expert_scale[ids]`` -> ``mul`` -> ``cast`` chain in
``Gemma4MoE.routing_function`` with one Triton kernel.

The reference design comes from vLLM PR #39083
(``_gemma4_routing_kernel`` / ``gemma4_fused_routing_kernel_triton``),
which is apache-2.0.  Our kernel is rewritten in SGLang style and uses
the identity ``softmax(all)[topk] / sum(softmax(all)[topk]) =
softmax(topk_logits)`` already exploited by SGLang's torch routing
function, so the math is bitwise-comparable to the prior fp32 path.
"""

from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _gemma_rmsnorm_residual_kernel(
    X_ptr,
    W_ptr,
    Residual_ptr,
    Scalar_ptr,
    Out_ptr,
    stride_x,
    stride_r,
    stride_o,
    N,
    eps,
    HAS_SCALAR: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused kernel: out = rmsnorm(x, w) + residual [* scalar]

    When HAS_SCALAR is True, also multiplies by a scalar loaded from Scalar_ptr.
    """
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(Residual_ptr + row * stride_r + cols, mask=mask, other=0.0).to(
        tl.float32
    )

    var = tl.sum(x * x, axis=0) / N
    rrms = tl.rsqrt(var + eps)
    out = x * rrms * w + r

    if HAS_SCALAR:
        scalar = tl.load(Scalar_ptr).to(tl.float32)
        out = out * scalar

    tl.store(Out_ptr + row * stride_o + cols, out.to(x.dtype), mask=mask)


def gemma_rmsnorm_residual_scalar(
    x: torch.Tensor,
    weight: torch.Tensor,
    residual: torch.Tensor,
    scalar: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Fused (rmsnorm(x) + residual) * scalar."""
    assert x.dim() == 2 and x.stride(-1) == 1, "Expected contiguous 2D input"
    M, N = x.shape
    BLOCK_SIZE = triton.next_power_of_2(N)
    out = torch.empty_like(x)

    _gemma_rmsnorm_residual_kernel[(M,)](
        x,
        weight,
        residual,
        scalar,
        out,
        x.stride(0),
        residual.stride(0),
        out.stride(0),
        N,
        eps,
        HAS_SCALAR=True,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


def gemma4_arf_rmsnorm_residual_scalar(
    x: torch.Tensor,
    weight: torch.Tensor,
    residual: torch.Tensor,
    scalar: torch.Tensor,
    eps: float = 1e-6,
    use_attn_tp_group: bool = True,
) -> torch.Tensor:
    """Fused TP all-reduce + (rmsnorm(x) + residual) * scalar for Gemma-4
    dense post-FF combine.

    Numerically equivalent to::

        x_reduced = tensor_model_parallel_all_reduce(x)
        return gemma_rmsnorm_residual_scalar(x_reduced, weight, residual, scalar, eps)

    but, when FlashInfer's fused AllReduce+RMSNorm pattern is applicable on
    this step (Hopper/Blackwell, ``--enable-flashinfer-allreduce-fusion``,
    batch <= ``FUSE_ALLREDUCE_MAX_BATCH_SIZE``, workspace healthy, etc.),
    collapses the TP all-reduce and the residual-add+RMSNorm into a single
    TRT-LLM communication kernel that overlaps the collective with the norm
    math.  The final ``* scalar`` tail runs as a one-launch broadcast mul
    (cheap; vectorized point-wise op).

    Caller contract:
      * The caller is responsible for passing ``skip_all_reduce=True`` to
        the upstream ``RowParallelLinear`` whose output is ``x`` so the
        all-reduce is not double-counted.
      * ``x`` must be the still-TP-sharded output of that ``down_proj``
        (i.e. the value RowParallelLinear would have all-reduced).
      * ``residual`` is the full pre-FF hidden state (already replicated).
      * ``scalar`` is the Gemma-4 ``layer_scalar`` persistent buffer
        (shape ``[1]``).
      * ``use_attn_tp_group=True`` selects the attention-TP group's
        FlashInfer workspace; for Gemma-4 (no DP-attn, no MoE-TP split)
        this is the full TP group.

    When the fused path is not applicable, falls back to the explicit
    ``tensor_model_parallel_all_reduce`` + ``gemma_rmsnorm_residual_scalar``
    sequence with bit-identical semantics to the pre-fusion code path.
    """
    # Lazy imports to avoid pulling in distributed/communicator at module
    # load time (matches the convention used by other call sites of
    # ``flashinfer_allreduce_residual_rmsnorm`` in SGLang).
    from sglang.srt.distributed import tensor_model_parallel_all_reduce
    from sglang.srt.layers.communicator import apply_flashinfer_allreduce_fusion
    from sglang.srt.layers.flashinfer_comm_fusion import (
        flashinfer_allreduce_residual_rmsnorm,
    )

    if x.is_cuda and x.dim() == 2 and apply_flashinfer_allreduce_fusion(x.shape[0]):
        norm_out, _residual_out = flashinfer_allreduce_residual_rmsnorm(
            input_tensor=x,
            residual=residual,
            weight=weight,
            eps=eps,
            use_attn_tp_group=use_attn_tp_group,
        )
        if norm_out is not None:
            # FlashInfer succeeded; apply the Gemma-4 layer_scalar tail.
            # The mul is fused by the eager bf16 elementwise path; one
            # extra launch on top of the fused AR+RMSNorm.  ``scalar`` is
            # shape ``[1]`` so broadcasting is free.
            return norm_out * scalar

    # Fallback: identical to the pre-fusion code path.
    x_reduced = tensor_model_parallel_all_reduce(x)
    return gemma_rmsnorm_residual_scalar(x_reduced, weight, residual, scalar, eps)


def gemma4_arf_rmsnorm_only(
    x: torch.Tensor,
    norm_module,
    use_attn_tp_group: bool = True,
) -> torch.Tensor:
    """Fused TP all-reduce + single-arg RMSNorm for Gemma-4
    ``post_attention_layernorm``.

    Numerically equivalent to::

        x_reduced = tensor_model_parallel_all_reduce(x)
        return norm_module.forward(x_reduced)

    where ``norm_module`` is a standard SGLang ``RMSNorm`` whose math is
    ``rmsnorm(x) * weight``.  This wrapper is the **correct fusion site**
    for Gemma-4's residual flow because Gemma-4 places a single-arg
    RMSNorm immediately after the attention all-reduce (before any
    residual addition).

    Why the zero-residual trick:
      FlashInfer's TRT-LLM ``allreduce_fusion`` API only exposes the
      ``kARResidualRMSNorm`` pattern (no residual-less variant).  vLLM's
      ``AllReduceRMSNormPattern`` solves this by synthesizing a
      ``torch.zeros_like(input)`` residual; the math
      ``rmsnorm(AR(x) + 0) == rmsnorm(AR(x))`` makes the residual
      contribution vanish.  We follow the same convention here.

    Caller contract:
      * Caller must pass ``skip_all_reduce=True`` to the upstream
        ``RowParallelLinear`` whose output is ``x``.
      * ``x`` must be the still-TP-sharded post-attention projection.
      * ``norm_module`` is the Gemma-4 layer's
        ``post_attention_layernorm`` (a ``RMSNorm`` instance — *not* a
        ``Gemma4RMSNorm``, because the latter's ``(weight + scale_shift)``
        gamma is not currently expressible in FlashInfer's pattern).

    Fallback: when FlashInfer is unavailable, batch too large, workspace
    not ready, or the predicate is False, falls back to
    ``tensor_model_parallel_all_reduce(x) + norm_module.forward(_)`` with
    bit-identical semantics to the pre-fusion path.
    """
    from sglang.srt.distributed import tensor_model_parallel_all_reduce
    from sglang.srt.layers.communicator import apply_flashinfer_allreduce_fusion
    from sglang.srt.layers.flashinfer_comm_fusion import (
        flashinfer_allreduce_residual_rmsnorm,
    )

    if x.is_cuda and x.dim() == 2 and apply_flashinfer_allreduce_fusion(x.shape[0]):
        zero_residual = torch.zeros_like(x)
        norm_out, _residual_out = flashinfer_allreduce_residual_rmsnorm(
            input_tensor=x,
            residual=zero_residual,
            weight=norm_module.weight.data,
            eps=norm_module.variance_epsilon,
            use_attn_tp_group=use_attn_tp_group,
        )
        if norm_out is not None:
            return norm_out

    # Fallback: identical to the pre-fusion code path.
    x_reduced = tensor_model_parallel_all_reduce(x)
    return norm_module.forward(x_reduced)


@triton.jit
def _gemma_dual_rmsnorm_residual_kernel(
    X1_ptr,
    W1_ptr,
    X2_ptr,
    W2_ptr,
    W3_ptr,
    Residual_ptr,
    Scalar_ptr,
    Out_ptr,
    stride_x1,
    stride_x2,
    stride_r,
    stride_o,
    N,
    eps1,
    eps2,
    eps3,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused: out = (rmsnorm(rmsnorm(x1,w1) + rmsnorm(x2,w2), w3) + residual) * scalar"""
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    x1 = tl.load(X1_ptr + row * stride_x1 + cols, mask=mask, other=0.0).to(tl.float32)
    w1 = tl.load(W1_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    x2 = tl.load(X2_ptr + row * stride_x2 + cols, mask=mask, other=0.0).to(tl.float32)
    w2 = tl.load(W2_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    w3 = tl.load(W3_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(Residual_ptr + row * stride_r + cols, mask=mask, other=0.0).to(
        tl.float32
    )

    var1 = tl.sum(x1 * x1, axis=0) / N
    norm1 = x1 * tl.rsqrt(var1 + eps1) * w1

    var2 = tl.sum(x2 * x2, axis=0) / N
    norm2 = x2 * tl.rsqrt(var2 + eps2) * w2

    combined = norm1 + norm2

    var3 = tl.sum(combined * combined, axis=0) / N
    norm3 = combined * tl.rsqrt(var3 + eps3) * w3

    scalar = tl.load(Scalar_ptr).to(tl.float32)
    out = (norm3 + r) * scalar

    tl.store(Out_ptr + row * stride_o + cols, out.to(x1.dtype), mask=mask)


@triton.jit
def _gemma_qkv_rmsnorm_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    Q_w_ptr,
    K_w_ptr,
    stride_q_m,
    stride_k_m,
    stride_v_m,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    eps,
    HAS_KV: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Per-token fused RMSNorm of Q (with q_w), K (with k_w), V (no scale).

    Layout assumption: each tensor's last dim packs (num_heads, head_dim) contiguously
    so per-head offset is `h * HEAD_DIM`. The token (M) stride is taken from
    stride_*_m so the kernel works on strided views (e.g. slices of a larger
    qkv buffer produced by `qkv.split`) without requiring `.contiguous()` copies.
    V uses `weight=ones` semantics so the multiply-by-weight is omitted.
    """
    m = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < HEAD_DIM

    qw = tl.load(Q_w_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    # Q heads
    for h in tl.static_range(NUM_Q_HEADS):
        off = m * stride_q_m + h * HEAD_DIM + cols
        x = tl.load(Q_ptr + off, mask=mask, other=0.0).to(tl.float32)
        rrms = tl.rsqrt(tl.sum(x * x, axis=0) / HEAD_DIM + eps)
        out = x * rrms * qw
        tl.store(Q_ptr + off, out.to(Q_ptr.dtype.element_ty), mask=mask)

    if HAS_KV:
        kw = tl.load(K_w_ptr + cols, mask=mask, other=0.0).to(tl.float32)

        # K heads
        for h in tl.static_range(NUM_KV_HEADS):
            off = m * stride_k_m + h * HEAD_DIM + cols
            x = tl.load(K_ptr + off, mask=mask, other=0.0).to(tl.float32)
            rrms = tl.rsqrt(tl.sum(x * x, axis=0) / HEAD_DIM + eps)
            out = x * rrms * kw
            tl.store(K_ptr + off, out.to(K_ptr.dtype.element_ty), mask=mask)

        # V heads (no scaling: V-norm uses weight=ones)
        for h in tl.static_range(NUM_KV_HEADS):
            off = m * stride_v_m + h * HEAD_DIM + cols
            x = tl.load(V_ptr + off, mask=mask, other=0.0).to(tl.float32)
            rrms = tl.rsqrt(tl.sum(x * x, axis=0) / HEAD_DIM + eps)
            out = x * rrms
            tl.store(V_ptr + off, out.to(V_ptr.dtype.element_ty), mask=mask)


def gemma_qkv_rmsnorm(
    q: torch.Tensor,
    k: Optional[torch.Tensor],
    v: Optional[torch.Tensor],
    q_weight: torch.Tensor,
    k_weight: Optional[torch.Tensor],
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    eps: float = 1e-6,
) -> None:
    """In-place fused RMSNorm on Q, K, V for Gemma4 attention.

    All three norms compute `x * rsqrt(mean(x^2) + eps)` independently per head.
    Q is scaled by `q_weight`, K by `k_weight`, V by 1 (Gemma4's V-norm has
    `with_scale=False`).

    Inputs may be 2D `(M, num_heads * head_dim)` or strided views of a larger
    buffer (such as q/k/v slices from `qkv.split`). The kernel uses the actual
    `stride(0)` so no `.contiguous()` copy is required. Within a token, the
    last dim must be contiguous so heads pack as `h * head_dim` offsets.

    If k and v are both None (KV-shared layer), only Q is normalized.
    """
    assert q.is_cuda
    assert q.stride(-1) == 1, "Q's last dim must be contiguous"
    assert q_weight.shape[-1] == head_dim
    M = q.shape[0] if q.dim() >= 2 else 1
    BLOCK = triton.next_power_of_2(head_dim)

    has_kv = k is not None and v is not None
    if has_kv:
        assert k.is_cuda and v.is_cuda
        assert k.stride(-1) == 1 and v.stride(-1) == 1
        assert k_weight is not None and k_weight.shape[-1] == head_dim

    _gemma_qkv_rmsnorm_kernel[(M,)](
        q,
        k if has_kv else q,
        v if has_kv else q,
        q_weight,
        k_weight if has_kv else q_weight,
        q.stride(0),
        k.stride(0) if has_kv else 0,
        v.stride(0) if has_kv else 0,
        NUM_Q_HEADS=num_q_heads,
        NUM_KV_HEADS=num_kv_heads if has_kv else 0,
        HEAD_DIM=head_dim,
        eps=eps,
        HAS_KV=has_kv,
        BLOCK=BLOCK,
    )


def gemma_dual_rmsnorm_residual_scalar(
    x1: torch.Tensor,
    weight1: torch.Tensor,
    x2: torch.Tensor,
    weight2: torch.Tensor,
    weight3: torch.Tensor,
    residual: torch.Tensor,
    scalar: torch.Tensor,
    eps1: float = 1e-6,
    eps2: float = 1e-6,
    eps3: float = 1e-6,
) -> torch.Tensor:
    """Fused (rmsnorm(rmsnorm(x1,w1) + rmsnorm(x2,w2), w3) + residual) * scalar."""
    assert x1.dim() == 2 and x1.stride(-1) == 1
    M, N = x1.shape
    BLOCK_SIZE = triton.next_power_of_2(N)
    out = torch.empty_like(x1)

    _gemma_dual_rmsnorm_residual_kernel[(M,)](
        x1,
        weight1,
        x2,
        weight2,
        weight3,
        residual,
        scalar,
        out,
        x1.stride(0),
        x2.stride(0),
        residual.stride(0),
        out.stride(0),
        N,
        eps1,
        eps2,
        eps3,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


# ---------------------------------------------------------------------------
# Fused Gemma4 routing kernel (one launch per layer)
# ---------------------------------------------------------------------------
#
# Equivalent to:
#
#     topk_logits, topk_ids = torch.topk(gating_output, k=topk, dim=-1)
#     topk_weights = torch.nn.functional.softmax(topk_logits, dim=-1)
#     topk_weights = topk_weights * per_expert_scale[topk_ids]
#     return topk_weights.float(), topk_ids.int()
#
# but completes the entire computation in one Triton program per token.
#
# Algorithm notes:
#   * Loads all E logits per token into one program; for Gemma4
#     ``E = num_experts = 128`` so ``BLOCK_E = next_pow2(E) = 128`` and the
#     work fits in a single warp with `num_warps=1`.
#   * Computes ``softmax-of-topk`` by:
#       - using ``tl.sort`` on (logit_bits_as_sortable_uint, expert_id) pairs
#         packed into int64 — this gives a fully vectorized top-K without a
#         K-step loop and matches the bitwise behavior of ``torch.topk``.
#       - taking the largest K via a mask on the sorted-descending sequence
#       - normalizing in fp32 (matches ``softmax`` default dtype)
#       - multiplying by ``per_expert_scale[topk_ids]``
#   * Writes ``topk_weights`` (fp32) and ``topk_ids`` (int32) in one
#     pass, matching the output dtypes the SGLang MoE topk wrapper
#     expects.
#
# Reference algorithm: vLLM PR #39083 ``_gemma4_routing_kernel`` (apache-2.0).
# Our independent implementation follows the same sort+mask+softmax scheme.
@triton.jit
def _gemma4_routing_kernel(
    gating_ptr,  # [T, E] router logits, any float dtype
    per_expert_scale_ptr,  # [E] per-expert scale (any float dtype)
    topk_weights_ptr,  # [T, K] fp32 out
    topk_ids_ptr,  # [T, K] int32 out
    stride_g_t,  # stride of gating in the token dim
    E: tl.constexpr,
    K: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_e = tl.arange(0, BLOCK_E)
    valid = offs_e < E

    # Load logits into fp32; out-of-bound lanes get -inf so they sort last.
    logits = tl.load(
        gating_ptr + pid * stride_g_t + offs_e,
        mask=valid,
        other=-float("inf"),
    ).to(tl.float32)

    # Build a sortable int64 key: high 32 bits = bijective(logit_bits) so
    # ascending-int sort == ascending-float sort; low 32 bits = expert id
    # (kept stable for ties matching torch.topk's default behavior).  This
    # avoids a separate index buffer / scatter pass after the sort.
    MIN32 = -2147483648
    logit_bits = logits.to(tl.int32, bitcast=True)
    sign = logit_bits >> 31
    key = tl.where(sign == 0, logit_bits ^ -1, logit_bits ^ MIN32)
    # Force invalid lanes to the max positive key so they end up *after* the
    # real logits when we sort ascending and read from the top of the
    # reversed list. (descending=True would flip the order.)
    key = tl.where(valid, key, 0x7FFFFFFF)
    sk64 = key.to(tl.int64) & 0x00000000FFFFFFFF
    packed = (sk64 << 32) | offs_e.to(tl.int64)

    # Sort ascending; the K smallest keys correspond to the K largest
    # logits because of the bijection above.
    sorted_p = tl.sort(packed, descending=False)
    all_keys = ((sorted_p >> 32) & 0x00000000FFFFFFFF).to(tl.int32)
    all_ids = (sorted_p & 0x00000000FFFFFFFF).to(tl.int32)

    # Invert the bijection to recover the original logit value.
    sign_k = all_keys >> 31
    all_bits = tl.where(sign_k < 0, all_keys ^ -1, all_keys ^ MIN32)
    all_logits = all_bits.to(tl.float32, bitcast=True)

    # Softmax over the K largest logits only (identity proven by SGLang's
    # torch routing function comment).  Subtract the max for stability;
    # since the list is sorted descending by logit value, the max sits at
    # index 0.
    top_mask = offs_e < K
    max_l = tl.max(tl.where(top_mask, all_logits, -float("inf")), axis=0)
    # exp2(x * log2(e)) is what tl.math.exp expands to; spell it out so we
    # can tolerate older Triton releases that lack tl.math.exp.
    raw_exp = tl.math.exp2((all_logits - max_l) * 1.4426950408889634)
    raw_exp = tl.where(top_mask, raw_exp, 0.0)

    denom = tl.sum(raw_exp, axis=0)
    denom = tl.where(denom > 0.0, denom, 1.0)
    weights = raw_exp / denom

    # Multiply by per_expert_scale[topk_ids].  per_expert_scale lives in
    # any float dtype; cast to fp32 for the final write.
    scales = tl.load(
        per_expert_scale_ptr + all_ids.to(tl.int64),
        mask=top_mask,
        other=1.0,
    ).to(tl.float32)
    weights = weights * scales

    base_off = pid * K + offs_e
    tl.store(topk_weights_ptr + base_off, weights, mask=top_mask)
    tl.store(topk_ids_ptr + base_off, all_ids, mask=top_mask)


def gemma4_fused_routing(
    gating_output: torch.Tensor,
    per_expert_scale: torch.Tensor,
    topk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One-launch Gemma4 router.

    Args:
        gating_output: [T, E] router logits in any floating dtype; will be
            cast to fp32 inside the kernel.
        per_expert_scale: [E] per-expert scale, any floating dtype.
        topk: number of experts to keep per token.

    Returns:
        topk_weights: [T, topk] fp32 (matches SGLang TopK contract).
        topk_ids: [T, topk] int32 (matches SGLang TopK contract).
    """
    assert gating_output.dim() == 2, "expected [T, E] router logits"
    assert per_expert_scale.dim() == 1
    assert per_expert_scale.shape[0] == gating_output.shape[1]
    T, E = gating_output.shape
    assert topk <= E

    # The kernel reads the token row with stride_g_t; force the inner-most
    # dim to be contiguous so the masked load is coalesced.  Most call
    # sites already pass a contiguous tensor (router proj output); contiguous
    # is cheap.
    gating_output = gating_output.contiguous()
    per_expert_scale = per_expert_scale.contiguous()

    BLOCK_E = triton.next_power_of_2(E)
    topk_weights = torch.empty(
        (T, topk), dtype=torch.float32, device=gating_output.device
    )
    topk_ids = torch.empty((T, topk), dtype=torch.int32, device=gating_output.device)

    if T == 0:
        return topk_weights, topk_ids

    _gemma4_routing_kernel[(T,)](
        gating_output,
        per_expert_scale,
        topk_weights,
        topk_ids,
        gating_output.stride(0),
        E,
        topk,
        BLOCK_E,
        num_warps=1,
    )
    return topk_weights, topk_ids
