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


# ---------------------------------------------------------------------------
# Fused ops for the Per-Layer-Embedding (PLE) tail of Gemma4 E2B / E4B.
#
# The slow path in Gemma4DecoderLayer.forward (the PLE branch, taken when
# `has_ple=True`) used to issue 7 separate kernels at the end of every layer
# (post_ff_norm; add residual; gate gelu; mul ple; project; norm; add+mul).
# Two of those (the gate and projection GEMMs) are unavoidable, but the
# remaining 5 are pointwise across the per-token dim and can be collapsed
# into 3 Triton launches:
#
#   `gemma_rmsnorm_add`        : out = rmsnorm(x, w) + r
#   `gemma_gelu_tanh_mul`      : out = gelu_tanh(gate) * per_layer_input
#   `gemma_rmsnorm_residual_scalar` (already defined above) for the tail
#
# This saves ~4 kernel launches per layer * num_layers per decode step.
# ---------------------------------------------------------------------------


@triton.jit
def _gemma_rmsnorm_add_kernel(
    X_ptr,
    W_ptr,
    Residual_ptr,
    Out_ptr,
    stride_x,
    stride_r,
    stride_o,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused kernel: out = rmsnorm(x, w) + residual.

    Identical to `_gemma_rmsnorm_residual_kernel` with HAS_SCALAR=False.
    Hoisted into its own kernel so the caller doesn't pay for the
    `tl.load(Scalar_ptr)` of a unit scalar.
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
    out = x * tl.rsqrt(var + eps) * w + r
    tl.store(Out_ptr + row * stride_o + cols, out.to(x.dtype), mask=mask)


def gemma_rmsnorm_add(
    x: torch.Tensor,
    weight: torch.Tensor,
    residual: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Fused (rmsnorm(x, w) + residual) — no scalar multiply."""
    assert x.dim() == 2 and x.stride(-1) == 1, "Expected contiguous 2D input"
    M, N = x.shape
    BLOCK_SIZE = triton.next_power_of_2(N)
    out = torch.empty_like(x)

    _gemma_rmsnorm_add_kernel[(M,)](
        x,
        weight,
        residual,
        out,
        x.stride(0),
        residual.stride(0),
        out.stride(0),
        N,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


@triton.jit
def _gemma_gelu_tanh_mul_kernel(
    Gate_ptr,
    Ple_ptr,
    Out_ptr,
    stride_g,
    stride_p,
    stride_o,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused kernel: out = gelu_tanh(gate) * per_layer_input."""
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    gate = tl.load(Gate_ptr + row * stride_g + cols, mask=mask, other=0.0).to(
        tl.float32
    )
    ple = tl.load(Ple_ptr + row * stride_p + cols, mask=mask, other=0.0).to(tl.float32)

    # GeLU with tanh approximation:
    #   0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    SQRT_2_OVER_PI = 0.7978845608028654  # sqrt(2 / pi)
    inner = SQRT_2_OVER_PI * (gate + 0.044715 * gate * gate * gate)
    gelu = 0.5 * gate * (1.0 + tl.extra.libdevice.tanh(inner))

    out = gelu * ple
    tl.store(Out_ptr + row * stride_o + cols, out.to(gate.dtype), mask=mask)


def gemma_gelu_tanh_mul(
    gate: torch.Tensor,
    per_layer_input: torch.Tensor,
) -> torch.Tensor:
    """Fused (gelu_tanh(gate) * per_layer_input) — pointwise."""
    assert gate.dim() == 2 and gate.stride(-1) == 1, "Expected contiguous 2D gate"
    assert (
        per_layer_input.dim() == 2 and per_layer_input.stride(-1) == 1
    ), "Expected contiguous 2D per_layer_input"
    assert gate.shape == per_layer_input.shape, "gate / ple must match"
    M, N = gate.shape
    BLOCK_SIZE = triton.next_power_of_2(N)
    out = torch.empty_like(gate)

    _gemma_gelu_tanh_mul_kernel[(M,)](
        gate,
        per_layer_input,
        out,
        gate.stride(0),
        per_layer_input.stride(0),
        out.stride(0),
        N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


# ---------------------------------------------------------------------------
# Triple-RMSNorm-with-shared-residual kernel (the MoE-branch pre-MLP block).
#
# Ports vLLM Inductor's ``triton_red_fused_add_moe_forward_mul_rms_norm_0``
# (captured from a torch.compile/Inductor run on Gemma-4-26B-A4B-IT). The
# pattern Inductor discovered:
#
#   1) post_attn_residual = rmsnorm(attn_out, w_post_attn) + residual_before
#   2) dense_ff_in        = rmsnorm(post_attn_residual, w_pre_ff)
#   3) router_in          = rmsnorm(post_attn_residual, ones) * router_scale
#   4) moe_in             = rmsnorm(post_attn_residual, w_pre_ff_2)
#
# Steps 2, 3 and 4 share the SAME ``rsqrt(variance(post_attn_residual))``;
# Inductor reuses the reduction across all three outputs. Doing the same
# in a hand-rolled Triton kernel lets us emit one launch instead of 3-4
# launches (post_attn_rmsnorm; pre_ff_rmsnorm_with_add; router_norm;
# pre_ff_2_rmsnorm) without depending on torch.compile.
#
# The kernel applies the classic 3-pass-reduction layout the Inductor
# kernel uses:
#   pass 1: variance(attn_out)              -> rsqrt for the first rmsnorm
#   pass 2: variance(rmsnorm(attn_out)+res) -> rsqrt shared by 3 outputs
#   pass 3: produce the 3 scaled outputs and the updated residual
#
# Pre-condition: with_scale=False for the router norm (true for Gemma4
# Gemma4Router). ``router_scale_per_dim`` MUST already be folded with
# the root_size (i.e. callers pass router._fused_scale, which is
# scale * hidden_size^{-0.5}).
# ---------------------------------------------------------------------------


@triton.jit
def _gemma_post_attn_triple_rmsnorm_kernel(
    Attn_ptr,  # in_ptr0 : [bs, H] bf16
    PostAttnW_ptr,  # in_ptr1 : [H]    bf16   - post_attention_layernorm weight
    Residual_ptr,  # in_ptr2 : [bs, H] bf16   - pre-attention residual (input_layernorm input)
    RouterScale_ptr,  # in_ptr3 : [H]    bf16   - router._fused_scale (= scale * root_size)
    PreFFW_ptr,  # in_ptr4 : [H]    bf16   - pre_feedforward_layernorm weight
    PreFF2W_ptr,  # in_ptr5 : [H]    bf16   - pre_feedforward_layernorm_2 weight (MoE)
    PostAttnResOut_ptr,  # out_ptr0: [bs, H] bf16   - updated residual (= rmsnorm(attn_out)+res)
    RouterIn_ptr,  # out_ptr1: [bs, H] bf16
    DenseFFIn_ptr,  # out_ptr2: [bs, H] bf16
    MoeIn_ptr,  # out_ptr3: [bs, H] bf16
    stride_attn,
    stride_res,
    stride_par,
    stride_rin,
    stride_dfn,
    stride_min,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    # ---------------- Pass 1: variance(attn_out) -----------------------------
    a = tl.load(Attn_ptr + row * stride_attn + cols, mask=mask, other=0.0).to(
        tl.float32
    )
    var_a = tl.sum(a * a, axis=0) / N
    rsqrt_a = tl.rsqrt(var_a + eps)

    # ---------------- Pass 2: build post_attn_residual; variance -------------
    # rmsnorm(attn_out, w_post_attn) + residual
    w_post = tl.load(PostAttnW_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    res = tl.load(Residual_ptr + row * stride_res + cols, mask=mask, other=0.0).to(
        tl.float32
    )
    post_attn_res = (a * rsqrt_a * w_post) + res
    var_par = tl.sum(post_attn_res * post_attn_res, axis=0) / N
    rsqrt_par = tl.rsqrt(var_par + eps)

    # ---------------- Pass 3: produce all three outputs ----------------------
    # base = rmsnorm(post_attn_res, ones) — shared by all three.
    base = post_attn_res * rsqrt_par

    rscale = tl.load(RouterScale_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    wff = tl.load(PreFFW_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    wff2 = tl.load(PreFF2W_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    router_out = base * rscale
    dense_out = base * wff
    moe_out_val = base * wff2

    # Store. The updated residual is also written so subsequent layers can
    # read it (downstream code expects the pre-attn residual to be the
    # post_attn rmsnorm output added to the prior residual).
    out_dtype = tl.bfloat16
    tl.store(
        PostAttnResOut_ptr + row * stride_par + cols,
        post_attn_res.to(out_dtype),
        mask=mask,
    )
    tl.store(
        RouterIn_ptr + row * stride_rin + cols, router_out.to(out_dtype), mask=mask
    )
    tl.store(
        DenseFFIn_ptr + row * stride_dfn + cols, dense_out.to(out_dtype), mask=mask
    )
    tl.store(MoeIn_ptr + row * stride_min + cols, moe_out_val.to(out_dtype), mask=mask)


def gemma_post_attn_triple_rmsnorm(
    attn_out: torch.Tensor,
    post_attn_weight: torch.Tensor,
    residual_before_attn: torch.Tensor,
    router_fused_scale: torch.Tensor,
    pre_ff_weight: torch.Tensor,
    pre_ff_2_weight: torch.Tensor,
    eps: float = 1e-6,
):
    """Fused launcher for the MoE-branch pre-MLP block.

    Returns ``(post_attn_residual, router_input, dense_ff_input, moe_input)``.

    Replaces SGLang's
        ``hidden = post_attn_norm(attn_out);
          hidden, residual = pre_ff_norm(hidden, residual);     # fused add+rmsnorm
          router_in = router.norm(residual) * router._fused_scale;
          moe_in = pre_ff_2_norm(residual);``
    with a single Triton kernel that walks the row 3 times for 2 reductions
    + 1 producer pass, mirroring the Inductor-generated kernel.
    """
    assert attn_out.dim() == 2 and attn_out.stride(-1) == 1
    M, N = attn_out.shape
    BLOCK_SIZE = triton.next_power_of_2(N)

    post_attn_res = torch.empty_like(attn_out)
    router_in = torch.empty_like(attn_out)
    dense_ff_in = torch.empty_like(attn_out)
    moe_in = torch.empty_like(attn_out)

    _gemma_post_attn_triple_rmsnorm_kernel[(M,)](
        attn_out,
        post_attn_weight,
        residual_before_attn,
        router_fused_scale,
        pre_ff_weight,
        pre_ff_2_weight,
        post_attn_res,
        router_in,
        dense_ff_in,
        moe_in,
        attn_out.stride(0),
        residual_before_attn.stride(0),
        post_attn_res.stride(0),
        router_in.stride(0),
        dense_ff_in.stride(0),
        moe_in.stride(0),
        N,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return post_attn_res, router_in, dense_ff_in, moe_in
