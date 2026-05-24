"""Unit tests for the Gemma4 PLE-tail fused ops added in
`python/sglang/srt/layers/gemma4_fused_ops.py`.

The PLE-tail (Per-Layer-Embedding) path in Gemma4 E2B / E4B used to issue
seven kernels per decoder layer; we collapse the five pointwise ones into
three Triton launches. These tests check numerical equivalence against a
clean PyTorch reference and require a CUDA device with bf16 support.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

cuda = pytest.importorskip("torch.cuda")
if not torch.cuda.is_available():
    pytest.skip("CUDA required for Gemma4 fused-op tests", allow_module_level=True)

from sglang.srt.layers.gemma4_fused_ops import (
    gemma_gelu_tanh_mul,
    gemma_rmsnorm_add,
    gemma_rmsnorm_residual_scalar,
)


def _ref_rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    var = x.float().pow(2).mean(-1, keepdim=True)
    return (x.float() * torch.rsqrt(var + eps) * w.float()).to(x.dtype)


@pytest.mark.parametrize("M,N", [(1, 1536), (7, 1536), (32, 2560), (128, 5376)])
def test_rmsnorm_add(M: int, N: int):
    """gemma_rmsnorm_add: out = rmsnorm(x, w) + r"""
    torch.manual_seed(0)
    x = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(N, dtype=torch.bfloat16, device="cuda") * 0.1
    r = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")

    ref = _ref_rmsnorm(x, w, eps=1e-6) + r
    out = gemma_rmsnorm_add(x, w, r, eps=1e-6)

    # bf16 reduction round-off — allow ~1/256 absolute slack at hidden=5376.
    assert torch.allclose(
        out.float(), ref.float(), atol=2e-2, rtol=2e-2
    ), f"rmsnorm_add diff at ({M},{N}): max={ (out.float()-ref.float()).abs().max().item() }"


@pytest.mark.parametrize("M,N", [(1, 256), (7, 256), (32, 512)])
def test_gelu_tanh_mul(M: int, N: int):
    """gemma_gelu_tanh_mul: out = gelu_tanh(gate) * ple"""
    torch.manual_seed(0)
    gate = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")
    ple = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")

    ref = F.gelu(gate.float(), approximate="tanh").to(torch.bfloat16) * ple
    out = gemma_gelu_tanh_mul(gate, ple)

    assert torch.allclose(
        out.float(), ref.float(), atol=5e-2, rtol=5e-2
    ), f"gelu_mul diff at ({M},{N}): max={ (out.float()-ref.float()).abs().max().item() }"


@pytest.mark.parametrize("M,N", [(1, 1536), (32, 2560)])
def test_rmsnorm_residual_scalar(M: int, N: int):
    """Existing op — verify the PLE-tail glue still matches reference."""
    torch.manual_seed(0)
    x = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(N, dtype=torch.bfloat16, device="cuda") * 0.1
    r = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")
    scalar = torch.tensor(0.7, dtype=torch.bfloat16, device="cuda")

    ref = (_ref_rmsnorm(x, w, eps=1e-6).float() + r.float()) * scalar.float()
    out = gemma_rmsnorm_residual_scalar(x, w, r, scalar, eps=1e-6)

    assert torch.allclose(
        out.float(), ref.float(), atol=2e-2, rtol=2e-2
    ), f"diff at ({M},{N}): max={ (out.float()-ref.float()).abs().max().item() }"


def test_chain_matches_eager_PLE_tail():
    """End-to-end PLE-tail composition matches the eager reference."""
    torch.manual_seed(0)
    M, H, P = 8, 1536, 256

    # Use small Linear layers as stand-ins for `per_layer_input_gate` /
    # `per_layer_projection` so the test is GEMM-independent.
    hidden_post = torch.randn(M, H, dtype=torch.bfloat16, device="cuda")

    norm_post_ff_w = torch.randn(H, dtype=torch.bfloat16, device="cuda") * 0.1
    residual = torch.randn(M, H, dtype=torch.bfloat16, device="cuda")
    eps = 1e-6

    # Synthetic outputs for the two GEMMs in the PLE tail
    gate = torch.randn(M, P, dtype=torch.bfloat16, device="cuda") * 0.3
    ple = torch.randn(M, P, dtype=torch.bfloat16, device="cuda") * 0.3
    proj_out = torch.randn(M, H, dtype=torch.bfloat16, device="cuda")
    norm_ple_w = torch.randn(H, dtype=torch.bfloat16, device="cuda") * 0.1
    layer_scalar = torch.tensor(0.7, dtype=torch.bfloat16, device="cuda")

    # Eager reference
    h_post_ref = _ref_rmsnorm(hidden_post, norm_post_ff_w, eps) + residual
    gated_ref = F.gelu(gate.float(), approximate="tanh").to(torch.bfloat16) * ple
    norm_proj = _ref_rmsnorm(proj_out, norm_ple_w, eps)
    ref = ((h_post_ref.float() + norm_proj.float()) * layer_scalar.float()).to(
        torch.bfloat16
    )

    # Fused
    h_post = gemma_rmsnorm_add(hidden_post, norm_post_ff_w, residual, eps=eps)
    gated = gemma_gelu_tanh_mul(gate, ple)
    out = gemma_rmsnorm_residual_scalar(
        proj_out, norm_ple_w, h_post, layer_scalar, eps=eps
    )

    # Sanity: gated has expected shape (the GEMM step uses it externally).
    assert gated.shape == (M, P)
    assert torch.allclose(
        out.float(), ref.float(), atol=5e-2, rtol=5e-2
    ), f"chain diff: max={ (out.float()-ref.float()).abs().max().item() }"


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
