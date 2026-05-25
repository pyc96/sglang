"""Unit tests for ``gemma4_arf_rmsnorm_residual_scalar``.

Three coverage points:
1. Success path: when the FlashInfer fused kernel returns a non-None
   ``norm_out``, the wrapper returns ``norm_out * scalar`` and does NOT
   call ``tensor_model_parallel_all_reduce``.
2. Fallback path: when the FlashInfer fused kernel returns ``(None, None)``
   (e.g. flashinfer unavailable, workspace not ready, non-contig input,
   batch too large), the wrapper falls back to
   ``tensor_model_parallel_all_reduce`` + ``gemma_rmsnorm_residual_scalar``
   with bit-identical semantics.
3. Predicate-off path: when ``apply_flashinfer_allreduce_fusion`` returns
   False (e.g. flag disabled), the wrapper takes the fallback path
   directly without even calling FlashInfer.
"""

import unittest
from unittest.mock import patch

import torch

from sglang.srt.layers.gemma4_fused_ops import gemma4_arf_rmsnorm_residual_scalar


class TestGemma4ArfRmsnormResidualScalar(unittest.TestCase):
    """All three branches of the new wrapper, with FlashInfer + all-reduce
    fully mocked so the test runs on CPU."""

    def setUp(self):
        # Use a tiny CUDA tensor to satisfy the ``x.is_cuda`` gate in the
        # wrapper without requiring real CUDA: the wrapper's branch
        # condition reads attributes; we provide a fake CUDA tensor via
        # mocking ``is_cuda`` to True on a CPU tensor.
        self.T = 4  # tokens
        self.H = 16  # hidden
        self.scalar_val = 2.5
        # All test tensors live on CPU; we patch ``is_cuda`` per-test.
        self.x = torch.randn(self.T, self.H)
        self.weight = torch.randn(self.H)
        self.residual = torch.randn(self.T, self.H)
        self.scalar = torch.tensor([self.scalar_val])

    def _force_cuda(self, *tensors):
        for t in tensors:
            # SimpleNamespace would break PyTorch ops; we instead patch the
            # is_cuda *property* on the tensor's class via a context.
            pass

    def test_success_path_uses_flashinfer_and_applies_scalar(self):
        # Sentinel "fused" tensor returned by flashinfer.  In real life it
        # would be (norm(allreduce(x) + residual) * weight).  Here it's
        # just a known tensor so the test can assert ``out == sentinel *
        # scalar`` without re-implementing the kernel.
        sentinel_norm = torch.full_like(self.x, fill_value=1.5)
        sentinel_residual = torch.full_like(self.residual, fill_value=0.5)

        with (
            patch(
                "sglang.srt.layers.gemma4_fused_ops.gemma_rmsnorm_residual_scalar"
            ) as mock_kernel,
            patch(
                "sglang.srt.layers.communicator.apply_flashinfer_allreduce_fusion",
                return_value=True,
            ),
            patch(
                "sglang.srt.layers.flashinfer_comm_fusion.flashinfer_allreduce_residual_rmsnorm",
                return_value=(sentinel_norm, sentinel_residual),
            ),
            patch("sglang.srt.distributed.tensor_model_parallel_all_reduce") as mock_ar,
            patch.object(torch.Tensor, "is_cuda", property(lambda self: True)),
        ):
            out = gemma4_arf_rmsnorm_residual_scalar(
                self.x,
                self.weight,
                self.residual,
                self.scalar,
                eps=1e-6,
            )

        # The wrapper should return sentinel_norm * scalar.
        expected = sentinel_norm * self.scalar
        torch.testing.assert_close(out, expected)
        # And critically: neither the AR helper nor the fallback kernel
        # was invoked, because the fused path succeeded.
        mock_ar.assert_not_called()
        mock_kernel.assert_not_called()

    def test_fallback_when_flashinfer_returns_none(self):
        # FlashInfer's wrapper returns (None, None) when its preconditions
        # aren't met at runtime (e.g. workspace init failed,
        # non-contiguous tensors).  Wrapper should fall back to the
        # ``tensor_model_parallel_all_reduce + gemma_rmsnorm_residual_scalar``
        # pair with bit-identical semantics.
        reduced_sentinel = torch.full_like(self.x, fill_value=7.7)
        kernel_sentinel = torch.full_like(self.x, fill_value=3.3)

        with (
            patch(
                "sglang.srt.layers.gemma4_fused_ops.gemma_rmsnorm_residual_scalar",
                return_value=kernel_sentinel,
            ) as mock_kernel,
            patch(
                "sglang.srt.layers.communicator.apply_flashinfer_allreduce_fusion",
                return_value=True,
            ),
            patch(
                "sglang.srt.layers.flashinfer_comm_fusion.flashinfer_allreduce_residual_rmsnorm",
                return_value=(None, None),
            ),
            patch(
                "sglang.srt.distributed.tensor_model_parallel_all_reduce",
                return_value=reduced_sentinel,
            ) as mock_ar,
            patch.object(torch.Tensor, "is_cuda", property(lambda self: True)),
        ):
            out = gemma4_arf_rmsnorm_residual_scalar(
                self.x,
                self.weight,
                self.residual,
                self.scalar,
                eps=1e-6,
            )

        # Wrapper must have called the AR + fallback kernel.
        mock_ar.assert_called_once_with(self.x)
        mock_kernel.assert_called_once()
        # And returned the kernel's output verbatim (no extra scalar mul
        # because the fallback kernel already applies the scalar).
        self.assertIs(out, kernel_sentinel)

    def test_predicate_off_uses_fallback_directly(self):
        # When apply_flashinfer_allreduce_fusion(...) is False (e.g. flag
        # disabled), the wrapper must take the fallback path without even
        # invoking flashinfer_allreduce_residual_rmsnorm.
        reduced_sentinel = torch.full_like(self.x, fill_value=7.7)
        kernel_sentinel = torch.full_like(self.x, fill_value=3.3)

        with (
            patch(
                "sglang.srt.layers.gemma4_fused_ops.gemma_rmsnorm_residual_scalar",
                return_value=kernel_sentinel,
            ) as mock_kernel,
            patch(
                "sglang.srt.layers.communicator.apply_flashinfer_allreduce_fusion",
                return_value=False,
            ),
            patch(
                "sglang.srt.layers.flashinfer_comm_fusion.flashinfer_allreduce_residual_rmsnorm",
            ) as mock_fi,
            patch(
                "sglang.srt.distributed.tensor_model_parallel_all_reduce",
                return_value=reduced_sentinel,
            ) as mock_ar,
            patch.object(torch.Tensor, "is_cuda", property(lambda self: True)),
        ):
            out = gemma4_arf_rmsnorm_residual_scalar(
                self.x,
                self.weight,
                self.residual,
                self.scalar,
                eps=1e-6,
            )

        mock_fi.assert_not_called()
        mock_ar.assert_called_once_with(self.x)
        mock_kernel.assert_called_once()
        self.assertIs(out, kernel_sentinel)

    def test_non_cuda_input_takes_fallback(self):
        # CPU tensors short-circuit through the fallback (the ``is_cuda``
        # gate prevents flashinfer from ever being called).
        reduced_sentinel = torch.full_like(self.x, fill_value=7.7)
        kernel_sentinel = torch.full_like(self.x, fill_value=3.3)

        with (
            patch(
                "sglang.srt.layers.gemma4_fused_ops.gemma_rmsnorm_residual_scalar",
                return_value=kernel_sentinel,
            ) as mock_kernel,
            patch(
                "sglang.srt.layers.communicator.apply_flashinfer_allreduce_fusion",
            ) as mock_pred,
            patch(
                "sglang.srt.layers.flashinfer_comm_fusion.flashinfer_allreduce_residual_rmsnorm",
            ) as mock_fi,
            patch(
                "sglang.srt.distributed.tensor_model_parallel_all_reduce",
                return_value=reduced_sentinel,
            ) as mock_ar,
        ):
            # Note: NOT patching is_cuda — leave it False on the CPU tensor.
            out = gemma4_arf_rmsnorm_residual_scalar(
                self.x,
                self.weight,
                self.residual,
                self.scalar,
                eps=1e-6,
            )

        mock_pred.assert_not_called()  # short-circuited before predicate
        mock_fi.assert_not_called()
        mock_ar.assert_called_once_with(self.x)
        mock_kernel.assert_called_once()
        self.assertIs(out, kernel_sentinel)


if __name__ == "__main__":
    unittest.main()
