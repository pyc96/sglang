"""Unit tests for the YOCO fast-prefill helpers on ``Gemma4TextModel``.

These tests cover three things:

* ``_can_run_yoco`` — the predicate that decides whether the YOCO branch
  should fire.  Covers the eligible case plus every individual reject
  condition.
* ``_build_cross_decoder_last_token_index`` — the gather index, matching
  ``LogitsProcessor._get_pruned_states`` (cumsum-1 in the non-padded
  case).
* ``_run_cross_decoder_with_yoco`` — verifies the temporary mutation of
  ``forward_batch.forward_mode`` and the restoration of the attention
  backend's ``forward_metadata`` after the back-half runs, with all
  external collaborators stubbed.

The actual end-to-end correctness (tokens match between YOCO-on and
YOCO-off greedy sampling) is exercised by the live integration test that
runs against a Gemma-4 31B-IT server during PR-C benchmarking, not here.
"""

import types
import unittest
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.models.gemma4_causal import Gemma4TextModel


def _stub_model(
    *,
    num_kv_shared_layers=5,
    num_hidden_layers=60,
    hidden_size_per_layer_input=0,
    kv_sharing_fast_prefill_enabled=True,
    start_layer=0,
    end_layer=None,
    layers_to_capture=None,
):
    """Build a bare ``Gemma4TextModel`` shell with just the fields the YOCO
    helpers read.  We deliberately avoid ``Gemma4TextModel.__init__`` (which
    would try to download weights, allocate embeddings, etc.) and inject
    only the attributes ``_can_run_yoco`` and ``_build_cross_decoder_*``
    consult.
    """
    if end_layer is None:
        end_layer = num_hidden_layers
    model = Gemma4TextModel.__new__(Gemma4TextModel)
    model._num_kv_shared_layers = num_kv_shared_layers
    model._first_kv_shared_layer_idx = num_hidden_layers - num_kv_shared_layers
    model._kv_sharing_fast_prefill_enabled = kv_sharing_fast_prefill_enabled
    model.hidden_size_per_layer_input = hidden_size_per_layer_input
    model.start_layer = start_layer
    model.end_layer = end_layer
    model.layers_to_capture = layers_to_capture or []
    return model


def _stub_forward_batch(
    *,
    mode=None,
    extend_seq_lens=None,
    extend_seq_lens_cpu=None,
    extend_logprob_start_lens_cpu=None,
    padded_static_len=-1,
):
    """Lightweight namespace standing in for ``ForwardBatch``."""
    if mode is None:
        mode = ForwardMode.EXTEND
    return types.SimpleNamespace(
        forward_mode=mode,
        extend_seq_lens=extend_seq_lens,
        extend_seq_lens_cpu=extend_seq_lens_cpu,
        extend_logprob_start_lens_cpu=extend_logprob_start_lens_cpu,
        padded_static_len=padded_static_len,
    )


class TestCanRunYoco(unittest.TestCase):
    def setUp(self):
        self.model = _stub_model()
        self.batch = _stub_forward_batch(
            mode=ForwardMode.EXTEND,
            extend_seq_lens=torch.tensor([4, 7, 3], dtype=torch.int32),
            extend_seq_lens_cpu=[4, 7, 3],
            extend_logprob_start_lens_cpu=[4, 7, 3],  # no input logprobs
        )

    def test_eligible_batch_returns_true(self):
        self.assertTrue(self.model._can_run_yoco(self.batch))

    def test_flag_off_returns_false(self):
        self.model._kv_sharing_fast_prefill_enabled = False
        self.assertFalse(self.model._can_run_yoco(self.batch))

    def test_no_kv_shared_layers_returns_false(self):
        self.model = _stub_model(num_kv_shared_layers=0)
        # When no shared layers, ``end_layer <= first_kv_shared_layer_idx``
        # because first_kv_shared_layer_idx == num_hidden_layers.
        self.assertFalse(self.model._can_run_yoco(self.batch))

    def test_decode_mode_returns_false(self):
        self.batch.forward_mode = ForwardMode.DECODE
        self.assertFalse(self.model._can_run_yoco(self.batch))

    def test_target_verify_returns_false(self):
        self.batch.forward_mode = ForwardMode.TARGET_VERIFY
        self.assertFalse(self.model._can_run_yoco(self.batch))

    def test_missing_extend_seq_lens_returns_false(self):
        self.batch.extend_seq_lens = None
        self.assertFalse(self.model._can_run_yoco(self.batch))

    def test_ple_enabled_still_returns_true(self):
        # PLE-enabled variants (Gemma-4 E4B/E2B) are supported: the YOCO
        # branch gathers per_layer_inputs alongside hidden_states.
        self.model.hidden_size_per_layer_input = 256
        self.assertTrue(self.model._can_run_yoco(self.batch))

    def test_input_logprobs_returns_false(self):
        # Any request with start < extend_len => input logprobs requested.
        self.batch.extend_logprob_start_lens_cpu = [0, 7, 3]
        self.assertFalse(self.model._can_run_yoco(self.batch))

    def test_eagle3_aux_capture_returns_false(self):
        self.model.layers_to_capture = [55]  # inside back half
        self.assertFalse(self.model._can_run_yoco(self.batch))

    def test_pp_no_back_half_layers_returns_false(self):
        # Front-only PP rank: no layers in the back half on this rank.
        self.model = _stub_model(start_layer=0, end_layer=30)
        self.assertFalse(self.model._can_run_yoco(self.batch))

    def test_pp_no_front_half_layers_returns_false(self):
        # Back-only PP rank.
        self.model = _stub_model(start_layer=55, end_layer=60)
        self.assertFalse(self.model._can_run_yoco(self.batch))


class TestBuildCrossDecoderLastTokenIndex(unittest.TestCase):
    def test_non_padded_matches_cumsum_minus_one(self):
        model = _stub_model()
        extend_lens = torch.tensor([4, 7, 3, 1], dtype=torch.int32)
        batch = _stub_forward_batch(
            extend_seq_lens=extend_lens,
            extend_seq_lens_cpu=[4, 7, 3, 1],
            extend_logprob_start_lens_cpu=[4, 7, 3, 1],
            padded_static_len=-1,
        )
        last_idx = model._build_cross_decoder_last_token_index(batch)
        # Expected: cumsum([4,7,3,1]) - 1 = [3, 10, 13, 14]
        torch.testing.assert_close(
            last_idx,
            torch.tensor([3, 10, 13, 14], dtype=torch.int32),
            check_dtype=False,
        )

    def test_padded_static_len(self):
        model = _stub_model()
        extend_lens = torch.tensor([3, 5, 2], dtype=torch.int32)
        batch = _stub_forward_batch(
            extend_seq_lens=extend_lens,
            extend_seq_lens_cpu=[3, 5, 2],
            extend_logprob_start_lens_cpu=[3, 5, 2],
            padded_static_len=8,
        )
        last_idx = model._build_cross_decoder_last_token_index(batch)
        # Padded layout: each req gets 8 slots; last valid token is at
        # i*8 + extend_len[i] - 1 => [0+3-1, 8+5-1, 16+2-1] = [2, 12, 17].
        torch.testing.assert_close(
            last_idx,
            torch.tensor([2, 12, 17], dtype=torch.int64),
            check_dtype=False,
        )


class TestRunCrossDecoderWithYoco(unittest.TestCase):
    """Verify the mode-mutation + metadata-restore contract."""

    def _make_model_with_back_half_layers(self, num_back_half=2, hidden=8):
        model = _stub_model(num_kv_shared_layers=num_back_half, num_hidden_layers=4)
        # The back-half layers must respond to ``__call__`` like Gemma4
        # decoder layers do (return (hidden_states, None)).  We replace them
        # with simple Linear stubs that increment by a known constant per
        # layer so the test can verify which layers ran.
        model.layers = [None] * 4
        for i in range(model._first_kv_shared_layer_idx, 4):
            layer = MagicMock()
            layer.return_value = (
                # Output: hidden_states + (i+1) — distinct per layer
                torch.zeros(0),  # placeholder; replaced in side_effect below
                None,
            )
            layer.side_effect = (
                lambda positions, hidden_states, per_layer_input, forward_batch, _i=i: (
                    hidden_states + (_i + 1),
                    None,
                )
            )
            model.layers[i] = layer
        return model

    def test_mode_and_metadata_are_restored(self):
        model = self._make_model_with_back_half_layers(num_back_half=2)
        original_mode = ForwardMode.EXTEND
        original_metadata = object()  # sentinel

        fb = _stub_forward_batch(
            mode=original_mode,
            extend_seq_lens=torch.tensor([3, 2], dtype=torch.int32),
            extend_seq_lens_cpu=[3, 2],
            extend_logprob_start_lens_cpu=[3, 2],
        )

        mock_backend = MagicMock()
        mock_backend.forward_metadata = original_metadata
        # init_forward_metadata replaces forward_metadata with a NEW sentinel
        # whenever called inside the YOCO scope.  We assert the swap happens
        # AND the original is restored on exit.
        new_metadata = object()

        def init_side_effect(_fb):
            mock_backend.forward_metadata = new_metadata

        mock_backend.init_forward_metadata.side_effect = init_side_effect

        with patch(
            "sglang.srt.models.gemma4_causal.get_attn_backend",
            return_value=mock_backend,
        ):
            positions = torch.arange(5)
            hidden = torch.zeros(5, 4)  # [T=5, H=4]
            last_idx = torch.tensor([2, 4])  # cumsum([3,2])-1
            out = model._run_cross_decoder_with_yoco(
                positions=positions,
                hidden_states=hidden,
                forward_batch=fb,
                last_token_index=last_idx,
            )

        # The cross-decoder ran on the gathered rows (shape [2, 4]).
        self.assertEqual(out.shape, (2, 4))
        # Both back-half layers fired (each added i+1 = 3 then 4 => sum 7).
        self.assertTrue(torch.equal(out, torch.full((2, 4), 7.0)))
        # Mode is restored.
        self.assertEqual(fb.forward_mode, original_mode)
        # Metadata is restored.
        self.assertIs(mock_backend.forward_metadata, original_metadata)
        # init_forward_metadata was called exactly once (to build the
        # decode-shaped metadata for the back half).
        self.assertEqual(mock_backend.init_forward_metadata.call_count, 1)

    def test_per_layer_inputs_are_gathered(self):
        """For PLE-enabled variants, per_layer_inputs must be gathered to
        the same rows as hidden_states before the back-half runs."""
        model = self._make_model_with_back_half_layers(num_back_half=2)
        captured_per_layer_inputs = []

        def layer_side_effect(positions, hidden_states, per_layer_input, forward_batch):
            captured_per_layer_inputs.append(
                None if per_layer_input is None else tuple(per_layer_input.shape)
            )
            return (hidden_states + 1, None)

        for layer in model.layers[model._first_kv_shared_layer_idx :]:
            layer.side_effect = layer_side_effect

        fb = _stub_forward_batch(
            mode=ForwardMode.EXTEND,
            extend_seq_lens=torch.tensor([3, 2], dtype=torch.int32),
            extend_seq_lens_cpu=[3, 2],
            extend_logprob_start_lens_cpu=[3, 2],
        )
        mock_backend = MagicMock()
        mock_backend.forward_metadata = object()

        # Provide a per_layer_inputs tensor of shape [T=5, num_layers=4, ple_dim=16].
        ple = torch.randn(5, 4, 16)

        with patch(
            "sglang.srt.models.gemma4_causal.get_attn_backend",
            return_value=mock_backend,
        ):
            model._run_cross_decoder_with_yoco(
                positions=torch.arange(5),
                hidden_states=torch.zeros(5, 4),
                forward_batch=fb,
                last_token_index=torch.tensor([2, 4]),
                per_layer_inputs=ple,
            )

        # Each back-half layer should have received a per_layer_input of
        # shape [B=2, ple_dim=16] (gathered to last-token rows + sliced by layer).
        self.assertEqual(len(captured_per_layer_inputs), 2)
        self.assertEqual(captured_per_layer_inputs, [(2, 16), (2, 16)])

    def test_exception_in_back_half_still_restores(self):
        """If a back-half layer raises, mode and metadata must still revert."""
        model = self._make_model_with_back_half_layers(num_back_half=1)
        # Make the single back-half layer raise.
        model.layers[3].side_effect = RuntimeError("intentional")

        original_mode = ForwardMode.EXTEND
        original_metadata = object()
        fb = _stub_forward_batch(
            mode=original_mode,
            extend_seq_lens=torch.tensor([2], dtype=torch.int32),
            extend_seq_lens_cpu=[2],
            extend_logprob_start_lens_cpu=[2],
        )
        mock_backend = MagicMock()
        mock_backend.forward_metadata = original_metadata
        mock_backend.init_forward_metadata.side_effect = lambda _fb: setattr(
            mock_backend, "forward_metadata", object()
        )

        with patch(
            "sglang.srt.models.gemma4_causal.get_attn_backend",
            return_value=mock_backend,
        ):
            with self.assertRaises(RuntimeError):
                model._run_cross_decoder_with_yoco(
                    positions=torch.arange(2),
                    hidden_states=torch.zeros(2, 4),
                    forward_batch=fb,
                    last_token_index=torch.tensor([1]),
                )

        # Even on exception, mode and metadata are restored.
        self.assertEqual(fb.forward_mode, original_mode)
        self.assertIs(mock_backend.forward_metadata, original_metadata)


if __name__ == "__main__":
    unittest.main()
