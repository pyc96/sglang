"""Unit tests for the ``--kv-sharing-fast-prefill`` server flag (YOCO opt-in).

These tests cover the dataclass field, argparse registration, ``ModelConfig``
plumbing, and the cross-cutting validator that runs inside
``ServerArgs.check_server_args``.  Architecture-specific validation
(Gemma-4 arch + triton attention backend + ``num_kv_shared_layers > 0``)
runs inside ``_handle_model_specific_adjustments`` and requires a real
Hugging Face config; that path is exercised by integration tests against
the live ``google/gemma-4-31B-it`` checkpoint and is not duplicated here.
"""

import unittest
from unittest.mock import patch

from sglang.srt.server_args import ServerArgs, prepare_server_args
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

# Mock get_device so all tests run on CPU-only CI runners.
_mock_device = patch("sglang.srt.server_args.get_device", return_value="cuda")
_mock_device.start()


class TestKvSharingFastPrefillFlag(CustomTestCase):
    """Plumbing-level tests: dataclass field, argparse, ModelConfig."""

    def test_default_is_disabled(self):
        server_args = ServerArgs(model_path="dummy")
        self.assertFalse(server_args.kv_sharing_fast_prefill)

    def test_cli_flag_parses_to_true(self):
        server_args = prepare_server_args(
            [
                "--model-path",
                "dummy",
                "--kv-sharing-fast-prefill",
            ]
        )
        self.assertTrue(server_args.kv_sharing_fast_prefill)

    def test_cli_help_lists_flag(self):
        # The presence of the help text is the simplest way to assert the
        # arg is registered without invoking the full parser.
        import argparse

        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)
        # ``--kv-sharing-fast-prefill`` should map to dest ``kv_sharing_fast_prefill``.
        kvsp = next(
            (
                a
                for a in parser._actions
                if "--kv-sharing-fast-prefill" in a.option_strings
            ),
            None,
        )
        self.assertIsNotNone(kvsp)
        self.assertEqual(kvsp.dest, "kv_sharing_fast_prefill")
        self.assertFalse(kvsp.default)

    def test_model_config_plumbing(self):
        # ``ModelConfig.__init__`` must accept the new keyword.
        from sglang.srt.configs.model_config import ModelConfig

        self.assertIn(
            "kv_sharing_fast_prefill",
            ModelConfig.__init__.__code__.co_varnames,
        )


class TestKvSharingFastPrefillValidators(CustomTestCase):
    """Cross-cutting validators that fire in ``check_server_args``."""

    def _server_args(self, **kwargs):
        return ServerArgs(model_path="dummy", **kwargs)

    def test_eagle_combo_rejected(self):
        sa = self._server_args(
            kv_sharing_fast_prefill=True,
            speculative_algorithm="EAGLE",
        )
        with self.assertRaisesRegex(ValueError, "incompatible with"):
            sa._validate_kv_sharing_fast_prefill_combos()

    def test_eagle3_combo_rejected(self):
        sa = self._server_args(
            kv_sharing_fast_prefill=True,
            speculative_algorithm="EAGLE3",
        )
        with self.assertRaisesRegex(ValueError, "incompatible with"):
            sa._validate_kv_sharing_fast_prefill_combos()

    def test_nextn_combo_accepted(self):
        # NEXTN (the algorithm that promotes to FROZEN_KV_MTP for Gemma-4
        # assistant drafts) is allowed and must not raise.
        sa = self._server_args(
            kv_sharing_fast_prefill=True,
            speculative_algorithm="NEXTN",
        )
        sa._validate_kv_sharing_fast_prefill_combos()  # should not raise

    def test_flag_off_does_not_validate(self):
        # When the flag is off, even EAGLE is fine for this validator.
        sa = self._server_args(
            kv_sharing_fast_prefill=False,
            speculative_algorithm="EAGLE",
        )
        sa._validate_kv_sharing_fast_prefill_combos()  # should not raise


if __name__ == "__main__":
    unittest.main()
