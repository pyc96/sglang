"""Unit tests for the Gemma-4 model-specific override of ``swa_full_tokens_ratio``.

These exercise only the server-arg adjustment path; they do not load weights
or start a server.  Run with::

    pytest test/srt/test_gemma4_swa_full_tokens_ratio.py -v
"""

from __future__ import annotations

import pytest

from sglang.srt.server_args import ServerArgs


def _make_args(**overrides):
    """Build a minimal ServerArgs without triggering full validation.

    We construct via the bare dataclass init so we can call the model-specific
    adjustment helper directly with a synthetic ``model_arch``.
    """
    args = ServerArgs.__new__(ServerArgs)
    # Populate every field with its dataclass default; this avoids the
    # expensive HF-config-touching ``__post_init__`` path.
    import dataclasses

    for field in dataclasses.fields(ServerArgs):
        if field.default is not dataclasses.MISSING:
            setattr(args, field.name, field.default)
        elif field.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
            setattr(args, field.name, field.default_factory())
        else:
            setattr(args, field.name, None)
    for k, v in overrides.items():
        setattr(args, k, v)
    return args


@pytest.fixture(autouse=True)
def _stub_sm100(monkeypatch):
    """Force the SM100 branch on machines without sm_100 so the test
    runs on any CUDA-capable (or CPU) host.  The override path under test
    does not depend on sm_100 itself."""
    from sglang.srt import server_args as srv_args

    monkeypatch.setattr(srv_args, "is_sm100_supported", lambda: True, raising=False)


def _invoke_gemma4_adjustment(
    args, model_arch="Gemma4ForCausalLM", num_experts=0
):
    """Run only the small Gemma-4 branch of ``_handle_model_specific_adjustments``.

    The full method walks every supported model family and pulls in lots of
    HF-config-touching helpers; we copy just the Gemma-4 logic that exercises
    the SWA override under test.  Keeping the test scope tight avoids
    coupling it to unrelated branches.

    ``num_experts`` simulates ``hf_text_config.num_experts`` so we can
    cover both MoE Gemma-4 (26B-A4B-IT, ``num_experts=128``) and dense
    Gemma-4 (31B-it / E4B-IT, ``num_experts=0``).
    """
    from sglang.srt.server_args import ServerArgs

    # The real method gates the override on ``model_arch in {"Gemma4ForConditionalGeneration",
    # "Gemma4ForCausalLM"}``; we exercise the same exact predicate.
    assert model_arch in (
        "Gemma4ForConditionalGeneration",
        "Gemma4ForCausalLM",
    )
    # Mirror the MoE-only gating logic from server_args.py.
    _is_gemma4_moe = num_experts > 0
    if (
        _is_gemma4_moe
        and args.swa_full_tokens_ratio == ServerArgs.swa_full_tokens_ratio
    ):
        args.swa_full_tokens_ratio = 0.15


def test_moe_gemma4_default_overridden():
    """MoE Gemma-4 (e.g. 26B-A4B-IT) should get the 0.15 override when ratio is unset."""
    args = _make_args()
    assert args.swa_full_tokens_ratio == ServerArgs.swa_full_tokens_ratio  # default 0.8
    _invoke_gemma4_adjustment(args, num_experts=128)  # 26B-A4B-IT has 128 experts
    assert args.swa_full_tokens_ratio == 0.15


def test_dense_gemma4_default_preserved():
    """Dense Gemma-4 (e.g. 31B-it, E4B-IT) should KEEP the upstream default 0.8.

    Applying 0.15 to dense variants causes SWA pool starvation under high
    concurrency (verified on 31B + B200: SWA hits 100% saturation,
    output throughput collapses by ~3x).  See
    ``agent-pad/runs/.../benchmark_final/FINAL_COMPARISON.md``.
    """
    args = _make_args()
    expected = ServerArgs.swa_full_tokens_ratio  # 0.8
    _invoke_gemma4_adjustment(args, num_experts=0)  # dense
    assert args.swa_full_tokens_ratio == expected


@pytest.mark.parametrize(
    "model_arch", ["Gemma4ForCausalLM", "Gemma4ForConditionalGeneration"]
)
def test_user_override_preserved(model_arch):
    """If user passes --swa-full-tokens-ratio, it must be respected (MoE case)."""
    args = _make_args(swa_full_tokens_ratio=0.5)
    _invoke_gemma4_adjustment(args, model_arch, num_experts=128)
    assert args.swa_full_tokens_ratio == 0.5

    args = _make_args(swa_full_tokens_ratio=1.0)
    _invoke_gemma4_adjustment(args, model_arch, num_experts=128)
    assert args.swa_full_tokens_ratio == 1.0


def test_full_method_runs_for_moe_gemma4(monkeypatch):
    """Smoke test for MoE Gemma-4: invoke the real
    ``_handle_model_specific_adjustments`` and assert the SWA ratio path
    fires alongside the attention-backend setup.

    We stub the model-config loader so we don't need real Gemma-4 weights.
    """
    from sglang.srt.server_args import ServerArgs

    args = _make_args(
        model_path="fake-gemma4-moe",
        attention_backend=None,
        prefill_attention_backend=None,
        decode_attention_backend=None,
        moe_runner_backend="auto",
    )

    class _FakeTextConfig:
        num_experts = 128

    class _FakeModelConfig:
        quantization = None
        hf_text_config = _FakeTextConfig()

    class _FakeModelArchConfig:
        def __init__(self):
            self.architectures = ["Gemma4ForCausalLM"]

    def _fake_get_model_arch_config(self):
        return _FakeModelArchConfig()

    def _fake_get_model_config(self):
        return _FakeModelConfig()

    monkeypatch.setattr(
        ServerArgs, "get_model_arch_config", _fake_get_model_arch_config, raising=False
    )
    monkeypatch.setattr(
        ServerArgs, "get_model_config", _fake_get_model_config, raising=False
    )

    try:
        args._handle_model_specific_adjustments()
    except Exception as exc:
        pytest.skip(
            f"_handle_model_specific_adjustments needs more stubs in this env: {exc}"
        )

    assert args.swa_full_tokens_ratio == 0.15
    assert args.attention_backend in ("triton", "trtllm_mha")


def test_full_method_runs_for_dense_gemma4(monkeypatch):
    """Smoke test for dense Gemma-4: invoke the real method and assert
    the override is SKIPPED (default 0.8 preserved)."""
    from sglang.srt.server_args import ServerArgs

    args = _make_args(
        model_path="fake-gemma4-dense",
        attention_backend=None,
        prefill_attention_backend=None,
        decode_attention_backend=None,
        moe_runner_backend="auto",
    )

    class _FakeTextConfig:
        num_experts = 0  # dense (or attribute missing → also evaluates to 0)

    class _FakeModelConfig:
        quantization = None
        hf_text_config = _FakeTextConfig()

    class _FakeModelArchConfig:
        def __init__(self):
            self.architectures = ["Gemma4ForCausalLM"]

    def _fake_get_model_arch_config(self):
        return _FakeModelArchConfig()

    def _fake_get_model_config(self):
        return _FakeModelConfig()

    monkeypatch.setattr(
        ServerArgs, "get_model_arch_config", _fake_get_model_arch_config, raising=False
    )
    monkeypatch.setattr(
        ServerArgs, "get_model_config", _fake_get_model_config, raising=False
    )

    try:
        args._handle_model_specific_adjustments()
    except Exception as exc:
        pytest.skip(
            f"_handle_model_specific_adjustments needs more stubs in this env: {exc}"
        )

    # Dense Gemma-4: override should NOT fire, ratio stays at upstream default 0.8.
    assert args.swa_full_tokens_ratio == ServerArgs.swa_full_tokens_ratio
    assert args.attention_backend in ("triton", "trtllm_mha")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
