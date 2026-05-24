"""
Unit tests for the YOCO ("You Only Cache Once") fast-prefill split in
``Gemma4TextModel.forward``.

The full forward path needs CUDA + a real Gemma4 checkpoint, so these
tests focus on the eligibility logic and the per-request "last token
index" math. They monkey-patch a minimal ``ForwardBatch``-like object
and exercise ``_yoco_eligibility`` and ``_yoco_truncate_to_last_tokens``
on CPU.

Larger end-to-end correctness is covered by the e2e benchmarks in the
PR description (E2B and E4B long-prompt runs both produced character-
identical outputs on the YOCO/non-YOCO single-prompt smoke test).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import List

import torch

from sglang.srt.models import gemma4_causal as gemma4_causal_module


class _FakeForwardMode:
    def is_extend_without_speculative(self):
        return True


class _DecodeForwardMode(_FakeForwardMode):
    def is_extend_without_speculative(self):
        return False


class _FakeAttnBackend:
    def __init__(self):
        self.init_calls: List[tuple] = []

    def init_forward_metadata(self, forward_batch):
        # Capture the metadata that the model sees at each rebuild so the
        # tests can assert the right truncation/restore happens.
        self.init_calls.append(
            (
                int(forward_batch.extend_seq_lens.sum().item()),
                int(forward_batch.extend_prefix_lens.sum().item()),
                list(forward_batch.extend_seq_lens_cpu),
            )
        )


def _make_fake_forward_batch(
    extend_seq_lens: List[int],
    seq_lens: List[int] | None = None,
    *,
    return_logprob: bool = False,
    decode_only: bool = False,
):
    if seq_lens is None:
        seq_lens = list(extend_seq_lens)
    return SimpleNamespace(
        extend_seq_lens=torch.tensor(extend_seq_lens, dtype=torch.int32),
        extend_seq_lens_cpu=list(extend_seq_lens),
        extend_prefix_lens=torch.tensor(
            [s - e for s, e in zip(seq_lens, extend_seq_lens)],
            dtype=torch.int32,
        ),
        extend_prefix_lens_cpu=[s - e for s, e in zip(seq_lens, extend_seq_lens)],
        extend_logprob_start_lens_cpu=(
            [0] * len(extend_seq_lens) if return_logprob else None
        ),
        extend_num_tokens=sum(extend_seq_lens),
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32),
        seq_lens_cpu=torch.tensor(seq_lens, dtype=torch.int32),
        return_logprob=return_logprob,
        forward_mode=_DecodeForwardMode() if decode_only else _FakeForwardMode(),
    )


class _FakePPGroup:
    is_first_rank = True
    is_last_rank = True


def _make_fake_model(
    *,
    num_hidden_layers: int = 35,
    num_kv_shared_layers: int = 20,
    layers_to_capture: List[int] | None = None,
):
    config = SimpleNamespace(
        num_hidden_layers=num_hidden_layers,
        num_kv_shared_layers=num_kv_shared_layers,
    )
    fake = SimpleNamespace(
        config=config,
        pp_group=_FakePPGroup(),
        layers_to_capture=layers_to_capture or [],
    )
    cls = gemma4_causal_module.Gemma4TextModel
    for name in ("_yoco_eligibility", "_yoco_truncate_to_last_tokens"):
        setattr(fake, name, getattr(cls, name).__get__(fake, type(fake)))
    return fake


def test_eligibility_default_on():
    fake = _make_fake_model()
    fb = _make_fake_forward_batch([10, 5, 7])
    assert fake._yoco_eligibility(fb)


def test_eligibility_no_kv_shared_layers():
    fake = _make_fake_model(num_kv_shared_layers=0)
    fb = _make_fake_forward_batch([10, 5, 7])
    assert not fake._yoco_eligibility(fb)


def test_eligibility_pure_decode_batch():
    fake = _make_fake_model()
    # All requests have a single new token -> nothing to truncate.
    fb = _make_fake_forward_batch([1, 1, 1])
    assert not fake._yoco_eligibility(fb)


def test_eligibility_decode_forward_mode():
    fake = _make_fake_model()
    fb = _make_fake_forward_batch([10], decode_only=True)
    assert not fake._yoco_eligibility(fb)


def test_eligibility_prompt_logprobs_disable():
    fake = _make_fake_model()
    fb = _make_fake_forward_batch([10, 5], return_logprob=True)
    # extend_logprob_start_lens_cpu = [0, 0] => starts before extend, prompt logprobs requested.
    assert not fake._yoco_eligibility(fb)


def test_eligibility_layer_capture_inside_kv_shared_range():
    # Capture targets sit inside [first_kv_shared_layer_idx, num_hidden_layers]
    # so the truncated tail would corrupt them. Disable.
    fake = _make_fake_model(layers_to_capture=[28])
    fb = _make_fake_forward_batch([10, 5])
    assert not fake._yoco_eligibility(fb)


def test_eligibility_layer_capture_outside_kv_shared_range_ok():
    fake = _make_fake_model(layers_to_capture=[2, 10])
    fb = _make_fake_forward_batch([10, 5])
    assert fake._yoco_eligibility(fb)


def test_eligibility_env_kill_switch(monkeypatch):
    monkeypatch.setenv("SGLANG_GEMMA4_YOCO", "0")
    fake = _make_fake_model()
    fb = _make_fake_forward_batch([10, 5])
    assert not fake._yoco_eligibility(fb)
    # Toggle back to default.
    monkeypatch.setenv("SGLANG_GEMMA4_YOCO", "1")
    assert fake._yoco_eligibility(fb)


def test_truncate_to_last_tokens_indices_and_restore():
    fake = _make_fake_model()
    fb = _make_fake_forward_batch(
        extend_seq_lens=[3, 4, 2],
        seq_lens=[3, 4, 2],
    )

    # Patch get_attn_backend to a fake.
    fake_backend = _FakeAttnBackend()
    gemma4_causal_module.get_attn_backend = lambda: fake_backend

    hidden = torch.arange(3 + 4 + 2, dtype=torch.float32).unsqueeze(-1).repeat(1, 8)
    positions = torch.arange(9, dtype=torch.int64)
    per_layer = torch.zeros(9, 35, 16)

    h_t, p_t, ple_t, last_indices, restore_fn = fake._yoco_truncate_to_last_tokens(
        fb, hidden, positions, per_layer
    )

    # last_indices = cumsum([3,4,2]) - 1 = [2, 6, 8]
    assert last_indices.tolist() == [2, 6, 8]
    assert h_t.shape == (3, 8)
    assert torch.equal(h_t[:, 0], torch.tensor([2.0, 6.0, 8.0]))
    assert p_t.tolist() == [2, 6, 8]
    assert ple_t.shape == (3, 35, 16)

    # forward_batch was mutated: extend_seq_lens is now all-1s, prefix is seq-1.
    assert fb.extend_seq_lens.tolist() == [1, 1, 1]
    assert fb.extend_prefix_lens.tolist() == [2, 3, 1]
    assert fb.extend_seq_lens_cpu == [1, 1, 1]
    assert fb.extend_num_tokens == 3
    # The backend was asked to rebuild its metadata for the truncated batch.
    assert len(fake_backend.init_calls) == 1
    assert fake_backend.init_calls[0] == (3, 6, [1, 1, 1])

    # Restore puts the original values back and rebuilds again.
    restore_fn()
    assert fb.extend_seq_lens.tolist() == [3, 4, 2]
    assert fb.extend_prefix_lens.tolist() == [0, 0, 0]
    assert fb.extend_seq_lens_cpu == [3, 4, 2]
    assert fb.extend_num_tokens == 9
    assert len(fake_backend.init_calls) == 2
    assert fake_backend.init_calls[1] == (9, 0, [3, 4, 2])


if __name__ == "__main__":
    test_eligibility_default_on()
    test_eligibility_no_kv_shared_layers()
    test_eligibility_pure_decode_batch()
    test_eligibility_decode_forward_mode()
    test_eligibility_prompt_logprobs_disable()
    test_eligibility_layer_capture_inside_kv_shared_range()
    test_eligibility_layer_capture_outside_kv_shared_range_ok()
    test_truncate_to_last_tokens_indices_and_restore()
    print("ALL TESTS PASSED")
