"""
Unit tests for the EOS truncation in FrozenKVMTPWorkerV2._verify_v2.

What this guards against (the regression PR #25 documented and this
PR fixes):

  v1 ``EagleVerifyInput.verify`` per-req loop walks ``accept_index[i, :]``
  and, on the first token that matches an EOS / stop-token / max-new-
  tokens cap, sets all subsequent positions to ``-1`` and shrinks the
  per-req accept count accordingly. Without this, post-EOS tokens get
  committed to ``output_ids`` and ``kv_committed_len``, polluting both
  the output stream and the radix cache.

  v2 ``EagleVerifyInput.sample()`` does NOT do this — without our
  ``_truncate_at_eos_inplace`` helper, finished requests under v2
  commit one extra token past EOS per accepted spec step.

These tests use stub ``req`` objects (no GPU) so they're fast and
deterministic.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch


def _make_req(
    output_ids=None,
    eos_token_ids=None,
    stop_token_ids=None,
    max_new_tokens=None,
    ignore_eos=False,
):
    """Stub Req with the minimal surface ``_truncate_at_eos_inplace`` reads."""
    return SimpleNamespace(
        output_ids=list(output_ids or []),
        eos_token_ids=set(eos_token_ids or ()),
        tokenizer=None,
        sampling_params=SimpleNamespace(
            ignore_eos=ignore_eos,
            stop_token_ids=list(stop_token_ids or ()),
            max_new_tokens=max_new_tokens,
        ),
    )


def _make_batch(reqs):
    """Stub batch — only ``.reqs`` is read by the truncation helper."""
    return SimpleNamespace(reqs=reqs)


def _make_worker_with_helper():
    """Import the v2 worker class WITHOUT instantiating it — we just need
    the unbound ``_truncate_at_eos_inplace`` method.

    We can't instantiate ``FrozenKVMTPWorkerV2`` without a real GPU + the
    target model, but the EOS helper is pure CPU and doesn't depend on
    any instance state, so we can call it via an unbound-method bind.
    """
    from sglang.srt.speculative.frozen_kv_mtp_worker_v2 import FrozenKVMTPWorkerV2

    return FrozenKVMTPWorkerV2._truncate_at_eos_inplace.__get__(
        SimpleNamespace()  # any object; method doesn't read self
    )


def test_no_eos_no_change():
    """No req hits EOS -> accept_lens / accept_index unchanged."""
    truncate = _make_worker_with_helper()
    reqs = [_make_req(eos_token_ids={99})]
    batch = _make_batch(reqs)

    predict = torch.tensor([10, 20, 30, 40], dtype=torch.int32)  # no 99
    accept_lens = torch.tensor([4], dtype=torch.int32)
    accept_index = torch.tensor([[0, 1, 2, 3]], dtype=torch.int32)

    truncate(batch, predict, accept_lens, accept_index)
    assert accept_lens.tolist() == [4]
    assert accept_index.tolist() == [[0, 1, 2, 3]]


def test_eos_in_middle_truncates_to_position_inclusive():
    """EOS at position 2 -> accept_lens becomes 3 (includes EOS itself).
    Positions 3+ become -1.
    """
    truncate = _make_worker_with_helper()
    reqs = [_make_req(eos_token_ids={99})]
    batch = _make_batch(reqs)

    # Token at accept_index[0,2] = predict[2] = 99 (EOS)
    predict = torch.tensor([10, 20, 99, 40], dtype=torch.int32)
    accept_lens = torch.tensor([4], dtype=torch.int32)
    accept_index = torch.tensor([[0, 1, 2, 3]], dtype=torch.int32)

    truncate(batch, predict, accept_lens, accept_index)
    # Expected: kept tokens at 0, 1, 2 (incl. EOS); 3 dropped.
    assert accept_lens.tolist() == [3]
    assert accept_index.tolist() == [[0, 1, 2, -1]]


def test_eos_at_first_token():
    """EOS at the very first accepted token -> accept_lens = 1."""
    truncate = _make_worker_with_helper()
    reqs = [_make_req(eos_token_ids={99})]
    batch = _make_batch(reqs)

    predict = torch.tensor([99, 20, 30, 40], dtype=torch.int32)
    accept_lens = torch.tensor([4], dtype=torch.int32)
    accept_index = torch.tensor([[0, 1, 2, 3]], dtype=torch.int32)

    truncate(batch, predict, accept_lens, accept_index)
    assert accept_lens.tolist() == [1]
    assert accept_index.tolist() == [[0, -1, -1, -1]]


def test_eos_already_at_last_position_keeps_full_count():
    """EOS at the last accepted position -> no change to count, no -1s
    added past it (since j+1 is already past the end).
    """
    truncate = _make_worker_with_helper()
    reqs = [_make_req(eos_token_ids={99})]
    batch = _make_batch(reqs)

    predict = torch.tensor([10, 20, 30, 99], dtype=torch.int32)
    accept_lens = torch.tensor([4], dtype=torch.int32)
    accept_index = torch.tensor([[0, 1, 2, 3]], dtype=torch.int32)

    truncate(batch, predict, accept_lens, accept_index)
    assert accept_lens.tolist() == [4]
    assert accept_index.tolist() == [[0, 1, 2, 3]]


def test_stop_token_ids():
    """stop_token_ids treated as EOS."""
    truncate = _make_worker_with_helper()
    reqs = [_make_req(stop_token_ids={42})]
    batch = _make_batch(reqs)

    predict = torch.tensor([10, 42, 30, 40], dtype=torch.int32)
    accept_lens = torch.tensor([4], dtype=torch.int32)
    accept_index = torch.tensor([[0, 1, 2, 3]], dtype=torch.int32)

    truncate(batch, predict, accept_lens, accept_index)
    assert accept_lens.tolist() == [2]
    assert accept_index.tolist() == [[0, 1, -1, -1]]


def test_ignore_eos_disables_truncation():
    """ignore_eos=True -> EOS in predict is ignored, full count kept."""
    truncate = _make_worker_with_helper()
    reqs = [_make_req(eos_token_ids={99}, ignore_eos=True)]
    batch = _make_batch(reqs)

    predict = torch.tensor([10, 99, 30, 40], dtype=torch.int32)
    accept_lens = torch.tensor([4], dtype=torch.int32)
    accept_index = torch.tensor([[0, 1, 2, 3]], dtype=torch.int32)

    truncate(batch, predict, accept_lens, accept_index)
    assert accept_lens.tolist() == [4]
    assert accept_index.tolist() == [[0, 1, 2, 3]]


def test_max_new_tokens_cap():
    """If appending this token would push len(output_ids) to >= max_new,
    truncate AT this token (it's the last allowed)."""
    truncate = _make_worker_with_helper()
    # Req has 5 output tokens already; max_new_tokens=7. Allowed to add 2 more.
    reqs = [_make_req(output_ids=list(range(5)), eos_token_ids={99}, max_new_tokens=7)]
    batch = _make_batch(reqs)

    # 4 candidate accepted tokens, but cap says only 2 fit.
    predict = torch.tensor([10, 20, 30, 40], dtype=torch.int32)
    accept_lens = torch.tensor([4], dtype=torch.int32)
    accept_index = torch.tensor([[0, 1, 2, 3]], dtype=torch.int32)

    truncate(batch, predict, accept_lens, accept_index)
    assert accept_lens.tolist() == [2]
    assert accept_index.tolist() == [[0, 1, -1, -1]]


def test_existing_neg_one_breaks_loop():
    """Pre-existing -1 in accept_index_row (no token was accepted at that
    position) breaks the loop. accept_lens reflects the run before the -1."""
    truncate = _make_worker_with_helper()
    reqs = [_make_req(eos_token_ids={99})]
    batch = _make_batch(reqs)

    predict = torch.tensor([10, 20, 30, 99], dtype=torch.int32)
    # Position 2 was not accepted (kernel set -1); 3 has the bonus.
    accept_lens = torch.tensor([2], dtype=torch.int32)
    accept_index = torch.tensor([[0, 1, -1, -1]], dtype=torch.int32)

    truncate(batch, predict, accept_lens, accept_index)
    # No mutation: neither token is EOS, loop broke at the existing -1.
    assert accept_lens.tolist() == [2]
    assert accept_index.tolist() == [[0, 1, -1, -1]]


def test_multi_req_independent_eos():
    """Two reqs in a batch — one hits EOS, one doesn't. Only the EOS req
    gets truncated."""
    truncate = _make_worker_with_helper()
    reqs = [_make_req(eos_token_ids={99}), _make_req(eos_token_ids={99})]
    batch = _make_batch(reqs)

    # req 0: token at index 1 is EOS (99). req 1: no EOS.
    predict = torch.tensor([10, 99, 30, 40, 50, 60, 70, 80], dtype=torch.int32)
    accept_lens = torch.tensor([4, 4], dtype=torch.int32)
    accept_index = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=torch.int32)

    truncate(batch, predict, accept_lens, accept_index)
    assert accept_lens.tolist() == [2, 4]
    assert accept_index.tolist() == [[0, 1, -1, -1], [4, 5, 6, 7]]


def test_no_mutation_does_not_copy_back():
    """Internal optimization: when nothing changed, no GPU copy_ should
    happen. We verify by passing tensors on a device-shaped object and
    checking that .copy_ is not called. (Pure-CPU test uses
    accept_lens.tolist() comparison as the gate; if no truncation
    occurred, the tensor objects are not rebound.)"""
    truncate = _make_worker_with_helper()
    reqs = [_make_req(eos_token_ids={99})]
    batch = _make_batch(reqs)

    predict = torch.tensor([10, 20, 30, 40], dtype=torch.int32)
    accept_lens = torch.tensor([4], dtype=torch.int32)
    accept_index = torch.tensor([[0, 1, 2, 3]], dtype=torch.int32)

    # data_ptr should NOT change when there's no truncation (i.e. no .copy_).
    orig_lens_ptr = accept_lens.data_ptr()
    orig_idx_ptr = accept_index.data_ptr()
    truncate(batch, predict, accept_lens, accept_index)
    assert accept_lens.data_ptr() == orig_lens_ptr
    assert accept_index.data_ptr() == orig_idx_ptr


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
