"""
Unit tests for the batched vision-encoder code path in
``Gemma4ForConditionalGeneration`` (``gemma4_mm.py``).

These tests stub the (otherwise heavy) vision tower and embedder with
deterministic functions so they can run without GPU and without loading the
real Gemma-4 checkpoint.  They cover the three things the patch promised:

1. Multi-image requests with one resolution bucket go through exactly one
   encoder forward and exactly one embedder forward.
2. Mixed-resolution requests fall back into per-bucket batching with the
   correct per-item ordering preserved in the output.
3. The encoder-batch chunking respects ``_encoder_max_batch`` when set
   explicitly.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import List

import torch

# Import the module-level helpers without instantiating
# Gemma4ForConditionalGeneration (which would require a full Gemma4Config and
# real weights). We monkey-patch a minimal subset of the class instead.
from sglang.srt.models import gemma4_mm as gemma4_mm_module


def _make_fake_model(
    hidden_size: int = 16,
    *,
    encoder_max_batch: int | None = None,
    fail_pad: bool = False,
):
    """Return a lightweight stand-in that exposes only the attributes the
    encoder helpers touch.  The vision tower behaves like an identity pool:
    every patch becomes a hidden_size vector equal to ``[idx, idx+1, ...]``
    so the caller can verify item ordering.
    """

    class _FakeTower:
        device = torch.device("cpu")

        def __init__(self):
            self.calls: List[tuple[torch.Tensor, torch.Tensor]] = []

        def __call__(self, pv: torch.Tensor, pp: torch.Tensor):
            # pv: (B, num_patches, patch_pixels)
            # Record the call shape so the test can assert how many encoder
            # invocations happened and at what batch size.
            self.calls.append((pv.clone(), pp.clone()))
            b, n, _ = pv.shape
            # Mark every patch valid except where pp == -1 (the padding
            # convention used by the real Gemma4 vision encoder).
            pooler_mask = (pp != -1).all(dim=-1)  # (B, n)
            # Embed each patch as a constant vector keyed on the item index
            # and the patch row, so per-item output is recoverable downstream.
            hidden = (
                torch.arange(b, dtype=torch.float32)
                .view(b, 1, 1)
                .repeat(1, n, hidden_size)
            )
            return hidden, pooler_mask

    class _FakeEmbedVision(torch.nn.Module):
        def __init__(self, hidden):
            super().__init__()
            self.hidden = hidden
            self.calls: List[torch.Tensor] = []

        def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
            self.calls.append(inputs_embeds.clone())
            # identity projection so we can compare expected per-token outputs
            return inputs_embeds

    class _LM:
        def __init__(self, hidden):
            self.config = SimpleNamespace(hidden_size=hidden)
            self.device = torch.device("cpu")

        def dtype(self):
            return torch.float32

    text_config = SimpleNamespace(hidden_size=hidden_size)
    config = SimpleNamespace(text_config=text_config)

    # The real `_encoder_max_batch` returns 1 when the per-patch cost has not
    # been initialized yet (the fail-safe path for unloaded models). To
    # exercise the batching code we set a very large budget by default and
    # let the `encoder_max_batch` kwarg override it.
    if encoder_max_batch is None:
        budget = 1 << 40  # 1 TB — effectively no bound
        per_patch = 1
    else:
        budget = encoder_max_batch
        per_patch = 1

    fake = SimpleNamespace(
        config=config,
        vision_tower=_FakeTower(),
        embed_vision=_FakeEmbedVision(hidden_size),
        language_model=_LM(hidden_size),
        _encoder_budget_bytes=budget,
        _encoder_bytes_per_patch=per_patch,
    )
    # Bind the real (unbound) methods to the fake instance.
    cls = gemma4_mm_module.Gemma4ForConditionalGeneration
    for name in [
        "_flatten_pixel_lists",
        "_batched_encode",
        "_gather_mm_features",
        "_encoder_max_batch",
        "get_image_feature",
        "get_video_feature",
    ]:
        fn = getattr(cls, name)
        setattr(fake, name, fn.__get__(fake, type(fake)))

    fake._fail_pad = fail_pad
    # parameters() helper used in the empty path; return at least one tensor
    fake.parameters = lambda: iter([torch.zeros(1)])
    return fake


def _make_item(num_images: int, num_patches: int):
    """Construct a minimal MultimodalDataItem-like object with `num_images`
    images each shaped (num_patches, 4)."""
    pv_list = [torch.full((num_patches, 4), float(i)) for i in range(num_images)]
    pp_list = [
        torch.arange(num_patches).unsqueeze(-1).repeat(1, 2).float()
        for _ in range(num_images)
    ]
    return SimpleNamespace(feature=pv_list, image_position_ids=pp_list)


def test_single_resolution_single_call():
    fake = _make_fake_model()
    item = _make_item(num_images=6, num_patches=10)
    out = fake.get_image_feature([item])

    # 1 encoder forward over [6, 10, 4]
    assert len(fake.vision_tower.calls) == 1, fake.vision_tower.calls
    pv, _ = fake.vision_tower.calls[0]
    assert pv.shape == (6, 10, 4)

    # 1 batched embedder call over (1, 60, 16)
    assert len(fake.embed_vision.calls) == 1
    assert fake.embed_vision.calls[0].shape == (1, 60, 16)

    # Output is (60, 16): 6 images × 10 valid patches × hidden 16
    assert out.shape == (60, 16)


def test_mixed_resolution_bucketing():
    fake = _make_fake_model()
    # 2 small images (5 patches each) and 1 big image (12 patches)
    small = _make_item(num_images=2, num_patches=5)
    big = _make_item(num_images=1, num_patches=12)
    fake.get_image_feature([small, big])

    # Two buckets: one for 5 patches (batch=2), one for 12 patches (batch=1).
    assert len(fake.vision_tower.calls) == 2
    shapes = sorted(call[0].shape for call in fake.vision_tower.calls)
    assert shapes == [(1, 12, 4), (2, 5, 4)]

    # Still a single embedder call over all valid tokens.
    assert len(fake.embed_vision.calls) == 1
    total_tokens = 2 * 5 + 1 * 12
    assert fake.embed_vision.calls[0].shape == (1, total_tokens, 16)


def test_chunking_when_max_batch_set():
    # With per_patch=1 and patches=2, cost-per-item = 2.
    # budget=4 -> 4//2 = 2 items per chunk; 6 items -> 3 encoder calls.
    fake = _make_fake_model(encoder_max_batch=4)
    item = _make_item(num_images=6, num_patches=2)
    fake.get_image_feature([item])
    assert len(fake.vision_tower.calls) == 3
    # Still 1 embedder call.
    assert len(fake.embed_vision.calls) == 1


def test_empty_returns_empty_tensor():
    fake = _make_fake_model()
    out = fake.get_image_feature([])
    assert out.shape == (0, 16)


if __name__ == "__main__":
    test_single_resolution_single_call()
    test_mixed_resolution_bucketing()
    test_chunking_when_max_batch_set()
    test_empty_returns_empty_tensor()
    print("ALL TESTS PASSED")
