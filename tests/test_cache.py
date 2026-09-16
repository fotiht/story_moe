"""Day 6 gate: a cached decode must compute the same thing as no cache at all.

The central test is `test_cached_decode_matches_uncached_forward`. Everything
else here is a guard on one way that test could pass for the wrong reason.

A KV cache is an optimization that changes no mathematics, so the only
acceptable evidence that it works is that it reproduces the unoptimized result.
The failure modes are silent. A RoPE offset left at zero gives fluent-looking
text at the wrong positions. A per-layer length skew gives each layer a
slightly wrong history. Neither crashes, and both move the logits.

Tokens are teacher-forced rather than sampled, so a mismatch is the cache and
never the sampler.
"""

from __future__ import annotations

import pytest
import torch

from story_moe.cache import KVCache
from story_moe.generate import GenerationConfig, generate, sample_next
from story_moe.model import StoryLM

from _helpers import VOCAB, tiny_cfg


def build(seed: int = 0, **overrides) -> StoryLM:
    torch.manual_seed(seed)
    model = StoryLM(tiny_cfg(**overrides))
    model.eval()  # dropout is 0.0 in tiny_cfg, but eval() states the intent
    return model


@pytest.mark.parametrize("use_moe", [False, True])
@pytest.mark.parametrize("prompt_len", [1, 3, 8])
def test_cached_decode_matches_uncached_forward(use_moe: bool, prompt_len: int) -> None:
    """Feed known tokens one at a time. Every step's logits must match."""
    overrides = dict(use_moe=True, n_experts=4, top_k=2, expert_width=32) if use_moe else {}
    model = build(**overrides)

    torch.manual_seed(1234)
    seq_len = 16
    seq = torch.randint(0, VOCAB, (2, seq_len))

    with torch.no_grad():
        reference = model(seq).logits                  # [2, seq_len, V]

    cache = model.new_cache(max_len=seq_len)
    with torch.no_grad():
        step = model(seq[:, :prompt_len], logits_to_keep=1, cache=cache).logits
        torch.testing.assert_close(
            step[:, 0, :], reference[:, prompt_len - 1, :], atol=1e-4, rtol=1e-3
        )
        for pos in range(prompt_len, seq_len):
            step = model(seq[:, pos : pos + 1], logits_to_keep=1, cache=cache).logits
            torch.testing.assert_close(
                step[:, 0, :], reference[:, pos, :], atol=1e-4, rtol=1e-3,
                msg=lambda s, pos=pos: f"logits diverge at position {pos}\n{s}",
            )
    assert cache.length == seq_len


def test_chunked_prefill_matches_single_prefill() -> None:
    """Prefilling 4+4+4 must equal prefilling 12. Exercises past_len > 0, T > 1."""
    model = build()
    torch.manual_seed(7)
    seq = torch.randint(0, VOCAB, (1, 12))

    whole = model.new_cache(max_len=12)
    with torch.no_grad():
        full = model(seq, logits_to_keep=1, cache=whole).logits

        chunked = model.new_cache(max_len=12)
        for start in range(0, 12, 4):
            out = model(seq[:, start : start + 4], logits_to_keep=1, cache=chunked)

    torch.testing.assert_close(out.logits, full, atol=1e-4, rtol=1e-3)
    assert chunked.length == whole.length == 12


def test_rope_offset_is_actually_applied() -> None:
    """A cache that ignored position would make these two agree. They must not.

    Same token at position 0 and at position 5. RoPE makes the query rotation
    differ, so the predictions differ. If someone drops the offset and rotates
    every new token at position 0, this test fails and the equivalence test
    above fails with it.
    """
    model = build()
    token = torch.tensor([[3]])

    with torch.no_grad():
        fresh = model.new_cache(max_len=8)
        first = model(token, logits_to_keep=1, cache=fresh).logits

        later = model.new_cache(max_len=8)
        torch.manual_seed(11)
        model(torch.randint(0, VOCAB, (1, 5)), logits_to_keep=1, cache=later)
        assert later.length == 5
        sixth = model(token, logits_to_keep=1, cache=later).logits

    assert not torch.allclose(first, sixth, atol=1e-4)


def test_greedy_generation_identical_with_and_without_cache() -> None:
    """The end-to-end statement. Same prompt, same tokens, cache or no cache."""
    model = build()
    prompt = torch.tensor([[5, 9, 2]])
    cfg = GenerationConfig(max_new_tokens=10, temperature=0.0)

    cached = generate(model, prompt, cfg, use_cache=True)
    uncached = generate(model, prompt, cfg, use_cache=False)

    assert torch.equal(cached, uncached)
    assert cached.shape == (1, 13)
    assert torch.equal(cached[:, :3], prompt)


def test_sampled_generation_identical_under_a_shared_seed() -> None:
    """Sampling is not an excuse for the paths to diverge, given one generator."""
    model = build()
    prompt = torch.tensor([[1, 4]])
    cfg = GenerationConfig(max_new_tokens=8, temperature=0.8, top_k=10)

    # The two paths produce probabilities that agree to floating-point noise
    # rather than bit-exactly, so in principle a draw could land within that
    # noise of a bucket boundary and diverge. The window is ~1e-7 wide per draw.
    g1 = torch.Generator().manual_seed(99)
    g2 = torch.Generator().manual_seed(99)
    assert torch.equal(
        generate(model, prompt, cfg, use_cache=True, generator=g1),
        generate(model, prompt, cfg, use_cache=False, generator=g2),
    )


def test_generation_leaves_the_model_in_training_mode_if_it_started_there() -> None:
    model = build()
    model.train()
    generate(model, torch.tensor([[1, 2]]), GenerationConfig(max_new_tokens=2))
    assert model.training


def test_eos_stops_early_and_pads() -> None:
    model = build()
    prompt = torch.tensor([[1, 2]])
    cfg = GenerationConfig(max_new_tokens=6, temperature=0.0, eos_id=None)
    free = generate(model, prompt, cfg)
    forced = free[0, 2].item()  # whatever greedy picks first

    stopped = generate(
        model, prompt,
        GenerationConfig(max_new_tokens=6, temperature=0.0, eos_id=forced),
    )
    assert stopped.size(1) == 3
    assert stopped[0, -1].item() == forced


# --- KVCache mechanics -------------------------------------------------------


def test_length_is_shared_and_advances_once_per_forward() -> None:
    """The layer-skew invariant, read directly off the cache."""
    model = build(n_layers=3)
    cache = model.new_cache(max_len=8)
    with torch.no_grad():
        model(torch.tensor([[1, 2, 3]]), logits_to_keep=1, cache=cache)
    assert cache.length == 3
    for layer in range(3):
        assert cache.keys[layer] is not None
        assert cache.keys[layer].shape[2] == 8       # capacity, not length


def test_append_returns_every_valid_position() -> None:
    cache = KVCache(n_layers=1, max_len=10)
    k = torch.randn(2, 4, 3, 8)
    keys, values = cache.append(0, k, k.clone())
    assert keys.shape == (2, 4, 3, 8)
    torch.testing.assert_close(keys, k)
    cache.advance(3)

    k2 = torch.randn(2, 4, 1, 8)
    keys, _ = cache.append(0, k2, k2.clone())
    assert keys.shape == (2, 4, 4, 8)
    torch.testing.assert_close(keys[:, :, :3], k)
    torch.testing.assert_close(keys[:, :, 3:], k2)


def test_capacity_is_enforced_rather_than_wrapping() -> None:
    cache = KVCache(n_layers=1, max_len=4)
    k = torch.randn(1, 2, 4, 8)
    cache.append(0, k, k.clone())
    cache.advance(4)
    with pytest.raises(ValueError, match="exceeds capacity"):
        cache.append(0, torch.randn(1, 2, 1, 8), torch.randn(1, 2, 1, 8))


def test_switching_precision_mid_generation_is_rejected() -> None:
    """The one mismatch torch would not catch is a silent dtype cast on copy."""
    cache = KVCache(n_layers=1, max_len=8)
    k = torch.randn(1, 2, 1, 8)
    cache.append(0, k, k.clone())
    cache.advance(1)
    half = k.half()
    with pytest.raises(ValueError, match="mid-generation"):
        cache.append(0, half, half.clone())


def test_grad_tracking_tensors_are_rejected() -> None:
    cache = KVCache(n_layers=1, max_len=8)
    k = torch.randn(1, 2, 1, 8, requires_grad=True)
    with pytest.raises(RuntimeError, match="no_grad"):
        cache.append(0, k, k.clone())


def test_training_path_refuses_a_cache() -> None:
    model = build()
    ids = torch.randint(0, VOCAB, (1, 4))
    with pytest.raises(ValueError, match="generation, not training"):
        model(ids, targets=ids, cache=model.new_cache(max_len=8))


def test_exceeding_max_seq_len_reports_absolute_positions() -> None:
    # 16 is the floor here. tiny_cfg fixes block_size at 16, and the config
    # validator rejects block_size > max_seq_len.
    model = build(max_seq_len=16)
    cache = model.new_cache(max_len=16)
    with torch.no_grad():
        model(torch.randint(0, VOCAB, (1, 16)), logits_to_keep=1, cache=cache)
    with pytest.raises(ValueError, match="exceed max_seq_len"):
        model(torch.tensor([[1]]), logits_to_keep=1, cache=cache)


def test_cache_from_a_different_model_depth_is_rejected() -> None:
    model = build(n_layers=2)
    with pytest.raises(ValueError, match="cache holds"):
        model(torch.tensor([[1]]), logits_to_keep=1, cache=KVCache(n_layers=5, max_len=8))


# --- sampler -----------------------------------------------------------------


def test_temperature_zero_is_argmax() -> None:
    logits = torch.tensor([[0.1, 5.0, 0.2], [3.0, 0.0, 1.0]])
    got = sample_next(logits, GenerationConfig(temperature=0.0))
    assert torch.equal(got, torch.tensor([[1], [0]]))


def test_top_k_never_returns_a_token_outside_the_top_k() -> None:
    logits = torch.tensor([[10.0, 9.0, -50.0, -60.0]])
    g = torch.Generator().manual_seed(0)
    for _ in range(50):
        tok = sample_next(logits, GenerationConfig(temperature=1.0, top_k=2), g).item()
        assert tok in (0, 1)


def test_negative_temperature_is_rejected() -> None:
    with pytest.raises(ValueError, match="temperature"):
        sample_next(torch.zeros(1, 4), GenerationConfig(temperature=-1.0))
