"""Day 3 gate: RoPE.

The offset test and the relative-position test are the two that matter. Together
they are why a key can be rotated once, cached, and stay correct forever. The
whole KV cache milestone rests on that assumption.
"""

import math

import pytest
import torch

import story_moe.attention as attention_module
from story_moe.model import StoryLM
from story_moe.rope import apply_rope, rope_tables

from _helpers import VOCAB, tiny_cfg


def test_table_shapes_and_theta_values():
    Dh, T, base = 8, 5, 10000.0
    cos, sin = rope_tables(Dh, T, base)
    assert cos.shape == sin.shape == (T, Dh // 2)
    assert cos.dtype == torch.float32

    i = torch.arange(Dh // 2, dtype=torch.float32)
    theta = base ** (-2.0 * i / Dh)
    angles = torch.arange(T, dtype=torch.float32).unsqueeze(1) * theta.unsqueeze(0)
    torch.testing.assert_close(cos, angles.cos())
    torch.testing.assert_close(sin, angles.sin())
    # theta_0 is always 1 -> the first pair rotates by the raw position
    assert math.isclose(float(theta[0]), 1.0)


def test_position_zero_is_the_identity():
    cos, sin = rope_tables(8, 16)
    x = torch.randn(2, 3, 1, 8)
    torch.testing.assert_close(apply_rope(x, cos, sin, offset=0), x)


def test_rotation_preserves_norms():
    cos, sin = rope_tables(8, 32)
    x = torch.randn(2, 3, 5, 8)
    y = apply_rope(x, cos, sin, offset=4)
    torch.testing.assert_close(x.norm(dim=-1), y.norm(dim=-1), rtol=1e-5, atol=1e-6)
    # and it is not a no-op at nonzero positions
    assert not torch.allclose(x, y)


def test_offset_chunk_matches_the_tail_of_a_full_pass():
    """The KV-cache assumption. Rotating a chunk at offset P gives the same
    tensors as rotating the whole sequence from 0 and slicing off the tail."""
    cos, sin = rope_tables(8, 64)
    x = torch.randn(2, 3, 20, 8)
    P = 12

    full = apply_rope(x, cos, sin, offset=0)
    chunk = apply_rope(x[:, :, P:], cos, sin, offset=P)
    torch.testing.assert_close(full[:, :, P:], chunk, rtol=1e-5, atol=1e-6)


def test_dot_product_depends_only_on_relative_position():
    cos, sin = rope_tables(8, 64)
    torch.manual_seed(0)
    q = torch.randn(1, 1, 1, 8)
    k = torch.randn(1, 1, 1, 8)

    def dot(p_q: int, p_k: int) -> torch.Tensor:
        return (apply_rope(q, cos, sin, p_q) * apply_rope(k, cos, sin, p_k)).sum()

    # same gap, different absolute positions -> same score
    torch.testing.assert_close(dot(3, 5), dot(10, 12), rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(dot(0, 7), dot(20, 27), rtol=1e-5, atol=1e-6)
    # different gap -> different score
    assert not torch.isclose(dot(3, 5), dot(3, 9))


def test_odd_head_dim_rejected():
    with pytest.raises(ValueError):
        rope_tables(7, 8)


def test_position_beyond_table_rejected():
    cos, sin = rope_tables(8, 10)
    x = torch.randn(1, 1, 4, 8)
    with pytest.raises(ValueError):
        apply_rope(x, cos, sin, offset=8)          # would need positions 8..11


def test_head_dim_mismatch_rejected():
    cos, sin = rope_tables(8, 10)
    with pytest.raises(ValueError):
        apply_rope(torch.randn(1, 1, 2, 16), cos, sin)


# --- wiring into the model -------------------------------------------------


def test_model_has_no_positional_embedding_table():
    """Day 2's learned positions were deleted outright. A table left in place
    but unused would fail here too."""
    m = StoryLM(tiny_cfg())
    names = [n for n, _ in m.named_parameters()]
    assert not any("pos" in n for n in names), names
    assert hasattr(m, "rope_cos") and hasattr(m, "rope_sin")
    # RoPE is parameter-free, so the buffers must not show up as trainable params
    assert "rope_cos" not in names and "rope_sin" not in names


def test_rope_tables_reach_attention():
    torch.manual_seed(0)
    m = StoryLM(tiny_cfg())
    m.eval()
    x = torch.randint(0, VOCAB, (1, 8))

    with torch.no_grad():
        real = m(x).logits
        m.rope_cos = torch.ones_like(m.rope_cos)
        m.rope_sin = torch.zeros_like(m.rope_sin)
        neutered = m(x).logits

    assert not torch.allclose(real, neutered), "RoPE tables never reached attention"


def test_rope_applied_to_q_and_k_only(monkeypatch):
    """V must never be rotated. Count the calls, two per layer for q and k."""
    calls: list[tuple] = []
    original = attention_module.apply_rope

    def counting(x, cos, sin, offset=0):
        calls.append(tuple(x.shape))
        return original(x, cos, sin, offset=offset)

    monkeypatch.setattr(attention_module, "apply_rope", counting)

    cfg = tiny_cfg()
    m = StoryLM(cfg)
    m.eval()
    with torch.no_grad():
        m(torch.randint(0, VOCAB, (1, 8)))

    assert len(calls) == 2 * cfg.model.n_layers, (
        f"expected 2 rotations per layer (q and k), got {len(calls)} over "
        f"{cfg.model.n_layers} layers -- is V being rotated?"
    )
