"""Day 2 gate: masking, the loss, causality, and a tiny overfit.

These build a small Config in code with a made-up vocabulary, so they need
neither the tokenizer, the dataset, nor a network connection.
"""

import math

import pytest
import torch

from _helpers import VOCAB, tiny_cfg
from story_moe.attention import offset_causal_mask
from story_moe.model import StoryLM, count_parameters, expected_initial_loss


@pytest.fixture
def model():
    torch.manual_seed(0)
    m = StoryLM(tiny_cfg())
    m.eval()
    return m


# --- masking ---------------------------------------------------------------


def test_mask_with_no_past_is_lower_triangular():
    mask = offset_causal_mask(5, 5, past_len=0, device=torch.device("cpu"))
    assert torch.equal(mask, torch.ones(5, 5, dtype=torch.bool).tril())


def test_mask_with_past_allows_all_cached_keys():
    # 2 new queries after 3 cached tokens, so query 0 is position 3 and query 1 is 4.
    mask = offset_causal_mask(2, 5, past_len=3, device=torch.device("cpu"))
    expected = torch.tensor(
        [[True, True, True, True, False],
         [True, True, True, True, True]]
    )
    assert torch.equal(mask, expected)


def test_single_token_decode_sees_everything():
    """One new query after P cached tokens excludes nothing. Every key is its past."""
    mask = offset_causal_mask(1, 9, past_len=8, device=torch.device("cpu"))
    assert mask.all()


# --- shapes and loss -------------------------------------------------------


def test_forward_shapes_and_finiteness(model):
    x = torch.randint(0, VOCAB, (3, 16))
    out = model(x)
    assert out.logits.shape == (3, 16, VOCAB)
    assert torch.isfinite(out.logits).all()
    assert out.lm_loss is None and out.total_loss is None


def test_loss_is_computed_and_separated_from_aux(model):
    x = torch.randint(0, VOCAB, (2, 16))
    y = torch.randint(0, VOCAB, (2, 16))
    out = model(x, targets=y)
    assert torch.isfinite(out.lm_loss)
    assert out.aux_loss.item() == 0.0
    assert torch.allclose(out.total_loss, out.lm_loss)


def test_untrained_loss_is_near_log_vocab(model):
    """The cheapest bug detector there is. A uniform predictor scores ln(V)."""
    x = torch.randint(0, VOCAB, (4, 16))
    y = torch.randint(0, VOCAB, (4, 16))
    loss = model(x, targets=y).lm_loss.item()
    assert abs(loss - math.log(VOCAB)) < 0.25, f"{loss} vs ln({VOCAB})={math.log(VOCAB)}"
    assert abs(expected_initial_loss(VOCAB) - math.log(VOCAB)) < 1e-12


def test_mismatched_target_shape_rejected(model):
    with pytest.raises(ValueError):
        model(torch.randint(0, VOCAB, (2, 16)), targets=torch.randint(0, VOCAB, (2, 15)))


# --- causality -------------------------------------------------------------


def test_future_tokens_do_not_change_earlier_logits(model):
    """Rewrite the suffix. Every logit at or before the cut must be unchanged."""
    torch.manual_seed(1)
    x = torch.randint(0, VOCAB, (2, 16))
    cut = 9

    baseline = model(x).logits
    changed = x.clone()
    changed[:, cut:] = (changed[:, cut:] + 37) % VOCAB
    assert not torch.equal(x, changed)

    perturbed = model(changed).logits
    torch.testing.assert_close(
        baseline[:, :cut], perturbed[:, :cut], rtol=1e-4, atol=1e-5
    )
    # and the suffix really did move, or the test proves nothing
    assert not torch.allclose(baseline[:, cut:], perturbed[:, cut:])


# --- logits_to_keep --------------------------------------------------------


def test_logits_to_keep_matches_tail_of_full_logits(model):
    x = torch.randint(0, VOCAB, (2, 16))
    full = model(x).logits
    last = model(x, logits_to_keep=1).logits
    assert last.shape == (2, 1, VOCAB)
    torch.testing.assert_close(last[:, -1], full[:, -1], rtol=1e-4, atol=1e-5)


def test_targets_with_logits_to_keep_is_rejected(model):
    x = torch.randint(0, VOCAB, (2, 16))
    with pytest.raises(ValueError):
        model(x, targets=x, logits_to_keep=1)


# --- parameters ------------------------------------------------------------


def test_tied_weights_counted_once():
    cfg = tiny_cfg(tie_weights=True)
    counts = count_parameters(StoryLM(cfg))
    assert counts["embedding"] + counts["body"] == counts["total"]

    untied = count_parameters(StoryLM(tiny_cfg(tie_weights=False)))
    assert untied["total"] - counts["total"] == VOCAB * cfg.model.d_model


def test_moe_and_dense_share_everything_but_the_feedforward():
    """Only the feed-forward slot may differ. See tests/test_moe.py for the rest."""
    dense = StoryLM(tiny_cfg())
    sparse = StoryLM(tiny_cfg(use_moe=True, n_experts=4, top_k=2, expert_width=48))

    dense_names = {n.split(".feed_forward")[0] for n, _ in dense.named_parameters()}
    sparse_names = {n.split(".feed_forward")[0] for n, _ in sparse.named_parameters()}
    assert dense_names == sparse_names

    assert not dense.blocks[0].use_moe and sparse.blocks[0].use_moe
    assert count_parameters(dense)["embedding"] == count_parameters(sparse)["embedding"]


# --- the overfit gate ------------------------------------------------------


def test_tiny_batch_overfits():
    torch.manual_seed(0)
    cfg = tiny_cfg()
    m = StoryLM(cfg)
    m.train()

    x = torch.randint(0, VOCAB, (2, 16))
    y = torch.randint(0, VOCAB, (2, 16))

    opt = torch.optim.AdamW(m.parameters(), lr=1e-2, betas=(0.9, 0.95), weight_decay=0.0)
    first = m(x, targets=y).lm_loss.item()

    for _ in range(400):
        out = m(x, targets=y)
        opt.zero_grad(set_to_none=True)
        out.total_loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()

    m.eval()
    with torch.no_grad():
        out = m(x, targets=y)
        acc = (out.logits.argmax(-1) == y).float().mean().item()

    assert out.lm_loss.item() < 0.2 * first, f"loss {out.lm_loss.item()} from {first}"
    assert acc > 0.9, f"next-token accuracy only {acc:.2%}"


def test_every_parameter_receives_gradient():
    torch.manual_seed(0)
    m = StoryLM(tiny_cfg())
    out = m(torch.randint(0, VOCAB, (2, 16)), targets=torch.randint(0, VOCAB, (2, 16)))
    out.total_loss.backward()

    missing = [n for n, p in m.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing, f"no gradient reached: {missing}"
    total = sum(float(p.grad.pow(2).sum()) for _, p in m.named_parameters() if p.grad is not None)
    assert total > 0
