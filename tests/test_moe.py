"""Day 4 gate: sparse Top-2 dispatch.

The two tests that matter are the oracle comparisons. A routing bug that
produces plausible outputs but wrong gradients is the classic way an MoE
silently fails to train, and nothing except a gradient comparison catches it.

The dense oracle lives HERE and only here. It evaluates every expert on every
token, which is exactly what the production path must never do.
"""

import pytest
import torch

from _helpers import VOCAB, tiny_cfg
from story_moe.model import StoryLM, count_parameters
from story_moe.moe import SparseMoE, balanced_reference_value

E, K, D, WIDTH = 4, 2, 32, 48


def moe_cfg(**overrides):
    base = dict(use_moe=True, n_experts=E, top_k=K, expert_width=WIDTH, dense_width=2 * WIDTH)
    base.update(overrides)
    return tiny_cfg(**base)


@pytest.fixture
def moe():
    torch.manual_seed(0)
    return SparseMoE(moe_cfg())


# --- the oracle ------------------------------------------------------------


def dense_oracle(module: SparseMoE, x: torch.Tensor) -> torch.Tensor:
    """Every expert on every token, combined with the SAME Top-k weights.

    Correct by construction and hopelessly wasteful. Its only job is to say what
    the sparse path should have produced.
    """
    B, T, d = x.shape
    flat = x.reshape(-1, d)
    _, topk_idx, weights = module.route(flat)
    weights = weights.to(x.dtype)

    all_out = torch.stack([e(flat) for e in module.experts], dim=1)   # [N, E, D]
    rows = torch.arange(flat.shape[0], device=flat.device)

    out = torch.zeros_like(flat)
    for slot in range(module.top_k):
        out = out + all_out[rows, topk_idx[:, slot]] * weights[:, slot].unsqueeze(-1)
    return out.view(B, T, d)


def _grads(fn, module, x, seed=7):
    """Run fn, backprop a fixed random projection, return (input grad, param grads)."""
    torch.manual_seed(seed)
    g = torch.randn(x.shape)
    module.zero_grad(set_to_none=True)
    xx = x.clone().requires_grad_(True)
    (fn(module, xx) * g).sum().backward()
    params = {
        n: (p.grad.clone() if p.grad is not None else torch.zeros_like(p))
        for n, p in module.named_parameters()
    }
    return xx.grad.clone(), params


# --- routing invariants ----------------------------------------------------


def test_output_shape_and_finiteness(moe):
    x = torch.randn(2, 16, D)
    out, aux, stats = moe(x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()
    assert torch.isfinite(aux)
    assert stats is None            # not requested


def test_exactly_k_distinct_assignments_per_token(moe):
    flat = torch.randn(64, D)
    probs, topk_idx, weights = moe.route(flat)

    assert topk_idx.shape == (64, K)
    assert probs.shape == (64, E)
    # k distinct experts per token -> exactly k*N assignments in total
    assert topk_idx.numel() == K * 64
    for row in topk_idx:
        assert len(set(row.tolist())) == K, "top-k returned a duplicate expert"


def test_selected_weights_sum_to_one(moe):
    flat = torch.randn(64, D)
    _, _, weights = moe.route(flat)
    torch.testing.assert_close(
        weights.sum(dim=-1), torch.ones(64), rtol=1e-5, atol=1e-6
    )
    assert (weights >= 0).all()


def test_full_probabilities_sum_to_one(moe):
    probs, _, _ = moe.route(torch.randn(64, D))
    torch.testing.assert_close(probs.sum(dim=-1), torch.ones(64), rtol=1e-5, atol=1e-6)
    assert probs.dtype == torch.float32, "router softmax must be float32"


def test_only_assigned_tokens_reach_each_expert(moe):
    """The whole point of sparsity. Dense evaluation would total E*N, not k*N."""
    seen: list[int] = []
    for expert in moe.experts:
        original = expert.forward
        expert.forward = (lambda f: lambda t: (seen.append(t.shape[0]), f(t))[1])(original)

    x = torch.randn(2, 16, D)
    moe(x)

    N = 2 * 16
    assert sum(seen) == K * N, f"expected {K * N} token-expert evaluations, got {sum(seen)}"
    assert sum(seen) < E * N, "this is dense evaluation, not sparse dispatch"


# --- agreement with the oracle --------------------------------------------


def test_sparse_output_matches_dense_oracle(moe):
    x = torch.randn(2, 16, D)
    with torch.no_grad():
        sparse, _, _ = moe(x)
        oracle = dense_oracle(moe, x)
    torch.testing.assert_close(sparse, oracle, rtol=1e-4, atol=1e-5)
    print(f"max abs error: {(sparse - oracle).abs().max():.3e}")


def test_gradients_match_dense_oracle(moe):
    """Input, expert and router gradients must all agree.

    This is the test that catches a scatter-add that drops a term, a weight
    applied to the wrong slot, or a detached routing weight.
    """
    x = torch.randn(2, 16, D)

    # every expert must be used somewhere, or the comparison is vacuous
    _, topk_idx, _ = moe.route(x.reshape(-1, D))
    assert set(topk_idx.reshape(-1).tolist()) == set(range(E)), "fixture leaves an expert unused"

    gx_sparse, gp_sparse = _grads(lambda m, t: m(t)[0], moe, x)
    gx_oracle, gp_oracle = _grads(dense_oracle, moe, x)

    torch.testing.assert_close(gx_sparse, gx_oracle, rtol=1e-4, atol=1e-5)
    assert set(gp_sparse) == set(gp_oracle)
    for name in gp_sparse:
        torch.testing.assert_close(
            gp_sparse[name], gp_oracle[name], rtol=1e-4, atol=1e-5,
            msg=lambda m, n=name: f"gradient mismatch for {n}\n{m}",
        )


def test_router_receives_gradient_from_the_language_path(moe):
    """Selected weights must not be detached, or the router never learns."""
    x = torch.randn(2, 16, D)
    moe.zero_grad(set_to_none=True)
    out, _, _ = moe(x, need_aux=False)          # no auxiliary loss in the graph
    out.sum().backward()

    g = moe.router.weight.grad
    assert g is not None, "router got no gradient at all"
    assert g.abs().sum() > 0, "router gradient is zero without the auxiliary loss"


# --- degenerate routing ----------------------------------------------------


class ConstRouter(torch.nn.Module):
    """Every token gets the same logits, so expert choice is fully controlled."""

    def __init__(self, logits):
        super().__init__()
        self.logits = torch.nn.Parameter(logits)

    def forward(self, x):
        return self.logits.unsqueeze(0).expand(x.shape[0], -1)


def test_empty_expert_does_not_crash(moe):
    # experts 2 and 3 always lose: nothing is ever routed to them
    moe.router = ConstRouter(torch.tensor([5.0, 4.0, -5.0, -6.0]))
    x = torch.randn(2, 16, D)

    out, aux, stats = moe(x, collect_stats=True)
    assert torch.isfinite(out).all()
    assert torch.isfinite(aux)
    assert stats.assignment_fraction[2] == 0.0
    assert stats.assignment_fraction[3] == 0.0
    torch.testing.assert_close(sum(stats.assignment_fraction), 1.0, rtol=0, atol=1e-6)

    out.sum().backward()   # an inactive expert simply has no gradient this step


def test_starved_routing_is_visible_in_the_statistics(moe):
    moe.router = ConstRouter(torch.tensor([5.0, 4.0, -5.0, -6.0]))
    _, _, stats = moe(torch.randn(2, 16, D), collect_stats=True)
    assert stats.assignment_fraction[:2] == [0.5, 0.5]
    assert stats.tokens == 32
    assert 0.0 <= stats.router_entropy <= float(torch.tensor(float(E)).log())


# --- the balancing objective ----------------------------------------------


def test_balance_loss_is_one_under_uniform_routing(moe):
    """Not zero. One. A healthy auxiliary loss sits near 1 and stays there.

    Assignments are hand-built rather than routed: a constant router produces
    exact ties, and top-k breaks ties by index, so it cannot actually spread
    tokens evenly. Here every expert gets exactly k*N/E of the assignments.
    """
    N = 8
    probs = torch.full((N, E), 1.0 / E)
    topk_idx = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 0]] * (N // 4))

    counts = torch.bincount(topk_idx.reshape(-1), minlength=E)
    assert counts.tolist() == [K * N // E] * E, "fixture is not balanced"

    loss = moe.balance_loss(probs, topk_idx)
    torch.testing.assert_close(loss, torch.tensor(1.0), rtol=1e-4, atol=1e-5)
    assert balanced_reference_value(E) == 1.0


def test_constant_router_gives_uniform_probabilities(moe):
    moe.router = ConstRouter(torch.zeros(E))
    probs, _, _ = moe.route(torch.randn(16, D))
    torch.testing.assert_close(probs[0], torch.full((E,), 1.0 / E), rtol=1e-5, atol=1e-6)


def test_balance_loss_exceeds_one_when_routing_collapses(moe):
    moe.router = ConstRouter(torch.tensor([9.0, 8.0, -9.0, -9.0]))
    flat = torch.randn(64, D)
    probs, topk_idx, _ = moe.route(flat)
    assert moe.balance_loss(probs, topk_idx).item() > 1.0


def test_balance_loss_gradient_flows_through_probabilities_only(moe):
    flat = torch.randn(64, D)
    probs, topk_idx, _ = moe.route(flat)
    moe.zero_grad(set_to_none=True)
    moe.balance_loss(probs, topk_idx).backward()
    assert moe.router.weight.grad.abs().sum() > 0


def test_need_aux_false_skips_the_reduction(moe):
    out, aux, stats = moe(torch.randn(2, 16, D), need_aux=False)
    assert aux is None and stats is None
    assert torch.isfinite(out).all()


# --- whole-model wiring ----------------------------------------------------


def test_moe_model_builds_and_separates_losses():
    torch.manual_seed(0)
    cfg = moe_cfg()
    m = StoryLM(cfg)
    x = torch.randint(0, VOCAB, (2, 16))
    y = torch.randint(0, VOCAB, (2, 16))

    out = m(x, targets=y, collect_stats=True)
    assert out.logits.shape == (2, 16, VOCAB)
    assert float(out.aux_loss) > 0.0, "MoE auxiliary loss should not be zero"
    expected = out.lm_loss + cfg.model.aux_loss_weight * out.aux_loss
    torch.testing.assert_close(out.total_loss, expected)
    assert out.router_stats is not None
    assert len(out.router_stats) == cfg.model.n_layers


def test_dense_model_auxiliary_loss_is_exactly_zero():
    m = StoryLM(tiny_cfg())
    out = m(torch.randint(0, VOCAB, (2, 16)), targets=torch.randint(0, VOCAB, (2, 16)))
    assert float(out.aux_loss) == 0.0
    torch.testing.assert_close(out.total_loss, out.lm_loss)
    assert out.router_stats is None


def test_inference_skips_the_auxiliary_reduction():
    """Section 12: no diagnostic work may distinguish the two benchmark paths."""
    m = StoryLM(moe_cfg())
    m.eval()
    with torch.no_grad():
        out = m(torch.randint(0, VOCAB, (1, 16)))
    assert float(out.aux_loss) == 0.0     # no terms collected
    assert out.router_stats is None


def test_moe_body_is_larger_than_dense_body():
    dense = count_parameters(StoryLM(tiny_cfg(dense_width=2 * WIDTH)))
    sparse = count_parameters(StoryLM(moe_cfg()))
    assert dense["embedding"] == sparse["embedding"]
    assert sparse["body"] > dense["body"]


def test_moe_overfits_a_tiny_batch():
    """Same gate the dense model passed: routing must not block learning."""
    torch.manual_seed(0)
    m = StoryLM(moe_cfg())
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
    assert out.lm_loss.item() < 0.2 * first
    assert acc > 0.9
