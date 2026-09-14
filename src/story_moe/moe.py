"""Sparse Top-k Mixture of Experts.

Replaces the feed-forward sublayer. Four experts, two selected per token, a
learned softmax router, and selected weights renormalized to sum to one.

Shapes, end to end:

    x                 [B, T, D]
    flat              [N, D]        N = B*T
    router logits     [N, E]        computed in float32
    probs             [N, E]        full softmax, BEFORE top-k truncation
    topk_idx          [N, k]        k distinct expert ids per token
    weights           [N, k]        selected probs / their sum -> rows sum to 1
    per-expert input  [n_e, D]      only the tokens that chose expert e
    output            [N, D] -> [B, T, D]

Three things here are easy to get wrong and are each pinned by a test:

  1. **Only assigned tokens are computed.** The production path gathers each
     expert's tokens, runs one MLP call on them, and scatter-adds the weighted
     result. Evaluating every expert on every token would be simpler and would
     defeat the entire point; that version exists only as a test oracle.

  2. **Selected weights are not detached.** The language loss has to be able to
     train the router through the selected probabilities. The top-k *indices*
     are discrete and carry no gradient, but the weights must.

  3. **The balancing loss is a Top-k adaptation of the Switch objective**, with
     the assignment fraction taken over k*N assignments rather than N. Its
     value under perfectly uniform routing is 1, not 0 -- see balance_loss.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Config


@dataclass
class RouterStats:
    """Detached diagnostics for one MoE layer. Never part of the graph."""

    assignment_fraction: list[float]   # over k*N assignments; sums to 1
    mean_probability: list[float]      # mean full softmax prob per expert; sums to 1
    router_entropy: float              # nats; ln(E) is maximum uncertainty
    aux_loss: float
    tokens: int

    def as_dict(self) -> dict:
        return {
            "assignment_fraction": self.assignment_fraction,
            "mean_probability": self.mean_probability,
            "router_entropy": self.router_entropy,
            "aux_loss": self.aux_loss,
            "tokens": self.tokens,
        }


class Expert(nn.Module):
    """One expert: Linear(D, m) -> GELU -> Linear(m, D). Same shape as the dense MLP."""

    def __init__(self, d_model: int, width: int, dropout: float = 0.0, bias: bool = False):
        super().__init__()
        self.fc_in = nn.Linear(d_model, width, bias=bias)
        self.fc_out = nn.Linear(width, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc_out(F.gelu(self.fc_in(x))))


class SparseMoE(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        m = cfg.model
        self.n_experts = m.n_experts
        self.top_k = m.top_k
        self.d_model = m.d_model

        self.router = nn.Linear(m.d_model, m.n_experts, bias=False)
        self.experts = nn.ModuleList(
            Expert(m.d_model, m.expert_width, m.dropout, m.bias) for _ in range(m.n_experts)
        )

    # -- routing -----------------------------------------------------------

    def route(self, flat: torch.Tensor):
        """[N, D] -> (probs [N, E], topk_idx [N, k], weights [N, k]).

        Softmax in float32: the router decides which experts see a token, and
        near-ties there turn into visibly different outputs downstream.
        """
        probs = F.softmax(self.router(flat).float(), dim=-1)
        topk_probs, topk_idx = probs.topk(self.top_k, dim=-1)
        # Renormalize so the selected weights sum to 1 per token. No detach:
        # this is the path the language loss uses to train the router.
        weights = topk_probs / topk_probs.sum(dim=-1, keepdim=True)
        return probs, topk_idx, weights

    # -- balancing objective -----------------------------------------------

    def balance_loss(self, probs: torch.Tensor, topk_idx: torch.Tensor) -> torch.Tensor:
        """Top-k adaptation of the Switch load-balancing objective.

            f_e = (assignments to expert e) / (k * N)        sums to 1 over e
            P_e = mean over tokens of the full softmax prob  sums to 1 over e
            L   = E * sum_e f_e * P_e

        Under uniform routing f_e = P_e = 1/E, so L = E * E * (1/E)(1/E) = 1.
        **The balanced value is 1, not 0.** An auxiliary loss sitting near 1 is
        healthy; do not chase it toward zero.

        f_e comes from discrete counts and carries no gradient -- it is detached
        explicitly so that is obvious rather than incidental. Gradients reach
        the router through P_e, which uses the full probabilities from BEFORE
        top-k truncation.
        """
        N, E = probs.shape
        k = self.top_k

        counts = torch.zeros(E, device=probs.device, dtype=probs.dtype)
        counts.scatter_add_(
            0,
            topk_idx.reshape(-1),
            torch.ones(N * k, device=probs.device, dtype=probs.dtype),
        )
        f = (counts / (k * N)).detach()
        P = probs.mean(dim=0)
        return E * torch.sum(f * P)

    def _stats(self, probs, topk_idx, aux) -> RouterStats:
        with torch.no_grad():
            N, E = probs.shape
            counts = torch.zeros(E, device=probs.device, dtype=probs.dtype)
            counts.scatter_add_(
                0, topk_idx.reshape(-1),
                torch.ones(N * self.top_k, device=probs.device, dtype=probs.dtype),
            )
            entropy = -(probs * probs.clamp_min(1e-9).log()).sum(-1).mean()
            return RouterStats(
                assignment_fraction=(counts / (self.top_k * N)).tolist(),
                mean_probability=probs.mean(dim=0).tolist(),
                router_entropy=float(entropy),
                aux_loss=float(aux),
                tokens=int(N),
            )

    # -- forward -----------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        need_aux: bool = True,
        collect_stats: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, RouterStats | None]:
        """[B, T, D] -> ([B, T, D], aux_loss or None, stats or None).

        `need_aux=False` skips the balancing reduction entirely, which is what
        the inference and benchmark paths want: no diagnostic work should
        distinguish the cached path from the uncached one.
        """
        B, T, D = x.shape
        flat = x.reshape(-1, D)

        probs, topk_idx, weights = self.route(flat)
        weights = weights.to(x.dtype)

        out = torch.zeros_like(flat)
        for e in range(self.n_experts):
            # Rows of topk_idx equal to e, and which of the k slots they used.
            token_idx, slot_idx = (topk_idx == e).nonzero(as_tuple=True)
            if token_idx.numel() == 0:
                continue        # an expert nobody chose contributes nothing this step
            expert_out = self.experts[e](flat[token_idx])
            out = out.index_add(
                0, token_idx, expert_out * weights[token_idx, slot_idx].unsqueeze(-1)
            )

        aux = self.balance_loss(probs, topk_idx) if (need_aux or collect_stats) else None
        stats = self._stats(probs, topk_idx, aux) if collect_stats else None
        return out.view(B, T, D), aux, stats


def balanced_reference_value(n_experts: int) -> float:
    """What balance_loss returns under perfectly uniform routing. Always 1.0.

    Kept as a named function so logs and the README can state the target
    instead of implying the auxiliary loss should approach zero.
    """
    return 1.0
