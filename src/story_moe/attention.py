"""Multi-head causal self-attention, written out explicitly.

This is the reference implementation: projections, scaled dot product, mask,
softmax, weighted sum, merge heads, output projection. PyTorch SDPA is faster
but its mask semantics are easy to get wrong with a cached offset, so it is only
introduced later and only after matching these numbers.

Shapes, tracked at every step:

    x                 [B, T, D]
    q / k / v         [B, T, D]  -> view [B, T, H, Dh] -> transpose [B, H, T, Dh]
    scores            [B, H, T_q, T_k]
    mask              [T_q, T_k]  broadcast over B and H
    attn @ v          [B, H, T_q, Dh]
    merged            [B, T_q, D]
    output            [B, T_q, D]
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Config
from .rope import apply_rope


def offset_causal_mask(
    q_len: int,
    k_len: int,
    past_len: int,
    device: torch.device,
) -> torch.Tensor:
    """Boolean mask, True where key j may be attended by query i.

    The rule is  j <= past_len + i.  Query i is the (past_len + i)-th token of
    the full sequence, so it may see every key up to and including itself.

    With past_len == 0 and q_len == k_len this is exactly a lower triangle. The
    general form is written now, rather than a tril, so the KV-cache milestone
    does not have to replace the masking logic -- only pass a nonzero past_len.
    """
    i = torch.arange(q_len, device=device).unsqueeze(1)   # [T_q, 1]
    j = torch.arange(k_len, device=device).unsqueeze(0)   # [1, T_k]
    return j <= (past_len + i)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        m = cfg.model
        self.n_heads = m.n_heads
        self.d_head = m.d_head
        self.d_model = m.d_model
        self.scale = 1.0 / math.sqrt(m.d_head)

        self.q_proj = nn.Linear(m.d_model, m.d_model, bias=m.bias)
        self.k_proj = nn.Linear(m.d_model, m.d_model, bias=m.bias)
        self.v_proj = nn.Linear(m.d_model, m.d_model, bias=m.bias)
        self.out_proj = nn.Linear(m.d_model, m.d_model, bias=m.bias)

        self.attn_dropout = nn.Dropout(m.dropout)
        self.resid_dropout = nn.Dropout(m.dropout)

    def _split_heads(self, t: torch.Tensor) -> torch.Tensor:
        """[B, T, D] -> [B, H, T, Dh]"""
        B, T, _ = t.shape
        return t.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

    def _merge_heads(self, t: torch.Tensor) -> torch.Tensor:
        """[B, H, T, Dh] -> [B, T, D]"""
        B, H, T, Dh = t.shape
        return t.transpose(1, 2).contiguous().view(B, T, H * Dh)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        past_len: int = 0,
    ) -> torch.Tensor:
        B, T, D = x.shape
        if D != self.d_model:
            raise ValueError(f"expected last dim {self.d_model}, got {D}")

        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))

        # RoPE on queries and keys only. V carries content, not position, and
        # rotating it would corrupt the values the attention weights average.
        # These T tokens occupy absolute positions past_len .. past_len+T-1.
        q = apply_rope(q, cos, sin, offset=past_len)
        k = apply_rope(k, cos, sin, offset=past_len)

        scores = (q @ k.transpose(-2, -1)) * self.scale      # [B, H, T, T]

        allowed = offset_causal_mask(T, k.shape[-2], past_len, x.device)
        scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)

        attn = self.attn_dropout(F.softmax(scores, dim=-1))
        out = self._merge_heads(attn @ v)                    # [B, T, D]
        return self.resid_dropout(self.out_proj(out))
