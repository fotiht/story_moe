"""Rotary Position Embeddings.

Convention used here: **adjacent even/odd pairs**. Feature dimension Dh is read
as Dh/2 two-dimensional pairs (0,1), (2,3), ..., and pair index i is rotated by
angle p * theta_i, where

    theta_i = base ** (-2*i / Dh)          i = 0 .. Dh/2 - 1

    a_rot = a*cos(p*theta_i) - b*sin(p*theta_i)
    b_rot = a*sin(p*theta_i) + b*cos(p*theta_i)

This is the GPT-J / interleaved layout. HuggingFace's Llama and GPT-NeoX rotate
halves instead, pairing (i, i + Dh/2). Both are valid and mathematically
equivalent under a permutation of the feature axis, but tensors from the two
conventions will not match element-wise. Noted in the README.

Two properties the tests pin down:

  - Position 0 is the identity (cos 0 = 1, sin 0 = 0).
  - The rotation is orthogonal per pair, so it preserves norms, and the dot
    product of a rotated query and a rotated key depends only on the difference
    of their positions. That relative property is the entire point of RoPE, and
    it is also why the KV cache can store rotated keys. A key rotated once at
    its absolute position stays correct as the sequence grows.

Q and K are rotated. V is never rotated.
"""

from __future__ import annotations

import torch


def rope_tables(
    d_head: int,
    max_seq_len: int,
    base: float = 10000.0,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute cos/sin for every position. Both are [max_seq_len, d_head//2].

    Always computed in float32 regardless of the model dtype. At long positions
    the angles are where low precision costs real accuracy.
    """
    if d_head % 2 != 0:
        raise ValueError(f"d_head must be even, got {d_head}")

    i = torch.arange(d_head // 2, dtype=torch.float32, device=device)
    theta = base ** (-2.0 * i / d_head)                      # [Dh/2]
    pos = torch.arange(max_seq_len, dtype=torch.float32, device=device)
    angles = pos.unsqueeze(1) * theta.unsqueeze(0)           # [T, Dh/2]
    return angles.cos(), angles.sin()


def apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    offset: int = 0,
) -> torch.Tensor:
    """Rotate [B, H, T, Dh] using absolute positions offset .. offset+T-1.

    A chunk of T new tokens arriving after P cached tokens occupies positions
    P .. P+T-1, so it is rotated with offset=P. Keys already in the cache were
    rotated when they arrived and must never be rotated again.
    """
    T, Dh = x.shape[-2], x.shape[-1]
    if offset + T > cos.shape[0]:
        raise ValueError(
            f"positions {offset}..{offset + T - 1} exceed the precomputed table "
            f"of length {cos.shape[0]}"
        )
    if Dh != 2 * cos.shape[1]:
        raise ValueError(f"head dim {Dh} does not match table pair count {cos.shape[1]}")

    c = cos[offset : offset + T].view(1, 1, T, Dh // 2)
    s = sin[offset : offset + T].view(1, 1, T, Dh // 2)

    pairs = x.float().unflatten(-1, (Dh // 2, 2))            # [B, H, T, Dh/2, 2]
    a, b = pairs[..., 0], pairs[..., 1]

    rotated = torch.stack([a * c - b * s, a * s + b * c], dim=-1)
    return rotated.flatten(-2).to(x.dtype)
