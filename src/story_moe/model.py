"""Dense MLP, pre-norm decoder block, and the language model.

DAY 3 SCOPE. Attention, the loss, and RoPE. The temporary learned positional
embedding from Day 2 has been DELETED, not merely disabled -- the final
architecture must not carry both. MoE and the KV cache are not here yet.

Parameter accounting: with the GPT-2 vocabulary (50,257) and a small d_model the
tied embedding dominates the total, so count_parameters() reports embedding and
body separately. The body number is the one to quote when comparing dense
against MoE.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import CausalSelfAttention
from .config import Config
from .rope import rope_tables


@dataclass
class ModelOutput:
    """Losses are kept separate on purpose.

    lm_loss is the next-token cross-entropy and is the ONLY thing reported as
    perplexity. aux_loss is the MoE balancing term (0.0 for a dense model).
    total_loss = lm_loss + aux_weight * aux_loss is what the optimizer sees.
    Never quote total_loss as a language-modeling result.
    """

    logits: torch.Tensor
    lm_loss: torch.Tensor | None = None
    aux_loss: torch.Tensor | None = None
    total_loss: torch.Tensor | None = None
    past_key_values: list | None = None      # Day 6
    router_stats: list | None = None         # Day 5


class DenseMLP(nn.Module):
    """Linear(D, width) -> GELU -> Linear(width, D). Also the shape of one expert."""

    def __init__(self, d_model: int, width: int, dropout: float = 0.0, bias: bool = False):
        super().__init__()
        self.fc_in = nn.Linear(d_model, width, bias=bias)
        self.fc_out = nn.Linear(width, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc_out(F.gelu(self.fc_in(x))))


class DecoderBlock(nn.Module):
    """Pre-norm: x = x + attn(norm1(x)); x = x + ff(norm2(x))."""

    def __init__(self, cfg: Config):
        super().__init__()
        m = cfg.model
        self.norm1 = nn.LayerNorm(m.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.norm2 = nn.LayerNorm(m.d_model)
        # Day 4 swaps this for the Top-k MoE when cfg.model.use_moe is set.
        self.feed_forward = DenseMLP(m.d_model, m.dense_width, m.dropout, m.bias)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        past_len: int = 0,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), cos, sin, past_len=past_len)
        x = x + self.feed_forward(self.norm2(x))
        return x


class StoryLM(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        m = cfg.model
        if m.vocab_size is None:
            raise ValueError(
                "cfg.model.vocab_size is unset. Resolve it from the tokenizer "
                "(len(tokenizer)) before building the model."
            )
        if m.use_moe:
            raise NotImplementedError("MoE arrives on Day 4; use a dense config for now")

        self.cfg = cfg
        self.token_emb = nn.Embedding(m.vocab_size, m.d_model)
        self.drop = nn.Dropout(m.dropout)

        # Position comes only from RoPE. There is no positional embedding table.
        # Buffers so .to(device) moves them; non-persistent so they are rebuilt
        # from the config on load instead of bloating every checkpoint.
        cos, sin = rope_tables(m.d_head, m.max_seq_len, m.rope_base)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.blocks = nn.ModuleList(DecoderBlock(cfg) for _ in range(m.n_layers))
        self.final_norm = nn.LayerNorm(m.d_model)
        self.lm_head = nn.Linear(m.d_model, m.vocab_size, bias=False)
        if m.tie_weights:
            self.lm_head.weight = self.token_emb.weight

        self.apply(self._init_weights)
        # Scale down the projections that write into the residual stream, so the
        # residual variance does not grow with depth (GPT-2's 1/sqrt(2*n_layers)).
        residual_std = 0.02 / math.sqrt(2 * m.n_layers)
        for name, p in self.named_parameters():
            if name.endswith("out_proj.weight") or name.endswith("fc_out.weight"):
                nn.init.normal_(p, mean=0.0, std=residual_std)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        logits_to_keep: int | None = None,
    ) -> ModelOutput:
        """[B, T] -> ModelOutput.

        logits_to_keep=k projects only the last k positions, giving [B, k, V].
        Generation uses k=1; training and evaluation need every position, so
        passing both targets and logits_to_keep is rejected rather than silently
        scoring a suffix.
        """
        if input_ids.dtype not in (torch.int64, torch.int32):
            raise TypeError(f"input_ids must be integer dtype, got {input_ids.dtype}")
        if targets is not None and logits_to_keep:
            raise ValueError("targets requires full logits; do not pass logits_to_keep")

        B, T = input_ids.shape
        if T > self.cfg.model.max_seq_len:
            raise ValueError(f"sequence length {T} exceeds max_seq_len {self.cfg.model.max_seq_len}")

        x = self.drop(self.token_emb(input_ids))

        for block in self.blocks:
            x = block(x, self.rope_cos, self.rope_sin, past_len=0)
        x = self.final_norm(x)

        if logits_to_keep:
            x = x[:, -logits_to_keep:, :]
        logits = self.lm_head(x)

        lm_loss = None
        if targets is not None:
            if targets.shape != input_ids.shape:
                raise ValueError(f"targets {tuple(targets.shape)} != inputs {tuple(input_ids.shape)}")
            lm_loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
            )

        # Dense model: the balancing term is exactly zero, and total == language.
        aux_loss = torch.zeros((), device=logits.device, dtype=logits.dtype)
        total_loss = None if lm_loss is None else lm_loss + 0.0 * aux_loss

        return ModelOutput(
            logits=logits, lm_loss=lm_loss, aux_loss=aux_loss, total_loss=total_loss
        )


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Unique trainable parameters, split into embedding and body.

    Tied weights are counted once: parameters are deduplicated by id() before
    summing, so a tied lm_head does not inflate the total. RoPE contributes no
    parameters at all, so "embedding" here is exactly the token embedding.
    """
    embedding_ids: set[int] = set()
    for module in model.modules():
        if isinstance(module, nn.Embedding):
            for p in module.parameters(recurse=False):
                embedding_ids.add(id(p))

    unique = {id(p): p for p in model.parameters() if p.requires_grad}
    total = sum(p.numel() for p in unique.values())
    embedding = sum(p.numel() for pid, p in unique.items() if pid in embedding_ids)
    return {"total": total, "embedding": embedding, "body": total - embedding}


def expected_initial_loss(vocab_size: int) -> float:
    """Cross-entropy of a uniform predictor: an untrained model must start here.

    For GPT-2's 50,257-token vocabulary this is ln(50257) = 10.825. A first-step
    loss far from it means a broken vocab size, a broken shift, or a broken tie.
    """
    return math.log(vocab_size)
