"""Pre-norm decoder block and the language model.

Attention, the loss, RoPE, the sparse Top-k MoE feed-forward, and the KV cache.
Position comes from RoPE alone. There is no positional embedding table.

Dense and MoE share every line of this file except which module fills the
feed-forward slot, so any measured difference between them comes from that slot.

Parameter accounting: with the GPT-2 vocabulary (50,257) and a small d_model the
tied embedding dominates the total, so count_parameters() reports embedding and
body separately. Quote the body number when comparing dense against MoE.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import CausalSelfAttention
from .cache import KVCache
from .config import Config
from .moe import FeedForward, RouterStats, SparseMoE
from .rope import rope_tables


@dataclass
class ModelOutput:
    """Losses are kept separate on purpose.

    lm_loss is the next-token cross-entropy and the only one of the three that
    becomes a reported perplexity. aux_loss is the MoE balancing term (0.0 for a
    dense model). total_loss = lm_loss + aux_weight * aux_loss is what the
    optimizer sees. Never quote total_loss as a language-modeling result.
    """

    logits: torch.Tensor
    lm_loss: torch.Tensor | None = None
    aux_loss: torch.Tensor | None = None
    total_loss: torch.Tensor | None = None
    # The same KVCache object that was passed in, advanced by this call. Handed
    # back so a generation loop can read it off the output instead of keeping a
    # second reference. It is the same object, not a copy.
    past_key_values: "KVCache | None" = None
    router_stats: list[RouterStats] | None = None  # detached, opt-in


class DecoderBlock(nn.Module):
    """Pre-norm: x = x + attn(norm1(x)); x = x + ff(norm2(x))."""

    def __init__(self, cfg: Config, layer_idx: int = 0):
        super().__init__()
        m = cfg.model
        self.layer_idx = layer_idx
        self.norm1 = nn.LayerNorm(m.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.norm2 = nn.LayerNorm(m.d_model)
        # The one architectural difference between the two models.
        self.use_moe = m.use_moe
        self.feed_forward = (
            SparseMoE(cfg) if m.use_moe
            else FeedForward(m.d_model, m.dense_width, m.dropout, m.bias)
        )

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        need_aux: bool = True,
        collect_stats: bool = False,
        cache: KVCache | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, RouterStats | None]:
        x = x + self.attn(self.norm1(x), cos, sin, cache=cache, layer_idx=self.layer_idx)
        h = self.norm2(x)
        if self.use_moe:
            ff, aux, stats = self.feed_forward(h, need_aux=need_aux, collect_stats=collect_stats)
        else:
            ff, aux, stats = self.feed_forward(h), None, None
        return x + ff, aux, stats


class StoryLM(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        m = cfg.model
        if m.vocab_size is None:
            raise ValueError(
                "cfg.model.vocab_size is unset. Resolve it from the tokenizer "
                "(len(tokenizer)) before building the model."
            )
        self.cfg = cfg
        self.token_emb = nn.Embedding(m.vocab_size, m.d_model)
        self.drop = nn.Dropout(m.dropout)

        # Position comes only from RoPE. There is no positional embedding table.
        # Buffers so .to(device) moves them. Non-persistent so they are rebuilt
        # from the config on load instead of bloating every checkpoint.
        cos, sin = rope_tables(m.d_head, m.max_seq_len, m.rope_base)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.blocks = nn.ModuleList(
            DecoderBlock(cfg, layer_idx=i) for i in range(m.n_layers)
        )
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
        collect_stats: bool = False,
        cache: KVCache | None = None,
    ) -> ModelOutput:
        """[B, T] -> ModelOutput.

        logits_to_keep=k projects only the last k positions, giving [B, k, V].
        Generation uses k=1. Training and evaluation need every position, so
        passing both targets and logits_to_keep raises instead of scoring a
        suffix and calling it the loss.

        With `cache`, this is an incremental step. input_ids holds only the
        tokens that are not in the cache yet. They occupy positions
        cache.length .. cache.length+T-1, and the cache is advanced by T before
        returning. Pass the whole prompt for prefill and one token per step
        after that.
        """
        if input_ids.dtype not in (torch.int64, torch.int32):
            raise TypeError(f"input_ids must be integer dtype, got {input_ids.dtype}")
        if targets is not None and logits_to_keep:
            raise ValueError("targets requires full logits; do not pass logits_to_keep")
        if cache is not None and targets is not None:
            raise ValueError(
                "the cache is for generation, not training; training recomputes "
                "every position and needs gradients through the keys and values"
            )

        B, T = input_ids.shape
        past_len = cache.length if cache is not None else 0
        if past_len + T > self.cfg.model.max_seq_len:
            raise ValueError(
                f"positions {past_len}..{past_len + T - 1} exceed max_seq_len "
                f"{self.cfg.model.max_seq_len}"
            )
        if cache is not None and cache.n_layers != len(self.blocks):
            raise ValueError(
                f"cache holds {cache.n_layers} layers, model has {len(self.blocks)}"
            )

        x = self.drop(self.token_emb(input_ids))

        # The balancing reduction runs only when it can be optimized. Inference
        # and the cache benchmark skip it, so no diagnostic work separates those
        # paths from each other.
        need_aux = targets is not None
        aux_terms: list[torch.Tensor] = []
        stats: list[RouterStats] = []

        for block in self.blocks:
            x, aux, stat = block(
                x, self.rope_cos, self.rope_sin,
                need_aux=need_aux, collect_stats=collect_stats, cache=cache,
            )
            if aux is not None:
                aux_terms.append(aux)
            if stat is not None:
                stats.append(stat)

        # Every layer appended the same T positions, so the shared length moves
        # once, here, after the last of them. Advancing inside append() would
        # skew layer 1 one chunk past layer 0.
        if cache is not None:
            cache.advance(T)

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

        # Averaged across MoE layers. A dense model contributes no terms, so its
        # auxiliary loss is exactly zero and total_loss == lm_loss.
        if aux_terms:
            aux_loss = torch.stack(aux_terms).mean()
        else:
            aux_loss = torch.zeros((), device=logits.device, dtype=torch.float32)

        total_loss = None
        if lm_loss is not None:
            total_loss = lm_loss + self.cfg.model.aux_loss_weight * aux_loss

        return ModelOutput(
            logits=logits,
            lm_loss=lm_loss,
            aux_loss=aux_loss,
            total_loss=total_loss,
            past_key_values=cache,
            router_stats=stats or None,
        )

    def new_cache(self, max_len: int | None = None) -> KVCache:
        """An empty cache sized for this model. Capacity defaults to max_seq_len."""
        limit = self.cfg.model.max_seq_len
        if max_len is None:
            max_len = limit
        elif max_len > limit:
            raise ValueError(f"max_len {max_len} exceeds max_seq_len {limit}")
        return KVCache(n_layers=len(self.blocks), max_len=max_len)


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Unique trainable parameters, split into embedding and body.

    Tied weights are counted once. Parameters are deduplicated by id() before
    summing, so a tied lm_head does not inflate the total. RoPE contributes no
    parameters, so "embedding" here is exactly the token embedding.
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
    """Cross-entropy of a uniform predictor. An untrained model starts here.

    For GPT-2's 50,257-token vocabulary this is ln(50257) = 10.825. A first-step
    loss far from it means a broken vocab size, a broken shift, or a broken tie.
    """
    return math.log(vocab_size)
