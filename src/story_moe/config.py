"""Validated configuration objects loaded from YAML.

Every run is described by one YAML file. The dataclasses here are the single
source of truth for shapes and hyperparameters; nothing else should hardcode a
dimension. `load_config` validates the invariants the project depends on
(D == H * Dh, even head dim, k <= E, block_size <= max_seq_len) so a bad config
fails at startup instead of deep inside a training loop.

Note on vocab_size: it is NOT set in YAML. It is resolved at runtime from the
tokenizer via len(tokenizer) and written into the config, so the model can never
disagree with the tokenizer that produced the data.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    n_layers: int
    d_model: int
    n_heads: int
    d_head: int
    max_seq_len: int

    # Feed-forward sublayer. use_moe switches between the dense MLP and Top-k MoE.
    use_moe: bool = False
    dense_width: int = 512       # intermediate width of the dense MLP
    n_experts: int = 4
    top_k: int = 2
    expert_width: int = 256      # intermediate width of ONE expert
    aux_loss_weight: float = 0.01

    rope_base: float = 10000.0
    dropout: float = 0.0
    tie_weights: bool = True
    bias: bool = False           # bias-free linear projections (simple param accounting)

    # Resolved from the tokenizer at load time; never written by hand.
    vocab_size: int | None = None

    def validate(self) -> None:
        if self.d_model != self.n_heads * self.d_head:
            raise ValueError(
                f"d_model ({self.d_model}) must equal n_heads * d_head "
                f"({self.n_heads} * {self.d_head} = {self.n_heads * self.d_head})"
            )
        if self.d_head % 2 != 0:
            raise ValueError(f"d_head must be even for RoPE pairing, got {self.d_head}")
        if self.use_moe and not (1 <= self.top_k <= self.n_experts):
            raise ValueError(f"top_k ({self.top_k}) must be in [1, n_experts={self.n_experts}]")
        if self.n_layers < 1 or self.d_model < 1:
            raise ValueError("n_layers and d_model must be positive")


@dataclass
class DataConfig:
    dataset: str = "roneneldan/TinyStories"
    tokenizer: str = "gpt2"
    train_stories: int = 1000
    val_stories: int = 200
    seed: int = 1234

    block_size: int = 128        # T: tokens the model sees per example
    # Stride between successive block starts, in tokens. Default block_size means
    # consecutive blocks share exactly one boundary token, so every token after the
    # first is a prediction target exactly once and none is wasted.
    stride: int | None = None

    cache_dir: str = "data/cache"

    def resolved_stride(self) -> int:
        return self.block_size if self.stride is None else self.stride

    def validate(self) -> None:
        if self.train_stories < 1 or self.val_stories < 1:
            raise ValueError("train_stories and val_stories must be positive")
        if self.resolved_stride() < 1:
            raise ValueError("stride must be positive")


@dataclass
class TrainConfig:
    lr: float = 3e-4
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0
    warmup_steps: int = 100
    microbatch_size: int = 4
    accum_steps: int = 8
    max_tokens: int = 1_000_000   # processed-token budget (counts repeated passes)
    eval_every: int = 250
    ckpt_every: int = 250
    out_dir: str = "checkpoints"

    # Device/precision pinning. The dense and MoE runs must agree on both or the
    # comparison in the project spec (same device, same precision) is void. Colab
    # reassigns GPUs between sessions, so these are asserted at startup rather
    # than trusted.
    device: str = "cuda"
    precision: str = "fp32"          # fp32 | fp16 | bf16
    require_gpu_name: str | None = None   # substring match, e.g. "A100"


@dataclass
class Config:
    name: str
    model: ModelConfig
    data: DataConfig
    train: TrainConfig = field(default_factory=TrainConfig)

    def validate(self) -> None:
        self.model.validate()
        self.data.validate()
        if self.data.block_size > self.model.max_seq_len:
            raise ValueError(
                f"block_size ({self.data.block_size}) exceeds max_seq_len "
                f"({self.model.max_seq_len})"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(path: str | Path) -> Config:
    """Read a YAML config, build the dataclasses, and validate invariants."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    cfg = Config(
        name=raw.get("name", path.stem),
        model=ModelConfig(**raw["model"]),
        data=DataConfig(**raw.get("data", {})),
        train=TrainConfig(**raw.get("train", {})),
    )
    cfg.validate()
    return cfg
