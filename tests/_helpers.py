"""Shared test fixtures: a small model config that needs no tokenizer or network."""

from story_moe.config import Config, DataConfig, ModelConfig

VOCAB = 97


def tiny_cfg(**overrides) -> Config:
    model_kwargs = dict(
        n_layers=2, d_model=32, n_heads=4, d_head=8, max_seq_len=32,
        dense_width=64, dropout=0.0, vocab_size=VOCAB,
    )
    model_kwargs.update(overrides)
    cfg = Config(
        name="tiny",
        model=ModelConfig(**model_kwargs),
        data=DataConfig(block_size=16),
    )
    cfg.validate()
    return cfg
