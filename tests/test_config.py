"""Config validation, and the parity guard for the dense-vs-MoE comparison.

test_dense_and_moe_configs_differ_only_in_feedforward is the one that matters.
The whole comparison is void if the two runs disagree about context, seed,
batch size, precision or data. Keep this test passing, or disclose the
difference in the README.
"""

from pathlib import Path

import pytest
import yaml

from story_moe.config import Config, DataConfig, ModelConfig, load_config

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def test_debug_configs_load_and_validate():
    for name in ("debug_dense.yaml", "debug_moe.yaml", "train_dense.yaml", "train_moe.yaml"):
        cfg = load_config(CONFIG_DIR / name)
        assert cfg.model.d_model == cfg.model.n_heads * cfg.model.d_head
        assert cfg.model.d_head % 2 == 0
        assert cfg.data.block_size <= cfg.model.max_seq_len


def test_default_stride_is_block_size():
    cfg = load_config(CONFIG_DIR / "debug_dense.yaml")
    assert cfg.data.stride is None
    assert cfg.data.resolved_stride() == cfg.data.block_size


def test_rejects_mismatched_head_dims():
    with pytest.raises(ValueError):
        ModelConfig(n_layers=2, d_model=128, n_heads=5, d_head=32, max_seq_len=256).validate()


def test_rejects_odd_head_dim():
    with pytest.raises(ValueError):
        ModelConfig(n_layers=2, d_model=126, n_heads=2, d_head=63, max_seq_len=256).validate()


def test_rejects_top_k_above_n_experts():
    with pytest.raises(ValueError):
        ModelConfig(
            n_layers=2, d_model=128, n_heads=4, d_head=32, max_seq_len=256,
            use_moe=True, n_experts=4, top_k=9,
        ).validate()


def test_rejects_block_size_above_context():
    cfg = Config(
        name="bad",
        model=ModelConfig(n_layers=2, d_model=128, n_heads=4, d_head=32, max_seq_len=128),
        data=DataConfig(block_size=256),
    )
    with pytest.raises(ValueError):
        cfg.validate()


@pytest.mark.parametrize("pair", [("debug_dense", "debug_moe"), ("train_dense", "train_moe")])
def test_dense_and_moe_configs_differ_only_in_feedforward(pair):
    dense = yaml.safe_load((CONFIG_DIR / f"{pair[0]}.yaml").read_text(encoding="utf-8"))
    moe = yaml.safe_load((CONFIG_DIR / f"{pair[1]}.yaml").read_text(encoding="utf-8"))

    allowed = {"use_moe", "n_experts", "top_k", "expert_width", "dense_width", "aux_loss_weight"}
    keys = set(dense["model"]) | set(moe["model"])
    differing = {k for k in keys if dense["model"].get(k) != moe["model"].get(k)}
    assert differing <= allowed, f"uncontrolled model difference: {sorted(differing - allowed)}"

    assert dense["data"] == moe["data"], "data configs must be identical"

    d_train, m_train = dict(dense["train"]), dict(moe["train"])
    d_train.pop("out_dir"), m_train.pop("out_dir")
    assert d_train == m_train, "training configs must match except out_dir"
