"""Training-reliability gates: sampling, cache/config agreement, resume state.

These cover three failure modes that produce *plausible* runs rather than
crashes, which is what makes them dangerous:

  - sampling with replacement while the log claims "one pass"
  - training on a cache whose block size disagrees with the config, so every
    processed-token count is off by the block-size ratio
  - a resume that loses the sampler position

The GPU half of the resume fix (RNG byte tensors must stay on CPU) cannot be
tested here; it needs an actual CUDA device. See the GPU resume cell in
notebooks/story_moe_colab.ipynb.
"""

import ast
import copy
import pathlib

import numpy as np
import pytest

import story_moe
from _helpers import tiny_cfg
from story_moe.data import Batcher, check_cache_matches_config


def blocks(n=13, t=9):
    return np.arange(n * t, dtype=np.uint16).reshape(n, t)


# --- sampling without replacement ------------------------------------------


def test_one_epoch_visits_every_block_exactly_once():
    """With replacement, n draws from n blocks reach only ~63% of them."""
    arr = blocks(n=12, t=9)
    b = Batcher(arr, batch_size=3, seed=0)

    seen = []
    for _ in range(4):                       # 4 batches x 3 = 12 = one epoch
        x, _ = b.next_batch()
        seen.extend(int(row[0]) for row in x)

    assert sorted(seen) == sorted(int(r[0]) for r in arr)
    assert len(set(seen)) == 12, "a block was drawn twice inside a single epoch"


def test_epoch_boundary_reshuffles_and_counts():
    arr = blocks(n=10, t=9)
    b = Batcher(arr, batch_size=4, seed=0)

    assert b.epochs_seen() == 0.0
    b.next_batch(); b.next_batch()            # 8 of 10
    assert b.epoch == 0 and b.cursor == 8
    assert abs(b.epochs_seen() - 0.8) < 1e-9

    b.next_batch()                            # wraps: 2 left + 2 from the next epoch
    assert b.epoch == 1
    assert b.epochs_seen() > 1.0


def test_batch_shapes_and_shift():
    arr = blocks(n=8, t=9)
    x, y = Batcher(arr, batch_size=4, seed=0).next_batch()
    assert x.shape == (4, 8) and y.shape == (4, 8)
    assert (y[:, :-1] == x[:, 1:]).all()


def test_unshuffled_batcher_is_in_order():
    arr = blocks(n=6, t=9)
    b = Batcher(arr, batch_size=2, seed=0, shuffle=False)
    x, _ = b.next_batch()
    assert [int(r[0]) for r in x] == [0, 9]


# --- sampler state survives a resume ---------------------------------------


def test_sampler_state_round_trips():
    arr = blocks(n=12, t=9)
    a = Batcher(arr, batch_size=3, seed=0)
    a.next_batch(); a.next_batch()
    state = copy.deepcopy(a.state_dict())

    fresh = Batcher(arr, batch_size=3, seed=999)   # deliberately a different seed
    fresh.load_state_dict(state)

    assert fresh.cursor == a.cursor and fresh.epoch == a.epoch
    xa, _ = a.next_batch()
    xb, _ = fresh.next_batch()
    assert (xa == xb).all(), "resumed sampler diverged from the original"


def test_sampler_state_rejects_a_different_cache_size():
    a = Batcher(blocks(n=12), batch_size=3, seed=0)
    state = a.state_dict()
    with pytest.raises(ValueError):
        Batcher(blocks(n=20), batch_size=3, seed=0).load_state_dict(state)


# --- cache / config agreement ----------------------------------------------


def good_meta(cfg, split="train", n=None):
    return {
        "split": split,
        "block_size": cfg.data.block_size,
        "stride": cfg.data.resolved_stride(),
        "tokenizer": {
            "name_or_path": cfg.data.tokenizer,
            "vocab_size_len": cfg.model.vocab_size,
        },
        "provenance": {
            "dataset": cfg.data.dataset,
            "seed": cfg.data.seed,
            "n_selected": n if n is not None else cfg.data.train_stories,
        },
    }


def test_matching_cache_is_accepted():
    cfg = tiny_cfg()
    check_cache_matches_config(good_meta(cfg), cfg, "train")


def test_wrong_block_size_is_rejected():
    """The silent-failure case: a 512 config reading the 128 cache runs fine and
    overcounts processed tokens by 4x."""
    cfg = tiny_cfg()
    meta = good_meta(cfg)
    meta["block_size"] = cfg.data.block_size * 4
    with pytest.raises(ValueError, match="block_size"):
        check_cache_matches_config(meta, cfg, "train")


@pytest.mark.parametrize(
    "mutate, pattern",
    [
        (lambda m: m.update(stride=3), "stride"),
        (lambda m: m["tokenizer"].update(name_or_path="gpt2-large"), "tokenizer"),
        (lambda m: m["tokenizer"].update(vocab_size_len=123), "vocab size"),
        (lambda m: m["provenance"].update(dataset="something/else"), "dataset"),
        (lambda m: m["provenance"].update(seed=999), "seed"),
        (lambda m: m["provenance"].update(n_selected=7), "stories"),
    ],
)
def test_every_mismatch_is_reported(mutate, pattern):
    cfg = tiny_cfg()
    meta = good_meta(cfg)
    mutate(meta)
    with pytest.raises(ValueError, match=pattern):
        check_cache_matches_config(meta, cfg, "train")


def test_validation_split_checks_against_val_stories():
    cfg = tiny_cfg()
    meta = good_meta(cfg, split="validation", n=cfg.data.val_stories)
    check_cache_matches_config(meta, cfg, "validation")
    with pytest.raises(ValueError, match="stories"):
        check_cache_matches_config(meta, cfg, "train")


# --- call sites stay in sync with the Batcher API --------------------------


def test_no_call_site_uses_a_missing_batcher_method():
    """Catch a rename that updates some call sites and not others.

    This exact bug shipped: `Batcher.random_batch` became `next_batch`,
    `train.py` was updated, the CLI in `data.py` was not, and the whole suite
    passed because nothing exercises that display path. The failure surfaced as
    an AttributeError three steps into a Colab session.

    Static rather than behavioural: it walks every `batcher.<attr>` in the
    package and checks the attribute exists. It only sees call sites where the
    variable is literally named `batcher`, which is the convention here.
    """
    known = {name for name in dir(Batcher) if not name.startswith("_")}
    problems = []

    for path in sorted(pathlib.Path(story_moe.__file__).parent.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "batcher"
                and node.attr not in known
            ):
                problems.append(f"{path.name}:{node.lineno} calls batcher.{node.attr}")

    assert not problems, "Batcher call sites out of sync:\n  " + "\n  ".join(problems)
