"""Day 7: the benchmark harness itself, on CPU with a tiny model.

These do not check that the cache is fast. Speed is hardware, and asserting a
speedup on whatever machine CI happens to use produces a flaky test that
everyone learns to ignore. What is checked here is that the harness reports
honestly: that it compares two paths computing the same thing, that its
bookkeeping subtracts prefill from the cached total before dividing, and that
its grid respects the model's context limit.
"""

from __future__ import annotations

import pytest
import torch

from contextlib import nullcontext

from story_moe.benchmark import (
    _cache_reserved_mib,
    _fmt,
    benchmark_point,
    default_grid,
    lockstep_compare,
    time_median_ms,
)
from story_moe.config import config_from_dict
from story_moe.model import StoryLM

from _helpers import tiny_cfg


@pytest.fixture
def model() -> StoryLM:
    torch.manual_seed(0)
    m = StoryLM(tiny_cfg())
    m.eval()
    return m


def test_config_round_trips_through_a_dict() -> None:
    """The benchmark rebuilds models from a checkpoint's own config copy."""
    cfg = tiny_cfg()
    back = config_from_dict(cfg.to_dict())
    assert back.to_dict() == cfg.to_dict()


def test_benchmark_point_reports_agreeing_outputs(model: StoryLM) -> None:
    row = benchmark_point(model, prompt_len=4, replay_len=8, device="cpu",
                          precision="fp32", warmup=0, trials=1)
    assert row["outputs_agree_fp32"] is True
    assert row["outputs_agree_native"] is True
    assert row["prompt_tokens"] == 4 and row["replay_tokens"] == 8


def test_lockstep_finds_no_difference_when_the_cache_is_right(model: StoryLM) -> None:
    """In fp32 the two paths are the same computation, so the delta is noise.

    This is the quantitative form of the Day 6 gate. A wrong RoPE offset or a
    per-layer length skew would show up here as a large delta at step 0, which
    is what distinguishes a broken cache from reduced-precision tie-breaking.
    """
    torch.manual_seed(3)
    prompt = torch.randint(0, model.cfg.model.vocab_size, (1, 6))
    report = lockstep_compare(model, prompt, n_steps=8, amp=nullcontext)

    assert report["argmax_disagreements"] == 0
    assert report["first_disagreement"] is None
    assert report["max_logit_delta"] < 1e-4
    assert report["steps"] == 8


def test_per_token_cost_excludes_prefill(model: StoryLM) -> None:
    """cached_ms_per_token must come from the total minus one prefill.

    Getting this wrong would fold the prompt's cost into the decode rate and
    make the cache look worse at long prompts, which is exactly where it helps
    most.
    """
    row = benchmark_point(model, prompt_len=8, replay_len=8, device="cpu",
                          precision="fp32", warmup=0, trials=1)
    expected = (row["cached_total_ms"] - row["prefill_ms"]) / 8
    if expected > 0:
        assert row["cached_ms_per_token"] == pytest.approx(expected)
    else:
        # Below timer resolution. Reported as missing rather than as zero, which
        # would otherwise divide into an infinite speedup.
        assert row["cached_ms_per_token"] is None
        assert row["speedup"] is None
    assert row["uncached_ms_per_token"] == pytest.approx(row["uncached_total_ms"] / 8)


def test_missing_timings_print_as_n_a_instead_of_crashing() -> None:
    """The formatter has to survive a None, since run_grid prints every row.

    Short grid points on a fast GPU are exactly where a decode measurement
    falls below timer resolution, so this is the common case, not a corner.
    """
    assert _fmt(None, 5, 2).strip() == "n/a"
    assert _fmt(1.5, 6, 3).strip() == "1.500"


def test_timings_are_positive_and_speedup_is_finite(model: StoryLM) -> None:
    row = benchmark_point(model, prompt_len=4, replay_len=4, device="cpu",
                          precision="fp32", warmup=0, trials=1)
    assert row["prefill_ms"] > 0
    assert row["uncached_total_ms"] > 0
    assert row["speedup"] is None or row["speedup"] > 0


def test_grid_point_past_the_context_is_rejected(model: StoryLM) -> None:
    with pytest.raises(ValueError, match="exceeds max_seq_len"):
        benchmark_point(model, prompt_len=24, replay_len=16, device="cpu",
                        precision="fp32", warmup=0, trials=1)


def test_default_grid_fits_the_context() -> None:
    for limit in (32, 256, 512):
        grid = default_grid(limit)
        assert grid, f"no grid points for max_seq_len {limit}"
        assert all(p + r <= limit for p, r in grid)
    assert len(default_grid(512)) > len(default_grid(32))


def test_time_median_ms_discards_warmup() -> None:
    calls = []
    median, best = time_median_ms(lambda: calls.append(1), "cpu", warmup=3, trials=4)
    assert len(calls) == 7
    assert median >= best >= 0


def test_cache_reserved_size_follows_the_shape_formula(model: StoryLM) -> None:
    m = model.cfg.model
    elements = 2 * m.n_layers * 2 * m.n_heads * 64 * m.d_head
    assert _cache_reserved_mib(model, 64, 2, "fp32") == pytest.approx(
        elements * 4 / (1024 * 1024)
    )
    # Under autocast the buffers take the autocast dtype, so a bf16 run reserves
    # half of what the fp32 formula would claim.
    assert _cache_reserved_mib(model, 64, 2, "bf16") == pytest.approx(
        elements * 2 / (1024 * 1024)
    )
