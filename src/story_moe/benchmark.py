"""Day 7: what the KV cache actually buys, measured.

The claim being tested is narrow. The cache changes no arithmetic, so it cannot
change what the model says; it changes how much of that arithmetic is repeated.
At generated position t:

    uncached step   re-runs the whole network over all t tokens
                    ~ O(t * D^2)  projections and MLP
                    + O(t^2 * D)  attention
    cached step     runs the network over ONE token
                    ~ O(D^2)      projections and MLP
                    + O(t * D)    attention against t stored keys

So the saving is not mainly in attention. It is that the uncached path re-runs
every projection, every expert and the output head over the entire prefix, every
step. Expect the per-token speedup to grow roughly linearly with the sequence
length, and expect the cached path to use LESS peak memory despite storing keys,
because the uncached path materializes a [B, H, t, t] score matrix each step.

Honesty notes baked into the numbers below:

  - Every timed region is bracketed by torch.cuda.synchronize(). CUDA launches
    are asynchronous, and timing without it measures how fast Python can queue
    work.
  - Warmup iterations are discarded. The first call on a shape pays allocator
    and kernel selection costs that no later call pays.
  - The median of repeated trials is reported, not the mean. One Colab
    preemption blip skews a mean and leaves a median alone.
  - Both modes decode greedily and the tokens are compared. A benchmark whose
    two sides compute different things measures nothing, so that comparison is
    an assertion, not a footnote.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable

import torch

from .config import Config, config_from_dict
from .generate import GenerationConfig, generate
from .model import StoryLM, count_parameters


def load_for_inference(
    ckpt_path: str | Path, device: str, precision: str | None = None
) -> tuple[StoryLM, Config, dict[str, Any]]:
    """Rebuild the trained model from a checkpoint, in eval mode.

    The config comes from inside the checkpoint, so the shapes are the ones the
    weights were trained with and a since-edited YAML cannot cause a silent
    mismatch.
    """
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = config_from_dict(payload["config"])
    if precision is not None:
        cfg.train.precision = precision

    model = StoryLM(cfg)
    model.load_state_dict(payload["model"])
    model.to(device).eval()
    meta = {
        "step": payload.get("step"),
        "processed_tokens": payload.get("processed_tokens"),
        "best_val_nll": payload.get("best_val_nll"),
        "tokenizer": payload.get("tokenizer"),
    }
    return model, cfg, meta


def autocast_for(device: str, precision: str) -> Callable[[], Any]:
    if not device.startswith("cuda") or precision == "fp32":
        return nullcontext
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return lambda: torch.autocast("cuda", dtype=dtype)


def _sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def time_median_ms(
    fn: Callable[[], Any], device: str, warmup: int = 2, trials: int = 5
) -> tuple[float, float]:
    """Run fn, discard `warmup`, return (median ms, min ms) over `trials`."""
    for _ in range(warmup):
        fn()
    _sync(device)

    samples = []
    for _ in range(trials):
        _sync(device)
        start = time.perf_counter()
        fn()
        _sync(device)
        samples.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(samples), min(samples)


def peak_mib(device: str) -> float | None:
    if not device.startswith("cuda"):
        return None
    return torch.cuda.max_memory_allocated() / (1024 * 1024)


@torch.no_grad()
def lockstep_compare(
    model: StoryLM,
    prompt: torch.Tensor,
    n_steps: int,
    amp: Callable[[], Any],
) -> dict[str, Any]:
    """Step both paths on IDENTICAL tokens and compare their logits directly.

    Free-running greedy decode answers "do the two paths produce the same
    story", but it answers it badly: one differing token at step 40 makes every
    later token differ too, so a single disagreement looks like total failure
    and there is no way to see how big the underlying numerical difference was.

    Here both paths are fed the same token every step, chosen by the cached
    path. That isolates per-step disagreement from cumulative drift and lets the
    question become quantitative: how far apart are the logits, and when argmax
    does disagree, how close were the top two candidates?

    A real cache bug (wrong RoPE offset, per-layer length skew) produces a large
    logit delta at the FIRST step. Reduced-precision tie-breaking produces a
    tiny delta that only flips argmax when the top-2 gap is smaller than it.
    Those two stories are distinguishable, which is the point of measuring.
    """
    ids = prompt.clone()
    cache = model.new_cache(max_len=prompt.size(1) + n_steps)
    with amp():
        out_cached = model(prompt, logits_to_keep=1, cache=cache)
        out_uncached = model(prompt, logits_to_keep=1)

    max_delta = 0.0
    max_logit_scale = 0.0
    disagreements = 0
    first: dict[str, Any] | None = None

    for step in range(n_steps):
        lc = out_cached.logits[:, -1, :].float()
        lu = out_uncached.logits[:, -1, :].float()

        delta = (lc - lu).abs().max().item()
        max_delta = max(max_delta, delta)
        max_logit_scale = max(max_logit_scale, lu.abs().max().item())

        pick_c = lc.argmax(dim=-1)
        pick_u = lu.argmax(dim=-1)
        if not torch.equal(pick_c, pick_u):
            disagreements += 1
            if first is None:
                top2 = lu.topk(2, dim=-1).values
                first = {
                    "step": step,
                    "logit_delta": delta,
                    "top2_gap": (top2[:, 0] - top2[:, 1]).min().item(),
                    "cached_token": int(pick_c[0].item()),
                    "uncached_token": int(pick_u[0].item()),
                }

        nxt = pick_c.unsqueeze(1)
        ids = torch.cat([ids, nxt], dim=1)
        with amp():
            out_cached = model(nxt, logits_to_keep=1, cache=cache)
            out_uncached = model(ids, logits_to_keep=1)

    return {
        "steps": n_steps,
        "max_logit_delta": max_delta,
        "max_logit_magnitude": max_logit_scale,
        "relative_delta": (max_delta / max_logit_scale) if max_logit_scale else None,
        "argmax_disagreements": disagreements,
        "first_disagreement": first,
    }


@torch.no_grad()
def benchmark_point(
    model: StoryLM,
    prompt_len: int,
    replay_len: int,
    device: str,
    precision: str,
    batch_size: int = 1,
    warmup: int = 2,
    trials: int = 5,
    seed: int = 0,
) -> dict[str, Any]:
    """One (prompt, replay) cell of the grid, both modes.

    "Replay" is the spec's word for how many tokens are generated after the
    prompt. Greedy decoding throughout, so the two modes are directly
    comparable and their outputs can be checked against each other.
    """
    vocab = model.cfg.model.vocab_size
    limit = model.cfg.model.max_seq_len
    if prompt_len + replay_len > limit:
        raise ValueError(
            f"prompt {prompt_len} + replay {replay_len} exceeds max_seq_len {limit}"
        )

    gen = torch.Generator(device="cpu").manual_seed(seed)
    prompt = torch.randint(0, vocab, (batch_size, prompt_len), generator=gen).to(device)
    amp = autocast_for(device, precision)
    gcfg = GenerationConfig(max_new_tokens=replay_len, temperature=0.0)

    # Correctness first. If these disagree the timings below are meaningless.
    #
    # The gate runs in fp32, not in the timing precision. The cache is a
    # mathematical identity, so it has to be tested where the arithmetic is
    # precise enough to test it. Greedy token equality is a DISCONTINUOUS
    # function of the logits: under bf16, two candidates within ~1e-2 of each
    # other can swap on a rounding difference, and one swapped token makes every
    # later token differ. Failing the run on that would be reporting a property
    # of bf16 as though it were a broken cache.
    fp32 = autocast_for(device, "fp32")
    with fp32():
        agree_fp32 = bool(torch.equal(
            generate(model, prompt, gcfg, use_cache=True),
            generate(model, prompt, gcfg, use_cache=False),
        ))
    lockstep_fp32 = lockstep_compare(model, prompt, min(replay_len, 32), fp32)

    # And the same question at the precision actually being timed, reported
    # rather than enforced, with the logit deltas that explain the answer.
    with amp():
        agree_native = bool(torch.equal(
            generate(model, prompt, gcfg, use_cache=True),
            generate(model, prompt, gcfg, use_cache=False),
        ))
    lockstep_native = lockstep_compare(model, prompt, min(replay_len, 32), amp)

    def run_prefill() -> None:
        cache = model.new_cache(max_len=prompt_len + replay_len)
        with amp():
            model(prompt, logits_to_keep=1, cache=cache)

    def run_cached() -> None:
        with amp():
            generate(model, prompt, gcfg, use_cache=True)

    def run_uncached() -> None:
        with amp():
            generate(model, prompt, gcfg, use_cache=False)

    prefill_ms, _ = time_median_ms(run_prefill, device, warmup, trials)

    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    cached_ms, cached_best = time_median_ms(run_cached, device, warmup, trials)
    cached_peak = peak_mib(device)

    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    uncached_ms, uncached_best = time_median_ms(run_uncached, device, warmup, trials)
    uncached_peak = peak_mib(device)

    # Per-token decode cost excludes prefill, which both modes pay identically:
    # the cached total includes one prefill, so subtract it before dividing.
    # Two separately-timed medians can subtract to something non-positive when
    # the decode work is near timer resolution (short replay on a fast GPU).
    # That is a measurement too small to report, not a decode that took no time,
    # so it becomes None rather than a flattering zero or a divide by zero.
    cached_decode_ms = cached_ms - prefill_ms
    per_token_cached = (
        cached_decode_ms / replay_len if replay_len and cached_decode_ms > 0 else None
    )
    per_token_uncached = uncached_ms / replay_len if replay_len else None

    return {
        "prompt_tokens": prompt_len,
        "replay_tokens": replay_len,
        "batch_size": batch_size,
        "outputs_agree_fp32": agree_fp32,
        "outputs_agree_native": agree_native,
        "lockstep_fp32": lockstep_fp32,
        "lockstep_native": lockstep_native,
        "prefill_ms": prefill_ms,
        "cached_total_ms": cached_ms,
        "uncached_total_ms": uncached_ms,
        "cached_ms_per_token": per_token_cached,
        "uncached_ms_per_token": per_token_uncached,
        "speedup": (
            per_token_uncached / per_token_cached
            if per_token_cached and per_token_uncached
            else None
        ),
        "cached_best_total_ms": cached_best,
        "uncached_best_total_ms": uncached_best,
        "cached_peak_mib": cached_peak,
        "uncached_peak_mib": uncached_peak,
        "cache_reserved_mib": _cache_reserved_mib(model, prompt_len + replay_len, batch_size),
    }


def _cache_reserved_mib(model: StoryLM, max_len: int, batch_size: int) -> float:
    """What the cache buffers cost, from shapes rather than from the allocator.

    2 (keys and values) * layers * B * H * max_len * Dh * bytes_per_element.
    Reported at fp32; under autocast the stored tensors are the autocast dtype,
    so treat this as an upper bound.
    """
    m = model.cfg.model
    elements = 2 * m.n_layers * batch_size * m.n_heads * max_len * m.d_head
    return elements * 4 / (1024 * 1024)


def run_grid(
    model: StoryLM,
    grid: list[tuple[int, int]],
    device: str,
    precision: str,
    batch_size: int = 1,
    warmup: int = 2,
    trials: int = 5,
) -> list[dict[str, Any]]:
    rows = []
    for prompt_len, replay_len in grid:
        row = benchmark_point(
            model, prompt_len, replay_len, device, precision,
            batch_size=batch_size, warmup=warmup, trials=trials,
        )
        rows.append(row)
        flag = "" if row["outputs_agree_fp32"] else "   <-- fp32 MISMATCH, CACHE IS WRONG"
        print(
            f"  prompt {prompt_len:>4}  replay {replay_len:>4}   "
            f"prefill {row['prefill_ms']:7.2f} ms   "
            f"cached {_fmt(row['cached_ms_per_token'], 6, 3)} ms/tok   "
            f"uncached {_fmt(row['uncached_ms_per_token'], 7, 3)} ms/tok   "
            f"speedup {_fmt(row['speedup'], 5, 2)}x{flag}"
        )
        _print_precision_note(row)
    return rows


def _print_precision_note(row: dict[str, Any]) -> None:
    """Show why the timed precision disagreed, when it did."""
    if row["outputs_agree_native"]:
        return
    ls = row["lockstep_native"]
    first = ls["first_disagreement"]
    detail = ""
    if first:
        detail = (
            f", first at step {first['step']} where the top-2 gap was "
            f"{first['top2_gap']:.4f} against a logit delta of "
            f"{first['logit_delta']:.4f}"
        )
    print(
        f"      note: same tokens in fp32, different tokens at the timed "
        f"precision. Lockstep max logit delta "
        f"{ls['max_logit_delta']:.5f} on logits of magnitude up to "
        f"{ls['max_logit_magnitude']:.2f} "
        f"({ls['argmax_disagreements']}/{ls['steps']} steps flipped argmax"
        f"{detail})."
    )


def _fmt(value: float | None, width: int, places: int) -> str:
    """Print a missing measurement as n/a instead of inventing a number."""
    return "n/a".rjust(width) if value is None else f"{value:{width}.{places}f}"


def default_grid(max_seq_len: int) -> list[tuple[int, int]]:
    """Points that fit the model's context, chosen to show the trend with length.

    The short points are not filler. The speedup should start near 1x and grow
    with sequence length, and a grid that only samples long sequences shows the
    headline number without the trend that explains it. The smallest point also
    keeps this usable on a short-context model instead of returning nothing.
    """
    candidates = [
        (8, 8), (8, 16), (16, 32), (32, 64),
        (64, 64), (128, 64), (128, 128), (256, 256),
    ]
    return [(p, r) for p, r in candidates if p + r <= max_seq_len]


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="story_moe.benchmark")
    p.add_argument("--checkpoint", required=True, help="path to a trained .pt")
    p.add_argument("--device", default="cuda")
    p.add_argument("--precision", default=None, help="override the checkpoint's precision")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--trials", type=int, default=5)
    p.add_argument("--out", default=None, help="write the rows to this JSON file")
    p.add_argument(
        "--grid", default=None,
        help="semicolon-separated prompt,replay pairs, e.g. '32,64;128,128'",
    )
    args = p.parse_args(argv)

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("--device cuda but torch.cuda.is_available() is False")

    model, cfg, meta = load_for_inference(args.checkpoint, device, args.precision)
    precision = args.precision or cfg.train.precision
    if not device.startswith("cuda"):
        precision = "fp32"

    params = count_parameters(model)
    gpu = torch.cuda.get_device_name(0) if device.startswith("cuda") else "cpu"
    print(f"model          : {cfg.name}  use_moe={cfg.model.use_moe}")
    print(f"parameters     : total={params['total']:,}  body={params['body']:,}")
    print(f"checkpoint     : step={meta['step']}  tokens={meta['processed_tokens']:,}")
    print(f"device         : {gpu}  precision: {precision}")
    print(f"context        : max_seq_len={cfg.model.max_seq_len}")
    print(f"timing         : median of {args.trials} trials after {args.warmup} warmup\n")

    if args.grid:
        grid = [
            tuple(int(v) for v in pair.split(","))
            for pair in args.grid.split(";") if pair.strip()
        ]
    else:
        grid = default_grid(cfg.model.max_seq_len)
    if not grid:
        raise ValueError(f"no grid points fit max_seq_len {cfg.model.max_seq_len}")

    rows = run_grid(
        model, grid, device, precision,
        batch_size=args.batch_size, warmup=args.warmup, trials=args.trials,
    )

    broken = [r for r in rows if not r["outputs_agree_fp32"]]
    if broken:
        worst = max(r["lockstep_fp32"]["max_logit_delta"] for r in broken)
        raise SystemExit(
            f"\nFAIL: {len(broken)} grid point(s) produced different tokens with "
            f"and without the cache IN FP32, where the two paths are the same "
            f"computation. Largest lockstep logit delta {worst:.6f}. The cache is "
            "wrong; do not quote any speedup."
        )
    print("\nfp32 gate: all grid points produced identical tokens with and without "
          "the cache")

    flipped = [r for r in rows if not r["outputs_agree_native"]]
    if flipped:
        worst = max(r["lockstep_native"]["max_logit_delta"] for r in flipped)
        print(
            f"at the timed precision, {len(flipped)}/{len(rows)} points decoded "
            f"different tokens. Largest lockstep logit delta {worst:.5f}, which is "
            "rounding, not a different computation: greedy argmax flips whenever "
            "the top-2 gap is narrower than that. Quote the fp32 gate as the "
            "correctness result and this as a property of the precision."
        )

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": cfg.to_dict(),
            "checkpoint": str(args.checkpoint),
            "checkpoint_meta": meta,
            "parameters": params,
            "device": gpu,
            "precision": precision,
            "batch_size": args.batch_size,
            "warmup": args.warmup,
            "trials": args.trials,
            "torch_version": torch.__version__,
            "rows": rows,
        }
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
