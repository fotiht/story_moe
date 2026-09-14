"""Training: the tiny-batch overfit check, the real loop, and resume.

    python -m story_moe.train overfit      --config configs/debug_dense.yaml
    python -m story_moe.train train        --config configs/debug_dense.yaml
    python -m story_moe.train resume-smoke --config configs/debug_dense.yaml

The cache is disabled during training (there is no cache yet, and there will not
be one in this path). Validation runs under eval() and no_grad, then restores
training mode.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import Config, load_config
from .data import Batcher, check_cache_matches_config, load_cached_split, load_tokenizer
from .evaluate import describe_protocol, evaluate_blocks
from .model import StoryLM, count_parameters, expected_initial_loss


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


def resolve_device(cfg: Config) -> str:
    """Check the device and precision actually match what the config demands.

    Colab hands out whichever accelerator is free. The dense and MoE runs must
    share a device and a precision or the comparison is void, so a mismatch is
    an error here rather than a footnote discovered after both runs finish.
    """
    want = cfg.train.device
    if not want.startswith("cuda"):
        print(f"device         : {want}  precision: fp32 (precision setting ignored off CUDA)")
        return want

    if not torch.cuda.is_available():
        raise RuntimeError(
            f"config asks for {want!r} but torch.cuda.is_available() is False. "
            "Set train.device: cpu, or run this where a GPU exists."
        )
    name = torch.cuda.get_device_name(0)
    required = cfg.train.require_gpu_name
    if required and required.lower() not in name.lower():
        raise RuntimeError(
            f"config requires a GPU matching {required!r} but this machine has {name!r}. "
            "Runs on different GPUs are not comparable; change require_gpu_name only "
            "deliberately, and disclose it."
        )
    if cfg.train.precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError(f"{name} does not support bf16; use fp16 or fp32")

    print(f"device         : {name}  precision: {cfg.train.precision}")
    return want


def precision_tools(cfg: Config, device: str):
    """Returns (autocast_factory, scaler). Both are no-ops in fp32."""
    p = cfg.train.precision
    if not device.startswith("cuda") or p == "fp32":
        return nullcontext, None
    dtype = torch.float16 if p == "fp16" else torch.bfloat16
    factory = lambda: torch.autocast("cuda", dtype=dtype)  # noqa: E731
    scaler = torch.amp.GradScaler("cuda") if p == "fp16" else None
    return factory, scaler


def resolve_vocab_size(cfg: Config):
    """Set cfg.model.vocab_size from the tokenizer. Returns the tokenizer."""
    tok = load_tokenizer(cfg.data.tokenizer)
    cfg.model.vocab_size = len(tok)
    return tok


def build_model(cfg: Config, device: str = "cpu") -> StoryLM:
    if cfg.model.vocab_size is None:
        raise ValueError("call resolve_vocab_size(cfg) before build_model(cfg)")
    return StoryLM(cfg).to(device)


def report_parameters(model: StoryLM) -> dict[str, int]:
    counts = count_parameters(model)
    share = 100.0 * counts["embedding"] / counts["total"]
    print(
        f"parameters     : total={counts['total']:,}  "
        f"embedding={counts['embedding']:,} ({share:.1f}%)  body={counts['body']:,}"
    )
    print("                 (quote BODY when comparing dense against MoE)")
    return counts


# ---------------------------------------------------------------------------
# Optimizer and schedule
# ---------------------------------------------------------------------------


def make_optimizer(model: torch.nn.Module, cfg: Config) -> torch.optim.AdamW:
    """Weight decay on matrices only; none on LayerNorm gains or biases.

    named_parameters() deduplicates shared tensors, so a tied lm_head appears in
    exactly one group -- it is not decayed twice.
    """
    decay, no_decay = [], []
    for _, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)

    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": cfg.train.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=cfg.train.lr,
        betas=tuple(cfg.train.betas),
    )


def lr_at(step: int, cfg: Config, total_updates: int) -> float:
    """Linear warmup, then cosine decay to 10% of the peak."""
    peak, warm = cfg.train.lr, cfg.train.warmup_steps
    if warm > 0 and step < warm:
        return peak * (step + 1) / warm
    if total_updates <= warm:
        return peak
    progress = min(1.0, (step - warm) / max(1, total_updates - warm))
    return 0.1 * peak + 0.9 * peak * 0.5 * (1.0 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    """Write to a temp file and rename, so an interrupted save cannot corrupt."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def _to_cpu(x):
    """RNG states must be CPU byte tensors. torch.load(map_location='cuda') moves
    every tensor in the payload, RNG states included, and torch.set_rng_state
    then rejects them. Loading on CPU avoids this; normalizing here as well means
    a checkpoint written by an older version still restores."""
    return x.cpu() if isinstance(x, torch.Tensor) else x


def checkpoint_payload(
    model, optimizer, scaler, cfg, tok, step, processed_tokens, batcher, best_val
) -> dict[str, Any]:
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "config": cfg.to_dict(),
        "tokenizer": {"name_or_path": tok.name_or_path, "len": len(tok)},
        "step": step,
        "processed_tokens": processed_tokens,
        "best_val_nll": best_val,
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        "batcher": batcher.state_dict(),
        "epochs_seen": batcher.epochs_seen(),
    }


def restore_checkpoint(payload: dict[str, Any], model, optimizer, scaler, batcher) -> dict:
    """Restore training state. The payload must be loaded with map_location='cpu'.

    load_state_dict copies into the model's existing (already on-device) tensors,
    and torch.optim moves optimizer state to each parameter's device, so nothing
    is lost by loading on CPU -- while RNG states, which must stay CPU byte
    tensors, survive.
    """
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    if scaler is not None and payload.get("scaler") is not None:
        scaler.load_state_dict(payload["scaler"])

    rng = payload.get("rng") or {}
    if rng.get("python") is not None:
        random.setstate(rng["python"])
    if rng.get("numpy") is not None:
        np.random.set_state(rng["numpy"])
    if rng.get("torch") is not None:
        torch.set_rng_state(_to_cpu(rng["torch"]))
    if rng.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([_to_cpu(s) for s in rng["cuda"]])
    if payload.get("batcher") is not None:
        batcher.load_state_dict(payload["batcher"])

    return {
        "step": payload["step"],
        "processed_tokens": payload["processed_tokens"],
        "best_val_nll": payload.get("best_val_nll", float("inf")),
    }


# ---------------------------------------------------------------------------
# The tiny-batch overfit check (Day 2 gate, kept as a regression test)
# ---------------------------------------------------------------------------


def overfit(
    cfg: Config,
    n_blocks: int = 2,
    steps: int = 200,
    lr: float = 1e-3,
    device: str = "cpu",
    log_every: int = 25,
) -> dict[str, float]:
    """Train on a handful of fixed blocks and confirm the loss collapses.

    A wiring test, not an experiment. If it fails, the bug is in target shifting,
    the causal mask, parameter registration, or the optimizer step.
    """
    cfg.model.dropout = 0.0
    tok = resolve_vocab_size(cfg)

    blocks, meta = load_cached_split(Path(cfg.data.cache_dir), "train")
    check_cache_matches_config(meta, cfg, "train")
    window = torch.from_numpy(blocks[:n_blocks].astype(np.int64)).to(device)
    x, y = window[:, :-1], window[:, 1:]

    model = build_model(cfg, device)
    model.train()
    counts = report_parameters(model)

    V = cfg.model.vocab_size
    print(f"tokenizer      : {tok.name_or_path}  V={V}")
    print(f"fixed batch    : {tuple(x.shape)} from {meta['n_blocks']} cached blocks")
    print(f"expected step-0 loss ~= ln(V) = {expected_initial_loss(V):.3f}\n")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.0)

    first_loss = None
    for step in range(steps + 1):
        out = model(x, targets=y)
        if first_loss is None:
            first_loss = out.lm_loss.item()

        if step % log_every == 0 or step == steps:
            with torch.no_grad():
                acc = (out.logits.argmax(-1) == y).float().mean().item()
            print(
                f"step {step:4d}  lm_loss {out.lm_loss.item():8.4f}  "
                f"aux {out.aux_loss.item():6.4f}  next-token acc {acc:6.2%}"
            )

        if step == steps:
            break
        opt.zero_grad(set_to_none=True)
        out.total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        opt.step()

    with torch.no_grad():
        out = model(x, targets=y)
        final_loss = out.lm_loss.item()
        final_acc = (out.logits.argmax(-1) == y).float().mean().item()

    expected = expected_initial_loss(V)
    print(f"\nstep-0 loss    : {first_loss:.4f}  (ln(V) = {expected:.3f}, "
          f"off by {abs(first_loss - expected):.3f})")
    print(f"final loss     : {final_loss:.4f}   final accuracy: {final_acc:.2%}")
    verdict = "PASS" if final_loss < 0.5 * first_loss and final_acc > 0.9 else "CHECK"
    print(f"overfit verdict: {verdict}  (want a large drop and high accuracy)")

    return {
        "first_loss": first_loss,
        "final_loss": final_loss,
        "final_acc": final_acc,
        "expected_initial_loss": expected,
        **{f"params_{k}": v for k, v in counts.items()},
    }


# ---------------------------------------------------------------------------
# The real loop
# ---------------------------------------------------------------------------


def train(
    cfg: Config,
    resume: str | None = None,
    max_updates: int | None = None,
    log_every: int = 10,
) -> dict[str, Any]:
    torch.manual_seed(cfg.data.seed)
    np.random.seed(cfg.data.seed)
    random.seed(cfg.data.seed)

    device = resolve_device(cfg)
    autocast_factory, scaler = precision_tools(cfg, device)
    tok = resolve_vocab_size(cfg)

    cache = Path(cfg.data.cache_dir)
    train_blocks, train_meta = load_cached_split(cache, "train")
    val_blocks, val_meta = load_cached_split(cache, "validation")
    # Before anything is counted: a mismatched cache runs fine and reports
    # processed-token numbers that are wrong by the block-size ratio.
    check_cache_matches_config(train_meta, cfg, "train")
    check_cache_matches_config(val_meta, cfg, "validation")

    model = build_model(cfg, device)
    counts = report_parameters(model)
    is_moe = cfg.model.use_moe
    opt = make_optimizer(model, cfg)
    batcher = Batcher(train_blocks, cfg.train.microbatch_size, seed=cfg.data.seed)

    T = cfg.data.block_size
    tokens_per_update = cfg.train.microbatch_size * cfg.train.accum_steps * T
    total_updates = max_updates or max(1, cfg.train.max_tokens // tokens_per_update)
    unique_tokens = train_meta["unique_stream_tokens"]
    # Sampling is without replacement, so this ratio really is data coverage.
    epochs = (total_updates * tokens_per_update) / unique_tokens

    print(f"tokenizer      : {tok.name_or_path}  V={cfg.model.vocab_size}")
    print(f"train protocol : {describe_protocol(train_meta, cfg.train.microbatch_size)}")
    print(f"val protocol   : {describe_protocol(val_meta, cfg.train.microbatch_size)}")
    print(
        f"budget         : {total_updates} updates x {tokens_per_update} tokens "
        f"= {total_updates * tokens_per_update:,} processed tokens"
    )
    print(
        f"                 unique dataset tokens = {unique_tokens:,}  ->  "
        f"{epochs:.2f} passes (sampling without replacement)"
    )
    if epochs > 2.0:
        print("                 WARNING: repeated passes. Report processed and unique "
              "token counts separately; do not call this a single-epoch run.")
    warm_frac = cfg.train.warmup_steps / total_updates
    if warm_frac > 0.2:
        print(f"                 WARNING: warmup is {warm_frac:.0%} of the run "
              f"({cfg.train.warmup_steps}/{total_updates} updates); lower warmup_steps.")

    start_step, processed_tokens, best_val = 0, 0, float("inf")
    if resume:
        # CPU, always: map_location="cuda" would move the RNG byte tensors to
        # the GPU and torch.set_rng_state rejects them. A CPU-only resume test
        # cannot catch this, which is why it is asserted here rather than there.
        payload = torch.load(resume, map_location="cpu", weights_only=False)
        state = restore_checkpoint(payload, model, opt, scaler, batcher)
        start_step = state["step"]
        processed_tokens = state["processed_tokens"]
        best_val = state["best_val_nll"]
        print(f"resumed        : {resume}  step={start_step}  tokens={processed_tokens:,}")

    out_dir = Path(cfg.train.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []
    model.train()
    t0 = time.time()
    tokens_at_t0 = processed_tokens

    for step in range(start_step, total_updates):
        lr = lr_at(step, cfg, total_updates)
        for group in opt.param_groups:
            group["lr"] = lr

        # Collecting routing statistics on logging steps only keeps the cost off
        # the hot path; they are aggregated over every microbatch of the update,
        # because one microbatch of expert usage is far too noisy to read.
        want_stats = is_moe and (step % log_every == 0 or step == total_updates - 1)
        frac_sum = [0.0] * cfg.model.n_experts
        prob_sum = [0.0] * cfg.model.n_experts
        stat_layers = 0

        opt.zero_grad(set_to_none=True)
        lm_sum = aux_sum = 0.0
        for _ in range(cfg.train.accum_steps):
            x, y = batcher.next_batch(device)
            with autocast_factory():
                out = model(x, targets=y, collect_stats=want_stats)
            if want_stats and out.router_stats:
                for st in out.router_stats:
                    stat_layers += 1
                    for e in range(cfg.model.n_experts):
                        frac_sum[e] += st.assignment_fraction[e]
                        prob_sum[e] += st.mean_probability[e]
            # Divide before backward so the accumulated gradient is the mean,
            # not the sum, over microbatches.
            loss = out.total_loss / cfg.train.accum_steps
            (scaler.scale(loss) if scaler else loss).backward()
            lm_sum += out.lm_loss.item()
            aux_sum += out.aux_loss.item()

        if scaler:
            scaler.unscale_(opt)                      # unscale BEFORE clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        if scaler:
            scaler.step(opt)
            scaler.update()
        else:
            opt.step()

        processed_tokens += tokens_per_update
        accum = cfg.train.accum_steps

        if step % log_every == 0 or step == total_updates - 1:
            elapsed = time.time() - t0
            tps = (processed_tokens - tokens_at_t0) / elapsed if elapsed > 0 else 0.0
            mem = (
                torch.cuda.max_memory_allocated() / 2**20
                if device.startswith("cuda") else 0.0
            )
            row = {
                "step": step,
                "lm_loss": lm_sum / accum,
                "aux_loss": aux_sum / accum,
                "total_loss": lm_sum / accum + cfg.model.aux_loss_weight * (aux_sum / accum),
                "lr": lr,
                "grad_norm": float(grad_norm),
                "processed_tokens": processed_tokens,
                "elapsed_s": elapsed,
                "tokens_per_s": tps,
                "peak_mem_mib": mem,
                "epochs_seen": batcher.epochs_seen(),
            }
            if want_stats and stat_layers:
                row["assignment_fraction"] = [v / stat_layers for v in frac_sum]
                row["mean_probability"] = [v / stat_layers for v in prob_sum]
            history.append(row)
            print(
                f"step {step:5d}/{total_updates}  lm {row['lm_loss']:7.4f}  "
                f"aux {row['aux_loss']:6.4f}  lr {lr:.2e}  gn {row['grad_norm']:5.2f}  "
                f"tok {processed_tokens:>10,}  {tps:8.0f} tok/s"
            )
            if "assignment_fraction" in row:
                frac = " ".join(f"{v:.3f}" for v in row["assignment_fraction"])
                prob = " ".join(f"{v:.3f}" for v in row["mean_probability"])
                # The assignment fractions are the real balance diagnostic. The
                # auxiliary loss is NOT: with near-uniform probabilities it sits
                # at 1.0 even when every token goes to the same expert.
                print(f"  routing      assign [{frac}]  prob [{prob}]")

        is_last = step == total_updates - 1
        if (step + 1) % cfg.train.eval_every == 0 or is_last:
            val = evaluate_blocks(
                model, val_blocks, cfg.train.microbatch_size, device,
                autocast_ctx=autocast_factory,
            )
            print(
                f"  validation   mean_nll {val['mean_nll']:.4f}  "
                f"ppl {val['perplexity']:.2f}  over {val['scored_tokens']:,} scored tokens"
            )
            history.append({"step": step, "validation": val})
            if val["mean_nll"] < best_val:
                best_val = val["mean_nll"]
                save_checkpoint(
                    out_dir / "best.pt",
                    checkpoint_payload(model, opt, scaler, cfg, tok, step + 1,
                                       processed_tokens, batcher, best_val),
                )

        if (step + 1) % cfg.train.ckpt_every == 0 or is_last:
            save_checkpoint(
                out_dir / "latest.pt",
                checkpoint_payload(model, opt, scaler, cfg, tok, step + 1,
                                   processed_tokens, batcher, best_val),
            )

    results = {
        "config": cfg.to_dict(),
        "parameters": counts,
        "total_updates": total_updates,
        "tokens_per_update": tokens_per_update,
        "processed_tokens": processed_tokens,
        "unique_dataset_tokens": unique_tokens,
        "passes_over_data": epochs,
        "epochs_seen": batcher.epochs_seen(),
        "sampling": "without replacement (shuffled permutation per epoch)",
        "best_val_nll": best_val,
        "best_val_perplexity": math.exp(best_val) if best_val < float("inf") else None,
        "history": history,
    }
    res_path = Path("results") / f"{cfg.name}_train.json"
    res_path.parent.mkdir(parents=True, exist_ok=True)
    res_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {res_path}")
    return results


def resume_smoke(cfg: Config, updates: int = 3) -> None:
    """Day 3 gate: train a few updates, reload, and take one more."""
    out_dir = Path(cfg.train.out_dir)
    print(f"--- phase 1: {updates} updates ---")
    train(cfg, max_updates=updates, log_every=1)

    ckpt = out_dir / "latest.pt"
    if not ckpt.exists():
        raise RuntimeError(f"no checkpoint at {ckpt}; nothing to resume from")
    before = torch.load(ckpt, map_location="cpu", weights_only=False)
    print(f"\ncheckpoint     : step={before['step']} tokens={before['processed_tokens']:,} "
          f"keys={sorted(before.keys())}")

    print("\n--- phase 2: resume and take 1 more update ---")
    train(cfg, resume=str(ckpt), max_updates=updates + 1, log_every=1)

    after = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert after["step"] == updates + 1, f"expected step {updates + 1}, got {after['step']}"
    print(f"\nresume smoke   : PASS  step {before['step']} -> {after['step']}")
    print("Note: this proves the state reloads and training continues. It does NOT "
          "claim bitwise-identical resumed training.")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="story_moe.train")
    p.add_argument("command", choices=["overfit", "train", "resume-smoke"])
    p.add_argument("--config", required=True)
    p.add_argument("--n-blocks", type=int, default=2)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default=None, help="override train.device for this run")
    p.add_argument("--resume", default=None)
    p.add_argument("--max-updates", type=int, default=None)
    # Colab overrides. out-dir and data-cache normally point into mounted Drive,
    # because the Colab VM's own disk disappears with the session.
    p.add_argument("--out-dir", default=None)
    p.add_argument("--data-cache", default=None)
    p.add_argument("--precision", default=None, choices=["fp32", "fp16", "bf16"])
    p.add_argument("--require-gpu-name", default=None,
                   help="refuse to run unless the GPU name contains this, e.g. A100")
    p.add_argument("--max-tokens", type=int, default=None)
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    if args.device:
        cfg.train.device = args.device
    if args.out_dir:
        cfg.train.out_dir = args.out_dir
    if args.data_cache:
        cfg.data.cache_dir = args.data_cache
    if args.precision:
        cfg.train.precision = args.precision
    if args.require_gpu_name:
        cfg.train.require_gpu_name = args.require_gpu_name
    if args.max_tokens:
        cfg.train.max_tokens = args.max_tokens

    if args.command == "overfit":
        overfit(cfg, args.n_blocks, args.steps, args.lr, args.device or "cpu")
    elif args.command == "train":
        train(cfg, resume=args.resume, max_updates=args.max_updates)
    else:
        resume_smoke(cfg)


if __name__ == "__main__":
    main()
