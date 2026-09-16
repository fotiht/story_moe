"""Autoregressive sampling, with and without the KV cache.

Two paths that must produce identical tokens:

  cached     prefill the prompt once, then feed one token per step and let the
             cache supply the keys and values for everything before it.
  uncached   re-run the whole prefix through the model at every step.

The uncached path is the control the cached path is measured against, for
correctness (tests/test_cache.py) and for speed (benchmark.py). Both live here
and share one sampler, so the only difference between them is where the keys
come from.

Shapes:

    prompt_ids      [B, P]           integer token ids
    step logits     [B, 1, V]        logits_to_keep=1 projects one position
    return          [B, P + N]       prompt followed by the sampled tokens
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .data import load_tokenizer
from .model import StoryLM
from .train import autocast_for, load_for_inference


@dataclass
class GenerationConfig:
    """Decoding knobs.

    temperature 0.0 means greedy (argmax). Above 0 it divides the logits before
    the softmax, so below 1 sharpens and above 1 flattens.

    top_k keeps only the k highest-probability tokens and renormalizes over
    them. 0 or None disables it. It has no effect when temperature is 0, since
    argmax of the full distribution and argmax of its top-k truncation agree.

    eos_id stops a sequence early. With eos_id None, exactly max_new_tokens are
    produced, which a timing benchmark needs.
    """

    max_new_tokens: int = 64
    temperature: float = 1.0
    top_k: int | None = None
    eos_id: int | None = None


def sample_next(
    logits: torch.Tensor,
    cfg: GenerationConfig,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """[B, V] -> [B, 1]. The one sampler both decode paths call.

    Sampling happens in float32 whatever the model dtype. A bf16 softmax over
    50,257 logits loses enough resolution in the tail to change which tokens are
    reachable at all.
    """
    if cfg.temperature < 0:
        raise ValueError(f"temperature must be >= 0, got {cfg.temperature}")

    logits = logits.float()
    if cfg.temperature == 0:
        return logits.argmax(dim=-1, keepdim=True)

    logits = logits / cfg.temperature
    if cfg.top_k:
        kth = logits.topk(min(cfg.top_k, logits.size(-1)), dim=-1).values[:, -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))

    probs = F.softmax(logits, dim=-1)
    if generator is not None and generator.device.type != probs.device.type:
        raise ValueError(
            f"generator is on {generator.device.type} but the logits are on "
            f"{probs.device.type}; torch.multinomial requires them to match. "
            f"Build it as torch.Generator(device={probs.device.type!r})."
        )
    return torch.multinomial(probs, num_samples=1, generator=generator)


@torch.no_grad()
def generate(
    model: StoryLM,
    prompt_ids: torch.Tensor,
    cfg: GenerationConfig,
    use_cache: bool = True,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """[B, P] -> [B, P + N]. Prompt tokens are returned unchanged.

    With use_cache the prompt is prefilled in one forward pass and each later
    step sees one token. Without it, every step re-runs the full prefix. The two
    are the same computation. Only the amount of recomputation differs.
    """
    B, P = prompt_ids.shape
    if P == 0:
        # logits_to_keep on an empty sequence produces an empty tensor rather
        # than an error, so this would otherwise fail silently.
        raise ValueError("prompt must hold at least one token")

    limit = model.cfg.model.max_seq_len
    if P + cfg.max_new_tokens > limit:
        raise ValueError(
            f"{P} prompt + {cfg.max_new_tokens} new tokens exceeds max_seq_len {limit}"
        )

    was_training = model.training
    model.eval()  # dropout would make the two paths disagree, and both wrong
    try:
        ids = prompt_ids
        finished = torch.zeros(B, dtype=torch.bool, device=prompt_ids.device)
        cache = model.new_cache(max_len=P + cfg.max_new_tokens) if use_cache else None

        # Only the last position's logits matter. Every earlier one predicts a
        # token we were already given.
        out = model(ids, logits_to_keep=1, cache=cache)

        for step in range(cfg.max_new_tokens):
            next_ids = sample_next(out.logits[:, -1, :], cfg, generator)
            if cfg.eos_id is not None:
                # Hold finished rows at EOS so the batch keeps stepping together.
                eos = torch.full_like(next_ids, cfg.eos_id)
                next_ids = torch.where(finished.unsqueeze(1), eos, next_ids)
                finished |= next_ids.squeeze(1) == cfg.eos_id
            ids = torch.cat([ids, next_ids], dim=1)

            # Stop before the forward pass whose logits nobody will read.
            if step == cfg.max_new_tokens - 1:
                break
            if cfg.eos_id is not None and bool(finished.all()):
                break

            if cache is not None:
                out = model(next_ids, logits_to_keep=1, cache=cache)
            else:
                out = model(ids, logits_to_keep=1)

        return ids
    finally:
        model.train(was_training)


def main(argv: list[str] | None = None) -> None:
    """Sample a continuation from a trained checkpoint.

    python -m story_moe.generate --checkpoint path/to/latest.pt \\
        --prompt "Once upon a time" --max-new-tokens 120 --temperature 0.8
    """
    p = argparse.ArgumentParser(prog="story_moe.generate")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--max-new-tokens", type=int, default=100)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--precision", default=None)
    p.add_argument("--no-cache", action="store_true",
                   help="decode without the KV cache; same tokens, slower")
    args = p.parse_args(argv)

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
        print("note: no CUDA available, falling back to cpu")

    model, model_cfg, meta = load_for_inference(args.checkpoint, device, args.precision)
    tok = load_tokenizer(model_cfg.data.tokenizer)
    if len(tok) != model_cfg.model.vocab_size:
        raise ValueError(
            f"tokenizer has {len(tok)} tokens but the checkpoint was trained with "
            f"vocab_size {model_cfg.model.vocab_size}"
        )

    ids = torch.tensor(
        [tok.encode(args.prompt, add_special_tokens=False)],
        dtype=torch.long, device=device,
    )
    cfg = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        eos_id=tok.eos_token_id,
    )
    # The generator must live where the probabilities do, which is wherever the
    # model is. A CPU generator with CUDA logits is a hard error in multinomial.
    generator = torch.Generator(device=device).manual_seed(args.seed)

    print(f"checkpoint     : step={meta['step']} val_nll={meta['best_val_nll']}")
    print(f"decoding       : temperature={cfg.temperature} top_k={cfg.top_k} "
          f"seed={args.seed} cache={not args.no_cache}\n")

    with autocast_for(device, args.precision or model_cfg.train.precision)():
        out = generate(model, ids, cfg, use_cache=not args.no_cache, generator=generator)

    print(tok.decode(out[0].tolist(), skip_special_tokens=True))


if __name__ == "__main__":
    main()
