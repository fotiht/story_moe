"""Token-weighted validation loss and perplexity.

Two rules this module exists to enforce:

  1. **Language loss only.** The MoE balancing term never enters a perplexity.
     Reporting exp(total_loss) would silently inflate the MoE's number and make
     the comparison meaningless.
  2. **Token weighting, not batch averaging.** Batches can hold different
     numbers of scored tokens, so the mean of per-batch losses is not the mean
     per-token loss. Sum the NLL over every scored target and divide once:

         mean_nll   = total_token_nll / scored_token_count
         perplexity = exp(mean_nll)

Each validation window is scored independently with context reset at its start,
so the first targets in a window are predicted from very little context. That
inflates perplexity relative to protocols that carry context across windows.
Both models use the identical protocol, so the comparison stays fair, but the
number is not comparable to a published result.
"""

from __future__ import annotations

import math
from contextlib import nullcontext
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()
def evaluate_blocks(
    model,
    blocks: np.ndarray,
    batch_size: int,
    device: str = "cpu",
    max_batches: int | None = None,
    autocast_ctx=None,
) -> dict[str, Any]:
    """Score every window in `blocks`. Returns NLL, perplexity and token counts.

    Restores the model's previous train/eval mode on the way out, so calling
    this mid-training cannot leave the model in eval by accident.
    """
    was_training = model.training
    model.eval()
    try:
        total_nll = 0.0
        scored = 0
        used = 0

        for start in range(0, blocks.shape[0], batch_size):
            if max_batches is not None and used >= max_batches:
                break
            chunk = torch.from_numpy(blocks[start : start + batch_size].astype(np.int64))
            window = chunk.to(device)
            x, y = window[:, :-1], window[:, 1:]

            ctx = autocast_ctx() if autocast_ctx is not None else nullcontext()
            with ctx:
                logits = model(x).logits
            # float32 for the reduction: summing thousands of fp16 terms drifts.
            nll = F.cross_entropy(
                logits.float().reshape(-1, logits.size(-1)),
                y.reshape(-1),
                reduction="sum",
            )
            total_nll += float(nll.detach())
            scored += int(y.numel())
            used += 1
    finally:
        model.train(was_training)

    mean_nll = total_nll / scored if scored else float("nan")
    return {
        "total_nll": total_nll,
        "scored_tokens": scored,
        "batches": used,
        "mean_nll": mean_nll,
        "perplexity": math.exp(mean_nll) if scored else float("nan"),
    }


def describe_protocol(meta: dict[str, Any], batch_size: int) -> str:
    """One line naming exactly what was scored, for the README and logs."""
    return (
        f"split={meta['split']} blocks={meta['n_blocks']} context={meta['block_size']} "
        f"stride={meta['stride']} discarded_tokens={meta['discarded_tokens']} "
        f"scored_targets={meta['total_scored_targets']} batch_size={batch_size} "
        f"context_reset_at_each_block_start=True"
    )
