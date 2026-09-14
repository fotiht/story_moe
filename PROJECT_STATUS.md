# PROJECT_STATUS

Updated: 2026-09-14

## Current milestone

**Days 1–3 complete and verified on CPU. Day 4 (sparse MoE) is next.**
No GPU run yet — the Colab throughput probe is the gating measurement.

## Day 1 — PASSED

Windows, Python 3.13, torch 2.14.0, transformers 5.17.0, datasets 5.0.1,
tokenizer `gpt2` (len 50257, EOS 50256).

| Split | Blocks | Unique stream tokens | Discarded | Scored targets |
| --- | --- | --- | --- | --- |
| train | 1,763 | 225,790 | 125 | 225,664 |
| validation | 339 | 43,427 | 34 | 43,392 |

`1 + (225790 - 129) // 128 = 1763` windows covering `1762*128 + 129 = 225665`
tokens, leaving exactly 125 discarded. ~226 tokens per story.

## Day 2 — PASSED

`pytest -q` → 29 passed. Overfit on 2 fixed blocks:

```
parameters : total=6,860,160  embedding=6,465,664 (94.2%)  body=394,496
step   0   lm_loss 10.7838   next-token acc   1.56%
step  25   lm_loss  4.8486   next-token acc  92.58%
step  50   lm_loss  1.2257   next-token acc  99.61%
step 200   lm_loss  0.0018   next-token acc 100.00%
step-0 loss 10.7838 vs ln(V) = 10.825, off by 0.041      verdict: PASS
```

`body = 394,496` reproduces exactly: 131,072 attention (4·128²·2) + 262,144 MLP
(2·128·512·2) + 1,280 LayerNorm. The 6,465,664 embedding was 6,432,896 token
embedding **plus the 32,768 temporary learned position table** (256×128), which
Day 3 deletes — expect `embedding = 6,432,896` and `total = 6,827,392` now.

## Day 3 — PASSED (CPU)

`pytest -q` → 40 passed in 10.49s.

```
parameters : total=6,827,392  embedding=6,432,896 (94.2%)  body=394,496
```

The learned position table is gone: embedding dropped from 6,465,664 to
6,432,896 = 50,257 x 128 exactly, and RoPE added no parameters.

`resume-smoke` → **PASS**, step 3 -> 4. The checkpoint carried
`['batcher_generator', 'best_val_nll', 'config', 'model', 'optimizer',
'processed_tokens', 'rng', 'scaler', 'step', 'tokenizer']`.

20-update CPU run:

```
step  0/20  lm 10.8206  lr 1.50e-05  gn 1.52   2634 tok/s
step 10/20  lm 10.4766  lr 1.65e-04  gn 1.41   2477 tok/s
step 19/20  lm 10.0686  lr 3.00e-04  gn 1.39   2192 tok/s
  validation  mean_nll 10.0315  ppl 22730.92  over 43,392 scored tokens
```

Loss decreases from ln(V) = 10.825; perplexity 22,731 against a uniform-predictor
50,257. That is a wiring result, not a model result — 81,920 tokens is nothing.
CPU throughput ~2,200-2,600 tok/s sets the floor the GPU must beat.

The "warmup is 667% of the run" warnings on the smoke tests are the guard working
as intended: `warmup_steps: 20` against a 3-update probe. Ignore it on probes;
heed it on real runs.

Implementation notes:

- **`rope.py`** — `rope_tables(d_head, max_seq_len, base)` precomputes cos/sin
  in float32, `[max_seq_len, d_head/2]`. `apply_rope(x, cos, sin, offset)`
  rotates `[B, H, T, Dh]` at absolute positions `offset .. offset+T-1`.
  Adjacent even/odd pair convention (GPT-J style), `theta_i = base**(-2i/Dh)`.
- **`attention.py`** — rotates Q and K, never V. Takes `past_len` and passes it
  to both the rotation offset and the mask.
- **`model.py`** — the learned positional embedding is **deleted**, not disabled.
  `rope_cos`/`rope_sin` are non-persistent buffers, so they move with `.to()`
  but do not bloat checkpoints.
- **`evaluate.py`** — token-weighted NLL: sum over every scored target, divide
  once, `ppl = exp(mean_nll)`. Language loss only. Restores train/eval mode.
- **`train.py`** — full loop: AdamW with decay on matrices only, linear warmup
  then cosine to 10%, gradient accumulation (loss divided before backward),
  fp16 autocast with unscale-before-clip, periodic validation, atomic
  `latest.pt` / `best.pt`, and resume that restores optimizer, scaler, RNG and
  sampler state.
- **`tests/test_rope.py`** — 12 tests. Two matter most: rotating a chunk at
  offset P equals rotating the whole sequence and slicing the tail, and the
  q·k dot product depends only on the position *difference*. Those are why a
  key can be rotated once, cached, and stay correct.
- **`tests/_helpers.py`** — shared tiny config; `pythonpath` now includes
  `tests`.

## Next action

Run the Colab throughput probe (`notebooks/story_moe_colab.ipynb`, cell 6). Its
`tok/s` figure decides open items 1 and 2 below, and nothing after Day 4 should
be launched until both are settled.

Day 4 then builds `moe.py`: four experts, a Top-2 softmax router with
renormalized selected weights, gather/MLP/scatter dispatch, and a dense
all-experts oracle used only in tests. The gate is output and gradient agreement
with that oracle, plus empty-expert handling and exactly 2N assignments.

## Decisions log

- `max_seq_len` 256, not 128 — the benchmark grid needs prompt 128 + 64 replay.
- Stride = `block_size`; windows share one boundary token, every token after the
  first is a target exactly once.
- `vocab_size` resolved at runtime from `len(tokenizer)`.
- `train.device` / `precision` / `require_gpu_name` asserted at startup.
- `count_parameters` splits embedding from body; body is the comparison number.
- Attention written explicitly, no SDPA, until the reference is matched.
- `warmup_steps` lowered 100 → 20: at 4,096 tokens per update a 1M-token budget
  is only 244 updates, so 100 would have been 41% of the run.
- Resume is documented as "state reloads and training continues", never as
  bitwise-identical resumed training.

## Open items

1. **Subset size for the real run.** 1,000 stories = 225,790 unique tokens, so
   the default 1M-token budget is **4.4 passes** over the same text (the trainer
   now prints this and warns above 2.0). For a single pass at 5M tokens the
   subset needs roughly 22,000 stories. Decide before launching the baseline.
2. **Debug tier vs 6-layer/384 tier for the headline comparison.** Decide from
   the Colab throughput probe above.
3. **Measure MoE throughput before freezing the dense token budget** — a
   ~200-step smoke run on Day 4, before the dense run is treated as final.
4. Repo is inside OneDrive; `.gitignore` keeps `data/` and `checkpoints/` out of
   Git but not out of sync. Checkpoints will be ~80 MB each at the debug tier
   (6.8M params × 4 bytes × 3 for weights + Adam moments), so `best.pt` plus
   `latest.pt` per run is real sync traffic once training starts.

## Resolved

- Initial-loss assertion — added and passing (10.7838 vs ln(V) = 10.825).
- Day 2's `float(tensor)` autograd warnings — replaced with `.item()`.
