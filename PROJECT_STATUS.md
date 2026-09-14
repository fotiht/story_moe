# PROJECT_STATUS

Updated: 2026-09-14

## Current milestone

**Days 1–2 complete and verified. Day 3 written, not yet run.**

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

## Day 3 — WRITTEN, UNVERIFIED

Still unrunnable in the cloud session (egress 403s pypi.org,
files.pythonhosted.org, download.pytorch.org, huggingface.co; the Linux
workspace on the Windows machine still fails to start). Syntax and config
arithmetic are checked; nothing touching torch has been executed.

New/changed:

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

```bash
pytest -q
python -m story_moe.train resume-smoke --config configs/debug_dense.yaml --device cpu
python -m story_moe.train train --config configs/debug_dense.yaml --device cpu --max-updates 20
```

Expect ~41 tests. `resume-smoke` should print `step 3 -> 4` and PASS.
The 20-update run should show `lm_loss` starting near 10.8 and falling, plus a
`tokens/s` figure — **that throughput number is what decides the real budget**.

Run the same throughput probe on Colab before committing to anything:

```bash
python -m story_moe.train train --config configs/debug_dense.yaml --max-updates 50
```

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
