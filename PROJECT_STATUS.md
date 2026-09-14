# PROJECT_STATUS

Updated: 2026-09-14

## Current milestone

**Days 1–3 verified. A100 probe measured. Day 4 (sparse MoE) written, not yet run.**

## Working constraint

The Claude session cannot run PyTorch: egress returns 403 for pypi.org,
files.pythonhosted.org, download.pytorch.org and huggingface.co, and the Linux
workspace on the Windows machine fails to start. Loop: Claude writes code and
commits it to the folder → push → run locally or in Colab → paste output back.
Pure-Python logic is verified in the sandbox; anything touching torch is
unverified until run.

## Verified results

**Day 1** — cache for 1,000/200 stories, block_size 128, stride 128, gpt2
(len 50257, EOS 50256):

| Split | Blocks | Unique tokens | Discarded | Scored targets |
| --- | --- | --- | --- | --- |
| train | 1,763 | 225,790 | 125 | 225,664 |
| validation | 339 | 43,427 | 34 | 43,392 |

~226 tokens per story.

**Day 2** — overfit collapsed 10.7838 → 0.0018 at 100% next-token accuracy;
step-0 loss within 0.041 of ln(50257) = 10.825.

**Day 3** — 40 tests pass. `resume-smoke` PASS (step 3 → 4). Parameters
`total=6,827,392 embedding=6,432,896 (94.2%) body=394,496`; body reproduces
exactly as 131,072 attention + 262,144 MLP + 1,280 LayerNorm.

**A100 probe** — 50 updates, debug tier, bf16, `NVIDIA A100-SXM4-40GB`,
torch 2.11.0+cu128:

```
step  0/50  lm 10.8205   4,537 tok/s (cumulative)
step 49/50  lm  8.9928  45,546 tok/s (cumulative)
  validation  mean_nll 9.0150  ppl 8225.16  over 43,392 scored tokens
```

Marginal rate between logged steps: 55,764 / 55,966 / 55,978 / 55,483 / 56,075
tok/s — **flat at ~55,800**, within 1%. CPU was ~2,200–2,600, so the A100 is
roughly 22x. At 55,800 tok/s, 20M tokens is about 6 minutes at the debug tier.

## Decisions from the probe

**Tier: the 6-layer / d_model 384 configuration carries the comparison.** At the
debug tier 94% of forward FLOPs are the embedding and output projection, so the
feed-forward change barely moves quality or throughput, and the models differ by
3.8% in parameters. At the larger tier it is 41.72M vs 60.60M total (bodies
22.42M vs 41.30M) and a 46% embedding share. New: `configs/train_dense.yaml`,
`configs/train_moe.yaml`.

**Budget: 20M processed tokens over 90,000 stories = 0.98 passes.** The spec's
1–5M suggestion is 18–90 seconds on this GPU; there is no reason to be that
small. 90k stories x 226 tokens ≈ 20.3M unique, so the run is close to a single
epoch rather than 4–22 repeats.

Experiment settings: block_size 512, microbatch 8, accum 4 → 16,384 tokens per
update, 1,220 updates, warmup 60 (5%), bf16, `require_gpu_name: A100`, cache at
`data/cache_512` (separate from the 128-token debug cache).

## Day 4 — WRITTEN, UNVERIFIED

`src/story_moe/moe.py` plus `tests/test_moe.py` (24 tests).

- `SparseMoE.route` — float32 softmax, `topk`, selected weights renormalized to
  sum to 1. **Not detached**: the language loss trains the router through them.
- Dispatch — per expert, gather its assigned rows, one MLP call, weight, and
  `index_add` back. k*N token-expert evaluations, not E*N.
- `balance_loss` — `f_e` over k*N assignments (detached), `P_e` the mean full
  probability before truncation, `L = E * sum_e f_e * P_e`. **Balanced value is
  1, not 0.**
- `RouterStats` — assignment fractions, mean probabilities, router entropy,
  aux loss, token count. Detached, opt-in via `collect_stats`.
- `need_aux=False` skips the balancing reduction entirely, so no diagnostic work
  distinguishes the cached and uncached benchmark paths later.
- `DecoderBlock` now returns `(x, aux, stats)`; `StoryLM` averages aux across MoE
  layers and computes `total = lm + aux_weight * aux`. Dense aux is exactly 0.

The gate is `test_gradients_match_dense_oracle`: input, expert **and** router
gradients must match a dense all-experts oracle using identical Top-2 weights.
Routing that produces correct outputs with wrong gradients is the classic silent
MoE failure.

## Next action

```bash
pytest -q                                   # expect ~64 tests
python -m story_moe.train overfit --config configs/debug_moe.yaml --steps 200
```

The MoE overfit should collapse like the dense one did, with `aux` sitting near
1.0 throughout — not falling toward zero.

Then, in Colab (notebook section 7): prepare the 512 cache, run
`train_dense.yaml`, then `train_moe.yaml` on the same GPU, same precision, same
budget.

## Decisions log

- `max_seq_len` 256 at the debug tier (benchmark grid needs prompt 128 + 64).
- Stride = `block_size`; every token after the first is a target exactly once.
- `vocab_size` resolved at runtime from `len(tokenizer)`.
- `train.device` / `precision` / `require_gpu_name` asserted at startup.
- `count_parameters` splits embedding from body; body is the comparison number.
- Attention explicit, no SDPA, until the reference is matched.
- `warmup_steps` 100 → 20 at the debug tier, 60 at the experiment tier (~5%).
- Resume is "state reloads and training continues", never bitwise-identical.
- Colab runs torch 2.11.0+cu128, Windows runs 2.14.0 — every quoted number must
  say which.

## Open items

1. **Microbatch is likely far too small for an A100.** The probe pushed 4 x 128 =
   512 tokens per forward; the bf16 logits tensor was 49 MiB against 40 GB of
   memory. Raising microbatch and lowering accum by the same factor leaves the
   effective batch and the optimization trajectory identical but should improve
   throughput substantially. The experiment config uses 8 x 4; worth a quick
   sweep before the real runs.
2. **Measure MoE throughput before treating the dense run as final.** Gather/
   scatter dispatch can be slower per token; if it is much slower, the matched
   token budget may not fit the wall clock.
3. Repo is inside OneDrive; `.gitignore` keeps `data/` and `checkpoints/` out of
   Git but not out of sync. Experiment checkpoints will be ~500 MB each at
   41.7M/60.6M parameters with Adam moments — these go to Drive via `--out-dir`,
   not into the repo folder.

## Resolved

- Subset size and tier — settled by the probe, above.
- Initial-loss assertion — passing.
- `float(tensor)` autograd warnings — replaced with `.item()`.
