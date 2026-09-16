# PROJECT_STATUS

Updated: 2026-09-16 (days 1 to 7 complete)

## Current milestone

All seven days are done and measured. Days 1 to 5 as before; Day 6 added the KV
cache and Day 7 the generation CLI, the cache benchmark and the results tables.
The suite passes at 111, so the earlier dead-code cleanup is confirmed.

Nothing in the README is projected any more. Every figure came off a run.

## Day 6 and 7 results

### The cache is correct

The fp32 gate passed at all 22 benchmark grid points on the trained models:
cached and uncached decode produced identical tokens. The lockstep comparison,
which feeds both paths the same tokens and compares logits directly, put the
largest fp32 difference at 2.1e-5 against logits of magnitude 16, a relative
1e-6. That is reassociation noise, which is what an identity computed two ways
should look like.

### bf16 token divergence is rounding, not a bug

In bf16 some points decoded different tokens. The lockstep delta there is 0.0625
at batch 1, which is exactly one ULP: bf16 has 8 mantissa bits, so near 15 the
spacing is 2^3 * 2^-8 = 0.0625. Several recorded flips had a top-2 gap of
exactly 0.0, meaning two tokens with identical bf16 logits, where argmax falls
to index order and any perturbation flips it. One flipped token then makes every
later token differ, which is why free-running greedy comparison looked like
total failure.

The benchmark now fails only on an fp32 mismatch and reports bf16 divergence
with the logit deltas beside it. That change was made after the measurement, not
before it, and the first version did hard-fail the run.

### Timings, A100, bf16, median of 5 after 2 warmup

| Model | Batch | Prompt | Replay | Prefill ms | Cached ms/tok | Uncached ms/tok | Speedup |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Dense | 1 | 8 | 8 | 9.35 | 7.05 | 8.27 | 1.17x |
| Dense | 1 | 256 | 256 | 9.14 | 7.97 | 8.13 | 1.02x |
| Dense | 32 | 64 | 64 | 9.38 | 8.09 | 8.02 | 0.99x |
| Dense | 32 | 256 | 256 | 9.73 | 8.39 | 15.19 | 1.81x |
| MoE | 1 | 8 | 8 | 17.25 | 12.38 | 17.69 | 1.43x |
| MoE | 1 | 256 | 256 | 19.32 | 13.94 | 17.63 | 1.26x |
| MoE | 32 | 256 | 256 | 19.99 | 17.64 | 25.90 | 1.47x |

### The prediction was wrong, and why

I predicted the per-token speedup would grow roughly linearly with sequence
length. It does not. Decode at this model size is bound by Python dispatch and
kernel launch, not arithmetic:

- Dense cached decode costs 7.05 ms/token at 16 tokens of context and 7.97 at
  512. A 32x range of context, 13% of time.
- Dense cached decode costs 8.39 ms/token at batch 32, where each step produces
  32 tokens instead of 1. Thirty-two times the work, 5% more time.
- Prefill costs 9.35 ms for 8 tokens and 9.73 ms for 8,192.

That is a floor of roughly 8 ms per step dense and 18 ms MoE, under which saved
arithmetic is invisible. The uncached path matches the cached path until its
recomputation exceeds that floor. At batch 1 it never does within a 512-token
context; at batch 32 the crossover sits between 256 and 512 tokens of context
(8.02 ms at 256, 15.19 at 512).

MoE shows a bigger batch-1 speedup than dense (1.26x to 1.43x against 1.02x to
1.17x) for the same reason inverted: gather, four experts and scatter put more
real arithmetic under the same overhead.

Batching is the larger lever: dense cached decode goes from 125 tok/s at batch 1
to 3,814 tok/s at batch 32, 30x from the same cache, because batching is what
makes a step compute-bound in the first place.

### The memory prediction did hold

At batch 32 and 512 tokens of context, the cached path peaks at 517 MiB against
the uncached path's 815 MiB (MoE: 615 against 936), even though only the cached
path stores keys and values. The uncached path materializes a [32, 6, t, t]
score matrix every step, which costs more than the cache avoids.

### Bugs found and fixed during Day 7

1. `torch.multinomial` needs its generator on the logits' device. The CLI built
   a CPU generator against CUDA logits and crashed. The CPU-only tests could not
   catch this, the same shape as the Day 3 GPU-resume bug.
2. `default_grid` started at (16, 32) and returned nothing for any model with
   max_seq_len under 48, so the benchmark refused to run rather than picking
   smaller points. Caught by its own test.
3. Fixing 2 exposed a latent crash: at short grid points on a fast GPU,
   `cached_total - prefill` falls below timer resolution, making the speedup
   None, which the print line formatted with `:5.2f`. Now prints `n/a`.
4. The lockstep diagnostic reported batch row 0 rather than the row that
   actually diverged, so at batch 32 it printed two identical tokens as evidence
   of a disagreement. Found by reading the run output, not by a test.
5. `_cache_reserved_mib` computed at fp32 always, overstating a bf16 run's cache
   by exactly 2x.

## Day 5 results

Dense and MoE, 1,220 updates each at 16,384 tokens per update, 19,988,480
processed tokens, 0.9918 passes, `NVIDIA A100-SXM4-40GB`, bf16, torch
2.11.0+cu128, one shared 512-token cache.

| | Total params | Body params | Val NLL | Perplexity | Cumulative tok/s | Wall clock | Peak mem |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `train_dense` | 41,721,984 | 22,423,296 | 2.3764 | 10.77 | 102,579 | 195 s | 4,139 MiB |
| `train_moe` | 60,605,568 | 41,306,880 | 2.3054 | 10.03 | 54,967 | 364 s | 4,648 MiB |

The MoE wins by 0.071 nats, 6.9% lower perplexity, and leads at all seven
validation points (dense 30.65 / 18.61 / 14.55 / 12.41 / 11.31 / 10.79 / 10.77
against MoE 28.23 / 17.01 / 13.33 / 11.50 / 10.51 / 10.06 / 10.03). It carries
1.84 times the body parameters and cost 1.87 times the wall clock, so this is a
matched-token win, not a compute-matched one. Neither curve has flattened.

```
dense  step     0/1220  lm 10.9144  gn 7.61   24,664 tok/s
dense  step  1219/1220  lm  2.3408  gn 1.02  102,579 tok/s
moe    step     0/1220  lm 10.8465  aux 1.0214  gn 7.21   14,765 tok/s
moe    step  1219/1220  lm  2.2724  aux 1.0220  gn 0.73   54,967 tok/s
```

Both step-0 language losses sit within 0.09 of ln(50257) = 10.825, so the
initial-loss assertion held at the experiment tier too.

MoE gradient norms ran consistently below dense in steady state, about 0.73
against 1.02 at the last step. Not investigated; noting it rather than
explaining it.

### Routing over the full run

The imbalance the 30-update probe showed forming did not continue. Fractions
averaged over the six MoE layers and the four microbatches of the logged update:

```
step    0  [0.277 0.234 0.235 0.254]  max/min 1.18  aux 1.0214
step  100  [0.265 0.314 0.222 0.199]  max/min 1.58  aux 1.0588
step  200  [0.256 0.288 0.225 0.232]  max/min 1.28  aux 1.0276
step  600  [0.257 0.283 0.222 0.238]  max/min 1.27  aux 1.0203
step 1219  [0.259 0.286 0.218 0.237]  max/min 1.31  aux 1.0220
```

Worst point is step 100, at the end of the 60-update warmup. From step 200 on the
fractions hold a stable band, max/min 1.27 to 1.32, no expert under 0.21. The
probe's max/min of 1.97 at step 20 was a transient, and reading 30 updates as a
trend was the wrong call. `aux` moved across a range of about 0.04 over the whole
run while the fractions moved visibly, which is one more reason not to use it as
the balance diagnostic.

Caveat on the numbers above: the log records the layer average, so per-layer skew
in opposite directions would partly cancel. `RouterStats` is collected per layer;
only the mean is written out. Worth a per-layer dump on Day 7 if expert behaviour
gets discussed at all.

### Verified on the A100 (torch 2.11.0+cu128, bf16)

`pytest -q` gives 77 passed on Windows, torch 2.14.0.

GPU resume smoke passed, step 3 to 4. Checkpoint keys
`['batcher', 'best_val_nll', 'config', 'epochs_seen', 'model', 'optimizer',
'processed_tokens', 'rng', 'scaler', 'step', 'tokenizer']`. The CPU run could
not perform this test at all, and passing it is what makes a long Colab run
survivable.

First real training result, at the debug tier: 244 updates, 999,424 processed
tokens over 1,000 stories, which is 4.43 passes and therefore memorization-
flavoured.

```
step   0/244  lm 10.8170
step 243/244  lm  5.4466
  validation  mean_nll 5.5436  ppl 255.59  over 43,392 scored tokens
```

Perplexity 255.6 against a uniform predictor's 50,257. Marginal throughput over
the last 40 updates was about 54,900 tok/s, matching the earlier probe within 2%.

The `results/*.json` history field works: a controlled 3-update run with the
file deleted first produced `history entries: 3`. An earlier check reported 0 on
a file that could not be traced to a specific run. Unexplained, not reproduced,
and not blocking. Re-check if it recurs.

### Reliability fixes from external review, applied and verified

Three real bugs, all of which produce plausible runs rather than crashes:

1. GPU resume was broken, and is now verified fixed on an A100.
   `torch.load(..., map_location=device)` moved the
   saved RNG byte tensors onto the GPU, and `torch.set_rng_state` requires a CPU
   byte tensor. The CPU resume-smoke test could not catch this because
   map_location was "cpu" there. Now always loaded on CPU (`load_state_dict`
   copies into the model's on-device tensors, and the optimizer moves its own
   state), with a defensive `.cpu()` on every RNG tensor.
2. There was no cache/config agreement check. Pointing a 512-token config at the
   128-token cache ran fine and reported 4x the true processed tokens.
   `check_cache_matches_config` now validates block size, stride, tokenizer
   identity, vocab size, dataset, seed and story count before training starts.
3. Sampling was with replacement. `torch.randint` over n blocks reaches only
   `1-(1-1/n)^n ~= 63%` of them in n draws, so "0.98 passes" was a budget ratio,
   not coverage. `Batcher` now consumes a shuffled permutation and reshuffles at
   epoch boundaries; permutation, cursor and epoch are checkpointed.

Plus: routing statistics are now collected on logging steps and aggregated over
every microbatch of the update, and several overstated claims were corrected
(see the Day 4 note on the auxiliary loss, and the FLOP accounting above).

## Day 5 probes on the A100

30 updates of each experiment config, bf16, `NVIDIA A100-SXM4-40GB`, torch
2.11.0+cu128, sharing one 512-token cache.

| | Steady tok/s | Peak memory | Total params | Body params | Val NLL at 30 updates |
| --- | --- | --- | --- | --- | --- |
| `train_dense` | 122,100 | 4,139 MiB | 41,721,984 | 22,423,296 | 8.1304 |
| `train_moe` | 60,800 | 4,642 MiB | 60,605,568 | 41,306,880 | 7.9990 |

The MoE is 2.01 times slower per token. That is the gather and scatter dispatch
cost, measured rather than assumed, and it does not threaten the plan: at 20M
tokens the runs cost 2.7 and 5.5 minutes. Open item 2 is closed.

The experiment tier is 2.2 times faster per token than the debug tier's 55,800
despite six times the compute, because the larger tensors use the card better.
Peak memory is 11% of 40 GB.

Cache, prepared once and shared: train 39,361 blocks over 90,000 stories,
20,152,860 unique tokens, 27 discarded, 20,152,832 scored targets; validation
854 blocks over 2,000 stories, 210 discarded, 437,248 scored targets. 223.9
tokens per story, and the 20M-token budget covers 0.99 passes.

Expert assignment fractions over the 30 MoE updates, where uniform is 0.250:

```
step  0  [0.277 0.234 0.235 0.254]   max/min 1.18
step 10  [0.268 0.299 0.188 0.245]   max/min 1.59
step 20  [0.312 0.303 0.227 0.158]   max/min 1.97
step 29  [0.286 0.310 0.215 0.189]   max/min 1.64
```

Imbalance forming, not collapse. Warmup is 60 updates and the probe was 30, so
the learning rate never reached peak; the trend is suggestive only. `aux` moved
1.021 to 1.076 alongside it, so once the probabilities stop being uniform the
auxiliary loss does carry some signal. The assignment fractions remain the
direct reading.

## Code cleanup, written but not re-tested

A de-AI pass removed four things. None changes behaviour, and all four are
unverified until `pytest -q` runs again.

1. `DenseMLP` in model.py and `Expert` in moe.py were byte-identical after
   docstrings were stripped. Now one `FeedForward` in moe.py, imported by
   model.py, which is the direction the dependency already ran.
2. `balanced_reference_value(n_experts)` deleted. It ignored its argument and
   returned a literal 1.0, and its only caller was a test asserting it returns
   1.0. The same test already checks the loss itself.
3. `Batcher.sequential_batches` and `RouterStats.as_dict` deleted, zero callers
   each.
4. The `Batcher` no longer stashes the torch module on the instance. The lazy
   import stays, because data.py's packing functions have to import without
   torch present.

## Working constraint

Code was written in an environment that could not run PyTorch, so nothing was
verified where it was written. Every change had to be run on the Windows machine
or in Colab before it could be believed. Two habits came out of that and are
worth keeping regardless: `data.py` is layered so its packing logic imports
without torch and can be tested anywhere, and any claim about behaviour in this
file is labelled by the machine and precision that produced it.

## Verified results

Day 1. Cache for 1,000/200 stories, block_size 128, stride 128, gpt2
(len 50257, EOS 50256):

| Split | Blocks | Unique tokens | Discarded | Scored targets |
| --- | --- | --- | --- | --- |
| train | 1,763 | 225,790 | 125 | 225,664 |
| validation | 339 | 43,427 | 34 | 43,392 |

~226 tokens per story.

Day 2. The overfit collapsed from 10.7838 to 0.0018 at 100% next-token accuracy;
step-0 loss within 0.041 of ln(50257) = 10.825.

Day 3. 40 tests pass, `resume-smoke` passes (step 3 to 4), and the parameters read
`total=6,827,392 embedding=6,432,896 (94.2%) body=394,496`; body reproduces
exactly as 131,072 attention + 262,144 MLP + 1,280 LayerNorm.

A100 probe, 50 updates, debug tier, bf16, `NVIDIA A100-SXM4-40GB`,
torch 2.11.0+cu128:

```
step  0/50  lm 10.8205   4,537 tok/s (cumulative)
step 49/50  lm  8.9928  45,546 tok/s (cumulative)
  validation  mean_nll 9.0150  ppl 8225.16  over 43,392 scored tokens
```

Marginal rate between logged steps: 55,764 / 55,966 / 55,978 / 55,483 / 56,075
tok/s, flat at about 55,800 and within 1%. CPU was 2,200 to 2,600, so the A100
is roughly 22 times faster. At 55,800 tok/s, 20M tokens is about 6 minutes at the debug tier.

## Decisions from the probe

Tier: the 6-layer, d_model 384 configuration carries the comparison.
Counting forward matmul FLOPs per token (output projection `2VD`, attention
projections `2*4D^2*L`, QK^T and AV `2*2TDL`, MLP `2*2DmL`), the debug tier is
93.3% output projection, 3.8% MLP and 2.9% attention, so the feed-forward change
touches under 4% of the compute and the models differ by 3.8% in parameters. The
experiment tier splits 43.8% / 42.8% / 13.4%, so the MoE swap touches over 40%.
At the larger tier the totals are 41.72M against 60.60M (bodies 22.42M against
41.30M) with a 46% embedding share. New: `configs/train_dense.yaml`,
`configs/train_moe.yaml`.

Budget: 20M processed tokens over 90,000 stories, which the runs confirmed at
0.9918 passes. (Sampling is
now without replacement, so this ratio is genuine coverage; with the previous
`torch.randint` sampler the same budget would have reached only 63% of blocks.)
The spec's 1M to 5M suggestion is 18 to 90 seconds on this GPU; there is no reason to be that
small. 90k stories x 226 tokens ≈ 20.3M unique, so the run is close to a single
epoch rather than 4 to 22 repeats.

Experiment settings: block_size 512, microbatch 8, accum 4, giving 16,384 tokens per
update, 1,220 updates, warmup 60 (5%), bf16, `require_gpu_name: A100`, cache at
`data/cache_512` (separate from the 128-token debug cache).

## Day 4 passed

`pytest -q` gave 61 passed. The gate, `test_gradients_match_dense_oracle`, passed:
input, expert and router gradients all agree with a dense all-experts oracle
using identical Top-2 weights.

MoE overfit on `configs/debug_moe.yaml`:

```
parameters : total=7,090,560  embedding=6,432,896 (90.7%)  body=657,664
step   0   lm_loss 10.8397  aux 1.0061  acc   1.56%
step  50   lm_loss  1.5534  aux 1.0175  acc  87.50%
step 200   lm_loss  0.0025  aux 0.9989  acc 100.00%
step-0 loss 10.8397 vs ln(V) = 10.825, off by 0.015     verdict: PASS
```

`body = 657,664` reproduces exactly: 131,072 attention + 524,288 experts
(4 x 2 x 128 x 256 x 2 layers) + 1,280 LayerNorm + 1,024 router. MoE body /
dense body = 657,664 / 394,496 = 1.667, matching the config-shape derivation.

`aux` stayed between 0.9989 and 1.0175 for all 200 steps. This corrects an
earlier reading: that does NOT demonstrate balanced routing. With
`L = E * sum_e f_e * P_e`, near-uniform probabilities `P_e ~= 1/E` give
`L = E * (1/E) * sum_e f_e = 1` for ANY assignment distribution, including every
token collapsing onto one expert. Worked counterexample: E=4, P uniform,
f = [1, 0, 0, 0] still yields exactly 1.0000. An untrained router has
near-uniform probabilities, so aux ~= 1 early in training is close to
uninformative. `assignment_fraction` carries the information instead, which is
why the training loop now logs it.

MoE converged slightly slower than dense (step 25: 5.06 vs 4.85), which is
expected: the router is an additional thing to learn.

### What Day 4 added

- `SparseMoE.route` does a float32 softmax and `topk`, with selected weights renormalized to
  sum to 1, and never detached, so the language loss trains the router.
- Dispatch, per expert: gather assigned rows, one MLP call, weight,
  `index_add` back. k*N token-expert evaluations, not E*N, with a test that
  counts them.
- `balance_loss` uses `f_e` over k*N assignments (detached), `P_e` the mean full
  probability before truncation, `L = E * sum_e f_e * P_e`. Balanced value 1.
- `RouterStats` holds assignment fractions, mean probabilities, entropy, aux loss,
  token count. Detached, opt-in.
- `need_aux=False` skips the balancing reduction in inference, so no diagnostic
  work distinguishes the cached and uncached benchmark paths on Day 7.
- `DecoderBlock` returns `(x, aux, stats)`; `StoryLM` averages aux across MoE
  layers. Dense aux is exactly 0.

## Next action

The 7-day plan is finished. Remaining housekeeping:

1. Commit and push the Day 7 fixes and the updated docs.
2. Copy `bench_dense.json`, `bench_moe.json`, `bench_dense_b32.json` and
   `bench_moe_b32.json` into the repo under `results/` so the README's tables
   have their source data beside them. They currently sit in Drive next to the
   checkpoints.

The trained checkpoints live in Drive under
`story_moe_artifacts/checkpoints/train_dense` and `.../train_moe`. Keep them:
they are what any further generation sample or benchmark needs, and retraining
costs 10 minutes of A100 time but loses the exact weights the README describes.

Optional work, if the project continues past the plan:

- Per-layer routing statistics in the log rather than the six-layer mean, which
  would answer whether the layers specialize or all route alike.
- A batch-size sweep on the cache benchmark (1, 4, 16, 32, 64) to locate the
  overhead crossover properly instead of bracketing it between two points.
- SDPA in place of the explicit attention, now that the explicit version is the
  verified reference to match against.

## Decisions log

- `max_seq_len` 256 at the debug tier (benchmark grid needs prompt 128 + 64).
- Stride = `block_size`; every token after the first is a target exactly once.
- `vocab_size` resolved at runtime from `len(tokenizer)`.
- `train.device` / `precision` / `require_gpu_name` asserted at startup.
- `count_parameters` splits embedding from body; body is the comparison number.
- Attention explicit, no SDPA, until the reference is matched.
- `warmup_steps` cut from 100 to 20 at the debug tier, 60 at the experiment tier (~5%).
- Resume is "state reloads and training continues", never bitwise-identical.
- Colab runs torch 2.11.0+cu128 and Windows runs 2.14.0, so every quoted number
  must say which.

## Open items

1. Microbatch may still be small for an A100, though less so than at the debug
   tier. The experiment probes peaked at 4,139 and 4,642 MiB, which is 11% of the
   card. Raising microbatch and lowering accum by the same factor keeps the
   effective batch and the language loss identical up to floating point. It does
   not keep the objective as a whole identical: `balance_loss` estimates `f_e` and `P_e` from
   one microbatch, so changing the microbatch size changes the auxiliary
   gradient. Treat a microbatch change as a config change to disclose, not a
   free optimization. Worth a sweep before the real runs; keep it identical
   across the dense and MoE runs either way.
2. Resolved. MoE throughput measured at 60,800 tok/s against the dense model's
   122,100, a 2.01x slowdown that costs 5.5 minutes instead of 2.7 at the full
   budget.
3. Repo is inside OneDrive; `.gitignore` keeps `data/` and `checkpoints/` out of
   Git but not out of sync. Experiment checkpoints will be ~500 MB each at
   41.7M/60.6M parameters with Adam moments, so these go to Drive via `--out-dir`,
   not into the repo folder.

## Resolved

- Subset size and tier, settled by the probe above.
- Initial-loss assertion, passing.
- `float(tensor)` autograd warnings, replaced with `.item()`.
