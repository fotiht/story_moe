# story_moe: decoder-only transformer with RoPE, sparse Top-2 MoE, and KV cache

A next-token language model trained from random initialization on a fixed
TinyStories subset. It continues short story prompts; it is not an
instruction-tuned assistant.

Status: day 5 of 7 complete. The data pipeline, causal attention, RoPE, the loss,
validation, checkpoint/resume and the sparse Top-2 MoE are all verified, the last
of them against a dense all-experts oracle on outputs and gradients. Resume is
verified on an A100, not only on CPU. Both full 20M-token runs have finished on
the same A100 in bf16, so the headline comparison below is measured. The KV
cache and its benchmarks are Day 6 and Day 7 and do not exist yet; the cache
benchmark table still reads NOT MEASURED.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate           # Windows
pip install -e ".[dev]"
```

PyTorch is installed from PyPI by the line above. On Colab, torch is already
present, so install the rest with `pip install transformers datasets pyyaml`.

## Commands that exist today

```bash
# 1. Tokenize the TinyStories subset and write the packed uint16 cache.
python -m story_moe.data prepare --config configs/debug_dense.yaml

# 2. Print a batch and assert the Day 1 data contract (shape, dtype, ID range,
#    single shift), plus a tokenizer round trip.
python -m story_moe.data show --config configs/debug_dense.yaml

# 3. Overfit a couple of fixed blocks. Wiring test, not an experiment.
python -m story_moe.train overfit --config configs/debug_dense.yaml --steps 200

# 4. Save, reload, and take another update.
python -m story_moe.train resume-smoke --config configs/debug_dense.yaml

# 5. Train the baseline.
python -m story_moe.train train --config configs/debug_dense.yaml
python -m story_moe.train train --config configs/debug_dense.yaml --resume checkpoints/debug_dense/latest.pt

# 6. Run the test suite.
pytest -q
```

`train.device` defaults to `cuda`; pass `--device cpu` to run anywhere. The run
refuses to start if the GPU does not match `train.require_gpu_name`, or if the
requested precision is unsupported. Colab reassigns accelerators between
sessions, and a dense/MoE pair split across two of them is not comparable.

Overrides that do not need a config edit: `--device`, `--precision`,
`--require-gpu-name`, `--out-dir`, `--data-cache`, `--max-tokens`,
`--max-updates`, `--resume` on `train`; `--cache-dir`, `--train-stories`,
`--val-stories` on `data prepare`.

## Running on Colab

`notebooks/story_moe_colab.ipynb` sets up and launches; it contains no model code.

Code reaches Colab through GitHub. One-time setup, from this folder on Windows:

```bash
git init
git add .
git status                 # confirm .venv/, data/ and checkpoints/ are NOT listed
git commit -m "story_moe: days 1-3"
gh repo create story_moe --private --source=. --push
# without the gh CLI: create an empty PRIVATE repo on github.com, then
#   git remote add origin https://github.com/<you>/story_moe.git
#   git branch -M main && git push -u origin main
```

After that: `git add -A && git commit -m "..." && git push` on Windows, and the
notebook's clone/pull cell picks the change up in Colab. Put the repo URL in that
cell once. Keep the repo private. Nothing here is secret, but an `HF_TOKEN` or a
stray checkpoint is much easier to leak than to un-leak.

Colab behaves differently from the other environments in three ways:

- Never run `pip install torch` there. Its torch is built against the driver on
  that VM, and the PyPI wheel can replace it with one that does not match, after
  which the GPU fails to initialize. The notebook uses `pip install -e . --no-deps`
  so `pyproject.toml` cannot pull torch in.
- The VM's disk is deleted when the session ends, and sessions end on their own.
  Checkpoints and the token cache go to mounted Drive via `--out-dir` and
  `--data-cache`; `--resume` picks the run back up from there.
- Pin the accelerator. The notebook's first cell prints the GPU name and whether
  bf16 is supported, then tells you what to pass for `--require-gpu-name` and
  `--precision`. Use the same pair for the dense and the MoE run, and record it.
  Otherwise the two runs are not comparable and the tokens/s figures mean nothing
  next to each other.

Both configs share one data cache, so step 1 is run once and serves the dense
and MoE runs alike.

Measured caches, both with the GPT-2 tokenizer and stride equal to block_size:

| Cache | Split | Stories | Blocks | Unique stream tokens | Discarded | Scored targets |
| --- | --- | --- | --- | --- | --- | --- |
| debug, T=128 | train | 1,000 | 1,763 | 225,790 | 125 | 225,664 |
| debug, T=128 | validation | 200 | 339 | 43,427 | 34 | 43,392 |
| experiment, T=512 | train | 90,000 | 39,361 | 20,152,860 | 27 | 20,152,832 |
| experiment, T=512 | validation | 2,000 | 854 | 437,459 | 210 | 437,248 |

The validation row's unique-token count is derived from the block count and the
discarded remainder rather than read off the log; the same arithmetic reproduces
the train row's reported 20,152,860 exactly.

That is 226 tokens per story in the debug cache and 223.9 in the experiment
cache, which sets the epoch count for any given processed-token budget. At
16,384 tokens per update, the 20M-token budget covers 0.99 passes.

## Data contract

Fixed for both models. Changing any of it invalidates the comparison.

| Property | Value |
| --- | --- |
| Dataset | `roneneldan/TinyStories`, official train / validation splits |
| Tokenizer | pretrained GPT-2 (`AutoTokenizer.from_pretrained("gpt2")`), no custom BPE |
| Special tokens | `add_special_tokens=False`, then exactly one EOS appended per story |
| Padding | none; there is no pad token and no padded batch |
| Packing | stories concatenated per split, cut into windows of `block_size + 1` |
| Stride | `block_size` (default), so consecutive windows share one boundary token |
| Sampling | shuffled permutation per epoch, without replacement |
| Shift | `inputs = window[:-1]`, `targets = window[1:]` |
| Remainder | discarded, never padded, never scored; count recorded in `data/cache/<split>.json` |
| Story boundaries | attention crosses them inside a block; EOS does not reset attention |

Subset selection is `random.Random(seed).sample(...)`, and the selected index
list, its hash, the split size and the tokenizer fingerprint are all written to
`data/cache/<split>.json` so a later run can prove it used the same data.

## Architecture

Pre-norm decoder blocks, multi-head causal self-attention, RoPE on queries and
keys only, LayerNorm, GELU experts, tied input/output embeddings, bias-free
linear projections. The feed-forward sublayer is either a dense MLP or four
experts with Top-2 routing and a renormalized softmax gate.

The MoE computes only the experts a token selected: each expert's assigned rows
are gathered, run through one MLP call, weighted, and scatter-added back. Over
N tokens that is k*N token-expert evaluations, not E*N. A dense all-experts
version exists solely as a test oracle.

The balancing objective is a Top-k adaptation of the Switch loss. `f_e` is
expert e's share of the k*N assignments, `P_e` the mean full softmax probability,
and `L = E * sum_e f_e * P_e`. Under uniform routing this equals 1, not 0, so it
should never be driven toward zero. A value of 1 is also no evidence that routing
is balanced; see "Reading the routing diagnostics" below. Gradients reach the
router through `P_e` (the probabilities before top-k truncation) and through the
renormalized selected weights, which are never detached. The top-k indices
themselves are discrete and carry no gradient.

Position comes from RoPE alone. There is no learned positional embedding table.
Attention masking uses `j <= past_len + i` in its general form from the start, so
enabling the KV cache changes the arguments and not the masking logic.

### Parameter accounting

With the GPT-2 vocabulary (50,257) the tied embedding dominates a small model,
so report both the total and the non-embedding ("body") count. Neither
substitutes for the other: the total is what a reader means by "model size",
while the body is what differs between the two architectures. RoPE contributes
no parameters, so "embedding" is exactly the tied token embedding.

The debug dense row is confirmed against an instantiated model: `body = 394,496`
= 131,072 attention + 262,144 MLP + 1,280 LayerNorm, and `embedding = 6,432,896`
= 50,257 × 128. The other rows are computed from config shapes.

| Config | Embedding | Attention | MLP / experts | Total | Embedding share |
| --- | --- | --- | --- | --- | --- |
| Debug dense (2L, D=128, m=512) | 6.43M | 0.13M | 0.26M | 6.83M | 94% |
| Debug MoE (2L, D=128, 4×m=256) | 6.43M | 0.13M | 0.52M | 7.09M | 91% |
| Experiment dense (6L, D=384, m=4096) | 19.30M | 3.54M | 18.87M | 41.72M | 46% |
| Experiment MoE (6L, D=384, 4×m=2048) | 19.30M | 3.54M | 37.75M | 60.60M | 32% |

All four rows are now confirmed against instantiated models. Exact counts:
debug dense 6,827,392 total and 394,496 body; debug MoE 7,090,560 and 657,664;
experiment dense 41,721,984 and 22,423,296; experiment MoE 60,605,568 and
41,306,880. The debug MoE body reproduces as 131,072 attention + 524,288 experts
+ 1,280 LayerNorm + 1,024 router.

The debug tier is a plumbing demonstration. Counting forward matmul FLOPs per
token (output projection `2VD`; attention projections `2·4D²L`; the QK^T and AV
matmuls `2·2TDL`; MLP `2·2DmL`), the debug tier spends 93.3% on the tied output
projection, 3.8% on the MLP and 2.9% on attention. The feed-forward change
therefore touches under 4% of the compute, and the two models differ by only
3.8% in total parameters. At the experiment tier the split is 43.8% output
projection, 42.8% MLP and 13.4% attention, so the MoE swap touches over 40% of
the forward cost. The main experiment runs at 6 layers and d_model 384:
`configs/train_dense.yaml` and `configs/train_moe.yaml`, 41.72M against 60.60M
parameters, bodies 22.42M against 41.30M.

### Measured throughput

All on `NVIDIA A100-SXM4-40GB` except the CPU row, marginal rate rather than
cumulative, measured over the later updates of each probe.

| Setup | Precision | Tokens/s | Peak memory |
| --- | --- | --- | --- |
| Debug tier, microbatch 4, T=128 | bf16 | 55,800 | not recorded |
| Experiment dense, microbatch 8, T=512 | bf16 | 122,100 | 4,139 MiB |
| Experiment MoE, microbatch 8, T=512 | bf16 | 60,800 | 4,642 MiB |
| Debug tier, microbatch 4, T=128, CPU (Windows) | fp32 | 2,200 to 2,600 | n/a |

Two things fall out of this. The MoE is 2.01 times slower per token than the
dense model at the same tier, which is the gather and scatter dispatch cost and
is why the spec warns against assuming a speedup. At 20M tokens that predicted
2.7 minutes for the dense run against 5.5 minutes for the MoE. The full runs came
in at 3.2 and 6.1 minutes, a ratio of 1.87, the difference being validation
passes and checkpoint writes that the probes did not include.

The experiment tier also runs 2.2 times faster per token than the debug tier
despite six times the compute, because the larger tensors use the GPU better.
Peak memory is 11% of the card, so microbatch 8 still leaves headroom.

Marginal rate within each probe was flat to within 1%. Colab runs torch
2.11.0+cu128 and the Windows venv runs 2.14.0, so any number quoted has to say
which produced it.

## Comparison protocol

Same tokenizer, data subset, validation protocol, attention backbone, context,
precision, effective batch, and processed-token budget. Both models trained from
scratch. `tests/test_config.py` enforces that the two YAML files differ only in
the feed-forward fields.

Default matching is approximate active-MLP compute: with expert width `m` and
Top-2, dense width is `2*m`. This matches selected MLP arithmetic per token; it
is not equal runtime and not equal full-model FLOPs. The MoE still stores four
experts. Router and dispatch overhead are extra.

Because Colab reassigns accelerators between sessions, `train.device`,
`train.precision` and `train.require_gpu_name` are asserted at startup. A dense
run on an A100 in bf16 and an MoE run on a T4 in fp16 are not comparable.

## Sampling and coverage

Training batches come from a shuffled permutation consumed in order, reshuffled
at each epoch boundary. With `torch.randint`, which samples with replacement,
drawing n batches' worth from n blocks reaches only `1 - (1-1/n)^n ≈ 63%` of
them, so a budget described as "one pass" would leave a third of the data unseen
while showing other blocks twice. With a permutation, "0.98 passes" means 98% of
blocks, each exactly once. The sampler's permutation, cursor and epoch are saved
in every checkpoint, so a resume continues from the same position instead of
reshuffling.

## Reading the routing diagnostics

Uniform routing over four experts would put 0.250 of the assignments on each.
The 30-update probe showed the fractions drifting away from that, reaching
max/min 1.97 at step 20. The full run shows that drift reversing. Fractions
below are averaged over the six MoE layers and over all four microbatches of the
logged update:

| Step | Assignment fractions | max/min | aux |
| --- | --- | --- | --- |
| 0 | 0.277, 0.234, 0.235, 0.254 | 1.18 | 1.0214 |
| 100 | 0.265, 0.314, 0.222, 0.199 | 1.58 | 1.0588 |
| 200 | 0.256, 0.288, 0.225, 0.232 | 1.28 | 1.0276 |
| 600 | 0.257, 0.283, 0.222, 0.238 | 1.27 | 1.0203 |
| 1219 | 0.259, 0.286, 0.218, 0.237 | 1.31 | 1.0220 |

Worst imbalance is around step 100, near the end of the 60-update warmup. From
roughly step 200 the fractions sit in a stable band with max/min between 1.27 and
1.32, and no expert falls below 0.21. Nothing starved and nothing collapsed.

Two caveats on reading that table. The fractions are a layer average, so a layer
skewed one way and a layer skewed the other would partly cancel; per-layer
statistics are collected but only the average is written to the log. And a
balanced router is not by itself evidence of expert specialization, which this
project does not claim to have measured.

`assignment_fraction` is the balance diagnostic. The auxiliary loss is a poor
one: with `L = E · Σ f_e P_e`, near-uniform probabilities `P_e ≈ 1/E` give
`L = E · (1/E) · Σ f_e = 1` for any assignment distribution at all, including
total collapse onto one expert. An untrained router has near-uniform
probabilities, so an auxiliary loss sitting at 1.0 early in training carries
almost no information about load balance. The full run bears this out from the
other side: `aux` moved over a range of about 0.04 across 1,220 updates while the
assignment fractions moved visibly, so it was never the thing to watch.

## Results

Both models trained from random initialization on the same 90,000-story
TinyStories subset, same tokenizer, same 512-token context, same effective batch
of 16,384 tokens, same 1,220 updates, same `NVIDIA A100-SXM4-40GB` in bf16 under
torch 2.11.0+cu128. Perplexity is the exponential of token-weighted mean NLL over
437,248 scored validation targets, language loss only, with the auxiliary term
excluded.

| Model | Body params | Total params | Width | Processed tokens | Val NLL | Perplexity | Train tok/s | Peak mem |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Dense + RoPE | 22,423,296 | 41,721,984 | m=4096 | 19,988,480 | 2.3764 | 10.77 | 102,579 | 4,139 MiB |
| Top-2 MoE + RoPE | 41,306,880 | 60,605,568 | 4×m=2048 | 19,988,480 | 2.3054 | 10.03 | 54,967 | 4,648 MiB |

Tokens/s here is cumulative over the whole run, including the slow first update
and every validation pause, so it is lower than the marginal rates in the
throughput table above. Wall clock was 195 s for the dense run and 364 s for the
MoE run.

The MoE is ahead by 0.071 nats, which is 6.9% lower perplexity. It leads at every
validation point, not just at the end:

| Step | Dense ppl | MoE ppl |
| --- | --- | --- |
| 199 | 30.65 | 28.23 |
| 399 | 18.61 | 17.01 |
| 599 | 14.55 | 13.33 |
| 799 | 12.41 | 11.50 |
| 999 | 11.31 | 10.51 |
| 1199 | 10.79 | 10.06 |
| 1219 | 10.77 | 10.03 |

What this does and does not show. It is a matched-token, matched-data,
matched-hardware comparison, and under those conditions the sparse model wins.
It is not a compute-matched or parameter-matched win: the MoE carries 1.84 times
the body parameters and took 1.87 times the wall clock. Active MLP arithmetic per
token is matched by construction (`dense_width = 2 * expert_width` under Top-2),
which is the protocol stated above, but that is one specific notion of "fair" and
a reader should know which one is being used. Neither curve has flattened at 0.99
passes, so these are the numbers at this budget, not converged numbers.

### Probe runs, not results

Short runs used to size the experiment, kept here because they are what the
configuration decisions were based on. None is long enough to say anything about
model quality, and the 30-update rows are superseded by the full runs above.

| Run | Updates | Processed tokens | Passes | Val NLL | Perplexity |
| --- | --- | --- | --- | --- | --- |
| Debug dense, T=128 | 244 | 999,424 | 4.43 | 5.5436 | 255.59 |
| Experiment dense, T=512 | 30 | 491,520 | 0.024 | 8.1304 | 3,396.07 |
| Experiment MoE, T=512 | 30 | 491,520 | 0.024 | 7.9990 | 2,977.96 |

A uniform predictor over the GPT-2 vocabulary scores 50,257, so the 244-update
debug run at 255.59 shows the model learning. It saw the same 1,000 stories 4.43
times, so it is partly memorization. The two 30-update rows are far too short to
compare against each other.

| Prompt tokens | Replay tokens | Prefill ms | Uncached ms/token | Cached ms/token | Speedup | Peak mem by mode |
| --- | --- | --- | --- | --- | --- | --- |
| CONFIG | CONFIG | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED |

## Limitations

- Packed training lets attention cross story boundaries. Separate-document
  masking is out of scope.
- Validation uses fixed windows with context reset at each window start, so the
  first tokens of a window are predicted with little context. This inflates
  perplexity relative to protocols that carry context across windows. Both
  models use the identical protocol, so the comparison is still fair, but the
  number is not comparable to a published result.
- RoPE here rotates adjacent even/odd feature pairs (the GPT-J / interleaved
  convention). HuggingFace Llama and GPT-NeoX rotate halves `(i, i+Dh/2)`
  instead. Both are valid; a direct tensor comparison against those
  implementations will not match.
- No target perplexity, speedup, or parameter count is promised.

## Attribution

- TinyStories dataset: https://huggingface.co/datasets/roneneldan/TinyStories
- RoFormer / RoPE: https://arxiv.org/abs/2104.09864
- Switch Transformers (balancing loss, originally Top-1; adapted here to Top-2):
  https://arxiv.org/abs/2101.03961
- Mixtral of Experts (token-wise Top-2 selection): https://arxiv.org/abs/2401.04088
- PyTorch SDPA mask/dropout semantics:
  https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
