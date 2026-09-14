# story_moe — decoder-only transformer with RoPE, sparse Top-2 MoE, and KV cache

A next-token language model trained from random initialization on a fixed
TinyStories subset. It continues short story prompts; it is not an
instruction-tuned assistant.

**Status: Day 4 of 7.** Data pipeline, causal attention, RoPE, the loss,
validation and checkpoint/resume are verified. The sparse Top-2 MoE is written
and awaiting its gate. No KV cache and no completed training runs yet. Every
results table below reads NOT MEASURED and will stay that way until a run
actually produces the number.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate           # Windows
pip install -e ".[dev]"
```

PyTorch is installed from PyPI by the line above. On Colab, torch is already
present — install the rest with `pip install transformers datasets pyyaml`.

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
requested precision is unsupported — Colab reassigns accelerators between
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
cell once. Keep the repo private — nothing here is secret, but an `HF_TOKEN` or a
stray checkpoint is much easier to leak than to un-leak.

Three things that matter there and nowhere else:

- **Never `pip install torch` in Colab.** Its torch is built against the driver on
  that VM; the PyPI wheel can replace it with one that does not match and the GPU
  then fails to initialize. The notebook uses
  `pip install -e . --no-deps` so `pyproject.toml` cannot pull torch in.
- **The VM's disk is deleted when the session ends**, and sessions end on their
  own. Checkpoints and the token cache go to mounted Drive via `--out-dir` and
  `--data-cache`; `--resume` picks the run back up from there.
- **Pin the accelerator.** The notebook's first cell prints the GPU name and
  whether bf16 is supported, then tells you what to pass for
  `--require-gpu-name` and `--precision`. Use the same pair for the dense and the
  MoE run, and record it — otherwise the two runs are not comparable and the
  tokens/s figures mean nothing next to each other.

Both configs share one data cache, so step 1 is run once and serves the dense
and MoE runs alike.

Measured cache for the debug config (1,000 train / 200 validation stories,
block_size 128, stride 128, GPT-2 tokenizer):

| Split | Blocks | Unique stream tokens | Discarded | Scored targets |
| --- | --- | --- | --- | --- |
| train | 1,763 | 225,790 | 125 | 225,664 |
| validation | 339 | 43,427 | 34 | 43,392 |

That is roughly 226 tokens per story, which is what sets the epoch count for any
given processed-token budget.

## Data contract

Fixed for both models. Changing any of it invalidates the comparison.

| Property | Value |
| --- | --- |
| Dataset | `roneneldan/TinyStories`, official train / validation splits |
| Tokenizer | pretrained GPT-2 (`AutoTokenizer.from_pretrained("gpt2")`), no custom BPE |
| Special tokens | `add_special_tokens=False`, then exactly one EOS appended per story |
| Padding | none — there is no pad token and no padded batch |
| Packing | stories concatenated per split, cut into windows of `block_size + 1` |
| Stride | `block_size` (default) — consecutive windows share one boundary token |
| Shift | `inputs = window[:-1]`, `targets = window[1:]` |
| Remainder | discarded, never padded, never scored; count recorded in `data/cache/<split>.json` |
| Story boundaries | attention **does** cross them inside a block; EOS does not reset attention |

Subset selection is `random.Random(seed).sample(...)`, and the selected index
list, its hash, the split size and the tokenizer fingerprint are all written to
`data/cache/<split>.json` so a later run can prove it used the same data.

## Architecture (target, not yet built)

Pre-norm decoder blocks, multi-head causal self-attention, RoPE on queries and
keys only, LayerNorm, GELU experts, tied input/output embeddings, bias-free
linear projections. The feed-forward sublayer is either a dense MLP or four
experts with Top-2 routing and a renormalized softmax gate.

The MoE computes only the experts a token selected: each expert's assigned rows
are gathered, run through one MLP call, weighted, and scatter-added back. Over
N tokens that is k*N token-expert evaluations, not E*N. A dense all-experts
version exists solely as a test oracle.

The balancing objective is a Top-k adaptation of the Switch loss:
`f_e` is expert e's share of the k*N assignments, `P_e` the mean full softmax
probability, and `L = E * sum_e f_e * P_e`. **Under uniform routing this equals
1, not 0** — an auxiliary loss near 1 is healthy and should not be driven down.
Gradients reach the router through `P_e` (the probabilities before top-k
truncation) and through the renormalized selected weights, which are never
detached. The top-k indices themselves are discrete and carry no gradient.

Position comes from RoPE alone — there is no learned positional embedding table.
Attention masking uses `j <= past_len + i` in its general form from the start, so
enabling the KV cache changes the arguments and not the masking logic.

### Parameter accounting

With the GPT-2 vocabulary (50,257) the tied embedding dominates a small model,
so **body (non-embedding) parameters are the number to compare**, not the total.
RoPE contributes no parameters, so "embedding" is exactly the token embedding.

The debug dense row is confirmed against an instantiated model: `body = 394,496`
= 131,072 attention + 262,144 MLP + 1,280 LayerNorm, and `embedding = 6,432,896`
= 50,257 × 128. The other rows are computed from config shapes.

| Config | Embedding | Attention | MLP / experts | Total | Embedding share |
| --- | --- | --- | --- | --- | --- |
| Debug dense (2L, D=128, m=512) | 6.43M | 0.13M | 0.26M | 6.83M | 94% |
| Debug MoE (2L, D=128, 4×m=256) | 6.43M | 0.13M | 0.52M | 7.09M | 91% |
| **Experiment dense** (6L, D=384, m=4096) | 19.30M | 3.54M | 18.87M | **41.72M** | 46% |
| **Experiment MoE** (6L, D=384, 4×m=2048) | 19.30M | 3.54M | 37.75M | **60.60M** | 32% |

The debug tier is a plumbing demonstration, and the measured A100 probe confirms
why: 94% of its forward FLOPs are the embedding and output projection, so the
feed-forward change moves neither quality nor throughput much, and the two
models differ by only 3.8% in total parameters. The main experiment therefore
runs at 6 layers / d_model 384 — `configs/train_dense.yaml` and
`configs/train_moe.yaml`, 41.72M vs 60.60M parameters, bodies 22.42M vs 41.30M.

### Measured throughput

| Setup | GPU | Precision | Tokens/s |
| --- | --- | --- | --- |
| Debug tier, microbatch 4, T=128 | A100-SXM4-40GB | bf16 | ~55,800 (steady state) |
| Debug tier, microbatch 4, T=128 | CPU (Windows) | fp32 | ~2,200–2,600 |

Marginal rate over the last 40 updates of the probe was flat within 1%. Colab
runs torch 2.11.0+cu128; the Windows venv runs 2.14.0 — any number quoted must
say which produced it.

## Comparison protocol

Same tokenizer, data subset, validation protocol, attention backbone, context,
precision, effective batch, and processed-token budget. Both models trained from
scratch. `tests/test_config.py` enforces that the two YAML files differ only in
the feed-forward fields.

Default matching is **approximate active-MLP compute**: with expert width `m`
and Top-2, dense width is `2*m`. This matches selected MLP arithmetic per token;
it is not equal runtime and not equal full-model FLOPs. The MoE still stores four
experts. Router and dispatch overhead are extra.

Because Colab reassigns accelerators between sessions, `train.device`,
`train.precision` and `train.require_gpu_name` are asserted at startup. A dense
run on an A100 in bf16 and an MoE run on a T4 in fp16 are not comparable.

## Results

| Model | Body params | Total params | Width | Processed tokens | Val NLL | Perplexity | Train tok/s | Peak mem |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Dense + RoPE | NOT MEASURED | NOT MEASURED | CONFIG | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED |
| Top-2 MoE + RoPE | NOT MEASURED | NOT MEASURED | CONFIG | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED |

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
