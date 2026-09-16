"""TinyStories tokenization, packing, and batching.

Layout of this module, in dependency order:

  1. pack_blocks / block_count. Pure Python, no torch, no HuggingFace. This is
     the part that decides input/target alignment, so it stays dependency-free
     and is unit-tested on hand-written token lists.
  2. load_tokenizer / select_stories / prepare_split. Needs `transformers` and
     `datasets`. Writes a uint16 cache so repeated runs do not retokenize.
  3. Batcher. Needs torch. Turns cached blocks into [B, T] input/target pairs.

Data contract, fixed for the dense run and the MoE run alike:
  - Each story is tokenized with add_special_tokens=False, then exactly one EOS
    is appended as the story separator.
  - Stories in a split are concatenated into one stream, then cut into windows of
    block_size + 1 tokens taken every `stride` tokens.
  - inputs = window[:-1], targets = window[1:], so target[i] is the token that
    follows input[i].
  - The trailing remainder that cannot fill a whole window is discarded. It is
    never padded and never scored. The discarded count goes into meta.json.
  - Attention is allowed to cross story boundaries inside a block. EOS does not
    reset attention. It is only a token the model can learn to predict.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# 1. Pure-Python packing core
# ---------------------------------------------------------------------------


def block_count(stream_len: int, block_size: int, stride: int) -> int:
    """How many (block_size + 1)-token windows fit in a stream of this length."""
    window = block_size + 1
    if stream_len < window:
        return 0
    return 1 + (stream_len - window) // stride


def pack_blocks(
    token_lists: Iterable[Sequence[int]],
    eos_id: int,
    block_size: int,
    stride: int,
) -> tuple[list[list[int]], int]:
    """Append EOS to each story, concatenate, and cut into overlapping windows.

    Returns (blocks, discarded_tokens) where every block has length block_size+1
    and discarded_tokens counts stream positions that never appear in any block.
    """
    if block_size < 1:
        raise ValueError("block_size must be >= 1")
    if stride < 1:
        raise ValueError("stride must be >= 1")

    stream: list[int] = []
    for toks in token_lists:
        stream.extend(toks)
        stream.append(eos_id)

    window = block_size + 1
    n = block_count(len(stream), block_size, stride)
    blocks = [stream[i * stride : i * stride + window] for i in range(n)]

    covered = 0 if n == 0 else (n - 1) * stride + window
    return blocks, len(stream) - covered


def split_inputs_targets(block: Sequence[int]) -> tuple[list[int], list[int]]:
    """The one place the shift happens. Everything else must call this."""
    return list(block[:-1]), list(block[1:])


# ---------------------------------------------------------------------------
# 2. Tokenizer + dataset -> cached uint16 blocks
# ---------------------------------------------------------------------------


def load_tokenizer(name: str = "gpt2"):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name)
    if tok.eos_token_id is None:
        raise ValueError(f"tokenizer {name!r} has no EOS token; it is required as a separator")
    return tok


def tokenizer_fingerprint(tok) -> dict[str, Any]:
    """Recorded in meta.json so a later run can prove it used the same tokenizer."""
    return {
        "name_or_path": getattr(tok, "name_or_path", None),
        "vocab_size_len": len(tok),
        "vocab_size_attr": getattr(tok, "vocab_size", None),
        "eos_token_id": tok.eos_token_id,
        "eos_token": tok.eos_token,
        "pad_token_id": tok.pad_token_id,
        "class": type(tok).__name__,
    }


def select_stories(dataset: str, split: str, n: int, seed: int) -> tuple[list[str], dict[str, Any]]:
    """Deterministically pick n stories from a split. Returns (texts, provenance)."""
    from datasets import load_dataset

    ds = load_dataset(dataset, split=split)
    total = len(ds)
    if n > total:
        raise ValueError(f"asked for {n} stories but split {split!r} has {total}")

    rng = random.Random(seed)
    idx = sorted(rng.sample(range(total), n))
    texts = [ds[i]["text"] for i in idx]

    provenance = {
        "dataset": dataset,
        "split": split,
        "split_size": total,
        "n_selected": n,
        "seed": seed,
        "selection": "random.Random(seed).sample(range(split_size), n), sorted",
        "index_sha256": hashlib.sha256(
            ",".join(map(str, idx)).encode("utf-8")
        ).hexdigest()[:16],
        "first_indices": idx[:10],
    }
    return texts, provenance


def prepare_split(
    cfg,
    split: str,
    n_stories: int,
    cache_dir: Path,
    tok=None,
) -> dict[str, Any]:
    """Tokenize, pack, and cache one split. Returns its meta dict.

    Cache layout:  <cache_dir>/<split>.bin   uint16, shape (n_blocks, block_size+1)
                   <cache_dir>/<split>.json  meta (counts, provenance, fingerprint)
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    bin_path = cache_dir / f"{split}.bin"
    meta_path = cache_dir / f"{split}.json"

    tok = tok or load_tokenizer(cfg.data.tokenizer)
    if len(tok) > np.iinfo(np.uint16).max + 1:
        raise ValueError(f"vocab {len(tok)} does not fit in uint16; change the cache dtype")

    texts, provenance = select_stories(cfg.data.dataset, split, n_stories, cfg.data.seed)
    token_lists = [tok(t, add_special_tokens=False)["input_ids"] for t in texts]

    stride = cfg.data.resolved_stride()
    blocks, discarded = pack_blocks(token_lists, tok.eos_token_id, cfg.data.block_size, stride)
    if not blocks:
        raise ValueError(
            f"{split}: {sum(len(t) for t in token_lists)} tokens is too few for one "
            f"block of {cfg.data.block_size + 1}; raise n_stories or lower block_size"
        )

    arr = np.asarray(blocks, dtype=np.uint16)
    arr.tofile(bin_path)

    unique_tokens = sum(len(t) for t in token_lists) + len(token_lists)  # + one EOS each
    meta = {
        "split": split,
        "n_blocks": int(arr.shape[0]),
        "block_size": cfg.data.block_size,
        "stride": stride,
        "window": cfg.data.block_size + 1,
        "unique_stream_tokens": unique_tokens,
        "discarded_tokens": discarded,
        "scored_targets_per_block": cfg.data.block_size,
        "total_scored_targets": int(arr.shape[0]) * cfg.data.block_size,
        "dtype": "uint16",
        "shape": list(arr.shape),
        "tokenizer": tokenizer_fingerprint(tok),
        "provenance": provenance,
    }
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return meta


def load_cached_split(cache_dir: Path, split: str) -> tuple[np.ndarray, dict[str, Any]]:
    cache_dir = Path(cache_dir)
    meta = json.loads((cache_dir / f"{split}.json").read_text(encoding="utf-8"))
    arr = np.fromfile(cache_dir / f"{split}.bin", dtype=np.uint16).reshape(meta["shape"])
    return arr, meta


def check_cache_matches_config(meta: dict[str, Any], cfg, split: str) -> None:
    """Refuse to train on a cache that does not match the active config.

    Point a 512-token config at the 128-token cache and nothing complains. The
    batcher yields shorter rows, the model takes them, and every processed-token
    count in the logs and the results JSON overcounts by a factor of four.
    Silent bad accounting is worse than a crash.
    """
    expected = {
        "block_size": cfg.data.block_size,
        "stride": cfg.data.resolved_stride(),
    }
    problems = [
        f"{key}: cache has {meta.get(key)!r}, config wants {want!r}"
        for key, want in expected.items()
        if meta.get(key) != want
    ]

    tok = meta.get("tokenizer", {})
    if tok.get("name_or_path") != cfg.data.tokenizer:
        problems.append(
            f"tokenizer: cache has {tok.get('name_or_path')!r}, config wants {cfg.data.tokenizer!r}"
        )
    if cfg.model.vocab_size is not None and tok.get("vocab_size_len") != cfg.model.vocab_size:
        problems.append(
            f"vocab size: cache has {tok.get('vocab_size_len')}, model has {cfg.model.vocab_size}"
        )

    prov = meta.get("provenance", {})
    if prov.get("dataset") != cfg.data.dataset:
        problems.append(f"dataset: cache has {prov.get('dataset')!r}, config wants {cfg.data.dataset!r}")
    if prov.get("seed") != cfg.data.seed:
        problems.append(f"seed: cache has {prov.get('seed')}, config wants {cfg.data.seed}")

    wanted_stories = cfg.data.train_stories if split == "train" else cfg.data.val_stories
    if prov.get("n_selected") != wanted_stories:
        problems.append(
            f"stories: cache has {prov.get('n_selected')}, config wants {wanted_stories}"
        )

    if problems:
        raise ValueError(
            f"cached '{split}' split does not match the config:\n  - "
            + "\n  - ".join(problems)
            + f"\nRe-run: python -m story_moe.data prepare --config <cfg> "
              f"--cache-dir {cfg.data.cache_dir}"
        )


# ---------------------------------------------------------------------------
# 3. Batching (torch)
# ---------------------------------------------------------------------------


class Batcher:
    """Yields (inputs, targets) of shape [B, T], both int64, from cached blocks.

    Sampling is without replacement. A shuffled permutation is consumed in
    order and reshuffled once exhausted, so a coverage figure means what it
    says. Drawing n times with replacement from n blocks touches only
    1 - (1-1/n)^n ~= 63% of them, so a "one pass" budget built on
    `torch.randint` would leave a third of the data unseen while showing other
    blocks twice. With a permutation, "0.98 passes" means 98% of the blocks,
    each exactly once.

    `epochs_seen()` reports the true fractional position in the data.
    """

    def __init__(self, blocks: np.ndarray, batch_size: int, seed: int = 0, shuffle: bool = True):
        import torch

        self.blocks = blocks
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.generator = torch.Generator().manual_seed(seed)
        self.epoch = 0
        self.cursor = 0
        self._perm = self._new_permutation()

    def _new_permutation(self):
        import torch

        n = self.blocks.shape[0]
        if self.shuffle:
            return torch.randperm(n, generator=self.generator)
        return torch.arange(n)

    def next_batch(self, device: str = "cpu"):
        """The next batch_size blocks in permutation order, wrapping at epoch end."""
        import torch

        taken = []
        remaining = self.batch_size
        while remaining > 0:
            available = len(self._perm) - self.cursor
            if available == 0:
                self.epoch += 1
                self._perm = self._new_permutation()
                self.cursor = 0
                continue
            take = min(remaining, available)
            taken.append(self._perm[self.cursor : self.cursor + take])
            self.cursor += take
            remaining -= take

        idx = torch.cat(taken).numpy()
        # int64 first, because uint16 is not a torch dtype.
        window = torch.from_numpy(self.blocks[idx].astype(np.int64))
        return window[:, :-1].to(device), window[:, 1:].to(device)

    def epochs_seen(self) -> float:
        return self.epoch + self.cursor / max(1, len(self._perm))

    def state_dict(self) -> dict[str, Any]:
        return {
            "generator": self.generator.get_state(),
            "perm": self._perm,
            "cursor": self.cursor,
            "epoch": self.epoch,
            "n_blocks": int(self.blocks.shape[0]),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("n_blocks") not in (None, int(self.blocks.shape[0])):
            raise ValueError(
                f"checkpoint sampler covered {state['n_blocks']} blocks but this "
                f"cache has {self.blocks.shape[0]}; the data changed under the run"
            )
        self.generator.set_state(state["generator"].cpu())
        self._perm = state["perm"].cpu()
        self.cursor = int(state["cursor"])
        self.epoch = int(state["epoch"])



# ---------------------------------------------------------------------------
# CLI:  python -m story_moe.data prepare|show --config configs/debug_dense.yaml
# ---------------------------------------------------------------------------


def _cmd_prepare(cfg) -> None:
    tok = load_tokenizer(cfg.data.tokenizer)
    cache = Path(cfg.data.cache_dir)
    for split, n in (("train", cfg.data.train_stories), ("validation", cfg.data.val_stories)):
        meta = prepare_split(cfg, split, n, cache, tok=tok)
        print(
            f"{split:10s} blocks={meta['n_blocks']:<7d} "
            f"unique_stream_tokens={meta['unique_stream_tokens']:<9d} "
            f"discarded={meta['discarded_tokens']:<5d} "
            f"scored_targets={meta['total_scored_targets']}"
        )
    print(f"\ncache written to {cache.resolve()}")


def _cmd_show(cfg) -> None:
    import torch

    tok = load_tokenizer(cfg.data.tokenizer)
    arr, meta = load_cached_split(Path(cfg.data.cache_dir), "train")
    V = len(tok)

    batcher = Batcher(arr, batch_size=2, seed=cfg.data.seed)
    x, y = batcher.next_batch()

    print(f"tokenizer      : {meta['tokenizer']['name_or_path']}  len={V}  eos={tok.eos_token_id}")
    print(f"inputs  shape  : {tuple(x.shape)}  dtype={x.dtype}")
    print(f"targets shape  : {tuple(y.shape)}  dtype={y.dtype}")

    # --- Day 1 gate assertions -------------------------------------------------
    assert x.shape == y.shape == (2, cfg.data.block_size), "batch is not [B, T]"
    assert x.dtype == torch.int64 and y.dtype == torch.int64, "token IDs must be int64"
    assert int(x.min()) >= 0 and int(x.max()) < V, f"input IDs outside [0, {V})"
    assert int(y.min()) >= 0 and int(y.max()) < V, f"target IDs outside [0, {V})"
    # Check the shift directly. target[:, i] must equal input[:, i+1] everywhere.
    assert torch.equal(y[:, :-1], x[:, 1:]), "targets are not the inputs shifted by one"
    print("assertions     : OK (shape, dtype, ID range, single shift)")

    print("\n-- first 12 (input -> target) pairs of row 0 --")
    for i in range(12):
        xi, yi = int(x[0, i]), int(y[0, i])
        print(f"  {i:3d}  {xi:>6d} {tok.decode([xi])!r:<14s} -> {yi:>6d} {tok.decode([yi])!r}")

    print("\n-- round trip of row 0 (first 240 chars) --")
    print(repr(tok.decode(x[0].tolist())[:240]))


def main(argv: list[str] | None = None) -> None:
    import argparse

    from .config import load_config

    p = argparse.ArgumentParser(prog="story_moe.data")
    p.add_argument("command", choices=["prepare", "show"])
    p.add_argument("--config", required=True)
    # Overrides, for Colab and for sizing experiments. Whatever you pass here is
    # recorded in the cache meta.json, so a run can always say what it used.
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--train-stories", type=int, default=None)
    p.add_argument("--val-stories", type=int, default=None)
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    if args.cache_dir:
        cfg.data.cache_dir = args.cache_dir
    if args.train_stories:
        cfg.data.train_stories = args.train_stories
    if args.val_stories:
        cfg.data.val_stories = args.val_stories

    {"prepare": _cmd_prepare, "show": _cmd_show}[args.command](cfg)


if __name__ == "__main__":
    main()
