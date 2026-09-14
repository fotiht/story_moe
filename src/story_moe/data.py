"""TinyStories tokenization, packing, and batching.

Layout of this module, in dependency order:

  1. pack_blocks / block_count  - pure Python, no torch, no HuggingFace.
     This is the part that decides input/target alignment, so it is kept
     dependency-free and is unit-tested on hand-written token lists.
  2. load_tokenizer / select_stories / prepare_split - needs `transformers`
     and `datasets`. Writes a uint16 cache so repeated runs do not retokenize.
  3. Batcher - needs torch. Turns cached blocks into [B, T] input/target pairs.

Data contract (fixed for BOTH the dense and MoE runs):
  - Each story is tokenized with add_special_tokens=False, then exactly one EOS
    is appended as the story separator.
  - Stories in a split are concatenated into one stream, then cut into windows of
    block_size + 1 tokens taken every `stride` tokens.
  - inputs = window[:-1], targets = window[1:]  ->  target[i] is the token that
    follows input[i].
  - The trailing remainder that cannot fill a whole window is DISCARDED. It is
    never padded and never scored. The discarded count is recorded in meta.json.
  - Attention is allowed to cross story boundaries inside a block. EOS does not
    reset attention; it is only a token the model can learn to predict.
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


# ---------------------------------------------------------------------------
# 3. Batching (torch)
# ---------------------------------------------------------------------------


class Batcher:
    """Yields (inputs, targets) of shape [B, T], both int64, from cached blocks."""

    def __init__(self, blocks: np.ndarray, batch_size: int, seed: int = 0, shuffle: bool = True):
        import torch

        self.torch = torch
        self.blocks = blocks
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.generator = torch.Generator().manual_seed(seed)

    def random_batch(self, device: str = "cpu"):
        torch = self.torch
        idx = torch.randint(
            0, self.blocks.shape[0], (self.batch_size,), generator=self.generator
        )
        # int64 first: uint16 is not a torch dtype.
        window = torch.from_numpy(self.blocks[idx.numpy()].astype(np.int64))
        return window[:, :-1].to(device), window[:, 1:].to(device)

    def sequential_batches(self, device: str = "cpu"):
        """Every block exactly once, in order. Used for validation."""
        torch = self.torch
        for start in range(0, self.blocks.shape[0], self.batch_size):
            chunk = self.blocks[start : start + self.batch_size].astype(np.int64)
            window = torch.from_numpy(chunk)
            yield window[:, :-1].to(device), window[:, 1:].to(device)


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
    x, y = batcher.random_batch()

    print(f"tokenizer      : {meta['tokenizer']['name_or_path']}  len={V}  eos={tok.eos_token_id}")
    print(f"inputs  shape  : {tuple(x.shape)}  dtype={x.dtype}")
    print(f"targets shape  : {tuple(y.shape)}  dtype={y.dtype}")

    # --- Day 1 gate assertions -------------------------------------------------
    assert x.shape == y.shape == (2, cfg.data.block_size), "batch is not [B, T]"
    assert x.dtype == torch.int64 and y.dtype == torch.int64, "token IDs must be int64"
    assert int(x.min()) >= 0 and int(x.max()) < V, f"input IDs outside [0, {V})"
    assert int(y.min()) >= 0 and int(y.max()) < V, f"target IDs outside [0, {V})"
    # The shift: target[:, i] must equal input[:, i+1] for every position.
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
