"""Per-layer key/value cache for incremental decoding.

DAY 6 SCOPE. What the cache is for: during generation, step t recomputes the
keys and values for every token 0..t-1 that it already computed at step t-1.
Those tensors do not change, because a key rotated by RoPE at its absolute
position stays correct as the sequence grows (see rope.py). Storing them turns
per-step attention work from O(t^2) into O(t).

Layout. One pair of buffers per layer, each [B, H, max_len, Dh], allocated on
the first append so the dtype and device come from the model rather than being
guessed. `length` is the number of valid positions and is SHARED across layers:
every layer appends the same T new tokens during one forward pass, so the
counter advances once at the end of that pass, not once per layer.

    cache.append(layer=0, k, v)  ->  writes rows [length, length+T)
    cache.append(layer=1, k, v)  ->  writes the same rows, layer 1's buffers
    ...
    cache.advance(T)             ->  length += T, once, by the caller

That split is deliberate. If `append` advanced the counter itself, layer 1 would
write its keys one chunk further along than layer 0 and the cache would be
silently skewed. Every test in tests/test_cache.py that checks a multi-layer
model is really checking that this invariant holds.

"Dynamic" here means the caller never declares the generation length up front;
it means the same thing it means in a production serving stack, where the buffer
is reserved once and a length counter moves. Capacity is bounded by
`max_seq_len` anyway, since the RoPE tables are only that long.
"""

from __future__ import annotations

import torch


class KVCache:
    """Key/value storage for `n_layers` layers, up to `max_len` positions.

    Attributes:
        length: positions currently valid. Starts at 0.
        max_len: capacity. Appending past it raises rather than wrapping.
    """

    def __init__(self, n_layers: int, max_len: int):
        if n_layers < 1:
            raise ValueError(f"n_layers must be positive, got {n_layers}")
        if max_len < 1:
            raise ValueError(f"max_len must be positive, got {max_len}")
        self.n_layers = n_layers
        self.max_len = max_len
        self.keys: list[torch.Tensor | None] = [None] * n_layers
        self.values: list[torch.Tensor | None] = [None] * n_layers
        self.length = 0

    def append(
        self, layer: int, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Store T new positions for one layer; return every valid position.

        k, v are [B, H, T, Dh], already RoPE-rotated for k at absolute positions
        `length .. length+T-1`. The returned views are [B, H, length+T, Dh] and
        are what attention scores against. They are views into the buffer, not
        copies, so they stay valid only until the next append on this layer.
        """
        if not 0 <= layer < self.n_layers:
            raise IndexError(f"layer {layer} out of range for {self.n_layers} layers")
        if k.shape != v.shape:
            raise ValueError(f"k {tuple(k.shape)} and v {tuple(v.shape)} must match")
        if k.dim() != 4:
            raise ValueError(f"expected [B, H, T, Dh], got {tuple(k.shape)}")
        if k.requires_grad or v.requires_grad:
            raise RuntimeError(
                "KVCache stores detached activations; run generation under "
                "torch.no_grad(). Training does not use the cache."
            )

        B, H, T, Dh = k.shape
        end = self.length + T
        if end > self.max_len:
            raise ValueError(
                f"appending {T} positions at length {self.length} exceeds "
                f"capacity {self.max_len}"
            )

        if self.keys[layer] is None:
            self.keys[layer] = torch.empty(
                B, H, self.max_len, Dh, dtype=k.dtype, device=k.device
            )
            self.values[layer] = torch.empty_like(self.keys[layer])
        else:
            buf = self.keys[layer]
            if (B, H, Dh) != (buf.shape[0], buf.shape[1], buf.shape[3]):
                raise ValueError(
                    f"layer {layer} was allocated for [B, H, *, Dh] = "
                    f"{(buf.shape[0], buf.shape[1], buf.shape[3])}, got {(B, H, Dh)}. "
                    "Start a new cache instead of reusing one across batch shapes."
                )
            if k.dtype != buf.dtype:
                raise ValueError(
                    f"layer {layer} holds {buf.dtype} but received {k.dtype}. "
                    "Mixing precisions inside one generation would silently "
                    "round the stored keys."
                )

        self.keys[layer][:, :, self.length : end] = k
        self.values[layer][:, :, self.length : end] = v
        return self.keys[layer][:, :, :end], self.values[layer][:, :, :end]

    def advance(self, n: int) -> None:
        """Commit n newly appended positions. Called once per forward pass."""
        if n < 0:
            raise ValueError(f"cannot advance by {n}")
        if self.length + n > self.max_len:
            raise ValueError(
                f"advancing to {self.length + n} exceeds capacity {self.max_len}"
            )
        self.length += n

    def reset(self) -> None:
        """Forget the contents, keep the allocation."""
        self.length = 0

    def memory_bytes(self) -> int:
        """Bytes reserved, which is capacity and not the used prefix.

        Reserved rather than used is the honest number for a benchmark: the
        allocation is what the GPU is holding either way.
        """
        total = 0
        for buf in self.keys + self.values:
            if buf is not None:
                total += buf.numel() * buf.element_size()
        return total

    def __repr__(self) -> str:
        allocated = sum(buf is not None for buf in self.keys)
        return (
            f"KVCache(layers={self.n_layers}, length={self.length}/{self.max_len}, "
            f"allocated_layers={allocated}, bytes={self.memory_bytes()})"
        )
