"""Day 1 gate: the data contract.

These tests use hand-written token lists, not the real dataset, so they run
without `transformers`, `datasets` or a network connection. The one thing they
check is input/target alignment, which ruins a language model silently when it
is wrong.
"""

import pytest

from story_moe.data import block_count, pack_blocks, split_inputs_targets

EOS = 99


def test_eos_appended_once_per_story():
    stories = [[1, 2, 3], [4, 5]]
    blocks, _ = pack_blocks(stories, EOS, block_size=6, stride=6)
    # stream is [1,2,3,EOS,4,5,EOS] -> exactly one window of 7
    assert blocks == [[1, 2, 3, EOS, 4, 5, EOS]]
    assert blocks[0].count(EOS) == 2


def test_every_target_is_the_next_token():
    stories = [list(range(1, 40))]
    blocks, _ = pack_blocks(stories, EOS, block_size=8, stride=8)
    for block in blocks:
        x, y = split_inputs_targets(block)
        assert len(x) == len(y) == 8
        # The shift, stated the only way that matters, is that y[i] follows x[i].
        for i in range(len(x)):
            assert y[i] == block[i + 1]
        # And no double shift, so y[:-1] must equal x[1:].
        assert y[:-1] == x[1:]


def test_no_token_is_scored_twice_with_default_stride():
    """With stride == block_size, every stream position after the first is a target once."""
    stories = [list(range(1, 33))]
    T = 8
    blocks, _ = pack_blocks(stories, EOS, block_size=T, stride=T)
    targets_seen = []
    for b_i, block in enumerate(blocks):
        _, y = split_inputs_targets(block)
        # absolute stream positions of this block's targets
        targets_seen.extend(range(b_i * T + 1, b_i * T + 1 + T))
    assert len(targets_seen) == len(set(targets_seen)), "a token is scored more than once"
    assert targets_seen == list(range(1, len(targets_seen) + 1))


def test_remainder_is_discarded_not_padded():
    stories = [list(range(1, 20))]  # stream length 20 after EOS
    T = 8
    blocks, discarded = pack_blocks(stories, EOS, block_size=T, stride=T)
    # windows of 9 every 8, starting at 0 and 8 -> covers 0..16, so 3 tokens are dropped
    assert len(blocks) == 2
    assert discarded == 3
    assert all(len(b) == T + 1 for b in blocks)
    # nothing invented, no padding value appears
    flat = [t for b in blocks for t in b]
    assert all(t != 0 for t in flat)


def test_short_stream_yields_no_blocks():
    blocks, discarded = pack_blocks([[1, 2]], EOS, block_size=8, stride=8)
    assert blocks == []
    assert discarded == 3


def test_block_count_matches_pack_blocks():
    for stream_len, T, stride in [(100, 8, 8), (100, 8, 4), (9, 8, 8), (8, 8, 8), (37, 12, 12)]:
        stories = [list(range(stream_len - 1))]  # +1 EOS -> stream_len
        blocks, _ = pack_blocks(stories, EOS, T, stride)
        assert len(blocks) == block_count(stream_len, T, stride)


@pytest.mark.parametrize("bad", [dict(block_size=0, stride=4), dict(block_size=4, stride=0)])
def test_invalid_arguments_rejected(bad):
    with pytest.raises(ValueError):
        pack_blocks([[1, 2, 3]], EOS, **bad)
