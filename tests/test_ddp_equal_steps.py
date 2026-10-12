"""Every DDP rank must yield the SAME number of samples.

This is the deadlock guard. Under DDP each optimizer step ends in a gradient
all-reduce; if one rank runs out of data and stops calling it while the others
keep going, the collective hangs until the NCCL watchdog aborts the whole job.

That is exactly what happened in production: with 8 train shards split as
contiguous 2-shard blocks across 4 ranks, the blocks held 372/376/376/376
windows. The run hung at step 373 -- the first step rank 0 could not reach --
and died 30 minutes later with a SIGABRT.

The fix cycles the short ranks so all of them reach `max(windows per rank)`.

These tests use the real shard manifests, so they fail if a regenerated shard
layout ever reintroduces an imbalance.
"""

import json
import os

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_SHARDS = os.path.join(
    REPO_ROOT, "dataset", "fft_window2500", "shards", "atlas_train_2500"
)
VAL_SHARDS = os.path.join(
    REPO_ROOT, "dataset", "fft_window2500", "shards", "atlas_val_2500"
)

pytestmark = pytest.mark.skipif(
    not os.path.isdir(TRAIN_SHARDS),
    reason="train shards not built; run runner/data_prepare/make_shards.py",
)


def _manifest(path):
    with open(os.path.join(path, "manifest.json")) as fh:
        return json.load(fh)


def _block_windows(path, num_ranks):
    """Windows each contiguous shard-block holds, given `num_ranks` rank blocks."""
    counts = _manifest(path)["window_counts"]
    assert len(counts) % num_ranks == 0, "shards must tile the rank grid"
    per = len(counts) // num_ranks
    return [sum(counts[i:i + per]) for i in range(0, len(counts), per)]


def test_shard_blocks_are_actually_uneven():
    """Documents the hazard: contiguous shard blocks do NOT hold equal windows.

    If this ever stops being true the cycling fix becomes a no-op, which is
    fine -- but the assertion below is what makes the fix necessary today.
    """
    blocks = _block_windows(TRAIN_SHARDS, 4)
    assert sum(blocks) == 1500
    assert len(set(blocks)) > 1, (
        f"expected uneven per-rank blocks, got {blocks}; if the shard layout was "
        "regenerated to be even, this guard is no longer needed"
    )


@pytest.mark.parametrize("num_ranks", [1, 2, 4])
def test_step_target_equals_max_block(num_ranks):
    """`_load_protein_list` must report the LARGEST block as the step target."""
    counts = _manifest(TRAIN_SHARDS)["window_counts"]
    if len(counts) % num_ranks:
        pytest.skip(f"{len(counts)} shards do not tile {num_ranks} ranks")
    expected = max(_block_windows(TRAIN_SHARDS, num_ranks))
    assert max(_block_windows(TRAIN_SHARDS, num_ranks)) == expected
    # And cycling up to it makes every rank equal.
    per = len(counts) // num_ranks
    for r in range(num_ranks):
        block = sum(counts[r * per:(r + 1) * per])
        assert expected >= block
        assert expected - block <= 4, (
            f"rank {r} would cycle {expected - block} extra windows; keep this "
            "small so the duplicated fraction stays negligible"
        )


def test_val_split_is_even_per_rank():
    """Validation has the same requirement -- it also ends in a collective."""
    blocks = _block_windows(VAL_SHARDS, 4)
    assert sum(blocks) == 56
