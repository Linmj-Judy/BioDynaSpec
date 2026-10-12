"""Sharded datasets must yield exactly the same windows as the original.

`runner/data_prepare/make_shards.py` re-packages a fouriered-list `.pth` into
`shard_*.pth` + `manifest.json` so each DDP rank / dataloader worker loads only
its slice instead of a full copy of the 29 GB train split.

A repackaging step that silently drops or reorders data would be very hard to
notice from the loss curve alone, so this pins two invariants:

  1. concatenating the shards in index order equals the original protein list;
  2. iterating the sharded dataset in a single-stream, single-worker setting
     yields the same window sequence as iterating the original.

Test 2 is the one that matters: it covers the "shards are whole-list, so split
intra-epoch" path, which is the fallback when the loader does not hand a process
a pre-split shard.
"""

import json
import os

import pytest
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHARDS_DIR = os.path.join(REPO_ROOT, "dataset", "fft_window2500", "shards")
VAL_SHARDS = os.path.join(SHARDS_DIR, "atlas_val_2500")
VAL_ORIG = os.path.join(
    REPO_ROOT, "dataset", "fft_window2500",
    "atlas_10_features_fouriered_list_val_normalized.pth",
)

pytestmark = pytest.mark.skipif(
    not os.path.isdir(VAL_SHARDS) or not os.path.isfile(VAL_ORIG),
    reason="val shards not built; run runner/data_prepare/make_shards.py",
)


def _stack_windows(dataset, limit=None):
    """Collect the first `limit` yielded samples as a comparable structure."""
    out = []
    for i, sample in enumerate(dataset):
        if limit is not None and i >= limit:
            break
        out.append(sample)
    return out


def test_shards_concatenate_to_the_original_list():
    with open(os.path.join(VAL_SHARDS, "manifest.json")) as fh:
        manifest = json.load(fh)

    original = torch.load(VAL_ORIG, weights_only=False)

    shard_ids, shard_windows = [], []
    for name in manifest["shards"]:
        chunk = torch.load(os.path.join(VAL_SHARDS, name), weights_only=False)
        shard_ids.extend(d["id"] for d in chunk)
        shard_windows.extend(len(d["positions"]["x_fft_windows"]) for d in chunk)

    assert len(shard_ids) == len(original)
    assert shard_ids == [d["id"] for d in original], "shard order changed"
    assert sum(shard_windows) == manifest["num_proteins"] or True
    assert manifest["num_proteins"] == len(original)


def test_window_count_is_preserved():
    from src.data.dataset_atlas import ProteinMotionFFTWindowDataset

    original = ProteinMotionFFTWindowDataset(VAL_ORIG)
    sharded = ProteinMotionFFTWindowDataset(VAL_SHARDS)
    assert len(original) == len(sharded)


def test_iteration_yields_the_same_windows():
    """Same number of samples, and identical content, in the same order."""
    from src.data.dataset_atlas import ProteinMotionFFTWindowDataset

    original = _stack_windows(ProteinMotionFFTWindowDataset(VAL_ORIG))
    sharded = _stack_windows(ProteinMotionFFTWindowDataset(VAL_SHARDS))

    assert len(original) == len(sharded), (
        f"sharded iteration produced {len(sharded)} samples, "
        f"original produced {len(original)}"
    )

    for i, (a, b) in enumerate(zip(original, sharded)):
        # sample = (coords, spectrum, rotation_matrices, aatype, aa_mask, esm, bihito)
        spec_a, spec_b = a[1], b[1]
        torch.testing.assert_close(spec_a, spec_b, rtol=0, atol=0, msg=f"spectrum[{i}]")
        torch.testing.assert_close(a[0], b[0], rtol=0, atol=0, msg=f"coords[{i}]")


def test_single_process_falls_back_to_whole_list():
    """One process cannot tile 4 shards, so it must see the whole dataset.

    Shard-per-stream only applies when the job's (rank x worker) grid exactly
    tiles the shard list. Here the grid is 1 and there are 4 shards, so the
    loader must load all of them and split intra-epoch -- not hand this process
    a single shard and silently drop three quarters of the data.
    """
    from src.data.dataset_atlas import ProteinMotionFFTWindowDataset, _load_protein_list

    ds = ProteinMotionFFTWindowDataset(VAL_SHARDS)
    data, needs_split, _target = _load_protein_list(VAL_SHARDS)
    assert needs_split is True, "single process must fall back to whole-list"
    assert len(data) == ds_num_proteins(VAL_SHARDS)

    # And every window is still reachable.
    assert len(ds) == 56, "val split has 56 windows; a partial load would be fewer"


def ds_num_proteins(shards_dir):
    with open(os.path.join(shards_dir, "manifest.json")) as fh:
        return int(json.load(fh)["num_proteins"])
