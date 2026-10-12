"""Add per-shard WINDOW counts to an existing shard manifest.

`make_shards.py` records `counts` = proteins per shard, but the DDP step-count
balancing needs WINDOWS per shard: proteins differ in how many windows they
hold, so equal proteins does not mean equal steps. (This was a real bug: the
first balancing attempt used protein counts and produced a wrong step target.)

Rewriting the 28 GB of shard `.pth` files just to record this would be wasteful,
so this loads each shard once, counts its windows, and rewrites only
`manifest.json`.

Usage:
    python runner/data_prepare/count_shard_windows.py \
        --shards dataset/fft_window2500/shards/atlas_train_2500
"""

import argparse
import json
import os
import sys
import time

import torch


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--shards", required=True, help="Shard directory with manifest.json")
    p.add_argument("--window-key", default="x_fft_windows")
    return p.parse_args()


def main():
    args = parse_args()
    mpath = os.path.join(args.shards, "manifest.json")
    with open(mpath) as fh:
        manifest = json.load(fh)

    if manifest.get("window_counts"):
        print(f"manifest already has window_counts: {manifest['window_counts']}")
        return 0

    t0 = time.time()
    counts = []
    for name in manifest["shards"]:
        chunk = torch.load(os.path.join(args.shards, name), weights_only=False)
        windows = sum(
            len(d["positions"].get(args.window_key) or []) for d in chunk
        )
        counts.append(windows)
        print(f"  {name}: {len(chunk)} proteins, {windows} windows", flush=True)

    manifest["window_counts"] = counts
    manifest["num_windows"] = sum(counts)
    with open(mpath, "w") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"\ntotal windows: {sum(counts)} (in {time.time() - t0:.1f}s)")
    print(f"updated {mpath}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
