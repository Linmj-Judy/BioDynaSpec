"""Split a fouriered-list dataset into per-rank shards.

Why: `ProteinMotionFFTWindowDataset.__init__` does a full
`torch.load(data_path)`, so every dataloader worker (and every DDP rank) holds a
complete copy of the split. At 29 GB for the train split that is the memory
problem: `num_workers > 0` multiplies it, and DDP multiplies it again per rank.

Sharding fixes both: rank r / worker w loads only its slice, so total resident
memory stays ~one full copy across the whole job instead of N copies.

This is a re-packaging step only. The element dictionaries, their key order, and
the per-protein window lists are copied through untouched, so a sharded dataset
is read by `ProteinMotionFFTWindowDataset` exactly as the original was. Nothing
is re-encoded or re-normalised.

Usage:
    python runner/data_prepare/make_shards.py \
        --src   dataset/fft_window2500/atlas_10_features_fouriered_list_train_normalized.pth \
        --out-dir dataset/fft_window2500/shards/atlas_train_2500 \
        --num-shards 8

The output directory gets `manifest.json` plus `shard_000.pth ... shard_NNN.pth`.
Concatenating the shards in index order reproduces the original protein order.
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
    p.add_argument("--src", required=True, help="Input .pth (a list of protein dicts)")
    p.add_argument("--out-dir", required=True, help="Directory for shard_*.pth + manifest.json")
    p.add_argument("--num-shards", type=int, default=8,
                   help="Number of shards. Use a multiple of (ranks * workers) so "
                        "every worker gets a disjoint slice.")
    p.add_argument("--num-workers", type=int, default=0,
                   help="Recorded in the manifest for the loader to read back.")
    p.add_argument("--overwrite", action="store_true",
                   help="Rebuild shards even if the manifest already exists")
    return p.parse_args()


def main():
    args = parse_args()
    manifest_path = os.path.join(args.out_dir, "manifest.json")

    if os.path.exists(manifest_path) and not args.overwrite:
        with open(manifest_path) as fh:
            m = json.load(fh)
        print(f"Shards already exist ({m['num_shards']} shards, "
              f"{m['num_proteins']} proteins); pass --overwrite to rebuild.")
        return 0

    if args.num_shards < 1:
        raise SystemExit("--num-shards must be >= 1")

    os.makedirs(args.out_dir, exist_ok=True)

    t0 = time.time()
    print(f"loading {args.src} ...", flush=True)
    data = torch.load(args.src, weights_only=False)
    print(f"  {len(data)} proteins in {time.time() - t0:.1f}s", flush=True)

    # Deterministic contiguous split: concatenating shards in order reproduces
    # the original order, so downstream window indices stay stable.
    n = len(data)
    num_shards = min(args.num_shards, n)
    base, extra = divmod(n, num_shards)
    bounds, start = [], 0
    for i in range(num_shards):
        size = base + (1 if i < extra else 0)
        bounds.append((start, start + size))
        start += size

    shard_files, counts = [], []
    for i, (lo, hi) in enumerate(bounds):
        chunk = data[lo:hi]
        name = f"shard_{i:03d}.pth"
        path = os.path.join(args.out_dir, name)
        torch.save(chunk, path)
        shard_files.append(name)
        counts.append(len(chunk))
        print(f"  wrote {name}: proteins [{lo}:{hi}] = {len(chunk)}", flush=True)

    manifest = {
        "num_shards": num_shards,
        "num_proteins": n,
        "num_workers": args.num_workers,
        "counts": counts,
        "shards": shard_files,
        "source": os.path.abspath(args.src),
        "window_key": "x_fft_windows",
    }
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"\ndone in {time.time() - t0:.1f}s -> {manifest_path}", flush=True)
    print(f"total on disk: {sum(counts)} proteins across {num_shards} shards", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
