#!/usr/bin/env python
"""Extract ESM embeddings from the existing pth into an assemblable shim dir.

Why this is safe: an ESM-2 embedding is a function of the amino-acid SEQUENCE
only. It is completely independent of trajectory processing, so the embeddings
already stored in the training pth are equally valid for a freshly assembled
dataset built from PBC-corrected trajectories. Recomputing them would cost hours
of GPU for a byte-identical result.

(Only `esm2_3b_embed`, 2560-d, exists under data/esm_cache; the config and paper
need 650M/1280-d, so that cache cannot be used.)

Writes `<structure_id>.pt` per protein, which is what
`assemble_training_dataset.load_esm` globs for.

Usage:
    python runner/data_prepare/extract_esm_from_pth.py \
        --pth dataset/fft_window2500/atlas_10_features_fouriered_list_train.pth \
        --out /data_lmj/database/Dynamics/data/esm_cache/esm2_650m_embed_frompth
"""

import argparse
import os
import sys
import time

import numpy as np
import torch


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pth", action="append", required=True,
                   help="Source .pth (repeatable). Any split works; ESM is per sequence.")
    p.add_argument("--out", required=True, help="Output shim directory")
    p.add_argument("--esm-dim", type=int, default=1280)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    seen, written, skipped, bad = {}, 0, 0, 0
    t0 = time.time()
    for path in args.pth:
        print(f"reading {path}", flush=True)
        data = torch.load(path, weights_only=False, map_location="cpu")
        for prot in data:
            tid = prot["id"]                       # e.g. "6h49_A_R1"
            struct = tid.rsplit("_", 1)[0].lower()  # e.g. "6h49_a"
            emb = prot["esm"]
            # The pth stores ESM as a numpy array (the dataset yields
            # `torch.tensor(esm)` at __iter__ time), so accept both.
            if not (torch.is_tensor(emb) or isinstance(emb, np.ndarray)):
                bad += 1
                continue
            emb = torch.as_tensor(emb)
            if emb.dim() != 2:
                bad += 1
                continue
            if emb.shape[1] != args.esm_dim:
                print(f"  ! {tid}: dim {emb.shape[1]} != {args.esm_dim}")
                bad += 1
                continue

            # Guard against two different sequences sharing a structure id.
            prev = seen.get(struct)
            if prev is not None:
                if prev.shape == emb.shape and torch.allclose(prev, emb, atol=1e-5):
                    skipped += 1
                    continue
                print(f"  ! {struct}: conflicting embeddings "
                      f"{tuple(prev.shape)} vs {tuple(emb.shape)}; keeping first")
                skipped += 1
                continue
            seen[struct] = emb

            dest = os.path.join(args.out, f"{struct}.pt")
            if os.path.exists(dest) and not args.overwrite:
                skipped += 1
                continue
            torch.save(emb.float().contiguous(), dest)
            written += 1

    print(f"\nwrote {written}  skipped(existing/dup) {skipped}  bad {bad}")
    print(f"unique structures: {len(seen)}  in {time.time() - t0:.1f}s")
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
