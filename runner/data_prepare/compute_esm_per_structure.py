#!/usr/bin/env python
"""Compute ESM-2 650M embeddings for EVERY structure, one file each.

Why not just use `src/data/basic/compute_esm.py`
------------------------------------------------
That script groups the sequence table by unique `seq` and keeps only the FIRST
`pdb_entity_id` as the label:

    df_seq["seq_label"] = df_seq["pdb_entity_id"].apply(lambda x: x.split(",")[0])

Among the 1389 ATLAS structures there are only 1378 distinct sequences -- 11
sequences are shared by 2+ structures -- so the stock path would silently leave
11 structures with no embedding file, and `assemble_training_dataset.load_esm`
would then drop them from the dataset.

This driver reuses the same tested inference code (`load_esm_model` +
`compute_ESM_embeddings`) but builds the label list from the structure list, so
every structure gets its own file. Duplicated sequences are computed once per
sequence and the result is written under each structure that shares it, which is
exactly correct: the embedding is a function of the sequence alone.

Output layout matches what `load_esm` globs and what the existing caches use:
    <out>/<id[1:3]>/<pdb_id>_<entity_id>.pt      holding a (L, 1280) tensor

Usage:
    python runner/data_prepare/compute_esm_per_structure.py \
        --pdb_seq_csv /data_lmj/database/Dynamics/data/sequences/pdb_seq.csv \
        --structure_ids_from /data_lmj/database/Dynamics/data/trajectories/atlas \
        --out /data_lmj/database/Dynamics/data/esm_cache/esm2_650m_embed \
        --model_root /data/L202500336/Protenix_1 \
        --gpu 0
"""

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.basic.compute_esm import (  # noqa: E402
    compute_ESM_embeddings,
    load_esm_model,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pdb_seq_csv", required=True)
    p.add_argument("--structure_ids_from", required=True,
                   help="Dir whose entry names are <pdb>_<chain>[_R{1,2,3}]")
    p.add_argument("--out", required=True)
    p.add_argument("--model_name", default="esm2-650m")
    p.add_argument("--model_root", default="/data/L202500336/Protenix_1",
                   help="Dir containing release_model/esm2_t33_650M_UR50D.pt")
    p.add_argument("--gpu", type=int, default=None)
    p.add_argument("--limit", type=int, default=0, help="Debug: only N structures")
    return p.parse_args()


def main():
    args = parse_args()

    # Structures actually present in the trajectory dir.
    needed = set()
    for d in os.listdir(args.structure_ids_from):
        if os.path.isdir(os.path.join(args.structure_ids_from, d)):
            needed.add(d.rsplit("_", 1)[0].lower())
    print(f"structures present: {len(needed)}")

    df = pd.read_csv(args.pdb_seq_csv)
    df = df[df["mol_type"] == "protein"].copy()
    df["sid"] = df["pdb_id"].astype(str).str.lower()
    df = df[df["sid"].isin(needed)]
    # One row per structure; if a structure somehow has several entities, the
    # assembler resolves by residue count, so keep the first.
    df = df.drop_duplicates(subset=["sid"], keep="first")

    missing = needed - set(df["sid"])
    if missing:
        print(f"!! {len(missing)} structures have no sequence row: {sorted(missing)[:5]}")
    if args.limit:
        df = df.head(args.limit)
    print(f"structures to embed: {len(df)}")

    device = f"cuda:{args.gpu}" if args.gpu is not None else None
    model, alphabet = load_esm_model(
        args.model_name, local_esm_model_dir=args.model_root,
        device_type="cuda" if torch.cuda.is_available() else "cpu", device=device,
    )

    # Compute once per unique sequence, then write under every structure id.
    seq_to_sids = df.groupby("seq")["sid"].apply(list).to_dict()
    uniq = list(seq_to_sids.items())
    print(f"unique sequences: {len(uniq)}  "
          f"(will write {sum(len(v) for _, v in uniq)} files)")

    # `compute_ESM_embeddings` always writes `<save_dir>/<label>.pt` and SKIPS any
    # label whose file already exists (`_check_files_exist` -> `continue`),
    # returning an empty dict in that case. A fixed scratch dir therefore breaks
    # the second run: the stale `seq00001.pt` makes it return nothing and the
    # caller hits a KeyError. Use a scratch dir unique to this process.
    scratch = os.path.join(args.out, f"_scratch_{os.getpid()}_{int(time.time())}")
    os.makedirs(scratch, exist_ok=True)

    t0 = time.time()
    written = skipped = 0
    for i, (seq, sids) in enumerate(uniq, 1):
        # Skip only if EVERY target already exists.
        todo = []
        for sid in sids:
            row = df[df["sid"] == sid].iloc[0]
            dest = os.path.join(args.out, sid[1:3], f"{sid}_{row['entity_id']}.pt")
            if not os.path.exists(dest):
                todo.append(dest)
        if not todo:
            skipped += len(sids)
            continue

        label = f"seq{i:05d}"
        embs = compute_ESM_embeddings(
            args.model_name, model, alphabet, [label], [seq],
            save_dir=scratch, truncation_seq_length=4094, toks_per_batch=16384,
            device=device,
        )
        emb = embs[label]

        for dest in todo:
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            torch.save(emb, dest)
            written += 1

        if i % 100 == 0 or i == len(uniq):
            el = time.time() - t0
            print(f"  {i}/{len(uniq)} seqs  written={written}  {el:.0f}s "
                  f"({el / i * 1000:.0f} ms/seq)", flush=True)

    import shutil
    shutil.rmtree(scratch, ignore_errors=True)

    print(f"\nwrote {written} files, skipped {skipped} existing, in {time.time() - t0:.1f}s")
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
