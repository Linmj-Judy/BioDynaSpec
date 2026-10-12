#!/usr/bin/env python
"""Scan `trajectories/atlas/*/**_frames_coords.npy` for PBC jumps.

Why this exists
---------------
The `source_manifest.json` next to each dump claims
`atlas_protein_fit_xtc_pbc_removed_and_aligned`, but the assembled training pth
contains a clear PBC fingerprint (55% of large steps clustered in a narrow
42-52 A band -- a box-length quantum, not thermal motion). So the manifests are
NOT sufficient evidence. This reads the arrays and measures.

Detector
--------
For each trajectory, take a contiguous block of early frames, all atoms, and
compute the per-atom frame-to-frame displacement. A PBC wrap moves a group of
atoms by ~box length in one step, so we report:

  * `med`   -- median per-atom step (thermal scale, ~1 A)
  * `ratio` -- max step / med
  * `frac`  -- fraction of (frame, atom) pairs above 20x med
  * `together` -- most atoms jumping in a single frame (thermal motion does not
                  move a whole group coherently)

A clean (PBC-removed) trajectory should have ratio of order 10-15 and
`together` ~1-2. The dirty pth showed ratio ~50-70 with `together` up to 6.

Reads only the first `--frames` frames via mmap, so a full sweep costs a
fraction of the 1.2 TB of dumps.
"""

import argparse
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--traj-dir", default="/data_lmj/database/Dynamics/data/trajectories/atlas")
    p.add_argument("--frames", type=int, default=1000,
                   help="Contiguous frames from the start to analyse")
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--limit", type=int, default=0, help="Only scan N trajectories (0 = all)")
    p.add_argument("--out", default=None, help="Write per-trajectory JSONL here")
    p.add_argument("--jump-threshold", type=float, default=20.0,
                   help="Multiple of the median step that counts as a jump")
    return p.parse_args()


def find_dump(traj_dir, traj_id):
    """Locate <traj_id>_frames_coords.npy inside the trajectory's subdirectory."""
    base = os.path.join(traj_dir, traj_id)
    if not os.path.isdir(base):
        return None
    for sub in sorted(os.listdir(base)):
        d = os.path.join(base, sub)
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if fn.endswith("_frames_coords.npy"):
                return os.path.join(d, fn)
    return None


def scan_one(args, traj_dir, traj_id):
    try:
        path = find_dump(traj_dir, traj_id)
        if path is None:
            return {"id": traj_id, "status": "no_dump"}

        arr = np.load(path, mmap_mode="r")          # (T, N, 3)
        T = arr.shape[0]
        n = min(args.frames, T - 1) if T > 1 else 0
        if n < 2:
            return {"id": traj_id, "status": "too_short", "T": int(T)}

        x = np.asarray(arr[:n + 1], dtype=np.float64)      # materialise the block
        step = np.linalg.norm(np.diff(x, axis=0), axis=-1)  # (n, N)
        med = float(np.median(step))
        mx = float(step.max())
        thr = args.jump_threshold * med if med > 0 else np.inf
        jump = step > thr
        per_frame = jump.sum(axis=1)

        return {
            "id": traj_id,
            "status": "ok",
            "T": int(T),
            "N": int(arr.shape[1]),
            "frames_used": int(n),
            "med": med,
            "max": mx,
            "ratio": mx / med if med > 0 else float("inf"),
            "frac": float(jump.mean()),
            "together": int(per_frame.max()),
            "jump_frames": int((per_frame > 0).sum()),
            "range": [float(x.min()), float(x.max())],
        }
    except Exception as exc:  # pragma: no cover
        return {"id": traj_id, "status": "error", "error": f"{type(exc).__name__}: {exc}",
                "tb": traceback.format_exc()[-300:]}


def main():
    args = parse_args()
    ids = sorted(d for d in os.listdir(args.traj_dir)
                 if os.path.isdir(os.path.join(args.traj_dir, d)))
    if args.limit:
        ids = ids[:args.limit]
    print(f"scanning {len(ids)} trajectories from {args.traj_dir} "
          f"(first {args.frames} frames)", flush=True)

    out = open(args.out, "w") if args.out else None
    results, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(scan_one, args, args.traj_dir, i): i for i in ids}
        for k, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            results.append(r)
            if out:
                out.write(json.dumps(r) + "\n")
            if k % 200 == 0 or k == len(ids):
                el = time.time() - t0
                print(f"  {k}/{len(ids)}  {el:.0f}s  "
                      f"({el / k * 1000:.1f} ms/traj)", flush=True)
    if out:
        out.close()

    ok = [r for r in results if r["status"] == "ok"]
    bad_status = [r for r in results if r["status"] != "ok"]
    print(f"\n=== SUMMARY ===")
    print(f"scanned ok : {len(ok)}")
    print(f"other      : {len(bad_status)}")
    if bad_status:
        from collections import Counter
        print("  ", Counter(r["status"] for r in bad_status))

    if not ok:
        return 1

    import statistics as st
    ratios = [r["ratio"] for r in ok]
    together = [r["together"] for r in ok]
    fracs = [r["frac"] for r in ok]
    print(f"\nratio (max step / median):")
    print(f"  median={st.median(ratios):.1f}  p90={np.percentile(ratios, 90):.1f}  "
          f"max={max(ratios):.1f}")
    print(f"atoms jumping together in one frame:")
    print(f"  median={st.median(together):.0f}  p90={np.percentile(together, 90):.0f}  "
          f"max={max(together)}")
    print(f"frac of (frame,atom) above threshold:")
    print(f"  median={st.median(fracs):.2e}  p90={np.percentile(fracs, 90):.2e}  "
          f"max={max(fracs):.2e}")

    # A trajectory is suspect when a whole group moves coherently by ~box length.
    suspect = [r for r in ok if r["together"] >= 4 and r["ratio"] >= 25]
    clean = [r for r in ok if not (r["together"] >= 4 and r["ratio"] >= 25)]
    print(f"\nsuspect (>=4 atoms jumping together AND ratio>=25): {len(suspect)} "
          f"({100 * len(suspect) / len(ok):.1f}%)")
    print(f"clean                                            : {len(clean)} "
          f"({100 * len(clean) / len(ok):.1f}%)")
    if suspect:
        print("\nfirst 15 suspect ids:")
        for r in suspect[:15]:
            print(f"  {r['id']:16s} ratio={r['ratio']:7.1f} together={r['together']:3d} "
                  f"frac={r['frac']:.4f} N={r['N']}")

    if args.out:
        with open(args.out + ".summary.json", "w") as fh:
            json.dump({"n_ok": len(ok), "n_suspect": len(suspect),
                       "suspect_ids": [r["id"] for r in suspect]}, fh, indent=2)
        print(f"\nper-traj jsonl: {args.out}")
        print(f"summary       : {args.out}.summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
