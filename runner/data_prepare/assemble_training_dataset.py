#!/usr/bin/env python
"""Assemble the `atlas_10_features_fouriered_list_*.pth` training dataset.

This is the step that was missing from the repository. The downstream stages
(`replace_positions_full_fft_atlas.py`, `compute_fft_window_scaling.py`,
`src/data/dataset_atlas.py`, training) all *consume* a
`atlas_10_features_fouriered_list_{train,val,test}.pth`, but nothing created it.

Pipeline position
-----------------
    raw MD (.tpr/.xtc)
      -> extract_frames/save_frames_to_npy_atlas.py     => <ID>_frames_coords.npy  (Cα)
    mmCIF -> prepare_md_training_data*.py
      -> bioassembly/<pdb>.pkl.gz                       => rotation matrices, aatype
      -> sequences/pdb_seq.csv
      -> 4_compute_esm/compute_esm.sh                   => esm_cache/esm2_650m_embed/<pdb>_<ent>.pt
    >>> THIS SCRIPT <<<                                  => dataset/fft_window<W>/*.pth
    compute_fft_window_scaling.py                        => *_normalized.pth

What it writes
--------------
A pickled `list[dict]`, one entry per trajectory, with exactly the schema the
consumers require (verified against `ProteinMotionFFTWindowDataset` in
`src/data/dataset_atlas.py` and `replace_positions_full_fft_atlas.py`):

    id                         str
    amino_acid_types           LongTensor (N,)          # 0..19, non-standard -> 20
    rotation_matrices0         FloatTensor (N, 3, 3)    # from bioassembly backbone frames
    esm                        FloatTensor (N, esm_dim) # ESM-2-650M -> 1280
    positions
      # full-trajectory spectrum (used by load_full_trajectory_data / inference eval)
      x, y, z                  list[N] of ndarray (F, 2) float32
      x0, y0, z0               ndarray (N,) float32
      coords_preprocessed      ndarray (T, N, 3) float32
      preprocess_state         dict
      # windowed spectrum (what training iterates over)
      x_fft_windows            list[n_win] of list[N] of ndarray (F, 2) float32
      y_fft_windows, z_fft_windows   same
      x0_fft_windows           list[n_win] of ndarray (N,) float32
      y0_fft_windows, z0_fft_windows same
      coords_preprocessed_fft_windows   list[n_win] of ndarray (W, N, 3) float32
      preprocess_state_fft_windows      list[n_win] of dict
      fft_window_size          int
      fft_hop_ratio            float
      num_fft_windows          int

The FFT itself is delegated to
`replace_positions_full_fft_atlas.compute_{full,windowed}_full_fft_positions`
so that this script cannot drift from the preprocessing used downstream.

Usage
-----
    python runner/data_prepare/assemble_training_dataset.py \
        --traj_dir        /data_lmj/database/Dynamics/data/trajectories/atlas \
        --bioassembly_dir /data_lmj/database/Dynamics/data/bioassembly/atlas \
        --esm_dir         /data_lmj/database/Dynamics/data/esm_cache/esm2_650m_embed \
        --indices_csv     /data_lmj/database/Dynamics/data/indices/atlas_traj_indices.csv \
        --output_dir      ./dataset/fft_window2500 \
        --window_size 2500 --hop_ratio 1.0 --dt_str 10ps
"""
import argparse
import csv
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from runner.data_prepare.streaming_pth import StreamingPthWriter  # noqa: E402
from runner.data_prepare.replace_positions_full_fft_atlas import (  # noqa: E402
    compute_full_rfft_positions,
    compute_windowed_full_fft_positions,
)

# ---------------------------------------------------------------------------
# Amino-acid encoding
# ---------------------------------------------------------------------------
# Matches the released dataset (values observed in 0..19, non-standard -> 20).
AA3_TO_ID = {
    "ALA": 0, "ARG": 1, "ASN": 2, "ASP": 3, "CYS": 4,
    "GLN": 5, "GLU": 6, "GLY": 7, "HIS": 8, "ILE": 9,
    "LEU": 10, "LYS": 11, "MET": 12, "PHE": 13, "PRO": 14,
    "SER": 15, "THR": 16, "TRP": 17, "TYR": 18, "VAL": 19,
}
UNKNOWN_AA_ID = 20

CA_ATOM_NAME = "CA"


def _log(msg: str) -> None:
    print(f"[assemble] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
def load_traj_ids(indices_csv: Optional[str], traj_dir: str) -> List[str]:
    """Trajectory ids, from a CSV (`traj_name` column) or by listing traj_dir."""
    if indices_csv and os.path.exists(indices_csv):
        ids = []
        with open(indices_csv, newline="") as f:
            for row in csv.DictReader(f):
                tid = (row.get("traj_name") or "").strip()
                if tid:
                    ids.append(tid)
        if ids:
            return sorted(set(ids))
    return sorted(d for d in os.listdir(traj_dir) if os.path.isdir(os.path.join(traj_dir, d)))


def find_frames_npy(traj_dir: str, traj_id: str) -> Optional[str]:
    """Locate `<traj_id>_frames_coords.npy` under the traj's subdirectories.

    Trajectory ids sourced from the indices CSV are lowercase (`16pk_a_r1`)
    while the extracted-frame directories preserve the original case
    (`16pk_A_R1`), so resolve the directory case-insensitively.
    """
    base = os.path.join(traj_dir, traj_id)
    if not os.path.isdir(base):
        want = traj_id.lower()
        match = None
        for d in os.listdir(traj_dir):
            if d.lower() == want and os.path.isdir(os.path.join(traj_dir, d)):
                match = d
                break
        if match is None:
            return None
        base = os.path.join(traj_dir, match)
    for root, _dirs, files in os.walk(base):
        for fn in files:
            if fn.endswith("_frames_coords.npy"):
                return os.path.join(root, fn)
    return None


def to_structure_id(traj_id: str) -> str:
    """`16pk_A_R1` -> `16pk_a` (how bioassembly/ESM files are named)."""
    return "_".join(traj_id.split("_")[:2]).lower()


def find_any_dump_file(traj_dir: str, traj_id: str, suffix: str) -> Optional[str]:
    """Case-insensitive search for the first file under the traj dir ending in `suffix`."""
    base = os.path.join(traj_dir, traj_id)
    if not os.path.isdir(base):
        want = traj_id.lower()
        for d in os.listdir(traj_dir):
            if d.lower() == want and os.path.isdir(os.path.join(traj_dir, d)):
                base = os.path.join(traj_dir, d)
                break
        else:
            return None
    for root, _dirs, files in os.walk(base):
        for fn in files:
            if fn.endswith(suffix):
                return os.path.join(root, fn)
    return None


def find_topology_pdb(traj_dir: str, traj_id: str) -> Optional[str]:
    """PDB describing the atom order of the trajectory.

    Prefers the per-trajectory first-frame dump; falls back to the original
    topology recorded in `source_manifest.json` (some replicates ship without a
    first-frame PDB). Either way the atom order is the same, because every
    replicate of a structure shares one topology.
    """
    p = find_any_dump_file(traj_dir, traj_id, "_first_frame.pdb")
    if p is not None:
        return p

    import json

    manifest = find_any_dump_file(traj_dir, traj_id, "source_manifest.json")
    if manifest is None:
        return None
    try:
        with open(manifest) as f:
            meta = json.load(f)
    except Exception:
        return None
    topo = (meta.get("topology_source") or {}).get("path")
    if topo and os.path.exists(topo):
        return topo
    return None


def parse_pdb_atom_names(pdb_path: str) -> List[str]:
    """Atom names in file order for ATOM/HETATM records (altloc A or blank only)."""
    names: List[str] = []
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            altloc = line[16]
            if altloc not in (" ", "A"):
                continue
            names.append(line[12:16].strip())
    return names


def select_ca_coords(
    coords: np.ndarray,
    first_frame_pdb: Optional[str],
    n_res_expected: Optional[int],
    tol: float = 1e-2,
) -> Tuple[Optional[np.ndarray], str]:
    """Reduce an all-atom trajectory to Cα-only (paper Sec. 3.1 / Eq. 1).

    The extractor was run against all non-water atoms, so the `.npy` holds
    every heavy atom; BioDynaSpec is residue-level and needs one Cα per
    residue. Indices come from the accompanying first-frame PDB, which was
    written from the same atom selection, and the first frame is compared
    against the PDB coordinates as a sanity check.
    """
    n_atoms = coords.shape[1]
    if first_frame_pdb is None:
        return None, "no first-frame PDB to derive Cα indices from"

    names = parse_pdb_atom_names(first_frame_pdb)
    if len(names) != n_atoms:
        return None, f"PDB atoms {len(names)} != trajectory atoms {n_atoms}"

    ca_idx = [i for i, nm in enumerate(names) if nm == CA_ATOM_NAME]
    if not ca_idx:
        return None, "no CA atoms in first-frame PDB"
    if n_res_expected is not None and len(ca_idx) != n_res_expected:
        return None, f"Cα count {len(ca_idx)} != expected residues {n_res_expected}"

    # Sanity-check atom ordering: first frame vs PDB coordinates.
    try:
        pdb_xyz = parse_pdb_coords(first_frame_pdb)
        if pdb_xyz is not None and pdb_xyz.shape == (n_atoms, 3):
            if not np.allclose(coords[0], pdb_xyz, atol=tol):
                return None, "trajectory atom order does not match the first-frame PDB"
    except Exception:
        pass  # non-fatal: the index count check above is the primary guard

    return coords[:, ca_idx, :], "ok"


def parse_pdb_coords(pdb_path: str) -> Optional[np.ndarray]:
    """Coordinates in file order for ATOM/HETATM records (altloc A or blank)."""
    xyz: List[Tuple[float, float, float]] = []
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            if line[16] not in (" ", "A"):
                continue
            try:
                xyz.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
            except ValueError:
                return None
    return np.asarray(xyz, dtype=np.float32) if xyz else None


def load_bioassembly(bioassembly_dir: str, structure_id: str) -> Optional[Dict]:
    """Load `bioassembly/<structure_id>.pkl.gz`, with a `.pkl` fallback."""
    import gzip
    import pickle

    for name in (f"{structure_id}.pkl.gz", f"{structure_id}.pkl"):
        p = os.path.join(bioassembly_dir, name)
        if not os.path.exists(p):
            continue
        opener = gzip.open if p.endswith(".gz") else open
        with opener(p, "rb") as f:
            return pickle.load(f)
    return None


def extract_aatype_and_rotations(bioassembly: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-Cα-residue amino-acid ids and backbone rotation matrices.

    Uses the token array's `centre_atom_index` to pick the representative atom
    of each token (the Cα for standard residues), matching how the trajectory's
    Cα-only coordinates are extracted.
    """
    atom_array = bioassembly["atom_array"]
    token_array = bioassembly["token_array"]

    atom_names = np.asarray(atom_array.atom_name)
    res_names = np.asarray(atom_array.res_name)

    aatypes: List[int] = []
    centre_idx: List[int] = []
    for tok in token_array:
        ci = int(tok._annot["centre_atom_index"])
        centre_idx.append(ci)
        aatypes.append(AA3_TO_ID.get(str(res_names[ci]).upper(), UNKNOWN_AA_ID))

    coords = np.asarray(atom_array.coord, dtype=np.float64)
    n = len(centre_idx)

    # Prefer real backbone frames from the bioassembly if present.
    rot = None
    for key in ("backbone_rigid_tensor", "bb_rigid_tensor"):
        t = bioassembly.get(key)
        if t is not None:
            arr = t.detach().cpu().numpy() if torch.is_tensor(t) else np.asarray(t)
            if arr.shape[0] == n:
                rot = arr[:, :3, :3].astype(np.float32)
                break
    if rot is None:
        rot = estimate_backbone_frames(coords[centre_idx])

    return (
        torch.tensor(aatypes, dtype=torch.int64),
        torch.tensor(rot, dtype=torch.float32),
    )


def estimate_backbone_frames(ca_coords: np.ndarray) -> np.ndarray:
    """Build a right-handed frame per residue from consecutive Cα positions.

    Fallback when the bioassembly has no rigid-group tensor. The frames are
    derived from the local chain direction, so they are orthonormal by
    construction (`rotation_matrices0` is consumed as an SE(3) rotation).
    """
    n = ca_coords.shape[0]
    ca = ca_coords.astype(np.float64)
    frames = np.tile(np.eye(3, dtype=np.float32), (n, 1, 1))
    for i in range(n):
        prev = ca[i - 1] if i > 0 else ca[i]
        nxt = ca[i + 1] if i + 1 < n else ca[i]
        fwd = nxt - prev
        nrm = np.linalg.norm(fwd)
        if nrm < 1e-8:
            continue
        e1 = fwd / nrm
        ref = np.array([0.0, 0.0, 1.0])
        if abs(float(np.dot(e1, ref))) > 0.9:
            ref = np.array([0.0, 1.0, 0.0])
        e2 = np.cross(e1, ref)
        e2 /= max(np.linalg.norm(e2), 1e-8)
        e3 = np.cross(e1, e2)
        frames[i] = np.stack([e1, e2, e3], axis=0).astype(np.float32)
    return frames


def load_esm(esm_dir: str, structure_id: str, n_res: int, esm_dim: int) -> Optional[torch.Tensor]:
    """Load the ESM embedding for `structure_id` and validate its shape.

    Files are named `<pdb_id>_<entity_number>.pt` (e.g. `16pk_a_1.pt`), stored
    in a two-character shard subdirectory, so glob rather than guess the
    entity number.
    """
    import glob

    candidates = []
    for sub in ("", *_subdirs(esm_dir)):
        candidates.extend(glob.glob(os.path.join(esm_dir, sub, f"{structure_id}_*.pt")))
        candidates.extend(glob.glob(os.path.join(esm_dir, sub, f"{structure_id}.pt")))
    if not candidates:
        return None

    for p in sorted(set(candidates)):
        try:
            emb = torch.load(p, map_location="cpu", weights_only=False)
        except Exception:
            continue
        if not torch.is_tensor(emb) or emb.dim() != 2:
            continue
        if emb.shape[0] != n_res:
            continue
        if emb.shape[1] != esm_dim:
            _log(f"  ! {os.path.basename(p)}: dim {emb.shape[1]} != expected {esm_dim}")
            continue
        return emb.float()
    return None


_ESM_SUBDIR_CACHE: List[str] = []


def _subdirs(root: str) -> List[str]:
    if not _ESM_SUBDIR_CACHE:
        _ESM_SUBDIR_CACHE.extend(
            d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))
        )
    return _ESM_SUBDIR_CACHE


# ---------------------------------------------------------------------------
# Per-trajectory assembly
# ---------------------------------------------------------------------------
class FilteredOut(Exception):
    """Raised when a trajectory is intentionally excluded (not an error)."""


def build_entry(
    traj_id: str,
    traj_dir: str,
    bioassembly_dir: str,
    esm_dir: str,
    window_size: int,
    hop_ratio: float,
    dt: float,
    esm_dim: int,
    max_residues: int,
    include_full_spectrum: bool = True,
) -> Optional[Dict]:
    npy_path = find_frames_npy(traj_dir, traj_id)
    if npy_path is None:
        _log(f"  ! {traj_id}: frames_coords.npy not found")
        return None

    coords = np.load(npy_path, mmap_mode="r")
    coords = np.asarray(coords, dtype=np.float32)
    if coords.ndim != 3 or coords.shape[-1] != 3:
        _log(f"  ! {traj_id}: expected (T,N,3), got {coords.shape}")
        return None
    T = coords.shape[0]
    if T < window_size:
        _log(f"  ! {traj_id}: T={T} < window_size={window_size}; skipping")
        return None

    structure_id = to_structure_id(traj_id)

    bio = load_bioassembly(bioassembly_dir, structure_id)
    if bio is None:
        _log(f"  ! {traj_id}: bioassembly {structure_id}.pkl.gz not found")
        return None
    aatype, rot = extract_aatype_and_rotations(bio)
    n_res = int(aatype.shape[0])

    # The extractor wrote every non-water atom; reduce to Cα (residue-level).
    ca_coords, why = select_ca_coords(
        coords,
        find_topology_pdb(traj_dir, traj_id),
        n_res_expected=n_res,
    )
    if ca_coords is None:
        _log(f"  ! {traj_id}: Cα selection failed: {why}")
        return None
    coords = ca_coords
    if n_res > max_residues:
        # Deliberate, documented filter (the trainer cannot take longer chains),
        # NOT a failure. Raised so the caller can report it separately and not
        # trip the "refuse to write a partial dataset" guard.
        _log(f"  ! {traj_id}: {n_res} residues > {max_residues}; filtered out")
        raise FilteredOut(f"{n_res} residues > max_residues={max_residues}")

    esm = load_esm(esm_dir, structure_id, n_res, esm_dim)
    if esm is None:
        _log(f"  ! {traj_id}: ESM ({esm_dim}-d) not found/usable for {structure_id}")
        return None

    positions: Dict[str, object] = {}

    if include_full_spectrum:
        xl, yl, zl, frame0, coords_pre, pre_state = compute_full_rfft_positions(coords, dt=dt)
        positions["x"], positions["y"], positions["z"] = xl, yl, zl
        positions["x0"] = frame0[:, 0].astype(np.float32, copy=False)
        positions["y0"] = frame0[:, 1].astype(np.float32, copy=False)
        positions["z0"] = frame0[:, 2].astype(np.float32, copy=False)
        positions["coords_preprocessed"] = coords_pre
        positions["preprocess_state"] = pre_state

    (xw, yw, zw, f0w, pre_w, state_w) = compute_windowed_full_fft_positions(
        coords, dt=dt, window_size=window_size, hop_ratio=hop_ratio
    )
    if not xw:
        _log(f"  ! {traj_id}: no windows produced")
        return None
    positions["x_fft_windows"] = xw
    positions["y_fft_windows"] = yw
    positions["z_fft_windows"] = zw
    positions["x0_fft_windows"] = [f[:, 0].astype(np.float32, copy=False) for f in f0w]
    positions["y0_fft_windows"] = [f[:, 1].astype(np.float32, copy=False) for f in f0w]
    positions["z0_fft_windows"] = [f[:, 2].astype(np.float32, copy=False) for f in f0w]
    positions["coords_preprocessed_fft_windows"] = pre_w
    positions["preprocess_state_fft_windows"] = state_w
    positions["fft_window_size"] = int(window_size)
    positions["fft_hop_ratio"] = float(hop_ratio)
    positions["num_fft_windows"] = int(len(xw))

    return {
        "id": traj_id,
        "positions": positions,
        "rotation_matrices0": rot,
        "amino_acid_types": aatype,
        "esm": esm,
    }


def _limit_threads():
    """Cap per-process BLAS/OpenMP threads.

    Worker processes inherit the parent's thread count (96 here) because
    `torch` is already imported when the fork happens; with 48 workers on 192
    cores that oversubscribes badly and measured CPU use sat near 200-250%
    instead of ~100% per worker. Must run inside the child.
    """
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, "2")
    try:
        import torch as _torch
        _torch.set_num_threads(2)
    except Exception:
        pass


def _worker(kwargs: Dict) -> Tuple[str, Optional[Dict], str]:
    _limit_threads()
    traj_id = kwargs["traj_id"]
    try:
        entry = build_entry(**kwargs)
        return traj_id, entry, "ok" if entry is not None else "skipped"
    except FilteredOut as e:
        return traj_id, None, f"filtered: {e}"
    except Exception as e:  # keep one bad trajectory from killing the run
        return traj_id, None, f"error: {type(e).__name__}: {e}\n{traceback.format_exc()}"


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------
def split_ids(ids: List[str], fracs: Tuple[float, float, float], seed: int) -> Dict[str, List[str]]:
    """Split TRAJECTORY ids into train/val/test **by structure**.

    Every trajectory is `<pdb>_<chain>_R<replica>`, and the three replicas of a
    structure are different simulations of the SAME protein. Splitting the flat
    trajectory list (the previous behaviour) therefore put e.g. `6h49_A_R1` in
    train and `6h49_A_R2` in test: measured on this dataset, all 156 test
    structures and all 120 val structures also appeared in train, i.e. the
    held-out sets were 100% leaked and any reported test metric would be
    meaningless.

    Grouping by structure first and assigning whole groups keeps a structure's
    replicas together, which is the paper's protocol (1269/39/82 by structure).
    """
    import random

    def structure_of(traj_id: str) -> str:
        return traj_id.rsplit("_", 1)[0].lower()

    by_structure: Dict[str, List[str]] = {}
    for t in sorted(ids):
        by_structure.setdefault(structure_of(t), []).append(t)

    structures = sorted(by_structure)
    rng = random.Random(seed)
    rng.shuffle(structures)
    n = len(structures)
    n_train = int(round(n * fracs[0]))
    n_val = int(round(n * fracs[1]))

    groups = {
        "train": structures[:n_train],
        "val": structures[n_train:n_train + n_val],
        "test": structures[n_train + n_val:],
    }
    return {k: [t for s in v for t in by_structure[s]] for k, v in groups.items()}


def _split_ids_old(ids: List[str], fracs: Tuple[float, float, float], seed: int) -> Dict[str, List[str]]:
    import random

    ids = sorted(ids)
    rng = random.Random(seed)
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(round(n * fracs[0]))
    n_val = int(round(n * fracs[1]))
    return {
        "train": ids[:n_train],
        "val": ids[n_train:n_train + n_val],
        "test": ids[n_train + n_val:],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traj_dir", required=True, help="Root with <traj_id>/<traj_id>_no_water/<traj_id>_frames_coords.npy")
    ap.add_argument("--bioassembly_dir", required=True, help="Dir with <structure_id>.pkl.gz")
    ap.add_argument("--esm_dir", required=True, help="Dir with <structure_id>_<entity>.pt (ESM embeddings)")
    ap.add_argument("--indices_csv", default=None, help="CSV with a traj_name column (else list traj_dir)")
    ap.add_argument("--output_dir", required=True, help="Where to write the .pth files")
    ap.add_argument("--window_size", type=int, default=2500)
    ap.add_argument("--hop_ratio", type=float, default=1.0)
    ap.add_argument("--dt_str", type=str, default="10ps")
    ap.add_argument("--esm_dim", type=int, default=1280, help="ESM-2-650M = 1280")
    ap.add_argument("--max_residues", type=int, default=1000, help="Skip longer chains (matches the trainer filter)")
    ap.add_argument("--ratios", type=float, nargs=3,
                    default=(0.91295, 0.02806, 0.05899),
                    metavar=("TRAIN", "VAL", "TEST"),
                    help="Structure-level split ratios. Defaults reproduce the paper's "
                         "1269/39/82 (out of 1390) protocol.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_workers", type=int, default=max(1, (os.cpu_count() or 2) // 4))
    ap.add_argument("--max_proteins", type=int, default=None, help="Cap the number of trajectories (debug)")
    ap.add_argument("--no_full_spectrum", action="store_true", help="Skip the full-trajectory rFFT (smaller files)")
    ap.add_argument("--shard_size", type=int, default=0, help="If >0, write shards of this many proteins instead of one file")
    args = ap.parse_args()

    from src.data.fourier.utils import parse_time_str_to_seconds

    dt = float(parse_time_str_to_seconds(args.dt_str))

    os.makedirs(args.output_dir, exist_ok=True)

    ids = load_traj_ids(args.indices_csv, args.traj_dir)
    if args.max_proteins:
        ids = ids[: args.max_proteins]
    _log(f"{len(ids)} trajectories to assemble")

    splits = split_ids(ids, tuple(args.ratios), args.seed)
    for k, v in splits.items():
        _log(f"  split {k}: {len(v)}")

    common = dict(
        traj_dir=args.traj_dir,
        bioassembly_dir=args.bioassembly_dir,
        esm_dir=args.esm_dir,
        window_size=args.window_size,
        hop_ratio=args.hop_ratio,
        dt=dt,
        esm_dim=args.esm_dim,
        max_residues=args.max_residues,
        include_full_spectrum=not args.no_full_spectrum,
    )

    t0 = time.time()
    stats = {"ok": 0, "skipped": 0, "error": 0, "filtered": 0}

    # Entries are streamed straight to disk rather than accumulated. Holding all
    # ~4150 entries cost ~987 GB and the cgroup OOM-killed the run mid-write.
    # Each entry is ~237 MB (the bulk is `coords_preprocessed_fft_windows`,
    # (W, N, 3) per window), so peak memory is now one entry.
    base = "atlas_10_features_fouriered_list"
    split_of = {t: sp for sp, ts in splits.items() for t in ts}
    n_win = {sp: 0 for sp in splits}
    seen = {sp: 0 for sp in splits}
    written = set()

    # When --shard_size is set each split becomes a DIRECTORY of shard files plus
    # a manifest, so no consumer ever has to materialise the whole split.
    # The single-file layout needs `torch.load` of the entire split to read it,
    # and pickle cannot yield list elements one at a time (a list's items share
    # one STOP, so `Unpickler.load()` always returns the whole list). At 638 GB
    # train that is both an OOM risk and unusable for normalization.
    #
    # The shard-directory layout is the one `ProteinMotionFFTWindowDataset`
    # already accepts (see `_load_protein_list` in src/data/dataset_atlas.py:
    # it reads `manifest.json`, loads only this rank/worker's shards and keeps a
    # no-op iterator).
    shard_size = int(getattr(args, "shard_size", 0) or 0)
    shard_paths = {sp: [] for sp in splits}
    shard_counts = {sp: [] for sp in splits}
    shard_windows = {sp: [] for sp in splits}
    writers = {}
    cur = {}

    def _split_dir(sp):
        return os.path.join(args.output_dir, sp)

    def _next_shard(sp):
        prev = cur.get(sp)
        if prev is not None:
            prev.close()
        if shard_size:
            os.makedirs(_split_dir(sp), exist_ok=True)
            path = os.path.join(_split_dir(sp), f"shard_{len(shard_paths[sp]):04d}.pth")
        else:
            path = os.path.join(args.output_dir, f"{base}_{sp}.pth")
        writers[sp] = StreamingPthWriter(path)
        cur[sp] = writers[sp]
        shard_paths[sp].append(path)
        shard_counts[sp].append(0)
        shard_windows[sp].append(0)
        return writers[sp]

    def _record(tid, entry):
        written.add(tid)
        sp = split_of[tid]
        w = cur.get(sp)
        if w is None:
            w = _next_shard(sp)
        elif shard_size and shard_counts[sp][-1] >= shard_size:
            w = _next_shard(sp)
        w.append(entry)
        nw = int(entry["positions"]["num_fft_windows"])
        shard_counts[sp][-1] += 1
        shard_windows[sp][-1] += nw
        n_win[sp] += nw
        seen[sp] += 1

    if args.num_workers > 1:
        with ProcessPoolExecutor(max_workers=args.num_workers) as ex:
            futs = {ex.submit(_worker, {**common, "traj_id": t}): t for t in ids}
            for i, fut in enumerate(as_completed(futs), 1):
                tid, entry, status = fut.result()
                if entry is not None:
                    _record(tid, entry)
                    stats["ok"] += 1
                elif status == "skipped":
                    stats["skipped"] += 1
                elif status.startswith("filtered"):
                    stats["filtered"] += 1
                else:
                    stats["error"] += 1
                    _log(f"  x {tid}: {status.splitlines()[0]}")
                if i % 25 == 0 or i == len(ids):
                    _log(f"  {i}/{len(ids)} done  ({time.time() - t0:.0f}s)")
    else:
        for i, tid in enumerate(ids, 1):
            _tid, entry, status = _worker({**common, "traj_id": tid})
            if entry is not None:
                _record(tid, entry)
                stats["ok"] += 1
            elif status == "skipped":
                stats["skipped"] += 1
            elif status.startswith("filtered"):
                stats["filtered"] += 1
            else:
                stats["error"] += 1
                _log(f"  x {tid}: {status}")
            if i % 10 == 0 or i == len(ids):
                _log(f"  {i}/{len(ids)} done  ({time.time() - t0:.0f}s)")

    _log(f"assembled ok={stats['ok']} filtered={stats['filtered']} "
         f"skipped={stats['skipped']} error={stats['error']}")

    # Fail loudly. A per-trajectory error previously only produced a log line, so
    # a partially-failed assembly still wrote a .pth and looked successful -- the
    # OOM run had already dropped 42 trajectories before anyone noticed. Refuse
    # to write output unless every trajectory is accounted for, so a "successful"
    # run always means a complete dataset. Set ALLOW_PARTIAL_ASSEMBLY=1 to
    # override (e.g. when the few failures are the known-bad PBC trajectories).
    n_bad = stats["skipped"] + stats["error"]
    if n_bad and os.environ.get("ALLOW_PARTIAL_ASSEMBLY") != "1":
        _log(f"ABORT: {n_bad}/{len(ids)} trajectories FAILED to assemble "
             f"(skipped={stats['skipped']} error={stats['error']}; "
             f"the {stats['filtered']} filtered ones are deliberate and do not "
             f"count). "
             f"Writing a partial .pth would silently shrink the dataset. "
             f"Fix the cause, or set ALLOW_PARTIAL_ASSEMBLY=1 to accept it.")
        # The writers already hold partial files; drop them so an aborted run
        # cannot be mistaken for a finished dataset.
        for w in writers.values():
            w.close()
        for sp in splits:
            paths = shard_paths[sp] or [os.path.join(args.output_dir, f"{base}_{sp}.pth")]
            for p in paths:
                if os.path.exists(p):
                    os.remove(p)
            mpath = os.path.join(_split_dir(sp), "manifest.json")
            if os.path.exists(mpath):
                os.remove(mpath)
        return 1

    dropped = sorted(t for t in ids if split_of.get(t) and t not in written)
    if dropped:
        _log(f"  {len(dropped)} trajectories absent from the output "
             f"({len({d.rsplit('_', 1)[0] for d in dropped})} structures): "
             f"{dropped[:8]}{' ...' if len(dropped) > 8 else ''}")
        _log("  -> the split will be smaller than the nominal ratio. If these are "
             "deliberate filters this is expected; check the reasons above.")

    import json

    for split in splits:
        if split in writers:
            writers[split].close()
        if shard_size:
            # An empty split (e.g. val when the sample is tiny) never had a
            # shard opened, so its directory does not exist yet -- create it or
            # the manifest write below raises FileNotFoundError.
            os.makedirs(_split_dir(split), exist_ok=True)
            # Manifest with the SAME schema `make_shards.py` writes, so
            # `ProteinMotionFFTWindowDataset` loads it with no changes: it reads
            # `shards`, and `window_counts` is what the DDP step balancer needs
            # (protein counts are not enough -- proteins differ in window count).
            names = [os.path.basename(p) for p in shard_paths[split]]
            manifest = {
                "num_shards": len(names),
                "num_proteins": seen[split],
                "num_workers": 0,
                "counts": shard_counts[split],
                "window_counts": shard_windows[split],
                "num_windows": n_win[split],
                "shards": names,
                "source": "assemble_training_dataset.py --shard_size",
                "window_key": "x_fft_windows",
            }
            mpath = os.path.join(_split_dir(split), "manifest.json")
            with open(mpath, "w") as fh:
                json.dump(manifest, fh, indent=2)
            total = sum(os.path.getsize(p) for p in shard_paths[split]) / 1e9
            _log(f"wrote {_split_dir(split)}/  ({len(names)} shards, "
                 f"{seen[split]} proteins, {n_win[split]} windows, {total:.2f} GB)")
        else:
            out = shard_paths[split][0]
            _log(f"wrote {out}  ({seen[split]} proteins, {n_win[split]} windows, "
                 f"{os.path.getsize(out)/1e9:.2f} GB)")

    _log(f"done in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
