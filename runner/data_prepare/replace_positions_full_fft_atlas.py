import argparse
import os
import concurrent.futures
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from tqdm import tqdm

import MDAnalysis as mda

from src.data.fourier.utils import (
    extract_positive_frequency_half_spectrum,
    parse_time_str_to_seconds,
    TrajectoryFFTConfig,
    TrajectoryFFTProcessor,
)
from src.protenix.utils import aggregate_atom_to_token


# Device for the rFFT. This runs inside a ProcessPoolExecutor, so a CUDA device
# means every worker allocates on the SAME GPU -- with 48 workers that exhausts
# an 80 GB card and trajectories fail with OutOfMemoryError / cuFFT
# CUFFT_ALLOC_FAILED, silently dropping them from the assembled dataset.
#
# The transform is a few 2500-point rFFTs per trajectory, which is far cheaper
# than the IPC cost of shipping coordinates to a GPU, so CPU is both safer and
# not slower in practice. `BDS_FFT_DEVICE=cuda` restores the old behaviour.
_FFT_DEVICE = os.environ.get("BDS_FFT_DEVICE", "cpu")


def _build_fft_processor() -> TrajectoryFFTProcessor:
    config = TrajectoryFFTConfig(
        norm="ortho",
        use_double=False,
        window=None,
        use_sliding_window=False,
        store_dtype=torch.float32,
        store_complex_as_float2=True,
        device=_FFT_DEVICE,
    )
    return TrajectoryFFTProcessor(config)


def _preprocess_window(
    processor: TrajectoryFFTProcessor,
    coords_tn3: np.ndarray,
    dt: float,
) -> Tuple[torch.Tensor, np.ndarray, Dict]:
    coords_tn3 = np.asarray(coords_tn3, dtype=np.float32)
    coords_t = torch.from_numpy(coords_tn3)  # (T, N, 3)
    x_proc, state = processor.preprocess(
        coords_t,
        dt=dt,
        layout_hint="TN3",
        remove_com=True,
        align_rotation=True,
    )  # (N, 3, T)
    coords_preprocessed_tn3 = x_proc.permute(2, 0, 1).contiguous().cpu().numpy().astype(np.float32, copy=False)
    return x_proc, coords_preprocessed_tn3, state.to_dict(store_on_cpu=True)


def load_xtc_trajectory(
    pdb_path: str,
    xtc_path: str,
    atom_selection: Optional[str],
) -> np.ndarray:
    """
    Load trajectory coordinates from a PDB+XTC pair.

    Returns:
        coords: numpy array of shape (T, N, 3), float32
    """
    u = mda.Universe(pdb_path, xtc_path)
    ag = u.select_atoms(atom_selection) if atom_selection else u.atoms
    n_frames = len(u.trajectory)
    n_atoms = len(ag)
    coords = np.empty((n_frames, n_atoms, 3), dtype=np.float32)
    for i, ts in enumerate(u.trajectory):
        coords[i] = ag.positions
    return coords


def subsample_trajectory(
    coords_tn3: np.ndarray,
    num_frames: int,
    total_length: Optional[int] = None,
) -> np.ndarray:
    """
    Uniformly subsample a trajectory to a fixed number of frames.

    Args:
        coords_tn3: numpy array (T, N, 3)
        num_frames: number of frames to keep
        total_length: optional expected total length for computing stride;
            defaults to coords_tn3.shape[0]

    Returns:
        Subsampled coords: numpy array (num_frames, N, 3)
    """
    if num_frames <= 0:
        raise ValueError(f"Invalid num_frames={num_frames}")
    T = coords_tn3.shape[0]
    total_len = int(total_length) if total_length is not None else T
    if total_len <= 1:
        raise ValueError(f"Invalid total_length={total_len}")
    if T < total_len:
        # Trajectory shorter than expected; fall back to actual length
        total_len = T
    if num_frames > total_len:
        raise ValueError(
            f"num_frames={num_frames} is larger than total_length={total_len}"
        )
    # Inclusive sampling from [0, total_len - 1]
    indices = np.round(np.linspace(0, total_len - 1, num_frames)).astype(np.int64)
    return coords_tn3[indices]


def compute_full_rfft_positions(
    coords_tn3: np.ndarray,
    dt: float,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], np.ndarray, np.ndarray, Dict]:
    """
    Compute full (non-truncated) FFT and keep only positive frequencies (Hermitian half).
    Apply CoM removal + Kabsch alignment before FFT.

    Args:
        coords_tn3: numpy array (T, N, 3)

    Returns:
        x_list, y_list, z_list: lists of length N, each element is (F, 2) real/imag
        frame0: numpy array (N, 3) first-frame coordinates (after preprocessing)
        coords_preprocessed: numpy array (T, N, 3) preprocessed coords
        preprocess_state: dict for inverting preprocessing
    """
    processor = _build_fft_processor()
    x_proc, coords_preprocessed_tn3, preprocess_state = _preprocess_window(
        processor, coords_tn3, dt=dt
    )
    frame0 = x_proc[:, :, 0]  # (N, 3)
    displacement = x_proc - frame0.unsqueeze(-1)  # (N, 3, T)

    # Full FFT then keep positive half spectrum (ortho norm)
    full_fft = torch.fft.fft(displacement, dim=-1, norm="ortho")
    half_fft = extract_positive_frequency_half_spectrum(full_fft)  # (N, 3, F)

    half_fft_np = half_fft.cpu().numpy()
    # Build (N, F, 2) arrays for each axis
    x_spec = np.stack([half_fft_np[:, 0, :].real, half_fft_np[:, 0, :].imag], axis=-1)
    y_spec = np.stack([half_fft_np[:, 1, :].real, half_fft_np[:, 1, :].imag], axis=-1)
    z_spec = np.stack([half_fft_np[:, 2, :].real, half_fft_np[:, 2, :].imag], axis=-1)

    x_list = [x_spec[i].astype(np.float32, copy=False) for i in range(x_spec.shape[0])]
    y_list = [y_spec[i].astype(np.float32, copy=False) for i in range(y_spec.shape[0])]
    z_list = [z_spec[i].astype(np.float32, copy=False) for i in range(z_spec.shape[0])]

    return (
        x_list,
        y_list,
        z_list,
        frame0.cpu().numpy().astype(np.float32, copy=False),
        coords_preprocessed_tn3,
        preprocess_state,
    )


def compute_windowed_full_fft_positions(
    coords_tn3: np.ndarray,
    dt: float,
    window_size: int = 200,
    hop_ratio: float = 1.0,
) -> Tuple[
    List[List[np.ndarray]],
    List[List[np.ndarray]],
    List[List[np.ndarray]],
    np.ndarray,
    List[np.ndarray],
    List[Dict],
]:
    """
    Split long trajectory into windows, run full FFT in each window (no truncation),
    and keep only positive frequencies. No COLA/overlap reconstruction assumed.
    """
    coords_tn3 = np.asarray(coords_tn3, dtype=np.float32)
    T, N, _ = coords_tn3.shape
    hop_size = int(window_size * hop_ratio)
    if hop_size <= 0:
        raise ValueError(f"Invalid hop_size={hop_size}, window_size={window_size}, hop_ratio={hop_ratio}")
    if T < window_size:
        return [], [], [], np.empty((0, N, 3), dtype=np.float32)

    x_windows: List[List[np.ndarray]] = []
    y_windows: List[List[np.ndarray]] = []
    z_windows: List[List[np.ndarray]] = []
    frame0_list = []
    preprocessed_windows: List[np.ndarray] = []
    preprocess_states: List[Dict] = []

    n_windows = 1 + (T - window_size) // hop_size
    for w in range(n_windows):
        start = w * hop_size
        end = start + window_size
        coords_window = coords_tn3[start:end]  # (window_size, N, 3)
        x_list, y_list, z_list, frame0, coords_preprocessed_tn3, preprocess_state = compute_full_rfft_positions(
            coords_window, dt=dt
        )
        x_windows.append(x_list)
        y_windows.append(y_list)
        z_windows.append(z_list)
        frame0_list.append(frame0)
        preprocessed_windows.append(coords_preprocessed_tn3)
        preprocess_states.append(preprocess_state)

    frame0_windows = np.stack(frame0_list, axis=0)
    return x_windows, y_windows, z_windows, frame0_windows, preprocessed_windows, preprocess_states


def build_xtc_paths(base_dir: str, protein_id: str) -> Tuple[str, str]:
    """
    Build PDB/XTC paths for a given protein_id like "16pk_A_R1".
    Files are expected as: {base_dir}/{protein_id}_gt.pdb / .xtc
    """
    pdb_path = os.path.join(base_dir, f"{protein_id}_gt.pdb")
    xtc_path = os.path.join(base_dir, f"{protein_id}_gt.xtc")
    return pdb_path, xtc_path


def load_bihito_repr_dict(
    bihito_repr_dir: str,
    protein_id: str,
    verbose: bool = False,
) -> Optional[Dict]:
    bihito_path = os.path.join(bihito_repr_dir, f"{protein_id.lower()}.pt")
    if not os.path.exists(bihito_path):
        if verbose:
            print(f"[WARN] bihito repr not found: {bihito_path}")
        return None
    bihito_repr_dict = torch.load(bihito_path, map_location="cpu", weights_only=False)
    for key, value in bihito_repr_dict.items():
        if isinstance(value, torch.Tensor):
            bihito_repr_dict[key] = value.detach().cpu()
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, torch.Tensor):
                    value[sub_key] = sub_value.detach().cpu()
    return bihito_repr_dict


def extract_atom_to_token_idx(feature_cache: Dict) -> Optional[torch.Tensor]:
    return feature_cache["data"]["input_feature_dict"]["atom_to_token_idx"]


def load_atom_to_token_idx(
    feature_cache_dir: str,
    protein_id: str,
    verbose: bool = False,
) -> Optional[torch.Tensor]:
    feature_path = os.path.join(feature_cache_dir, f"{protein_id.lower()}.pt")
    if not os.path.exists(feature_path):
        if verbose:
            print(f"[WARN] feature_cache not found: {feature_path}")
        return None
    feature_cache = torch.load(feature_path, map_location="cpu", weights_only=False)
    if not isinstance(feature_cache, dict):
        if verbose:
            print(f"[WARN] feature_cache format not dict for {feature_path}")
        return None
    atom_to_token_idx = extract_atom_to_token_idx(feature_cache)
    if atom_to_token_idx is None:
        if verbose:
            print(f"[WARN] atom_to_token_idx missing in feature_cache: {feature_path}")
        return None
    return torch.as_tensor(atom_to_token_idx, dtype=torch.long)


def replace_positions_in_data_list(
    data_list: List[Dict],
    xtc_base_dir: str,
    atom_selection: Optional[str],
    fallback_selection: Optional[str],
    bihito_repr_dir: str,
    feature_cache_dir: str,
    dt: float,

    use_fft_windows: bool = False,
    window_size: int = 200,
    hop_ratio: float = 0.5,
    sample_num_frames: Optional[int] = None,
    sample_total_length: Optional[int] = None,
    truncate_to_expected_n: bool = False,
    num_workers: Optional[int] = None,
    drop_missing: bool = True,
    verbose: bool = False,
) -> List[Dict]:
    """
    Replace positions (x,y,z,x0,y0,z0) in a data_list in-place.
    """
    def _process_one(index: int, item: Dict) -> Tuple[int, Optional[Dict], str]:
        protein_id = item.get("id", "")
        if not protein_id:
            return index, item, "no_id"

        pdb_path, xtc_path = build_xtc_paths(xtc_base_dir, protein_id)
        if not (os.path.exists(pdb_path) and os.path.exists(xtc_path)):
            if verbose:
                print(f"[WARN] Missing files for {protein_id}: {pdb_path}, {xtc_path}")
            return index, item if not drop_missing else None, "missing_xtc_pdb"

        expected_n = len(item["positions"]["x"])

        # Try primary selection
        coords = load_xtc_trajectory(pdb_path, xtc_path, atom_selection)
        if coords.shape[1] != expected_n and fallback_selection:
            # Try fallback selection if atom count mismatch
            coords = load_xtc_trajectory(pdb_path, xtc_path, fallback_selection)

        if coords.shape[1] != expected_n:
            if truncate_to_expected_n and expected_n < coords.shape[1]:
                if verbose:
                    print(
                        f"[WARN] Atom count mismatch for {protein_id}: "
                        f"expected {expected_n}, got {coords.shape[1]} -> truncate"
                    )
                coords = coords[:, :expected_n, :]
            else:
                if verbose:
                    print(
                        f"[WARN] Atom count mismatch for {protein_id}: "
                        f"expected {expected_n}, got {coords.shape[1]}"
                    )
                return index, item if not drop_missing else None, "atom_count_mismatch"

        # bihito_repr_dict = load_bihito_repr_dict(bihito_repr_dir, protein_id, verbose=verbose)
        # atom_to_token_idx = load_atom_to_token_idx(feature_cache_dir, protein_id, verbose=verbose)
        # if bihito_repr_dict is None or atom_to_token_idx is None:
        #     return index, item if not drop_missing else None, "missing_bihito_or_feature_cache"

        # if 0 in bihito_repr_dict:
        #     first_frame_idx = 0
        # else:
        #     first_frame_idx = min(bihito_repr_dict.keys())
        # first_frame_bihito_repr_dict = bihito_repr_dict[first_frame_idx]
        # if "level3_encoding" not in first_frame_bihito_repr_dict:
        #     if verbose:
        #         print(f"[WARN] level3_encoding missing for {protein_id}")
        #     return index, item if not drop_missing else None, "missing_level3_encoding"

        # a_token = first_frame_bihito_repr_dict["level3_encoding"]
        # if not isinstance(a_token, torch.Tensor):
        #     a_token = torch.as_tensor(a_token)
        # if a_token.dim() < 2:
        #     if verbose:
        #         print(f"[WARN] level3_encoding shape invalid for {protein_id}: {a_token.shape}")
        #     return index, item if not drop_missing else None, "invalid_level3_shape"

        # if atom_to_token_idx.dim() > 1:
        #     atom_to_token_idx_flat = atom_to_token_idx.reshape(-1)
        # else:
        #     atom_to_token_idx_flat = atom_to_token_idx
        # n_token = int(atom_to_token_idx_flat.max().item() + 1)

        # num_residues = None
        # positions = item.get("positions", {})
        # if "x0" in positions:
        #     num_residues = len(positions["x0"])
        # elif "x" in positions:
        #     num_residues = len(positions["x"])
        # if num_residues is not None and n_token != num_residues:
        #     if verbose:
        #         print(
        #             f"[WARN] N_token({n_token}) != num_residues({num_residues}) for {protein_id}"
        #         )
        #     return index, item if not drop_missing else None, "token_residue_mismatch"

        # if a_token.shape[-2] != atom_to_token_idx.shape[-1]:
        #     if verbose:
        #         print(
        #             f"[WARN] atom count mismatch in bihito/feature_cache for {protein_id}: "
        #             f"{a_token.shape[-2]} vs {atom_to_token_idx.shape[-1]}"
        #         )
        #     return index, item if not drop_missing else None, "bihito_feature_atom_mismatch"

        # a_token = aggregate_atom_to_token(
        #     x_atom=a_token,
        #     atom_to_token_idx=atom_to_token_idx,
        #     n_token=n_token,
        #     reduce="mean",
        # )

        # if truncate_to_expected_n and expected_n < a_token.shape[-2]:
        #     if verbose:
        #         print(
        #             f"[WARN] a_token atom count larger than expected for {protein_id}: "
        #             f"{a_token.shape[-2]} -> truncate to {expected_n}"
        #         )
        #     a_token = a_token[..., :expected_n, :]

        # item["bihito_token_repr"] = a_token.detach().cpu()

        if sample_num_frames is not None:
            try:
                coords = subsample_trajectory(
                    coords, num_frames=sample_num_frames, total_length=sample_total_length
                )
            except Exception as e:
                if verbose:
                    print(f"[WARN] Subsample failed for {protein_id}: {e}")
                return index, item if not drop_missing else None, "subsample_failed"

        if use_fft_windows:
            if hop_ratio != 1.0:
                print(f"[WARN] use_fft_windows expects hop_ratio=1.0 for simple concatenation, got {hop_ratio}")
            (
                x_windows,
                y_windows,
                z_windows,
                frame0_windows,
                preprocessed_windows,
                preprocess_states,
            ) = compute_windowed_full_fft_positions(coords, dt=dt, window_size=window_size, hop_ratio=hop_ratio)
            item["positions"]["x_fft_windows"] = x_windows
            item["positions"]["y_fft_windows"] = y_windows
            item["positions"]["z_fft_windows"] = z_windows
            item["positions"]["x0_fft_windows"] = [f[:, 0] for f in frame0_windows]
            item["positions"]["y0_fft_windows"] = [f[:, 1] for f in frame0_windows]
            item["positions"]["z0_fft_windows"] = [f[:, 2] for f in frame0_windows]
            item["positions"]["coords_preprocessed_fft_windows"] = preprocessed_windows
            item["positions"]["preprocess_state_fft_windows"] = preprocess_states
            item["positions"]["fft_window_size"] = window_size
            item["positions"]["fft_hop_ratio"] = hop_ratio
            item["positions"]["num_fft_windows"] = len(x_windows)
        else:
            (
                x_list,
                y_list,
                z_list,
                frame0,
                coords_preprocessed,
                preprocess_state,
            ) = compute_full_rfft_positions(coords, dt=dt)
            item["positions"]["x"] = x_list
            item["positions"]["y"] = y_list
            item["positions"]["z"] = z_list
            item["positions"]["x0"] = frame0[:, 0].astype(np.float32, copy=False)
            item["positions"]["y0"] = frame0[:, 1].astype(np.float32, copy=False)
            item["positions"]["z0"] = frame0[:, 2].astype(np.float32, copy=False)
            item["positions"]["coords_preprocessed"] = coords_preprocessed
            item["positions"]["preprocess_state"] = preprocess_state

        item["positions"]["dt"] = float(dt)

        return index, item, "ok"

    if num_workers is None:
        num_workers = min(32, (os.cpu_count() or 1))
    if num_workers <= 1:
        results = [
            _process_one(i, item)
            for i, item in tqdm(enumerate(data_list), total=len(data_list), desc="Updating spectra")
        ]
    else:
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(_process_one, i, item) for i, item in enumerate(data_list)]
            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc="Updating spectra",
            ):
                results.append(future.result())

    results.sort(key=lambda x: x[0])
    updated = [item for _, item, _ in results if item is not None]
    reason_counts: Dict[str, int] = {}
    for _, item, reason in results:
        if item is None:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    if reason_counts:
        print("[SUMMARY] Dropped counts by reason:")
        for reason, count in sorted(reason_counts.items(), key=lambda x: (-x[1], x[0])):
            print(f"  - {reason}: {count}")
    return updated


def parse_args():
    parser = argparse.ArgumentParser(
        description="Replace positions with full (non-truncated) FFT spectra for ATLAS data lists."
    )
    parser.add_argument(
        "--data_list_paths",
        nargs="+",
        default=[
            "./dataset/atlas_10_features_fouriered_list_train.pth",
            "./dataset/atlas_10_features_fouriered_list_val.pth",
            "./dataset/atlas_10_features_fouriered_list_test.pth",
        ],
        help="Paths to data_list .pth files to update in-place.",
    )
    parser.add_argument(
        "--xtc_base_dir",
        type=str,
        default="./data/trajectories_xtcs/atlas",
        help="Directory containing *_gt.pdb and *_gt.xtc files.",
    )
    parser.add_argument(
        "--atom_selection",
        type=str,
        default="name CA",
        help="MDAnalysis selection string for atoms to use.",
    )
    parser.add_argument(
        "--fallback_selection",
        type=str,
        default=None,
        help="Fallback MDAnalysis selection if atom count mismatches.",
    )
    parser.add_argument(
        "--bihito_repr_dir",
        type=str,
        default="./dataset/bihito_repr_cache/atlas",
        help="Directory containing BiHiTo repr .pt files.",
    )
    parser.add_argument(
        "--feature_cache_dir",
        type=str,
        default="./dataset/feature_cache/atlas",
        help="Directory containing feature_cache .pt files (for atom_to_token_idx).",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=None,
        help="Time step in seconds (float). Overrides --dt_str when provided.",
    )
    parser.add_argument(
        "--dt_str",
        type=str,
        default="10ps",
        help="Time step string like '10ps' or '1ns'. Used when --dt is not set.",
    )
    parser.add_argument(
        "--output_suffix",
        type=str,
        default="",
        help="Suffix to add before .pth when saving (empty = overwrite).",
    )
    parser.add_argument(
        "--save_new",
        action="store_true",
        help="Save to a new .pth file instead of overwriting the original.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="",
        help="Explicit output path when using --save_new (optional).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Output directory when using --save_new; keeps original filenames.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print warnings for missing/mismatched files.",
    )
    parser.add_argument(
        "--use_fft_windows",
        action="store_true",
        help="Use windowed full FFT spectra (no COLA; windows concatenated).",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=200,
        help="STFT window size (default: 200).",
    )
    parser.add_argument(
        "--hop_ratio",
        type=float,
        default=0.5,
        help="STFT hop ratio (default: 0.5).",
    )
    parser.add_argument(
        "--sample_num_frames",
        type=int,
        default=None,
        help="Uniformly subsample to this many frames before FFT/STFT (optional).",
    )
    parser.add_argument(
        "--sample_total_length",
        type=int,
        default=None,
        help="Expected total length for sampling stride (optional).",
    )
    parser.add_argument(
        "--keep_missing",
        action="store_true",
        help="Keep samples without xtc/pdb or with atom mismatch (default: drop).",
    )
    parser.add_argument(
        "--truncate_to_expected_n",
        action="store_true",
        help="When expected_n is smaller than XTC or bihito sizes, truncate to expected_n.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dt = float(args.dt) if args.dt is not None else parse_time_str_to_seconds(args.dt_str)

    for data_list_path in args.data_list_paths:
        if not os.path.exists(data_list_path):
            print(f"[WARN] data_list not found: {data_list_path}")
            continue

        print(f"Loading data_list: {data_list_path}")
        data_list = torch.load(data_list_path, weights_only=False)
        print(f"Loaded {len(data_list)} items")

        updated = replace_positions_in_data_list(
            data_list,
            xtc_base_dir=args.xtc_base_dir,
            atom_selection=args.atom_selection,
            fallback_selection=args.fallback_selection,
            bihito_repr_dir=args.bihito_repr_dir,
            feature_cache_dir=args.feature_cache_dir,
            dt=dt,

            use_fft_windows=args.use_fft_windows,
            window_size=args.window_size,
            hop_ratio=args.hop_ratio,
            sample_num_frames=args.sample_num_frames,
            sample_total_length=args.sample_total_length,
            truncate_to_expected_n=args.truncate_to_expected_n,
            drop_missing=not args.keep_missing,
            verbose=args.verbose,
            num_workers=1,
        )

        if args.save_new:
            if args.output_dir:
                os.makedirs(args.output_dir, exist_ok=True)
                base_name = os.path.basename(data_list_path)
                out_path = os.path.join(args.output_dir, base_name)
            elif args.output_path:
                if len(args.data_list_paths) > 1 and not os.path.isdir(args.output_path):
                    raise ValueError(
                        "--output_path must be a directory when multiple data_list_paths are provided. "
                        "Use --output_dir instead."
                    )
                if len(args.data_list_paths) > 1 and os.path.isdir(args.output_path):
                    os.makedirs(args.output_path, exist_ok=True)
                    base_name = os.path.basename(data_list_path)
                    out_path = os.path.join(args.output_path, base_name)
                else:
                    out_path = args.output_path
            else:
                base, ext = os.path.splitext(data_list_path)
                suffix = args.output_suffix if args.output_suffix else "_updated"
                out_path = f"{base}{suffix}{ext}"
        else:
            if args.output_suffix:
                base, ext = os.path.splitext(data_list_path)
                out_path = f"{base}{args.output_suffix}{ext}"
            else:
                out_path = data_list_path

        print(f"Saving updated data_list: {out_path}")
        torch.save(updated, out_path)


if __name__ == "__main__":
    main()
