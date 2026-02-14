"""
Compute per-frequency scaling factors for FFT-window spectra.

Data format reference: inferencer_far.py
- positions["x_fft_windows"/"y_fft_windows"/"z_fft_windows"]
  each is a list of windows; each window is a list of residues;
  each residue is an array shaped [F, 2] (real, imag).

We compute magnitude = sqrt(real^2 + imag^2) for x/y/z, pool all samples,
and take the percentile (default 95) per frequency as scaling factor.
"""

import os
import argparse
from typing import List

import torch
import numpy as np
import concurrent.futures


def _iter_items(data_obj):
    if isinstance(data_obj, dict):
        if "data_list" in data_obj and isinstance(data_obj["data_list"], (list, tuple)):
            for item in data_obj["data_list"]:
                yield item
            return
        if "positions" in data_obj:
            yield data_obj
            return
        raise TypeError(f"Unsupported dict format with keys: {list(data_obj.keys())}")
    if isinstance(data_obj, (list, tuple)):
        for item in data_obj:
            yield item
        return
    raise TypeError(f"Unsupported data type: {type(data_obj)}")


def iter_fft_window_magnitudes(positions: dict):
    x_windows = positions.get("x_fft_windows", [])
    y_windows = positions.get("y_fft_windows", [])
    z_windows = positions.get("z_fft_windows", [])
    if not x_windows or not y_windows or not z_windows:
        return
    num_windows = min(len(x_windows), len(y_windows), len(z_windows))
    for w in range(num_windows):
        x_w = x_windows[w]
        y_w = y_windows[w]
        z_w = z_windows[w]
        num_residues = min(len(x_w), len(y_w), len(z_w))
        for r in range(num_residues):
            for dim_hat in (x_w[r], y_w[r], z_w[r]):
                arr = np.asarray(dim_hat)
                if arr.ndim != 2 or arr.shape[1] != 2:
                    raise ValueError(f"Unexpected spectrum shape: {arr.shape}")
                mag = np.sqrt(arr[:, 0] ** 2 + arr[:, 1] ** 2)
                yield mag


def _get_scale_for_length(scale: np.ndarray, length: int) -> np.ndarray:
    if length <= len(scale):
        return scale[:length]
    pad = np.full(length - len(scale), scale[-1], dtype=scale.dtype)
    return np.concatenate([scale, pad])


def normalize_fft_windows_in_item(item: dict, scale: np.ndarray) -> bool:
    positions = item.get("positions", {})
    x_windows = positions.get("x_fft_windows")
    y_windows = positions.get("y_fft_windows")
    z_windows = positions.get("z_fft_windows")
    if not x_windows or not y_windows or not z_windows:
        return False

    num_windows = min(len(x_windows), len(y_windows), len(z_windows))
    for w in range(num_windows):
        x_w = x_windows[w]
        y_w = y_windows[w]
        z_w = z_windows[w]
        num_residues = min(len(x_w), len(y_w), len(z_w))
        for r in range(num_residues):
            for dim_list in (x_w, y_w, z_w):
                arr = np.asarray(dim_list[r], dtype=np.float32)
                if arr.ndim != 2 or arr.shape[1] != 2:
                    raise ValueError(f"Unexpected spectrum shape: {arr.shape}")
                scale_used = _get_scale_for_length(scale, arr.shape[0])
                arr[:, 0] = arr[:, 0] / scale_used
                arr[:, 1] = arr[:, 1] / scale_used
                dim_list[r] = arr
    return True


def _process_item(item, path, num_freq):
    """Helper to process one item for parallel execution."""
    if not isinstance(item, dict):
        return None, f"[WARN] skip non-dict item in {path}: {type(item)}"
    positions = item.get("positions", {})
    item_freq_mags = [[] for _ in range(num_freq)]
    item_yielded = 0
    try:
        for mag in iter_fft_window_magnitudes(positions):
            if len(mag) < num_freq:
                raise ValueError(
                    f"Frequency length {len(mag)} < expected {num_freq} in {path}"
                )
            for f in range(num_freq):
                item_freq_mags[f].append(float(mag[f]))
            item_yielded += 1
    except Exception as e:
        return None, f"[WARN] error in {path} ({type(e).__name__}): {e}"
    return item_freq_mags, None if item_yielded > 0 else "[WARN] no valid fft_window_magnitudes"

def compute_scaling_factors_from_dir(
    data_dir: str,
    percentile: float,
    num_freq: int,
):
    import threading
    # Fix for numpy + threads and PyTorch openblas 
    torch.set_num_threads(1)

    filenames = [
        "atlas_10_features_fouriered_list_train.pth",
        "atlas_10_features_fouriered_list_val.pth",
        "atlas_10_features_fouriered_list_test.pth",
    ]
    pth_files = [os.path.join(data_dir, name) for name in filenames]
    missing = [p for p in pth_files if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"Missing required .pth files: {missing}")

    freq_magnitudes: List[List[float]] = [[] for _ in range(num_freq)]
    total_items = 0

    for path in pth_files:
        data_list = torch.load(path, weights_only=False)
        batch_items = list(_iter_items(data_list))
        # Use ThreadPoolExecutor for simple parallelism
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(32, (os.cpu_count() // 2 or 1))) as executor:
            futures = [
                executor.submit(_process_item, item, path, num_freq)
                for item in batch_items
            ]
            for fut in concurrent.futures.as_completed(futures):
                res, warn = fut.result()
                if res is not None:
                    for f in range(num_freq):
                        freq_magnitudes[f].extend(res[f])
                    total_items += 1
                elif warn:
                    print(warn)

    if total_items == 0:
        raise ValueError("No valid items found in dataset.")

    scaling_factors = np.zeros(num_freq, dtype=np.float32)
    for f in range(num_freq):
        if freq_magnitudes[f]:
            scaling_factors[f] = np.percentile(freq_magnitudes[f], percentile)
        else:
            scaling_factors[f] = 1.0

    return torch.tensor(scaling_factors, dtype=torch.float32)


def normalize_dataset_files(
    data_dir: str,
    scale_tensor: torch.Tensor,
    output_suffix: str,
):
    scale = scale_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
    filenames = [
        "atlas_10_features_fouriered_list_train.pth",
        "atlas_10_features_fouriered_list_val.pth",
        "atlas_10_features_fouriered_list_test.pth",
    ]
    for name in filenames:
        path = os.path.join(data_dir, name)
        if not os.path.exists(path):
            print(f"[WARN] missing file: {path}")
            continue
        data_list = torch.load(path, weights_only=False)
        updated = 0
        for item in _iter_items(data_list):
            if not isinstance(item, dict):
                print(f"[WARN] skip non-dict item in {path}: {type(item)}")
                continue
            if normalize_fft_windows_in_item(item, scale):
                updated += 1
        base, ext = os.path.splitext(path)
        out_path = f"{base}{output_suffix}{ext}"
        torch.save(data_list, out_path)
        print(f"Saved normalized: {out_path} (updated_items={updated}/{len(data_list)})")


def main():
    default_dir = "./dataset/fft_window5000"
    default_output = os.path.join(default_dir, "fft_window5000_scaling_factors.pth")

    parser = argparse.ArgumentParser(description="Compute scaling factors for fft_window5000 dataset")
    parser.add_argument("--data_dir", type=str, default=default_dir, help="Directory containing .pth files")
    parser.add_argument("--num_freq", type=int, default=2501, help="Number of frequency components")
    parser.add_argument("--percentile", type=float, default=95.0, help="Percentile for scaling factors")
    parser.add_argument("--output_path", type=str, default=default_output, help="Output .pth path")
    parser.add_argument("--normalize", action="store_true", help="Normalize train/valid/test with scaling factors")
    parser.add_argument("--output_suffix", type=str, default="_normalized", help="Suffix for normalized .pth files")
    args = parser.parse_args()

    data_dir = os.path.abspath(os.path.expanduser(os.path.expandvars(args.data_dir)))
    output_path = os.path.abspath(os.path.expanduser(os.path.expandvars(args.output_path)))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if os.path.exists(output_path):
        scaling = torch.load(output_path, weights_only=False)
    else:
        scaling = compute_scaling_factors_from_dir(
            data_dir=data_dir,
            percentile=args.percentile,
            num_freq=args.num_freq,
        )
        torch.save(scaling, output_path)
        print(f"Saved scaling factors: {output_path}")
    print(f"shape={tuple(scaling.shape)}, dtype={scaling.dtype}")
    print(f"first10={scaling[:10]}")
    print(f"last10={scaling[-10:]}")

    if args.normalize:
        normalize_dataset_files(
            data_dir=data_dir,
            scale_tensor=scaling,
            output_suffix=args.output_suffix,
        )


if __name__ == "__main__":
    main()
