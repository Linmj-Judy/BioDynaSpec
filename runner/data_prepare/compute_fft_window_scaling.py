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

from streaming_pth import StreamingPthWriter
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
    """Return this item's per-frequency magnitudes as `(num_mags, num_freq)`.

    Vectorised: the previous version built `num_freq` Python lists per item and
    appended one Python float per element, i.e. ~1251 lists and ~1.9M float
    conversions per protein (~125k x 1251 = 156M for the full split), then moved
    them through the thread pool. That cost dominated the run (measured ~4.4 min
    per 10 GB shard, ~4.4 h for the split). One `np.stack` + one `__getitem__`
    over the frequency axis does the same work in C.
    """
    if not isinstance(item, dict):
        return None, f"[WARN] skip non-dict item in {path}: {type(item)}"
    positions = item.get("positions", {})
    try:
        mags = list(iter_fft_window_magnitudes(positions))
        if not mags:
            return None, "[WARN] no valid fft_window_magnitudes"
        # All magnitudes in one item share the frequency length; a ragged item
        # (different length per window) is truncated to the shortest, which is
        # what `len(mag) < num_freq` used to reject outright.
        m = min(len(x) for x in mags)
        if m < num_freq:
            raise ValueError(
                f"Frequency length {m} < expected {num_freq} in {path}"
            )
        arr = np.stack([np.asarray(x[:num_freq], dtype=np.float32) for x in mags])
    except Exception as e:
        return None, f"[WARN] error in {path} ({type(e).__name__}): {e}"
    return arr, None

def _split_inputs(data_dir, split):
    """Input .pth files for `split`: every shard, or the single legacy file.

    Sharded splits are what keep this script inside memory: it needs the whole
    split's items, and one `torch.load` of the 638 GB train file would not fit
    the container's 1030 GB cgroup limit. Each shard is loaded independently.
    """
    import json
    d = os.path.join(data_dir, split)
    if os.path.isdir(d):
        mpath = os.path.join(d, "manifest.json")
        if os.path.exists(mpath):
            with open(mpath) as fh:
                names = json.load(fh)["shards"]
            return [os.path.join(d, n) for n in names]
        return sorted(
            os.path.join(d, f) for f in os.listdir(d) if f.endswith(".pth")
        )
    single = os.path.join(data_dir, f"atlas_10_features_fouriered_list_{split}.pth")
    return [single] if os.path.exists(single) else []


def _iter_items_stream(path):
    """Iterate one input file's items.

    Loaded whole: a pickled list's elements share a single STOP opcode, so
    `Unpickler.load()` always returns the entire list and cannot yield items one
    at a time (verified directly -- the second `load()` raises EOFError). Peak
    memory is therefore one FILE, which is why callers pass shards rather than a
    whole split.
    """
    yield from _iter_items(torch.load(path, weights_only=False))


def _shard_magnitudes(args):
    """Return `(num_items, (num_mags, num_freq))` for one shard file.

    Top-level so a ProcessPoolExecutor can pickle it. Only the small magnitude
    array crosses the process boundary -- never the shard's coordinates -- so
    parallelising costs ~20 KB per protein instead of ~237 MB.
    """
    path, num_freq = args
    try:
        torch.set_num_threads(1)
    except Exception:
        pass
    parts, n_items = [], 0
    for item in _iter_items_stream(path):
        res, warn = _process_item(item, path, num_freq)
        if res is None:
            if warn:
                print(warn)
            continue
        parts.append(res)
        n_items += 1
    if not parts:
        return 0, None
    return n_items, np.concatenate(parts, axis=0)


def compute_scaling_factors_from_dir(
    data_dir: str,
    percentile: float,
    num_freq: int,
    fit_split: str = "train",
):
    """Per-frequency scaling factors s_f.

    Paper Sec. 4.2: "The normalization scaling factors s_f for each frequency are
    computed *solely using the training set*." So s_f is fitted on `fit_split`
    only; val/test are normalized with the same factors but never contribute to
    them. Fitting on all three splits (the previous behaviour) leaks test/val
    magnitudes into a quantity the model consumes.
    """
    import threading
    # Fix for numpy + threads and PyTorch openblas
    torch.set_num_threads(1)

    all_splits = ["train", "val", "test"]
    if fit_split not in all_splits:
        raise ValueError(f"fit_split must be one of {all_splits}, got {fit_split!r}")

    fit_paths = _split_inputs(data_dir, fit_split)
    if not fit_paths:
        raise FileNotFoundError(
            f"No input found for the {fit_split} split under {data_dir} "
            f"(looked for {fit_split}/ shards and the single-file layout)"
        )
    print(f"[s_f] fitting on the {fit_split} split only: {len(fit_paths)} file(s)")
    print(f"[s_f] val/test will be normalized with these factors but do NOT contribute")

    # Shards are independent and the per-shard result is small
    # (`num_windows * num_freq` floats, ~20 KB per protein), while the cost is
    # deserialising the shard (~10 GB) -- so the work parallelises cleanly. The
    # sequential loop used ~17% of one core and took ~1 min/shard.
    workers = max(1, min(len(fit_paths), (os.cpu_count() or 8) // 8))
    print(f"[s_f] scanning {len(fit_paths)} shard(s) with {workers} process(es)")

    per_item_blocks = []
    total_items = 0
    if workers > 1:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
            for n_items, block in ex.map(_shard_magnitudes,
                                         [(p, num_freq) for p in fit_paths]):
                if block is None:
                    continue
                per_item_blocks.append(block)
                total_items += n_items
    else:
        for p in fit_paths:
            n_items, block = _shard_magnitudes((p, num_freq))
            if block is None:
                continue
            per_item_blocks.append(block)
            total_items += n_items

    # One (num_mags, num_freq) array for the whole split; the percentiles below
    # then read straight along the frequency axis.
    all_mags = (np.concatenate(per_item_blocks, axis=0)
                if per_item_blocks else np.zeros((0, num_freq), dtype=np.float32))

    if total_items == 0:
        raise ValueError(f"No valid items found in the {fit_split} split.")

    scaling_factors = np.percentile(all_mags, percentile, axis=0).astype(np.float32)
    scaling_factors = np.nan_to_num(scaling_factors, nan=1.0, posinf=1.0, neginf=1.0)
    scaling_factors[scaling_factors == 0] = 1.0

    print(f"[s_f] fitted on {total_items} {fit_split} items")
    return torch.tensor(scaling_factors, dtype=torch.float32)


def normalize_dataset_files(
    data_dir: str,
    scale_tensor: torch.Tensor,
    output_suffix: str,
):
    scale = scale_tensor.detach().cpu().numpy().astype(np.float32, copy=False)
    # Streamed: `torch.load` + `torch.save` held the whole split resident, and
    # the train split is 638 GB against a 1030 GB cgroup limit -- the load alone
    # nearly fills it, and serialising a second copy on top OOM-killed the run.
    # Reading and writing one item at a time makes peak memory one item.
    import json

    for split in ("train", "val", "test"):
        inputs = _split_inputs(data_dir, split)
        if not inputs:
            continue
        # Write to a SIBLING directory, not next to the originals: the loader
        # reads `<dir>/manifest.json` and loads exactly the shards it lists, so
        # normalized shards dropped beside the originals would be ignored (the
        # manifest would still point at the un-normalized ones).
        out_dir = os.path.join(data_dir, f"{split}{output_suffix}")
        os.makedirs(out_dir, exist_ok=True)

        names, counts, win_counts, total_items, total_windows = [], [], [], 0, 0
        for path in inputs:
            name = os.path.basename(path)
            if name.endswith(f"{output_suffix}.pth"):
                continue                      # never re-normalize an output
            out_path = os.path.join(out_dir, f"{os.path.splitext(name)[0]}{output_suffix}.pth")
            updated = items = windows = 0
            # torch.load + torch.save held a whole split resident; one shard at
            # a time keeps peak memory at a single shard.
            with StreamingPthWriter(out_path) as writer:
                for item in _iter_items_stream(path):
                    items += 1
                    if isinstance(item, dict) and normalize_fft_windows_in_item(item, scale):
                        updated += 1
                    if isinstance(item, dict):
                        windows += int(item["positions"]["num_fft_windows"])
                    writer.append(item)
            names.append(os.path.basename(out_path))
            counts.append(items)
            win_counts.append(windows)
            total_items += items
            total_windows += windows
            print(f"Saved normalized: {out_path} (updated_items={updated}/{items})")

        manifest = {
            "num_shards": len(names),
            "num_proteins": total_items,
            "num_workers": 0,
            "counts": counts,
            "window_counts": win_counts,
            "num_windows": total_windows,
            "shards": names,
            "source": f"compute_fft_window_scaling.py from {split}/",
            "window_key": "x_fft_windows",
        }
        with open(os.path.join(out_dir, "manifest.json"), "w") as fh:
            json.dump(manifest, fh, indent=2)
        print(f"wrote {out_dir}/  ({len(names)} shards, {total_items} proteins, "
              f"{total_windows} windows)")


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
    parser.add_argument("--fit_split", type=str, default="train", choices=["train", "val", "test"],
                        help="Split used to fit s_f. Paper Sec. 4.2 requires 'train' (default).")
    parser.add_argument("--recompute_sf", action="store_true",
                        help="Recompute s_f even if the output file already exists. "
                             "Use this after changing --fit_split, otherwise a stale "
                             "(leaky) scaling_factors.pth is silently reused.")
    args = parser.parse_args()

    data_dir = os.path.abspath(os.path.expanduser(os.path.expandvars(args.data_dir)))
    output_path = os.path.abspath(os.path.expanduser(os.path.expandvars(args.output_path)))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if os.path.exists(output_path) and not args.recompute_sf:
        scaling = torch.load(output_path, weights_only=False)
        print(f"[s_f] reusing existing {output_path}")
        print(f"[s_f] NOTE: pass --recompute_sf to refit (e.g. after changing --fit_split).")
    else:
        scaling = compute_scaling_factors_from_dir(
            data_dir=data_dir,
            percentile=args.percentile,
            num_freq=args.num_freq,
            fit_split=args.fit_split,
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
