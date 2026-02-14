"""
Inspect GT spectrum data from atlas_10_features_fouriered_list_test.pth.
参考 inferencer_far.py 的数据结构:
- fft: positions["x"/"y"/"z"] -> list of [F, 2] (real/imag)
- stft: positions["x_windows"/"y_windows"/"z_windows"] -> list of windows, each window is list of [F, 2]
- fft_window: positions["x_fft_windows"/"y_fft_windows"/"z_fft_windows"] -> list of windows, each window is list of [F, 2]
"""

import os
import argparse
from typing import Dict, List

import torch
import numpy as np


def detect_data_mode(data_list: List[Dict]) -> str:
    for item in data_list:
        positions = item.get("positions", {})
        if positions.get("x_fft_windows") is not None:
            return "fft_window"
        if positions.get("x_windows") is not None:
            return "stft"
    return "fft"


def build_fft_spectrum(positions: Dict) -> torch.Tensor:
    spectrum_x_real = [item[:, 0] for item in positions["x"]]
    spectrum_x_imag = [item[:, 1] for item in positions["x"]]
    spectrum_y_real = [item[:, 0] for item in positions["y"]]
    spectrum_y_imag = [item[:, 1] for item in positions["y"]]
    spectrum_z_real = [item[:, 0] for item in positions["z"]]
    spectrum_z_imag = [item[:, 1] for item in positions["z"]]
    spectrum = np.stack(
        [
            spectrum_x_real,
            spectrum_x_imag,
            spectrum_y_real,
            spectrum_y_imag,
            spectrum_z_real,
            spectrum_z_imag,
        ],
        axis=-1,
    )  # [N, F, 6]
    return torch.tensor(spectrum, dtype=torch.float32)


def build_window_spectrum(positions: Dict, window_idx: int) -> torch.Tensor:
    x_windows = positions["x_windows"][window_idx]
    y_windows = positions["y_windows"][window_idx]
    z_windows = positions["z_windows"][window_idx]
    spectrum_x_real = [item[:, 0] for item in x_windows]
    spectrum_x_imag = [item[:, 1] for item in x_windows]
    spectrum_y_real = [item[:, 0] for item in y_windows]
    spectrum_y_imag = [item[:, 1] for item in y_windows]
    spectrum_z_real = [item[:, 0] for item in z_windows]
    spectrum_z_imag = [item[:, 1] for item in z_windows]
    spectrum = np.stack(
        [
            spectrum_x_real,
            spectrum_x_imag,
            spectrum_y_real,
            spectrum_y_imag,
            spectrum_z_real,
            spectrum_z_imag,
        ],
        axis=-1,
    )  # [N, F, 6]
    return torch.tensor(spectrum, dtype=torch.float32)


def build_fft_window_spectrum(positions: Dict, window_idx: int) -> torch.Tensor:
    x_windows = positions["x_fft_windows"][window_idx]
    y_windows = positions["y_fft_windows"][window_idx]
    z_windows = positions["z_fft_windows"][window_idx]
    spectrum_x_real = [item[:, 0] for item in x_windows]
    spectrum_x_imag = [item[:, 1] for item in x_windows]
    spectrum_y_real = [item[:, 0] for item in y_windows]
    spectrum_y_imag = [item[:, 1] for item in y_windows]
    spectrum_z_real = [item[:, 0] for item in z_windows]
    spectrum_z_imag = [item[:, 1] for item in z_windows]
    spectrum = np.stack(
        [
            spectrum_x_real,
            spectrum_x_imag,
            spectrum_y_real,
            spectrum_y_imag,
            spectrum_z_real,
            spectrum_z_imag,
        ],
        axis=-1,
    )  # [N, F, 6]
    return torch.tensor(spectrum, dtype=torch.float32)


def summarize_spectrum(
    spectrum: torch.Tensor,
    max_residues: int,
    max_freqs: int,
    label: str,
):
    n_res, n_freq, n_feat = spectrum.shape
    print(f"  {label}: shape={tuple(spectrum.shape)} (N={n_res}, F={n_freq}, C={n_feat})")
    max_res = min(max_residues, n_res)
    max_f = min(max_freqs, n_freq)
    for r in range(max_res):
        snippet = spectrum[r, :max_f].cpu().numpy()
        print(f"    residue[{r}] first {max_f} freqs:")
        print(snippet)


def main():
    # default_path = (
    #     "./"
    #     "dataset/stft_window5000/atlas_10_features_fouriered_list_test.pth"
    # )
    default_path = (
        "./"
        "dataset/atlas_10_features_fouriered_list_test.pth"
    )

    parser = argparse.ArgumentParser(description="Inspect GT spectrum data")
    parser.add_argument("--data_path", type=str, default=default_path, help="Path to .pth file")
    parser.add_argument("--num_proteins", type=int, default=2, help="Number of proteins to inspect")
    parser.add_argument("--num_windows", type=int, default=1, help="Number of windows to inspect (for stft/fft_window)")
    parser.add_argument("--max_residues", type=int, default=2, help="Max residues to print")
    parser.add_argument("--max_freqs", type=int, default=3, help="Max freqs to print per residue")
    args = parser.parse_args()

    data_path = os.path.abspath(os.path.expanduser(os.path.expandvars(args.data_path)))
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"data_path not found: {data_path}")

    data_list = torch.load(data_path, weights_only=False)
    print(f"Loaded: {data_path}")
    print(f"Total proteins: {len(data_list)}")
    if not data_list:
        return

    data_mode = detect_data_mode(data_list)
    print(f"Detected data_mode: {data_mode}")

    for idx, item in enumerate(data_list[: max(0, args.num_proteins)]):
        protein_id = item.get("id", f"sample_{idx}")
        positions = item.get("positions", {})
        print(f"\n[{idx}] protein_id={protein_id}")
        print(f"positions keys: {list(positions.keys())}")

        if data_mode == "fft":
            spectrum = build_fft_spectrum(positions)
            summarize_spectrum(spectrum, args.max_residues, args.max_freqs, "gt_fft_spectrum")
        elif data_mode == "stft":
            num_windows = len(positions.get("x_windows", []))
            if num_windows == 0:
                print("  no x_windows found")
                continue
            for w in range(min(args.num_windows, num_windows)):
                spectrum = build_window_spectrum(positions, w)
                summarize_spectrum(
                    spectrum,
                    args.max_residues,
                    args.max_freqs,
                    f"gt_stft_window[{w}]",
                )
        elif data_mode == "fft_window":
            num_windows = len(positions.get("x_fft_windows", []))
            if num_windows == 0:
                print("  no x_fft_windows found")
                continue
            for w in range(min(args.num_windows, num_windows)):
                spectrum = build_fft_window_spectrum(positions, w)
                summarize_spectrum(
                    spectrum,
                    args.max_residues,
                    args.max_freqs,
                    f"gt_fft_window[{w}]",
                )
        else:
            print(f"Unsupported data_mode: {data_mode}")


if __name__ == "__main__":
    main()
