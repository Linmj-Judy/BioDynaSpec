"""Coordinate-space metrics after inverse FFT and preprocessing reversal."""

from __future__ import annotations

from typing import Any

import torch

from .trajectory import framewise_kabsch_rmsd_angstrom


def coordinate_metrics(
    processor: Any,
    pred_real: torch.Tensor,
    pred_imag: torch.Tensor,
    *,
    target_coords_processed: torch.Tensor,
    scaling_factors: torch.Tensor | None,
    dc_residual: torch.Tensor | None,
    preprocess_state: dict[str, Any],
    start: int,
    stop: int,
    window_size: int,
) -> tuple[dict[str, float], torch.Tensor]:
    """Return processed/physical MSE and framewise Kabsch RMSD."""
    from src.inference_spectrum_vae import restore_physical_coords, safe_irfft_window

    pred_processed = safe_irfft_window(
        processor,
        pred_real,
        pred_imag,
        window_size,
        scaling_factors,
        dc_residual,
    )
    pred_processed = pred_processed[..., : target_coords_processed.shape[-1]]
    processed_mse = torch.mean(
        (pred_processed.float() - target_coords_processed.float()) ** 2
    )
    pred_physical = restore_physical_coords(
        processor, pred_processed, preprocess_state, start=start, end=stop
    )
    target_physical = restore_physical_coords(
        processor, target_coords_processed, preprocess_state, start=start, end=stop
    )
    physical_mse = torch.mean(
        (pred_physical.float() - target_physical.float()) ** 2
    )
    pred_tn3 = pred_physical
    target_tn3 = target_physical
    if pred_tn3.ndim != 3 or target_tn3.ndim != 3:
        raise ValueError(
            "Expected one window of physical coordinates with rank 3, got "
            f"{tuple(pred_tn3.shape)} and {tuple(target_tn3.shape)}"
        )
    if pred_tn3.shape[-1] != 3:
        pred_tn3 = pred_tn3.permute(2, 0, 1).contiguous()
        target_tn3 = target_tn3.permute(2, 0, 1).contiguous()
    rmsd = framewise_kabsch_rmsd_angstrom(
        pred_tn3.detach().float().cpu().numpy(),
        target_tn3.detach().float().cpu().numpy(),
    )
    return {
        "processed_coordinate_mse": float(processed_mse.detach().cpu()),
        "physical_coordinate_mse": float(physical_mse.detach().cpu()),
        "framewise_kabsch_rmsd_angstrom_mean": float(rmsd.mean()),
        "framewise_kabsch_rmsd_angstrom_max": float(rmsd.max()),
        "framewise_kabsch_rmsd_angstrom_values": rmsd.tolist(),
    }, pred_physical.detach()
