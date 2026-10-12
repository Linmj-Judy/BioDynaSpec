"""Trajectory metrics with explicit alignment semantics.

The functions in this module keep alignment policy separate from model and
decoder code.  In particular, framewise Kabsch RMSD and single-transform
trajectory RMSD answer different scientific questions and must not be mixed.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _validate_trajectory_pair(
    prediction: np.ndarray, reference: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    prediction = np.asarray(prediction, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if prediction.shape != reference.shape or prediction.ndim != 3:
        raise ValueError(
            "Trajectory inputs must have identical (T, N_atom, 3) shapes, got "
            f"{prediction.shape} and {reference.shape}"
        )
    if prediction.shape[-1] != 3:
        raise ValueError(f"Trajectory inputs must end in xyz=3, got {prediction.shape}")
    return prediction, reference


def _kabsch_transform(
    source: np.ndarray, target: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``source @ rotation + translation`` aligned to ``target``."""
    source_center = source.mean(axis=0, keepdims=True)
    target_center = target.mean(axis=0, keepdims=True)
    source_centered = source - source_center
    target_centered = target - target_center
    u, _, vt = np.linalg.svd(source_centered.T @ target_centered)
    correction = np.eye(3)
    if np.linalg.det(u @ vt) < 0:
        correction[-1, -1] = -1.0
    rotation = u @ correction @ vt
    translation = target_center - source_center @ rotation
    return rotation, translation


def framewise_kabsch_rmsd_angstrom(
    prediction: np.ndarray, reference: np.ndarray
) -> np.ndarray:
    """Compute RMSD after an independent optimal rigid fit for every frame."""
    prediction, reference = _validate_trajectory_pair(prediction, reference)
    values = np.empty(prediction.shape[0], dtype=np.float32)
    for frame_idx, (pred_frame, ref_frame) in enumerate(zip(prediction, reference)):
        rotation, translation = _kabsch_transform(pred_frame, ref_frame)
        delta = pred_frame @ rotation + translation - ref_frame
        values[frame_idx] = np.sqrt(np.mean(np.sum(delta * delta, axis=-1)))
    return values


def global_alignment_rmsd_angstrom(
    prediction: np.ndarray, reference: np.ndarray
) -> np.ndarray:
    """Apply one frame-0 Kabsch transform to the complete prediction trajectory."""
    prediction, reference = _validate_trajectory_pair(prediction, reference)
    rotation, translation = _kabsch_transform(prediction[0], reference[0])
    aligned = prediction @ rotation + translation
    return np.sqrt(np.mean(np.sum((aligned - reference) ** 2, axis=-1), axis=-1)).astype(
        np.float32
    )


def rmsf(coords: np.ndarray) -> np.ndarray:
    """Return per-atom RMSF for a ``(T, N_atom, 3)`` trajectory."""
    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 3 or coords.shape[-1] != 3:
        raise ValueError(f"Expected (T, N_atom, 3) coordinates, got {coords.shape}")
    centered = coords - coords.mean(axis=0, keepdims=True)
    return np.sqrt(np.mean(np.sum(centered * centered, axis=-1), axis=0))


def target_reference_rmsf_metrics(
    prediction: np.ndarray, reference: np.ndarray
) -> dict[str, float]:
    """Compare RMSF after target-derived frame-0 trajectory alignment.

    Each target frame is aligned to target frame 0.  The same target-derived
    transform is applied to the corresponding prediction frame, preserving
    motion errors while removing target trajectory-wide rigid-body drift.
    """
    prediction, reference = _validate_trajectory_pair(prediction, reference)
    target_reference = reference[0]
    aligned_prediction = []
    aligned_reference = []
    for pred_frame, ref_frame in zip(prediction, reference):
        rotation, translation = _kabsch_transform(ref_frame, target_reference)
        aligned_prediction.append(pred_frame @ rotation + translation)
        aligned_reference.append(ref_frame @ rotation + translation)
    pred_rmsf = rmsf(np.stack(aligned_prediction))
    ref_rmsf = rmsf(np.stack(aligned_reference))
    diff = pred_rmsf - ref_rmsf
    pred_centered = pred_rmsf - pred_rmsf.mean()
    ref_centered = ref_rmsf - ref_rmsf.mean()
    denominator = np.linalg.norm(pred_centered) * np.linalg.norm(ref_centered)
    return {
        "aligned_rmsf_mean_prediction": float(pred_rmsf.mean()),
        "aligned_rmsf_mean_reference": float(ref_rmsf.mean()),
        "aligned_rmsf_mae": float(np.mean(np.abs(diff))),
        "aligned_rmsf_rmse": float(np.sqrt(np.mean(diff**2))),
        "aligned_rmsf_bias": float(diff.mean()),
        "aligned_rmsf_pearson": (
            float(pred_centered.dot(ref_centered) / denominator)
            if denominator
            else float("nan")
        ),
    }


def summarize_rmsd(values: np.ndarray) -> dict[str, Any]:
    """Summarize per-frame RMSD values with explicit Angstrom units."""
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError("Cannot summarize an empty RMSD array")
    return {
        "mean_angstrom": float(values.mean()),
        "median_angstrom": float(np.median(values)),
        "p95_angstrom": float(np.quantile(values, 0.95)),
        "max_angstrom": float(values.max()),
        "num_frames": int(values.size),
    }
