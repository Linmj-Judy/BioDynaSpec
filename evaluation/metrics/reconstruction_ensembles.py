"""Adapt analyze_ensembles to cached reconstructions without ATLAS file reads.

Preserves its atom selections: all cached heavy atoms for RMSF, atomic EMD
and SASA; C-alpha for PCA, contacts and pairwise RMSD. All-atom fits to target
frame zero precede the C-alpha fit. Sampling is paired and uniform rather
than the original CLI's bootstrap, to compare the two checkpoints fairly.
"""
import argparse
import json
from pathlib import Path

import mdtraj as md
import numpy as np
import torch

from evaluation.metrics.analyze_ensembles import analyze_ensemble_trajectories
from src.data.basic.data_pipeline import DataPipeline


def _json_default(value):
    """Serialize NumPy scalars/arrays emitted by mdtraj and PCA helpers."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def cached_topology(atoms):
    """Preserve cached atom order; SASA requires elements, but not bonds."""
    top = md.Topology()
    chains, residues = {}, {}
    for a in atoms:
        c = str(a.chain_id)
        if c not in chains:
            chains[c] = top.add_chain(c)
        key = (c, int(a.res_id), str(a.ins_code), str(a.res_name))
        if key not in residues:
            residues[key] = top.add_residue(str(a.res_name), chains[c], resSeq=int(a.res_id))
        top.add_atom(str(a.atom_name), md.element.get_by_symbol(str(a.element)), residues[key])
    return top


def analyze_ensembles_compat_metrics(prediction, reference, atoms, *, sample_frames=256):
    """Adapt cached arrays to the shared :mod:`analyze_ensembles` metric core.

    This function deliberately contains no metric formulas.  It only converts
    the cached Angstrom arrays and AtomArray topology into mdtraj objects;
    ``analyze_ensemble_trajectories`` owns the complete metric implementation.
    """
    prediction = np.asarray(prediction, dtype=np.float32)
    reference = np.asarray(reference, dtype=np.float32)
    if prediction.shape != reference.shape:
        raise ValueError("prediction and reference shapes must match")
    top = cached_topology(atoms)
    return analyze_ensemble_trajectories(
        md.Trajectory(prediction / 10, top),
        md.Trajectory(reference / 10, top),
        sample_frames=sample_frames,
        source="evaluation.metrics.analyze_ensembles",
    )


def reconstruction_ensemble_metrics(prediction, reference, atoms, *, sample_frames=256):
    """Return the compact table derived from the shared full metric pass."""
    full = analyze_ensembles_compat_metrics(
        prediction, reference, atoms, sample_frames=sample_frames
    )
    summary = full["summary"]
    return {
        "num_heavy_atoms": full["num_heavy_atoms"],
        "num_ca": full["num_ca"],
        "num_frames": full["num_frames"],
        "pca2_wasserstein_reference_angstrom": full["pca2_wasserstein_reference_angstrom"],
        "pca2_wasserstein_prediction_angstrom": full["pca2_wasserstein_prediction_angstrom"],
        "pca2_wasserstein_joint_angstrom": full["pca2_wasserstein_joint_angstrom"],
        "pc1_cosine": full["cosine_sim"],
        "pc1_absolute_cosine": full["cosine_sim_absolute"],
        "reference_pca_variance_first10_angstrom2": full["ref_variance"][:10],
        "prediction_pca_variance_first10_angstrom2": full["af_variance"][:10],
        "ca_pairwise_rmsd_reference_angstrom": full["ref_mean_pairwise_rmsd"],
        "ca_pairwise_rmsd_prediction_angstrom": full["af_mean_pairwise_rmsd"],
        "ca_contact_probability_mae": summary["ca_contact_probability_mae"],
        "atomic_mean_emd_angstrom": summary["all_heavy_atomic_mean_emd_angstrom"],
        "atomic_covariance_emd_angstrom": summary["all_heavy_atomic_covariance_emd_angstrom"],
        "all_atom_rmsf_mae_angstrom": summary["all_heavy_rmsf_mae_angstrom"],
        "all_atom_rmsf_rmse_angstrom": summary["all_heavy_rmsf_rmse_angstrom"],
        "all_atom_rmsf_reference_mean_angstrom": float(np.asarray(full["ref_rmsf"]).mean()),
        "all_atom_rmsf_prediction_mean_angstrom": float(np.asarray(full["af_rmsf"]).mean()),
        "sidechain_exposure_probability_mae": summary["sidechain_exposure_probability_mae"],
        "sidechain_sasa_mi_mae_nats": summary["sidechain_sasa_mi_mae_nats"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['control', 'protenix', 'feature-cache', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--sample-frames', type=int, default=256)
    args = parser.parse_args()
    names = sorted(p.name for p in args.control.glob('*_trajectory.pt'))
    if not names or set(names) != {p.name for p in args.protenix.glob('*_trajectory.pt')}:
        raise ValueError('Both arms must contain the same nonempty sample set')
    out = {'protocol': {'atom_selection': 'analyze_ensembles: all-heavy RMSF/EMD/SASA; CA PCA/contacts/pairwise',
                       'alignment': 'independent all-heavy fit to target frame 0; additional CA fit for CA metrics',
                       'topology': 'cached AtomArray, atom order preserved',
                       'sampling': f'{args.sample_frames} uniform paired frames (not bootstrap)',
                       'scope': 'PARETO exploratory reconstruction, not ATLAS formal test',
                       'covariance': 'PSD-stable evaluation of Gaussian Wasserstein trace'}}
    for name in names:
        cache_path = args.feature_cache / name.replace('_trajectory.pt', '.pt')
        _, atoms, _, ok = DataPipeline.load_cached_feature(
            str(cache_path.parent), cache_path.stem
        )
        if not ok:
            raise ValueError(f'Invalid feature cache: {cache_path}')
        target = None
        for arm in ['control', 'protenix']:
            record = torch.load(getattr(args, arm) / name, map_location='cpu', weights_only=False, mmap=True)
            count = len(record['pred_coords'])
            idx = np.linspace(0, count - 1, min(args.sample_frames, count)).astype(int)
            ref = record['target_coords'][idx].float().numpy()
            if target is not None and not np.array_equal(target, ref):
                raise ValueError('Target coordinates differ between branches')
            target = ref
            pred_sample = record['pred_coords'][idx].float().numpy()
            compact = reconstruction_ensemble_metrics(pred_sample, ref, atoms)
            compact['analyze_ensembles_compat'] = analyze_ensembles_compat_metrics(
                pred_sample, ref, atoms, sample_frames=0)
            out.setdefault(arm, {})[name] = compact
            args.output.write_text(json.dumps(out, indent=2, allow_nan=False, default=_json_default))
            print(arm, name, flush=True)
            del record


if __name__ == '__main__':
    main()
