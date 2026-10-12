"""Analyze reference and predicted molecular dynamics ensembles.

The module is import-safe: analysis starts only when :func:`run` or the
command-line entry point is called.
"""

from __future__ import annotations

import argparse
import os
import pickle
from multiprocessing import Pool
from functools import partial
from typing import Any

import mdtraj
import numpy as np
import tqdm
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA


def get_pca(xyz):
    traj_reshaped = xyz.reshape(xyz.shape[0], -1)
    pca = PCA(n_components=min(traj_reshaped.shape))
    coords = pca.fit_transform(traj_reshaped)
    return pca, coords

def get_rmsds(traj1, traj2, broadcast=False):
    n_atoms = traj1.shape[1]
    traj1 = traj1.reshape(traj1.shape[0], n_atoms * 3)
    traj2 = traj2.reshape(traj2.shape[0], n_atoms * 3)
    if broadcast:
        traj1, traj2 = traj1[:,None], traj2[None]
    distmat = np.square(traj1 - traj2).sum(-1)**0.5 / n_atoms**0.5 * 10
    return distmat

def condense_sidechain_sasas(sasas, top):
    assert top.n_residues > 1

    if top.n_atoms != sasas.shape[1]:
        raise ValueError(
            f"The number of atoms in top ({top.n_atoms}) didn't match the "
            f"number of SASAs provided ({sasas.shape[1]}). Make sure you "
            f"computed atom-level SASAs (mode='atom') and that you've passed "
            "the correct topology file and array of SASAs"
        )

    sc_mask = np.array([a.name not in ['CA', 'C', 'N', 'O', 'OXT'] for a in top.atoms])
    res_id = np.array([a.residue.index for a in top.atoms])
    
    rsd_sasas = np.zeros((sasas.shape[0], top.n_residues), dtype='float32')

    for i in range(top.n_residues):
        rsd_sasas[:, i] = sasas[:, sc_mask & (res_id == i)].sum(1)
    return rsd_sasas

def sasa_mi(sasa): 
    N, L = sasa.shape
    joint_probs = np.zeros((L, L, 2, 2))

    joint_probs[:,:,1,1] = (sasa[:,:,None] & sasa[:,None,:]).mean(0)
    joint_probs[:,:,1,0] = (sasa[:,:,None] & ~sasa[:,None,:]).mean(0)
    joint_probs[:,:,0,1] = (~sasa[:,:,None] & sasa[:,None,:]).mean(0)
    joint_probs[:,:,0,0] = (~sasa[:,:,None] & ~sasa[:,None,:]).mean(0)

    marginal_probs = np.stack([1-sasa.mean(0), sasa.mean(0)], -1)
    indep_probs = marginal_probs[None,:,None,:] * marginal_probs[:,None,:,None]
    mi = np.nansum(joint_probs * np.log(joint_probs / indep_probs), (-1, -2))
    mi[np.arange(L), np.arange(L)] = 0
    return mi

       
def get_mean_covar(xyz):
    mean = xyz.mean(0)
    xyz = xyz - mean
    covar = (xyz[...,None] * xyz[...,None,:]).mean(0)
    return mean, covar


def get_contact_probability(xyz, threshold=0.8):
    """Return the per-pair contact probability used by :func:`main`.

    ``xyz`` is in nm and the default threshold is therefore 0.8 nm, matching
    the original command-line analysis.
    """
    distmat = np.linalg.norm(xyz[:, :, None] - xyz[:, None, :], axis=-1)
    return (distmat < threshold).mean(0)


def get_atomic_emd(ref_xyz, af_xyz):
    """Return the original per-atom mean/covariance EMD diagnostics.

    Coordinates are expected in nm, as in the original ``main`` function;
    returned distances are in Angstrom.
    """
    ref_mean, ref_covar = get_mean_covar(ref_xyz)
    af_mean, af_covar = get_mean_covar(af_xyz)
    emd_mean = np.sqrt(np.square(ref_mean - af_mean).sum(-1)) * 10
    # Symmetric PSD form of the same Gaussian Wasserstein covariance term.
    # It avoids small complex round-off values from a generic matrix square
    # root while preserving the metric's intended definition.
    values, vectors = np.linalg.eigh(ref_covar)
    ref_root = (vectors * np.sqrt(np.maximum(values, 0))[:, None, :]) @ vectors.transpose(0, 2, 1)
    cross = ref_root @ af_covar @ ref_root
    cross_trace = np.sqrt(np.maximum(np.linalg.eigvalsh(cross), 0)).sum(-1)
    cov_distance = np.sqrt(
        np.maximum(
            np.trace(ref_covar + af_covar, axis1=-2, axis2=-1)
            - 2 * cross_trace,
            0,
        )
    )
    emd_var = cov_distance * 10
    return emd_mean, emd_var


def get_sasa_metrics(traj, probe_radius=0.28, exposure_threshold=0.02):
    """Return sidechain SASA probabilities and mutual information.

    The atom selection, probe radius, threshold, and MI definition are shared
    with the original ensemble analysis.
    """
    sasa = mdtraj.shrake_rupley(traj, probe_radius=probe_radius)
    sasa = condense_sidechain_sasas(sasa, traj.top)
    exposed = sasa > exposure_threshold
    return {
        "sasa": sasa,
        "exposed": exposed,
        "exposure_probability": exposed.mean(0),
        "mutual_information": sasa_mi(exposed),
    }


def sqrtm(M):
    D, P = np.linalg.eig(M)
    out = (P * np.sqrt(D[:,None])) @ np.linalg.inv(P)
    return out


def get_wasserstein(distmat, p=2):
    assert distmat.shape[0] == distmat.shape[1]
    distmat = distmat ** p
    row_ind, col_ind = linear_sum_assignment(distmat)
    return distmat[row_ind, col_ind].mean() ** (1/p)


def get_pca2_wasserstein(ref_coords, predicted_coords, n_atoms):
    """Return the PCA-2 Wasserstein distance used by the ensemble analysis.

    ``ref_coords`` and ``predicted_coords`` are coordinates in a common PCA
    basis.  The normalization and Angstrom conversion intentionally match the
    original ``main`` implementation.  Keeping this at module scope makes
    cached reconstruction evaluation use the same definition as the ATLAS
    evaluator.
    """
    ref_coords = np.asarray(ref_coords)
    predicted_coords = np.asarray(predicted_coords)
    if ref_coords.ndim != 2 or predicted_coords.ndim != 2:
        raise ValueError("PCA coordinates must have shape [frames, components]")
    if ref_coords.shape[1] < 2 or predicted_coords.shape[1] < 2:
        raise ValueError("PCA-2 Wasserstein requires at least two components")
    if n_atoms <= 0:
        raise ValueError("n_atoms must be positive")
    pca2_dist = np.linalg.norm(
        ref_coords[:, None, :2] - predicted_coords[None, :, :2], axis=-1
    ) * 10 / np.sqrt(n_atoms)
    return get_wasserstein(pca2_dist)


def summarize_ensemble_errors(
    ref_rmsf,
    predicted_rmsf,
    atomic_emd_mean,
    atomic_emd_var,
    ref_exposure_probability,
    predicted_exposure_probability,
    ref_sasa_mi,
    predicted_sasa_mi,
    ref_contact_probability=None,
    predicted_contact_probability=None,
):
    """Summarize per-atom/per-residue ensemble diagnostics.

    This is only aggregation; the arrays themselves are produced by the
    shared metric helpers above.  It is intentionally reusable by cached
    decoder evaluation so the compact tables cannot drift from ``main``.
    """
    def mean_abs(lhs, rhs):
        return float(np.abs(np.asarray(lhs) - np.asarray(rhs)).mean())

    def rmse(lhs, rhs):
        return float(np.square(np.asarray(lhs) - np.asarray(rhs)).mean() ** 0.5)

    summary = {
        "all_heavy_rmsf_mae_angstrom": mean_abs(ref_rmsf, predicted_rmsf),
        "all_heavy_rmsf_rmse_angstrom": rmse(ref_rmsf, predicted_rmsf),
        "all_heavy_atomic_mean_emd_angstrom": float(np.asarray(atomic_emd_mean).mean()),
        "all_heavy_atomic_covariance_emd_angstrom": float(np.asarray(atomic_emd_var).mean()),
        "sidechain_exposure_probability_mae": mean_abs(
            ref_exposure_probability, predicted_exposure_probability
        ),
        "sidechain_sasa_mi_mae_nats": mean_abs(ref_sasa_mi, predicted_sasa_mi),
    }
    if ref_contact_probability is not None or predicted_contact_probability is not None:
        if ref_contact_probability is None or predicted_contact_probability is None:
            raise ValueError("contact probabilities must be provided together")
        summary["ca_contact_probability_mae"] = mean_abs(
            ref_contact_probability, predicted_contact_probability
        )
    return summary


def get_emd(ref_coords1, ref_coords2, af_coords, seed_coords, n_atoms, K=None):
    """Compute the PCA-coordinate EMD diagnostics used by :func:`main`.

    This was originally nested inside ``main``.  Keeping it at module scope
    lets cached-trajectory evaluators reuse the exact same implementation
    instead of maintaining a second copy with subtly different field names or
    scaling.
    """
    arrays = [ref_coords1, ref_coords2, af_coords, seed_coords]
    if any(len(value.shape) == 3 for value in arrays):
        ref_coords1 = ref_coords1.reshape(ref_coords1.shape[0], -1)
        ref_coords2 = ref_coords2.reshape(ref_coords2.shape[0], -1)
        af_coords = af_coords.reshape(af_coords.shape[0], -1)
        seed_coords = seed_coords.reshape(seed_coords.shape[0], -1)
    if K is not None:
        ref_coords1 = ref_coords1[:, :K]
        ref_coords2 = ref_coords2[:, :K]
        af_coords = af_coords[:, :K]
        seed_coords = seed_coords[:, :K]
    emd = {}
    emd['ref|ref mean'] = (
        np.square(ref_coords1 - ref_coords1.mean(0)).sum(-1)
    ).mean() ** 0.5 / n_atoms ** 0.5 * 10

    distmat = np.square(ref_coords1[:, None] - ref_coords2[None]).sum(-1)
    distmat = distmat ** 0.5 / n_atoms ** 0.5 * 10
    emd['ref|ref2'] = get_wasserstein(distmat)
    emd['ref mean|ref2 mean'] = (
        np.square(ref_coords1.mean(0) - ref_coords2.mean(0)).sum()
    ) ** 0.5 / n_atoms ** 0.5 * 10

    distmat = np.square(ref_coords1[:, None] - af_coords[None]).sum(-1)
    distmat = distmat ** 0.5 / n_atoms ** 0.5 * 10
    emd['ref|af'] = get_wasserstein(distmat)
    emd['ref mean|af mean'] = (
        np.square(ref_coords1.mean(0) - af_coords.mean(0)).sum()
    ) ** 0.5 / n_atoms ** 0.5 * 10

    emd['ref|seed'] = (
        np.square(ref_coords1 - seed_coords).sum(-1)
    ).mean() ** 0.5 / n_atoms ** 0.5 * 10
    emd['ref mean|seed'] = (
        np.square(ref_coords1.mean(0) - seed_coords).sum(-1)
    ).mean() ** 0.5 / n_atoms ** 0.5 * 10
    emd['af|seed'] = (
        np.square(af_coords - seed_coords).sum(-1)
    ).mean() ** 0.5 / n_atoms ** 0.5 * 10
    emd['af|af mean'] = (
        np.square(af_coords - af_coords.mean(0)).sum(-1)
    ).mean() ** 0.5 / n_atoms ** 0.5 * 10
    emd['af mean|seed'] = (
        np.square(af_coords.mean(0) - seed_coords).sum(-1)
    ).mean() ** 0.5 / n_atoms ** 0.5 * 10
    return emd


def analyze_ensemble_trajectories(
    prediction,
    reference,
    *,
    sample_frames=256,
    source="evaluation.metrics.analyze_ensembles",
):
    """Compute the coordinate-compatible ensemble metrics for two trajectories.

    This is the reusable coordinate-compatible metric core for cached and
    other preloaded-trajectory evaluation.  The legacy ATLAS CLI retains its
    crystal-reference and bootstrap protocol in ``main``. Inputs are
    ``mdtraj.Trajectory`` objects in
    nanometers with matching atom order.  File discovery, topology matching,
    and checkpoint-specific sampling belong to the caller; metric definitions
    and atom selections live here so they cannot drift between evaluators.
    """
    if not isinstance(prediction, mdtraj.Trajectory) or not isinstance(
        reference, mdtraj.Trajectory
    ):
        raise TypeError("prediction and reference must be mdtraj.Trajectory objects")
    if prediction.xyz.shape != reference.xyz.shape:
        raise ValueError("prediction and reference trajectories must have matching shapes")
    if prediction.n_atoms != reference.n_atoms:
        raise ValueError("prediction and reference must have matching atom counts")

    if sample_frames > 0 and reference.n_frames > sample_frames:
        sample_idx = np.linspace(0, reference.n_frames - 1, sample_frames).astype(int)
        prediction = prediction[sample_idx]
        reference = reference[sample_idx]

    heavy = [atom.index for atom in reference.top.atoms if atom.element.symbol != "H"]
    pred = prediction.atom_slice(heavy)
    ref = reference.atom_slice(heavy)

    # Match the original evaluator: remove global motion with an all-heavy fit
    # to the target frame 0, then fit the C-alpha subsets for CA-only metrics.
    seed = ref[0]
    pred.superpose(seed)
    ref.superpose(seed)
    ca_indices = ref.top.select("name CA")
    pca_traj = pred.atom_slice(ca_indices)
    ref_ca = ref.atom_slice(ca_indices)
    seed_ca = seed.atom_slice(ca_indices)
    pca_traj.superpose(seed_ca)
    ref_ca.superpose(seed_ca)
    n_atoms = pca_traj.n_atoms

    ref_pca, ref_coords = get_pca(ref_ca.xyz)
    pred_pca, pred_coords = get_pca(pca_traj.xyz)
    split_a = np.arange(len(ref_coords))
    split_b = np.roll(split_a, -1)
    joint_pca, _ = get_pca(np.concatenate([ref_ca.xyz[split_a], pca_traj.xyz]))
    ref_coords_ref = ref_pca.transform(ref_ca.xyz.reshape(len(ref_ca), -1))
    pred_coords_ref = ref_pca.transform(pca_traj.xyz.reshape(len(pca_traj), -1))
    ref_coords_pred = pred_pca.transform(ref_ca.xyz.reshape(len(ref_ca), -1))
    pred_coords_pred = pred_pca.transform(pca_traj.xyz.reshape(len(pca_traj), -1))
    ref_coords_joint = joint_pca.transform(ref_ca.xyz.reshape(len(ref_ca), -1))
    pred_coords_joint = joint_pca.transform(pca_traj.xyz.reshape(len(pca_traj), -1))
    seed_ref = ref_pca.transform(ref_ca.xyz[:1].reshape(1, -1))
    seed_pred = pred_pca.transform(ref_ca.xyz[:1].reshape(1, -1))
    seed_joint = joint_pca.transform(ref_ca.xyz[:1].reshape(1, -1))

    out = {
        "protocol": {
            "source": source,
            "random_sampling_replacement": "uniform paired frames; cyclic reference pairing for EMD",
            "atom_selection": "all heavy atoms for RMSF/atomic EMD/SASA; C-alpha for PCA/contact/pairwise/PCA-EMD",
            "alignment": "all-heavy fit to target frame 0, then C-alpha fit for C-alpha metrics",
            "crystal_reference": None,
            "crystal_unavailable_reason": "trajectory API does not receive an external crystal PDB",
            "sasa_probe_radius_nm": 0.28,
            "sasa_exposure_threshold_nm2": 0.02,
            "contact_threshold_nm": 0.8,
        },
        "ca_mask": ca_indices.tolist(),
        "num_heavy_atoms": len(heavy),
        "num_ca": n_atoms,
        "num_frames": len(ref),
        "ref_variance": (ref_pca.explained_variance_ / n_atoms * 100).tolist(),
        "af_variance": (pred_pca.explained_variance_ / n_atoms * 100).tolist(),
        "joint_variance": (joint_pca.explained_variance_ / n_atoms * 100).tolist(),
        "cosine_sim": float(ref_pca.components_[0] @ pred_pca.components_[0]),
        "cosine_sim_absolute": float(abs(ref_pca.components_[0] @ pred_pca.components_[0])),
        "af_rmsf": (mdtraj.rmsf(pred, seed) * 10).tolist(),
        "ref_rmsf": (mdtraj.rmsf(ref, seed) * 10).tolist(),
        "crystal_sasa": None,
        "crystal_distmat": None,
    }

    for basis_name, ref_basis, pred_basis in (
        ("reference", ref_coords_ref, pred_coords_ref),
        ("prediction", ref_coords_pred, pred_coords_pred),
        ("joint", ref_coords_joint, pred_coords_joint),
    ):
        out[f"pca2_wasserstein_{basis_name}_angstrom"] = float(
            get_pca2_wasserstein(ref_basis, pred_basis, n_atoms)
        )

    emd_mean, emd_var = get_atomic_emd(ref.xyz.astype(np.float64), pred.xyz.astype(np.float64))
    out["emd_mean"] = emd_mean.tolist()
    out["emd_var"] = emd_var.tolist()

    ref_sasa = get_sasa_metrics(ref, probe_radius=0.28, exposure_threshold=0.02)
    pred_sasa = get_sasa_metrics(pred, probe_radius=0.28, exposure_threshold=0.02)
    out["ref_sa_prob"] = ref_sasa["exposure_probability"].tolist()
    out["af_sa_prob"] = pred_sasa["exposure_probability"].tolist()
    out["ref_mi_mat"] = np.nan_to_num(
        ref_sasa["mutual_information"], nan=0.0, posinf=0.0, neginf=0.0
    ).tolist()
    out["af_mi_mat"] = np.nan_to_num(
        pred_sasa["mutual_information"], nan=0.0, posinf=0.0, neginf=0.0
    ).tolist()

    out["ref_contact_prob"] = get_contact_probability(ref_ca.xyz, threshold=0.8).tolist()
    out["af_contact_prob"] = get_contact_probability(pca_traj.xyz, threshold=0.8).tolist()
    ref_pair = get_rmsds(ref_ca.xyz[split_a], ref_ca.xyz[split_b], broadcast=True)
    pred_pair = get_rmsds(pca_traj.xyz, pca_traj.xyz, broadcast=True)
    ref_self_pair = get_rmsds(ref_ca.xyz[split_a], ref_ca.xyz[split_a], broadcast=True)
    out["ref_mean_pairwise_rmsd"] = float(ref_pair.mean())
    out["af_mean_pairwise_rmsd"] = float(pred_pair.mean())
    out["ref_rms_pairwise_rmsd"] = float(np.square(ref_pair).mean() ** 0.5)
    out["af_rms_pairwise_rmsd"] = float(np.square(pred_pair).mean() ** 0.5)
    out["ref_self_mean_pairwise_rmsd"] = float(ref_self_pair.mean())
    out["ref_self_rms_pairwise_rmsd"] = float(np.square(ref_self_pair).mean() ** 0.5)
    out["EMD,ref"] = get_emd(ref_coords_ref[split_a], ref_coords_ref[split_b], pred_coords_ref, seed_ref, n_atoms, K=2)
    out["EMD,af2"] = get_emd(ref_coords_pred[split_a], ref_coords_pred[split_b], pred_coords_pred, seed_pred, n_atoms, K=2)
    out["EMD,joint"] = get_emd(ref_coords_joint[split_a], ref_coords_joint[split_b], pred_coords_joint, seed_joint, n_atoms, K=2)
    out["summary"] = summarize_ensemble_errors(
        out["ref_rmsf"], out["af_rmsf"], out["emd_mean"], out["emd_var"],
        out["ref_sa_prob"], out["af_sa_prob"], out["ref_mi_mat"], out["af_mi_mat"],
        out["ref_contact_prob"], out["af_contact_prob"],
    )
    return out

def align_tops(top1, top2):
    names1 = [repr(a) for a in top1.atoms]
    names2 = [repr(a) for a in top2.atoms]

    intersection = [nam for nam in names1 if nam in names2]
    
    mask1 = [names1.index(nam) for nam in intersection]
    mask2 = [names2.index(nam) for nam in intersection]
    return mask1, mask2
    
def main(name: str, args: argparse.Namespace):
    print('Analyzing', name)
    out = {}
    ref_aa = mdtraj.load(f'{args.atlas_dir}/{name}/{name}.pdb')
    topfile = f'{args.atlas_dir}/{name}/{name}.pdb'
    print('Loading reference trajectory')
    traj_aa = mdtraj.load(f'{args.atlas_dir}/{name}/{name}_prod_R1_fit.xtc', top=topfile) \
        + mdtraj.load(f'{args.atlas_dir}/{name}/{name}_prod_R2_fit.xtc', top=topfile) \
        + mdtraj.load(f'{args.atlas_dir}/{name}/{name}_prod_R3_fit.xtc', top=topfile)
    print(f'Loaded {traj_aa.n_frames} reference frames')
    
    print('Loading AF2 conformers')
    aftraj_aa = mdtraj.load(f'{args.pdbdir}/{name}.pdb')
        
    print(f'Loaded {aftraj_aa.n_frames} AF2 conformers')   
    print(f'Reference has {traj_aa.n_atoms} atoms')
    print(f'Crystal has {ref_aa.n_atoms} atoms')
    print(f'AF has {aftraj_aa.n_atoms} atoms')

    print('Removing hydrogens')

    traj_aa = traj_aa.atom_slice([a.index for a in traj_aa.top.atoms if a.element.symbol != 'H'], True)
    ref_aa = ref_aa.atom_slice([a.index for a in ref_aa.top.atoms if a.element.symbol != 'H'], True)
    aftraj_aa = aftraj_aa.atom_slice([a.index for a in aftraj_aa.top.atoms if a.element.symbol != 'H'], True)

    print(f'Reference has {traj_aa.n_atoms} atoms')
    print(f'Crystal has {ref_aa.n_atoms} atoms')
    print(f'AF has {aftraj_aa.n_atoms} atoms')
    
    if args.bb_only:
        print('Removing sidechains')
        aftraj_aa = aftraj_aa.atom_slice([a.index for a in aftraj_aa.top.atoms if a.name in ['CA', 'C', 'N', 'O', 'OXT']], True)
        print(f'AF has {aftraj_aa.n_atoms} atoms')

    elif args.ca_only:
        print('Removing sidechains')
        aftraj_aa = aftraj_aa.atom_slice([a.index for a in aftraj_aa.top.atoms if a.name == 'CA'], True)
        print(f'AF has {aftraj_aa.n_atoms} atoms')

    
    refmask, afmask = align_tops(traj_aa.top, aftraj_aa.top)
    traj_aa = traj_aa.atom_slice(refmask, True)
    ref_aa = ref_aa.atom_slice(refmask, True)
    aftraj_aa = aftraj_aa.atom_slice(afmask, True)

    print(f'Aligned on {aftraj_aa.n_atoms} atoms')

    np.random.seed(137)
    RAND1 = np.random.randint(0, traj_aa.n_frames, aftraj_aa.n_frames)
    RAND2 = np.random.randint(0, traj_aa.n_frames, aftraj_aa.n_frames)
    RAND1K = np.random.randint(0, traj_aa.n_frames, 1000)
        
    traj_aa.superpose(ref_aa)
    aftraj_aa.superpose(ref_aa)

    out['ca_mask'] = ca_mask = [a.index for a in traj_aa.top.atoms if a.name == 'CA']
    traj = traj_aa.atom_slice(ca_mask, False)
    ref = ref_aa.atom_slice(ca_mask, False)
    aftraj = aftraj_aa.atom_slice(ca_mask, False)
    print(f'Sliced {aftraj.n_atoms} C-alphas')
    
    traj.superpose(ref)
    aftraj.superpose(ref)

    
    n_atoms = aftraj.n_atoms

    print(f'Doing PCA')

    ref_pca, ref_coords = get_pca(traj.xyz)
    af_coords_ref_pca = ref_pca.transform(aftraj.xyz.reshape(aftraj.n_frames, -1))
    seed_coords_ref_pca = ref_pca.transform(ref.xyz.reshape(1, -1))
    
    af_pca, af_coords = get_pca(aftraj.xyz)
    ref_coords_af_pca = af_pca.transform(traj.xyz.reshape(traj.n_frames, -1))
    seed_coords_af_pca = af_pca.transform(ref.xyz.reshape(1, -1))
    
    joint_pca, _ = get_pca(np.concatenate([traj[RAND1].xyz, aftraj.xyz]))
    af_coords_joint_pca = joint_pca.transform(aftraj.xyz.reshape(aftraj.n_frames, -1))
    ref_coords_joint_pca = joint_pca.transform(traj.xyz.reshape(traj.n_frames, -1))
    seed_coords_joint_pca = joint_pca.transform(ref.xyz.reshape(1, -1))
    
    out['ref_variance'] = ref_pca.explained_variance_ / n_atoms * 100
    out['af_variance'] = af_pca.explained_variance_ / n_atoms * 100
    out['joint_variance'] = joint_pca.explained_variance_ / n_atoms * 100

    out['af_rmsf'] = mdtraj.rmsf(aftraj_aa, ref_aa) * 10
    out['ref_rmsf'] = mdtraj.rmsf(traj_aa, ref_aa) * 10
    
    print(f'Computing atomic EMD')
    out['emd_mean'], out['emd_var'] = get_atomic_emd(
        traj_aa[RAND1K].xyz, aftraj_aa.xyz
    )


    print(f'Analyzing SASA')
    af_sasa_metrics = get_sasa_metrics(aftraj_aa, probe_radius=0.28, exposure_threshold=0.02)
    ref_sasa_metrics = get_sasa_metrics(traj_aa[RAND1K], probe_radius=0.28, exposure_threshold=0.02)
    crystal_sasa = mdtraj.shrake_rupley(ref_aa, probe_radius=0.28)
    out['crystal_sasa'] = condense_sidechain_sasas(crystal_sasa, ref_aa.top)
    
    out['ref_sa_prob'] = ref_sasa_metrics['exposure_probability']
    out['af_sa_prob'] = af_sasa_metrics['exposure_probability']
    out['ref_mi_mat'] = ref_sasa_metrics['mutual_information']
    out['af_mi_mat'] = af_sasa_metrics['mutual_information']
    
    out['ref_contact_prob'] = get_contact_probability(traj[RAND1].xyz, threshold=0.8)
    out['af_contact_prob'] = get_contact_probability(aftraj.xyz, threshold=0.8)
    out['crystal_distmat'] = np.linalg.norm(ref.xyz[0,None,:] - ref.xyz[0,:,None], axis=-1)
    
    out['ref_mean_pairwise_rmsd'] = get_rmsds(traj[RAND1].xyz, traj[RAND2].xyz, broadcast=True).mean()
    out['af_mean_pairwise_rmsd'] = get_rmsds(aftraj.xyz, aftraj.xyz, broadcast=True).mean()

    out['ref_rms_pairwise_rmsd'] = np.square(get_rmsds(traj[RAND1].xyz, traj[RAND2].xyz, broadcast=True)).mean() ** 0.5
    out['af_rms_pairwise_rmsd'] = np.square(get_rmsds(aftraj.xyz, aftraj.xyz, broadcast=True)).mean() ** 0.5

    out['ref_self_mean_pairwise_rmsd'] = get_rmsds(traj[RAND1].xyz, traj[RAND1].xyz, broadcast=True).mean()
    out['ref_self_rms_pairwise_rmsd'] = np.square(get_rmsds(traj[RAND1].xyz, traj[RAND1].xyz, broadcast=True)).mean() ** 0.5
    
    out['cosine_sim'] = (ref_pca.components_[0] * af_pca.components_[0]).sum() 
    

    K=2
    out[f'EMD,ref'] = get_emd(ref_coords[RAND1], ref_coords[RAND2], af_coords_ref_pca, seed_coords_ref_pca, n_atoms, K=K)
    out[f'EMD,af2'] = get_emd(ref_coords_af_pca[RAND1], ref_coords_af_pca[RAND2], af_coords, seed_coords_af_pca, n_atoms, K=K)
    out[f'EMD,joint'] = get_emd(ref_coords_joint_pca[RAND1], ref_coords_joint_pca[RAND2], af_coords_joint_pca, seed_coords_joint_pca, n_atoms, K=K)
    return name, out 


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas_dir", required=True)
    parser.add_argument("--pdbdir", required=True)
    parser.add_argument("--pdb_id", nargs="*", default=[])
    parser.add_argument("--bb_only", action="store_true")
    parser.add_argument("--ca_only", action="store_true")
    parser.add_argument("--num_workers", type=int, default=1)
    return parser


def run(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    """Analyze all requested PDB ensembles and write ``out.pkl``."""
    pdb_ids = args.pdb_id or [
        name.rsplit(".", 1)[0]
        for name in os.listdir(args.pdbdir)
        if name.endswith(".pdb")
    ]
    if args.num_workers > 1:
        with Pool(args.num_workers) as pool:
            results = pool.imap(partial(main, args=args), pdb_ids)
            out = dict(tqdm.tqdm(results, total=len(pdb_ids)))
    else:
        out = dict(tqdm.tqdm((main(name, args) for name in pdb_ids), total=len(pdb_ids)))
    with open(os.path.join(args.pdbdir, "out.pkl"), "wb") as handle:
        pickle.dump(out, handle)
    return out


def cli() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    cli()
