"""
ProteinSpectrumFAR Inference Script
直接在频谱上进行自回归生成，不再需要VAE和Flow Matching
"""

import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.nn.functional as F
from collections import defaultdict
import json
import argparse
from tqdm import tqdm
import time

from dataset_atlas import ProteinMotionDataset, collate_fn_single
from protein_spectrum_far import protein_spectrum_far_large, protein_spectrum_far_base, protein_spectrum_far_huge, sample_multiple
from data.fourier.old_torch_utils import spectrum_to_coordinates, inverse_normalize_and_ifft


# Global variables
# map_ids = [368, 156, 41, 45, 227, 94, 196, 170, 192, 129, 4, 384, 313, 141, 461, 154, 382, 237, 184, 190, 424, 207, 7, 145, 21]
map_ids = ['5znj_A_R1', '6d7y_A_R1', '6d7y_B_R1', '6e7e_A_R1', '6gus_A_R1', '6h49_A_R1', '6h86_A_R1', '6iah_A_R1', '6in7_A_R1', '6j56_A_R1', '6jpt_A_R1', '6jv8_A_R1', '6jwh_A_R1', '6kty_A_R1', '6l34_A_R1', '6l3r_E_R1', '6l4l_A_R1', '6l4p_B_R1', '6l8s_A_R1', '6lrd_A_R1', '6lus_A_R1', '6ndw_B_R1', '6nl2_A_R1', '6o2v_A_R1', '6o6y_A_R1', '6odd_B_R1', '6okd_C_R1', '6ono_C_R1', '6ovk_R_R1', '6oz1_A_R1', '6p5h_A_R1', '6p5x_B_R1', '6pce_B_R1', '6pnv_A_R1', '6pxz_B_R1', '6q10_A_R1', '6q9c_A_R1', '6q9c_B_R1', '6qj0_A_R1', '6ro6_A_R1', '6rrv_A_R1', '6rwt_A_R1', '6sms_A_R1', '6tgk_C_R1', '6tly_A_R1', '6uof_A_R1', '6vjg_A_R1', '6xb3_H_R1', '6xds_A_R1', '6xrx_A_R1', '6y2x_A_R1', '6yhu_B_R1', '6ypi_A_R1', '6zsl_B_R1', '7a66_B_R1', '7aex_A_R1', '7aqx_A_R1', '7asg_A_R1', '7buy_A_R1', '7bwf_B_R1', '7c45_A_R1', '7dmn_A_R1', '7e2s_A_R1', '7ead_A_R1', '7ec1_A_R1', '7jfl_C_R1', '7jrq_A_R1', '7k7p_B_R1', '7la6_A_R1', '7lao_A_R1', '7lp1_A_R1', '7mf4_A_R1', '7n0j_E_R1', '7ned_A_R1', '7nmq_A_R1', '7onn_A_R1', '7p41_D_R1', '7p46_A_R1', '7qsu_A_R1', '7rm7_A_R1', '7s86_A_R1', '7wab_A_R1']
# map_ids = ['6d7y_A_R1', '6d7y_B_R1', '6e7e_A_R1', '6gus_A_R1', '6h49_A_R1', '6h86_A_R1', '6iah_A_R1', '6in7_A_R1', '6j56_A_R1', '6jpt_A_R1', '6jv8_A_R1', '6kty_A_R1', '6l34_A_R1', '6l4p_B_R1', '6lus_A_R1', '6ndw_B_R1', '6o2v_A_R1', '6odd_B_R1', '6okd_C_R1', '6ono_C_R1', '6ovk_R_R1', '6p5h_A_R1', '6pce_B_R1', '6pxz_B_R1', '6q10_A_R1', '6q9c_A_R1', '6ro6_A_R1', '6rrv_A_R1', '6rwt_A_R1', '6tgk_C_R1', '6tly_A_R1', '6uof_A_R1', '6vjg_A_R1', '6xb3_H_R1', '6y2x_A_R1', '6yhu_B_R1', '6ypi_A_R1', '7a66_B_R1', '7bwf_B_R1', '7e2s_A_R1', '7ead_A_R1', '7jfl_C_R1', '7jrq_A_R1', '7k7p_B_R1', '7la6_A_R1', '7lp1_A_R1', '7n0j_E_R1', '7rm7_A_R1', '7s86_A_R1'] #256
# map_ids = ['6d7y_A_R1', '6h86_A_R1', '6in7_A_R1', '6jpt_A_R1', '6jv8_A_R1', '6l34_A_R1', '6odd_B_R1', '6okd_C_R1', '6ono_C_R1', '6p5h_A_R1', '6pce_B_R1', '6pxz_B_R1', '6ro6_A_R1', '6rrv_A_R1', '6tgk_C_R1', '6tly_A_R1', '6uof_A_R1', '6vjg_A_R1', '6yhu_B_R1', '6ypi_A_R1', '7a66_B_R1', '7bwf_B_R1', '7jfl_C_R1', '7k7p_B_R1', '7lp1_A_R1', '7n0j_E_R1', '7s86_A_R1'] # 128
# map_ids = ['6okd_C_R1', '6ro6_A_R1', '7jfl_C_R1', '7lp1_A_R1'] # 64
_default_scaling_path = os.environ.get("ATLAS_SCALING_PATH", "./dataset/atlas_10_scaling_factors.pth")
scaling_factor = torch.tensor(torch.load(_default_scaling_path, weights_only=False))


def compute_coordinate_reconstruction_metrics(original_coords, reconstructed_coords):
    """
    Calculate coordinate reconstruction quality metrics
    Args:
        original_coords: [num_frames, num_residues, 3] original coordinates
        reconstructed_coords: [num_residues, 3, num_frames] reconstructed coordinates
    """
    # Convert reconstructed coordinates to same dimension order as original coordinates
    reconstructed_coords = reconstructed_coords.permute(2, 0, 1).contiguous()  # [num_frames, num_residues, 3]
    
    # Calculate overall displacement RMSE
    diff = original_coords - reconstructed_coords
    displacement_squared = torch.sum(diff ** 2, dim=-1)  # [num_frames, num_residues]
    overall_rmse = torch.sqrt(torch.mean(displacement_squared))
    
    # Calculate RMSE for each residue
    rmsd_per_residue = torch.sqrt(torch.mean(displacement_squared, dim=0))  # [num_residues]
    
    # Calculate RMSE for each frame
    rmsd_per_frame = torch.sqrt(torch.mean(displacement_squared, dim=1))  # [num_frames]
    
    # Calculate MAE
    mae = torch.mean(torch.abs(original_coords.cpu() - reconstructed_coords.cpu()))
    
    # Calculate correlation coefficient
    original_flat = original_coords.view(-1)
    reconstructed_flat = reconstructed_coords.view(-1)
    correlation = torch.corrcoef(torch.stack([original_flat.cpu(), reconstructed_flat.cpu()]))[0, 1]
    
    return {
        'overall_rmse': overall_rmse.item(),
        'mae': mae.item(),
        'correlation': correlation.item(),
        'rmse_per_frame': rmsd_per_frame.cpu().numpy(),
        'rmse_per_residue': rmsd_per_residue.cpu().numpy()
    }


def load_test_data(data_path, batch_size=1):
    """Load test data"""
    dataset = ProteinMotionDataset(data_path=data_path)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn_single,
        pin_memory=True
    )
    return dataloader


def evaluate_far_reconstruction(far_model, dataloader, trajectory_data_dict, device, 
                                num_samples=1, cfg=2.0, temperature=1.0, 
                                sample_method='nomask', seed=42):
    """
    Evaluate FAR reconstruction performance
    
    Args:
        far_model: ProteinSpectrumFAR model
        dataloader: test data loader
        trajectory_data_dict: 从ProteinMotionDataset.load_full_trajectory_data()加载的轨迹数据字典
        device: device
        num_samples: number of samples per protein (for ensemble)
        cfg: classifier-free guidance scale
        temperature: sampling temperature
        sample_method: sampling method ('sample', 'nomask', 'mask')
        seed: random seed
    
    Returns:
        results: evaluation results
    """
    
    results = {
        'far_rmse': [],
        'far_mae': [],
        'far_correlation': [],
        'per_protein_metrics': defaultdict(list),
        'timing': {
            'total_times': [],
            'far_sampling_times': [],
            'trajectory_reconstruction_times': [],
            'metrics_computation_times': []
        }
    }
    
    print("Starting FAR reconstruction evaluation...")
    print(f"Sampling method: {sample_method}")
    print(f"Number of samples per protein: {num_samples}")
    print(f"CFG scale: {cfg}, Temperature: {temperature}")
    
    # Iterate through data loader
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating FAR reconstruction")):
        batch_start_time = time.time()
        
        # Unpack data
        packed_graph, coords, spectrum_gt, rotation_matrices, aatype, aa_mask, esm_features = batch
        
        # Move data to device
        coords = coords.to(device)
        rotation_matrices = rotation_matrices.to(device)
        aatype = aatype.to(device)
        aa_mask = aa_mask.to(device)
        esm_features = esm_features.to(device)
        spectrum_gt = spectrum_gt.to(device)
        
        num_residues = coords.shape[0]
        seq_len = spectrum_gt.shape[1] // 3  # Assuming spectrum is [num_residues, 3*seq_len, 1]
        
        # Use FAR model to sample spectrum
        far_start_time = time.time()
        with torch.no_grad():
            if num_samples > 1:
                # Multiple samples for ensemble
                reconstructed_spectra = sample_multiple(
                    model=far_model,
                    coords=coords,
                    rotation_matrices=rotation_matrices,
                    aatype=aatype,
                    aa_mask=aa_mask,
                    esm_features=esm_features,
                    n_samples=num_samples,
                    num_iter=seq_len,
                    cfg=cfg,
                    temperature=temperature,
                    progress=False,
                    sample_method=sample_method,
                    seed=seed + batch_idx
                )
                # Use mean of ensemble
                reconstructed_spectrum = reconstructed_spectra.mean(dim=0)  # [num_residues, 3, seq_len, 1]
            else:
                # Single sample
                if sample_method == 'nomask':
                    reconstructed_spectrum = far_model.sample_nomask(
                        coords=coords,
                        rotation_matrices=rotation_matrices,
                        aatype=aatype,
                        aa_mask=aa_mask,
                        esm_features=esm_features,
                        num_iter=seq_len,
                        cfg=cfg,
                        temperature=temperature,
                        progress=False
                    )
                elif sample_method == 'mask':
                    reconstructed_spectrum = far_model.sample_mask(
                        coords=coords,
                        rotation_matrices=rotation_matrices,
                        aatype=aatype,
                        aa_mask=aa_mask,
                        esm_features=esm_features,
                        num_iter=seq_len,
                        cfg=cfg,
                        temperature=temperature,
                        progress=False
                    )
                else:  # 'sample'
                    reconstructed_spectrum = far_model.sample(
                        coords=coords,
                        rotation_matrices=rotation_matrices,
                        aatype=aatype,
                        aa_mask=aa_mask,
                        esm_features=esm_features,
                        num_iter=seq_len,
                        cfg=cfg,
                        temperature=temperature,
                        progress=False
                    )
        reconstructed_spectrum = reconstructed_spectrum.squeeze(-1) # (num_residues, seq_len, 6) or (num_residues, seq_len, 3)
        
        far_end_time = time.time()
        far_sampling_time = far_end_time - far_start_time
        results['timing']['far_sampling_times'].append(far_sampling_time)
        
        # Get original trajectory data from the preloaded dictionary
        protein_id = map_ids[batch_idx]
        protein_data = trajectory_data_dict[protein_id]
        
        # Reconstruct ground truth trajectory from spectrum data
        trajectory_start_time = time.time()
        
        # Get frame 0 positions
        frame0_positions = torch.tensor(np.stack([
            protein_data['positions']['x0'],
            protein_data['positions']['y0'],
            protein_data['positions']['z0']
        ], axis=-1), dtype=torch.float32).to(device)
        
        # Determine num_frames - 使用100帧作为默认（与ATLAS数据集一致）
        num_frames = 100
        # num_frames = len(protein_data['positions']['x'][0])  # 假设每个残基的频谱数据长度即为帧数
        
        # Convert reconstructed spectrum to coordinates
        reconstructed_coords = spectrum_to_coordinates(
            reconstructed_spectrum, frame0_positions, scaling_factor, num_frames=num_frames
        )
        
        # Get original coordinate sequence by reconstructing from ground truth spectrum
        original_coords = ProteinMotionDataset.reconstruct_trajectory_from_spectrum(
            protein_data, scaling_factor, num_frames=num_frames
        ).to(device)
        
        trajectory_end_time = time.time()
        trajectory_time = trajectory_end_time - trajectory_start_time
        results['timing']['trajectory_reconstruction_times'].append(trajectory_time)
        
        # Calculate reconstruction metrics
        metrics_start_time = time.time()
        coord_metrics = compute_coordinate_reconstruction_metrics(original_coords, reconstructed_coords)
        metrics_end_time = time.time()
        metrics_time = metrics_end_time - metrics_start_time
        results['timing']['metrics_computation_times'].append(metrics_time)
        
        results['far_rmse'].append(coord_metrics['overall_rmse'])
        results['far_mae'].append(coord_metrics['mae'])
        results['far_correlation'].append(coord_metrics['correlation'])
        
        # Record metrics for each protein
        results['per_protein_metrics']['rmse'].append(coord_metrics['overall_rmse'])
        results['per_protein_metrics']['mae'].append(coord_metrics['mae'])
        results['per_protein_metrics']['correlation'].append(coord_metrics['correlation'])
        results['per_protein_metrics']['num_residues'].append(num_residues)
        results['per_protein_metrics']['num_frames'].append(num_frames)
        
        # Calculate total time
        batch_end_time = time.time()
        total_time = batch_end_time - batch_start_time
        results['timing']['total_times'].append(total_time)
        
        print(f"\nBatch {batch_idx + 1} ({protein_id}): RMSE={coord_metrics['overall_rmse']:.4f}Å, MAE={coord_metrics['mae']:.4f}Å, Corr={coord_metrics['correlation']:.4f}")
        print(f"  Time: Total={total_time:.3f}s, FAR={far_sampling_time:.3f}s, Traj={trajectory_time:.3f}s")
    
    return results


def compute_statistics(metrics_list):
    """Calculate statistics"""
    metrics_array = np.array(metrics_list)
    stats = {
        'mean': np.mean(metrics_array),
        'std': np.std(metrics_array),
        'min': np.min(metrics_array),
        'max': np.max(metrics_array),
        'median': np.median(metrics_array),
        'q25': np.percentile(metrics_array, 25),
        'q75': np.percentile(metrics_array, 75)
    }
    return stats


def plot_far_results(results, save_dir):
    """Plot FAR results charts"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Create charts
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. FAR RMSE distribution
    axes[0, 0].hist(results['far_rmse'], bins=30, alpha=0.7, color='blue')
    axes[0, 0].set_title('FAR Reconstruction RMSE Distribution')
    axes[0, 0].set_xlabel('RMSE (Å)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].axvline(np.mean(results['far_rmse']), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(results["far_rmse"]):.4f}Å')
    axes[0, 0].legend()
    
    # 2. FAR MAE distribution
    axes[0, 1].hist(results['far_mae'], bins=30, alpha=0.7, color='green')
    axes[0, 1].set_title('FAR Reconstruction MAE Distribution')
    axes[0, 1].set_xlabel('MAE (Å)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].axvline(np.mean(results['far_mae']), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(results["far_mae"]):.4f}Å')
    axes[0, 1].legend()
    
    # 3. FAR correlation coefficient distribution
    axes[0, 2].hist(results['far_correlation'], bins=30, alpha=0.7, color='orange')
    axes[0, 2].set_title('FAR Reconstruction Correlation Distribution')
    axes[0, 2].set_xlabel('Correlation')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].axvline(np.mean(results['far_correlation']), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(results["far_correlation"]):.4f}')
    axes[0, 2].legend()
    
    # 4. Protein size vs reconstruction RMSE scatter plot
    num_residues = results['per_protein_metrics']['num_residues']
    protein_rmse = results['per_protein_metrics']['rmse']
    axes[1, 0].scatter(num_residues, protein_rmse, alpha=0.6, color='purple')
    axes[1, 0].set_title('Protein Size vs FAR Reconstruction RMSE')
    axes[1, 0].set_xlabel('Number of Residues')
    axes[1, 0].set_ylabel('RMSE (Å)')
    
    # 5. Time distribution histogram
    axes[1, 1].hist(results['timing']['total_times'], bins=20, alpha=0.7, color='red')
    axes[1, 1].set_title('Total Processing Time Distribution')
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].axvline(np.mean(results['timing']['total_times']), color='blue', linestyle='--', 
                       label=f'Mean: {np.mean(results["timing"]["total_times"]):.3f}s')
    axes[1, 1].legend()
    
    # 6. Time distribution pie chart by processing step
    far_time = np.mean(results['timing']['far_sampling_times'])
    trajectory_time = np.mean(results['timing']['trajectory_reconstruction_times'])
    metrics_time = np.mean(results['timing']['metrics_computation_times'])
    
    times = [far_time, trajectory_time, metrics_time]
    labels = ['FAR Sampling', 'Trajectory Reconstruction', 'Metrics Computation']
    colors = ['#ff9999', '#99ff99', '#ffcc99']
    
    axes[1, 2].pie(times, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    axes[1, 2].set_title('Time Distribution by Processing Step')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'far_reconstruction_results.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"FAR results chart saved to {os.path.join(save_dir, 'far_reconstruction_results.png')}")


def save_far_results(results, save_path):
    """Save FAR evaluation results"""
    detailed_results = {
        'far_rmse_stats': compute_statistics(results['far_rmse']),
        'far_mae_stats': compute_statistics(results['far_mae']),
        'far_correlation_stats': compute_statistics(results['far_correlation']),
        'per_protein_stats': {
            'rmse_stats': compute_statistics(results['per_protein_metrics']['rmse']),
            'mae_stats': compute_statistics(results['per_protein_metrics']['mae']),
            'correlation_stats': compute_statistics(results['per_protein_metrics']['correlation']),
            'num_residues_stats': compute_statistics(results['per_protein_metrics']['num_residues']),
            'num_frames_stats': compute_statistics(results['per_protein_metrics']['num_frames'])
        },
        'timing_stats': {
            'total_time_stats': compute_statistics(results['timing']['total_times']),
            'far_sampling_time_stats': compute_statistics(results['timing']['far_sampling_times']),
            'trajectory_reconstruction_time_stats': compute_statistics(results['timing']['trajectory_reconstruction_times']),
            'metrics_computation_time_stats': compute_statistics(results['timing']['metrics_computation_times'])
        }
    }
    
    torch.save(detailed_results, save_path)
    print(f"FAR detailed results saved to {save_path}")


def print_far_summary(results):
    """Print FAR evaluation results summary"""
    print("\n" + "="*60)
    print("ProteinSpectrumFAR Reconstruction Performance Evaluation")
    print("="*60)
    
    print(f"\nNumber of evaluation samples: {len(results['far_rmse'])}")
    
    print("\n--- FAR Trajectory Reconstruction Performance ---")
    far_rmse_stats = compute_statistics(results['far_rmse'])
    far_mae_stats = compute_statistics(results['far_mae'])
    far_corr_stats = compute_statistics(results['far_correlation'])
    
    print(f"FAR RMSE: {far_rmse_stats['mean']:.4f} ± {far_rmse_stats['std']:.4f} Å")
    print(f"FAR MAE: {far_mae_stats['mean']:.4f} ± {far_mae_stats['std']:.4f} Å")
    print(f"FAR Correlation: {far_corr_stats['mean']:.4f} ± {far_corr_stats['std']:.4f}")
    
    print("\n--- Statistics by Protein ---")
    protein_rmse_stats = compute_statistics(results['per_protein_metrics']['rmse'])
    protein_mae_stats = compute_statistics(results['per_protein_metrics']['mae'])
    protein_corr_stats = compute_statistics(results['per_protein_metrics']['correlation'])
    num_residues_stats = compute_statistics(results['per_protein_metrics']['num_residues'])
    num_frames_stats = compute_statistics(results['per_protein_metrics']['num_frames'])
    
    print(f"Protein RMSE: {protein_rmse_stats['mean']:.4f} ± {protein_rmse_stats['std']:.4f} Å")
    print(f"Protein MAE: {protein_mae_stats['mean']:.4f} ± {protein_mae_stats['std']:.4f} Å")
    print(f"Protein Correlation: {protein_corr_stats['mean']:.4f} ± {protein_corr_stats['std']:.4f}")
    print(f"Average number of residues: {num_residues_stats['mean']:.1f} ± {num_residues_stats['std']:.1f}")
    print(f"Average number of frames: {num_frames_stats['mean']:.1f} ± {num_frames_stats['std']:.1f}")
    
    print("\n--- Time Statistics ---")
    total_time_stats = compute_statistics(results['timing']['total_times'])
    far_time_stats = compute_statistics(results['timing']['far_sampling_times'])
    trajectory_time_stats = compute_statistics(results['timing']['trajectory_reconstruction_times'])
    metrics_time_stats = compute_statistics(results['timing']['metrics_computation_times'])
    
    print(f"Total time: {total_time_stats['mean']:.3f} ± {total_time_stats['std']:.3f} s")
    print(f"FAR sampling time: {far_time_stats['mean']:.3f} ± {far_time_stats['std']:.3f} s ({far_time_stats['mean']/total_time_stats['mean']*100:.1f}%)")
    print(f"Trajectory reconstruction time: {trajectory_time_stats['mean']:.3f} ± {trajectory_time_stats['std']:.3f} s ({trajectory_time_stats['mean']/total_time_stats['mean']*100:.1f}%)")
    print(f"Metrics computation time: {metrics_time_stats['mean']:.3f} ± {metrics_time_stats['std']:.3f} s ({metrics_time_stats['mean']/total_time_stats['mean']*100:.1f}%)")
    
    print("="*60)


def amino_acid_id_to_name(aatype_id):
    """
    Convert amino acid type ID to three-letter name
    Based on OpenFold's restype_1to3 mapping
    Args:
        aatype_id: amino acid type ID (0-19)
    Returns:
        amino_acid_name: three-letter amino acid name
    """
    # Based on OpenFold's restypes order (alphabetically sorted)
    amino_acid_names = [
        'ALA', 'ARG', 'ASN', 'ASP', 'CYS',  # 0-4: A, R, N, D, C
        'GLN', 'GLU', 'GLY', 'HIS', 'ILE',  # 5-9: Q, E, G, H, I
        'LEU', 'LYS', 'MET', 'PHE', 'PRO',  # 10-14: L, K, M, F, P
        'SER', 'THR', 'TRP', 'TYR', 'VAL'   # 15-19: S, T, W, Y, V
    ]
    return amino_acid_names[aatype_id]

def amino_acid_name_to_id(aa_name):
    """
    Convert three-letter amino acid name to ID
    Based on OpenFold's restype_3to1 mapping
    Args:
        aa_name: three-letter amino acid name
    Returns:
        aatype_id: amino acid type ID (0-19)
    """
    # Based on OpenFold's restype_3to1 mapping
    aa_name_to_id = {
        'ALA': 0, 'ARG': 1, 'ASN': 2, 'ASP': 3, 'CYS': 4,
        'GLN': 5, 'GLU': 6, 'GLY': 7, 'HIS': 8, 'ILE': 9,
        'LEU': 10, 'LYS': 11, 'MET': 12, 'PHE': 13, 'PRO': 14,
        'SER': 15, 'THR': 16, 'TRP': 17, 'TYR': 18, 'VAL': 19
    }
    return aa_name_to_id.get(aa_name)

def coordinates_to_pdb(coordinates, aatype, output_path, protein_name="PROTEIN", chain_id="A", ca_only=True):
    """
    Convert reconstructed coordinate sequence to PDB file
    Args:
        coordinates: [num_residues, 3, num_frames] reconstructed CA atom coordinate sequence
        aatype: [num_residues] amino acid type ID (0-19)
        output_path: output PDB file path
        protein_name: protein name
        chain_id: chain ID
        ca_only: whether to write only CA atoms (recommended, since reconstructed coordinates are CA coordinates)
    """
    num_residues, _, num_frames = coordinates.shape
    
    with open(output_path, 'w') as f:
        f.write("REMARK  Generated by ProteinSpectrumFAR Reconstruction\n")
        f.write(f"REMARK  Protein: {protein_name}\n")
        f.write(f"REMARK  Number of residues: {num_residues}\n")
        f.write(f"REMARK  Number of frames: {num_frames}\n")
        f.write(f"TITLE     {protein_name} FAR Reconstruction\n")
        
        for frame_idx in range(num_frames):
            f.write(f"MODEL     {frame_idx + 1:4d}\n")
            
            atom_idx = 1
            for res_idx in range(num_residues):
                ca_coord = coordinates[res_idx, :, frame_idx].cpu().numpy()
                aa_name = amino_acid_id_to_name(aatype[res_idx].item())
                
                if ca_only:
                    # Write only CA atoms (recommended) - directly use reconstructed coordinates
                    f.write(f"ATOM  {atom_idx:5d}  CA  {aa_name} {chain_id}{res_idx + 1:4d}    "
                           f"{ca_coord[0]:8.3f}{ca_coord[1]:8.3f}{ca_coord[2]:8.3f}  1.00  0.00           C\n")
                    atom_idx += 1
                else:
                    # Write complete backbone atoms (CA, C, N, O)
                    # Note: C, N, O coordinates here are estimated, not reconstructed
                    # Write CA atom (reconstructed coordinates)
                    f.write(f"ATOM  {atom_idx:5d}  CA  {aa_name} {chain_id}{res_idx + 1:4d}    "
                           f"{ca_coord[0]:8.3f}{ca_coord[1]:8.3f}{ca_coord[2]:8.3f}  1.00  0.00           C\n")
                    atom_idx += 1
                    
                    # Write C atom (estimated coordinates)
                    c_coord = ca_coord + np.array([1.53, 0.0, 0.0])  # estimated position
                    f.write(f"ATOM  {atom_idx:5d}  C   {aa_name} {chain_id}{res_idx + 1:4d}    "
                           f"{c_coord[0]:8.3f}{c_coord[1]:8.3f}{c_coord[2]:8.3f}  1.00  0.00           C\n")
                    atom_idx += 1
                    
                    # Write N atom (estimated coordinates)
                    n_coord = ca_coord + np.array([-1.33, 0.0, 0.0])  # estimated position
                    f.write(f"ATOM  {atom_idx:5d}  N   {aa_name} {chain_id}{res_idx + 1:4d}    "
                           f"{n_coord[0]:8.3f}{n_coord[1]:8.3f}{n_coord[2]:8.3f}  1.00  0.00           N\n")
                    atom_idx += 1
                    
                    # Write O atom (estimated coordinates)
                    o_coord = c_coord + np.array([0.0, 1.24, 0.0])  # estimated position
                    f.write(f"ATOM  {atom_idx:5d}  O   {aa_name} {chain_id}{res_idx + 1:4d}    "
                           f"{o_coord[0]:8.3f}{o_coord[1]:8.3f}{o_coord[2]:8.3f}  1.00  0.00           O\n")
                    atom_idx += 1
            
            f.write("TER\n")
            f.write("ENDMDL\n")
        
        f.write("END\n")
    
    print(f"PDB file saved to: {output_path}")
    print(f"Mode: {'CA atoms only' if ca_only else 'Complete backbone atoms'}")
    print(f"Coordinate range: X[{coordinates[:, 0, :].min():.3f}, {coordinates[:, 0, :].max():.3f}], "
          f"Y[{coordinates[:, 1, :].min():.3f}, {coordinates[:, 1, :].max():.3f}], "
          f"Z[{coordinates[:, 2, :].min():.3f}, {coordinates[:, 2, :].max():.3f}]")


def reconstruct_and_save_pdb(far_model, dataloader, trajectory_data_dict, device,
                            output_dir, protein_indices=None, num_frames=100, 
                            save_gt=True, cfg=2.0, temperature=1.0, 
                            sample_method='nomask', num_samples=1, seed=42):
    """
    Reconstruct trajectories and save as PDB files
    
    Args:
        far_model: ProteinSpectrumFAR model
        dataloader: test data loader
        trajectory_data_dict: 从ProteinMotionDataset.load_full_trajectory_data()加载的轨迹数据字典
        device: device
        output_dir: output directory
        protein_indices: protein indices to save
        num_frames: number of frames to reconstruct
        save_gt: whether to save ground truth trajectories
        cfg: classifier-free guidance scale
        temperature: sampling temperature
        sample_method: sampling method
        num_samples: number of samples per protein
        seed: random seed
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if protein_indices is None:
        protein_indices = list(range(min(10, len(dataloader))))
    
    print("Starting trajectory reconstruction and saving as PDB files...")
    print(f"Processing protein indices: {protein_indices}")
    print(f"Output directory: {output_dir}")
    
    dataloader_list = list(dataloader)
    
    for batch_idx in protein_indices:
        if batch_idx >= len(dataloader_list):
            break
            
        print(f"\nProcessing protein {batch_idx + 1}/{len(protein_indices)}...")
        
        # Get data
        batch = dataloader_list[batch_idx]
        packed_graph, coords, spectrum_gt, rotation_matrices, aatype, aa_mask, esm_features = batch
        
        # Move data to device
        coords = coords.to(device)
        rotation_matrices = rotation_matrices.to(device)
        aatype = aatype.to(device)
        aa_mask = aa_mask.to(device)
        esm_features = esm_features.to(device)
        
        num_residues = coords.shape[0]
        seq_len = spectrum_gt.shape[1] // 3
        
        # Use FAR model to sample spectrum
        with torch.no_grad():
            if num_samples > 1:
                reconstructed_spectra = sample_multiple(
                    model=far_model,
                    coords=coords,
                    rotation_matrices=rotation_matrices,
                    aatype=aatype,
                    aa_mask=aa_mask,
                    esm_features=esm_features,
                    n_samples=num_samples,
                    num_iter=seq_len,
                    cfg=cfg,
                    temperature=temperature,
                    progress=False,
                    sample_method=sample_method,
                    seed=seed + batch_idx
                )
                reconstructed_spectrum = reconstructed_spectra.mean(dim=0)
            else:
                if sample_method == 'nomask':
                    reconstructed_spectrum = far_model.sample_nomask(
                        coords=coords,
                        rotation_matrices=rotation_matrices,
                        aatype=aatype,
                        aa_mask=aa_mask,
                        esm_features=esm_features,
                        num_iter=seq_len,
                        cfg=cfg,
                        temperature=temperature,
                        progress=False
                    )
                elif sample_method == 'mask':
                    reconstructed_spectrum = far_model.sample_mask(
                        coords=coords,
                        rotation_matrices=rotation_matrices,
                        aatype=aatype,
                        aa_mask=aa_mask,
                        esm_features=esm_features,
                        num_iter=seq_len,
                        cfg=cfg,
                        temperature=temperature,
                        progress=False
                    )
                else:
                    reconstructed_spectrum = far_model.sample(
                        coords=coords,
                        rotation_matrices=rotation_matrices,
                        aatype=aatype,
                        aa_mask=aa_mask,
                        esm_features=esm_features,
                        num_iter=seq_len,
                        cfg=cfg,
                        temperature=temperature,
                        progress=False
                    )
        reconstructed_spectrum = reconstructed_spectrum.squeeze(-1) # (num_residues, seq_len, 6) or (num_residues, seq_len, 3)

        # Get original trajectory data from the preloaded dictionary
        protein_id = map_ids[batch_idx]
        protein_data = trajectory_data_dict[protein_id]
        
        # Get frame 0 positions
        frame0_positions = torch.tensor(np.stack([
            protein_data['positions']['x0'],
            protein_data['positions']['y0'],
            protein_data['positions']['z0']
        ], axis=-1), dtype=torch.float32).to(device)
        
        # Get ground truth trajectory by reconstructing from spectrum
        if save_gt:
            gt_coords = ProteinMotionDataset.reconstruct_trajectory_from_spectrum(
                protein_data, scaling_factor, num_frames=num_frames
            ).to(device)
            # Convert to [num_residues, 3, num_frames] format
            gt_coords = gt_coords.permute(1, 2, 0).contiguous()  # [num_frames, num_residues, 3] -> [num_residues, 3, num_frames]
        
        # Convert reconstructed spectrum to coordinates
        reconstructed_coords = spectrum_to_coordinates(
            reconstructed_spectrum, frame0_positions, scaling_factor, num_frames=num_frames
        )
        
        print(f"  Reconstructed coordinate shape: {reconstructed_coords.shape}")
        
        # Save PDB files
        protein_name = f"protein_{protein_id}"
        reconstructed_output_path = os.path.join(output_dir, f"{protein_name}_reconstructed.pdb")
        
        coordinates_to_pdb(
            coordinates=reconstructed_coords.cpu(),
            aatype=aatype.cpu(),
            output_path=reconstructed_output_path,
            protein_name=f"{protein_name}_reconstructed",
            chain_id="A",
            ca_only=True
        )
        
        if save_gt:
            gt_output_path = os.path.join(output_dir, f"{protein_name}_ground_truth.pdb")
            coordinates_to_pdb(
                coordinates=gt_coords.cpu(),
                aatype=aatype.cpu(),
                output_path=gt_output_path,
                protein_name=f"{protein_name}_ground_truth",
                chain_id="A",
                ca_only=True
            )
            print(f"  Ground truth trajectory saved: {gt_output_path}")
        
            # Calculate metrics
            mse = torch.mean((reconstructed_coords.cpu() - gt_coords.cpu()) ** 2)
            rmse = torch.sqrt(mse)
            mae = torch.mean(torch.abs(reconstructed_coords.cpu() - gt_coords.cpu()))
            print(f"  Reconstruction quality - RMSE: {rmse.item():.4f}Å, MAE: {mae.item():.4f}Å")
        
        print(f"  Reconstructed trajectory saved: {reconstructed_output_path}")
    
    print(f"\nPDB file saving completed! Processed {len(protein_indices)} proteins")


def main():
    parser = argparse.ArgumentParser(description='ProteinSpectrumFAR model inference')
    parser.add_argument('--checkpoint_path', type=str, 
                       default="./checkpoints/protein_far.ckpt",
                       help='FAR model checkpoint path')
    parser.add_argument('--data_path', type=str, 
                       default="./dataset/atlas_10_features_fouriered_list_test.pth",
                       help='Test data path (用于inference和加载ground truth轨迹)')
    parser.add_argument('--output_dir', type=str, 
                       default='./inference_results', 
                       help='Output directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--model_size', type=str, default='base', 
                       choices=['base', 'large', 'huge'],
                       help='Model size')
    
    # Sampling parameters
    parser.add_argument('--cfg', type=float, default=2.0, 
                       help='Classifier-free guidance scale')
    parser.add_argument('--temperature', type=float, default=1.0, 
                       help='Sampling temperature')
    parser.add_argument('--sample_method', type=str, default='nomask', 
                       choices=['sample', 'nomask', 'mask'],
                       help='Sampling method')
    parser.add_argument('--num_samples', type=int, default=1, 
                       help='Number of samples per protein (for ensemble)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # PDB saving parameters
    parser.add_argument('--save_pdb', action='store_true', 
                       help='Whether to save reconstructed trajectories as PDB files')
    parser.add_argument('--pdb_output_dir', type=str, 
                       default='./reconstructed_pdbs_far', 
                       help='PDB file output directory')
    parser.add_argument('--num_pdb_proteins', type=int, default=100, 
                       help='Number of proteins to save PDB files for')
    parser.add_argument('--num_frames', type=int, default=100, 
                       help='Number of reconstructed frames')
    parser.add_argument('--save_gt', action='store_true', default=True, 
                       help='Whether to save ground truth trajectories')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Check files
    if not os.path.exists(args.checkpoint_path):
        raise FileNotFoundError(f"Checkpoint does not exist: {args.checkpoint_path}")
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Data does not exist: {args.data_path}")
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load trajectory data from the same data_path (for ground truth)
    print(f"Loading trajectory data from: {args.data_path}")
    trajectory_data_dict = ProteinMotionDataset.load_full_trajectory_data(args.data_path)
    print(f"Loaded {len(trajectory_data_dict)} proteins' trajectory data")
    
    # Load checkpoint first to read model config
    print(f"Loading checkpoint: {args.checkpoint_path}")
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    
    # Try to read model config from checkpoint
    model_config = None
    if 'hyper_parameters' in checkpoint and 'model_config' in checkpoint['hyper_parameters']:
        model_config = checkpoint['hyper_parameters']['model_config']
        print(f"Found model config in checkpoint:")
        for key, value in model_config.items():
            print(f"  {key}: {value}")
    else:
        print("Warning: No model config found in checkpoint, using default/fallback configuration")
    
    # Create FAR model
    print(f"\nCreating ProteinSpectrumFAR model ({args.model_size})...")
    
    if args.model_size == 'base':
        if model_config is not None:
            # 使用checkpoint中保存的配置
            print("Using model config from checkpoint")
            far_model = protein_spectrum_far_base(**model_config)
        else:
            # Fallback: 使用与训练时一致的配置（见train_far.py line 173-188）
            print("Using fallback configuration (train_far.py compatible)")
            far_model = protein_spectrum_far_base(
                spectrum_token_dim=1,
                num_xyz_channels=3,
                aatype_vocab_size=21,
                esm_dim=1280,
                encoder_depth=4,
                encoder_num_heads=8,  # 训练时使用的是8，不是默认的16
                decoder_embed_dim=512,  # 训练时使用的是512，不是默认的768
                decoder_depth=4,
                decoder_num_heads=8,  # 训练时使用的是8，不是默认的12
                buffer_size=32,
                diffloss_d=2,
                diffloss_w=512,
                num_sampling_steps='50',
                diffusion_batch_mul=1,
            )
    elif args.model_size == 'large':
        if model_config is not None:
            far_model = protein_spectrum_far_large(**model_config)
        else:
            far_model = protein_spectrum_far_large()
    else:  # huge
        if model_config is not None:
            far_model = protein_spectrum_far_huge(**model_config)
        else:
            far_model = protein_spectrum_far_huge()
    
    # Load state dict
    if 'state_dict' in checkpoint:
        # PyTorch Lightning format - need to handle 'model.' prefix
        state_dict = checkpoint['state_dict']
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('model.'):
                new_state_dict[k[6:]] = v  # Remove 'model.' prefix
            else:
                new_state_dict[k] = v
        far_model.load_state_dict(new_state_dict)
    else:
        far_model.load_state_dict(checkpoint)
    
    far_model = far_model.to(device)
    far_model.eval()
    
    print(f"Model loaded successfully!")
    print(f"Model parameters: {sum(p.numel() for p in far_model.parameters()):,}")
    
    # Load test data
    print(f"Loading test data: {args.data_path}")
    dataloader = load_test_data(args.data_path, batch_size=1)
    print(f"Test dataset size: {len(dataloader.dataset)}")
    
    print(f"\nSampling parameters:")
    print(f"  Method: {args.sample_method}")
    print(f"  CFG scale: {args.cfg}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Samples per protein: {args.num_samples}")
    print(f"  Random seed: {args.seed}")
    
    # Evaluate FAR reconstruction
    results = evaluate_far_reconstruction(
        far_model, dataloader, trajectory_data_dict, device,
        num_samples=args.num_samples,
        cfg=args.cfg,
        temperature=args.temperature,
        sample_method=args.sample_method,
        seed=args.seed
    )
    
    # Print summary
    print_far_summary(results)
    
    # Save results
    save_far_results(results, os.path.join(args.output_dir, 'far_detailed_results.pth'))
    
    # Plot results
    plot_far_results(results, args.output_dir)
    
    # Save PDB files if requested
    if args.save_pdb:
        print("\nStarting to save reconstructed trajectories as PDB files...")
        reconstruct_and_save_pdb(
            far_model, dataloader, trajectory_data_dict, device,
            args.pdb_output_dir, 
            protein_indices=list(range(args.num_pdb_proteins)),
            num_frames=args.num_frames,
            save_gt=args.save_gt,
            cfg=args.cfg,
            temperature=args.temperature,
            sample_method=args.sample_method,
            num_samples=args.num_samples,
            seed=args.seed
        )
    
    print(f"\nFAR inference completed! Results saved in: {args.output_dir}")
    if args.save_pdb:
        print(f"PDB files saved in: {args.pdb_output_dir}")


if __name__ == "__main__":
    main() 
 