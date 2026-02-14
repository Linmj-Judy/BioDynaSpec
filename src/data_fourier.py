import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import json
import argparse
from tqdm import tqdm

def load_features_list(features_path):
    """
    Load features_list data (consistent with inference_vae.py)
    """
    features_list = torch.load(features_path)
    return features_list

def extract_trajectory_from_protein_data(protein_data):
    """
    Extract trajectory data from protein_data
    Args:
        protein_data: protein data loaded from features_list
    Returns:
        trajectory: [N, T, 3] trajectory data
        frame0_positions: [N, 3] frame 0 positions
    """
    # Get positions from all frames
    positions_list = [frame['positions'] for frame in protein_data]
    trajectory = torch.stack(positions_list, dim=0)  # [T, N, 3]
    
    # Convert to [N, T, 3] format
    trajectory = trajectory.permute(1, 0, 2)  # [N, T, 3]
    
    # Get frame 0 positions
    frame0_positions = trajectory[:, 0, :]  # [N, 3]
    
    return trajectory.numpy(), frame0_positions.numpy()

# # lmj: 废弃，早期画图用
# def compute_displacement_fft(trajectory):
#     """
#     Calculate displacement relative to frame 0 and perform FFT
#     Args:
#         trajectory: [N, T, 3] trajectory data, N residues, T frames, 3 coordinate dimensions
#     Returns:
#         fft_result: [N, 3, T//2+1] complex FFT result (only positive frequency part)
#         frame0_positions: [N, 3] frame 0 positions
#         amplitudes: [N, 3, T//2+1] amplitude spectrum
#     """
#     N, T, _ = trajectory.shape
    
#     # Get frame 0 positions
#     frame0_positions = trajectory[:, 0, :]  # [N, 3]
    
#     # Calculate displacement relative to frame 0
#     displacement = trajectory[:, :, :] - frame0_positions[:, None, :]  # [N, T, 3]
    
#     # Perform FFT for each residue and each dimension
#     fft_result = np.zeros((N, 3, T//2+1), dtype=np.complex128)  # T//2+1 = (T+1)//2
#     amplitudes = np.zeros((N, 3, T//2+1))
    
#     for i in range(N):
#         for dim in range(3):
#             # Perform FFT on displacement sequence
#             fft_vals = np.fft.rfft(displacement[i, :, dim])
#             fft_result[i, dim, :] = fft_vals
#             # Calculate amplitude
#             amplitudes[i, dim, :] = np.abs(fft_vals)
    
#     return fft_result, frame0_positions, amplitudes

def compute_global_scaling_factors_from_spectra(saved_spectra, num_freq_components=None):
    """
    Calculate global scaling factors from saved spectra (using 95th percentile amplitude)
    支持不同长度的频谱，使用完整频谱计算（不截断）
    Args:
        saved_spectra: list of full-length spectra from all samples
        num_freq_components: 如果指定，返回前num_freq_components个scaling factors；
                            如果为None，返回所有频率的scaling factors
    Returns:
        scaling_factors: [max_freq_len or num_freq_components] global scaling factors
    """
    # 首先找到最大的频率长度
    max_freq_len = 0
    for sample_spectra in saved_spectra:
        for x_hat, y_hat, z_hat in sample_spectra:
            max_freq_len = max(max_freq_len, len(x_hat), len(y_hat), len(z_hat))
    
    print(f"Maximum frequency length found: {max_freq_len}")
    
    # 为每个频率分量收集所有幅值（支持不同长度）
    freq_magnitudes = [[] for _ in range(max_freq_len)]
    
    for sample_spectra in saved_spectra:
        for x_hat, y_hat, z_hat in sample_spectra:
            # 处理x, y, z三个维度，使用完整频谱长度
            for dim_hat in [x_hat, y_hat, z_hat]:
                # 遍历该频谱的完整长度
                for freq_idx in range(len(dim_hat)):
                    freq_magnitudes[freq_idx].append(np.abs(dim_hat[freq_idx]))
    
    # 计算每个频率分量的95百分位幅值作为scaling factor
    all_scaling_factors = []
    for freq_idx in range(max_freq_len):
        if len(freq_magnitudes[freq_idx]) > 0:
            scaling_factor = np.percentile(freq_magnitudes[freq_idx], 95)
            all_scaling_factors.append(scaling_factor)
            if freq_idx < 20 or freq_idx % 10 == 0:  # 只打印部分频率避免输出过多
                print(f"Frequency {freq_idx}: {len(freq_magnitudes[freq_idx])} samples, 95th percentile = {scaling_factor:.6f}")
        else:
            # 如果某个频率没有数据（所有轨迹都太短），使用默认值
            scaling_factor = 1.0
            all_scaling_factors.append(scaling_factor)
            print(f"Frequency {freq_idx}: No data available, using default = {scaling_factor:.6f}")
    
    all_scaling_factors = np.array(all_scaling_factors)
    
    # 如果指定了num_freq_components，只返回前N个
    if num_freq_components is not None:
        if num_freq_components > len(all_scaling_factors):
            # 如果要求的长度超过实际长度，用1.0填充
            padding = np.ones(num_freq_components - len(all_scaling_factors))
            scaling_factors = np.concatenate([all_scaling_factors, padding])
            print(f"Warning: Requested {num_freq_components} components but only {len(all_scaling_factors)} available, padding with 1.0")
        else:
            scaling_factors = all_scaling_factors[:num_freq_components]
    else:
        scaling_factors = all_scaling_factors
    
    print(f"\nTotal scaling factors computed: {len(all_scaling_factors)}")
    print(f"Returned scaling factors length: {len(scaling_factors)}")
    print(f"First 10 scaling factors: {scaling_factors[:10]}")
    print(f"Last 10 scaling factors: {scaling_factors[-10:]}")
    
    return scaling_factors

def process_single_trajectory_to_spectra(trajectory):
    """
    Process single trajectory, extract no-truncated spectra
    Args:
        trajectory: [N, T, 3] trajectory data
        num_freq_components: number of frequency components
    Returns:
        sample_spectra: no-truncated spectra list for current sample
        frame0_positions: [N, 3] frame 0 positions
    """
    N, T, _ = trajectory.shape
    
    # Get frame 0 positions
    frame0_positions = trajectory[:, 0, :]  # [N, 3]
    
    # Calculate displacement relative to frame 0
    displacement = trajectory[:, :, :] - frame0_positions[:, None, :]  # [N, T, 3]
    
    # Store truncated spectra for all atoms in current sample
    sample_spectra = []
    
    # Iterate through position sequence for each atom
    for i in range(N):
        x_list = displacement[i, :, 0]  # x direction displacement
        y_list = displacement[i, :, 1]  # y direction displacement
        z_list = displacement[i, :, 2]  # z direction displacement
        
        # FFT for x, y, z 分别做FFT，独立
        x_hat = np.fft.rfft(x_list)
        y_hat = np.fft.rfft(y_list)
        z_hat = np.fft.rfft(z_list)
        
        x_amplitude = np.abs(x_hat)
        x_phase = np.angle(x_hat)
        y_amplitude = np.abs(y_hat)
        y_phase = np.angle(y_hat)
        z_amplitude = np.abs(z_hat)
        z_phase = np.angle(z_hat)
        
        # # Truncate first K frequency components and save
        # x_hat_truncated = x_hat[:num_freq_components]
        # y_hat_truncated = y_hat[:num_freq_components]
        # z_hat_truncated = z_hat[:num_freq_components]
        sample_spectra.append((x_hat, y_hat, z_hat))
    
    return sample_spectra, frame0_positions

def normalize_spectra_with_scaling_factors(saved_spectra, scaling_factors):
    """
    Normalize spectra using global scaling factors
    支持不同长度的频谱和scaling_factors
    Args:
        saved_spectra: list of full-length spectra from all samples
        scaling_factors: [K] global scaling factors, K可以是任意长度
    Returns:
        normalized_spectra: list of normalized spectra (保持原始长度)
    """
    normalized_spectra = []
    
    for sample_spectra in saved_spectra:
        sample_normalized = []
        for x_hat, y_hat, z_hat in sample_spectra:
            # Reverse derivation: reconstructed magnitude = (original magnitude)² × scale
            # So compressed magnitude = √(original magnitude / scale)
            # Direction remains unchanged
            
            # Calculate original magnitude
            x_magnitude = np.abs(x_hat)
            y_magnitude = np.abs(y_hat)
            z_magnitude = np.abs(z_hat)
            
            # 确定每个维度使用的scaling_factors长度
            # 如果频谱比scaling_factors长，超出部分使用最后一个scaling_factor
            # 如果频谱比scaling_factors短，只使用前面部分的scaling_factors
            def get_scaling_for_length(length):
                if length <= len(scaling_factors):
                    return scaling_factors[:length]
                else:
                    # 如果频谱更长，用最后一个值填充
                    padding = np.full(length - len(scaling_factors), scaling_factors[-1])
                    return np.concatenate([scaling_factors, padding])
            
            scale_x = get_scaling_for_length(len(x_hat))
            scale_y = get_scaling_for_length(len(y_hat))
            scale_z = get_scaling_for_length(len(z_hat))
            
            # Calculate normalized magnitude
            x_normalized_magnitude = np.sqrt(x_magnitude / scale_x)
            y_normalized_magnitude = np.sqrt(y_magnitude / scale_y)
            z_normalized_magnitude = np.sqrt(z_magnitude / scale_z)
            
            # Keep direction unchanged, only change magnitude
            x_normalized = x_hat / x_magnitude * x_normalized_magnitude
            y_normalized = y_hat / y_magnitude * y_normalized_magnitude
            z_normalized = z_hat / z_magnitude * z_normalized_magnitude
            
            # Handle zero magnitude case
            x_normalized = np.where(x_magnitude == 0, 0, x_normalized)
            y_normalized = np.where(y_magnitude == 0, 0, y_normalized)
            z_normalized = np.where(z_magnitude == 0, 0, z_normalized)
            
            sample_normalized.append((x_normalized, y_normalized, z_normalized))
        normalized_spectra.append(sample_normalized)
    
    return normalized_spectra

def convert_spectra_to_spectrum_format(normalized_spectra, num_freq_components=None):
    """
    Convert normalized spectra to [N, K*6] format
    支持不同长度的频谱
    Args:
        normalized_spectra: list of normalized spectra (可能长度不同)
        num_freq_components: 目标频率分量数量，如果为None则使用该样本的最大频率长度
    Returns:
        spectra_list: list of spectra in [N, K*6] format
    """
    spectra_list = []
    
    for sample_normalized in normalized_spectra:
        N = len(sample_normalized)
        
        # 如果没有指定num_freq_components，自动确定该样本的最大频率长度
        if num_freq_components is None:
            max_len = 0
            for x_normalized, y_normalized, z_normalized in sample_normalized:
                max_len = max(max_len, len(x_normalized), len(y_normalized), len(z_normalized))
            target_len = max_len
        else:
            target_len = num_freq_components
        
        spectrum = np.zeros((N, target_len * 6))
        
        for i, (x_normalized, y_normalized, z_normalized) in enumerate(sample_normalized):
            # 获取实际的频率长度（可能小于target_len）
            actual_len = min(len(x_normalized), target_len)
            
            for freq_idx in range(actual_len):
                # Convert to real and imaginary parts
                spectrum[i, freq_idx*6 + 0] = x_normalized[freq_idx].real  # x real part
                spectrum[i, freq_idx*6 + 1] = x_normalized[freq_idx].imag  # x imaginary part
                spectrum[i, freq_idx*6 + 2] = y_normalized[freq_idx].real  # y real part
                spectrum[i, freq_idx*6 + 3] = y_normalized[freq_idx].imag  # y imaginary part
                spectrum[i, freq_idx*6 + 4] = z_normalized[freq_idx].real  # z real part
                spectrum[i, freq_idx*6 + 5] = z_normalized[freq_idx].imag  # z imaginary part
            
            # 如果actual_len < target_len，剩余部分已经是0（通过np.zeros初始化）
        
        spectra_list.append(spectrum)
    
    return spectra_list

def spectrum_to_coords(spectrum, num_freq_components=None):
    """
    Convert [N, K*6] spectrum to [N, 3, 2, K] coordinate form
    Args:
        spectrum: tensor or numpy array with shape [N, K*6]
        num_freq_components: 频率分量数量，如果为None则自动从spectrum形状推断
    Returns:
        coords_complex: tensor or numpy array with shape [N, 3, 2, K],
                       where coords_complex[:, 0/1/2, 0] are real parts of x/y/z
                       coords_complex[:, 0/1/2, 1] are imaginary parts of x/y/z
    """
    if isinstance(spectrum, torch.Tensor):
        device = spectrum.device
        spectrum = spectrum.cpu().numpy()
        return_tensor = True
    else:
        return_tensor = False
    
    N = spectrum.shape[0]
    # TODO：# TODO：模型同时预测出实部和虚部，但这里没有加上实部和虚部是反对称的这个硬约束条件，所以应该只预测出实部，然后再根据反共轭给出虚部（要写一个根据T//2+1个独立频率数获得其他所有频率的函数），所以K实际上就是T
    
    # 如果没有指定num_freq_components，从spectrum形状自动推断
    if num_freq_components is None:
        assert spectrum.shape[1] % 6 == 0, f"Spectrum shape[1]={spectrum.shape[1]} must be divisible by 6"
        num_freq_components = spectrum.shape[1] // 6
    
    # First reshape K*6 dimensions to [N, K, 6]
    spectrum = spectrum.reshape(N, num_freq_components, 6)
    
    # Reorganize data
    coords_complex = np.zeros((N, 3, 2, num_freq_components))
    
    # x dimension
    coords_complex[:, 0, 0, :] = spectrum[:, :, 0]  # x real part
    coords_complex[:, 0, 1, :] = spectrum[:, :, 1]  # x imaginary part
    # y dimension
    coords_complex[:, 1, 0, :] = spectrum[:, :, 2]  # y real part
    coords_complex[:, 1, 1, :] = spectrum[:, :, 3]  # y imaginary part
    # z dimension
    coords_complex[:, 2, 0, :] = spectrum[:, :, 4]  # z real part
    coords_complex[:, 2, 1, :] = spectrum[:, :, 5]  # z imaginary part
    
    if return_tensor:
        return torch.from_numpy(coords_complex).to(device)
    else:
        return coords_complex

def inverse_normalize_and_ifft(data_normalized, num_frames, scale):
    """
    Inverse normalize and perform inverse Fourier transform
    支持不同长度的频谱重建
    Args:
        data_normalized: [2, K] normalized data (real and imaginary parts), K是频率分量数
        num_frames: 目标时域长度（帧数T）
        scale: [K] scaling factors for each frequency component
    Returns:
        reconstructed displacement sequence: [num_frames] array
    """
    # Separate real and imaginary parts
    real_normalized = data_normalized[0, ...]
    imag_normalized = data_normalized[1, ...]
    # Merge real and imaginary parts
    data_fft = real_normalized + 1j * imag_normalized
    
    # 获取实际的频率分量长度
    actual_freq_len = len(data_fft)
    
    # 只使用对应长度的scale
    scale_used = scale[:actual_freq_len] if len(scale) >= actual_freq_len else scale
    
    # Inverse normalize
    # 注意：避免除以0
    magnitude = np.abs(data_fft)
    data_fft_denorm = np.where(
        magnitude > 0,
        (magnitude ** 2) * scale_used * (data_fft / magnitude),
        0
    )
    
    # 执行逆FFT，指定输出长度为num_frames
    # irfft会自动处理padding（如果频域数据不足）或截断（如果超出）
    return np.fft.irfft(data_fft_denorm, n=num_frames)

def spectrum_to_coordinates(spectrum, frame0_positions, scaling_factors, num_frames=100, num_freq_components=None):
    """
    Convert spectrum to coordinate sequence
    支持不同长度频谱的重建
    Args:
        spectrum: [num_residues, K*6] spectrum
        frame0_positions: [num_residues, 3] frame 0 positions
        scaling_factors: [K] scaling factors
        num_frames: 目标时域长度（帧数T）
        num_freq_components: 频率分量数量，如果为None则自动从spectrum推断
    Returns:
        trajectory: [num_residues, num_frames, 3] reconstructed coordinate sequence
    """
    num_residues = spectrum.shape[0]
    
    # Convert to complex coordinate form (自动推断频率数量)
    coords_complex = spectrum_to_coords(spectrum, num_freq_components)  # [N, 3, 2, K]
    
    # Initialize trajectory array
    trajectory = np.zeros((num_residues, num_frames, 3))
    
    # Process x,y,z three dimensions separately for each residue
    for i in range(num_residues):
        # 对每个残基的x, y, z分别做逆FFT重建
        trajectory[i, :, 0] = inverse_normalize_and_ifft(coords_complex[i, 0, :], num_frames, scaling_factors)
        trajectory[i, :, 1] = inverse_normalize_and_ifft(coords_complex[i, 1, :], num_frames, scaling_factors)
        trajectory[i, :, 2] = inverse_normalize_and_ifft(coords_complex[i, 2, :], num_frames, scaling_factors)
    
    # 第0帧应该是初始位置（位移为0）
    # 但由于频域截断/padding，可能第0帧不完全为0，所以强制设为0
    trajectory[:, 0, :] = 0
    
    # Add displacement to initial coordinates
    for t in range(num_frames):
        trajectory[:, t, :] = frame0_positions + trajectory[:, t, :]
    
    return trajectory

def compute_rmse(original_trajectory, reconstructed_trajectory):
    """
    Calculate RMSE
    Args:
        original_trajectory: [N, num_frames, 3] original trajectory
        reconstructed_trajectory: [N, num_frames, 3] reconstructed trajectory
    Returns:
        rmse: overall RMSE
        rmse_per_residue: [N] RMSE for each residue
        rmse_per_frame: [num_frames] RMSE for each frame
    """
    # Calculate squared error for each residue and each frame
    squared_error = (original_trajectory - reconstructed_trajectory) ** 2  # [N, num_frames, 3]
    
    # Overall RMSE
    rmse = np.sqrt(np.mean(squared_error))
    
    # RMSE for each residue
    rmse_per_residue = np.sqrt(np.mean(squared_error, axis=(1, 2)))  # [N]
    
    # RMSE for each frame
    rmse_per_frame = np.sqrt(np.mean(squared_error, axis=(0, 2)))  # [num_frames]
    
    return rmse, rmse_per_residue, rmse_per_frame

def process_multiple_trajectories_from_features_list(features_list, map_ids, num_freq_components=16): # lmj: used this in main function
    """
    Process multiple trajectories from features_list, calculate global scaling factors and normalized spectra
    Args:
        features_list: data loaded from features_list.pth
        map_ids: list of protein IDs to process
        num_freq_components: number of frequency components
    Returns:
        global_scaling_factors: [num_freq_components] global scaling factors
        all_spectra: list of all spectra
        all_frame0_positions: list of all frame 0 positions
        all_normalized_spectra: list of all normalized spectra
        all_trajectories: list of all original trajectories
    """
    saved_spectra = []
    all_frame0_positions = []
    all_trajectories = []
    
    print("First pass: calculate all Fourier transforms and save full-length spectra...")
    for protein_id in tqdm(map_ids):
        # Get protein data
        # protein_data = features_list[protein_id]
        protein_data = torch.load(features_list[protein_id])
        
        # Extract trajectory data
        trajectory, frame0_positions = extract_trajectory_from_protein_data(protein_data)
        all_trajectories.append(trajectory)
        all_frame0_positions.append(frame0_positions)
        
        # Process single trajectory, extract full-length spectra
        sample_spectra, _ = process_single_trajectory_to_spectra(trajectory)
        saved_spectra.append(sample_spectra)
    
    # Calculate global scaling factors
    print("Calculate global scaling factors (95th percentile amplitude)...")
    global_scaling_factors = compute_global_scaling_factors_from_spectra(saved_spectra, num_freq_components)
    
    # Normalize using global scaling factors
    print("Normalize using global scaling factors...")
    all_normalized_spectra = normalize_spectra_with_scaling_factors(
        saved_spectra, global_scaling_factors
    )
    
    # Convert to [N, num_freq_components*6] format
    print("Convert to spectrum format...")
    all_spectra = convert_spectra_to_spectrum_format(all_normalized_spectra, num_freq_components)
    
    return global_scaling_factors, all_spectra, all_frame0_positions, all_normalized_spectra, all_trajectories

def process_single_trajectory_with_global_scaling(trajectory, global_scaling_factors, num_freq_components=None):
    """
    Process single trajectory using global scaling factors
    Args:
        trajectory: [N, T, 3] trajectory data
        global_scaling_factors: [K] global scaling factors
        num_freq_components: 可选，如果指定则将频谱截断/填充到该长度；如果为None则使用完整频谱
    Returns:
        spectrum: [N, K*6] spectrum
        frame0_positions: [N, 3] frame 0 positions
        reconstructed_trajectory: [N, T, 3] reconstructed trajectory (for validation)
    """
    # Extract full-length spectra
    sample_spectra, frame0_positions = process_single_trajectory_to_spectra(trajectory)
    
    # Normalize using global scaling factors
    normalized_spectra = normalize_spectra_with_scaling_factors([sample_spectra], global_scaling_factors)
    
    # Convert to [N, K*6] format
    spectra_list = convert_spectra_to_spectrum_format(normalized_spectra, num_freq_components)
    spectrum = spectra_list[0]
    
    # Reconstruct trajectory for validation (自动从spectrum推断频率数量)
    reconstructed_trajectory = spectrum_to_coordinates(
        spectrum, frame0_positions, global_scaling_factors, num_frames=trajectory.shape[1]
    )
    
    return spectrum, frame0_positions, reconstructed_trajectory

def visualize_spectrum_analysis(spectrum, frame0_positions, scaling_factors, 
                               original_trajectory, reconstructed_trajectory, save_path=None, num_freq_components=None):
    """
    Visualize spectrum analysis results
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 自动推断频率数量
    if num_freq_components is None:
        assert spectrum.shape[1] % 6 == 0, "Spectrum shape[1] must be divisible by 6"
        num_freq_components = spectrum.shape[1] // 6
    
    # Select first residue for visualization
    residue_idx = 0
    orig_residue = original_trajectory[:, residue_idx, :]
    recon_residue = reconstructed_trajectory[:, residue_idx, :]
    
    # Plot X coordinate over time
    axes[0, 0].plot(orig_residue[:, 0], label='Original X', alpha=0.8, linewidth=2)
    axes[0, 0].plot(recon_residue[:, 0], label='Reconstructed X', alpha=0.7)
    axes[0, 0].set_title('X-coordinate over Time (Residue 0)')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Plot Y coordinate over time
    axes[0, 1].plot(orig_residue[:, 1], label='Original Y', alpha=0.8, linewidth=2)
    axes[0, 1].plot(recon_residue[:, 1], label='Reconstructed Y', alpha=0.7)
    axes[0, 1].set_title('Y-coordinate over Time (Residue 0)')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Plot Z coordinate over time
    axes[0, 2].plot(orig_residue[:, 2], label='Original Z', alpha=0.8, linewidth=2)
    axes[0, 2].plot(recon_residue[:, 2], label='Reconstructed Z', alpha=0.7)
    axes[0, 2].set_title('Z-coordinate over Time (Residue 0)')
    axes[0, 2].legend()
    axes[0, 2].grid(True)
    
    # Plot RMSE over time
    rmse_per_frame = np.sqrt(np.mean((orig_residue - recon_residue) ** 2, axis=1))
    axes[1, 0].plot(rmse_per_frame, label='RMSE', color='red')
    axes[1, 0].set_title('RMSE over Time (Residue 0)')
    axes[1, 0].set_xlabel('Frame')
    axes[1, 0].set_ylabel('RMSE (Å)')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Plot scaling factors magnitude
    scaling_magnitudes = np.abs(scaling_factors)
    axes[1, 1].plot(scaling_magnitudes, marker='o')
    axes[1, 1].set_title('Scaling Factors Magnitude')
    axes[1, 1].set_xlabel('Frequency Component')
    axes[1, 1].set_ylabel('Magnitude')
    axes[1, 1].grid(True)
    
    # Plot spectrum magnitude
    # spectrum格式: [freq0: x_real, x_imag, y_real, y_imag, z_real, z_imag, freq1: ...]
    spectrum_reshaped = spectrum[residue_idx].reshape(num_freq_components, 6)
    x_magnitude = np.sqrt(spectrum_reshaped[:, 0]**2 + spectrum_reshaped[:, 1]**2)
    y_magnitude = np.sqrt(spectrum_reshaped[:, 2]**2 + spectrum_reshaped[:, 3]**2)
    z_magnitude = np.sqrt(spectrum_reshaped[:, 4]**2 + spectrum_reshaped[:, 5]**2)
    
    axes[1, 2].plot(x_magnitude, marker='o', label='X')
    axes[1, 2].plot(y_magnitude, marker='s', label='Y')
    axes[1, 2].plot(z_magnitude, marker='^', label='Z')
    axes[1, 2].set_title('Spectrum Magnitude (Residue 0)')
    axes[1, 2].set_xlabel('Frequency Component')
    axes[1, 2].set_ylabel('Magnitude')
    axes[1, 2].legend()
    axes[1, 2].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization results saved to: {save_path}")
    
    plt.show()

import json
from pathlib import Path
from typing import List, Any, Iterable

def load_list_from_json_file(p: Path, key: str = None) -> List[Any]:
    with p.open('r', encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if key is not None and isinstance(data, dict) and key in data and isinstance(data[key], list):
        return data[key]
    # 尝试在 dict 中找到第一个 list 值
    if isinstance(data, dict):
        for v in data.values():
            if isinstance(v, list):
                return v
    raise ValueError(f"No list found in file {p}")

def load_and_concat_json_lists(paths: Iterable[str], key: str = None, sort_paths: bool = True) -> List[Any]:
    """
    paths: 可迭代文件路径或目录
    key: 如果每个 JSON 文件中列表存放在某个键下，传入该键名
    sort_paths: 是否对 paths 进行排序（通常推荐 True）
    返回：合并后的列表（按 paths 的顺序拼接）
    """
    all_paths = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            # 加入目录下所有 .json 文件
            files = sorted([x for x in p.glob("*.json")])
            all_paths.extend(files)
        else:
            all_paths.append(p)

    if sort_paths:
        all_paths = sorted(all_paths, key=lambda x: str(x))

    concatenated: List[Any] = []
    for p in all_paths:
        try:
            lst = load_list_from_json_file(p, key=key)
            print(f"Loaded {len(lst)} items from {p}")
            concatenated.extend(lst)
        except Exception as e:
            print(f"Warning: failed to load list from {p}: {e}")
    return concatenated


def main():
    parser = argparse.ArgumentParser(description='Extract spectrum and scaling factors from real trajectories (using 95th percentile amplitude)')
    # parser.add_argument('-f', '--features_list_path', type=str, default="/path/to/features_list_atlas", help='features_list path')
    parser.add_argument('--output_dir', type=str, default='./fourier_analysis_results', help='output directory')
    parser.add_argument('--num_freq_components', type=int, default=16, help='number of frequency components')
    parser.add_argument('--visualize', action='store_true', help='whether to generate visualization')
    parser.add_argument('--max_proteins', type=int, default=None, help='maximum number of proteins to process (for testing)')
    parser.add_argument('--data_list_path', type=str, default="/path/to/atlas_features_fouriered_list.pth", help='original data_list path')
    parser.add_argument('--output_scaling_path', type=str, default="/path/to/atlas_scaling_factors.pth", help='new scaling factors save path')
    parser.add_argument('--output_data_list_path', type=str, default="/path/to/atlas_features_fouriered_list.pth", help='new data_list save path')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load features_list
    print(f"Loading features_list: {json_dir}")
    json_dir = os.environ.get("ATLAS_JSON_DIR", "./dataset/atlas/")
    features_list = load_and_concat_json_lists([json_dir], key=None)
    name_2_id = {}
    for f in features_list:
        name_2_id[f.split('.')[0]] = f
    features_list = [os.path.join(args.features_list_path, f) for f in features_list if f.endswith('_R1.pth')]
    print(f"features_list contains {len(features_list)} proteins")
    
    # Load original data_list
    print(f"Loading original data_list: {args.data_list_path}")
    original_data_list = torch.load(args.data_list_path)
    original_data_list = [data for data in original_data_list if data['id'].endswith('_R1')]
    print(f"Original data_list contains {len(original_data_list)} proteins")
    
    # Create protein_id to index mapping
    protein_id_to_index = {}
    for i, item in enumerate(original_data_list):
        protein_id_to_index[item['id']] = i
    
    # Define protein IDs to process
    map_ids = list(range(len(features_list)))
    
    # Limit processing quantity (for testing)
    if args.max_proteins is not None:
        map_ids = map_ids[:args.max_proteins]
    
    print(f"Processing {len(map_ids)} proteins: {map_ids}")
    
    # Process multiple trajectories, calculate global scaling factors
    print("Processing trajectories and calculating global scaling factors...")
    global_scaling_factors, all_spectra, all_frame0_positions, all_normalized_spectra, all_trajectories = process_multiple_trajectories_from_features_list(
        features_list, map_ids, args.num_freq_components
    )
    
    # Calculate reconstruction quality for all trajectories
    print("Calculating reconstruction quality...")
    all_rmses = []
    all_spectra_processed = []
    all_frame0_positions_processed = []
    
    for i, (trajectory, spectrum, frame0_positions) in enumerate(zip(all_trajectories, all_spectra, all_frame0_positions)):
        # Reconstruct trajectory using global scaling factors
        # num_freq_components会从spectrum形状自动推断
        reconstructed_trajectory = spectrum_to_coordinates(
            spectrum, frame0_positions, global_scaling_factors, num_frames=trajectory.shape[1]
        )
        
        # Calculate reconstruction quality
        rmse, rmse_per_residue, rmse_per_frame = compute_rmse(trajectory, reconstructed_trajectory)
        all_rmses.append(rmse)
        
        print(f"Protein {map_ids[i]}: RMSE={rmse:.4f}Å")
        
        # Save processed data
        all_spectra_processed.append(spectrum)
        all_frame0_positions_processed.append(frame0_positions)
    
    # Calculate average RMSE
    avg_rmse = np.mean(all_rmses)
    print(f"\nAverage RMSE: {avg_rmse:.4f}Å")
    print(f"RMSE range: {np.min(all_rmses):.4f}Å - {np.max(all_rmses):.4f}Å")
    
    # Save new scaling factors as list format
    print(f"Saving new scaling factors: {args.output_scaling_path}")
    scaling_factors_list = global_scaling_factors.tolist()
    torch.save(scaling_factors_list, args.output_scaling_path)
    print(f"Scaling factors saved, total {len(scaling_factors_list)}")
    
    # Put normalized spectra back into data_list
    print("Putting normalized spectra back into data_list...")
    new_data_list = original_data_list.copy()
    
    for i, protein_id in enumerate(map_ids):
        # Get protein filename
        protein_file = features_list[protein_id]
        protein_name = os.path.basename(protein_file).split('.')[0]
        
        # Check if in original data_list
        if protein_name in protein_id_to_index:
            data_index = protein_id_to_index[protein_name]
            spectrum = all_spectra[i]  # [N, num_freq_components*6]
            
            # Convert [N, num_freq_components*6] to [N, num_freq_components, 6] format
            spectrum_reshaped = spectrum.reshape(spectrum.shape[0], args.num_freq_components, 6)
            
            # Organize data according to original format
            # spectrum_reshaped[:, :, 0] = x real part, spectrum_reshaped[:, :, 1] = x imaginary part
            # spectrum_reshaped[:, :, 2] = y real part, spectrum_reshaped[:, :, 3] = y imaginary part
            # spectrum_reshaped[:, :, 4] = z real part, spectrum_reshaped[:, :, 5] = z imaginary part
            
            # Update positions in data_list
            new_data_list[data_index]['positions']['x'] = [
                np.column_stack([spectrum_reshaped[res_id, :, 0], spectrum_reshaped[res_id, :, 1]]) 
                for res_id in range(spectrum_reshaped.shape[0])
            ]
            new_data_list[data_index]['positions']['y'] = [
                np.column_stack([spectrum_reshaped[res_id, :, 2], spectrum_reshaped[res_id, :, 3]]) 
                for res_id in range(spectrum_reshaped.shape[0])
            ]
            new_data_list[data_index]['positions']['z'] = [
                np.column_stack([spectrum_reshaped[res_id, :, 4], spectrum_reshaped[res_id, :, 5]]) 
                for res_id in range(spectrum_reshaped.shape[0])
            ]
            
            print(f"Updated spectrum data for protein {protein_name}")
        else:
            print(f"Warning: protein {protein_name} not in original data_list")
    
    # Save new data_list
    print(f"Saving new data_list: {args.output_data_list_path}")
    torch.save(new_data_list, args.output_data_list_path)
    print(f"New data_list saved, contains {len(new_data_list)} proteins")
    
    # Print scaling factors information
    print("\nScaling Factors Information (95th percentile amplitude):")
    for i in range(len(global_scaling_factors)):
        print(f"  Frequency {i}: {global_scaling_factors[i]:.6f}")
    
    print(f"\nAnalysis complete!")
    print(f"New scaling factors saved to: {args.output_scaling_path}")
    print(f"New data_list saved to: {args.output_data_list_path}")

if __name__ == "__main__":
    main() 