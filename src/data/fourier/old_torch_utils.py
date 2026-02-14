import torch
import numpy as np


def spectrum_to_coords(spectrum):
    """
    Convert spectrum to coordinate complex form
    Args:
        spectrum: tensor with shape [N, seq_len*6] or [N, seq_len, 6]
    Returns:
        coords_complex: tensor with shape [N, 3, 2, seq_len]
    """
    N = spectrum.shape[0]
    
    # Determine seq_len and reshape if necessary
    if spectrum.dim() == 2:
        # [N, seq_len*6] -> [N, seq_len, 6]
        total_features = spectrum.shape[-1]
        if total_features % 6 != 0:
            raise ValueError(f"Spectrum second dimension ({total_features}) must be divisible by 6")
        seq_len = total_features // 6
        spectrum = spectrum.reshape(N, seq_len, 6)
    elif spectrum.dim() == 3:
        # [N, seq_len, 6] - already correct
        N, seq_len, features = spectrum.shape
        if features != 6:
            raise ValueError(f"Expected last dimension to be 6, got {features}")
    else:
        raise ValueError(f"Unexpected spectrum dimensions: {spectrum.shape}")
    
    # Reorganize data: [N, seq_len, 6] -> [N, 3, 2, seq_len]
    coords_complex = torch.zeros(N, 3, 2, seq_len, device=spectrum.device)
    
    # x dimension
    coords_complex[:, 0, 0, :] = spectrum[:, :, 0]  # x real part
    coords_complex[:, 0, 1, :] = spectrum[:, :, 1]  # x imaginary part
    # y dimension
    coords_complex[:, 1, 0, :] = spectrum[:, :, 2]  # y real part
    coords_complex[:, 1, 1, :] = spectrum[:, :, 3]  # y imaginary part
    # z dimension
    coords_complex[:, 2, 0, :] = spectrum[:, :, 4]  # z real part
    coords_complex[:, 2, 1, :] = spectrum[:, :, 5]  # z imaginary part
    
    return coords_complex


def inverse_normalize_and_ifft(data_normalized, raw_len, scale):
    """
    Inverse normalize and perform IFFT
    
    Args:
        data_normalized: [2, K] normalized data (real and imaginary parts), K is the number of frequency components
        raw_len: target time domain length (number of frames T)
        scale: [K] scaling factor for each frequency component (or shorter)
    
    Returns:
        Reconstructed displacement sequence: [raw_len] array
    """
    # Convert to numpy if it's a torch tensor
    if isinstance(data_normalized, torch.Tensor):
        data_normalized = data_normalized.detach().cpu().numpy()
    
    # Separate real and imaginary parts
    real_normalized = data_normalized[0, ...]
    imag_normalized = data_normalized[1, ...]
    # Combine real and imaginary parts
    data_fft = real_normalized + 1j * imag_normalized
    
    # Get actual frequency component length
    actual_freq_len = len(data_fft)
    
    # Only use the corresponding length of scale
    # If scale is shorter than data_fft, use the last value of scale to pad
    # If scale is longer than data_fft, only use the first part
    if len(scale) >= actual_freq_len:
        scale_used = scale[:actual_freq_len]
    else:
        # If scale is shorter, pad with the last value
        padding = np.full(actual_freq_len - len(scale), scale[-1])
        scale_used = np.concatenate([scale, padding])
    
    # Inverse normalization
    magnitude = np.abs(data_fft)
    # Avoid division by zero
    data_fft_denorm = np.where(
        magnitude > 0,
        (magnitude ** 2) * scale_used * (data_fft / magnitude),
        0
    )
    
    # Perform inverse FFT with specified output length
    return np.fft.irfft(data_fft_denorm, n=raw_len)

def spectrum_to_coordinates(spectrum, frame0_positions, scaling_factor, num_frames=100):
    """
    Convert spectrum to coordinate sequence
    Args:
        spectrum: [num_residues, seq_len, 6] spectrum
        frame0_positions: [num_residues, 3] positions of frame 0
        num_frames: total number of frames
    Returns:
        coordinates: [num_residues, 3, num_frames] reconstructed coordinate sequence
    """
    # Convert to numpy
    spectrum_np = spectrum.cpu().numpy()
    frame0_positions_np = frame0_positions.cpu().numpy()
    scaling_factor_np = scaling_factor.cpu().numpy()
    
    num_residues = spectrum_np.shape[0]
    
    # Use correct spectrum_to_coords function
    spectrum_tensor = torch.from_numpy(spectrum_np).to(spectrum.device)
    coords_complex = spectrum_to_coords(spectrum_tensor)
    
    # Initialize trajectory array
    trajectory = np.zeros((num_residues, num_frames, 3))
    
    # Process x,y,z three dimensions for each residue separately
    for i in range(num_residues):
        trajectory[i, :, 0] = inverse_normalize_and_ifft(coords_complex[i, 0, :], num_frames, scaling_factor_np)
        trajectory[i, :, 1] = inverse_normalize_and_ifft(coords_complex[i, 1, :], num_frames, scaling_factor_np)
        trajectory[i, :, 2] = inverse_normalize_and_ifft(coords_complex[i, 2, :], num_frames, scaling_factor_np)
    
    # Set frame 0 coordinates
    trajectory[:, 0, :] = frame0_positions_np
    
    # Add displacements to initial coordinates
    for i in range(1, num_frames):
        trajectory[:, i, :] = trajectory[:, 0, :] + trajectory[:, i, :]
    
    # Convert to tensor and adjust dimension order to [num_residues, 3, num_frames]
    coordinates = torch.from_numpy(trajectory).to(spectrum.device)
    coordinates = coordinates.permute(0, 2, 1).contiguous()  # [num_residues, 3, num_frames]
    
    return coordinates
