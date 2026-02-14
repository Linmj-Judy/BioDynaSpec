
"""
Protein Trajectory FFT/IFT Pipeline (PyTorch)

What this provides:
- Robust preprocessing for MD-like protein trajectories: optional PBC unwrapping (orthorhombic),
  missing-data imputation, COM removal (mass-weighted), rigid-body rotational alignment (Kabsch),
  and detrending (constant or linear).
- Full complex spectrum extraction via FFT along time, with frequency axes and optional windowing
  (for spectral estimation) and zero-padding.
- Exact reconstruction by IFFT and inversion of all preprocessing steps (no idealized assumptions).
- Power/Amplitude/PSD utilities with correct normalization and window compensation.
- Works on CPU/GPU, float32/float64, and handles a range of edge cases.

Input convention:
- Coordinates tensor: positions of shape (N_atoms, 3, T). This follows your stated convention.
- Time sampling is assumed uniform with interval dt (float, seconds).
- Optional data:
  - masses: (N_atoms,) in atomic mass units or any consistent mass unit (only relative masses matter for weighting)
  - box_lengths: (3,) or (T, 3) for orthorhombic PBC unwrapping; unit must match positions' coordinate unit.
  - missing_mask: boolean mask of shape (N_atoms, 3, T), True=valid, False=missing; will be linearly interpolated.
  - norm for FFT: {'backward','ortho','forward'} identical to torch.fft API.

Notes on windowing:
- To get an exact time-domain reconstruction, you must NOT window the signal you invert. This pipeline
  computes the FFT from the unwindowed series for reconstruction and (optionally) also from a windowed
  series for better spectral estimation. Both can be returned.

Caveats:
- Rotational alignment uses mass-weighted Kabsch to remove rigid-body motion against a reference frame
  (default: first frame). This is optional and costs SVD per frame. For large T, this is compute-intensive.
- PBC unwrapping assumes orthorhombic boxes, and applies minimum-image continuity across frames. If the
  displacement between two frames exceeds half the box length, the step is "unwrapped" accordingly.
- Missing data interpolation is per (atom, coordinate) series; for many series it may be slow in Python.
  Vectorization can be added if needed.

"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Tuple, Literal, Any, Union
import torch.nn.functional as F
import math
import torch
import numpy as np


# -------------------------
# Utility and math helpers
# -------------------------

# ------------- Shape & Validation -------------

def _ensure_positions(x: torch.Tensor) -> None:
    if x.ndim != 3:
        raise ValueError(f"positions must be 3D tensor. Got shape={tuple(x.shape)}")
    if x.shape[1] == 3 and x.shape[0] > 3:
        return  # (N,3,T)
    if x.shape[-1] == 3 and x.shape[0] > 3:
        return  # (T,N,3)
    raise ValueError(f"Unsupported positions layout. Expected (N,3,T) or (T,N,3). Got {tuple(x.shape)}")


def _normalize_to_N3T(x: torch.Tensor, layout_hint: Optional[str] = None) -> Tuple[torch.Tensor, str]:
    """
    Normalize positions to (N,3,T). Returns (x_n3t, original_layout).
    layout_hint in {"N3T","TN3"} can override auto-detect.
    """
    _ensure_positions(x)
    if layout_hint is None:
        # auto-detect
        if x.shape[1] == 3:
            return x.contiguous(), "N3T"
        elif x.shape[-1] == 3:
            # (T,N,3) -> (N,3,T)
            return x.permute(1, 2, 0).contiguous(), "TN3"
    else:
        if layout_hint.upper() == "N3T":
            return x.contiguous(), "N3T"
        elif layout_hint.upper() == "TN3":
            return x.permute(1, 2, 0).contiguous(), "TN3"
    raise ValueError("layout detection failed")


def _restore_layout(x_n3t: torch.Tensor, original_layout: str) -> torch.Tensor:
    if original_layout.upper() == "N3T":
        return x_n3t
    elif original_layout.upper() == "TN3":
        return x_n3t.permute(2, 0, 1).contiguous()
    else:
        raise ValueError(f"Unknown original_layout={original_layout}")


def _validate_masses(masses: Optional[torch.Tensor], n_atoms: int, device, dtype):
    if masses is None:
        return torch.ones(n_atoms, device=device, dtype=dtype)
    if masses.ndim != 1 or masses.shape[0] != n_atoms:
        print(f"[DEBUG] masses shape: {masses.shape}")
        raise ValueError(f"masses must be 1D tensor of length N_atoms={n_atoms}")
    if not torch.all(torch.isfinite(masses)):
        raise ValueError("masses contain non-finite values")
    if (masses <= 0).any():
        raise ValueError("masses must be strictly positive")
    return masses.to(device=device, dtype=dtype)


def _validate_box_lengths(box_lengths: Optional[torch.Tensor], T: int, device, dtype):
    if box_lengths is None:
        return None
    if box_lengths.ndim == 1 and box_lengths.shape[0] == 3:
        return box_lengths.to(device=device, dtype=dtype)
    if box_lengths.ndim == 2 and box_lengths.shape == (T, 3):
        return box_lengths.to(device=device, dtype=dtype)
    raise ValueError("box_lengths must be (3,) or (T,3) for orthorhombic boxes")


def _validate_missing_mask(mask: Optional[torch.Tensor], shape: Tuple[int, int, int], device):
    if mask is None:
        return None
    if mask.shape != shape or mask.dtype != torch.bool:
        raise ValueError(f"missing_mask must be boolean tensor with shape {shape}")
    return mask.to(device=device)


def _poly_design_matrix(t: torch.Tensor, order: int) -> torch.Tensor:
    # t: (T,), returns A: (T, order+1) with columns t^0, t^1, ...
    cols = [torch.ones_like(t)]
    for k in range(1, order + 1):
        cols.append(t ** k)
    return torch.stack(cols, dim=-1)


def _interpolate_missing_series_1d(y: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """
    Linear interpolation along time for a single 1D series.
    y: (T,) float
    valid: (T,) bool (True means y[t] is valid)
    Returns a filled (T,) tensor.
    """
    T = y.shape[0]
    if valid.all():
        return y
    if not valid.any():
        raise ValueError("A time-series has no valid samples; cannot interpolate.")

    # Clone to avoid in-place on input
    out = y.clone()

    # # Indices
    # t_idx = torch.arange(T, device=y.device)

    # Forward fill for leading NaNs
    first_valid = torch.nonzero(valid, as_tuple=False)[0].item()
    out[:first_valid] = y[first_valid]

    # Backward fill for trailing NaNs
    last_valid = torch.nonzero(valid, as_tuple=False)[-1].item()
    out[last_valid + 1:] = y[last_valid]

    # Linear interpolation for gaps
    # # Find runs of invalid segments between valids
    # invalid = ~valid
    # boundaries are indices where valid changes
    changes = torch.where(valid[1:] ^ valid[:-1])[0] + 1
    # Build segments between change points
    segments = torch.cat([torch.tensor([0], device=y.device), changes, torch.tensor([T], device=y.device)])
    # Iterate segments
    for i in range(len(segments) - 1):
        start, end = segments[i].item(), segments[i + 1].item()
        # segment [start, end)
        seg_valid = valid[start:end]
        if seg_valid.all():
            continue
        # There must be at least one valid at each side because we did forward/backward fill
        # Find the leftmost valid index within [start, end)
        left = start + torch.where(seg_valid)[0][0].item()
        # For the invalid block before 'left', we interpolate between prev_valid (left-1) and left
        j = start
        while j < left:
            # Interpolate between (left-1) and left
            a, b = left - 1, left
            wa = (left - j) / (left - (left - 1))
            wb = 1.0 - wa
            out[j] = wa * y[a] + wb * y[b]
            j += 1
        # Also check for possible invalid tail inside segment (should not happen due to fill)
    return out


def _interpolate_missing_3d(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Interpolate missing values along time for x: (N,3,T) using mask: True=valid.
    """
    N, C, T = x.shape
    out = x.clone()
    # reshape to (N*C, T)
    flat = out.reshape(N * C, T)
    valid = mask.reshape(N * C, T)
    for i in range(N * C):
        if (~valid[i]).any():
            flat[i] = _interpolate_missing_series_1d(flat[i], valid[i])
    return flat.reshape(N, C, T)


def _compute_com(x: torch.Tensor, masses: torch.Tensor) -> torch.Tensor:
    """
    Center of mass per frame.
    x: (N,3,T), masses: (N,)
    returns com: (3,T)
    """
    # mass-weighted sum over atoms
    mw = masses[:, None, None] * x  # (N,3,T)
    com = mw.sum(dim=0) / masses.sum()  # (3,T)
    return com


def _remove_com(x: torch.Tensor, masses: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Remove COM per frame and return centered coordinates and COM(t).
    """
    com = _compute_com(x, masses)  # (3,T)
    xc = x - com.unsqueeze(0)  # broadcast (N,3,T)
    return xc, com


def _kabsch_rotations_to_ref(xc: torch.Tensor, ref_xc0: torch.Tensor, masses: torch.Tensor) -> torch.Tensor:
    """
    Compute mass-weighted Kabsch rotations aligning each frame of xc to ref_xc0.

    xc: (N,3,T) centered positions (COM=0)
    ref_xc0: (N,3) centered reference (e.g., first frame)
    masses: (N,)

    Returns R: (T,3,3) such that y_t = R_t @ xc_t best aligns to ref_xc0.

    Batch SVD over frames.
    """
    N, C, T = xc.shape
    assert C == 3
    w_sqrt = masses.sqrt()[:, None]  # (N,1)

    # Reshape to (T,N,3)
    X = xc.permute(2, 0, 1).contiguous()  # (T,N,3)
    Y = ref_xc0[None, :, :].expand(T, N, 3)  # (T,N,3)

    # Mass-weighted
    Xw = X * w_sqrt  # (T,N,3)
    Yw = Y * w_sqrt  # (T,N,3)

    # Covariance per frame: S_t = Xw_t^T @ Yw_t  => (3,3)
    # Use batch: (T,3,3)
    S = torch.matmul(Yw.transpose(1, 2), Xw)  # Y^T X

    # SVD per frame: S = U @ diag(s) @ Vh
    U, s, Vh = torch.linalg.svd(S)
    V = Vh.transpose(-2, -1)

    # Handle right-handedness (det correction)
    # det(V @ U^T) should be +1
    det = torch.det(torch.matmul(V, U.transpose(-2, -1)))
    # Build correction matrix
    I = torch.eye(3, device=xc.device, dtype=xc.dtype).expand(T, 3, 3).clone()
    I[:, -1, -1] = torch.where(det < 0, torch.tensor(-1.0, device=xc.device, dtype=xc.dtype), torch.tensor(1.0, device=xc.device, dtype=xc.dtype))
    R = torch.matmul(V, torch.matmul(I, U.transpose(-2, -1)))  # (T,3,3)
    return R


def _apply_rotations(xc: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """
    Apply per-frame rotations to centered coords.
    xc: (N,3,T), R: (T,3,3)
    returns y: (N,3,T) where y_t = R_t @ xc_t
    """
    # Bring frames to front: (T,N,3)
    X = xc.permute(2, 0, 1)  # (T,N,3)
    Y = torch.matmul(R, X.transpose(1, 2)).transpose(1, 2)  # (T,N,3)
    return Y.permute(1, 2, 0).contiguous()  # (N,3,T)


def _unwrap_pbc_orthorhombic(x: torch.Tensor, box_lengths: torch.Tensor) -> torch.Tensor:
    """
    Unwrap PBC across time with orthorhombic boxes using minimum image convention.
    x: (N,3,T) within box [0, L) or similar
    box_lengths: (3,) or (T,3)

    Strategy:
    - Compute step displacements d = x[...,1:] - x[...,:-1]
    - For each step k, adjust d_k by subtracting L_k * round(d_k / L_k) to enforce |d_k| <= L_k/2
    - Cumulative sum to reconstruct unwrapped trajectory.
    """
    N, C, T = x.shape
    if T < 2:
        return x.clone()
    if box_lengths.ndim == 1:
        L_steps = box_lengths[None, :, None].expand(N, C, T - 1)  # (N,3,T-1) broadcast
    else:
        # (T,3) -> steps associated with each delta are from frame k to k+1; use L at k (or average)
        L_steps = box_lengths[:-1, :].transpose(0, 1).unsqueeze(0).expand(N, C, T - 1)  # (N,3,T-1)
    if (L_steps <= 0).any():
        raise ValueError("box_lengths must be positive")

    d = x[..., 1:] - x[..., :-1]  # (N,3,T-1)
    # Avoid division by zero handled above
    shift = L_steps * torch.round(d / L_steps)
    d_corr = d - shift  # ensure steps minimal
    y = torch.empty_like(x)
    y[..., 0] = x[..., 0]
    y[..., 1:] = y[..., :-1] + d_corr
    return y


def _fft_full(x: torch.Tensor, dt: float, norm: Literal["backward", "ortho", "forward"], n_fft: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Full complex FFT along last dimension (time). Returns spectrum, freq (Hz), ang_freq (rad/s).
    x: real-valued (N,3,T)
    """
    T = x.shape[-1]
    if n_fft is None:
        n_fft = T
    # torch.fft.fft accepts n=n_fft to zero-pad/truncate.
    X = torch.fft.fft(x, n=n_fft, dim=-1, norm=norm)  # complex64/128
    freqs = torch.fft.fftfreq(n_fft, d=dt).to(device=x.device, dtype=x.real.dtype)  # (n_fft,)
    omega = 2.0 * math.pi * freqs
    return X, freqs, omega


def _ifft_full(X: torch.Tensor, norm: Literal["backward", "ortho", "forward"], T_target: Optional[int] = None) -> torch.Tensor:
    """
    IFFT along last dim. If T_target is not None and differs from X.size(-1), truncate to T_target after IFFT
    or, if X was zero-padded, IFFT gives interpolated series of length n_fft (cannot "truncate" frequency-domain and expect exact original unless n_fft==T).
    """
    x_rec = torch.fft.ifft(X, dim=-1, norm=norm).real  # (N,3,n_fft)
    if T_target is not None and T_target != x_rec.shape[-1]:
        # If user zero-padded in FFT, we cannot simply truncate in time domain for exact reconstruction from original X.
        # We expose truncation here only if the user explicitly set n_fft != T (interpolated series).
        x_rec = x_rec[..., :T_target]
    return x_rec

def _stft_full(coords_t: torch.Tensor, dt: float, window_size: int, window_vector: torch.Tensor, hop_ratio: float = 0.5):
    """
    Full complex STFT along last dimension (time). Returns spectrum, freq (Hz), ang_freq (rad/s).
    x: real-valued (N,3,T)
    """
    hop_size = int(window_size * hop_ratio)
    coords_t_xyz = [coords_t[:, i, :] for i in range(3)]  # 得到3个 (N, T), 分别对应 x, y, z
    freqs = torch.fft.rfftfreq(window_size, d=dt).to(device=coords_t.device, dtype=coords_t.real.dtype)  # (freq_bins,) 只有非负频半谱的频率
    omega = 2.0 * math.pi * freqs
    
    spectrum_xyz = []
    for NT in coords_t_xyz:
        windowed_spec = torch.stft(
            NT, n_fft=window_size, hop_length=hop_size, win_length=window_size,
            window=window_vector, center=False, return_complex=True, normalized=True, 
        ) # (N, freq_bins, n_frames) # 已经是非负频半谱 # torch.complex
        # Since center=True, x_hat length needs adjustment. Centered STFT with ISTFT reconstructs signal with padding,
        # so to get length T, set 'length=x.shape[-1]'
        spectrum_xyz.append(windowed_spec)
    spectrum_xyz = torch.stack(spectrum_xyz, dim=-3) # (N, 3, freq_bins, n_frames)
    return spectrum_xyz, freqs, omega

def _istft_full(spectrum: torch.Tensor, total_frames: int, window_size: int, window_vector: torch.Tensor, hop_ratio: float = 0.5):
    assert spectrum.shape[-3] == 3, "spectrum must be (N, 3, freq_bins, n_frames)"
    hop_size = int(window_size * hop_ratio)

    spectrum_xyz = spectrum.split(1, dim=-3)
    coords_t_xyz_hat = []
    for spec in spectrum_xyz:
        spec = spec.squeeze(dim=-3) # (N, freq_bins, n_frames)
        NT_hat = torch.istft(
            spec, n_fft=window_size, hop_length=hop_size, win_length=window_size,
            window=window_vector, center=False, length=total_frames, normalized=True, 
        ) # (N, T)
        coords_t_xyz_hat.append(NT_hat)
    coords_t_hat = torch.stack(coords_t_xyz_hat, dim=-1) # (N, 3, T)
    return coords_t_hat

def irfft_full(spectrum: torch.Tensor, window_size: int, norm: Literal["backward", "ortho", "forward"] = "ortho"):
    return torch.fft.irfft(spectrum, n=window_size, dim=-1, norm=norm) # norm: "ortho" 对应torch.stft的normalized=True

def compute_one_sided_psd_from_half_spectrum(
    spectrum_xyz: torch.Tensor,
    window: torch.Tensor,
    dt: float,
    n_fft: int,
    # average_frames: bool = True,
):
    """
    spectrum_xyz: (N, 3, freq_bins, n_frames), 非负频半谱 (onesided=True 的 STFT 输出)
    window:       (win_length,) 与 STFT 使用的窗口一致
    dt:           采样时间间隔，比如 1/fs
    n_fft:        STFT 时使用的 n_fft（通常 == win_length）
    average_frames: 是否沿 n_frames 维做 Welch 平均

    返回:
        freqs:         (freq_bins,) 频率轴 (0 ~ Nyquist), 单位 Hz
        psd_one_sided: (N, 3, freq_bins, n_frames)
    """
    device = spectrum_xyz.device
    dtype = spectrum_xyz.dtype

    # 1) 窗功率归一化 U = (1/N) sum w^2
    T = window.numel()
    assert T == n_fft, "window 长度要和 n_fft 一致"
    U = (window.to(dtype=dtype, device=device) ** 2).mean().item()  # 等价于 sum(w^2)/N

    # 2) 先算 two-sided PSD 的值（只是只看 0~Nyquist）
    #    power: (N, 3, freq_bins, n_frames)
    power = spectrum_xyz.abs() ** 2

    # Sxx_two_side(k) = (dt / (N * U)) * |X(k)|^2
    psd_two_sided_half = (dt / (n_fft * U)) * power

    # 3) 从「半边 two-sided」转成真正的 one-sided PSD
    psd_one_sided = psd_two_sided_half.clone()
    freq_bins = spectrum_xyz.shape[-2]

    # 检查 freq_bins 是否和 n_fft 对得上
    assert freq_bins == n_fft // 2 + 1, "freq_bins 应该等于 n_fft//2 + 1 (onesided=True)"

    if n_fft % 2 == 0:
        # N 偶数: bin 0 是 DC, bin -1 是 Nyquist，中间 1..-2 乘 2
        psd_one_sided[..., 1:-1, :] *= 2.0
    else:
        # N 奇数: 没有单独 Nyquist，1..end 全部乘 2
        psd_one_sided[..., 1:, :] *= 2.0

    # # 4) 如需 Welch 平均：在时间帧维度上做平均
    # if average_frames:
    #     psd_one_sided = psd_one_sided.mean(dim=-1)  # (N, 3, freq_bins)

    return psd_one_sided  # (N, 3, freq_bins, n_frames)



def extract_positive_frequency_half_spectrum(
    X_full: torch.Tensor,
    freqs_full: Optional[torch.Tensor] = None
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    从完整频谱中提取正频半谱（包括DC分量和可能的Nyquist频率）。
    
    对于实值信号的FFT，频谱具有厄米对称性：X[k] = X*[-k]。
    因此只需要保存正频率部分（包括DC和Nyquist）即可完整重建。
    
    Args:
        X_full: 完整复数频谱，shape (..., n_fft)，最后一维是频率维度
        freqs_full: 完整的频率数组，shape (n_fft,)，可选。如果提供，会同时返回对应的正频率数组
    
    Returns:
        如果 freqs_full 为 None：只返回 X_positive (torch.Tensor)
        如果 freqs_full 不为 None：返回 (X_positive, freqs_positive) 元组
        - X_positive: 正频半谱，shape (..., n_positive)，其中 n_positive = n_fft // 2 + 1
        - freqs_positive: 正频率数组，shape (n_positive,)
    
    注意：
        - 如果 n_fft 是偶数，正频半谱包括 [0, 1, ..., n_fft/2]（共 n_fft//2 + 1 个点）
        - 如果 n_fft 是奇数，正频半谱包括 [0, 1, ..., (n_fft-1)/2]（共 (n_fft+1)//2 个点）
    """
    n_fft = X_full.shape[-1]
    n_positive = n_fft // 2 + 1  # 包括DC和Nyquist（如果存在）
    
    # 提取正频率部分：索引 [0, 1, ..., n_positive-1]
    X_positive = X_full[..., :n_positive]
    
    if freqs_full is not None:
        freqs_positive = freqs_full[:n_positive]
        return X_positive, freqs_positive
    else:
        return X_positive


def reconstruct_full_spectrum_from_positive_half(
    X_positive: torch.Tensor,
    n_fft: Optional[int] = None
) -> torch.Tensor:
    """
    从正频半谱通过厄米共轭补齐负频，重建完整的频谱。
    
    对于实值信号的FFT，频谱具有厄米对称性：
    - X[0] 是DC分量（实数）
    - X[k] = X*[-k] for k > 0（厄米对称）
    - 如果 n_fft 是偶数，X[n_fft/2] 是Nyquist频率（实数）
    
    Args:
        X_positive: 正频半谱，shape (..., n_positive)，其中 n_positive = n_fft // 2 + 1（偶数时）或 n_positive = (n_fft + 1) // 2（奇数时）
        n_fft: 完整频谱的长度。如果为None，会从 X_positive 的形状推断：
                - 偶数情况：n_fft = 2 * (n_positive - 1)
                - 奇数情况：n_fft = 2 * n_positive - 1
    
    Returns:
        X_full: 完整频谱，shape (..., n_fft)
    
    注意：
        - 如果 n_fft 是偶数，X_positive 应该包含 [0, 1, ..., n_fft/2]（共 n_fft//2 + 1 个点）
        - 如果 n_fft 是奇数，X_positive 应该包含 [0, 1, ..., (n_fft-1)/2]（共 (n_fft+1)//2 个点）
        - 偶数: n_fft = 2 * (n_positive - 1)；奇数: n_fft = 2 * n_positive - 1
    """
    n_positive = X_positive.shape[-1]

    # 推断 n_fft：支持偶数和奇数情况
    if n_fft is None:
        # 1. 尝试偶数方案
        n_fft_even = 2 * (n_positive - 1)
        if n_fft_even // 2 + 1 == n_positive:
            n_fft = n_fft_even
        else:
            # 2. 否则尝试奇数方案
            n_fft_odd = 2 * n_positive - 1
            if (n_fft_odd + 1) // 2 == n_positive:
                n_fft = n_fft_odd
            else:
                raise ValueError(
                    f"Cannot infer n_fft from n_positive ({n_positive}). "
                    "Neither even (n_fft = 2*(n_positive-1)) nor odd (n_fft = 2*n_positive-1) rules match."
                )
    # 再次验证 n_positive/n_fft 匹配
    expected_n_positive_even = n_fft // 2 + 1
    expected_n_positive_odd = (n_fft + 1) // 2
    if (n_fft % 2 == 0 and n_positive != expected_n_positive_even) or \
       (n_fft % 2 == 1 and n_positive != expected_n_positive_odd):
        raise ValueError(
            f"n_positive ({n_positive}) 与 n_fft ({n_fft}) 不匹配。"
            f"对于 n_fft={n_fft}，期望 n_positive="
            f"{expected_n_positive_even if n_fft % 2 == 0 else expected_n_positive_odd}"
        )

    # 创建完整频谱
    shape = list(X_positive.shape)
    shape[-1] = n_fft
    X_full = torch.zeros(shape, dtype=X_positive.dtype, device=X_positive.device)

    # 复制正频率部分（包括DC和Nyquist）
    X_full[..., :n_positive] = X_positive

    # 通过厄米共轭补齐负频率部分
    if n_positive > 1:
        if n_fft % 2 == 0:
            # 偶数：Nyquist在最后一个正频
            # 正向索引: 1 ~ n_fft//2-1
            pos_part = X_positive[..., 1 : n_fft // 2]
            # 负频区间: n_fft//2+1 ~ n_fft-1
            neg_start = n_fft // 2 + 1
            neg_end = n_fft
        else:
            # 奇数：最后一个正频就是 (n_fft-1)//2
            # 索引: 1 ~ (n_fft-1)//2
            pos_part = X_positive[..., 1 : n_positive]
            # 负频区间: (n_fft+1)//2 ~ n_fft-1
            neg_start = (n_fft + 1) // 2
            neg_end = n_fft
        neg_part = torch.flip(pos_part, dims=[-1]).conj()
        X_full[..., neg_start:neg_end] = neg_part

    return X_full

def reconstruct_full_spectrum_from_positive_half_real_imag(
    real_positive: torch.Tensor,
    imag_positive: torch.Tensor,
    n_fft: Optional[int] = None
) -> torch.Tensor:
    """
    从正频半谱通过厄米共轭补齐负频，重建完整的频谱。
    
    对于实值信号的FFT，频谱具有厄米对称性：
    - X[0] 是DC分量（实数）
    - X[k] = X*[-k] for k > 0（厄米对称）
    - 如果 n_fft 是偶数，X[n_fft/2] 是Nyquist频率（实数）
    
    Args:
        X_positive: 正频半谱，shape (..., n_positive)，其中 n_positive = n_fft // 2 + 1（偶数时）或 n_positive = (n_fft + 1) // 2（奇数时）
        n_fft: 完整频谱的长度。如果为None，会从 X_positive 的形状推断：
                - 偶数情况：n_fft = 2 * (n_positive - 1)
                - 奇数情况：n_fft = 2 * n_positive - 1
    
    Returns:
        X_full: 完整频谱，shape (..., n_fft)
    
    注意：
        - 如果 n_fft 是偶数，X_positive 应该包含 [0, 1, ..., n_fft/2]（共 n_fft//2 + 1 个点）
        - 如果 n_fft 是奇数，X_positive 应该包含 [0, 1, ..., (n_fft-1)/2]（共 (n_fft+1)//2 个点）
        - 偶数: n_fft = 2 * (n_positive - 1)；奇数: n_fft = 2 * n_positive - 1
    """
    assert real_positive.shape[-1] == imag_positive.shape[-1], "real and imag must have the same length in the last dimension"
    *leading_dims, n_positive = real_positive.shape
    # 构造正频半谱复数
    pos_spec = torch.complex(real_positive, imag_positive)  # (..., F_pos)

    # 推断 n_fft：支持偶数和奇数情况
    if n_fft is None:
        # 1. 尝试偶数方案
        n_fft_even = 2 * (n_positive - 1)
        if n_fft_even // 2 + 1 == n_positive:
            n_fft = n_fft_even
        else:
            # 2. 否则尝试奇数方案
            n_fft_odd = 2 * n_positive - 1
            if (n_fft_odd + 1) // 2 == n_positive:
                n_fft = n_fft_odd
            else:
                raise ValueError(
                    f"Cannot infer n_fft from n_positive ({n_positive}). "
                    "Neither even (n_fft = 2*(n_positive-1)) nor odd (n_fft = 2*n_positive-1) rules match."
                )
    # 再次验证 n_positive/n_fft 匹配
    expected_n_positive_even = n_fft // 2 + 1
    expected_n_positive_odd = (n_fft + 1) // 2
    if (n_fft % 2 == 0 and n_positive != expected_n_positive_even) or \
       (n_fft % 2 == 1 and n_positive != expected_n_positive_odd):
        raise ValueError(
            f"n_positive ({n_positive}) 与 n_fft ({n_fft}) 不匹配。"
            f"对于 n_fft={n_fft}，期望 n_positive="
            f"{expected_n_positive_even if n_fft % 2 == 0 else expected_n_positive_odd}"
        )

    # 创建完整频谱
    shape = list(pos_spec.shape)
    shape[-1] = n_fft
    full_spec = pos_spec.new_zeros(*leading_dims, n_fft)

    # 复制正频率部分（包括DC和Nyquist）
    full_spec[..., :n_positive] = pos_spec

    # 通过厄米共轭补齐负频率部分
    if n_positive > 1:
        if n_fft % 2 == 0:
            # 偶数：Nyquist在最后一个正频
            # 正向索引: 1 ~ n_fft//2-1
            pos_part = pos_spec[..., 1 : n_fft // 2]
            # 负频区间: n_fft//2+1 ~ n_fft-1
            neg_start = n_fft // 2 + 1
            neg_end = n_fft
        else:
            # 奇数：最后一个正频就是 (n_fft-1)//2
            # 索引: 1 ~ (n_fft-1)//2
            pos_part = pos_spec[..., 1 : n_positive]
            # 负频区间: (n_fft+1)//2 ~ n_fft-1
            neg_start = (n_fft + 1) // 2
            neg_end = n_fft
        neg_part = torch.flip(pos_part, dims=[-1]).conj()
        full_spec[..., neg_start:neg_end] = neg_part

    return full_spec  # (..., n_fft) complex


def _window_vector(T: int, win_type: Optional[str], device, dtype) -> Optional[torch.Tensor]:
    if win_type is None:
        return None
    wt = win_type.lower()
    if wt == "hann":
        w = torch.hann_window(T, periodic=True, dtype=dtype, device=device)
    elif wt == "hamming":
        w = torch.hamming_window(T, periodic=True, dtype=dtype, device=device)
    elif wt == "blackman":
        w = torch.blackman_window(T, periodic=True, dtype=dtype, device=device)
    else:
        raise ValueError(f"Unsupported window type: {win_type}")
    return w


def _window_power_correction(w: Optional[torch.Tensor]) -> float:
    """
    U = (1/T) * sum(w^2), used for PSD normalization correction.
    """
    if w is None:
        return 1.0
    T = w.numel()
    return float((w.pow(2).sum() / T).item())

def parse_time_str_to_seconds(time_str: str) -> float:
    """
    将"10ps"/"200ns"/"100ms"/"3 us"/"2 s"/"1.5 ns"格式的字符串解析为秒（float）。
    支持单位: s, ms, us, ns, ps。大小写不敏感，数字和单位间可有空格。
    """
    import re

    time_str = time_str.strip().lower()
    match = re.match(r"^\s*([0-9]*\.?[0-9]+)\s*([a-z]+)\s*$", time_str)
    if not match:
        raise ValueError(f"Invalid time string format: {time_str}")
    value, unit = match.groups()
    value = float(value)
    unit_factors = {
        "s": 1.0,
        "ms": 1e-3,
        "us": 1e-6,
        "ns": 1e-9,
        "ps": 1e-12,
    }
    if unit not in unit_factors:
        raise ValueError(f"Unsupported time unit '{unit}' in: {time_str}")
    return value * unit_factors[unit]

def compute_velocity_spectrum_from_displacement(
    X: torch.Tensor,  # 位移谱 (N, 3, n_fft) 或 (..., n_fft)
    omega: torch.Tensor,  # 角频率 (n_fft,) rad/s
) -> torch.Tensor:
    """
    从位移谱直接计算速度谱（频域方法）。
    
    理论依据：
    - 速度 v(t) = dx/dt，在频域中：V(ω) = iω * X(ω)
    
    Args:
        X: 位移谱，复数张量，shape (..., n_fft)
        omega: 角频率数组，shape (n_fft,)，单位 rad/s
    
    Returns:
        V: 速度谱，复数张量，shape 与 X 相同
    """
    n_fft = X.shape[-1]
    if omega.shape[0] != n_fft:
        raise ValueError(f"omega shape {omega.shape} 与 X 的最后一个维度 {n_fft} 不匹配")
    
    # 将 omega 扩展到与 X 相同的形状
    omega_expanded = omega.view(*([1] * (X.ndim - 1)), n_fft)
    
    # 计算速度谱：V(ω) = iω * X(ω)
    # 使用 1j 表示虚数单位
    V = 1j * omega_expanded * X
    
    return V


def compute_acceleration_spectrum_from_displacement(
    X: torch.Tensor,  # 位移谱 (N, 3, n_fft) 或 (..., n_fft)
    omega: torch.Tensor,  # 角频率 (n_fft,) rad/s
    dt: float  # 时间步长（秒），用于验证
) -> torch.Tensor:
    """
    从位移谱直接计算加速度谱（频域方法）。
    
    理论依据：
    - 位移 x(t) 的傅里叶变换为 X(ω)
    - 速度 v(t) = dx/dt，在频域中：V(ω) = iω * X(ω)
    - 加速度 a(t) = dv/dt = d²x/dt²，在频域中：A(ω) = (iω)² * X(ω) = -ω² * X(ω)
    
    Args:
        X: 位移谱，复数张量，shape (..., n_fft)
        omega: 角频率数组，shape (n_fft,)，单位 rad/s
        dt: 时间步长（秒），用于验证频率数组的正确性
    
    Returns:
        A: 加速度谱，复数张量，shape 与 X 相同
    """
    # 确保 omega 的形状与 X 的最后一个维度匹配
    n_fft = X.shape[-1]
    if omega.shape[0] != n_fft:
        raise ValueError(f"omega shape {omega.shape} 与 X 的最后一个维度 {n_fft} 不匹配")
    
    # 将 omega 扩展到与 X 相同的形状（除了最后一个维度）
    # omega: (n_fft,) -> (1, 1, ..., 1, n_fft)
    omega_expanded = omega.view(*([1] * (X.ndim - 1)), n_fft)
    
    # 计算加速度谱：A(ω) = -ω² * X(ω)
    # 注意：这里使用 -omega² 而不是 (iω)²，因为 (iω)² = i²ω² = -ω²
    A = -omega_expanded ** 2 * X
    
    return A


# ------------- Serializable State & Config -------------

@dataclass
class PreprocessStateStruct:
    # Stored to invert preprocessing
    used_unwrap: bool
    box_lengths: Optional[torch.Tensor]  # (3,) or (T,3)
    removed_com: bool
    com: Optional[torch.Tensor]          # (3,T) if removed_com
    aligned_rotation: bool
    rotations: Optional[torch.Tensor]    # (T,3,3) if aligned_rotation
    detrend_order: Optional[int]
    trend_coeffs: Optional[torch.Tensor] # ((order+1), N*3) coefficients for trend(y) = A @ coeffs per series
    trend_time: Optional[torch.Tensor]   # (T,) time vector used for detrending
    # additional metadata for inversion and reconstruction
    original_layout: str                 # "N3T" or "TN3"
    dtype_name: str                      # "float32" or "float64"
    
    def to_dict(self, store_on_cpu: bool = True) -> Dict[str, Any]:
        def cpu_or_same(t):
            if t is None:
                return None
            return t.detach().cpu() if store_on_cpu else t.detach()

        return {
            "used_unwrap": self.used_unwrap,
            "box_lengths": cpu_or_same(self.box_lengths),
            "removed_com": self.removed_com,
            "com": cpu_or_same(self.com),
            "aligned_rotation": self.aligned_rotation,
            "rotations": cpu_or_same(self.rotations),
            "detrend_order": self.detrend_order,
            "trend_coeffs": cpu_or_same(self.trend_coeffs),
            "trend_time": cpu_or_same(self.trend_time),
            "original_layout": self.original_layout,
            "dtype_name": self.dtype_name,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> PreprocessStateStruct:
        return PreprocessStateStruct(
            used_unwrap=bool(d["used_unwrap"]),
            box_lengths=d["box_lengths"],
            removed_com=bool(d["removed_com"]),
            com=d["com"],
            aligned_rotation=bool(d["aligned_rotation"]),
            rotations=d["rotations"],
            detrend_order=d["detrend_order"],
            trend_coeffs=d["trend_coeffs"],
            trend_time=d["trend_time"],
            original_layout=d.get("original_layout", "N3T"),
            dtype_name=d.get("dtype_name", "float64"),
        )


@dataclass
class TrajectoryFFTConfig:
    # FFT
    norm: Literal["backward", "ortho", "forward"] = "backward"
    use_double: bool = True
    # Preprocess options (default values; can be overridden per call)
    perform_unwrap: bool = False
    remove_com: bool = False
    align_rotation: bool = False
    detrend_order: Optional[int] = None  # None/0/1
    impute_missing: bool = True
    # Spectrum estimation
    window: Optional[str] = "hamming"       # None/'hamming'/'blackman' ('hann' is not recommended)
    window_size: Optional[int] = None
    hop_ratio: float = 0.5
    use_sliding_window: bool = True
    # Storage options
    store_dtype: torch.dtype = torch.float32  # stored tensors (freqs, real/imag etc.)
    store_complex_as_float2: bool = True      # True: store real/imag separately
    # Device
    device: Optional[str] = None  # "cpu" recommended inside Dataset
    dtype: torch.dtype = torch.float32

    def to_dict(self) -> Dict[str, Any]:
        return {
            "norm": self.norm,
            "use_double": self.use_double,
            "perform_unwrap": self.perform_unwrap,
            "remove_com": self.remove_com,
            "align_rotation": self.align_rotation,
            "detrend_order": self.detrend_order,
            "impute_missing": self.impute_missing,
            "window": self.window,
            "window_size": self.window_size,
            "hop_ratio": self.hop_ratio,
            "use_sliding_window": self.use_sliding_window,
            "store_dtype": str(self.store_dtype).replace("torch.", ""),
            "store_complex_as_float2": self.store_complex_as_float2,
            "device": self.device,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> TrajectoryFFTConfig:
        dtype = getattr(torch, d.get("store_dtype", "float32"))
        return TrajectoryFFTConfig(
            norm=d.get("norm", "ortho"),
            use_double=bool(d.get("use_double", True)),
            perform_unwrap=bool(d.get("perform_unwrap", False)),
            remove_com=bool(d.get("remove_com", False)),
            align_rotation=bool(d.get("align_rotation", False)),
            detrend_order=d.get("detrend_order", None),
            impute_missing=bool(d.get("impute_missing", True)),
            window=d.get("window", "hamming"),
            hop_ratio=d.get("hop_ratio", 0.5),
            window_size=d.get("window_size", None),
            use_sliding_window=bool(d.get("use_sliding_window", True)),
            store_dtype=dtype,
            store_complex_as_float2=bool(d.get("store_complex_as_float2", True)),
            device=d.get("device", None),
        )


# ------------- Stateless Processor -------------

class TrajectoryFFTProcessor:
    """
    Stateless multi-protein / multi-trajectory processor.
    - No per-trajectory state retained in this object
    - All per-trajectory invert info is returned as PreprocessStateStruct
    - Safe for Dataset multi-worker usage and serializable
    """

    VERSION = "1.0.0"

    def __init__(self, config: TrajectoryFFTConfig):
        self.config = config

    # ----- Preprocess -----
    def preprocess(
        self,
        positions: torch.Tensor,                 # (N,3,T)
        *,
        dt: float,                          # seconds (uniform sampling)
        masses: Optional[torch.Tensor] = None,   # (N,)
        box_lengths: Optional[torch.Tensor] = None, # (3,) or (T,3)
        missing_mask: Optional[torch.Tensor] = None, # shape like positions (True=valid)
        layout_hint: Optional[str] = None,
        perform_unwrap: Optional[bool] = None,
        impute_missing: Optional[bool] = None,
        remove_com: Optional[bool] = None,
        align_rotation: Optional[bool] = None,
        detrend_order: Optional[int] = None,
    ) -> Tuple[torch.Tensor, PreprocessStateStruct]:
        """
        Preprocess the trajectory and store state for exact inversion after IFFT.
        Returns:
          - x_proc: preprocessed positions (N,3,T) with dtype float32/64 per config
          - state: per-trajectory PreprocessStateStruct for exact inversion
        """
        if dt <= 0:
            raise ValueError("dt must be positive and uniform for this FFT")

        # Normalize to (N,3,T)
        x_in = positions
        x, orig_layout = _normalize_to_N3T(x_in, layout_hint=layout_hint)
        # print(f"x.shape: {x.shape}")
        N, C, T = x.shape
        if C != 3:
            raise ValueError("Second dimension of position must be 3 (x,y,z)")

        # Determine target device: use GPU if configured, otherwise use input device or CPU
        dtype = torch.float64 if self.config.use_double else torch.float32
        
        # If GPU is configured and available, use GPU for all operations
        if self.config.device is not None and self.config.device == "cuda" and torch.cuda.is_available():
            preprocess_device = torch.device("cuda")
        else:
            # Use input device if it's already on GPU, otherwise use CPU
            preprocess_device = x.device if x.device.type == "cuda" else torch.device("cpu")
        
        # Move to target device and dtype (only if needed)
        if x.device != preprocess_device or x.dtype != dtype:
            x = x.to(device=preprocess_device, dtype=dtype)
        elif x.dtype != dtype:
            x = x.to(dtype=dtype)

        masses_t = _validate_masses(masses, N, preprocess_device, dtype)
        boxL = _validate_box_lengths(box_lengths, T, preprocess_device, dtype)
        mask = _validate_missing_mask(missing_mask, x.shape, preprocess_device)

        # Resolve flags
        perform_unwrap = self.config.perform_unwrap if perform_unwrap is None else perform_unwrap
        impute_missing = self.config.impute_missing if impute_missing is None else impute_missing
        remove_com = self.config.remove_com if remove_com is None else remove_com
        align_rotation = self.config.align_rotation if align_rotation is None else align_rotation
        detrend_order = self.config.detrend_order if detrend_order is None else detrend_order

        # Missing data handling
        if mask is not None and impute_missing:
            if x.dtype not in (torch.float32, torch.float64):
                raise ValueError("positions dtype must be float32/float64 for interpolation")
            x = _interpolate_missing_3d(x, mask)

        # Optional PBC unwrap
        used_unwrap = False
        if perform_unwrap:
            if boxL is None:
                raise ValueError("perform_unwrap=True requires box_lengths")
            x = _unwrap_pbc_orthorhombic(x, boxL)
            used_unwrap = True

        # Optional COM removal
        removed_com = False
        com = None
        if remove_com:
            x, com = _remove_com(x, masses_t)
            removed_com = True

        # Optional rotational alignment (Kabsch)
        aligned_rotation = False
        rotations = None
        if align_rotation:
            if not removed_com:
                # Kabsch should operate on centered coords; enforce COM removal for correctness
                # We can center by subtracting per-frame unweighted centroid, but mass-weighted COM is preferred.
                # Enforce remove_com to avoid numerical drift.
                x, com = _remove_com(x, masses_t)
                removed_com = True
            ref = x[..., 0]  # (N,3), first frame as reference (centered)
            R = _kabsch_rotations_to_ref(x, ref, masses_t)  # (T,3,3)
            x = _apply_rotations(x, R)
            aligned_rotation = True
            rotations = R

        # Optional detrend along time per (atom, coord) series
        trend_coeffs = None
        trend_time = None
        if detrend_order is not None:
            if detrend_order not in (0, 1):
                raise ValueError("Only detrend_order in {0,1} supported")
            # Build time vector in seconds for numerical stability
            # Use the same dtype as the working trajectory to avoid dtype mismatch in lstsq
            # Use preprocess_device (CPU) to avoid unnecessary GPU memory allocation
            t = torch.arange(T, device=preprocess_device, dtype=dtype) * dt  # (T,)
            A = _poly_design_matrix(t, detrend_order)  # (T, p)
            # Solve min ||A @ c - y||_2 per series y
            # Reshape to (T, M)
            Y = x.reshape(N * C, T).transpose(0, 1)  # (T, M)
            # Use lstsq
            # Xhat: (p, M)
            coeffs = torch.linalg.lstsq(A, Y).solution  # (p, M)
            trend = (A @ coeffs)  # (T, M)
            X_detrended = (Y - trend).transpose(0, 1).reshape(N, C, T)
            x = X_detrended
            trend_coeffs = coeffs  # keep to add back later
            trend_time = t

        # Save state for inversion
        state = PreprocessStateStruct(
            used_unwrap=used_unwrap,
            box_lengths=boxL,
            removed_com=removed_com,
            com=com,
            aligned_rotation=aligned_rotation,
            rotations=rotations,
            detrend_order=detrend_order,
            trend_coeffs=trend_coeffs,
            trend_time=trend_time,
            original_layout=orig_layout,
            dtype_name="float64" if dtype == torch.float64 else "float32",
        )
        return x, state

    # ----- Spectrum -----
    def compute_spectrum_and_metrics(
        self,
        processed_positions: torch.Tensor,     # (N,3,T) ideally from preprocess()
        *,
        dt: float,                             # 单位：秒 (uniform sampling)
        window: Optional[str] = None,          # None, 'hann', 'hamming', 'blackman'
        window_size: int = None,
        use_sliding_window: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Compute full complex spectrum along time (last dimension).

        Returns dict with:
        - X: complex spectrum from unwindowed series, shape (N,3,n_fft)
        - freqs: (n_fft,) Hz
        - omega: (n_fft,) rad/s
        - X_windowed: (optional) spectrum from windowed series for spectral estimation
        - window: (optional) window used (T,)
        """
        x = processed_positions
        if x.ndim != 3 or x.shape[1] != 3:
            raise ValueError(f"processed_positions must be (N,3,T). Got {tuple(x.shape)}")

        T = x.shape[-1]
        if not use_sliding_window or window_size is None:
            window_size = T
        # Use the same device as input (should already be on target device from preprocess)
        # All computations stay on the same device
        # x_fft = x  # Already on correct device from preprocess
        if window == "hann":
            raise ValueError("Hann window is not supported for this task, because it must set center=True for STFT and ISTFT, which is not supported for sliding window prediction in this task.")
        window_vector = _window_vector(window_size, window, x.device, x.dtype)

        if use_sliding_window:
            # 滑动窗口STFT得到位移谱
            spectrum_xyz, freqs, omega = _stft_full(x, dt, window_size, window_vector, hop_ratio=self.config.hop_ratio)
            # spectrum_xyz: (N, 3, freq_bins, n_frames)
            # freqs: (freq_bins,) Hz
            # omega: (freq_bins,) rad/s
            psd_one_sided = compute_one_sided_psd_from_half_spectrum(spectrum_xyz, window_vector, dt, window_size) # (N,3,freq_bins,n_frames)
            amp = spectrum_xyz.abs()  # (N,3,freq_bins,n_frames)
            # Power spectrum |X|^2
            power = amp.pow(2)  # (N,3,freq_bins,n_frames)

        else:
            # FFT得到完整位移谱
            # Unwindowed spectrum for exact IFFT
            spectrum_xyz, freqs, omega = _fft_full(x, dt=dt, norm=self.config.norm, n_fft=T)
            # Amplitude spectrum (complex magnitude)
            amp = spectrum_xyz.abs()  # (N,3,n_fft)

            # Power spectrum |X|^2
            power = amp.pow(2)

            # PSD scaling
            # For discrete-time sequence x[n], sampling interval dt:
            # With torch's 'backward' normalization (default), FFT has no 1/N factor in forward,
            # and IFFT has 1/N. A commonly used PSD estimate (two-sided) is:
            #   Sxx = (dt / (N * U)) * |FFT(windowed x)|^2
            # where U = (1/N) sum w^2 is window power normalization. For unwindowed, U=1.
            Tlen = T  # number of points in the FFT (could be zero-padded)
            U = _window_power_correction(window_vector)
            psd_two_sided = (dt / (Tlen * U)) * power  # (N,3,n_fft)
            
            spectrum_xyz, freqs = extract_positive_frequency_half_spectrum(spectrum_xyz, freqs)
            amp = spectrum_xyz.abs()  # (N,3,n_fft//2+1)

            # Power spectrum |X|^2
            power = amp.pow(2)  # (N,3,n_fft//2+1)

            psd_one_sided = psd_two_sided[..., :(T//2 + 1)] # (N,3,n_fft//2+1)
        
        out = {
            "spectrum_xyz": spectrum_xyz, 
            "freqs": freqs, "omega": omega,
            "use_sliding_window": use_sliding_window,
            "amplitude": amp,
            "power": power,
            "psd_one_sided": psd_one_sided,
        }
        return out

    # ----- High-level: end-to-end per trajectory -----
    def process_and_package(
        self,
        positions: torch.Tensor,              # (N,3,T) or (T,N,3)
        *,
        dt: float,
        masses: Optional[torch.Tensor] = None,
        box_lengths: Optional[torch.Tensor] = None,
        missing_mask: Optional[torch.Tensor] = None,
        layout_hint: Optional[str] = None,
        # override defaults if needed:
        perform_unwrap: Optional[bool] = None,
        impute_missing: Optional[bool] = None,
        remove_com: Optional[bool] = None,
        align_rotation: Optional[bool] = None,
        detrend_order: Optional[int] = None,
        # # metrics
        # compute_metrics_on: Literal["windowed", "unwindowed"] = "windowed",
        # storage optimization
        save_minimal: bool = False,  # If True, only save essential data (positive half spectrum, first frame, state)
        # save_full_spectrum: bool = False,  # If True, save full spectrum (X, V, A real/imag)
        # save_windowed: bool = False,  # If True, save windowed spectrum
        save_psd: bool = False,  # If True, save PSD metrics
    ) -> Dict[str, Any]:
        """
        Full pipeline for a single trajectory. Returns a dict ready to be attached
        into label_full_dict["trajectory_spectrum"] or similar key.
        """
        x_proc, state = self.preprocess(
            positions,
            dt=dt,
            masses=masses,
            box_lengths=box_lengths,
            missing_mask=missing_mask,
            layout_hint=layout_hint,
            perform_unwrap=perform_unwrap,
            impute_missing=impute_missing,
            remove_com=remove_com,
            align_rotation=align_rotation,
            detrend_order=detrend_order,
        )
        first_frame_positions = x_proc[:, :, 0] # (N,3)
        out = self.compute_spectrum_and_metrics(
            x_proc,
            dt=dt,
            window=self.config.window,
            window_size=self.config.window_size,
            use_sliding_window=self.config.use_sliding_window,
        )
        T = x_proc.shape[-1]

        # Package for storage (use store_dtype and split complex if needed)
        sdtype = self.config.store_dtype
        
        # Start with minimal essential data
        pack: Dict[str, Any] = {
            "version": self.VERSION,
            "norm": self.config.norm,
            "coords_t": x_proc.to("cpu", sdtype),
            "first_frame_coords": first_frame_positions.to("cpu", sdtype),
            "dt": float(dt),
            "T": int(T),
            "freqs": out["freqs"].to("cpu", sdtype),
            "omega": out["omega"].to("cpu", sdtype),
            "preprocess_state": state.to_dict(store_on_cpu=True),
        }
        
        # Always save positive frequency half spectrum (essential for reconstruction)
        if self.config.store_complex_as_float2:
            pack["X_positive_real"] = out["spectrum_xyz"].real.to("cpu", sdtype)
            pack["X_positive_imag"] = out["spectrum_xyz"].imag.to("cpu", sdtype)
        else:
            pack["X_positive"] = out["spectrum_xyz"].to("cpu")

        if save_psd:
            pack["psd_one_sided"] = out["psd_one_sided"].to("cpu", sdtype)
            pack["amplitude"] = out["amplitude"].to("cpu", sdtype)
            pack["power"] = out["power"].to("cpu", sdtype)
        
        # # If save_minimal is True, ensure we only have essential data
        # if save_minimal:
        #     # Remove any optional data that might have been added
        #     for key in ["X_real", "X_imag"]:
        #         pack.pop(key, None)
        
        # Clean up GPU memory: delete references to GPU tensors after moving to CPU
        # Data has already been moved to CPU during packing above
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return pack

    # ----- Reconstruction -----
    def reconstruct_package_from_positive_half_real_imag(
        self,
        positive_real: torch.Tensor,
        positive_imag: torch.Tensor,
        dt: float,
        T: int,
        first_frame_coords: torch.Tensor,
        preprocess_state=None,
        *,
        use_sliding_window: bool = False,
        window_size: Optional[int] = None,
        hop_ratio: float = 0.5,
        save_minimal: bool = False,  # If True, only save essential data (positive half spectrum, first frame, state)
        save_psd: bool = False,  # If True, save PSD metrics
    ):
        """
        从正频半谱（实部/虚部）和 preprocess_state 组装一个用于重建的 package。

        支持两种频谱来源：
        1) 完整 FFT（非滑窗）：正频半谱形状 (N, 3, n_fft//2+1)
        2) 滑窗 STFT：正频半谱形状 (N, 3, freq_bins, n_frames)，其中 freq_bins = n_fft//2+1，n_fft = window_size

        Args:
            positive_real / positive_imag: 正频半谱实部/虚部
            dt: 采样间隔（秒）
            T: 原始时间长度
            n_fft: FFT 长度；如果为 None，默认取 T（非滑窗）或从 freq_bins 反推（滑窗）
            first_frame_coords: 首帧坐标 (N,3)
            preprocess_state: 预处理状态（PreprocessStateStruct 或其 dict）
            use_sliding_window: 是否来自滑动窗口 STFT
            window_size: 滑窗长度（若滑窗模式必须提供或可从正频长度反推）
            hop_ratio: 滑窗重叠部分占比（可选，默认为 0.5）
            save_minimal: 若为 True，只保留最小必要字段
        """
        norm = self.config.norm
        device = positive_real.device
        dtype = positive_real.dtype
        spectrum_xyz = positive_real + 1j * positive_imag

        if use_sliding_window:
            # 频率维为倒数第二维
            freq_bins = positive_real.shape[-2]
            # 从正频长度反推 window_size
            if window_size is None:
                window_size = 2 * (freq_bins - 1)
            hop_size = int(window_size * hop_ratio)

            freqs = torch.fft.rfftfreq(window_size, dt, device=device, dtype=dtype)
            omega = 2 * np.pi * freqs
            window_vector = _window_vector(window_size, self.config.window, device, dtype)

            coords_t_hat = _istft_full(
                spectrum_xyz,
                total_frames=T,
                window_size=window_size,
                window_vector=window_vector,
                hop_ratio=hop_ratio
            )

            pack = {
                "version": self.VERSION,
                "dt": dt,
                "T": T,
                "norm": norm,
                "coords_t": coords_t_hat,
                "first_frame_coords": first_frame_coords,
                # "window_size": window_size,
                # "hop_size": hop_size, # 我的task中是window_size // 2
                # "use_sliding_window": True,
                "freqs": freqs,
                "omega": omega,
                "preprocess_state": preprocess_state,
            }
            if self.config.store_complex_as_float2:
                pack["X_positive_real"] = positive_real
                pack["X_positive_imag"] = positive_imag
            else:
                pack["X_positive"] = positive_real + 1j * positive_imag

            if preprocess_state is not None and hasattr(preprocess_state, "to_dict"):
                pack["preprocess_state"] = preprocess_state.to_dict(store_on_cpu=True)

            if save_psd:
                pack["psd_one_sided"] = compute_one_sided_psd_from_half_spectrum(
                    spectrum_xyz, window_vector, dt, window_size
                ) # (N,3,freq_bins,n_frames)
                pack["amplitude"] = spectrum_xyz.abs()
                pack["power"] = pack["amplitude"].pow(2)  # (N,3,freq_bins,n_frames)

            # 滑窗频谱不在此处组装全频/速度/加速度谱，保持最小必要信息
            return pack if save_minimal else pack

        # 非滑窗：完整 FFT
        freqs = torch.fft.rfftfreq(T, dt, device=device, dtype=dtype)
        omega = 2 * np.pi * freqs
        window_vector = _window_vector(T, self.config.window, device, dtype)

        pack = {
            "version": self.VERSION,
            "dt": dt,
            "T": T,
            "norm": norm,
            "freqs": freqs,
            "omega": omega,
            "coords_t": _ifft_full(spectrum_xyz, norm=self.config.norm, T_target=T),
            "first_frame_coords": first_frame_coords,
            "preprocess_state": preprocess_state,
            # "use_sliding_window": False,
        }
        if self.config.store_complex_as_float2:
            pack["X_positive_real"] = positive_real
            pack["X_positive_imag"] = positive_imag
        else:
            pack["X_positive"] = positive_real + 1j * positive_imag

        if preprocess_state is not None and hasattr(preprocess_state, "to_dict"):
            pack["preprocess_state"] = preprocess_state.to_dict(store_on_cpu=True)

        if save_psd:
            # PSD scaling
            Tlen = T  # number of points in the FFT (could be zero-padded)
            U = _window_power_correction(window_vector)
            psd_two_sided = (dt / (Tlen * U)) * spectrum_xyz.abs().pow(2)  # (N,3,n_fft)

            pos_spectrum_xyz = extract_positive_frequency_half_spectrum(spectrum_xyz)
            pack["amplitude"] = pos_spectrum_xyz.abs()  # (N,3,n_fft//2+1)
            # Power spectrum |X|^2
            pack["power"] = pack["amplitude"].pow(2)  # (N,3,n_fft//2+1)

            pack["psd_one_sided"] = psd_two_sided[..., :(T//2 + 1)] # (N,3,n_fft//2+1)

        if save_minimal:
            return pack

        return pack
    # def reconstruct_processed(self, X: torch.Tensor, original_T: int) -> torch.Tensor:
    #     """
    #     Reconstruct the processed (preprocessed) trajectory by IFFT of the provided full spectrum.
    #     X: (N,3,n_fft) full complex spectrum
    #     original_T: original time length T (for truncation if n_fft != T)
    #     Returns: x_rec_processed (N,3,T)
    #     """
    #     x_rec = _ifft_full(X, norm=self.config.norm, T_target=original_T)
    #     return x_rec

    def invert_preprocessing(
        self,
        x_processed: torch.Tensor,      # (N,3,T) processed
        state: PreprocessStateStruct,
        fold_back_to_box: bool = False,
    ) -> torch.Tensor:
        """
        Invert preprocessing to return to original coordinate frame.
        x_processed: (N,3,T), the reconstructed processed trajectory

        If fold_back_to_box=True and PBC unwrapping was used, fold coordinates back to [0,L)
        (requires constant box_lengths=(3,) or per-frame (T,3)).
        """
        x = x_processed

        # Re-apply detrend that was removed
        if state.detrend_order is not None:
            if state.trend_coeffs is None or state.trend_time is None:
                raise RuntimeError("Missing detrend info to invert preprocessing.")
            T = x.shape[-1]
            if state.trend_time.numel() != T:
                raise RuntimeError("Time length mismatch when restoring trend.")
            A = _poly_design_matrix(state.trend_time.to(x.device, x.dtype), state.detrend_order)  # (T,p)
            M = x.shape[0] * x.shape[1]
            trend = (A @ state.trend_coeffs.to(x.device, x.dtype))  # (T,M)
            x = (x.reshape(M, T) + trend.transpose(0, 1)).reshape_as(x)

        # Undo rotations
        if state.aligned_rotation:
            if state.rotations is None:
                raise RuntimeError("Missing rotations to invert rotational alignment.")
            RT = state.rotations.to(x.device, x.dtype).transpose(-2, -1)
            # Apply per-frame
            X = x.permute(2, 0, 1)  # (T,N,3)
            Y = torch.matmul(RT, X.transpose(1, 2)).transpose(1, 2)  # (T,N,3)
            x = Y.permute(1, 2, 0).contiguous()  # (N,3,T)

        # Re-add COM
        if state.removed_com:
            if state.com is None:
                raise RuntimeError("Missing COM to restore translation.")
            x = x + state.com.to(x.device, x.dtype).unsqueeze(0)

        # If PBC was unwrapped, optionally fold back into box
        if state.used_unwrap and fold_back_to_box:
            boxL = state.box_lengths
            if boxL is None:
                raise RuntimeError("Cannot fold back to box without box_lengths")
            if boxL.ndim == 1:
                L = boxL.to(x.device, x.dtype)[None, :, None]
                x = x - torch.floor(x / L) * L
            else:
                # Per frame box; fold per frame
                # Build L_t as (1,3,T)
                Lt = boxL.to(x.device, x.dtype).transpose(0, 1).unsqueeze(0)  # (1,3,T)
                x = x - torch.floor(x / Lt) * Lt

        return x

    def reconstruct_and_unpack(
        self,
        pack: Dict[str, Any],
        reconstruct_from: str = "X", # "X", "V", "A"
    ) -> torch.Tensor:
        """
        Reconstruct the trajectory and unpack the pack.
        """
        if reconstruct_from not in ["X", "V", "A"]:
            raise ValueError(f"Invalid reconstruct_from={reconstruct_from}, must be in ['X', 'V', 'A'], which represent the displacement, velocity, and acceleration respectively.")

        dt = float(pack["dt"])
        T = int(pack["T"])
        self.config.device = pack["first_frame_coords"].device if pack["first_frame_coords"] is not None else None

        state = PreprocessStateStruct.from_dict(pack["preprocess_state"])
        dtype = torch.float64 if state.dtype_name == "float64" else torch.float32
        x_orig = pack["coords_t"]

        # if reconstruct_from == "X":
        #     # 恢复频谱
        #     if "X" in pack:
        #         X = pack["X"]
        #     elif "X_real" in pack and "X_imag" in pack:
        #         X = pack["X_real"].to(dtype)
        #         X = X + 1j * pack["X_imag"].to(dtype)
        #     elif "X_positive" in pack:
        #         # Direct complex tensor (when store_complex_as_float2=False)
        #         X = reconstruct_full_spectrum_from_positive_half(pack["X_positive"], n_fft)
        #     elif "X_positive_real" in pack and "X_positive_imag" in pack:
        #         # Reconstruct from separate real/imag parts (when store_complex_as_float2=True)
        #         X_positive = pack["X_positive_real"].to(dtype) + 1j * pack["X_positive_imag"].to(dtype)
        #         X = reconstruct_full_spectrum_from_positive_half(X_positive, n_fft)
        #     else:
        #         raise ValueError(f"Invalid reconstruct_from={reconstruct_from}, must be in ['X', 'V', 'A'], which represent the displacement, velocity, and acceleration respectively.")

        #     # 逆变换得到预处理域信号
        #     x_proc = self.reconstruct_processed(X, original_T=T)
        #     # 反向还原预处理
        #     x_orig_n3t = self.invert_preprocessing(x_proc, state, fold_back_to_box=state.used_unwrap)

        #     # 恢复为保存时的布局
        #     x_orig = _restore_layout(x_orig_n3t, state.original_layout)

        # elif reconstruct_from == "V":
        #     # 恢复速度谱
        #     if "V" in pack:
        #         V = pack["V"]
        #     elif "V_real" in pack and "V_imag" in pack:
        #         V = pack["V_real"].to(dtype) + 1j * pack["V_imag"].to(dtype)
        #     elif "V_positive" in pack:
        #         # Direct complex tensor (when store_complex_as_float2=False)
        #         V = reconstruct_full_spectrum_from_positive_half(pack["V_positive"], n_fft)
        #     elif "V_positive_real" in pack and "V_positive_imag" in pack:
        #         # Reconstruct from separate real/imag parts (when store_complex_as_float2=True)
        #         V_positive = pack["V_positive_real"].to(dtype) + 1j * pack["V_positive_imag"].to(dtype)
        #         V = reconstruct_full_spectrum_from_positive_half(V_positive, n_fft)
        #     else:
        #         raise ValueError(f"Invalid reconstruct_from={reconstruct_from}, must be in ['X', 'V', 'A'], which represent the displacement, velocity, and acceleration respectively.")

        #     # IFFT得到速度时域信号
        #     v = torch.fft.ifft(V, dim=-1, norm=self.config.norm).real  # (N,3,T)
            
        #     # 从速度积分得到位移（使用首帧作为锚点）
        #     # x[t] = x[0] + dt * cumsum(v, dim=time)
        #     v_n3t = v  # (N,3,T)
        #     x0_n3t = pack["first_frame_coords"].unsqueeze(-1) # (N,3,1)
        #     assert x0_n3t.shape == (v_n3t.shape[0], 3, 1)
            
        #     # 积分：x[t] = x[0] + dt * cumsum(v[0:t+1])
        #     x_rec_n3t = x0_n3t + dt * torch.cumsum(v_n3t, dim=-1)  # (N,3,T)
            
        #     # 恢复为保存时的布局
        #     x_orig = _restore_layout(x_rec_n3t, state.original_layout)
            
        # elif reconstruct_from == "A":
        #     raise NotImplementedError(f"Reconstructing from acceleration spectrum is not implemented yet.")

        return x_orig

    # ----- Serialization for the processor (config only) -----

    def save(self, fpath: str) -> None:
        payload = {
            "version": self.VERSION,
            "config": self.config.to_dict(),
        }
        torch.save(payload, fpath)

    @staticmethod
    def load(fpath: str) -> TrajectoryFFTProcessor:
        payload = torch.load(fpath, map_location="cpu")
        cfg = TrajectoryFFTConfig.from_dict(payload["config"])
        return TrajectoryFFTProcessor(cfg)

# ------------- Dataset Integration Helpers -------------

def attach_spectrum_to_label_full(
    processor: TrajectoryFFTProcessor,
    *,
    label_full_dict: Dict[str, Any],
    positions: torch.Tensor,           # the trajectory to analyze
    dt: float,
    masses: Optional[torch.Tensor] = None,
    box_lengths: Optional[torch.Tensor] = None,
    missing_mask: Optional[torch.Tensor] = None,
    layout_hint: Optional[str] = None,
    spectrum_key: str = "trajectory_spectrum",
    remove_com_for_alignment: bool = False,  # 是否去除质心用于对齐
    align_rotation_for_alignment: bool = False,  # 是否旋转对齐用于对齐
    **override_flags
) -> Dict[str, Any]:
    """
    Run the full pipeline and attach packed results into label_full_dict[spectrum_key].
    
    Args:
        remove_com_for_alignment: 如果True，会去除质心并保存对齐后的轨迹到label_full_dict["aligned_trajectory"]
        align_rotation_for_alignment: 如果True，会旋转对齐并保存对齐后的轨迹到label_full_dict["aligned_trajectory"]
    
    Returns the updated label_full_dict.
    """
    pack = processor.process_and_package(
        positions,
        dt=dt,
        masses=masses,
        box_lengths=box_lengths,
        missing_mask=missing_mask,
        layout_hint=layout_hint,
        remove_com=remove_com_for_alignment,
        align_rotation=align_rotation_for_alignment,
        **override_flags
    )
        
    label_full_dict[spectrum_key] = pack
    return label_full_dict

def reconstruct_trajectory_from_label_full(
    processor: TrajectoryFFTProcessor,
    label_full_dict: Dict[str, Any],
    spectrum_key: str = "trajectory_spectrum",
    reconstruct_from: str = "X",
    fold_back_to_box: bool = False,
) -> torch.Tensor:
    """
    Reconstruct the trajectory from label_full_dict.
    """
    pack = label_full_dict[spectrum_key]
    dt = float(pack["dt"])
    T = int(pack["T"])
    n_fft = int(pack["n_fft"])
    state = PreprocessStateStruct.from_dict(pack["preprocess_state"])
    dtype = torch.float64 if state.dtype_name == "float64" else torch.float32
    original_layout = pack.get("original_layout", "TN3")
    if reconstruct_from == "X":
        if "X_positive" in pack:
            # Direct complex tensor (when store_complex_as_float2=False)
            X = reconstruct_full_spectrum_from_positive_half(pack["X_positive"], n_fft)
        elif "X_positive_real" in pack and "X_positive_imag" in pack:
            # Reconstruct from separate real/imag parts (when store_complex_as_float2=True)
            X_positive = pack["X_positive_real"].to(dtype) + 1j * pack["X_positive_imag"].to(dtype)
            X = reconstruct_full_spectrum_from_positive_half(X_positive, n_fft)
        else:
            if "X" in pack:
                X = pack["X"]
            else:
                X = pack["X_real"].to(dtype)
                X = X + 1j * pack["X_imag"].to(dtype)
    else:
        if "X" in pack:
            X = pack["X"]
        else:
            X = pack["X_real"].to(dtype)
            X = X + 1j * pack["X_imag"].to(dtype)
    x_proc = processor.reconstruct_processed(X, original_T=T)
    x_orig_n3t = processor.invert_preprocessing(x_proc, state, fold_back_to_box=fold_back_to_box)
    x_orig = _restore_layout(x_orig_n3t, state.original_layout)
    return x_orig

def reconstruct_from_velocity_spectrum(
    label_full_dict: Dict[str, Any],
    spectrum_key: str = "trajectory_spectrum",
    first_frame: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    从速度谱重建位移轨迹，使用首帧坐标作为锚点。
    
    Args:
        label_full_dict: 包含频谱数据的字典
        spectrum_key: 频谱数据的键
        first_frame: 首帧坐标 (N, 3)，如果为None，会尝试从label_full_dict["coordinate"]获取
    
    Returns:
        重建的位移轨迹，shape为保存时的original_layout
    """
    pack = label_full_dict[spectrum_key]
    dt = float(pack["dt"])
    T = int(pack["T"])
    n_fft = int(pack["n_fft"])
    original_layout = pack.get("original_layout", "TN3")
    
    # 获取首帧坐标
    if first_frame is None:
        if "coordinate" in label_full_dict:
            first_frame = label_full_dict["coordinate"]
        else:
            raise ValueError("first_frame must be provided or available in label_full_dict['coordinate']")
       
    # 恢复速度谱
    dtype_name = pack.get("preprocess_state", {}).get("dtype_name", "float64")
    dtype = torch.float64 if dtype_name == "float64" else torch.float32
    if "V_positive" in pack:
        # Direct complex tensor (when store_complex_as_float2=False)
        V = reconstruct_full_spectrum_from_positive_half(pack["V_positive"], n_fft)
    elif "V_positive_real" in pack and "V_positive_imag" in pack:
        # Reconstruct from separate real/imag parts (when store_complex_as_float2=True)
        V_positive = pack["V_positive_real"].to(dtype) + 1j * pack["V_positive_imag"].to(dtype)
        V = reconstruct_full_spectrum_from_positive_half(V_positive, n_fft)
    else:
        if "V" in pack:
            V = pack["V"]
        else:
            V = pack["V_real"].to(dtype) + 1j * pack["V_imag"].to(dtype)
    
    # IFFT得到速度时域信号
    v = torch.fft.ifft(V, dim=-1, norm=pack.get("norm", "ortho")).real  # (N,3,T)
    
    # 从速度积分得到位移（使用首帧作为锚点）
    # x[t] = x[0] + dt * cumsum(v, dim=time)
    v_n3t = v  # (N,3,T)
    x0_n3t = first_frame.unsqueeze(-1) # (N,3,1)
    assert x0_n3t.shape == (v_n3t.shape[0], 3, 1)
    
    # 积分：x[t] = x[0] + dt * cumsum(v[0:t+1])
    x_rec_n3t = x0_n3t + dt * torch.cumsum(v_n3t, dim=-1)  # (N,3,T)
    
    # 恢复为保存时的布局
    x_rec = _restore_layout(x_rec_n3t, original_layout)
    return x_rec


# ------------- Spectrum Analysis Functions -------------

def compute_per_atom_spectra_from_pack(
    pack: Dict[str, Any],
    per_coordinate: bool = True
) -> Dict[str, torch.Tensor]:
    """
    从pack中提取逐原子的频谱（位移、速度、加速度）。
    
    Args:
        pack: 包含频谱数据的字典
        per_coordinate: 如果True，返回每个坐标轴(x,y,z)的频谱；如果False，返回每个原子的总幅度频谱
    
    Returns:
        包含以下键的字典:
        - displacement_spectrum_per_atom: (N, 3, n_fft) 或 (N, n_fft) 如果per_coordinate=False
        - velocity_spectrum_per_atom: 同上
        - acceleration_spectrum_per_atom: 同上
        - frequencies: (n_fft,)
    """
    # 恢复位移谱
    if "X" in pack:
        X = pack["X"]
    else:
        dtype_name = pack.get("preprocess_state", {}).get("dtype_name", "float64")
        dtype = torch.float64 if dtype_name == "float64" else torch.float32
        X = pack["X_real"].to(dtype) + 1j * pack["X_imag"].to(dtype)
    
    # 恢复速度谱
    if "V" in pack:
        V = pack["V"]
    else:
        dtype_name = pack.get("preprocess_state", {}).get("dtype_name", "float64")
        dtype = torch.float64 if dtype_name == "float64" else torch.float32
        V = pack["V_real"].to(dtype) + 1j * pack["V_imag"].to(dtype)
    
    # 恢复加速度谱
    if "A" in pack:
        A = pack["A"]
    else:
        dtype_name = pack.get("preprocess_state", {}).get("dtype_name", "float64")
        dtype = torch.float64 if dtype_name == "float64" else torch.float32
        A = pack["A_real"].to(dtype) + 1j * pack["A_imag"].to(dtype)
    
    # 计算幅度谱
    X_amp = X.abs()  # (N,3,n_fft)
    V_amp = V.abs()  # (N,3,n_fft)
    A_amp = A.abs()  # (N,3,n_fft)
    
    if not per_coordinate:
        # 计算每个原子的总幅度（对所有坐标轴求和）
        X_amp = X_amp.sum(dim=1)  # (N, n_fft)
        V_amp = V_amp.sum(dim=1)
        A_amp = A_amp.sum(dim=1)
    # freqs_x、freqs_v、freqs_a 应该完全相同
    freqs = pack["freqs_x"]
    
    return {
        "displacement_spectrum_per_atom": X_amp,
        "velocity_spectrum_per_atom": V_amp,
        "acceleration_spectrum_per_atom": A_amp,
        "frequencies": freqs,
    }


def analyze_snr(
    spectrum_per_atom: torch.Tensor,
    frequencies: torch.Tensor,
    noise_freq_threshold: Optional[float] = None
) -> Dict[str, float]:
    """
    分析信噪比（逐原子频谱的平均）。
    
    Args:
        spectrum_per_atom: (N, n_fft) 或 (N, 3, n_fft) 的频谱幅度
        frequencies: (n_fft,) 频率数组
        noise_freq_threshold: 高于此频率视为噪声 (Hz)，如果None则使用Nyquist频率的一半
    
    Returns:
        包含SNR信息的字典
    """
    # 如果是(N,3,n_fft)，先平均所有坐标轴
    if spectrum_per_atom.ndim == 3:
        spectrum_avg = spectrum_per_atom.mean(dim=1)  # (N, n_fft)
    else:
        spectrum_avg = spectrum_per_atom  # (N, n_fft)
    
    # 平均所有原子
    spectrum_global = spectrum_avg.mean(dim=0)  # (n_fft,)
    
    if noise_freq_threshold is None:
        noise_freq_threshold = frequencies.abs().max().item() / 2
    
    # 分离信号和噪声频段
    signal_idx = frequencies.abs() < noise_freq_threshold
    noise_idx = frequencies.abs() >= noise_freq_threshold
    
    signal_power = (spectrum_global[signal_idx]**2).sum().item()
    noise_power = (spectrum_global[noise_idx]**2).sum().item()
    
    if noise_power == 0:
        snr_db = float('inf')
    else:
        snr = signal_power / noise_power
        snr_db = 10 * math.log10(snr)
    
    return {
        'snr_db': snr_db,
        'signal_power': signal_power,
        'noise_power': noise_power,
        'noise_threshold': noise_freq_threshold
    }


def analyze_frequency_bands(
    spectrum_per_atom: torch.Tensor,
    frequencies: torch.Tensor
) -> Dict[str, Dict[str, float]]:
    """
    分析不同频段的能量分布。
    
    Args:
        spectrum_per_atom: (N, n_fft) 或 (N, 3, n_fft) 的频谱幅度
        frequencies: (n_fft,) 频率数组
    
    Returns:
        频段能量分布字典
    """
    # 如果是(N,3,n_fft)，先平均所有坐标轴
    if spectrum_per_atom.ndim == 3:
        spectrum_avg = spectrum_per_atom.mean(dim=1)  # (N, n_fft)
    else:
        spectrum_avg = spectrum_per_atom  # (N, n_fft)
    
    # 平均所有原子
    spectrum_global = spectrum_avg.mean(dim=0)  # (n_fft,)
    
    max_freq = frequencies.abs().max().item()
    bands = {
        'Ultra-low (0-10%)': (0, 0.1 * max_freq),
        'Low (10-30%)': (0.1 * max_freq, 0.3 * max_freq),
        'Mid (30-60%)': (0.3 * max_freq, 0.6 * max_freq),
        'High (60-100%)': (0.6 * max_freq, max_freq)
    }
    
    results = {}
    for band_name, (f_low, f_high) in bands.items():
        band_idx = (frequencies.abs() >= f_low) & (frequencies.abs() < f_high)
        energy = (spectrum_global[band_idx]**2).sum().item()
        results[band_name] = {
            'energy': energy,
            'freq_range': (f_low, f_high)
        }
    
    return results


def identify_flexible_regions(
    displacement_spectrum_per_atom: torch.Tensor,
    top_n: int = 10
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    识别最灵活的原子区域。
    
    Args:
        displacement_spectrum_per_atom: (N, n_fft) 或 (N, 3, n_fft) 的位移频谱
        top_n: 返回前N个最灵活的原子
    
    Returns:
        (top_indices, residue_energy) 元组
    """
    # 如果是(N,3,n_fft)，先对所有坐标轴求和
    if displacement_spectrum_per_atom.ndim == 3:
        spectrum_avg = displacement_spectrum_per_atom.sum(dim=1)  # (N, n_fft)
    else:
        spectrum_avg = displacement_spectrum_per_atom  # (N, n_fft)
    
    # 计算每个原子的总能量
    residue_energy = (spectrum_avg**2).sum(dim=1)  # (N,)
    
    # 找到最灵活的原子
    top_indices = torch.argsort(residue_energy, descending=True)[:top_n]
    
    return top_indices, residue_energy


def compute_effective_modes(
    displacement_spectrum_per_atom: torch.Tensor,
    frequencies: torch.Tensor,
    n_modes: int = 5
) -> Dict[str, Any]:
    """
    计算有效振动模式。
    
    Args:
        displacement_spectrum_per_atom: (N, n_fft) 或 (N, 3, n_fft) 的位移频谱
        frequencies: (n_fft,) 频率数组
        n_modes: 返回前N个模式
    
    Returns:
        包含模式信息的字典
    """
    # 如果是(N,3,n_fft)，先对所有坐标轴求和
    if displacement_spectrum_per_atom.ndim == 3:
        spectrum_avg = displacement_spectrum_per_atom.sum(dim=1)  # (N, n_fft)
    else:
        spectrum_avg = displacement_spectrum_per_atom  # (N, n_fft)
    
    # 平均所有原子
    spectrum_global = spectrum_avg.mean(dim=0)  # (n_fft,)
    
    # 计算能量
    energy = spectrum_global**2
    
    # 找到能量最大的频率
    top_freq_indices = torch.argsort(energy, descending=True)[:n_modes]
    
    modes = []
    for i, idx in enumerate(top_freq_indices):
        freq = frequencies[idx].item()
        period = 1.0 / abs(freq) if abs(freq) > 0 else float('inf')
        amplitude = spectrum_global[idx].item()
        energy_val = energy[idx].item()
        energy_pct = energy_val / energy.sum().item() * 100
        
        modes.append({
            'rank': i + 1,
            'frequency': freq,
            'period': period,
            'amplitude': amplitude,
            'energy': energy_val,
            'energy_percentage': energy_pct,
            'frequency_index': idx.item()
        })
    
    # 累积能量分析
    sorted_energy = torch.sort(energy, descending=True)[0]
    cumulative_energy = torch.cumsum(sorted_energy, dim=0) / energy.sum()
    
    n_modes_90 = (cumulative_energy >= 0.90).nonzero(as_tuple=False)[0].item() + 1 if (cumulative_energy >= 0.90).any() else len(cumulative_energy)
    n_modes_95 = (cumulative_energy >= 0.95).nonzero(as_tuple=False)[0].item() + 1 if (cumulative_energy >= 0.95).any() else len(cumulative_energy)
    n_modes_99 = (cumulative_energy >= 0.99).nonzero(as_tuple=False)[0].item() + 1 if (cumulative_energy >= 0.99).any() else len(cumulative_energy)
    n_modes_999 = (cumulative_energy >= 0.999).nonzero(as_tuple=False)[0].item() + 1 if (cumulative_energy >= 0.999).any() else len(cumulative_energy)
    
    return {
        'modes': modes,
        'cumulative_energy': cumulative_energy.cpu().numpy(),
        'n_modes_90': n_modes_90,
        'n_modes_95': n_modes_95,
        'n_modes_99': n_modes_99,
        'n_modes_999': n_modes_999
    }


def test_process_trajectory(
    traj_coord_path: str,
    remove_com_for_alignment: bool = False,
    align_rotation_for_alignment: bool = False,
    dt_str: str = "10ps",
    perform_analysis: bool = True,
) -> Dict[str, Any]:
    """
    处理轨迹并提取频谱，可选进行对齐和分析。
    
    Args:
        traj_coord_path: 轨迹文件路径，numpy array shape (T, N, 3)
        remove_com_for_alignment: 是否去除质心用于对齐
        align_rotation_for_alignment: 是否旋转对齐用于对齐
        dt_str: 时间步长字符串，如"10ps"
        perform_analysis: 是否执行完整分析
    
    Returns:
        包含所有结果的字典
    """
    arr = np.load(traj_coord_path)  # (T, N, 3) numpy数组
    traj = torch.from_numpy(arr).to(torch.float32)  # 转为tensor，可根据需要选择dtype
    print(f"traj.shape: {traj.shape}")
    T, N, _ = traj.shape
    first_frame_positions = traj[0]  # shape is (N, 3)
    masses = torch.ones(N)
    dt = parse_time_str_to_seconds(dt_str)

    cfg = TrajectoryFFTConfig(
        norm="ortho", use_double=True,
        remove_com=True, align_rotation=True, detrend_order=None,  # 这个设置会被pack_and_package中的覆盖
        window=None, zero_pad_to=None, also_windowed=False,  # 不使用窗口以保证精确重建
        device="cpu", store_dtype=torch.float32, store_complex_as_float2=True
    )
    proc = TrajectoryFFTProcessor(cfg)

    # 打包并存入 label_full_dict
    label_full = {"coordinate": first_frame_positions}  # 首帧坐标
    label_full = attach_spectrum_to_label_full(
        proc,
        label_full_dict=label_full,
        positions=traj,
        dt=dt,
        masses=masses,
        layout_hint="TN3",
        spectrum_key="trajectory_spectrum",
        remove_com_for_alignment=remove_com_for_alignment,
        align_rotation_for_alignment=align_rotation_for_alignment,
    )

    # 从 label_full_dict 重建（使用位移谱）
    x_rec = reconstruct_trajectory_from_label_full(proc, label_full, fold_back_to_box=False)
    
    # 从速度谱重建
    x_rec_from_vel = reconstruct_from_velocity_spectrum(label_full, first_frame=first_frame_positions)
    
    # 计算重建误差（与对齐后的ground truth比较，如果有的话）
    if "aligned_trajectory" in label_full:
        traj_ref = label_full["aligned_trajectory"]
        error_displacement = torch.norm(x_rec - traj_ref).item()
        error_velocity = torch.norm(x_rec_from_vel - traj_ref).item()
        print(f"重建误差 (位移谱): {error_displacement:.6e}")
        print(f"重建误差 (速度谱): {error_velocity:.6e}")
    else:
        error_displacement = torch.norm(x_rec - traj).item()
        error_velocity = torch.norm(x_rec_from_vel - traj).item()
        print(f"重建误差 (位移谱): {error_displacement:.6e}")
        print(f"重建误差 (速度谱): {error_velocity:.6e}")
    
    print("Reconstructed layout shape (T,N,3):", tuple(x_rec.shape))
    print("label_full keys:", label_full.keys())
    
    # 执行分析
    if perform_analysis:
        pack = label_full["trajectory_spectrum"]
        
        # 提取逐原子频谱
        spectra = compute_per_atom_spectra_from_pack(pack, per_coordinate=False)
        disp_spectrum = spectra["displacement_spectrum_per_atom"]  # (N, n_fft)
        vel_spectrum = spectra["velocity_spectrum_per_atom"]
        acc_spectrum = spectra["acceleration_spectrum_per_atom"]
        freqs = spectra["frequencies"]
        
        print("\n" + "="*60)
        print("频谱分析结果")
        print("="*60)
        
        # 信噪比分析
        snr_disp = analyze_snr(disp_spectrum, freqs)
        snr_vel = analyze_snr(vel_spectrum, freqs)
        snr_acc = analyze_snr(acc_spectrum, freqs)
        
        print(f"\n信噪比 (dB):")
        print(f"  位移谱: {snr_disp['snr_db']:.2f} dB")
        print(f"  速度谱: {snr_vel['snr_db']:.2f} dB")
        print(f"  加速度谱: {snr_acc['snr_db']:.2f} dB")
        
        # 频段能量分析
        bands_disp = analyze_frequency_bands(disp_spectrum, freqs)
        print(f"\n频段能量分布 (位移谱):")
        for band_name, band_data in bands_disp.items():
            print(f"  {band_name}: {band_data['energy']:.6e}")
        
        # 识别灵活区域
        top_indices, residue_energy = identify_flexible_regions(disp_spectrum, top_n=10)
        print(f"\n最灵活的10个原子:")
        for i, idx in enumerate(top_indices):
            print(f"  #{i+1}: 原子 {idx.item()}, 能量: {residue_energy[idx].item():.6e}")
        
        # 有效振动模式
        modes_info = compute_effective_modes(disp_spectrum, freqs, n_modes=5)
        print(f"\n前5个有效振动模式:")
        for mode in modes_info['modes']:
            print(f"  模式 #{mode['rank']}: 频率={mode['frequency']:.6f} Hz, "
                  f"周期={mode['period']:.6f} s, 能量占比={mode['energy_percentage']:.2f}%")
        
        print(f"\n累积能量分析:")
        print(f"  90% 能量需要: {modes_info['n_modes_90']} 个模式")
        print(f"  95% 能量需要: {modes_info['n_modes_95']} 个模式")
        print(f"  99% 能量需要: {modes_info['n_modes_99']} 个模式")
        print(f"  99.9% 能量需要: {modes_info['n_modes_999']} 个模式")
        
        # 将分析结果保存到label_full
        label_full["analysis"] = {
            "snr": {
                "displacement": snr_disp,
                "velocity": snr_vel,
                "acceleration": snr_acc
            },
            "frequency_bands": bands_disp,
            "flexible_regions": {
                "top_indices": top_indices.cpu(),
                "residue_energy": residue_energy.cpu()
            },
            "effective_modes": modes_info
        }
    
    return label_full


# ------------- Minimal Example -------------

if __name__ == "__main__":
    traj_coord_path = "./test/data/1ear_A_R1_no_water_frames_coords.npy"

    # 示例：不使用对齐
    label_full = test_process_trajectory(
        traj_coord_path,
        remove_com_for_alignment=False,
        align_rotation_for_alignment=False,
        dt_str="10ps",
        perform_analysis=True
    )
    
    # 示例：使用对齐
    # label_full = test_process_trajectory(
    #     traj_coord_path,
    #     remove_com_for_alignment=True,
    #     align_rotation_for_alignment=True,
    #     dt_str="10ps",
    #     perform_analysis=True
    # )
    
    # torch.manual_seed(0)
    # # 造一个 (T,N,3) 形式的短轨迹并测试
    # T, N = 128, 10
    # dt = 1e-12
    # t = torch.arange(T) * dt
    # base = torch.randn(N, 3)
    # freq = 1.5e12
    # amp = 0.02
    # traj = torch.empty(T, N, 3)
    # for k in range(T):
    #     # 对每个时间点，添加随机噪声并施加周期性扰动
    #     traj[k] = base + amp * torch.sin(2 * math.pi * freq * t[k]) * torch.randn(N, 3)
    
    # masses = torch.ones(N)
    # cfg = TrajectoryFFTConfig(
    #     norm="backward", use_double=True,
    #     remove_com=True, align_rotation=False, detrend_order=1,
    #     window="hann", zero_pad_to=None, also_windowed=True,
    #     device="cpu", store_dtype=torch.float32, store_complex_as_float2=True
    # )
    # proc = TrajectoryFFTProcessor(cfg)

    # # 打包并存入 label_full_dict
    # label_full = {"coordinate": torch.randn(N, 3)}  # 仅示例，静态结构
    # label_full = attach_spectrum_to_label_full(
    #     proc,
    #     label_full_dict=label_full,
    #     positions=traj,
    #     dt=dt,
    #     masses=masses,
    #     layout_hint="TN3"
    # )

    # # 从 label_full_dict 重建
    # x_rec = reconstruct_trajectory_from_label_full(proc, label_full, fold_back_to_box=False)
    # print("Reconstructed layout shape (T,N,3):", tuple(x_rec.shape))
    # print("Reconstructed loss (T,N,3):", torch.norm(x_rec - traj))


    # # 造一个 (N,3,T) 形式的短轨迹并测试
    # torch.manual_seed(42)
    # base2 = torch.randn(N, 3)
    # traj2 = torch.empty(N, 3, T)
    # for k in range(T):
    #     traj2[..., k] = base2 + amp * torch.sin(2 * math.pi * freq * t[k]) * torch.randn(N, 3)
    # # masses 同样用
    # label_full2 = {"coordinate": torch.randn(N, 3)}
    # label_full2 = attach_spectrum_to_label_full(
    #     proc,
    #     label_full_dict=label_full2,
    #     positions=traj2,
    #     dt=dt,
    #     masses=masses,
    #     layout_hint="N3T"
    # )

    # x_rec2 = reconstruct_trajectory_from_label_full(proc, label_full2, fold_back_to_box=False)
    # print("Reconstructed layout shape (N,3,T):", tuple(x_rec2.shape))
    # print("Reconstructed loss (N,3,T):", torch.norm(x_rec2 - traj2))

# # -------------------------
# # Example usage and tests
# # -------------------------

# def _example():
#     """
#     Minimal example demonstrating:
#     - Synthetic trajectory
#     - Preprocess with COM removal + rotational alignment (optional)
#     - Full spectrum extraction
#     - Exact reconstruction in original coordinates
#     """
#     torch.manual_seed(0)
#     device = torch.device("cpu")
#     dtype = torch.float64

#     # Synthetic data
#     N, T = 5, 1024
#     dt = 1e-12  # 1 ps sampling
#     t = torch.arange(T, dtype=dtype, device=device) * dt

#     # Build a trajectory: small vibrations around initial positions + rigid translation and rotation
#     base = torch.randn(N, 3, dtype=dtype, device=device) * 1.0  # initial structure
#     # Internal vibrations at multiple freqs
#     freqs = torch.tensor([1e12, 3e12], dtype=dtype, device=device)  # Hz
#     vib = (
#         0.05 * torch.sin(2 * math.pi * freqs[0] * t)[None, None, :] * torch.randn(N, 3, 1, dtype=dtype, device=device)
#         + 0.03 * torch.sin(2 * math.pi * freqs[1] * t)[None, None, :] * torch.randn(N, 3, 1, dtype=dtype, device=device)
#     )  # (N,3,T)

#     # Rigid translation (drift)
#     drift = (torch.tensor([0.001, -0.002, 0.0005], dtype=dtype, device=device)[:, None]) * (t / t[-1])  # (3,T)
#     # Construct rotation around z by a small angle changing over time
#     theta = 0.5 * torch.sin(2 * math.pi * 0.5e12 * t)  # radians
#     cos_th, sin_th = torch.cos(theta), torch.sin(theta)
#     Rz = torch.stack([
#         torch.stack([cos_th, -sin_th, torch.zeros_like(theta)], dim=0),
#         torch.stack([sin_th,  cos_th, torch.zeros_like(theta)], dim=0),
#         torch.stack([torch.zeros_like(theta), torch.zeros_like(theta), torch.ones_like(theta)], dim=0),
#     ], dim=0)  # (3,3,T)

#     # Apply rigid rotation and translation
#     base_centered = base - base.mean(dim=0, keepdim=True)
#     X = torch.empty(N, 3, T, dtype=dtype, device=device)
#     for k in range(T):
#         X[:, :, k] = (base_centered @ Rz[:, :, k].T) + drift[:, k]  # rotate + translate frame k
#     X = X + vib  # add internal vibrations
#     # Make it non-centered by adding a base offset
#     X = X + torch.tensor([10.0, -20.0, 5.0], dtype=dtype, device=device)[None, :, None]

#     # Optional missing values mask
#     mask = torch.ones_like(X, dtype=torch.bool)
#     # Introduce some missing samples
#     mask[0, 0, 100:110] = False
#     X_missing = X.clone()
#     X_missing[~mask] = 0.0  # will be interpolated

#     # Masses (optional); if None, equal masses
#     masses = torch.rand(N, dtype=dtype, device=device) + 12.0  # e.g., ~carbon-like

#     # Build processor
#     proc = ProteinTrajectoryFFT(dt=dt, masses=masses, norm="backward", use_double=True, device=device)

#     # Preprocess: impute missing, remove COM, align rotation, detrend linear drift
#     Xp = proc.preprocess(
#         X_missing,
#         missing_mask=mask,
#         impute_missing=True,
#         remove_com=True,
#         align_rotation=True,
#         detrend_order=1,
#         perform_unwrap=False,
#         box_lengths=None,
#     )

#     # Compute spectrum (unwindowed for exact reconstruction; also compute Hann-windowed for PSD)
#     spec = proc.compute_spectrum(Xp, window="hann", zero_pad_to=None, also_windowed=True)
#     Xf = spec["X"]            # unwindowed full spectrum (N,3,T)
#     freqs = spec["freqs"]
#     omega = spec["omega"]

#     # Reconstruct processed series via IFFT
#     Xp_rec = proc.reconstruct_processed(Xf, original_T=Xp.shape[-1])

#     # Invert preprocessing to recover original coordinates
#     X_rec = proc.invert_preprocessing(Xp_rec, fold_back_to_box=False)

#     # Check reconstruction quality (should be close; loss from detrending+restoration is numerical only)
#     max_err = (X_rec - X).abs().max().item()
#     print(f"Max abs reconstruction error: {max_err:.3e}")

#     # Compute spectral metrics (use windowed spectrum for PSD)
#     if "X_windowed" in spec:
#         metrics = proc.compute_spectral_metrics(spec["X_windowed"], freqs, window=spec["window"])
#     else:
#         metrics = proc.compute_spectral_metrics(Xf, freqs, window=None)

#     print("Spectrum shape:", Xf.shape, "Freq bins:", freqs.numel())
#     print("PSD shape:", metrics["psd_two_sided"].shape)


# if __name__ == "__main__":
#     _example()
