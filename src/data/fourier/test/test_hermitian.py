import torch
from typing import Optional, Tuple, Union

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
        X_positive: 正频半谱，shape (..., n_positive)，其中 n_positive = n_fft // 2 + 1
        n_fft: 完整频谱的长度。如果为None，会从 X_positive 的形状推断：n_fft = 2 * (n_positive - 1)
    
    Returns:
        X_full: 完整频谱，shape (..., n_fft)
    
    注意：
        - 如果 n_fft 是偶数，X_positive 应该包含 [0, 1, ..., n_fft/2]（共 n_fft//2 + 1 个点）
        - 如果 n_fft 是奇数，X_positive 应该包含 [0, 1, ..., (n_fft-1)/2]（共 (n_fft+1)//2 个点）
        - 推断规则：n_fft = 2 * (n_positive - 1) 适用于偶数情况
    """
    n_positive = X_positive.shape[-1]
    
    # 推断 n_fft
    if n_fft is None:
        # 假设是偶数情况：n_positive = n_fft // 2 + 1
        n_fft = 2 * (n_positive - 1)
    
    # 验证 n_positive 是否与 n_fft 匹配
    expected_n_positive = n_fft // 2 + 1
    if n_positive != expected_n_positive:
        raise ValueError(
            f"n_positive ({n_positive}) 与 n_fft ({n_fft}) 不匹配。"
            f"对于 n_fft={n_fft}，期望 n_positive={expected_n_positive}"
        )
    
    # 创建完整频谱
    shape = list(X_positive.shape)
    shape[-1] = n_fft
    X_full = torch.zeros(shape, dtype=X_positive.dtype, device=X_positive.device)
    
    # 复制正频率部分（包括DC和Nyquist）
    X_full[..., :n_positive] = X_positive
    print(f"n_fft: {n_fft}, n_positive: {n_positive}")
    # 通过厄米共轭补齐负频率部分
    # 对于 k > 0: X[-k] = X*[k]
    # 注意：torch.fft.fftfreq 的负频率索引是 [n_fft//2+1, ..., n_fft-1]
    if n_positive > 1:
        if n_fft % 2 == 0:
            # 偶数情况：Nyquist频率在索引 n_fft//2，是实数
            # 负频率索引：n_fft//2+1 到 n_fft-1
            # 对应的正频率索引：n_fft//2-1 到 1（反向）
            # X_full[..., n_fft-k] = X_positive[..., k].conj() for k in [1, n_fft//2-1]

            # 提取需要共轭的部分：索引 1 到 n_fft//2-1
            pos_part = X_positive[..., 1:n_fft//2]  # (..., n_fft//2-1)
            # 反转并取共轭，填充到负频率位置 [n_fft-1, n_fft-2, ..., n_fft//2+1]
            neg_start = n_fft // 2 + 1
            neg_end = n_fft

        else:
            # 奇数情况：没有Nyquist频率
            # 负频率索引：(n_fft+1)//2 到 n_fft-1
            # 对应的正频率索引：(n_fft-1)//2 到 1（反向）
            
            # X_full[..., n_fft-k] = X_positive[..., k].conj() for k in [1, (n_fft-1)//2]
            # if (n_fft + 1) // 2 > 1:
            # 提取需要共轭的部分：索引 1 到 (n_fft-1)//2
            pos_part = X_positive[..., 1:n_positive] # (..., (n_fft-1)//2)
            # # 反转并取共轭，填充到负频率位置 [n_fft-1, n_fft-2, ..., (n_fft+1)//2]
            # X_full[..., n_fft-1:(n_fft+1)//2-1:-1] = pos_part.conj().flip(dims=[-1])
            neg_start = (n_fft + 1) // 2
            neg_end = n_fft

        neg_part = torch.flip(pos_part, dims=[-1]).conj()
        X_full[..., neg_start:neg_end] = neg_part

    return X_full

# 测试偶数情况
n_fft = 10
x = torch.randn(n_fft, dtype=torch.float64)
X_full = torch.fft.fft(x)
X_pos = extract_positive_frequency_half_spectrum(X_full)
X_reconstructed = reconstruct_full_spectrum_from_positive_half(X_pos, n_fft)
print(f"偶数重建误差: {torch.abs(X_full - X_reconstructed).max().item():.2e}")

# 测试奇数情况
n_fft = 11
x = torch.randn(n_fft, dtype=torch.float64)
X_full = torch.fft.fft(x)
X_pos = extract_positive_frequency_half_spectrum(X_full)
X_reconstructed = reconstruct_full_spectrum_from_positive_half(X_pos, n_fft)
print(f"奇数重建误差: {torch.abs(X_full - X_reconstructed).max().item():.2e}")
