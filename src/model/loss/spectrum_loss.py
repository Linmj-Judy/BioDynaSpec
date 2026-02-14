import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import logging
from src.protenix.metrics.rmsd import align_pred_to_true

def _amplitude_from_real_imag(
    real: torch.Tensor,
    imag: torch.Tensor,
    eps_amp: float = 1e-12,
) -> torch.Tensor:
    """
    从实部和虚部计算幅度谱:
        A = sqrt(Re^2 + Im^2 + eps_amp)
    real, imag: 形状相同
    返回: 与输入同形状的幅度张量
    """
    return torch.sqrt(real**2 + imag**2 + eps_amp)


def rigid_align_spectrum_by_dc(
    X_pred_real_positive: torch.Tensor,   # (B, N_atom, 3, seq_len) or (N_atom, 3, seq_len)
    X_pred_imag_positive: torch.Tensor,   # (B, N_atom, 3, seq_len) or (N_atom, 3, seq_len)
    true_real_b: torch.Tensor,            # (1 or B, N_atom, 3, seq_len) or (N_atom, 3, seq_len)
    true_imag_b: torch.Tensor,            # (1 or B, N_atom, 3, seq_len) or (N_atom, 3, seq_len)
    atom_mask: Optional[torch.Tensor] = None,  # (N_atom) or (B, N_atom)
    weight: Optional[torch.Tensor] = None,     # (N_atom) or (B, N_atom)
    allowing_reflection: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    用 0Hz 实部(平均结构)估计全局旋转 R，并将 R 应用到 Pred 的所有频率(实部+虚部)。
    支持 batched 和非 batched 输入。
    返回: 对齐后的 (pred_real_aligned, pred_imag_aligned, R)
    """
    # 判断输入是否带 batch 维 (B, N_atom, 3, seq_len) 或 (N_atom, 3, seq_len)
    is_batched = X_pred_real_positive.dim() == 4
    device = X_pred_real_positive.device
    dtype = X_pred_real_positive.dtype

    if is_batched:
        B, N, _, L = X_pred_real_positive.shape
        pred_dc = X_pred_real_positive[..., 0]  # (B, N_atom, 3)
        # true_real_b shape: (1, N_atom, 3, seq_len) or (B, N_atom, 3, seq_len)
        true_dc = true_real_b[..., 0]           # (1, N_atom, 3) or (B, N_atom, 3)
        if true_dc.dim() == 3:
            # shape: (N_atom, 3), must unsqueeze(0) to (1, N_atom, 3)
            true_dc = true_dc.unsqueeze(0)
        if true_dc.shape[0] == 1 and B != 1:
            true_dc = true_dc.expand(B, -1, -1)
        if atom_mask is not None:
            if atom_mask.dim() == 1:
                atom_mask_b = atom_mask.unsqueeze(0).expand(B, -1).to(device)
            else:
                atom_mask_b = atom_mask
        else:
            atom_mask_b = None
        if weight is not None:
            if weight.dim() == 1:
                weight_b = weight.unsqueeze(0).expand(B, -1).to(device)
            else:
                weight_b = weight
        else:
            weight_b = None
        # align
        _, rot, _ = align_pred_to_true(
            pred_pose=pred_dc,
            true_pose=true_dc,
            atom_mask=atom_mask_b,
            weight=weight_b,
            allowing_reflection=allowing_reflection,
        )  # rot: (B, 3, 3)
        # 应用旋转
        def apply_rot(x: torch.Tensor, rot: torch.Tensor) -> torch.Tensor:
            x_perm = x.permute(0, 1, 3, 2).contiguous()      # (B, N, L, 3)
            x_flat = x_perm.view(B, N * L, 3)               # (B, N*L, 3)
            x_rot = torch.matmul(x_flat, rot.transpose(-1, -2))  # (B, N*L, 3)
            x_out = x_rot.view(B, N, L, 3).permute(0, 1, 3, 2).contiguous()  # (B, N, 3, L)
            return x_out
        X_pred_real_aligned = apply_rot(X_pred_real_positive, rot).to(dtype)
        X_pred_imag_aligned = apply_rot(X_pred_imag_positive, rot).to(dtype)
    else:
        # 单个样本 (N_atom, 3, seq_len)
        N, _, L = X_pred_real_positive.shape
        pred_dc = X_pred_real_positive[..., 0]  # (N_atom, 3)
        true_dc = true_real_b[..., 0]           # (N_atom, 3)
        if atom_mask is not None:
            atom_mask_b = atom_mask.to(device)
        else:
            atom_mask_b = None
        if weight is not None:
            weight_b = weight.to(device)
        else:
            weight_b = None
        # align
        _, rot, _ = align_pred_to_true(
            pred_pose=pred_dc,
            true_pose=true_dc,
            atom_mask=atom_mask_b,
            weight=weight_b,
            allowing_reflection=allowing_reflection,
        )  # rot: (3, 3)
        # 应用旋转
        def apply_rot_single(x: torch.Tensor, rot: torch.Tensor) -> torch.Tensor:
            # x: (N, 3, L)
            x_perm = x.permute(0, 2, 1).contiguous()      # (N, L, 3)
            x_flat = x_perm.view(N * L, 3)                # (N*L, 3)
            x_rot = torch.matmul(x_flat, rot.transpose(-1, -2))  # (N*L, 3)
            x_out = x_rot.view(N, L, 3).permute(0, 2, 1).contiguous()   # (N, 3, L)
            return x_out
        X_pred_real_aligned = apply_rot_single(X_pred_real_positive, rot).to(dtype)
        X_pred_imag_aligned = apply_rot_single(X_pred_imag_positive, rot).to(dtype)
    return X_pred_real_aligned, X_pred_imag_aligned, rot


def compute_spectrum_mse_mae_after_rigid_align(
    X_pred_real_positive: torch.Tensor,   # (B, N_atom, 3, seq_len)
    X_pred_imag_positive: torch.Tensor,   # (B, N_atom, 3, seq_len)
    true_real_b: torch.Tensor,            # (1 or B, N_atom, 3, seq_len)
    true_imag_b: torch.Tensor,            # (1 or B, N_atom, 3, seq_len)
    atom_mask: Optional[torch.Tensor] = None,  # (N_atom) or (B, N_atom)
    weight: Optional[torch.Tensor] = None,     # (N_atom) or (B, N_atom)
    allowing_reflection: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    频域刚性对齐 + 频谱 MSE/MAE（与你现有实现一致：复数的实/虚差平方和取 mean）
    返回: mse_loss, mae_loss, rot
    """
    Xr_aligned, Xi_aligned, rot = rigid_align_spectrum_by_dc(
        X_pred_real_positive=X_pred_real_positive,
        X_pred_imag_positive=X_pred_imag_positive,
        true_real_b=true_real_b,
        true_imag_b=true_imag_b,
        atom_mask=atom_mask,
        weight=weight,
        allowing_reflection=allowing_reflection,
    )

    # # 广播 true 到 B
    # if true_real_b.shape[0] == 1 and Xr_aligned.shape[0] != 1:
    #     true_real = true_real_b.expand(Xr_aligned.shape[0], -1, -1, -1)
    #     true_imag = true_imag_b.expand(Xr_aligned.shape[0], -1, -1, -1)
    # else:
    #     true_real, true_imag = true_real_b, true_imag_b
    true_real, true_imag = true_real_b, true_imag_b

    diff_real = Xr_aligned - true_real
    diff_imag = Xi_aligned - true_imag

    mse_loss = (diff_real**2 + diff_imag**2).mean()
    mae_loss = (diff_real.abs() + diff_imag.abs()).mean()
    return mse_loss, mae_loss, rot

class SpectrumLoss(nn.Module):
    def __init__(self, configs):
        super(SpectrumLoss, self).__init__()
        self.configs = configs
        self.eps = self.configs.spectrum_loss.eps
        # 安全地获取权重，如果配置中不存在则使用默认值
        weight_config = self.configs.spectrum_loss.weight
        self.loss_weight = {
            "spectrum_mse_loss": getattr(weight_config, "mse_loss", 1.0),
            "spectrum_mae_loss": getattr(weight_config, "mae_loss", 1.0),
            "smooth_l1_loss": getattr(weight_config, "smooth_l1_loss", 1.0),
            "log_spectral_l1_loss": getattr(weight_config, "log_spectral_l1_loss", 1.0),
            "cosine_similarity_loss": getattr(weight_config, "cosine_similarity_loss", 1.0),
            "spectral_convergence_loss": getattr(weight_config, "spectral_convergence_loss", 1.0),
            "amplitude_mse_loss": getattr(weight_config, "amplitude_mse_loss", 1.0),  # 添加缺失的键
        }

    def _compute_relative_mse_mae_loss(self,
        X_pred_real_positive: torch.Tensor,   # (B, N_atom, 3, seq_len)
        X_pred_imag_positive: torch.Tensor,   # (B, N_atom, 3, seq_len)
        true_real_b: torch.Tensor,   # (1, N_atom, 3, seq_len)
        true_imag_b: torch.Tensor,   # (1, N_atom, 3, seq_len)
    ):
        return compute_spectrum_mse_mae_after_rigid_align(
            X_pred_real_positive=X_pred_real_positive,
            X_pred_imag_positive=X_pred_imag_positive,
            true_real_b=true_real_b,
            true_imag_b=true_imag_b,
            allowing_reflection=False,
        )
        # diff_real = X_pred_real_positive - true_real_b   # (B, N_atom, 3, seq_len)
        # diff_imag = X_pred_imag_positive - true_imag_b   # (B, N_atom, 3, seq_len)
     
        # mse_loss = (diff_real**2 + diff_imag**2).mean()
        # mae_loss = (diff_real.abs() + diff_imag.abs()).mean()
        # # # === 归一化因子 ===
        # # # 计算真实值的平均能量 (Scale)，加上 eps 防止除以 0
        # # scale = (true_real_b**2 + true_imag_b**2).mean() + 1e-6
        
        # # # 归一化后的 Loss
        # # mse_loss = mse_loss.mean() / scale
        # # mae_loss = mae_loss.mean() / torch.sqrt(scale)
        # return mse_loss, mae_loss

    def _compute_smooth_l1_loss(self,
        X_pred_real_positive: torch.Tensor,   # (B, N_atom, 3, seq_len)
        X_pred_imag_positive: torch.Tensor,   # (B, N_atom, 3, seq_len)
        true_real_b: torch.Tensor,   # (1, N_atom, 3, seq_len)
        true_imag_b: torch.Tensor,   # (1, N_atom, 3, seq_len)
    ):
        loss_real = F.smooth_l1_loss(X_pred_real_positive, true_real_b, beta=10.0)
        loss_imag = F.smooth_l1_loss(X_pred_imag_positive, true_imag_b, beta=10.0)
        # 归一化因子
        scale = (true_real_b.abs() + true_imag_b.abs()).mean().detach() + 1e-6
        return (loss_real.mean() + loss_imag.mean()) / scale

    def _compute_log_spectral_l1_loss(
        self,
        amplitude_pred: torch.Tensor, # (B, N_atom, 3, seq_len)
        amplitude_true: torch.Tensor, # (1, N_atom, 3, seq_len)
    ):
        log_amplitude_true = torch.log(amplitude_true + self.eps)      # (1, N_atom, 3, seq_len)
        log_amplitude_pred = torch.log(amplitude_pred + self.eps)        # (B, N_atom, 3, seq_len)
        diff = log_amplitude_pred - log_amplitude_true              # 广播到 (B, N_atom, 3, seq_len)
        return diff.abs().mean()

    def _compute_cosine_similarity_loss(
        self,
        amplitude_pred: torch.Tensor, # (B, N_atom, 3, seq_len)
        amplitude_true: torch.Tensor, # (1, N_atom, 3, seq_len)
        loss_weight: torch.Tensor=None, # (seq_len,) loss weight
    ):
        # 频率权重 w_k
        if loss_weight is None:
            w = torch.ones(amplitude_true.shape[-1], device=amplitude_true.device, dtype=amplitude_true.dtype) # (seq_len, )
        else:
            assert loss_weight.shape[-1] == amplitude_true.shape[-1], "loss_weight 长度必须等于 seq_len"
            w = loss_weight.to(device=amplitude_true.device, dtype=amplitude_true.dtype)

        # w: (seq_len,)
        # x: (..., N_atom, 3, seq_len)
        # 让 w 变成 (..., 1, 1, seq_len) 以便在倒数第三和倒数第二维上广播
        w_view = w.view(*([1] * (amplitude_true.dim() - 1)), -1)  # (..., 1, 1, seq_len)
        w_true = w_view.expand_as(amplitude_true)  # (...,N_atom,3,seq_len)
        w_pred = w_view.expand_as(amplitude_pred)  # (...,N_atom,3,seq_len)

        if amplitude_true.dim() == 3 and amplitude_pred.dim() == 3:
            v_true = (w_true * amplitude_true).reshape(1, -1)            # (..., N_atom*3*seq_len)
            v_pred = (w_pred * amplitude_pred).reshape(1, -1)  # (..., N_atom*3*seq_len)
        else:
            v_true = (w_true * amplitude_true).reshape(amplitude_true.shape[0], -1)            # (..., N_atom*3*seq_len)
            v_pred = (w_pred * amplitude_pred).reshape(amplitude_pred.shape[0], -1)  # (..., N_atom*3*seq_len)

        # 加权 cosine 相似度
        dot = (v_true * v_pred).sum(dim=-1)                          # (B,)
        norm_true = v_true.norm(dim=-1)                              # (B,)
        norm_pred = v_pred.norm(dim=-1)                              # (B,)
        cos_sim = dot / (norm_true * norm_pred + self.eps)                # (B,)
        loss_cos = 1.0 - cos_sim.mean()
        return loss_cos

    def _compute_spectral_convergence_loss(
        self,
        amplitude_pred: torch.Tensor, # (B, N_atom, 3, seq_len)
        amplitude_true: torch.Tensor, # (1, N_atom, 3, seq_len)
    ):
        # Spectral Convergence
        diff = (amplitude_pred - amplitude_true).reshape(amplitude_pred.shape[0], -1)                    # (B, D)
        num = diff.norm(dim=-1)                                      # (B,)
        den = amplitude_true.expand(amplitude_pred.shape[0], -1, -1, -1).reshape(amplitude_pred.shape[0], -1).norm(dim=-1) + self.eps             # (B,)
        spectral_convergence = (num / den).mean()                    # 标量
        return spectral_convergence

    def aggregate_losses(
        self, loss_fns: dict
    ) -> tuple[torch.Tensor, dict]:
        """
        Aggregates multiple loss functions and their respective metrics.

        Args:
            loss_fns (dict): Dictionary of loss functions to be aggregated.
            has_valid_resolution (Optional[torch.Tensor]): Tensor indicating valid resolutions. Defaults to None.

        Returns:
            tuple[torch.Tensor, dict]:
                - cum_loss (torch.Tensor): Cumulative loss.
                - all_metrics (dict): Dictionary containing all metrics.
        """
        cum_loss = 0.0
        all_metrics = {}
        for loss_name, loss_fn in loss_fns.items():
            # 防御性处理：如果权重不存在，使用默认值 1.0
            weight = self.loss_weight.get(loss_name, 1.0)
            if loss_name not in self.loss_weight:
                logging.warning(f"Loss weight for '{loss_name}' not found in config, using default 1.0")
            loss_outputs = loss_fn()
            if isinstance(loss_outputs, tuple):
                loss, metrics = loss_outputs
            else:
                assert isinstance(loss_outputs, torch.Tensor)
                loss, metrics = loss_outputs, {}

            all_metrics.update(
                {f"{loss_name}/{key}": val for key, val in metrics.items()}
            )
            if torch.isnan(loss) or torch.isinf(loss):
                logging.warning(f"{loss_name} loss is NaN. Skipping...")
            
            all_metrics[loss_name] = loss.detach().clone()
            all_metrics[f"weighted_{loss_name}"] = weight * loss.detach().clone()

            cum_loss = cum_loss + weight * loss
        all_metrics["loss"] = cum_loss.detach().clone()

        return cum_loss, all_metrics

    def forward(
        self,
        pred_fourier_package: dict,
        true_fourier_package: dict,
        loss_weight: Optional[torch.Tensor] = None, # (seq_len, )
    ):
        """
        复谱 MSE loss (仅在正频半谱上):

            L_MSE = mean_{B, i, c, k} [ (Re_hat - Re)^2 + (Im_hat - Im)^2 ]

        复谱 MAE loss (仅在正频半谱上):

            L_MAE = mean_{B, i, c, k} [ |Re_hat - Re| + |Im_hat - Im| ]

        log-spectral L1 loss, 在正频半谱的幅度谱上:

            A_k      = |X_k|
            \hat A_k = |\hat X_k|

            L_log-mag = mean_{B, i, c, k} | log(A_k + eps) - log(\hat A_k + eps) |

        其中 i 是原子索引，c 是坐标分量(x,y,z), k 是正频索引。

        """
        X_pred_real_positive = pred_fourier_package["X_positive_real"]   # (B, N_atom, 3, seq_len)
        X_pred_imag_positive = pred_fourier_package["X_positive_imag"]   # (B, N_atom, 3, seq_len)
        X_true_real_positive = true_fourier_package["X_positive_real"]   # (N_atom, 3, seq_len)
        X_true_imag_positive = true_fourier_package["X_positive_imag"]   # (N_atom, 3, seq_len)

        if X_pred_real_positive.dim() == 3 and X_pred_imag_positive.dim() == 3:
            true_real_b = X_true_real_positive  # (1, N_atom, 3, seq_len_chunk)
            true_imag_b = X_true_imag_positive  # (1, N_atom, 3, seq_len_chunk)
        else:
            true_real_b = X_true_real_positive.unsqueeze(0)  # (1, N_atom, 3, seq_len_chunk)
            true_imag_b = X_true_imag_positive.unsqueeze(0)  # (1, N_atom, 3, seq_len_chunk)
        
        # 复谱的「点对点匹配」
        # 幅度谱的「形状 + 相对能量分布」匹配
        A_true = _amplitude_from_real_imag(true_real_b, true_imag_b, eps_amp=self.eps) # (1, N_atom, 3, seq_len)
        A_pred = _amplitude_from_real_imag(X_pred_real_positive, X_pred_imag_positive, eps_amp=self.eps) # (B, N_atom, 3, seq_len)

        loss_fns = {}

        # 定义每个loss为lambda函数，返回loss tensor (以及可选metrics字典)
        loss_fns.update({
            "spectrum_mse_loss": lambda: self._compute_relative_mse_mae_loss(
                X_pred_real_positive, X_pred_imag_positive, true_real_b, true_imag_b
            )[0],
            "spectrum_mae_loss": lambda: self._compute_relative_mse_mae_loss(
                X_pred_real_positive, X_pred_imag_positive, true_real_b, true_imag_b
            )[1],
            "smooth_l1_loss": lambda: self._compute_smooth_l1_loss(
                X_pred_real_positive, X_pred_imag_positive, true_real_b, true_imag_b
            ),
            "amplitude_mse_loss": lambda: F.mse_loss(A_pred, A_true),
            "log_spectral_l1_loss": lambda: self._compute_log_spectral_l1_loss(
                A_pred, A_true
            ),
            "cosine_similarity_loss": lambda: self._compute_cosine_similarity_loss(
                A_pred, A_true,
                loss_weight=loss_weight
            ),
            "spectral_convergence_loss": lambda: self._compute_spectral_convergence_loss(
                A_pred, A_true
            ),
        })

        cum_loss, metrics = self.aggregate_losses(loss_fns)
        return cum_loss, metrics

class SpectrumChunkLoss(SpectrumLoss):
    def forward(
        self,
        X_pred_real_positive: torch.Tensor,   # (B, N_atom, 3, seq_len_chunk) or (N_atom, 3, seq_len_chunk)
        X_pred_imag_positive: torch.Tensor,   # (B, N_atom, 3, seq_len_chunk) or (N_atom, 3, seq_len_chunk)
        X_true_real_positive: torch.Tensor,   # (N_atom, 3, seq_len_chunk)
        X_true_imag_positive: torch.Tensor,   # (N_atom, 3, seq_len_chunk)
        loss_weight: Optional[torch.Tensor] = None, # (seq_len_chunk, )
    ):
        if X_pred_real_positive.dim() == 3 and X_pred_imag_positive.dim() == 3:
            true_real_b = X_true_real_positive  # (1, N_atom, 3, seq_len_chunk)
            true_imag_b = X_true_imag_positive  # (1, N_atom, 3, seq_len_chunk)
        else:
            true_real_b = X_true_real_positive.unsqueeze(0)  # (1, N_atom, 3, seq_len_chunk)
            true_imag_b = X_true_imag_positive.unsqueeze(0)  # (1, N_atom, 3, seq_len_chunk)

        # 复谱的「点对点匹配」& 幅度谱的「形状 + 相对能量分布」匹配
        A_true = _amplitude_from_real_imag(true_real_b, true_imag_b, eps_amp=self.eps) # (1, N_atom, 3, seq_len_chunk)
        A_pred = _amplitude_from_real_imag(X_pred_real_positive, X_pred_imag_positive, eps_amp=self.eps) # (B, N_atom, 3, seq_len_chunk)

        loss_fns = {}

        # 定义每个loss为lambda函数，返回loss tensor (以及可选metrics字典)
        loss_fns.update({
            "spectrum_mse_loss": lambda: self._compute_relative_mse_mae_loss(
                X_pred_real_positive, X_pred_imag_positive, true_real_b, true_imag_b
            )[0],
            "spectrum_mae_loss": lambda: self._compute_relative_mse_mae_loss(
                X_pred_real_positive, X_pred_imag_positive, true_real_b, true_imag_b
            )[1],
            "smooth_l1_loss": lambda: self._compute_smooth_l1_loss(
                X_pred_real_positive, X_pred_imag_positive, true_real_b, true_imag_b
            ),
            "amplitude_mse_loss": lambda: F.mse_loss(A_pred, A_true),
            "log_spectral_l1_loss": lambda: self._compute_log_spectral_l1_loss(
                A_pred, A_true
            ),
            "cosine_similarity_loss": lambda: self._compute_cosine_similarity_loss(
                A_pred, A_true,
                loss_weight=loss_weight
            ),
            "spectral_convergence_loss": lambda: self._compute_spectral_convergence_loss(
                A_pred, A_true
            ),
        })

        cum_loss, metrics = self.aggregate_losses(loss_fns)
        return cum_loss, metrics

    def merge_losses_chunk(
        self, cum_losses_and_metrics: list[tuple[torch.Tensor, dict]]
    ) -> tuple[torch.Tensor, dict]:
        """
        Merges cum_loss and metrics from multiple chunks (e.g., from aggregate_losses for each chunk).

        Args:
            cum_losses_and_metrics (list): List of (cum_loss, metrics) for each chunk.
        Returns:
            Tuple[torch.Tensor, dict]: (aggregated_cum_loss, aggregated_all_metrics)
        """
        if len(cum_losses_and_metrics) == 0:
            raise ValueError("cum_losses_and_metrics list is empty.")

        total_cum_loss = 0.0
        merged_metrics = {}

        # Sum all cum_loss
        for cum_loss, chunk_metrics in cum_losses_and_metrics:
            total_cum_loss = total_cum_loss + cum_loss

            for key, value in chunk_metrics.items():
                if key not in merged_metrics:
                    merged_metrics[key] = []
                merged_metrics[key].append(value)

        # Post-process: stack and average or sum as appropriate
        for key, values in merged_metrics.items():
            try:
                values_tensor = torch.stack([v.detach() for v in values])
                # For "loss" and each loss_name, average over chunks
                if (
                    key == "loss"
                    or "/" not in key  # for "mse_loss", etc.
                    or key.startswith("weighted_")
                ):
                    merged_metrics[key] = values_tensor.mean()
                else:
                    merged_metrics[key] = values_tensor.mean()
            except Exception:
                # If cannot stack (shouldn't happen), leave as is
                merged_metrics[key] = values

        merged_metrics["loss"] = total_cum_loss.detach()
        return total_cum_loss, merged_metrics