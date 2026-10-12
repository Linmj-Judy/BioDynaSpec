"""
ProteinSpectrumFAR Inference Script
类化推理器,采用FARTrainer的代码组织风格
"""

import os
import logging
import time

import torch
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from dataclasses import dataclass
from typing import Optional, Dict, List
import argparse
import contextlib

from data.fourier.utils import (
    PreprocessStateStruct,
    TrajectoryFFTConfig,
    TrajectoryFFTProcessor,
    reconstruct_full_spectrum_from_positive_half_real_imag,
)
from visualize.utils import plot_far_results
from protein_spectrum_far import (
    protein_spectrum_far_base,
    protein_spectrum_far_large,
    protein_spectrum_far_huge,
    sample_multiple,
)


def set_seed(seed: int):
    import random, numpy as np, torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class FARInferencer:
    """FAR推理器主类"""
    
    def __init__(self, config):
        self.config = config
        self.init_env()
        self.init_basics()
        self.init_data()
        self.init_model()
        
    def init_env(self):
        """初始化环境"""
        logging.info("初始化推理环境...")
        
        # 配置精度
        self.config_precision()
        
        # 检测设备
        self.use_cuda = torch.cuda.is_available() and self.config.device == 'cuda'
        if self.use_cuda:
            self.device = torch.device('cuda')
            logging.info(f"检测到GPU设备,使用CUDA")
        else:
            self.device = torch.device('cpu')
            logging.info("使用CPU推理")
    
    def init_basics(self):
        """初始化基础配置"""
        logging.info("初始化基础配置...")
        
        # 创建输出目录
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        if self.config.save_pdb:
            os.makedirs(self.config.pdb_output_dir, exist_ok=True)
        
        logging.info(f"输出目录: {self.config.output_dir}")
        if self.config.save_pdb:
            logging.info(f"PDB输出目录: {self.config.pdb_output_dir}")

        self.fft_processor = TrajectoryFFTProcessor(
            TrajectoryFFTConfig(
                norm=self.config.fft_norm,  # fft_window5000的时候统一使用 backward (1/N) 以匹配默认行为
                use_double=False,
                window=None,
                use_sliding_window=False,
                store_dtype=torch.float32,
                store_complex_as_float2=True,
                device=self.device.type,
            )
        )
        
        # 检查必要文件
        if not os.path.exists(self.config.checkpoint_path):
            raise FileNotFoundError(f"检查点文件不存在: {self.config.checkpoint_path}")
        if not os.path.exists(self.config.data_path):
            raise FileNotFoundError(f"数据文件不存在: {self.config.data_path}")
    
    def init_data(self):
        """初始化数据加载器"""
        logging.info("初始化数据加载器...")

        logging.info(f"加载轨迹数据: {self.config.data_path}")
        self.data_list = torch.load(self.config.data_path, weights_only=False)
        logging.info(f"已加载 {len(self.data_list)} 个蛋白质样本")

            # 自动检测数据模式
        self.data_mode = self._detect_data_mode(self.data_list, self.config.data_mode)
        logging.info(f"推理数据模式: {self.data_mode}")

        # 加载对应数据集的 scaling factors（用于频谱逆归一化）
        self.scaling_factors = self._load_scaling_factors()
        if self.scaling_factors is not None:
            logging.info(
                f"Scaling factors loaded: shape={tuple(self.scaling_factors.shape)}"
            )
            logging.info(f"Scaling factor sample (first 5): {self.scaling_factors.flatten()[:5]}")
        else:
            logging.warning("CRITICAL: 未加载 scaling factors! 如果数据是归一化的，重建结果将完全错误（极其微小）。")

        # 生成蛋白质ID映射
        self.protein_ids = [item.get("id", f"sample_{i}") for i, item in enumerate(self.data_list)]
        self.trajectory_data_dict = {item.get("id", f"sample_{i}"): item for i, item in enumerate(self.data_list)}

        # 加载对应数据集的 scaling factors（用于频谱逆归一化）
        # self.scaling_factors = self._load_scaling_factors()
        # if self.scaling_factors is not None:
        #     logging.info(
        #         f"Scaling factors loaded: shape={tuple(self.scaling_factors.shape)}"
        #     )
        # else:
        #     logging.info("未加载 scaling factors，频谱将按原尺度重建")
    
    def init_model(self):
        """初始化模型并加载权重"""
        logging.info("初始化模型并加载权重...")

        logging.info(f"Loading Checkpoint: {self.config.checkpoint_path}")
        checkpoint = torch.load(self.config.checkpoint_path, map_location=self.device)

        # 从 checkpoint 中获取模型配置
        model_config = None
        if "hyper_parameters" in checkpoint and "model_config" in checkpoint["hyper_parameters"]:
            model_config = checkpoint["hyper_parameters"]["model_config"]
            logging.info("从检查点中找到模型配置:")
            for key, value in model_config.items():
                logging.info(f"  {key}: {value}")
        else:
            raise ValueError("检查点中未找到模型配置 (hyper_parameters.model_config)")

        if self.data_mode == "fft_window":
            model_config = dict(model_config)
            model_config["enable_cross_window"] = False

        if self.config.model_size == "large":
            self.model = protein_spectrum_far_large(**model_config)
        elif self.config.model_size == "huge":
            self.model = protein_spectrum_far_huge(**model_config)
        else:
            self.model = protein_spectrum_far_base(**model_config)

        state_dict = checkpoint.get("state_dict", checkpoint)
        if any(k.startswith("model.") for k in state_dict.keys()):
            stripped = {}
            for key, value in state_dict.items():
                if key.startswith("model."):
                    stripped[key[len("model."):]] = value
                else:
                    stripped[key] = value
            state_dict = stripped

        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if missing:
            logging.warning(f"Missing keys when loading checkpoint: {missing}")
        if unexpected:
            logging.warning(f"Unexpected keys when loading checkpoint: {unexpected}")

        self.model = self.model.to(self.device)
        self.model.eval()

        num_params = sum(p.numel() for p in self.model.parameters())
        logging.info("Model loaded successfully!")
        logging.info(f"Model Parameters: {num_params:,}")

        ckpt_precision = None
        hp = checkpoint.get("hyper_parameters", {})
        if isinstance(hp, dict):
            if isinstance(hp.get("config"), dict):
                ckpt_precision = hp["config"].get("precision")
            if ckpt_precision is None:
                ckpt_precision = hp.get("precision")
        self.precision = self.config.precision or ckpt_precision or "fp32"
        self.use_autocast = (
            self.device.type == "cuda"
            and str(self.precision).lower() in ("bf16", "bfloat16")
        )
        logging.info(f"推理精度: {self.precision} (autocast={self.use_autocast})")
    
    def config_precision(self):
        # 使用支持 Tensor Cores 的 matmul 精度，以提升 A800 上的性能
        if torch.cuda.is_available():
            try:
                torch.set_float32_matmul_precision('high')  # 可改为 'medium' 以更激进提速
                # 进一步开启 TF32（对大部分 FP32 计算生效）
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            except Exception:
                pass

    def _load_scaling_factors(self) -> Optional[torch.Tensor]:
        import glob

        path = self.config.scaling_factor_path
        if not path:
            data_dir = os.path.dirname(self.config.data_path)
            # 搜索所有以 _scaling_factors.pth 结尾的文件
            candidates = glob.glob(os.path.join(data_dir, "*_scaling_factors.pth"))
            path = candidates[0] if candidates else None
        if path and os.path.exists(path):
            return torch.tensor(torch.load(path, weights_only=False), dtype=torch.float32)
        # if "SCALING_FACTOR" in globals() and isinstance(SCALING_FACTOR, torch.Tensor):
        #     return SCALING_FACTOR.to(torch.float32)
        return None

    def _expand_scaling(self, length: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.scaling_factors is None:
            raise ValueError("scaling_factors is None")
        scale = self.scaling_factors.to(device=device, dtype=dtype)
        if scale.numel() >= length:
            return scale[:length]
        pad = scale[-1].repeat(length - scale.numel())
        return torch.cat([scale, pad], dim=0)

    # 这里之所以不能直接 spectrum * scale，是因为 spectrum[..., 6] 的最后一维是 [x_real, x_imag, y_real, y_imag, z_real, z_imag]，
    # 而 scale 只有 (n_freqs,)（频率长度），主要作用是对不同频率（dim=-2的每个频点）缩放。直接 pointwise 相乘会导致广播不一致：
    # - spectrum[..., 6]，比如 (N, F, 6)
    # - scale.shape=(F,)，直接乘会沿最后一维 6 重复同一个 scale，而我们实际希望 x/y/z 的 real/imag 频谱每组都独立按 freq 缩放
    # 所以需要先把 spectrum 拆成 3 组 (real, imag)，分别对每组 (F,) 上的 real 和 imag 乘以 (F,) scale，再堆回去保证通道顺序。
    def _denormalize_real_imag(
        self,
        real: torch.Tensor,
        imag: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.scaling_factors is None:
            return real, imag
        if real.shape != imag.shape:
            raise ValueError("real/imag shape mismatch for denormalization")
        freq_len = real.shape[-1]
        scale = self._expand_scaling(freq_len, real.device, real.dtype).view(
            *([1] * (real.ndim - 1)), freq_len
        )
        return real * scale, imag * scale

    def _denormalize_spectrum(self, spectrum: torch.Tensor) -> torch.Tensor:
        if self.scaling_factors is None:
            return spectrum
        spec = spectrum
        if spec.dim() == 4 and spec.shape[-1] == 1:
            spec = spec.squeeze(-1)
        if spec.dim() != 3 or spec.shape[-1] != 6:
            raise ValueError(f"Unexpected spectrum shape for denorm: {spec.shape}")
        # 不能直接 spec * scale，否则每个 6 通道都用同一个 scale，实际上每组 (real, imag) 都需要对 freq 方向单独缩放
        # 所以要按 freq 拆开分别缩放后再组装确保语义正确
        real_imag = spec.split(1, dim=-1)
        real_x, imag_x = real_imag[0].squeeze(-1), real_imag[1].squeeze(-1)
        real_y, imag_y = real_imag[2].squeeze(-1), real_imag[3].squeeze(-1)
        real_z, imag_z = real_imag[4].squeeze(-1), real_imag[5].squeeze(-1)
        real_x, imag_x = self._denormalize_real_imag(real_x, imag_x)
        real_y, imag_y = self._denormalize_real_imag(real_y, imag_y)
        real_z, imag_z = self._denormalize_real_imag(real_z, imag_z)
        out = torch.stack(
            [real_x, imag_x, real_y, imag_y, real_z, imag_z], dim=-1
        )
        return out

    @staticmethod
    def _detect_data_mode(data_list: List[Dict], mode: str) -> str:
        if mode in ("fft", "fft_window"):
            return mode
        for item in data_list:
            positions = item.get("positions", {})
            if positions.get("x_fft_windows") is not None:
                return "fft_window"
        return "fft"

    @staticmethod
    def _build_fft_spectrum(positions: Dict) -> torch.Tensor:
        spectrum_x_real = [item[:, 0] for item in positions["x"]]
        spectrum_x_imag = [item[:, 1] for item in positions["x"]]
        spectrum_y_real = [item[:, 0] for item in positions["y"]]
        spectrum_y_imag = [item[:, 1] for item in positions["y"]]
        spectrum_z_real = [item[:, 0] for item in positions["z"]]
        spectrum_z_imag = [item[:, 1] for item in positions["z"]]
        spectrum = np.stack(
            [
                spectrum_x_real,
                spectrum_x_imag,
                spectrum_y_real,
                spectrum_y_imag,
                spectrum_z_real,
                spectrum_z_imag,
            ],
            axis=-1,
        )  # [N, F, 6]
        return torch.tensor(spectrum, dtype=torch.float32).unsqueeze(-1)

    @staticmethod
    def _build_fft_window_spectrum(positions: Dict, window_idx: int) -> torch.Tensor:
        x_windows = positions["x_fft_windows"][window_idx]
        y_windows = positions["y_fft_windows"][window_idx]
        z_windows = positions["z_fft_windows"][window_idx]
        spectrum_x_real = [item[:, 0] for item in x_windows]
        spectrum_x_imag = [item[:, 1] for item in x_windows]
        spectrum_y_real = [item[:, 0] for item in y_windows]
        spectrum_y_imag = [item[:, 1] for item in y_windows]
        spectrum_z_real = [item[:, 0] for item in z_windows]
        spectrum_z_imag = [item[:, 1] for item in z_windows]
        spectrum = np.stack(
            [
                spectrum_x_real,
                spectrum_x_imag,
                spectrum_y_real,
                spectrum_y_imag,
                spectrum_z_real,
                spectrum_z_imag,
            ],
            axis=-1,
        )  # [N, F, 6]
        return torch.tensor(spectrum, dtype=torch.float32).unsqueeze(-1)

    def _autocast_ctx(self):
        if self.use_autocast:
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    @contextlib.contextmanager
    def _no_mask_ctx(self):
        if hasattr(self.model, "mask"):
            original = self.model.mask
            self.model.mask = False
            try:
                yield
            finally:
                self.model.mask = original
        else:
            yield

    def _unpack_batch(self, batch):
        if len(batch) == 7:
            coords, spectrum, rotation_matrices, aatype, aa_mask, esm_features, _ = batch
            last_spectrum = None
            window_size = None
            hop_ratio = None
        elif len(batch) == 10:
            coords, spectrum, rotation_matrices, aatype, aa_mask, esm_features, _, last_spectrum, window_size, hop_ratio = batch
        else:
            raise ValueError(f"Unexpected batch length: {len(batch)}")
        return coords, spectrum, rotation_matrices, aatype, aa_mask, esm_features, last_spectrum, window_size, hop_ratio

    def _predict_spectrum(self, batch) -> torch.Tensor:
        with torch.no_grad():
            coords, spectrum, rotation_matrices, aatype, aa_mask, esm_features, last_spectrum, window_size, hop_ratio = self._unpack_batch(batch)
            with self._no_mask_ctx():
                with self._autocast_ctx():
                    pred, _ = self.model(
                        spectrum,
                        coords,
                        rotation_matrices,
                        aatype,
                        aa_mask,
                        esm_features,
                        None,
                        loss_weight=False,
                        return_pred_spectrum=True,
                        last_spectrum=last_spectrum,
                        window_size=window_size,
                        hop_ratio=hop_ratio,
                    )
        if pred.dim() == 5:
            pred = pred[0]
        return pred

    def _spectrum_to_full_fft_coords(
        self,
        spectrum: torch.Tensor,
        frame0: torch.Tensor,
        window_size,
        num_frames: int,
        preprocess_state: Optional[Dict] = None,
        skip_invert_preprocessing: bool = False,
    ) -> torch.Tensor:
        """
        从频谱重建坐标
        
        Args:
            spectrum: 频谱张量
            frame0: 初始帧坐标
            num_frames: 帧数
            preprocess_state: 预处理状态（用于invert_preprocessing）
            skip_invert_preprocessing: 是否跳过invert_preprocessing（用于与coords_preprocessed比较）
        """
        spectrum = self._denormalize_spectrum(spectrum).squeeze(-1)
        spec_x = torch.complex(spectrum[..., 0], spectrum[..., 1])
        spec_y = torch.complex(spectrum[..., 2], spectrum[..., 3])
        spec_z = torch.complex(spectrum[..., 4], spectrum[..., 5])
        # irfft_full 的实现: torch.fft.irfft(spectrum, n=num_frames, dim=-1, norm=norm)
        # 注意：对于 fft_window 模式，num_frames 就是 window_size
        disp_x = torch.fft.irfft(spec_x, n=window_size, dim=-1, norm=self.config.fft_norm)
        disp_y = torch.fft.irfft(spec_y, n=window_size, dim=-1, norm=self.config.fft_norm)
        disp_z = torch.fft.irfft(spec_z, n=window_size, dim=-1, norm=self.config.fft_norm)

        # spec_x_real = spectrum[..., 0]
        # spec_x_imag = spectrum[..., 1]
        # spec_y_real = spectrum[..., 2]
        # spec_y_imag = spectrum[..., 3]
        # spec_z_real = spectrum[..., 4]
        # spec_z_imag = spectrum[..., 5]
        # full_x = reconstruct_full_spectrum_from_positive_half_real_imag(spec_x_real, spec_x_imag)
        # full_y = reconstruct_full_spectrum_from_positive_half_real_imag(spec_y_real, spec_y_imag)
        # full_z = reconstruct_full_spectrum_from_positive_half_real_imag(spec_z_real, spec_z_imag)
        # disp_x = torch.fft.ifft(full_x, dim=-1, norm=self.config.fft_norm).real
        # disp_y = torch.fft.ifft(full_y, dim=-1, norm=self.config.fft_norm).real
        # disp_z = torch.fft.ifft(full_z, dim=-1, norm=self.config.fft_norm).real
        coords = torch.stack([disp_x, disp_y, disp_z], dim=1)[..., :num_frames]
        coords = coords + frame0.unsqueeze(-1)
        if skip_invert_preprocessing:
            return coords
        return self._invert_preprocessing_if_available(coords, preprocess_state)

    def _invert_preprocessing_if_available(
        self,
        coords: torch.Tensor,
        preprocess_state: Optional[Dict],
    ) -> torch.Tensor:
        if preprocess_state is None:
            return coords
        state = preprocess_state
        if isinstance(preprocess_state, dict):
            state = PreprocessStateStruct.from_dict(preprocess_state)
        return self.fft_processor.invert_preprocessing(coords, state, fold_back_to_box=False)

    def _warn_if_static(
        self,
        protein_id: str,
        reconstructed_coords: torch.Tensor,
        pred_spectrum: Optional[torch.Tensor] = None,
        gt_spectrum: Optional[torch.Tensor] = None,
    ):
        if reconstructed_coords.ndim != 3:
            return
        disp = reconstructed_coords - reconstructed_coords[..., :1]
        disp_mean = float(disp.abs().mean().item())
        print(f"{protein_id}: pred_spectrum", pred_spectrum)
        print(f"{protein_id}: gt_spectrum", gt_spectrum)
        if disp_mean < 1e-6:
            pred_norm = float(pred_spectrum.abs().mean().item()) if pred_spectrum is not None else None
            gt_norm = float(gt_spectrum.abs().mean().item()) if gt_spectrum is not None else None
            logging.warning(
                f"{protein_id}: 重建轨迹几乎静止(平均位移={disp_mean:.2e})，"
                f"pred_spectrum|mean|={pred_norm}, gt_spectrum|mean|={gt_norm}。"
                "可能是频谱幅值接近0或频谱与重建流程不匹配。"
            )

    # def _cola_merge(self, window_coords: List[torch.Tensor], window_size: int, hop_ratio: float) -> torch.Tensor:
    #     hop_size = int(window_size * hop_ratio)
    #     total_frames = window_size + (len(window_coords) - 1) * hop_size
    #     device = window_coords[0].device
    #     dtype = window_coords[0].dtype
    #     num_residues = window_coords[0].shape[0]

    #     numerator = torch.zeros(num_residues, 3, total_frames, device=device, dtype=dtype)
    #     denom = torch.zeros(total_frames, device=device, dtype=dtype)
    #     window = torch.hann_window(window_size, periodic=DEFAULT_HANN_PERIODIC, device=device, dtype=dtype)

    #     for w, coords in enumerate(window_coords):
    #         start = w * hop_size
    #         end = start + window_size
    #         weighted = coords * window.view(1, 1, -1)
    #         numerator[:, :, start:end] += weighted
    #         denom[start:end] += window

    #     denom = denom.clamp_min(1e-8)
    #     return numerator / denom.view(1, 1, -1)


    def compute_reconstruction_metrics(self, original_coords, reconstructed_coords):
        """
        计算坐标重建质量指标
        Args:
            original_coords: [num_frames, num_residues, 3] 原始坐标
            reconstructed_coords: [num_residues, 3, num_frames] 重建坐标
        Returns:
            metrics: 评估指标字典
        """
        # 将重建坐标转换为与原始坐标相同的维度顺序
        reconstructed_coords = reconstructed_coords.permute(2, 0, 1).contiguous()
        reconstructed_coords = self._kabsch_align_batch(
            original_coords, reconstructed_coords
        )
        
        # 计算整体位移RMSE
        diff = original_coords - reconstructed_coords
        displacement_squared = torch.sum(diff ** 2, dim=-1)
        overall_rmse = torch.sqrt(torch.mean(displacement_squared))
        
        # 计算每个残基的RMSE
        rmsd_per_residue = torch.sqrt(torch.mean(displacement_squared, dim=0))
        
        # 计算每帧的RMSE
        rmsd_per_frame = torch.sqrt(torch.mean(displacement_squared, dim=1))
        
        # 计算MAE
        mae = torch.mean(torch.abs(original_coords.cpu() - reconstructed_coords.cpu()))
        
        # 计算相关系数
        original_flat = original_coords.view(-1)
        reconstructed_flat = reconstructed_coords.view(-1)
        correlation = torch.corrcoef(
            torch.stack([original_flat.cpu(), reconstructed_flat.cpu()])
        )[0, 1]
        
        return {
            'overall_rmse': overall_rmse.item(),
            'mae': mae.item(),
            'correlation': correlation.item(),
            'rmse_per_frame': rmsd_per_frame.cpu().numpy(),
            'rmse_per_residue': rmsd_per_residue.cpu().numpy()
        }

    @staticmethod
    def _kabsch_align_batch(target_coords: torch.Tensor, source_coords: torch.Tensor) -> torch.Tensor:
        """
        对 source_coords 做 Kabsch 对齐到 target_coords（逐帧）。
        输入形状: (T, N, 3)，输出同形状（已对齐）。
        """
        if target_coords.ndim != 3 or source_coords.ndim != 3:
            raise ValueError("Kabsch对齐要求输入形状为 (T, N, 3)")
        if target_coords.shape != source_coords.shape:
            raise ValueError("Kabsch对齐要求 target/source 形状一致")

        target_center = target_coords.mean(dim=1, keepdim=True)
        source_center = source_coords.mean(dim=1, keepdim=True)
        Y = target_coords - target_center
        X = source_coords - source_center

        # H = X^T Y, shape (T, 3, 3)
        H = torch.matmul(X.transpose(1, 2), Y)
        U, _, Vh = torch.linalg.svd(H)
        V = Vh.transpose(-2, -1)

        # 处理反射
        det = torch.det(torch.matmul(V, U.transpose(-2, -1)))
        eye = torch.eye(3, device=H.device, dtype=H.dtype).expand(H.shape[0], 3, 3).clone()
        eye[:, -1, -1] = torch.where(
            det < 0,
            torch.tensor(-1.0, device=H.device, dtype=H.dtype),
            torch.tensor(1.0, device=H.device, dtype=H.dtype),
        )
        R = torch.matmul(V, torch.matmul(eye, U.transpose(-2, -1)))

        X_aligned = torch.matmul(X, R.transpose(-2, -1))
        return X_aligned + target_center
    
    # def evaluate_reconstruction(self):
    #     if self.data_mode == "stft":
    #         return self._evaluate_stft()
    #     if self.data_mode == "fft_window":
    #         return self._evaluate_fft_window()
    #     return self._evaluate_fft()

    # def _evaluate_fft(self):
    #     results = {
    #         "far_rmse": [],
    #         "far_mae": [],
    #         "far_correlation": [],
    #         "per_protein_metrics": defaultdict(list),
    #         "spectral_mse": [],  # <--- [新增] 初始化列表
    #         "timing": {
    #             "total_times": [],
    #             "far_sampling_times": [],
    #             "trajectory_reconstruction_times": [],
    #             "metrics_computation_times": [],
    #         },
    #         "diversity_metrics": {
    #             "sample_std": [],
    #             "sample_min_rmse": [],
    #             "sample_max_rmse": [],
    #             "ensemble_rmse": [],
    #         } if self.config.num_samples > 1 else None,
    #     }

    #     logging.info("开始FFT模式重建评估...")
    #     for idx, protein_id in enumerate(tqdm(self.protein_ids, desc="评估FFT重建")):
    #         batch_start_time = time.time()
    #         protein_data = self.trajectory_data_dict[protein_id]
    #         positions = protein_data["positions"]
    #         preprocess_state = positions.get("preprocess_state")

    #         coords = torch.tensor(
    #             np.stack([positions["x0"], positions["y0"], positions["z0"]], axis=-1),
    #             dtype=torch.float32,
    #         ).to(self.device)
    #         spectrum = self._build_fft_spectrum(positions).to(self.device)
    #         rotation_matrices = protein_data["rotation_matrices0"].clone().detach().to(self.device)
    #         aatype = protein_data["amino_acid_types"].clone().detach().to(self.device)
    #         aa_mask = torch.ones(aatype.shape, dtype=torch.bool, device=self.device)
    #         esm_features = torch.tensor(protein_data["esm"], device=self.device)

    #         pred_spectrum = None
    #         all_samples = None
    #         sampling_start_time = time.time()
    #         if self.config.num_samples > 1:
    #             with self._no_mask_ctx():
    #                 with self._autocast_ctx():
    #                     all_samples = sample_multiple(
    #                         model=self.model,
    #                         coords=coords,
    #                         rotation_matrices=rotation_matrices,
    #                         aatype=aatype,
    #                         aa_mask=aa_mask,
    #                         esm_features=esm_features,
    #                         bihito_token_repr=None,
    #                         n_samples=self.config.num_samples,
    #                         num_iter=spectrum.shape[1],
    #                         cfg=self.config.cfg,
    #                         temperature=self.config.temperature,
    #                         progress=False,
    #                         sample_method="nomask",
    #                         seed=self.config.seed + idx,
    #                     )
    #             pred_spectrum = all_samples.mean(dim=0)
    #         else:
    #             pred_spectrum = self._predict_spectrum(
    #                 (coords, spectrum, rotation_matrices, aatype, aa_mask, esm_features, None)
    #             ).squeeze(-1)
    #         # <--- [新增] 在这里添加计算逻辑 (约 713 行)
    #         # spectrum 是 GT，形状通常是 [N, F, 6, 1]，pred_spectrum 是 [N, F, 6]
    #         # 需要把 GT 的最后一维 squeeze 掉以匹配形状
    #         gt_spec_squeezed = spectrum.squeeze(-1)
    #         mse_loss = torch.nn.functional.mse_loss(pred_spectrum, gt_spec_squeezed)
    #         results["spectral_mse"].append(mse_loss.item())

    #         sampling_end_time = time.time()
    #         results["timing"]["far_sampling_times"].append(
    #             sampling_end_time - sampling_start_time
    #         )

    #         trajectory_start_time = time.time()
    #         reconstructed_coords = self._spectrum_to_full_fft_coords(
    #             pred_spectrum, coords, self.config.num_frames, preprocess_state=preprocess_state
    #         )
    #         original_coords = self._spectrum_to_full_fft_coords(
    #             gt_spec_squeezed, coords, self.config.num_frames, preprocess_state=preprocess_state
    #         )

    #         # original_coords = self._reconstruct_full_fft_coords_from_positions_dict(
    #         #     positions, coords, self.config.num_frames, preprocess_state=preprocess_state
    #         # ).permute(2, 0, 1).contiguous()
    #         self._warn_if_static(
    #             protein_id,
    #             reconstructed_coords,
    #             pred_spectrum=pred_spectrum,
    #             gt_spectrum=spectrum,
    #         )
    #         trajectory_end_time = time.time()
    #         results["timing"]["trajectory_reconstruction_times"].append(
    #             trajectory_end_time - trajectory_start_time
    #         )

    #         metrics_start_time = time.time()
    #         coord_metrics = self.compute_reconstruction_metrics(original_coords, reconstructed_coords)
    #         if all_samples is not None:
    #             diversity_metrics = self.analyze_sample_diversity(
    #                 all_samples, original_coords, coords, self.config.num_frames, use_full_fft=True
    #             )
    #             results["diversity_metrics"]["sample_std"].append(diversity_metrics["coords_diversity_std"])
    #             results["diversity_metrics"]["sample_min_rmse"].append(diversity_metrics["sample_min_rmse"])
    #             results["diversity_metrics"]["sample_max_rmse"].append(diversity_metrics["sample_max_rmse"])
    #             results["diversity_metrics"]["ensemble_rmse"].append(diversity_metrics["ensemble_rmse"])
    #         metrics_end_time = time.time()
    #         results["timing"]["metrics_computation_times"].append(
    #             metrics_end_time - metrics_start_time
    #         )

    #         results["far_rmse"].append(coord_metrics["overall_rmse"])
    #         results["far_mae"].append(coord_metrics["mae"])
    #         results["far_correlation"].append(coord_metrics["correlation"])
    #         results["per_protein_metrics"]["rmse"].append(coord_metrics["overall_rmse"])
    #         results["per_protein_metrics"]["mae"].append(coord_metrics["mae"])
    #         results["per_protein_metrics"]["correlation"].append(coord_metrics["correlation"])
    #         results["per_protein_metrics"]["num_residues"].append(coords.shape[0])
    #         results["per_protein_metrics"]["num_frames"].append(self.config.num_frames)

    #         batch_end_time = time.time()
    #         results["timing"]["total_times"].append(batch_end_time - batch_start_time)

    #         logging.info(
    #             f"\nBatch {idx + 1} ({protein_id}): "
    #             f"RMSE={coord_metrics['overall_rmse']:.4f}Å, "
    #             f"MAE={coord_metrics['mae']:.4f}Å, "
    #             f"Corr={coord_metrics['correlation']:.4f}"
    #         )
    #     return results

    # def _evaluate_stft(self):
    #     results = {
    #         "far_rmse": [],
    #         "far_mae": [],
    #         "far_correlation": [],
    #         "per_protein_metrics": defaultdict(list),
    #         "spectral_mse": [],  # <--- [新增] 初始化列表
    #         "timing": {
    #             "total_times": [],
    #             "far_sampling_times": [],
    #             "trajectory_reconstruction_times": [],
    #             "metrics_computation_times": [],
    #         },
    #     }

    #     logging.info("开始STFT模式重建评估（含COLA后处理）...")
    #     if self.config.num_samples > 1:
    #         logging.warning("STFT模式暂不支持多样性采样指标，将忽略 num_samples>1 的多样性统计。")
    #     for idx, protein_id in enumerate(tqdm(self.protein_ids, desc="评估STFT重建")):
    #         batch_start_time = time.time()
    #         protein_data = self.trajectory_data_dict[protein_id]
    #         positions = protein_data["positions"]
    #         x_windows = positions.get("x_windows", [])
    #         if not x_windows:
    #             logging.warning(f"{protein_id}: 未找到 x_windows，跳过")
    #             continue
    #         preprocess_states = positions.get("preprocess_state_windows")

    #         window_size = int(positions.get("window_size"))
    #         hop_ratio = float(positions.get("hop_ratio"))
    #         num_windows = len(x_windows)

    #         rotation_matrices = protein_data["rotation_matrices0"].clone().detach().to(self.device)
    #         aatype = protein_data["amino_acid_types"].clone().detach().to(self.device)
    #         aa_mask = torch.ones(aatype.shape, dtype=torch.bool, device=self.device)
    #         esm_features = torch.tensor(protein_data["esm"], device=self.device)

    #         predicted_windows: List[torch.Tensor] = []
    #         sampling_start_time = time.time()
    #         last_pred = None
    #         for w in range(num_windows):
    #             frame0 = torch.tensor(
    #                 np.stack(
    #                     [
    #                         positions["x0_windows"][w],
    #                         positions["y0_windows"][w],
    #                         positions["z0_windows"][w],
    #                     ],
    #                     axis=-1,
    #                 ),
    #                 dtype=torch.float32,
    #             ).to(self.device)
    #             spectrum = self._build_window_spectrum(positions, w).to(self.device)

    #             batch = (
    #                 frame0,
    #                 spectrum,
    #                 rotation_matrices,
    #                 aatype,
    #                 aa_mask,
    #                 esm_features,
    #                 None,
    #                 last_pred,
    #                 window_size,
    #                 hop_ratio,
    #             )
    #             if w == 0:
    #                 pred_spectrum = spectrum
    #             else:
    #                 pred_spectrum = self._predict_spectrum(batch)
    #             # <--- [新增] 在这里添加计算逻辑 (约 841 行)
    #             # spectrum 此时就是当前窗口的 GT spectrum (已在 loop 开头获取)
    #             # 同样注意形状，如果 spectrum 有最后一维 1，需要 squeeze
    #             gt_spec_squeezed = spectrum.squeeze(-1) if spectrum.dim() > pred_spectrum.dim() else spectrum
    #             mse_loss = torch.nn.functional.mse_loss(pred_spectrum, gt_spec_squeezed)
    #             results["spectral_mse"].append(mse_loss.item())

    #             last_pred = pred_spectrum
    #             predicted_windows.append(pred_spectrum)
    #         sampling_end_time = time.time()
    #         results["timing"]["far_sampling_times"].append(
    #             sampling_end_time - sampling_start_time
    #         )

    #         trajectory_start_time = time.time()
    #         window_coords = [
    #             self._reconstruct_window_coords(
    #                 pred_spectrum,
    #                 torch.tensor(
    #                     np.stack(
    #                         [
    #                             positions["x0_windows"][w],
    #                             positions["y0_windows"][w],
    #                             positions["z0_windows"][w],
    #                         ],
    #                         axis=-1,
    #                     ),
    #                     dtype=torch.float32,
    #                     device=self.device,
    #                 ),
    #                 window_size,
    #                 preprocess_state=(
    #                     preprocess_states[w] if preprocess_states is not None and w < len(preprocess_states) else None
    #                 ),
    #             )
    #             for w, pred_spectrum in enumerate(predicted_windows)
    #         ]
    #         reconstructed_coords = self._cola_merge(window_coords, window_size, hop_ratio)

    #         gt_window_coords = []
    #         for w in range(num_windows):
    #             frame0 = torch.tensor(
    #                 np.stack(
    #                     [
    #                         positions["x0_windows"][w],
    #                         positions["y0_windows"][w],
    #                         positions["z0_windows"][w],
    #                     ],
    #                     axis=-1,
    #                 ),
    #                 dtype=torch.float32,
    #             ).to(self.device)
    #             gt_spectrum = self._build_window_spectrum(positions, w).to(self.device)
    #             gt_window_coords.append(
    #                 self._reconstruct_window_coords(
    #                     gt_spectrum,
    #                     frame0,
    #                     window_size,
    #                     preprocess_state=(
    #                         preprocess_states[w]
    #                         if preprocess_states is not None and w < len(preprocess_states)
    #                         else None
    #                     ),
    #                 )
    #             )
    #         original_coords = self._cola_merge(gt_window_coords, window_size, hop_ratio)
    #         original_coords = original_coords.permute(2, 0, 1).contiguous()
    #         self._warn_if_static(
    #             protein_id,
    #             reconstructed_coords,
    #             pred_spectrum=predicted_windows[0] if predicted_windows else None,
    #             gt_spectrum=self._build_window_spectrum(positions, 0).to(self.device),
    #         )

    #         trajectory_end_time = time.time()
    #         results["timing"]["trajectory_reconstruction_times"].append(
    #             trajectory_end_time - trajectory_start_time
    #         )

    #         metrics_start_time = time.time()
    #         coord_metrics = self.compute_reconstruction_metrics(original_coords, reconstructed_coords)
    #         metrics_end_time = time.time()
    #         results["timing"]["metrics_computation_times"].append(
    #             metrics_end_time - metrics_start_time
    #         )

    #         results["far_rmse"].append(coord_metrics["overall_rmse"])
    #         results["far_mae"].append(coord_metrics["mae"])
    #         results["far_correlation"].append(coord_metrics["correlation"])
    #         results["per_protein_metrics"]["rmse"].append(coord_metrics["overall_rmse"])
    #         results["per_protein_metrics"]["mae"].append(coord_metrics["mae"])
    #         results["per_protein_metrics"]["correlation"].append(coord_metrics["correlation"])
    #         results["per_protein_metrics"]["num_residues"].append(reconstructed_coords.shape[0])
    #         results["per_protein_metrics"]["num_frames"].append(reconstructed_coords.shape[-1])

    #         batch_end_time = time.time()
    #         results["timing"]["total_times"].append(batch_end_time - batch_start_time)

    #         logging.info(
    #             f"\nBatch {idx + 1} ({protein_id}): "
    #             f"RMSE={coord_metrics['overall_rmse']:.4f}Å, "
    #             f"MAE={coord_metrics['mae']:.4f}Å, "
    #             f"Corr={coord_metrics['correlation']:.4f}"
    #         )
    #     return results

    # def _evaluate_fft_window(self):
    #     results = {
    #         "far_rmse": [],
    #         "far_mae": [],
    #         "far_correlation": [],
    #         "per_protein_metrics": defaultdict(list),
    #         "spectral_mse": [],  # <--- [新增] 初始化列表
    #         "timing": {
    #             "total_times": [],
    #             "far_sampling_times": [],
    #             "trajectory_reconstruction_times": [],
    #             "metrics_computation_times": [],
    #         },
    #     }

    #     logging.info("开始FFT窗口模式重建评估（按窗口拼接）...")
    #     if self.config.num_samples > 1:
    #         logging.warning("FFT窗口模式暂不支持多样性采样指标，将忽略 num_samples>1 的多样性统计。")
    #     for idx, protein_id in enumerate(tqdm(self.protein_ids, desc="评估FFT窗口重建")):
    #         batch_start_time = time.time()
    #         protein_data = self.trajectory_data_dict[protein_id]
    #         positions = protein_data["positions"]
    #         x_windows = positions.get("x_fft_windows", [])
    #         if not x_windows:
    #             logging.warning(f"{protein_id}: 未找到 x_fft_windows，跳过")
    #             continue
    #         preprocess_states = positions.get("preprocess_state_fft_windows")

    #         window_size = int(positions.get("fft_window_size"))
    #         num_windows = len(x_windows)

    #         rotation_matrices = protein_data["rotation_matrices0"].clone().detach().to(self.device)
    #         aatype = protein_data["amino_acid_types"].clone().detach().to(self.device)
    #         aa_mask = torch.ones(aatype.shape, dtype=torch.bool, device=self.device)
    #         esm_features = torch.tensor(protein_data["esm"], device=self.device)

    #         predicted_windows: List[torch.Tensor] = []
    #         sampling_start_time = time.time()
    #         for w in range(num_windows):
    #             frame0 = torch.tensor(
    #                 np.stack(
    #                     [
    #                         positions["x0_fft_windows"][w],
    #                         positions["y0_fft_windows"][w],
    #                         positions["z0_fft_windows"][w],
    #                     ],
    #                     axis=-1,
    #                 ),
    #                 dtype=torch.float32,
    #             ).to(self.device)
    #             spectrum = self._build_fft_window_spectrum(positions, w).to(self.device)
    #             pred_spectrum = self._predict_spectrum(
    #                 (frame0, spectrum, rotation_matrices, aatype, aa_mask, esm_features, None)
    #             ).squeeze(-1)
    #             # <--- [新增] 计算spectral_mse
    #             gt_spec_squeezed = spectrum.squeeze(-1) if spectrum.dim() > pred_spectrum.dim() else spectrum
    #             mse_loss = torch.nn.functional.mse_loss(pred_spectrum, gt_spec_squeezed)
    #             results["spectral_mse"].append(mse_loss.item())
    #             preprocess_state = (
    #                 preprocess_states[w]
    #                 if preprocess_states is not None and w < len(preprocess_states)
    #                 else None
    #             )
    #             predicted_windows.append((pred_spectrum, frame0, preprocess_state))
    #         sampling_end_time = time.time()
    #         results["timing"]["far_sampling_times"].append(
    #             sampling_end_time - sampling_start_time
    #         )

    #         trajectory_start_time = time.time()
    #         window_coords = [
    #             self._spectrum_to_full_fft_coords(
    #                 pred_spectrum,
    #                 frame0,
    #                 window_size,
    #                 preprocess_state=preprocess_state,
    #             )
    #             for pred_spectrum, frame0, preprocess_state in predicted_windows
    #         ]
    #         reconstructed_coords = torch.cat(window_coords, dim=-1)

    #         gt_window_coords = []
    #         for w in range(num_windows):
    #             frame0 = torch.tensor(
    #                 np.stack(
    #                     [
    #                         positions["x0_fft_windows"][w],
    #                         positions["y0_fft_windows"][w],
    #                         positions["z0_fft_windows"][w],
    #                     ],
    #                     axis=-1,
    #                 ),
    #                 dtype=torch.float32,
    #             ).to(self.device)
    #             gt_spectrum = self._build_fft_window_spectrum(positions, w).to(self.device)
    #             preprocess_state = (
    #                 preprocess_states[w]
    #                 if preprocess_states is not None and w < len(preprocess_states)
    #                 else None
    #             )
    #             gt_window_coords.append(
    #                 self._spectrum_to_full_fft_coords(
    #                     gt_spectrum.squeeze(-1),
    #                     frame0,
    #                     window_size,
    #                     preprocess_state=preprocess_state,
    #                 )
    #             )
    #         original_coords = torch.cat(gt_window_coords, dim=-1).permute(2, 0, 1).contiguous()

    #         trajectory_end_time = time.time()
    #         results["timing"]["trajectory_reconstruction_times"].append(
    #             trajectory_end_time - trajectory_start_time
    #         )

    #         if reconstructed_coords.shape[-1] != original_coords.shape[0]:
    #             pred_f = predicted_windows[0][0].shape[1] if predicted_windows else None
    #             gt_f = positions["x_fft_windows"][0][0].shape[0] if positions.get("x_fft_windows") else None
    #             logging.warning(
    #                 f"{protein_id}: pred/gt 轨迹帧数不一致, "
    #                 f"pred={reconstructed_coords.shape[-1]}, gt={original_coords.shape[0]}, "
    #                 f"window_size={window_size}, num_windows={num_windows}, "
    #                 f"pred_F={pred_f}, gt_F={gt_f}。将按最短长度对齐。"
    #             )
    #             min_frames = min(reconstructed_coords.shape[-1], original_coords.shape[0])
    #             if min_frames <= 0:
    #                 logging.warning(f"{protein_id}: min_frames<=0，跳过该样本")
    #                 continue
    #             reconstructed_coords = reconstructed_coords[..., :min_frames]
    #             original_coords = original_coords[:min_frames]
    #         self._warn_if_static(
    #             protein_id,
    #             reconstructed_coords,
    #             pred_spectrum=predicted_windows[0][0] if predicted_windows else None,
    #             gt_spectrum=self._build_fft_window_spectrum(positions, 0).squeeze(-1).to(self.device),
    #         )

    #         metrics_start_time = time.time()
    #         coord_metrics = self.compute_reconstruction_metrics(original_coords, reconstructed_coords)
    #         metrics_end_time = time.time()
    #         results["timing"]["metrics_computation_times"].append(
    #             metrics_end_time - metrics_start_time
    #         )

    #         results["far_rmse"].append(coord_metrics["overall_rmse"])
    #         results["far_mae"].append(coord_metrics["mae"])
    #         results["far_correlation"].append(coord_metrics["correlation"])
    #         results["per_protein_metrics"]["rmse"].append(coord_metrics["overall_rmse"])
    #         results["per_protein_metrics"]["mae"].append(coord_metrics["mae"])
    #         results["per_protein_metrics"]["correlation"].append(coord_metrics["correlation"])
    #         results["per_protein_metrics"]["num_residues"].append(reconstructed_coords.shape[0])
    #         results["per_protein_metrics"]["num_frames"].append(reconstructed_coords.shape[-1])

    #         batch_end_time = time.time()
    #         results["timing"]["total_times"].append(batch_end_time - batch_start_time)

    #         logging.info(
    #             f"\nBatch {idx + 1} ({protein_id}): "
    #             f"RMSE={coord_metrics['overall_rmse']:.4f}Å, "
    #             f"MAE={coord_metrics['mae']:.4f}Å, "
    #             f"Corr={coord_metrics['correlation']:.4f}"
    #         )
    #     return results
    
    
    def compute_statistics(self, metrics_list):
        """计算统计量"""
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

    def analyze_sample_diversity(self, all_samples, original_coords, frame0_positions, window_size, num_frames, use_full_fft=False):
        """
        分析多个样本的多样性
        Args:
            all_samples: (num_samples, num_residues, seq_len, 6/3, spectrum_token_dim)
            original_coords: (num_frames, num_residues, 3)
            frame0_positions: (num_residues, 3)
            num_frames: 帧数
            use_full_fft: 是否按完整FFT重建（不做截断/缩放）
        """
        num_samples = all_samples.shape[0]
        all_coords = []
        for i in range(num_samples):
            sample_spectrum = all_samples[i].squeeze(-1)
            if not use_full_fft:
                raise ValueError("当前推理仅支持完整FFT重建，请设置 use_full_fft=True")
            coords = self._spectrum_to_full_fft_coords(
                sample_spectrum, frame0_positions, window_size, num_frames
            )
            all_coords.append(coords)

        all_coords = torch.stack(all_coords, dim=0)  # (num_samples, num_residues, 3, num_frames)

        sample_rmses = []
        for i in range(num_samples):
            coords = all_coords[i].permute(2, 0, 1).contiguous()  # (num_frames, num_residues, 3)
            diff = original_coords - coords
            rmse = torch.sqrt(torch.mean(diff ** 2))
            sample_rmses.append(rmse.item())

        ensemble_coords = all_coords.mean(dim=0).permute(2, 0, 1).contiguous()
        diff_ensemble = original_coords - ensemble_coords
        ensemble_rmse = torch.sqrt(torch.mean(diff_ensemble ** 2)).item()
        coords_std = all_coords.std(dim=0).mean().item()

        return {
            'sample_rmses': sample_rmses,
            'sample_min_rmse': min(sample_rmses),
            'sample_max_rmse': max(sample_rmses),
            'sample_mean_rmse': np.mean(sample_rmses),
            'sample_std_rmse': np.std(sample_rmses),
            'ensemble_rmse': ensemble_rmse,
            'coords_diversity_std': coords_std,
        }
    
    # def plot_results(self, results):
    #     """绘制FAR结果图表"""
    #     save_dir = self.config.output_dir
        
    #     # 创建图表
    #     fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
    #     # 1. FAR RMSE分布
    #     axes[0, 0].hist(results['far_rmse'], bins=30, alpha=0.7, color='blue')
    #     axes[0, 0].set_title('FAR重建RMSE分布')
    #     axes[0, 0].set_xlabel('RMSE (Å)')
    #     axes[0, 0].set_ylabel('频数')
    #     axes[0, 0].axvline(np.mean(results['far_rmse']), color='red', linestyle='--',
    #                       label=f'均值: {np.mean(results["far_rmse"]):.4f}Å')
    #     axes[0, 0].legend()
        
    #     # 2. FAR MAE分布
    #     axes[0, 1].hist(results['far_mae'], bins=30, alpha=0.7, color='green')
    #     axes[0, 1].set_title('FAR重建MAE分布')
    #     axes[0, 1].set_xlabel('MAE (Å)')
    #     axes[0, 1].set_ylabel('频数')
    #     axes[0, 1].axvline(np.mean(results['far_mae']), color='red', linestyle='--',
    #                       label=f'均值: {np.mean(results["far_mae"]):.4f}Å')
    #     axes[0, 1].legend()
        
    #     # 3. FAR相关系数分布
    #     axes[0, 2].hist(results['far_correlation'], bins=30, alpha=0.7, color='orange')
    #     axes[0, 2].set_title('FAR重建相关性分布')
    #     axes[0, 2].set_xlabel('相关系数')
    #     axes[0, 2].set_ylabel('频数')
    #     axes[0, 2].axvline(np.mean(results['far_correlation']), color='red', linestyle='--',
    #                       label=f'均值: {np.mean(results["far_correlation"]):.4f}')
    #     axes[0, 2].legend()
        
    #     # 4. 蛋白质大小vs重建RMSE散点图
    #     num_residues = results['per_protein_metrics']['num_residues']
    #     protein_rmse = results['per_protein_metrics']['rmse']
    #     axes[1, 0].scatter(num_residues, protein_rmse, alpha=0.6, color='purple')
    #     axes[1, 0].set_title('蛋白质大小 vs FAR重建RMSE')
    #     axes[1, 0].set_xlabel('残基数')
    #     axes[1, 0].set_ylabel('RMSE (Å)')
        
    #     # 5. 时间分布直方图
    #     axes[1, 1].hist(results['timing']['total_times'], bins=20, alpha=0.7, color='red')
    #     axes[1, 1].set_title('总处理时间分布')
    #     axes[1, 1].set_xlabel('时间 (s)')
    #     axes[1, 1].set_ylabel('频数')
    #     axes[1, 1].axvline(np.mean(results['timing']['total_times']), color='blue', 
    #                       linestyle='--',
    #                       label=f'均值: {np.mean(results["timing"]["total_times"]):.3f}s')
    #     axes[1, 1].legend()
        
    #     # 6. 各处理步骤时间分布饼图
    #     far_time = np.mean(results['timing']['far_sampling_times'])
    #     trajectory_time = np.mean(results['timing']['trajectory_reconstruction_times'])
    #     metrics_time = np.mean(results['timing']['metrics_computation_times'])
        
    #     times = [far_time, trajectory_time, metrics_time]
    #     labels = ['FAR采样', '轨迹重建', '指标计算']
    #     colors = ['#ff9999', '#99ff99', '#ffcc99']
        
    #     axes[1, 2].pie(times, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    #     axes[1, 2].set_title('各处理步骤时间分布')
        
    #     plt.tight_layout()
    #     save_path = os.path.join(save_dir, 'far_reconstruction_results.png')
    #     plt.savefig(save_path, dpi=600, bbox_inches='tight')
    #     plt.close()
        
    #     logging.info(f"FAR结果图表已保存至 {save_path}")
    
    def save_results(self, results):
        """保存FAR评估结果"""
        detailed_results = {
            'far_rmse_stats': self.compute_statistics(results['far_rmse']),
            'far_mae_stats': self.compute_statistics(results['far_mae']),
            'far_correlation_stats': self.compute_statistics(results['far_correlation']),
            'spectral_mse_stats': self.compute_statistics(results.get('spectral_mse', [])),
            'per_protein_stats': {
                'rmse_stats': self.compute_statistics(results['per_protein_metrics']['rmse']),
                'mae_stats': self.compute_statistics(results['per_protein_metrics']['mae']),
                'correlation_stats': self.compute_statistics(results['per_protein_metrics']['correlation']),
                'num_residues_stats': self.compute_statistics(results['per_protein_metrics']['num_residues']),
                'num_frames_stats': self.compute_statistics(results['per_protein_metrics']['num_frames'])
            },
            'timing_stats': {
                'total_time_stats': self.compute_statistics(results['timing']['total_times']),
                'trajectory_reconstruction_time_stats': self.compute_statistics(results['timing']['trajectory_reconstruction_times']),
                'metrics_computation_time_stats': self.compute_statistics(results['timing']['metrics_computation_times'])
            }
        }

        if results.get('diversity_metrics') is not None:
            detailed_results['diversity_metrics_stats'] = {
                'sample_std_stats': self.compute_statistics(results['diversity_metrics']['sample_std']),
                'sample_min_rmse_stats': self.compute_statistics(results['diversity_metrics']['sample_min_rmse']),
                'sample_max_rmse_stats': self.compute_statistics(results['diversity_metrics']['sample_max_rmse']),
                'ensemble_rmse_stats': self.compute_statistics(results['diversity_metrics']['ensemble_rmse']),
            }
        
        save_path = os.path.join(self.config.output_dir, 'far_detailed_results.pth')
        torch.save(detailed_results, save_path)
        logging.info(f"FAR详细结果已保存至 {save_path}")
    
    def print_summary(self, results):
        """打印FAR评估结果摘要"""
        logging.info("\n" + "="*60)
        logging.info("ProteinSpectrumFAR 重建性能评估")
        logging.info("="*60)
        
        logging.info(f"\n评估样本数: {len(results['far_rmse'])}")
        logging.info(f"每个蛋白质采样数: {self.config.num_samples}")
        
        logging.info("\n--- FAR轨迹重建性能 ---")
        far_rmse_stats = self.compute_statistics(results['far_rmse'])
        far_mae_stats = self.compute_statistics(results['far_mae'])
        far_corr_stats = self.compute_statistics(results['far_correlation'])
        
        logging.info(f"FAR RMSE (集成平均): {far_rmse_stats['mean']:.4f} ± {far_rmse_stats['std']:.4f} Å")
        logging.info(f"FAR MAE (集成平均): {far_mae_stats['mean']:.4f} ± {far_mae_stats['std']:.4f} Å")
        logging.info(f"FAR相关性 (集成平均): {far_corr_stats['mean']:.4f} ± {far_corr_stats['std']:.4f}")
        if "spectral_mse" in results and results["spectral_mse"]:
            spec_mse_stats = self.compute_statistics(results["spectral_mse"])
            logging.info(f"Spectral MSE (未反归一化): {spec_mse_stats['mean']:.4e} ± {spec_mse_stats['std']:.4e}")

        # 多样性统计
        if results.get('diversity_metrics') is not None and self.config.num_samples > 1:
            logging.info("\n--- 多样性采样分析 ---")
            diversity_std_stats = self.compute_statistics(results['diversity_metrics']['sample_std'])
            min_rmse_stats = self.compute_statistics(results['diversity_metrics']['sample_min_rmse'])
            max_rmse_stats = self.compute_statistics(results['diversity_metrics']['sample_max_rmse'])
            ensemble_rmse_stats = self.compute_statistics(results['diversity_metrics']['ensemble_rmse'])
            
            logging.info(f"样本间坐标标准差: {diversity_std_stats['mean']:.4f} ± {diversity_std_stats['std']:.4f} Å")
            logging.info(f"最佳样本RMSE: {min_rmse_stats['mean']:.4f} ± {min_rmse_stats['std']:.4f} Å")
            logging.info(f"最差样本RMSE: {max_rmse_stats['mean']:.4f} ± {max_rmse_stats['std']:.4f} Å")
            logging.info(f"集成平均RMSE: {ensemble_rmse_stats['mean']:.4f} ± {ensemble_rmse_stats['std']:.4f} Å")
            logging.info(f"集成改进: {(min_rmse_stats['mean'] - ensemble_rmse_stats['mean']) / min_rmse_stats['mean'] * 100:.2f}%")

        logging.info("\n--- 按蛋白质统计 ---")
        protein_rmse_stats = self.compute_statistics(results['per_protein_metrics']['rmse'])
        protein_mae_stats = self.compute_statistics(results['per_protein_metrics']['mae'])
        protein_corr_stats = self.compute_statistics(results['per_protein_metrics']['correlation'])
        num_residues_stats = self.compute_statistics(results['per_protein_metrics']['num_residues'])
        num_frames_stats = self.compute_statistics(results['per_protein_metrics']['num_frames'])
        
        logging.info(f"蛋白质RMSE: {protein_rmse_stats['mean']:.4f} ± {protein_rmse_stats['std']:.4f} Å")
        logging.info(f"蛋白质MAE: {protein_mae_stats['mean']:.4f} ± {protein_mae_stats['std']:.4f} Å")
        logging.info(f"蛋白质相关性: {protein_corr_stats['mean']:.4f} ± {protein_corr_stats['std']:.4f}")
        logging.info(f"平均残基数: {num_residues_stats['mean']:.1f} ± {num_residues_stats['std']:.1f}")
        logging.info(f"平均帧数: {num_frames_stats['mean']:.1f} ± {num_frames_stats['std']:.1f}")
        
        logging.info("\n--- 时间统计 ---")
        total_time_stats = self.compute_statistics(results['timing']['total_times'])
        trajectory_time_stats = self.compute_statistics(results['timing']['trajectory_reconstruction_times'])
        metrics_time_stats = self.compute_statistics(results['timing']['metrics_computation_times'])
        
        logging.info(f"总时间: {total_time_stats['mean']:.3f} ± {total_time_stats['std']:.3f} s")
        logging.info(
            f"轨迹重建时间: {trajectory_time_stats['mean']:.3f} ± {trajectory_time_stats['std']:.3f} s "
            f"({trajectory_time_stats['mean']/total_time_stats['mean']*100:.1f}%)"
        )
        logging.info(
            f"指标计算时间: {metrics_time_stats['mean']:.3f} ± {metrics_time_stats['std']:.3f} s "
            f"({metrics_time_stats['mean']/total_time_stats['mean']*100:.1f}%)"
        )
        
        logging.info("="*60)
    
    def amino_acid_id_to_name(self, aatype_id):
        """将氨基酸类型ID转换为三字母名称"""
        amino_acid_names = [
            'ALA', 'ARG', 'ASN', 'ASP', 'CYS',
            'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
            'LEU', 'LYS', 'MET', 'PHE', 'PRO',
            'SER', 'THR', 'TRP', 'TYR', 'VAL'
        ]
        return amino_acid_names[aatype_id]
    
    def coordinates_to_pdb(self, coordinates, aatype, output_path, 
                          protein_name="PROTEIN", chain_id="A", ca_only=True):
        """
        将重建的坐标序列转换为PDB文件
        Args:
            coordinates: [num_residues, 3, num_frames] 重建的CA原子坐标序列
            aatype: [num_residues] 氨基酸类型ID (0-19)
            output_path: 输出PDB文件路径
            protein_name: 蛋白质名称
            chain_id: 链ID
            ca_only: 是否只写CA原子
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
                    aa_name = self.amino_acid_id_to_name(aatype[res_idx].item())
                    
                    if ca_only:
                        # 只写CA原子
                        f.write(
                            f"ATOM  {atom_idx:5d}  CA  {aa_name} {chain_id}{res_idx + 1:4d}    "
                            f"{ca_coord[0]:8.3f}{ca_coord[1]:8.3f}{ca_coord[2]:8.3f}  "
                            f"1.00  0.00           C\n"
                        )
                        atom_idx += 1
                    else:
                        # 写完整骨架原子(估计位置)
                        # CA原子
                        f.write(
                            f"ATOM  {atom_idx:5d}  CA  {aa_name} {chain_id}{res_idx + 1:4d}    "
                            f"{ca_coord[0]:8.3f}{ca_coord[1]:8.3f}{ca_coord[2]:8.3f}  "
                            f"1.00  0.00           C\n"
                        )
                        atom_idx += 1
                        
                        # C原子(估计位置)
                        c_coord = ca_coord + np.array([1.53, 0.0, 0.0])
                        f.write(
                            f"ATOM  {atom_idx:5d}  C   {aa_name} {chain_id}{res_idx + 1:4d}    "
                            f"{c_coord[0]:8.3f}{c_coord[1]:8.3f}{c_coord[2]:8.3f}  "
                            f"1.00  0.00           C\n"
                        )
                        atom_idx += 1
                        
                        # N原子(估计位置)
                        n_coord = ca_coord + np.array([-1.33, 0.0, 0.0])
                        f.write(
                            f"ATOM  {atom_idx:5d}  N   {aa_name} {chain_id}{res_idx + 1:4d}    "
                            f"{n_coord[0]:8.3f}{n_coord[1]:8.3f}{n_coord[2]:8.3f}  "
                            f"1.00  0.00           N\n"
                        )
                        atom_idx += 1
                        
                        # O原子(估计位置)
                        o_coord = c_coord + np.array([0.0, 1.24, 0.0])
                        f.write(
                            f"ATOM  {atom_idx:5d}  O   {aa_name} {chain_id}{res_idx + 1:4d}    "
                            f"{o_coord[0]:8.3f}{o_coord[1]:8.3f}{o_coord[2]:8.3f}  "
                            f"1.00  0.00           O\n"
                        )
                        atom_idx += 1
                
                f.write("TER\n")
                f.write("ENDMDL\n")
            
            f.write("END\n")
        
        logging.info(f"PDB文件已保存至: {output_path}")
        logging.info(f"模式: {'仅CA原子' if ca_only else '完整骨架原子'}")
        logging.info(
            f"坐标范围: X[{coordinates[:, 0, :].min():.3f}, {coordinates[:, 0, :].max():.3f}], "
            f"Y[{coordinates[:, 1, :].min():.3f}, {coordinates[:, 1, :].max():.3f}], "
            f"Z[{coordinates[:, 2, :].min():.3f}, {coordinates[:, 2, :].max():.3f}]"
        )

    def coordinates_to_pdb_per_frame(
        self,
        coordinates: torch.Tensor,
        aatype: torch.Tensor,
        output_dir: str,
        protein_name: str = "PROTEIN",
        chain_id: str = "A",
        ca_only: bool = True,
    ):
        """
        将重建轨迹按帧分别保存为PDB文件。
        文件名格式: frame00000.pdb (5位frame idx)
        """
        os.makedirs(output_dir, exist_ok=True)
        num_residues, _, num_frames = coordinates.shape

        for frame_idx in range(num_frames):
            frame_path = os.path.join(output_dir, f"frame{frame_idx:05d}.pdb")
            with open(frame_path, "w") as f:
                f.write("REMARK  Generated by ProteinSpectrumFAR Reconstruction\n")
                f.write(f"REMARK  Protein: {protein_name}\n")
                f.write(f"REMARK  Number of residues: {num_residues}\n")
                f.write(f"REMARK  Frame index: {frame_idx}\n")
                f.write(f"TITLE     {protein_name} FAR Reconstruction\n")
                f.write(f"MODEL     {frame_idx + 1:4d}\n")

                atom_idx = 1
                for res_idx in range(num_residues):
                    ca_coord = coordinates[res_idx, :, frame_idx].cpu().numpy()
                    aa_name = self.amino_acid_id_to_name(aatype[res_idx].item())

                    if ca_only:
                        f.write(
                            f"ATOM  {atom_idx:5d}  CA  {aa_name} {chain_id}{res_idx + 1:4d}    "
                            f"{ca_coord[0]:8.3f}{ca_coord[1]:8.3f}{ca_coord[2]:8.3f}  "
                            f"1.00  0.00           C\n"
                        )
                        atom_idx += 1
                    else:
                        f.write(
                            f"ATOM  {atom_idx:5d}  CA  {aa_name} {chain_id}{res_idx + 1:4d}    "
                            f"{ca_coord[0]:8.3f}{ca_coord[1]:8.3f}{ca_coord[2]:8.3f}  "
                            f"1.00  0.00           C\n"
                        )
                        atom_idx += 1

                        c_coord = ca_coord + np.array([1.53, 0.0, 0.0])
                        f.write(
                            f"ATOM  {atom_idx:5d}  C   {aa_name} {chain_id}{res_idx + 1:4d}    "
                            f"{c_coord[0]:8.3f}{c_coord[1]:8.3f}{c_coord[2]:8.3f}  "
                            f"1.00  0.00           C\n"
                        )
                        atom_idx += 1

                        n_coord = ca_coord + np.array([-1.33, 0.0, 0.0])
                        f.write(
                            f"ATOM  {atom_idx:5d}  N   {aa_name} {chain_id}{res_idx + 1:4d}    "
                            f"{n_coord[0]:8.3f}{n_coord[1]:8.3f}{n_coord[2]:8.3f}  "
                            f"1.00  0.00           N\n"
                        )
                        atom_idx += 1

                        o_coord = c_coord + np.array([0.0, 1.24, 0.0])
                        f.write(
                            f"ATOM  {atom_idx:5d}  O   {aa_name} {chain_id}{res_idx + 1:4d}    "
                            f"{o_coord[0]:8.3f}{o_coord[1]:8.3f}{o_coord[2]:8.3f}  "
                            f"1.00  0.00           O\n"
                        )
                        atom_idx += 1

                f.write("TER\n")
                f.write("ENDMDL\n")
                f.write("END\n")

        logging.info(f"逐帧PDB已保存至目录: {output_dir}")

    @staticmethod
    def _get_chain_id_from_protein_id(protein_id: str) -> str:
        # 示例: 5znj_A_R1 -> A
        if isinstance(protein_id, str):
            parts = protein_id.split("_")
            if len(parts) >= 2 and parts[1]:
                return parts[1][0]
        return "A"
    
    def _test_gt_constructed_acc(
        self,
        coords_preprocessed_direct: np.ndarray,
        x_proc: torch.Tensor,
        protein_id: str,
    ):
        """
        比较直接法和重建法得到的坐标，输出误差信息。

        Args:
            coords_preprocessed_direct (np.ndarray): 直接法得到的 (window_size, N, 3) 坐标
            x_proc (torch.Tensor): 重建方法输出的 (N, 3, window_size) tensor
            protein_id (str): 蛋白质ID

        Returns:
            dict
        """
        # 转换为 (window_size, N, 3) 格式以便比较
        coords_reconstructed_permuted = x_proc.permute(2, 0, 1).contiguous().cpu().numpy()  # (window_size, N, 3)
        logging.info(f"方法2 (重建): shape={coords_reconstructed_permuted.shape}")
        
        # 比较
        diff = coords_preprocessed_direct - coords_reconstructed_permuted
        abs_diff = np.abs(diff)
        max_diff = np.max(abs_diff)
        mean_diff = np.mean(abs_diff)
        std_diff = np.std(abs_diff)
        
        # 计算相对误差
        coords_norm = np.linalg.norm(coords_preprocessed_direct, axis=-1)
        rel_diff = abs_diff / (coords_norm[..., np.newaxis] + 1e-8)
        max_rel_diff = np.max(rel_diff)
        mean_rel_diff = np.mean(rel_diff)
        
        logging.info("\n" + "="*60)
        logging.info(f"{protein_id} 比较结果:")
        logging.info("="*60)
        logging.info(f"最大绝对差异: {max_diff:.6e}")
        logging.info(f"平均绝对差异: {mean_diff:.6e}")
        logging.info(f"标准差: {std_diff:.6e}")
        logging.info(f"最大相对差异: {max_rel_diff:.6e}")
        logging.info(f"平均相对差异: {mean_rel_diff:.6e}")
        
        # 检查是否几乎相同
        tolerance = 1e-5
        if max_diff < tolerance:
            logging.info(f"\n✓ 两种方法得到的坐标几乎相同 (max_diff < {tolerance})")
        else:
            logging.warning(f"\n✗ 两种方法得到的坐标有显著差异 (max_diff >= {tolerance})")
            
            # 找出差异最大的位置
            max_diff_idx = np.unravel_index(np.argmax(abs_diff), abs_diff.shape)
            logging.info(f"最大差异位置: frame={max_diff_idx[0]}, residue={max_diff_idx[1]}, axis={max_diff_idx[2]}")
            logging.info(f"  直接方法值: {coords_preprocessed_direct[max_diff_idx]:.6f}")
            logging.info(f"  重建方法值: {coords_reconstructed_permuted[max_diff_idx]:.6f}")
            logging.info(f"  差异: {diff[max_diff_idx]:.6e}")
        
        # 统计信息
        logging.info("\n" + "="*60)
        logging.info("统计信息:")
        logging.info("="*60)
        logging.info(f"直接方法坐标范围:")
        logging.info(f"  X: [{coords_preprocessed_direct[:, :, 0].min():.3f}, {coords_preprocessed_direct[:, :, 0].max():.3f}]")
        logging.info(f"  Y: [{coords_preprocessed_direct[:, :, 1].min():.3f}, {coords_preprocessed_direct[:, :, 1].max():.3f}]")
        logging.info(f"  Z: [{coords_preprocessed_direct[:, :, 2].min():.3f}, {coords_preprocessed_direct[:, :, 2].max():.3f}]")
        logging.info(f"重建方法坐标范围:")
        logging.info(f"  X: [{coords_reconstructed_permuted[:, :, 0].min():.3f}, {coords_reconstructed_permuted[:, :, 0].max():.3f}]")
        logging.info(f"  Y: [{coords_reconstructed_permuted[:, :, 1].min():.3f}, {coords_reconstructed_permuted[:, :, 1].max():.3f}]")
        logging.info(f"  Z: [{coords_reconstructed_permuted[:, :, 2].min():.3f}, {coords_reconstructed_permuted[:, :, 2].max():.3f}]")
        
        return {
            'max_diff': max_diff,
            'mean_diff': mean_diff,
            'std_diff': std_diff,
            'max_rel_diff': max_rel_diff,
            'mean_rel_diff': mean_rel_diff,
            'coords_direct': coords_preprocessed_direct,
            'coords_reconstructed': coords_reconstructed_permuted,
            'x_proc': x_proc,  # (N, 3, window_size) 格式，用于保存PDB
            'protein_id': protein_id,
        }


    def reconstruct_and_save_pdb(self):
        """重建轨迹、计算指标并保存为PDB文件（合并了原来的 evaluate_reconstruction 逻辑）"""
        output_dir = self.config.pdb_output_dir if self.config.save_pdb else None
        
        # 初始化结果字典
        results = {
            "far_rmse": [],
            "far_mae": [],
            "far_correlation": [],
            "per_protein_metrics": defaultdict(list),
            "spectral_mse": [],
            "timing": {
                "total_times": [],
                "far_sampling_times": [],
                "trajectory_reconstruction_times": [],
                "metrics_computation_times": [],
            },
            "diversity_metrics": {
                "sample_std": [],
                "sample_min_rmse": [],
                "sample_max_rmse": [],
                "ensemble_rmse": [],
            } if self.config.num_samples > 1 else None,
        }

        num_to_process = self.config.num_pdb_proteins or len(self.data_list)
        logging.info(f"开始轨迹重建、评估并保存为PDB文件...")
        logging.info(f"处理蛋白质索引: 0-{num_to_process-1}")
        if output_dir:
            logging.info(f"PDB输出目录: {output_dir}")

        for idx in range(min(num_to_process, len(self.data_list))):
            batch_start_time = time.time()
            protein_id = self.protein_ids[idx]
            protein_data = self.trajectory_data_dict[protein_id]
            positions = protein_data["positions"]
            aatype = protein_data["amino_acid_types"].clone().detach()
            warn_gt_spectrum = None

            logging.info(f"\n处理蛋白质 {idx + 1}/{num_to_process}: {protein_id}")

            sampling_start_time = time.time()
            pred_spectrum = None
            all_samples = None
            warn_gt_spectrum = None

            if self.data_mode == "fft_window":
                x_windows = positions.get("x_fft_windows", [])
                if not x_windows:
                    logging.warning(f"{protein_id}: 未找到 x_fft_windows，跳过")
                    continue
                window_size = int(positions.get("fft_window_size"))
                num_windows = len(x_windows)
                preprocess_states = positions.get("preprocess_state_fft_windows")

                rotation_matrices = (
                    protein_data["rotation_matrices0"].clone().detach().to(self.device)
                )
                aatype_device = aatype.to(self.device)
                aa_mask = torch.ones(
                    aatype_device.shape, dtype=torch.bool, device=self.device
                )
                esm_features = torch.tensor(
                    protein_data["esm"], device=self.device
                )
                coords_preprocessed_windows = positions.get("coords_preprocessed_fft_windows", []) # [(window_size, N, 3)]

                predicted_windows = []
                gt_windows = []
                # Recursive structural anchor (paper Sec. 3.3, Eq. 8):
                #   x_start^(k+1) = X_hat^(k)_W
                # Window 1 is seeded with the ground-truth initial conformation;
                # every later window is conditioned on the LAST FRAME OF THE
                # PREVIOUSLY GENERATED WINDOW, not on ground truth. This is what
                # makes the rollout a genuine autoregressive rollout.
                rollout_anchor = None
                for w in range(num_windows):
                    if rollout_anchor is None:
                        # first window: ground-truth anchor (aligned frame 0)
                        frame0 = torch.tensor(
                            np.stack(
                                [
                                    positions["x0_fft_windows"][w],
                                    positions["y0_fft_windows"][w],
                                    positions["z0_fft_windows"][w],
                                ],
                                axis=-1,
                            ),
                            dtype=torch.float32,
                        ).to(self.device)
                    else:
                        # subsequent windows: generated anchor from window w-1
                        frame0 = rollout_anchor

                    spectrum = self._build_fft_window_spectrum(positions, w).to(self.device)
                    pred_spectrum_window = self._predict_spectrum(
                        (
                            frame0,
                            spectrum,
                            rotation_matrices,
                            aatype_device,
                            aa_mask,
                            esm_features,
                            None,
                        )
                    ).squeeze(-1)
                    # 计算spectral_mse
                    gt_spec_squeezed = spectrum.squeeze(-1) if spectrum.dim() > pred_spectrum_window.dim() else spectrum
                    mse_loss = torch.nn.functional.mse_loss(pred_spectrum_window, gt_spec_squeezed)
                    results["spectral_mse"].append(mse_loss.item())

                    # Decode this window and take its last frame as the next anchor.
                    window_coords_w = self._spectrum_to_full_fft_coords(
                        pred_spectrum_window,
                        frame0,
                        window_size,
                        num_frames=self.config.num_frames,
                        preprocess_state=(
                            preprocess_states[w]
                            if preprocess_states is not None and w < len(preprocess_states)
                            else None
                        ),
                        skip_invert_preprocessing=True,
                    )  # (N, 3, window_size)
                    rollout_anchor = window_coords_w[..., -1].detach()  # (N, 3)

                    predicted_windows.append((pred_spectrum_window, frame0))
                    gt_windows.append((spectrum, frame0))
                    # 保存第一个窗口的pred_spectrum用于warn_if_static
                    if w == 0:
                        pred_spectrum = pred_spectrum_window
                if num_windows > 0:
                    warn_gt_spectrum = self._build_fft_window_spectrum(positions, 0).to(self.device)
                
                sampling_end_time = time.time()
                results["timing"]["far_sampling_times"].append(sampling_end_time - sampling_start_time)
                
                trajectory_start_time = time.time()

                window_coords = [
                    self._spectrum_to_full_fft_coords(
                        pred_spectrum,
                        frame0,
                        window_size,
                        num_frames=self.config.num_frames,
                        preprocess_state=(
                            preprocess_states[w]
                            if preprocess_states is not None and w < len(preprocess_states)
                            else None
                        ),
                    )
                    for w, (pred_spectrum, frame0) in enumerate(predicted_windows)
                ]
                reconstructed_coords = torch.cat(window_coords, dim=-1)
                # 为了与coords_preprocessed_windows比较，需要重建不做invert_preprocessing的坐标
                gt_window_coords_for_test = [
                    self._spectrum_to_full_fft_coords(
                        gt_spectrum,
                        frame0,
                        window_size,
                        num_frames=self.config.num_frames,
                        preprocess_state=(
                            preprocess_states[w]
                            if preprocess_states is not None and w < len(preprocess_states)
                            else None
                        ),
                        skip_invert_preprocessing=True,  # 不做invert_preprocessing，用于与coords_preprocessed比较
                    )
                    for w, (gt_spectrum, frame0) in enumerate(gt_windows)
                ]
                # 测试重建精度（比较预处理后的坐标）
                for w in range(num_windows):
                    self._test_gt_constructed_acc(
                        coords_preprocessed_windows[w],
                        gt_window_coords_for_test[w],
                        protein_id,
                    )
                # 重建用于后续指标计算的坐标（需要做invert_preprocessing）
                gt_window_coords = [
                    self._spectrum_to_full_fft_coords(
                        gt_spectrum,
                        frame0,
                        window_size,
                        num_frames=self.config.num_frames,
                        preprocess_state=(
                            preprocess_states[w]
                            if preprocess_states is not None and w < len(preprocess_states)
                            else None
                        ),
                        skip_invert_preprocessing=False,  # 做invert_preprocessing，用于后续指标计算
                    )
                    for w, (gt_spectrum, frame0) in enumerate(gt_windows)
                ]
                gt_coords = torch.cat(gt_window_coords, dim=-1)
                # 转换为 (T, N, 3) 格式用于指标计算
                original_coords = gt_coords.permute(2, 0, 1).contiguous()
                
                trajectory_end_time = time.time()
                results["timing"]["trajectory_reconstruction_times"].append(
                    trajectory_end_time - trajectory_start_time
                )

            else:
                coords = torch.tensor(
                    np.stack([positions["x0"], positions["y0"], positions["z0"]], axis=-1),
                    dtype=torch.float32,
                ).to(self.device)
                preprocess_state = positions.get("preprocess_state")
                spectrum = self._build_fft_spectrum(positions).to(self.device)
                rotation_matrices = (
                    protein_data["rotation_matrices0"].clone().detach().to(self.device)
                )
                aatype_device = aatype.to(self.device)
                aa_mask = torch.ones(
                    aatype_device.shape, dtype=torch.bool, device=self.device
                )
                esm_features = torch.tensor(
                    protein_data["esm"], device=self.device
                )

                if self.config.num_samples > 1:
                    with self._no_mask_ctx():
                        with self._autocast_ctx():
                            all_samples = sample_multiple(
                                model=self.model,
                                coords=coords,
                                rotation_matrices=rotation_matrices,
                                aatype=aatype_device,
                                aa_mask=aa_mask,
                                esm_features=esm_features,
                                bihito_token_repr=None,
                                n_samples=self.config.num_samples,
                                num_iter=spectrum.shape[1],
                                cfg=self.config.cfg,
                                temperature=self.config.temperature,
                                progress=False,
                                sample_method="nomask",
                                seed=self.config.seed + idx,
                            )
                    pred_spectrum = all_samples.mean(dim=0)
                else:
                    pred_spectrum = self._predict_spectrum(
                        (
                            coords,
                            spectrum,
                            rotation_matrices,
                            aatype_device,
                            aa_mask,
                            esm_features,
                            None,
                        )
                    ).squeeze(-1)
                
                # 计算spectral_mse
                gt_spec_squeezed = spectrum.squeeze(-1)
                mse_loss = torch.nn.functional.mse_loss(pred_spectrum, gt_spec_squeezed)
                results["spectral_mse"].append(mse_loss.item())
                
                sampling_end_time = time.time()
                results["timing"]["far_sampling_times"].append(sampling_end_time - sampling_start_time)
                
                trajectory_start_time = time.time()
                
                reconstructed_coords = self._spectrum_to_full_fft_coords(
                    pred_spectrum,
                    coords,
                    window_size,
                    num_frames=self.config.num_frames,
                    preprocess_state=preprocess_state,
                )
                warn_gt_spectrum = spectrum
                gt_coords = self._spectrum_to_full_fft_coords(
                    spectrum.squeeze(-1),
                    coords,
                    window_size,
                    num_frames=self.config.num_frames,
                    preprocess_state=preprocess_state,
                )
                # 转换为 (T, N, 3) 格式用于指标计算
                original_coords = gt_coords.permute(2, 0, 1).contiguous()
                
                trajectory_end_time = time.time()
                results["timing"]["trajectory_reconstruction_times"].append(
                    trajectory_end_time - trajectory_start_time
                )

            # 计算指标
            metrics_start_time = time.time()
            
            # 对齐帧数（如果不同）
            if reconstructed_coords.shape[-1] != original_coords.shape[0]:
                min_frames = min(reconstructed_coords.shape[-1], original_coords.shape[0])
                if min_frames <= 0:
                    logging.warning(f"{protein_id}: min_frames<=0，跳过该样本")
                    continue
                reconstructed_coords = reconstructed_coords[..., :min_frames]
                original_coords = original_coords[:min_frames]
            
            coord_metrics = self.compute_reconstruction_metrics(original_coords, reconstructed_coords)
            
            if all_samples is not None and self.data_mode == "fft":
                # 多样性采样目前只支持fft模式
                diversity_metrics = self.analyze_sample_diversity(
                    all_samples, original_coords, coords, window_size, self.config.num_frames, use_full_fft=True
                )
                results["diversity_metrics"]["sample_std"].append(diversity_metrics["coords_diversity_std"])
                results["diversity_metrics"]["sample_min_rmse"].append(diversity_metrics["sample_min_rmse"])
                results["diversity_metrics"]["sample_max_rmse"].append(diversity_metrics["sample_max_rmse"])
                results["diversity_metrics"]["ensemble_rmse"].append(diversity_metrics["ensemble_rmse"])
            elif all_samples is not None:
                logging.warning(f"{protein_id}: 多样性采样暂不支持 {self.data_mode} 模式，跳过多样性统计")
            
            metrics_end_time = time.time()
            results["timing"]["metrics_computation_times"].append(
                metrics_end_time - metrics_start_time
            )
            
            results["far_rmse"].append(coord_metrics["overall_rmse"])
            results["far_mae"].append(coord_metrics["mae"])
            results["far_correlation"].append(coord_metrics["correlation"])
            results["per_protein_metrics"]["rmse"].append(coord_metrics["overall_rmse"])
            results["per_protein_metrics"]["mae"].append(coord_metrics["mae"])
            results["per_protein_metrics"]["correlation"].append(coord_metrics["correlation"])
            results["per_protein_metrics"]["num_residues"].append(reconstructed_coords.shape[0])
            results["per_protein_metrics"]["num_frames"].append(reconstructed_coords.shape[-1])
            
            batch_end_time = time.time()
            results["timing"]["total_times"].append(batch_end_time - batch_start_time)
            
            spec_mse_stats = self.compute_statistics(results["spectral_mse"])
            logging.info(
                f"  RMSE={coord_metrics['overall_rmse']:.4f}Å, "
                f"MAE={coord_metrics['mae']:.4f}Å, "
                f"Corr={coord_metrics['correlation']:.4f}, "
                f"Spectral MSE Loss={spec_mse_stats['mean']:.4e} ± {spec_mse_stats['std']:.4e}"
            )

            protein_name = f"{protein_id}"
            chain_id = self._get_chain_id_from_protein_id(protein_id)

            # 保存PDB文件（如果需要）
            if output_dir and self.config.save_gt:
                if self.config.save_pdb_per_frame:
                    gt_dir = os.path.join(output_dir, f"{protein_id}_gt")
                    self.coordinates_to_pdb_per_frame(
                        coordinates=gt_coords.cpu(),
                        aatype=aatype.cpu(),
                        output_dir=gt_dir,
                        protein_name=f"{protein_name}_gt",
                        chain_id=chain_id,
                        ca_only=True,
                    )
                    logging.info(f"  Ground truth轨迹已保存: {gt_dir}")
                else:
                    gt_output_path = os.path.join(output_dir, f"{protein_name}_ground_truth.pdb")
                    self.coordinates_to_pdb(
                        coordinates=gt_coords.cpu(),
                        aatype=aatype.cpu(),
                        output_path=gt_output_path,
                        protein_name=f"{protein_name}_ground_truth",
                        chain_id=chain_id,
                        ca_only=True,
                    )
                    logging.info(f"  Ground truth轨迹已保存: {gt_output_path}")

            logging.info(f"  重建坐标形状: {reconstructed_coords.shape}")
            self._warn_if_static(
                protein_id,
                reconstructed_coords,
                pred_spectrum=pred_spectrum,
                gt_spectrum=warn_gt_spectrum,
            )

            if output_dir:
                if self.config.save_pdb_per_frame:
                    pred_dir = os.path.join(output_dir, f"{protein_id}_pred")
                    self.coordinates_to_pdb_per_frame(
                        coordinates=reconstructed_coords.cpu(),
                        aatype=aatype.cpu(),
                        output_dir=pred_dir,
                        protein_name=f"{protein_name}_pred",
                        chain_id=chain_id,
                        ca_only=True,
                    )
                else:
                    reconstructed_output_path = os.path.join(
                        output_dir, f"{protein_name}_pred.pdb"
                    )
                    self.coordinates_to_pdb(
                        coordinates=reconstructed_coords.cpu(),
                        aatype=aatype.cpu(),
                        output_path=reconstructed_output_path,
                        protein_name=f"{protein_name}_pred",
                        chain_id=chain_id,
                        ca_only=True,
                    )
                    logging.info(f"  重建轨迹已保存: {reconstructed_output_path}")

        if output_dir:
            logging.info(f"\nPDB文件保存完成! 已处理 {num_to_process} 个蛋白质")
        
        return results
    
    def run(self):
        """执行推理流程（合并了重建、评估和保存逻辑，避免重复推理）"""
        logging.info("=" * 80)
        logging.info("开始FAR推理")
        logging.info("=" * 80)
        logging.info(f"\n推理参数:")
        logging.info(f"  数据模式: {self.data_mode}")
        logging.info(f"  随机种子: {self.config.seed}")
        
        # 重建轨迹、计算指标并保存PDB（合并逻辑，只推理一次）
        results = self.reconstruct_and_save_pdb()
        
        # 打印摘要
        self.print_summary(results)
        
        # 保存结果
        self.save_results(results)
        
        # 绘制结果
        plot_far_results(results, self.config.output_dir)
        
        logging.info(f"\nFAR推理完成! 结果保存在: {self.config.output_dir}")
        if self.config.save_pdb:
            logging.info(f"PDB文件保存在: {self.config.pdb_output_dir}")
    
    def evaluate_reconstruction(self):
        """兼容性接口：调用 reconstruct_and_save_pdb（但可能不保存PDB）"""
        # 临时禁用PDB保存，只计算指标
        original_save_pdb = self.config.save_pdb
        self.config.save_pdb = False
        try:
            results = self.reconstruct_and_save_pdb()
        finally:
            self.config.save_pdb = original_save_pdb
        return results

@dataclass
class Config:
    # 基础配置
    checkpoint_path: str
    data_path: str
    output_dir: str
    data_mode: str = "auto"  # auto/fft/fft_window

    # 运行配置
    device: str = "cuda"           
    model_size: str = "base"       # 'base', 'large', 'huge'
    cfg: float = 2.0               # Classifier-free guidance scale
    temperature: float = 0.5
    sample_method: str = "nomask"  # 'sample', 'nomask', 'mask'
    num_samples: int = 1           # 每个蛋白质的采样数(用于集成)
    seed: int = 42
    num_frames: int = 100
    num_pdb_proteins: int = None
    precision: Optional[str] = None
    scaling_factor_path: Optional[str] = None
    fft_norm: str = "backward"     # 'backward', 'ortho'

    # 导出
    save_pdb: bool = False
    pdb_output_dir: Optional[str] = None
    save_gt: bool = False
    save_pdb_per_frame: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference configuration")

    # 基础配置（建议必填）
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--data_path", type=str, required=True, help="Input data path")
    parser.add_argument("--window_size", type=int, default=2500, help="Window size")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save inference outputs")
    parser.add_argument("--data_mode", type=str, default="auto", choices=["auto", "fft", "fft_window"],
                        help="Data mode: auto/fft/fft_window")

    # 运行配置
    parser.add_argument("--device", type=str, default="cuda", help='Device, e.g. "cuda", "cuda:0", "cpu"')
    parser.add_argument("--model_size", type=str, default="base", help='Model size, e.g. "tiny", "base", "large"')
    parser.add_argument("--cfg", type=float, default=2.0, help="Guidance scale")
    parser.add_argument("--temperature", type=float, default=0.5, help="Sampling temperature")
    parser.add_argument("--sample_method", type=str, default="nomask", help="Sampling method")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of samples to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_frames", type=int, default=100, help="Number of frames to generate")
    parser.add_argument("--num_pdb_proteins", type=int, default=None, help="Number of proteins to save as PDB (default: None, i.e., all proteins)")
    parser.add_argument("--precision", type=str, default=None, help="Inference precision: bf16/fp32 (default: from checkpoint)")
    parser.add_argument("--scaling_factor_path", type=str, default=None, help="Scaling factor path for spectrum denormalization")

    # 导出
    parser.add_argument("--save_pdb", action="store_true", help="Save reconstructed trajectories to PDB")
    parser.add_argument("--pdb_output_dir", type=str, default=None, help="Directory for PDB outputs")
    parser.add_argument("--save_gt", action="store_true", help="Save ground-truth data")
    parser.add_argument(
        "--save_pdb_per_frame",
        action="store_true",
        help="Save each frame as a separate PDB file under <id>_pred/<id>_gt",
    )

    return parser.parse_args()

def _expand(path: Optional[str]) -> Optional[str]:
    if path is None:
        return None
    return os.path.abspath(os.path.expanduser(os.path.expandvars(path)))

def build_config_from_args(args: argparse.Namespace) -> Config:
    if "fft_window5000" in args.data_path:
        fft_norm = "backward"
    elif "fft_window2500" in args.data_path or "stft" in args.data_path:
        fft_norm = "ortho"
    else:
        fft_norm = "backward"
    print(f"fft_norm: {fft_norm}")
    cfg_obj = Config(
        checkpoint_path=_expand(args.checkpoint_path),
        data_path=_expand(args.data_path),
        output_dir=_expand(args.output_dir),
        data_mode=args.data_mode,
        device=args.device,
        model_size=args.model_size,
        cfg=args.cfg,
        temperature=args.temperature,
        sample_method=args.sample_method,
        num_samples=args.num_samples,
        seed=args.seed,
        num_frames=args.num_frames,
        fft_norm=fft_norm,
        precision=args.precision,
        scaling_factor_path=_expand(args.scaling_factor_path) if args.scaling_factor_path else None,
        save_pdb=args.save_pdb,
        pdb_output_dir=_expand(args.pdb_output_dir) if args.pdb_output_dir else None,
        num_pdb_proteins=args.num_pdb_proteins,
        save_gt=args.save_gt,
        save_pdb_per_frame=args.save_pdb_per_frame,
    )

    # 默认 PDB 输出目录
    if cfg_obj.save_pdb and cfg_obj.pdb_output_dir is None:
        cfg_obj.pdb_output_dir = os.path.join(cfg_obj.output_dir, "reconstructed_trajs")
    
    if cfg_obj.save_pdb_per_frame:
        cfg_obj.save_pdb = True

    # 创建需要的目录
    os.makedirs(cfg_obj.output_dir, exist_ok=True)
    if cfg_obj.save_pdb and cfg_obj.pdb_output_dir:
        os.makedirs(cfg_obj.pdb_output_dir, exist_ok=True)

    return cfg_obj


def main():
    """主函数"""
    LOG_FORMAT = "%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
    logging.basicConfig(
        format=LOG_FORMAT,
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    args = parse_args()
    config = build_config_from_args(args)
    set_seed(config.seed)
    
    logging.info("=" * 80)
    logging.info("推理配置:")
    for key, value in vars(config).items():
        if not key.startswith('_'):
            logging.info(f"  {key}: {value}")
    logging.info("=" * 80)
    
    inferencer = FARInferencer(config)
    inferencer.run()


if __name__ == "__main__":
    main()

