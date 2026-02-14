import os
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F
import random
import numpy as np
import math
from types import SimpleNamespace
# import warnings

from protein_spectrum_far import protein_spectrum_far_base
from src.model.loss.spectrum_loss import SpectrumChunkLoss

def _mem():
    torch.cuda.synchronize()
    alloc = torch.cuda.memory_allocated() / 1024**2
    reserv = torch.cuda.memory_reserved() / 1024**2
    peak = torch.cuda.max_memory_allocated() / 1024**2
    return alloc, reserv, peak

class WarmupCosineWithPlateauScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self, 
        optimizer, 
        warmup_epochs, 
        max_epochs, 
        min_lr=1e-6, 
        warmup_start_lr=1e-7, 
        plateau_patience=5, 
        plateau_factor=0.5, 
        last_epoch=-1, 
        verbose=False
    ):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.min_lr = min_lr
        self.warmup_start_lr = warmup_start_lr
        self.plateau_patience = plateau_patience
        self.plateau_factor = plateau_factor
        self.verbose = verbose
        
        # Initialize base class
        super().__init__(optimizer, last_epoch, verbose)
        
        # Set initial learning rate
        if self.last_epoch == -1:
            for param_group in self.optimizer.param_groups:
                param_group['initial_lr'] = param_group['lr']
        
        self.base_lrs = [group['initial_lr'] for group in optimizer.param_groups]
        
        # Initialize plateau scheduler
        self.plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=plateau_factor,
            patience=plateau_patience,
            verbose=verbose
        )

    def get_lr(self):
        lrs = []
        for base_lr in self.base_lrs:
            if self.last_epoch < self.warmup_epochs:
                # Linear warmup
                lr = self.warmup_start_lr + (base_lr - self.warmup_start_lr) * (self.last_epoch / self.warmup_epochs)
            else:
                # Cosine annealing
                progress = (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
                progress = min(1.0, progress)  # Ensure not exceeding 1
                lr = self.min_lr + 0.5 * (base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))
            lrs.append(lr)
        return lrs

    def step(self, metrics=None):
        # Execute base class step first
        super().step()
        
        # If validation metrics are provided, update plateau scheduler
        if metrics is not None:
            self.plateau_scheduler.step(metrics)

    def state_dict(self):
        state = super().state_dict()
        state.update({
            'warmup_epochs': self.warmup_epochs,
            'max_epochs': self.max_epochs,
            'min_lr': self.min_lr,
            'warmup_start_lr': self.warmup_start_lr,
            'plateau_patience': self.plateau_patience,
            'plateau_factor': self.plateau_factor,
            'plateau_scheduler': self.plateau_scheduler.state_dict()
        })
        return state

    def load_state_dict(self, state_dict):
        # Save plateau scheduler state
        plateau_state = state_dict.pop('plateau_scheduler')
        
        # Update other attributes
        for key, value in state_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
        
        # Load base class state
        super().load_state_dict(state_dict)
        
        # Restore plateau scheduler state
        self.plateau_scheduler.load_state_dict(plateau_state)

class ProteinDynamicsFAR(pl.LightningModule):
    """蛋白质动力学FAR模型"""
    def __init__(self, config):
        super().__init__()
        self.config = config

        print("="*80)
        print("Initializing Protein Spectrum FAR")
        print("="*80)
    
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        
        # Create model
        print("\nCreating model...")
        # NOTE: c_s (default 1024) % 8 = 0 ✓, decoder_embed_dim (512) % 8 = 0 ✓
        
        # 保存模型配置以便推理时使用
        # self.model_config = {
        #     'spectrum_token_dim': 1,
        #     'num_xyz_channels': 3,
        #     'use_complex_spectrum': True,
        #     'aatype_vocab_size': 21,
        #     'esm_dim': 1280,
        #     'encoder_depth': 4,
        #     'encoder_num_heads': 8,  # c_s=1024 % 8 = 0 ✓
        #     'decoder_embed_dim': 512,  # 512 % 8 = 0 ✓
        #     'decoder_depth': 4,
        #     'decoder_num_heads': 8,
        #     'buffer_size': 32,
        #     'diffloss_d': 2,
        #     'diffloss_w': 512,
        #     'num_sampling_steps': '50',
        #     'diffusion_batch_mul': 1,
        # }
        train_path = getattr(config, "train_path", None)
        # use_cross_window = (
        #     isinstance(train_path, str)
        #     and "stft" in train_path
        #     and "fft_window" not in train_path
        # )
        use_cross_window = False
        self.model_config = {
            'spectrum_token_dim': config.spectrum_token_dim,
            'num_xyz_channels': config.num_xyz_channels,
            'use_complex_spectrum': config.use_complex_spectrum,
            'aatype_vocab_size': config.aatype_vocab_size,
            'esm_dim': config.esm_dim,
            'encoder_depth': config.encoder_depth,
            'encoder_num_heads': config.encoder_num_heads,
            'decoder_embed_dim': config.decoder_embed_dim,
            'decoder_depth': config.decoder_depth,
            'decoder_num_heads': config.decoder_num_heads,
            'buffer_size': config.buffer_size,
            'diffloss_d': config.diffloss_d,
            'diffloss_w': config.diffloss_w,
            'num_sampling_steps': config.num_sampling_steps,
            'diffusion_batch_mul': config.diffusion_batch_mul,
            # diffusion core knobs with safe defaults
            'learn_sigma': getattr(config, 'learn_sigma', False),
            'predict_xstart': getattr(config, 'predict_xstart', True),
            'sigma_small': getattr(config, 'sigma_small', False),
            'noise_schedule': getattr(config, 'noise_schedule', 'cosine'),
            'use_kl': getattr(config, 'use_kl', False),
            'rescale_learned_sigmas': getattr(config, 'rescale_learned_sigmas', False),
            'diffusion_steps': getattr(config, 'diffusion_steps', 1000),
            'max_spectrum_len': getattr(config, 'max_spectrum_len', None),
            'denoise_seq_chunk_size': getattr(config, 'denoise_seq_chunk_size', None),
            'denoise_block_type': getattr(config, 'denoise_block_type', 'residue'),
            'denoise_use_checkpoint': getattr(config, 'denoise_use_checkpoint', False),
            'use_checkpoint': getattr(config, 'use_checkpoint', False),
            'enable_cross_window': getattr(config, 'enable_cross_window', use_cross_window),
            'use_self_forcing': getattr(config, 'use_self_forcing', False),
            'self_forcing_schedule': getattr(config, 'self_forcing_schedule', None),
        }
                
        self.model = protein_spectrum_far_base(**self.model_config)
        
        # 将配置保存到hparams以便自动保存到checkpoint
        self.save_hyperparameters({'model_config': self.model_config})

        # self.model = self.model.to(device)

        num_params = sum(p.numel() for p in self.model.parameters())
        print(f"Model parameters: {num_params:,}")

        self.beta = config.beta  # KL loss weight parameter

        # 额外频谱损失（SpectrumChunkLoss）
        # 需要 config.spectrum_loss.eps / config.spectrum_loss.weight 等配置存在
        self.spectrum_chunk_loss = None
        self.spectrum_chunk_loss_scale = float(getattr(config, "spectrum_chunk_loss_scale", 1.0))
        # YAML 很可能把 spectrum_loss 覆盖成 dict，这里做一次兼容转换成可点号访问对象
        if hasattr(config, "spectrum_loss") and isinstance(getattr(config, "spectrum_loss"), dict):
            sl = getattr(config, "spectrum_loss")
            # weight 也可能是 dict
            w = sl.get("weight", None)
            if isinstance(w, dict):
                sl = dict(sl)
                sl["weight"] = SimpleNamespace(**w)
            setattr(config, "spectrum_loss", SimpleNamespace(**sl))

        if getattr(config, "enable_spectrum_chunk_loss", True) and hasattr(config, "spectrum_loss"):
            try:
                self.spectrum_chunk_loss = SpectrumChunkLoss(config)
            except Exception as e:
                # 不阻塞训练：如果用户没配齐 spectrum_loss 配置则跳过
                if self.trainer is None or self.trainer.is_global_zero:
                    print(f"[WARN] Failed to init SpectrumChunkLoss, will disable it. Error: {e}")
                self.spectrum_chunk_loss = None

    def on_after_backward(self):
        if self.global_step == 0 and self.trainer.is_global_zero:
            unused = []
            for n, p in self.named_parameters():
                if p.requires_grad and p.grad is None:
                    unused.append(n)
            if unused:
                print(f"[Step {self.global_step}] Unused parameters (no grad):")
            for n in unused:
                print(" -", n)
            else:
                print(f"[Step {self.global_step}] All trainable params got grad.")

    def forward(self, spectrum, coords, rotation_matrices, aatype, aa_mask, esm_features, bihito_token_repr, loss_weight=True, return_pred_spectrum=True, last_spectrum=None, window_size=None, hop_ratio=None):
        return self.model(
            spectrum,
            coords,
            rotation_matrices,
            aatype,
            aa_mask,
            esm_features,
            bihito_token_repr,
            loss_weight=loss_weight,
            return_pred_spectrum=return_pred_spectrum,
            last_spectrum=last_spectrum,
            window_size=window_size,
            hop_ratio=hop_ratio,
        )

    @staticmethod
    def _squeeze_batch1(x: torch.Tensor, name: str) -> torch.Tensor:
        """本模型的 backbone 期望无 batch 维；如果 DataLoader 额外加了 batch=1，则安全 squeeze 掉。"""
        if isinstance(x, torch.Tensor) and x.dim() >= 1 and x.shape[0] == 1:
            return x.squeeze(0)
        return x

    @staticmethod
    def _spectrum6_to_real_imag_xyz(spec: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        把 spectrum 的 6 通道 (x_re,x_im,y_re,y_im,z_re,z_im) 转成:
          real: (N, 3, L)
          imag: (N, 3, L)
        兼容输入形状:
          - (N, L, 6, token_dim)  (token_dim 通常为 1)
          - (N, L, 6)
        """
        if spec.dim() == 4:
            # (N, L, 6, token_dim) -> (N, L, 6)
            if spec.shape[-1] == 1:
                spec = spec.squeeze(-1)
            else:
                # 如果 token_dim>1，则按 token_dim 展平到通道里会更合理，但当前训练数据通常是 1
                raise ValueError(f"Unsupported spectrum token_dim={spec.shape[-1]} (expected 1). shape={tuple(spec.shape)}")
        if spec.dim() != 3 or spec.shape[-1] != 6:
            raise ValueError(f"Expected spectrum shape (N,L,6) or (N,L,6,1), got {tuple(spec.shape)}")

        # (N, L, 6) -> (N, 3, L)
        real = torch.stack([spec[..., 0], spec[..., 2], spec[..., 4]], dim=-1).permute(0, 2, 1).contiguous()
        imag = torch.stack([spec[..., 1], spec[..., 3], spec[..., 5]], dim=-1).permute(0, 2, 1).contiguous()
        return real, imag

    def _compute_spectrum_chunk_losses(
        self,
        pred_spectrum: torch.Tensor,
        true_spectrum: torch.Tensor,
        *,
        loss_weight: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """
        用 SpectrumChunkLoss 计算频谱额外损失（返回 cum_loss, metrics）。
        支持的频谱形状:
          - (N, L, 6, 1) 或 (N, L, 6)
          - (1, N, L, 6, 1) 或 (1, N, L, 6)（batch_mul=1 时的额外维度，会在此处 squeeze 掉）
        """
        if self.spectrum_chunk_loss is None:
            return torch.tensor(0.0, device=pred_spectrum.device, dtype=pred_spectrum.dtype), {}

        # 兼容 ProteinSpectrumFAR 返回的 batch_mul 维度：shape 可能是 (1, N, L, 6, 1)
        if pred_spectrum.dim() == 5 and pred_spectrum.shape[0] == 1:
            pred_spectrum = pred_spectrum.squeeze(0)
        if true_spectrum.dim() == 5 and true_spectrum.shape[0] == 1:
            true_spectrum = true_spectrum.squeeze(0)

        pred_real, pred_imag = self._spectrum6_to_real_imag_xyz(pred_spectrum)
        true_real, true_imag = self._spectrum6_to_real_imag_xyz(true_spectrum)
        # SpectrumChunkLoss.forward 接口:
        # (B,N,3,L) 或 (N,3,L)
        cum_loss, metrics = self.spectrum_chunk_loss(
            pred_real,
            pred_imag,
            true_real,
            true_imag,
            loss_weight=loss_weight,
        )
        return cum_loss, metrics

    def _unpack_batch(self, batch):
        # 老格式 7-tuple（原始 FFT 数据）
        # 新格式 10-tuple（滑窗 STFT 数据）
        if len(batch) == 7:
            coords, spectrum, rotation_matrices, aatype, aa_mask, esm_features, bihito_token_repr = batch
            last_spectrum = None
            window_size = None
            hop_ratio = None
        else:
            coords, spectrum, rotation_matrices, aatype, aa_mask, esm_features, bihito_token_repr, last_spectrum, window_size, hop_ratio = batch
        # 兼容 DataLoader 额外加了 batch=1 的情况（backbone 只接受无 batch 维）
        coords = self._squeeze_batch1(coords, "coords")
        spectrum = self._squeeze_batch1(spectrum, "spectrum")
        rotation_matrices = self._squeeze_batch1(rotation_matrices, "rotation_matrices")
        aatype = self._squeeze_batch1(aatype, "aatype")
        aa_mask = self._squeeze_batch1(aa_mask, "aa_mask")
        esm_features = self._squeeze_batch1(esm_features, "esm_features")
        if isinstance(bihito_token_repr, torch.Tensor):
            bihito_token_repr = self._squeeze_batch1(bihito_token_repr, "bihito_token_repr")
        if isinstance(last_spectrum, torch.Tensor):
            last_spectrum = self._squeeze_batch1(last_spectrum, "last_spectrum")
        if isinstance(window_size, torch.Tensor):
            window_size = self._squeeze_batch1(window_size, "window_size")
        if isinstance(hop_ratio, torch.Tensor):
            hop_ratio = self._squeeze_batch1(hop_ratio, "hop_ratio")

        return coords, spectrum, rotation_matrices, aatype, aa_mask, esm_features, bihito_token_repr, last_spectrum, window_size, hop_ratio
    
    def training_step(self, batch, batch_idx):
        coords, spectrum, rotation_matrices, aatype, aa_mask, esm_features, bihito_token_repr, last_spectrum, window_size, hop_ratio = self._unpack_batch(batch)
        
        # print("====== Input Shapes ======")
        # print(f"spectrum shape: {spectrum.shape}")
        # print(f"coords shape: {coords.shape}")
        # print(f"rotation_matrices shape: {rotation_matrices.shape}")
        # print(f"aatype shape: {aatype.shape}")
        
        # print(f"esm_features shape: {esm_features.shape}")
        
        # print(f"aa_mask shape: {aa_mask.shape}")
        # print("=========================")
        # # Forward pass
        # mu, logvar, decoded_spectrum = self(packed_graph, coords, spectrum, rotation_matrices, aatype, esm_features)
        reconstructed_spectrum, diffloss = self(
            spectrum,
            coords,
            rotation_matrices,
            aatype,
            aa_mask,
            esm_features,
            bihito_token_repr,
            loss_weight=True,
            return_pred_spectrum=True,
            last_spectrum=last_spectrum,
            window_size=window_size,
            hop_ratio=hop_ratio,
        ) # (num_residues, seq_len, 6, spectrum_token_dim) or (num_residues, seq_len, 3, spectrum_token_dim)
        # shape of reconstructed_spectrum: (num_residues, seq_len, 6, spectrum_token_dim) or (num_residues, seq_len, 3, spectrum_token_dim)
        
        # # Calculate reconstruction loss
        # recon_loss = F.mse_loss(reconstructed_spectrum, spectrum) 
        
        # # Calculate KL loss
        # kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        # # Total loss
        # loss = recon_loss + self.beta * kl_loss
        # 额外频谱损失（若 forward 未返回光谱，如自监督/仅loss模式，会是 None）
        if reconstructed_spectrum is not None:
            spec_cum_loss, spec_metrics = self._compute_spectrum_chunk_losses(
                pred_spectrum=reconstructed_spectrum,
                true_spectrum=spectrum,
                loss_weight=None,
            )
            loss = diffloss + self.spectrum_chunk_loss_scale * spec_cum_loss
        else:
            spec_cum_loss, spec_metrics = torch.tensor(0.0, device=diffloss.device, dtype=diffloss.dtype), {}
            loss = diffloss
        
        # Log losses
        # self.log('train_recon_loss', recon_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('train_diffloss', diffloss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        if self.spectrum_chunk_loss is not None:
            self.log('train_spectrum_chunk_loss', spec_cum_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            for k, v in spec_metrics.items():
                if isinstance(v, torch.Tensor):
                    self.log(f"train_spectrum/{k}", v, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        # self.log('train_kl_loss', kl_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
       
        if self.trainer.is_global_zero and batch_idx % 10 == 0:
            alloc, reserv, peak = _mem()
            print(f"[step {self.global_step}] alloc={alloc:.1f}MB reserved={reserv:.1f}MB peak={peak:.1f}MB")
            torch.cuda.reset_peak_memory_stats()
        return loss
    
    def validation_step(self, batch, batch_idx):
        coords, spectrum, rotation_matrices, aatype, aa_mask, esm_features, bihito_token_repr, last_spectrum, window_size, hop_ratio = self._unpack_batch(batch)
        
        # # Forward pass
        # mu, logvar, decoded_spectrum = self(packed_graph, coords, spectrum, rotation_matrices, aatype, esm_features)
        reconstructed_spectrum, diffloss = self(
            spectrum,
            coords,
            rotation_matrices,
            aatype,
            aa_mask,
            esm_features,
            bihito_token_repr,
            loss_weight=True,
            return_pred_spectrum=True,
            last_spectrum=last_spectrum,
            window_size=window_size,
            hop_ratio=hop_ratio,
        )
        # # Calculate reconstruction loss
        # recon_loss = F.mse_loss(reconstructed_spectrum, spectrum)
        
        # # Calculate KL loss
        # kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        # # Total loss
        # loss = recon_loss + self.beta * kl_loss
        if reconstructed_spectrum is not None:
            spec_cum_loss, spec_metrics = self._compute_spectrum_chunk_losses(
                pred_spectrum=reconstructed_spectrum,
                true_spectrum=spectrum,
                loss_weight=None,
            )
            loss = diffloss + self.spectrum_chunk_loss_scale * spec_cum_loss
        else:
            spec_cum_loss, spec_metrics = torch.tensor(0.0, device=diffloss.device, dtype=diffloss.dtype), {}
            loss = diffloss
        
        # Log losses
        # self.log('val_recon_loss', recon_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_diffloss', diffloss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        if self.spectrum_chunk_loss is not None:
            self.log('val_spectrum_chunk_loss', spec_cum_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            for k, v in spec_metrics.items():
                if isinstance(v, torch.Tensor):
                    self.log(f"val_spectrum/{k}", v, on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        # self.log('val_kl_loss', kl_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        
        return loss

    def test_step(self, batch, batch_idx):
        coords, spectrum, rotation_matrices, aatype, aa_mask, esm_features, bihito_token_repr, last_spectrum, window_size, hop_ratio = self._unpack_batch(batch)
        reconstructed_spectrum, diffloss = self(
            spectrum,
            coords,
            rotation_matrices,
            aatype,
            aa_mask,
            esm_features,
            bihito_token_repr,
            loss_weight=True,
            return_pred_spectrum=True,
            last_spectrum=last_spectrum,
            window_size=window_size,
            hop_ratio=hop_ratio,
        )
        spec_cum_loss, spec_metrics = self._compute_spectrum_chunk_losses(
            pred_spectrum=reconstructed_spectrum,
            true_spectrum=spectrum,
            loss_weight=None,
        )
        loss = diffloss + self.spectrum_chunk_loss_scale * spec_cum_loss
        self.log('test_diffloss', diffloss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        if self.spectrum_chunk_loss is not None:
            self.log('test_spectrum_chunk_loss', spec_cum_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            for k, v in spec_metrics.items():
                if isinstance(v, torch.Tensor):
                    self.log(f"test_spectrum/{k}", v, on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        coords, spectrum, rotation_matrices, aatype, aa_mask, esm_features, bihito_token_repr, last_spectrum, window_size, hop_ratio = self._unpack_batch(batch)
        pred_spectrum, _ = self(
            spectrum,
            coords,
            rotation_matrices,
            aatype,
            aa_mask,
            esm_features,
            bihito_token_repr,
            loss_weight=False,
            return_pred_spectrum=True,
            last_spectrum=last_spectrum,
            window_size=window_size,
            hop_ratio=hop_ratio,
        )
        return pred_spectrum
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.config.lr,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        scheduler = WarmupCosineWithPlateauScheduler(
            optimizer,
            warmup_epochs=self.config.warmup_epochs,
            max_epochs=self.config.max_epochs,
            min_lr=self.config.min_lr,
            warmup_start_lr=self.config.warmup_start_lr,
            plateau_patience=self.config.plateau_patience,
            plateau_factor=self.config.plateau_factor,
            verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss_epoch",
                "interval": "epoch",
                "frequency": 1,
                "reduce_on_plateau": True
            }
        }

# def seed_everything(seed=42):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False

# def main():
#     # Configuration parameters
#     class Config:
#         # Data related
#         train_path = "./dataset/atlas_10_features_fouriered_list_train.pth"
#         val_path = "./dataset/atlas_10_features_fouriered_list_val.pth"
#         batch_size = 1
#         num_workers = 8
        
#         # Model related
#         # input_dim = 1280 + 640 + 640  # ESM + spectrum + amino acid type
#         # hidden_dims = [256, ]  # 4-layer network
#         # num_relation = 7
#         # latent_dim = 64
#         dropout_rate = 0.0
#         # output_dim = 51 * 6
        
#         # Training related
#         lr = 1e-4
#         max_epochs = 500  # Total training epochs
        
#         # Save related
#         ckpt_dir = "./checkpoints/FAR_IPA_encoder_and_decoder/"
        
#         # New parameters
#         beta = 1e-6  # Assumed beta value
        
#     config = Config()
    
#     # Create save directory
#     os.makedirs(config.ckpt_dir, exist_ok=True)
    
#     # Set seed
#     seed_everything(42)
    
#     if int(os.environ.get('LOCAL_RANK', 0)) == 0:  # Only print in main process
#         print(f"\n{'='*50}")
#         print(f"Training with specified train/val lists")
#         print(f"{'='*50}")
    
#     # Initialize data module
#     dm = DataModule(config)
    
#     # Initialize model
#     model = ProteinDynamicsFAR(config)
    
#     # Set checkpoint callback
#     checkpoint_callback = ModelCheckpoint(
#         dirpath=config.ckpt_dir,
#         filename='protein_far-{epoch:02d}-{train_loss:.4f}-{val_loss:.4f}',
#         save_top_k=-1,
#         monitor='val_loss',
#         mode='min'
#     )
    
#     # Initialize trainer
#     # 这里的devices=-1表示会使用所有可用的GPU进行训练。
#     # 在PyTorch Lightning中，devices参数用于指定使用的GPU数目或者具体的GPU编号。
#     # 如果设置为-1，表示使用所有可用的GPU（类似devices='auto'行为）。
#     trainer = pl.Trainer(
#         accelerator='gpu',
#         devices=-1,
#         strategy=DDPStrategy(find_unused_parameters=True),
#         max_epochs=config.max_epochs,
#         callbacks=[checkpoint_callback],
#         precision='bf16',
#         gradient_clip_val=1,
#         accumulate_grad_batches=1,
#         log_every_n_steps=1,
#         sync_batchnorm=True,
#         enable_progress_bar=True,
#         enable_model_summary=True,
#         num_sanity_val_steps=0,
#         check_val_every_n_epoch=1,
#         limit_val_batches=1.0
#     )
#     # Start training
#     trainer.fit(model, dm)
    
#     # Only save final model in main process
#     if int(os.environ.get('LOCAL_RANK', 0)) == 0:
#         final_ckpt = os.path.join(config.ckpt_dir, "final.ckpt")
#         trainer.save_checkpoint(final_ckpt)
#         print(f"Saved final model to {final_ckpt}")

# if __name__ == "__main__":
#     main()