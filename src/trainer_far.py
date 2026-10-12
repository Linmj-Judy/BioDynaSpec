import os
import logging
import time
import yaml
from pathlib import Path

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.plugins.environments import LightningEnvironment
from pytorch_lightning.loggers import WandbLogger
import random
import numpy as np
from typing import Dict, Any, Optional

from main_model import ProteinDynamicsFAR
from data.datamodule import DataModule
from visualize.utils import plot_loss_curves_from_ckpt_dir

# Set your own WandB API key via environment variable: export WANDB_API_KEY="your_key"
# os.environ['WANDB_API_KEY'] = 'your_wandb_api_key'
    
class SpectrumLossWeight:
    """
    对应 src/model/loss/spectrum_loss.py 里读取的 config.spectrum_loss.weight.*
    必须是“可点号访问”的对象，不能是 dict。
    """
    def __init__(self):
        self.mse_loss = 0.1  # 降低权重，因为这部分对绝对位置太敏感
        self.mae_loss = 1.0
        self.smooth_l1_loss = 1.0
        self.log_spectral_l1_loss = 10.0
        self.cosine_similarity_loss = 5.0
        self.spectral_convergence_loss = 5.0
        # 5.0 -> 2.0. The term is now scale-normalised
        # (`_compute_relative_amplitude_mse_loss`), which changes its magnitude
        # by 1/mean(A_true^2) ≈ 7.2x on ATLAS data, so the old weight no longer
        # means what it did. 2.0 matches the successor project
        # (MacroMolecularDynamics configs/configs_base.py) and keeps the
        # amplitude term from dominating the summed loss in the large-error
        # regime.
        self.amplitude_mse_loss = 2.0


class SpectrumLossConfig:
    """提供 config.spectrum_loss.eps 与 config.spectrum_loss.weight.*"""
    def __init__(self):
        self.eps = 1e-6
        self.weight = SpectrumLossWeight()


class BaseConfig:
    """基础配置类，包含不需要优化的固定参数"""
    # 基础配置
    run_name = "BioDynaSpec"
    base_dir = os.environ.get("BIODYNASPEC_RUNS_DIR", "./runs")
    seed = 42
    deterministic = True
    
    # 数据相关
    train_path = os.environ.get("BIODYNASPEC_TRAIN_PATH", "./dataset/fft_window2500/atlas_10_features_fouriered_list_train_normalized.pth")
    val_path = os.environ.get("BIODYNASPEC_VAL_PATH", "./dataset/fft_window2500/atlas_10_features_fouriered_list_val_normalized.pth")
    bihito_repr_dir = os.environ.get("BIODYNASPEC_BIHITO_DIR", "./dataset/bihito_repr_cache/atlas")

    batch_size = 1
    num_workers = 32
    
    # 模型配置
    spectrum_token_dim = 1
    num_xyz_channels = 3
    use_complex_spectrum = True
    aatype_vocab_size = 21
    esm_dim = 1280
    with_irfc=True
    # Architecture knobs consumed unconditionally by main_model.py:164-171.
    # These MUST have defaults here: create_config() only overlays keys that the
    # YAML/tune actually provide, and both the shipped YAML and BaseConfig had
    # these commented out -> `ProteinDynamicsFAR(cfg)` raised AttributeError, so
    # the documented `bash scripts/pretrain.sh` path could not start at all.
    # Values mirror protein_spectrum_far_base() (paper: d_model=768, 12 heads).
    encoder_depth = 12
    encoder_num_heads = 16
    decoder_embed_dim = 768
    decoder_depth = 12
    decoder_num_heads = 12
    buffer_size = 32
    diffloss_d = 3
    diffloss_w = 1024
    num_sampling_steps = '50'
    diffusion_batch_mul = 1
    dropout_rate = 0.0
    denoise_seq_chunk_size = 2560  # 分块处理频率维度以降低显存
    denoise_block_type = "residue"  # "residue"=残基注意力, "mlp"=无残基注意力
    denoise_use_checkpoint = True  # 扩散网络激活检查点，降低显存
    use_checkpoint = True  # 编码器/解码器激活检查点，降低显存
    max_spectrum_len = 2560  # 限制频谱长度，避免超长序列导致OOM

    # diffusion loss parameters with safe defaults
    learn_sigma = False
    predict_xstart = True
    sigma_small = False
    noise_schedule = 'cosine'
    use_kl = False
    rescale_learned_sigmas = False
    diffusion_steps = 1000
    
    # Training related
    max_epochs = 50  # 每个trial的最大epoch数（比完整训练少）
    lr = 1e-4
    warmup_epochs = 10
    min_lr = 1e-6
    warmup_start_lr = 1e-7
    plateau_patience = 8
    plateau_factor = 0.7
    beta = 1e-6
    
    # Trainer configuration
    # devices = -1  # 一般：使用所有可用GPU
    devices = "auto"
    sync_batchnorm = True

    precision = 'bf16'
    gradient_clip_val = 1.0
    accumulate_grad_batches = 1
    log_every_n_steps = 1
    
    # 强制每轮重新加载数据加载器以实现随机子采样更新
    reload_dataloaders_every_n_epochs = 1
    
    num_sanity_val_steps = 0
    check_val_every_n_epoch = 1
    limit_val_batches = 1.0
    limit_train_batches = 1.0  # set <1.0 to bound a smoke run
    find_unused_parameters = True
    
    # 检查点相关
    # save_top_k = -1  # Save all checkpoints
    save_top_k = 10  # Save the best 10 checkpoints

    # monitor_metric = 'val_loss'
    # monitor_mode = 'min'
    monitor_metric = 'train_loss'
    monitor_mode = 'min'
    checkpoint_interval = 1  # Save checkpoint every epoch
    
    # WandB配置
    use_wandb = True  # Whether to use wandb
    wandb_project = "BioMacroDynamicsFAR"  # wandb project name
    wandb_id = None  # wandb run id for resuming training
    
    # 断点续训
    load_checkpoint_path = None  # 指定上一次训练的.ckpt路径以继续训练
    # 兼容旧ckpt：只加载权重（不恢复优化器/调度器）
    load_weights_only = False
    load_state_strict = False
    
    # 按步保存ckpt（None=按epoch保存）
    save_every_n_steps = 1

    # 数据集子采样（用于快速实验或大步长更新）
    use_dataset_subsample = True  # 是否启用数据集子采样
    subsample_ratio = 0.1  # 每次采样的比例 (0.0 ~ 1.0)
    
    # Self-forcing训练模式（弥合训练和推理gap）
    use_self_forcing = False  # 是否启用self-forcing模式
    self_forcing_schedule = None  # None=所有step都计算loss，或指定step列表如[0, 5, 10, 20, ...]

    # ---------------------------
    # SpectrumChunkLoss 相关配置
    # ---------------------------
    enable_spectrum_chunk_loss = True          # 是否启用 SpectrumChunkLoss（额外频谱损失）
    spectrum_chunk_loss_scale = 1.0            # 将 SpectrumChunkLoss 的 cum_loss 乘以该系数后加到 diffloss 上

    # 让其支持点号访问：config.spectrum_loss.eps / config.spectrum_loss.weight.mse_loss
    spectrum_loss = SpectrumLossConfig()


def load_config_from_yaml(yaml_path: str) -> Dict[str, Any]:
    """
    从YAML文件加载配置
    
    Args:
        yaml_path: YAML配置文件路径
        
    Returns:
        配置字典
    """
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {yaml_path}")
    
    with open(yaml_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    
    if config_dict is None:
        config_dict = {}
    
    logging.info(f"从YAML文件加载配置: {yaml_path}")
    return config_dict


def create_config(base_config: BaseConfig = None, yaml_config_path: Optional[str] = None) -> object:
    """
    创建配置对象，将yaml配置的内容合并/覆盖到base_config（BaseConfig实例）中

    优先级: yaml_config > base_config

    Args:
        base_config: 已有的BaseConfig实例，或None（此时用默认），优先级最低
        yaml_config_path: YAML配置文件路径，优先级更高

    Returns:
        config对象
    """
    if base_config is None:
        config = BaseConfig()
    else:
        import copy
        config = copy.deepcopy(base_config)

    # 从YAML文件加载配置（如果提供）
    if yaml_config_path:
        yaml_config = load_config_from_yaml(yaml_config_path)
        for key, value in yaml_config.items():
            setattr(config, key, value)
        logging.info(f"已应用YAML配置，覆盖了 {len(yaml_config)} 个参数")

    return config

def get_default_hyperparameters() -> Dict[str, Any]:
    """
    获取默认超参数（当不使用搜索时）
    
    Returns:
        默认超参数字典
    """
    # NOTE: this dict takes HIGHEST priority in create_config(), so it — not
    # BaseConfig — determines the architecture that `pretrain.sh` actually
    # trains. It previously held a small debug config (4/8/512/4/8/2/512, 158M
    # params) which silently disagreed with paper Sec. 4.2 (d_model=768,
    # 12 heads). Values below mirror protein_spectrum_far_base() and the paper.
    return {
        "lr": 1e-4,
        "encoder_depth": 12,
        "encoder_num_heads": 16,
        "decoder_embed_dim": 768,
        "decoder_depth": 12,
        "decoder_num_heads": 12,
        "buffer_size": 32,
        "diffloss_d": 3,
        "diffloss_w": 1024,
        "warmup_epochs": 10,
        "min_lr": 1e-6,
        "warmup_start_lr": 1e-7,
        "plateau_patience": 8,
        "plateau_factor": 0.7,
        "beta": 1e-6,
    }



class SafeModelCheckpoint(ModelCheckpoint):
    """为缺失的 val_loss 提供 'none' 占位，避免格式化失败。"""

    def format_checkpoint_name(self, metrics, filename=None, ver=None):
        safe_metrics = dict(metrics or {})
        val = safe_metrics.get("val_loss", None)
        if val is None:
            safe_metrics["val_loss_str"] = "none"
        else:
            try:
                safe_metrics["val_loss_str"] = f"{float(val):.4f}"
            except Exception:
                safe_metrics["val_loss_str"] = "none"
        return super().format_checkpoint_name(safe_metrics, filename=filename, ver=ver)


class FARTrainer:
    """FAR训练器主类"""
    
    def __init__(self, config):
        self.config = config
        self.init_env()
        self.init_basics()
        self.init_logger()
        self.init_data()
        self.init_model()
        self.init_callbacks()
        self.init_trainer()
    
    def init_env(self):
        """初始化环境"""
        logging.info("初始化训练环境...")
        
        # 设置随机种子
        self.seed_everything(self.config.seed)
        
        # 检测设备
        self.use_cuda = torch.cuda.device_count() > 0
        if self.use_cuda:
            logging.info(f"检测到 {torch.cuda.device_count()} 个GPU设备")
        else:
            logging.info("使用CPU训练")
        # 配置精度
        self.config_precision()


    def init_basics(self):
        """初始化基础配置"""
        self.run_name = self.config.run_name + "_" + time.strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(self.config.base_dir, self.run_name)
        self.checkpoint_dir = os.path.join(self.run_dir, "checkpoints")
        self.wandb_dir = os.path.join(self.run_dir, "wandb")
        
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        if self.config.use_wandb:
            os.makedirs(self.wandb_dir, exist_ok=True)
        
        logging.info(f"运行名称: {self.run_name}")
        logging.info(f"运行目录: {self.run_dir}")
        logging.info(f"检查点目录: {self.checkpoint_dir}")
        if self.config.use_wandb:
            logging.info(f"WandB目录: {self.wandb_dir}")
    
    def init_logger(self):
        """初始化WandB Logger（离线模式）"""
        self.logger = None
        if self.config.use_wandb:
            # 只在主进程中初始化
            local_rank = int(os.environ.get('LOCAL_RANK', 0))
            if local_rank == 0:
                logging.info("初始化WandB Logger（离线模式）...")
                
                # 设置wandb为离线模式
                os.environ['WANDB_MODE'] = 'offline'
                os.environ['WANDB_DIR'] = self.wandb_dir
                
                # 下面将config也写进wandb logger，让wandb记录所有配置信息
                self.logger = WandbLogger(
                    name=self.run_name,
                    save_dir=self.wandb_dir,
                    project=self.config.wandb_project,
                    log_model=False,  # 不自动上传模型
                    offline=True,  # 离线模式
                    id=self.config.wandb_id if hasattr(self.config, 'wandb_id') else None,
                    config=vars(self.config)  # 直接保存config，等价于wandb.init(config=...)效果
                )
                
                self.logger.experiment.config.update(vars(self.config))
                
                logging.info(f"WandB初始化完成（离线模式），日志保存在: {self.wandb_dir}")
                logging.info(f"训练完成后，可以使用 'wandb sync {self.wandb_dir}' 上传日志")
            else:
                logging.info("非主进程，跳过WandB初始化")
    
    def init_data(self):
        """初始化数据模块"""
        logging.info("Initializing Data Module...")
        
        # 传递子采样配置
        if hasattr(self.config, 'use_dataset_subsample') and self.config.use_dataset_subsample:
            logging.info(f"启用数据集子采样: ratio={self.config.subsample_ratio}")
            
        self.data_module = DataModule(self.config)
    
    def init_model(self):
        """初始化模型"""
        logging.info("Initializing Model...")
        self.model = ProteinDynamicsFAR(self.config)
    
    def init_callbacks(self):
        """初始化回调函数"""
        logging.info("Initializing Callbacks...")
        
        # 检查点回调
        ckpt_kwargs = dict(
            dirpath=self.checkpoint_dir,
            filename='protein_far-{epoch:02d}-{train_loss:.4f}-val{val_loss_str}',
            save_top_k=self.config.save_top_k,
            monitor=self.config.monitor_metric,
            mode=self.config.monitor_mode,
            save_last=True,
        )
        if getattr(self.config, "save_every_n_steps", None):
            ckpt_kwargs["every_n_train_steps"] = int(self.config.save_every_n_steps)
            ckpt_kwargs["every_n_epochs"] = None
        else:
            ckpt_kwargs["every_n_epochs"] = self.config.checkpoint_interval
        self.checkpoint_callback = SafeModelCheckpoint(**ckpt_kwargs)

        self.callbacks = [self.checkpoint_callback]

        # 学习率监控回调（用于wandb记录）
        if self.config.use_wandb:
            lr_monitor = LearningRateMonitor(logging_interval='step')
            self.callbacks.append(lr_monitor)

    @staticmethod
    def _resolve_device_count(devices, accelerator: str) -> int:
        """Number of devices PL will actually use, derived from `config.devices`.

        The strategy choice must follow the config, not the machine's raw GPU
        count — otherwise `devices: 1` on a multi-GPU node silently builds a
        DDPStrategy. Mirrors the forms Lightning accepts: an int, -1 (all),
        "auto"/None, or an explicit list of indices.
        """
        if accelerator != "gpu":
            n = devices
            if isinstance(n, (list, tuple)):
                return max(1, len(n))
            if n in (None, "auto", -1):
                return 1
            try:
                return max(1, int(n))
            except (TypeError, ValueError):
                return 1

        if isinstance(devices, (list, tuple)):
            return max(1, len(devices))
        if devices in (None, "auto", -1):
            return max(1, torch.cuda.device_count())
        try:
            return max(1, int(devices))
        except (TypeError, ValueError):
            # Unrecognised value: let Lightning interpret it.
            return max(1, torch.cuda.device_count())

    def init_trainer(self):
        """初始化PyTorch Lightning Trainer"""
        logging.info("Initializing Trainer...")

        accelerator = 'gpu' if self.use_cuda else 'cpu'
        n_devices = self._resolve_device_count(self.config.devices, accelerator)
        logging.info(f"devices={self.config.devices!r} -> {n_devices} device(s)")

        # 配置策略 - 由 config.devices 决定，而不是机器上的 GPU 数量
        if n_devices > 1:
            # When find_unused_parameters=True, static_graph must be False,
            # otherwise DDP will still error on unused params.
            use_ckpt = bool(getattr(self.config, "use_checkpoint", False) or getattr(self.config, "denoise_use_checkpoint", False))
            static_graph = use_ckpt and (not bool(getattr(self.config, "find_unused_parameters", False)))
            strategy = DDPStrategy(find_unused_parameters=self.config.find_unused_parameters, static_graph=static_graph)
            logging.info("使用 DDPStrategy")
        else:
            strategy = "auto"
            logging.info("使用 auto strategy（单卡）")

        self.trainer = pl.Trainer(
            accelerator=accelerator,
            devices=self.config.devices,
            strategy=strategy,
            max_epochs=self.config.max_epochs,
            callbacks=self.callbacks,
            plugins=[LightningEnvironment()],
            enable_checkpointing=True,
            logger=self.logger,  # WandB logger
            precision=self.config.precision,
            gradient_clip_val=self.config.gradient_clip_val,
            accumulate_grad_batches=self.config.accumulate_grad_batches,
            log_every_n_steps=self.config.log_every_n_steps,
            sync_batchnorm=self.config.sync_batchnorm,
            enable_progress_bar=True,
            enable_model_summary=True,
            num_sanity_val_steps=self.config.num_sanity_val_steps,
            check_val_every_n_epoch=self.config.check_val_every_n_epoch,
            limit_val_batches=self.config.limit_val_batches,
            limit_train_batches=getattr(self.config, "limit_train_batches", 1.0),
            reload_dataloaders_every_n_epochs=getattr(self.config, 'reload_dataloaders_every_n_epochs', 0),
            deterministic=self.config.deterministic
        )


    def seed_everything(self, seed: int = 42):
        """Setting all random seeds"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if self.config.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
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

    def run(self):
        """Running Training"""
        logging.info("=" * 80)
        logging.info("Starting Training")
        logging.info("=" * 80)
        
        # Resume from checkpoint (if specified)
        ckpt_path = self.config.load_checkpoint_path if hasattr(self.config, 'load_checkpoint_path') else None
        if ckpt_path and getattr(self.config, "load_weights_only", True):
            logging.info(f"Loading weights only (strict={getattr(self.config, 'load_state_strict', False)}): {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location="cpu")
            state_dict = ckpt.get("state_dict", ckpt)
            missing, unexpected = self.model.load_state_dict(
                state_dict, strict=getattr(self.config, "load_state_strict", False)
            )
            if missing:
                logging.warning(f"Missing keys: {missing}")
            if unexpected:
                logging.warning(f"Unexpected keys: {unexpected}")
            ckpt_path = None
        
        # Start Training
        self.trainer.fit(
            self.model, 
            self.data_module,
            ckpt_path=ckpt_path
        )
        
        # Plot loss curves
        logging.info(f"Plotting loss curves...")
        plot_loss_curves_from_ckpt_dir(self.checkpoint_dir)
        
        # Save final model
        if int(os.environ.get('LOCAL_RANK', 0)) == 0:
            final_ckpt = os.path.join(self.checkpoint_dir, "final.ckpt")
            self.trainer.save_checkpoint(final_ckpt)
            logging.info(f"训练完成！最终模型已保存至: {final_ckpt}")

def train_single_config(yaml_config_path: Optional[str] = None):
    """
    用一份配置跑一次训练（论文的默认入口）。

    Args:
        yaml_config_path: YAML配置文件路径（可选），其键覆盖 BaseConfig。
    """
    print("\n" + "="*80)
    print("Training (single config)")
    print("="*80 + "\n")

    # get_default_hyperparameters() takes priority over BaseConfig in
    # create_config(), so it — not BaseConfig — determines the architecture.
    config = create_config(
        BaseConfig(), yaml_config_path=yaml_config_path
    )
    for key, value in get_default_hyperparameters().items():
        setattr(config, key, value)

    logging.info("=" * 80)
    logging.info("Training Configuration:")
    for key, value in vars(config).items():
        if not key.startswith('_'):
            logging.info(f"  {key}: {value}")
    logging.info("=" * 80)

    FARTrainer(config).run()

    print("\nTraining finished.\n")


def main():
    """主函数"""
    
    LOG_FORMAT = "%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
    logging.basicConfig(
        format=LOG_FORMAT,
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    import argparse
    
    parser = argparse.ArgumentParser(
        description="BioDynaSpec training - FAR protein dynamics model"
    )
    parser.add_argument("--config", type=str, default=None,
                        help="YAML config path (optional); its keys override BaseConfig")
    # 断点续训与WandB恢复
    parser.add_argument("--resume_ckpt", type=str, default=None,
                        help="Resume from this checkpoint (.ckpt)")
    parser.add_argument("--wandb_id", type=str, default=None,
                        help="Resume logging into this existing WandB run id (optional)")

    args = parser.parse_args()

    os.makedirs(BaseConfig.base_dir, exist_ok=True)

    # 将命令行断点续训参数注入配置
    if args.resume_ckpt is not None and len(args.resume_ckpt) > 0:
        BaseConfig.load_checkpoint_path = args.resume_ckpt
    if args.wandb_id is not None and len(args.wandb_id) > 0:
        BaseConfig.wandb_id = args.wandb_id

    train_single_config(yaml_config_path=args.config)


if __name__ == "__main__":
    main()
