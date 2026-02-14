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
from torch.utils.data import DataLoader
import torch.nn.functional as F
import random
import numpy as np
import math
from typing import Dict, Any, Optional
# Ray Tune imports
from ray import tune, train
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback
from ray.train import ScalingConfig
from ray.train.lightning import RayDDPStrategy
from ray.train.lightning import RayLightningEnvironment
from ray.train.lightning import prepare_trainer

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
        self.amplitude_mse_loss = 5.0


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
    # encoder_depth = 4
    # encoder_num_heads = 8
    # decoder_embed_dim = 512
    # decoder_depth = 4
    # decoder_num_heads = 8
    # buffer_size = 32
    # diffloss_d = 2
    # diffloss_w = 512
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
    predict_xstart = False
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
    using_ray = False
    # devices = -1  # 一般：使用所有可用GPU
    devices = "auto"  # Ray Tune自动配置
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


def create_config(tune_config: Dict[str, Any] = None, base_config: BaseConfig = None, yaml_config_path: Optional[str] = None) -> object:
    """
    创建配置对象，将tune_config和yaml配置的内容合并/覆盖到base_config（BaseConfig实例）中
    
    优先级: tune_config > yaml_config > base_config

    Args:
        tune_config: Ray Tune给定的超参数字典（新定义的参数），优先级最高
        base_config: 已有的BaseConfig实例，或None（此时用默认），优先级最低
        yaml_config_path: YAML配置文件路径，优先级中等

    Returns:
        config对象，参数为三者合并，优先级高的覆盖优先级低的
    """
    # 只要 tune_config 里有的 key，就会覆盖 base_config 的原有值，这是标准写法
    if base_config is None:
        config = BaseConfig()
    else:
        import copy
        config = copy.deepcopy(base_config)
    
    # 首先从YAML文件加载配置（如果提供）
    yaml_config = {}
    if yaml_config_path:
        yaml_config = load_config_from_yaml(yaml_config_path)
        # 将yaml配置应用到config对象
        for key, value in yaml_config.items():
            setattr(config, key, value)
        logging.info(f"已应用YAML配置，覆盖了 {len(yaml_config)} 个参数")
    
    # 然后应用tune_config（优先级最高）
    if tune_config:
        for key, value in tune_config.items():
            setattr(config, key, value)
    
    return config


def get_search_space() -> Dict[str, Any]:
    """
    定义超参数搜索空间
    
    Returns:
        搜索空间字典
    """
    search_space = {
        # 学习率搜索（对数均匀采样）
        "lr": tune.loguniform(1e-5, 1e-3),
        
        # 模型架构超参数
        "encoder_depth": tune.choice([2, 4, 6]),
        "encoder_num_heads": tune.choice([4, 8, 16]),
        "decoder_embed_dim": tune.choice([256, 512, 768]),
        "decoder_depth": tune.choice([2, 4, 6]),
        "decoder_num_heads": tune.choice([4, 8, 16]),
        
        # Buffer size
        "buffer_size": tune.choice([16, 32, 64]),
        
        # Diffloss参数
        "diffloss_d": tune.choice([2, 4, 8]),
        "diffloss_w": tune.choice([256, 512, 1024]),
        
        # 优化器参数
        "warmup_epochs": tune.choice([5, 10, 15]),
        "min_lr": tune.loguniform(1e-7, 1e-5),
        "warmup_start_lr": tune.loguniform(1e-8, 1e-6),
        "plateau_patience": tune.choice([5, 8, 10]),
        "plateau_factor": tune.choice([0.5, 0.7, 0.9]),
        
        # Beta参数（KL loss权重）
        "beta": tune.loguniform(1e-7, 1e-5),
    }
    
    return search_space

def get_default_hyperparameters() -> Dict[str, Any]:
    """
    获取默认超参数（当不使用搜索时）
    
    Returns:
        默认超参数字典
    """
    return {
        "lr": 1e-4,
        "encoder_depth": 4,
        "encoder_num_heads": 8,
        "decoder_embed_dim": 512,
        "decoder_depth": 4,
        "decoder_num_heads": 8,
        "buffer_size": 32,
        "diffloss_d": 2,
        "diffloss_w": 512,
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

        # 尝试获取 Ray Train 上下文，如果失败则说明不在 Ray 环境中
        self.in_ray_context = False
        if self.config.using_ray:
            try:
                ctx = train.get_context()
                self.in_ray_context = True
                # [2] Configure scaling and resource requirements.
                self.scaling_config = ScalingConfig(num_workers=torch.cuda.device_count(), use_gpu=self.use_cuda)
                self.trial_id = str(ctx.get_trial_id()) if ctx else "local"
                logging.info(f"检测到 Ray Train 环境，Trial ID: {self.trial_id}")
            except RuntimeError as e:
                # 不在 Ray Train 环境中
                logging.warning(f"未检测到 Ray Train 环境 ({e})，使用本地模式")
                self.in_ray_context = False
                self.scaling_config = None
                self.trial_id = "no_tune"
        else:
            self.scaling_config = None
            self.trial_id = "no_tune"

    
    def init_basics(self):
        """初始化基础配置"""
        self.run_name = self.config.run_name + "_" + time.strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(self.config.base_dir, self.run_name, f"trial_{self.trial_id}") if self.trial_id != "no_tune" else os.path.join(self.config.base_dir, self.run_name)
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
        self.ray_tune_callback = TuneReportCheckpointCallback(
            metrics={
                "train_loss": "train_loss_epoch",      # ✓ 存在
                "val_loss": "val_loss_epoch",          # ✓ 存在
                "train_diffloss": "train_diffloss_epoch",  # ✓ 存在
                "val_diffloss": "val_diffloss_epoch",      # ✓ 存在
                # "train_recon_loss": "train_recon_loss_epoch",  # ✗ 已注释，不报告
                # "val_recon_loss": "val_recon_loss_epoch",      # ✗ 已注释，不报告
            },
            save_checkpoints=False,
            on="validation_end"
        )

        self.callbacks = [self.checkpoint_callback]
        
        # 学习率监控回调（用于wandb记录）
        if self.config.use_wandb:
            lr_monitor = LearningRateMonitor(logging_interval='step')
            self.callbacks.append(lr_monitor)

        # ✅ 只在 Ray Tune 环境中添加 TuneReportCheckpointCallback
        if self.in_ray_context:
            # 必须放在最后，确保它能获取到所有指标
            self.callbacks.append(self.ray_tune_callback)
            logging.info("已添加 TuneReportCheckpointCallback（Ray Tune 模式）")
        else:
            logging.info("跳过 TuneReportCheckpointCallback（本地训练模式）")
    
    def init_trainer(self):
        """初始化PyTorch Lightning Trainer"""
        logging.info("Initializing Trainer...")
        
        # 配置策略 - 根据实际的 Ray 上下文状态而不是配置值
        if self.in_ray_context and torch.cuda.device_count() > 1:
            strategy = RayDDPStrategy()
            logging.info("使用 RayDDPStrategy")
        elif torch.cuda.device_count() > 1:
            # When find_unused_parameters=True, static_graph must be False,
            # otherwise DDP will still error on unused params.
            use_ckpt = bool(getattr(self.config, "use_checkpoint", False) or getattr(self.config, "denoise_use_checkpoint", False))
            static_graph = use_ckpt and (not bool(getattr(self.config, "find_unused_parameters", False)))
            strategy = DDPStrategy(find_unused_parameters=self.config.find_unused_parameters, static_graph=static_graph)
            logging.info("使用 DDPStrategy")
        else:
            strategy = "auto"
            logging.info("使用 auto strategy")
        
        self.trainer = pl.Trainer(
            accelerator='gpu' if self.use_cuda else 'cpu',
            devices=self.config.devices,
            strategy=strategy,
            max_epochs=self.config.max_epochs,
            callbacks=self.callbacks,
            plugins=[RayLightningEnvironment()] if self.in_ray_context else [LightningEnvironment()],
            enable_checkpointing=False if self.in_ray_context else True,
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
            reload_dataloaders_every_n_epochs=getattr(self.config, 'reload_dataloaders_every_n_epochs', 0),
            deterministic=self.config.deterministic
        )
        
        # 只有在真正的 Ray 环境中才调用 prepare_trainer
        if self.in_ray_context:
            self.trainer = prepare_trainer(self.trainer)
            logging.info("Trainer 已通过 Ray prepare_trainer 准备")
            
    
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

    # def run_raytune(self):
    #     trainer = TorchTrainer(
    #         self.run,
    #         scaling_config=self.scaling_config,
    #         train_loop_config=self
    #         # [3a] If running in a multi-node cluster, this is where you
    #         # should configure the run's persistent storage that is accessible
    #         # across all worker nodes.
    #         # run_config=ray.train.RunConfig(storage_path="s3://..."),
    #     )
    #     result: ray.train.Result = trainer.fit()

def raytune_entrypoint(tune_config: Dict[str, Any], yaml_config_path: Optional[str] = None) -> object:
    """
    Ray Tune入口函数，用于训练
    
    Args:
        tune_config: Ray Tune给定的超参数字典
        yaml_config_path: YAML配置文件路径（可选）
        
    Returns:
        None
    """
    config = create_config(tune_config, BaseConfig(), yaml_config_path=yaml_config_path)
    logging.info("=" * 80)
    logging.info("Training Configuration:")
    for key, value in vars(config).items():
        if not key.startswith('_'):
            logging.info(f"  {key}: {value}")
    logging.info("=" * 80)
    trainer = FARTrainer(config)
    trainer.run()

def run_hyperparameter_optimization(
    num_samples: int = 10,
    num_epochs: int = 50,
    gpus_per_trial: int = 1,
    cpus_per_trial: int = 4,
    max_concurrent_trials: int = 4,
    grace_period: int = 10,
    reduction_factor: int = 2,
    use_asha: bool = True,
    yaml_config_path: Optional[str] = None,
):
    """
    运行超参数优化
    
    Args:
        num_samples: 采样的超参数组合数量
        num_epochs: 每个trial的最大训练epoch数
        gpus_per_trial: 每个trial使用的GPU数量
        cpus_per_trial: 每个trial使用的CPU数量
        max_concurrent_trials: 最大并发trial数量
        grace_period: ASHA调度器的grace period
        reduction_factor: ASHA调度器的reduction factor
        use_asha: 是否使用ASHA调度器
        yaml_config_path: YAML配置文件路径（可选）
    """
    # 更新BaseConfig中的max_epochs
    BaseConfig.max_epochs = num_epochs
    
    search_space = get_search_space() # Dict
    
    if use_asha:
        scheduler = ASHAScheduler(
            max_t=num_epochs,
            grace_period=grace_period,
            reduction_factor=reduction_factor
        )
    else:
        scheduler = None
    
    # 创建一个包装函数，将yaml_config_path传递给raytune_entrypoint
    def train_fn_with_yaml(tune_config):
        return raytune_entrypoint(tune_config, yaml_config_path=yaml_config_path)
    
    train_fn_with_resources = tune.with_resources(
        train_fn_with_yaml,
        resources={"CPU": cpus_per_trial, "GPU": gpus_per_trial}
    )
    
    reporter = CLIReporter(
        parameter_columns=[
            "lr", 
            "encoder_depth", 
            "decoder_embed_dim",
            "buffer_size",
            "diffloss_w"
        ],
        metric_columns=[
            "train_loss", 
            "val_loss", 
            "training_iteration"
        ],
    )
    
    # 创建Tuner
    tuner = tune.Tuner(
        train_fn_with_resources,
        param_space=search_space,
        tune_config=tune.TuneConfig(
            metric="val_loss",
            mode="min",
            num_samples=num_samples,
            scheduler=scheduler,
            max_concurrent_trials=max_concurrent_trials,
        ),
        run_config=tune.RunConfig(
            name=BaseConfig.run_name,
            storage_path=BaseConfig.base_dir,
            checkpoint_config=tune.CheckpointConfig(
                num_to_keep=BaseConfig.save_top_k,
                checkpoint_score_attribute="val_loss",
                checkpoint_score_order="min",
            ),
            progress_reporter=reporter,
            verbose=1,
        ),
    )
    
    results = tuner.fit()
    
    best_result = results.get_best_result(metric="val_loss", mode="min")
    
    print("\n" + "="*80)
    print("超参数优化完成!")
    print("="*80)
    print(f"\n最佳超参数配置:")
    for key, value in best_result.config.items():
        print(f"  {key}: {value}")
    
    print(f"\n最佳验证损失: {best_result.metrics['val_loss']:.6f}")
    print(f"最佳检查点路径: {best_result.checkpoint}")
    print("="*80 + "\n")
    
    return results, best_result

def test_single_config(yaml_config_path: Optional[str] = None):
    """
    测试单个配置（不进行超参数搜索）
    用于调试和验证训练流程
    
    Args:
        yaml_config_path: YAML配置文件路径（可选）
    """
    print("\n" + "="*80)
    print("测试单个配置（不进行超参数搜索）")
    print("="*80 + "\n")
    
    test_config = get_default_hyperparameters()
    # 在 test 模式下关闭 Ray，使用本地训练模式
    test_config['using_ray'] = False
    
    raytune_entrypoint(test_config, yaml_config_path=yaml_config_path)
    
    print("\n单配置测试完成!\n")


def main():
    """主函数"""
    
    LOG_FORMAT = "%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
    logging.basicConfig(
        format=LOG_FORMAT,
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Ray Tune超参数优化 - FAR蛋白质动力学模型")
    parser.add_argument("--mode", type=str, default="tune", choices=["tune", "test"],
                        help="运行模式: 'tune'进行超参数搜索, 'test'测试单个配置")
    parser.add_argument("--num_samples", type=int, default=10,
                        help="超参数采样数量")
    parser.add_argument("--num_epochs", type=int, default=50,
                        help="每个trial的最大训练epoch数")
    parser.add_argument("--gpus_per_trial", type=int, default=1,
                        help="每个trial使用的GPU数量")
    parser.add_argument("--cpus_per_trial", type=int, default=4,
                        help="每个trial使用的CPU数量")
    parser.add_argument("--max_concurrent_trials", type=int, default=4,
                        help="最大并发trial数量")
    parser.add_argument("--grace_period", type=int, default=10,
                        help="ASHA调度器的grace period")
    parser.add_argument("--reduction_factor", type=int, default=2,
                        help="ASHA调度器的reduction factor")
    parser.add_argument("--no_asha", action="store_true",
                        help="禁用ASHA调度器")
    # 断点续训与WandB恢复
    parser.add_argument("--resume_ckpt", type=str, default=None,
                        help="从指定checkpoint路径继续训练 (.ckpt)")
    parser.add_argument("--wandb_id", type=str, default=None,
                        help="恢复到同一个WandB run的ID（可选）")
    parser.add_argument("--config", type=str, default=None,
                        help="YAML配置文件路径（可选），配置文件中的参数会覆盖BaseConfig中的默认值")
    
    args = parser.parse_args()

    os.makedirs(BaseConfig.base_dir, exist_ok=True)
    
    # 将命令行断点续训参数注入配置
    if args.resume_ckpt is not None and len(args.resume_ckpt) > 0:
        BaseConfig.load_checkpoint_path = args.resume_ckpt
    if args.wandb_id is not None and len(args.wandb_id) > 0:
        BaseConfig.wandb_id = args.wandb_id

    if args.mode == "test":
        # 测试模式：运行单个配置
        test_single_config(yaml_config_path=args.config)
    else:
        # 超参数优化模式
        results, best_result = run_hyperparameter_optimization(
            num_samples=args.num_samples,
            num_epochs=args.num_epochs,
            gpus_per_trial=args.gpus_per_trial,
            cpus_per_trial=args.cpus_per_trial,
            max_concurrent_trials=args.max_concurrent_trials,
            grace_period=args.grace_period,
            reduction_factor=args.reduction_factor,
            use_asha=not args.no_asha,
            yaml_config_path=args.config,
        )
        
        # 保存最佳配置
        best_config_path = os.path.join(BaseConfig.base_dir, BaseConfig.run_name, "best_config.txt")
        with open(best_config_path, 'w') as f:
            f.write("最佳超参数配置:\n")
            f.write("="*80 + "\n")
            for key, value in best_result.config.items():
                f.write(f"{key}: {value}\n")
            f.write("\n")
            f.write(f"最佳验证损失: {best_result.metrics['val_loss']:.6f}\n")
            f.write(f"最佳检查点: {best_result.checkpoint}\n")
        
        print(f"\n最佳配置已保存到: {best_config_path}")


if __name__ == "__main__":
    main()

