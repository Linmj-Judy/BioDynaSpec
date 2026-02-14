"""
配置示例文件

展示如何创建自定义配置并使用 FARTrainer
"""

def create_custom_config():
    """
    创建自定义配置
    
    可以根据实验需求修改此函数中的配置
    """
    class Config:
        # ====================================================================
        # 基础配置
        # ====================================================================
        run_name = "FAR_experiment_v1"  # 修改为你的实验名称
        base_dir = "./runs"
        seed = 42  # 随机种子，保证可复现性
        deterministic = True  # 确定性训练（可能稍慢）
        
        # ====================================================================
        # 数据配置
        # ====================================================================
        train_path = "./dataset/atlas_10_features_fouriered_list_train.pth"
        val_path = "./dataset/atlas_10_features_fouriered_list_val.pth"
        
        batch_size = 1  # 每个GPU的批次大小
        num_workers = 0  # 数据加载线程数（0表示主线程加载）
        
        # ====================================================================
        # 模型架构配置
        # ====================================================================
        # 频谱相关
        spectrum_token_dim = 1  # 频谱token维度
        num_xyz_channels = 3  # 坐标通道数
        use_complex_spectrum = True  # 是否使用复数频谱
        
        # 序列相关
        aatype_vocab_size = 21  # 氨基酸类型数量（20种标准氨基酸 + 1个特殊token）
        esm_dim = 1280  # ESM特征维度
        
        # 编码器配置
        encoder_depth = 4  # 编码器层数
        encoder_num_heads = 8  # 编码器注意力头数
        
        # 解码器配置
        decoder_embed_dim = 512  # 解码器嵌入维度
        decoder_depth = 4  # 解码器层数
        decoder_num_heads = 8  # 解码器注意力头数
        
        # 其他模型参数
        buffer_size = 32  # 缓冲区大小
        diffloss_d = 2  # 扩散损失参数d
        diffloss_w = 512  # 扩散损失参数w
        num_sampling_steps = '100'  # 采样步数
        diffusion_batch_mul = 1  # 扩散批次倍数
        dropout_rate = 0.0  # Dropout率
        denoise_seq_chunk_size = 2560  # 分块处理频率维度以降低显存
        denoise_block_type = "residue"  # "residue" or "mlp"
        denoise_use_checkpoint = True  # 扩散网络激活检查点
        use_checkpoint = True  # 编码器/解码器激活检查点
        max_spectrum_len = 2560  # 限制频谱长度，避免超长序列导致OOM
        
        # ====================================================================
        # 训练超参数
        # ====================================================================
        lr = 1e-4  # 初始学习率
        max_epochs = 500  # 最大训练轮数
        
        # 学习率调度器参数
        warmup_epochs = 10  # 预热轮数
        min_lr = 1e-6  # 最小学习率
        warmup_start_lr = 1e-7  # 预热起始学习率
        plateau_patience = 8  # 平台期耐心值（多少个epoch不改善后降低学习率）
        plateau_factor = 0.7  # 平台期学习率衰减因子
        
        # 损失权重
        beta = 1e-6  # KL散度损失权重（如果使用VAE）
        
        # ====================================================================
        # PyTorch Lightning Trainer 配置
        # ====================================================================
        devices = -1  # GPU设备配置：-1表示使用所有可用GPU
                      # 也可以设置为具体数字：1, 2, 4等
                      # 或GPU ID列表：[0, 1, 2]
        
        precision = 'bf16'  # 训练精度：'bf16', '16', '32'
                           # bf16: bfloat16 (推荐，A100/H100等新GPU)
                           # 16: float16 (V100等老GPU)
                           # 32: float32 (最慢但最稳定)
        
        gradient_clip_val = 1.0  # 梯度裁剪值（防止梯度爆炸）
        accumulate_grad_batches = 1  # 梯度累积步数（增大有效batch size）
        
        log_every_n_steps = 1  # 每N步记录一次日志
        sync_batchnorm = True  # 多GPU时同步BatchNorm
        
        num_sanity_val_steps = 0  # 训练前的验证步数（用于检查）
        check_val_every_n_epoch = 1  # 每N个epoch验证一次
        limit_val_batches = 1.0  # 验证时使用的batch比例（1.0表示全部）
        
        find_unused_parameters = True  # DDP时查找未使用参数
        
        # ====================================================================
        # 检查点配置
        # ====================================================================
        save_top_k = -1  # 保存检查点数量：-1表示保存所有，N表示保存最好的N个
        monitor_metric = 'val_loss'  # 监控的指标
        monitor_mode = 'min'  # 监控模式：'min'或'max'
        checkpoint_interval = 1  # 每N个epoch保存一次检查点
        
        # 恢复训练（可选）
        # load_checkpoint_path = None  # 检查点路径，None表示从头训练
        # load_checkpoint_path = "/path/to/checkpoint.ckpt"
        
        # ====================================================================
        # WandB 配置
        # ====================================================================
        use_wandb = True  # 是否使用WandB记录训练过程
        wandb_project = "ProteinDynamics_FAR"  # WandB项目名称
        wandb_id = None  # WandB运行ID（恢复训练时使用）
        
        # 如果不想使用WandB，设置：
        # use_wandb = False
        
    return Config()


def create_debug_config():
    """
    调试配置 - 用于快速测试
    
    使用更小的模型和更少的训练步数
    """
    config = create_custom_config()
    
    # 修改为调试模式
    config.run_name = "FAR_debug"
    config.max_epochs = 2  # 只训练2个epoch
    config.batch_size = 1
    config.num_workers = 0
    config.log_every_n_steps = 1
    config.limit_val_batches = 10  # 只验证10个batch
    config.use_wandb = False  # 调试时关闭wandb
    
    # 使用更小的模型
    config.encoder_depth = 2
    config.decoder_depth = 2
    
    return config


def create_production_config():
    """
    生产配置 - 用于正式训练
    
    使用更大的模型和完整的训练设置
    """
    config = create_custom_config()
    
    # 优化配置
    config.run_name = "FAR_production_large"
    config.batch_size = 2  # 如果GPU内存足够可以增大
    config.accumulate_grad_batches = 2  # 有效batch_size = 2 * 2 = 4
    config.num_workers = 4  # 使用多线程加载数据
    
    # 使用更大的模型
    config.encoder_depth = 6
    config.decoder_depth = 6
    config.decoder_embed_dim = 768
    
    # 更长的训练
    config.max_epochs = 1000
    config.warmup_epochs = 20
    
    return config


def create_small_model_config():
    """
    小模型配置 - 用于资源受限的情况
    """
    config = create_custom_config()
    
    config.run_name = "FAR_small_model"
    config.encoder_depth = 2
    config.decoder_depth = 2
    config.encoder_num_heads = 4
    config.decoder_num_heads = 4
    config.decoder_embed_dim = 256
    
    # 使用float16以节省内存
    config.precision = '16'
    config.gradient_clip_val = 0.5
    
    return config


if __name__ == "__main__":
    """
    使用示例
    """
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))
    
    from trainer_far import FARTrainer
    
    # 选择配置
    print("可用配置:")
    print("1. custom - 自定义配置")
    print("2. debug - 调试配置（快速测试）")
    print("3. production - 生产配置（完整训练）")
    print("4. small - 小模型配置（资源受限）")
    
    choice = input("请选择配置 (1-4): ").strip()
    
    if choice == "1":
        config = create_custom_config()
    elif choice == "2":
        config = create_debug_config()
    elif choice == "3":
        config = create_production_config()
    elif choice == "4":
        config = create_small_model_config()
    else:
        print("无效选择，使用默认配置")
        config = create_custom_config()
    
    print(f"\n使用配置: {config.run_name}")
    print(f"最大轮数: {config.max_epochs}")
    print(f"批次大小: {config.batch_size}")
    print(f"学习率: {config.lr}")
    print(f"使用WandB: {config.use_wandb}")
    
    confirm = input("\n确认开始训练? (y/n): ").strip().lower()
    if confirm == 'y':
        trainer = FARTrainer(config)
        trainer.run()
    else:
        print("训练已取消")

