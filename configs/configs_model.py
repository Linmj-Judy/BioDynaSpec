import os
from typing import Dict, Any

ESM_CONFIG = {
    "esm2-650m": {
        "type": "esm2",
        "model_path": "release_model/esm2_t33_650M_UR50D.pt",
        "emb_dim": 1280,
        "n_layers": 33,
        "emb_dir": "esm_cache/esm2_650m_embed/", # default embedding directory
        "model_url":'https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt',
        "contact_model_url": "https://dl.fbaipublicfiles.com/fair-esm/regression/esm2_t33_650M_UR50D-contact-regression.pt"
    },
    "esm2-3b": {
        "type": "esm2",
        "model_path": "release_model/esm2_t36_3B_UR50D.pt",
        "emb_dim": 2560,
        "n_layers": 36,
        "emb_dir":"esm_cache/esm2_3b_embed/" , # default embedding directory
        "model_url":'https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t36_3B_UR50D.pt',
        "contact_model_url": 'https://dl.fbaipublicfiles.com/fair-esm/regression/esm2_t36_3B_UR50D-contact-regression.pt'
    },
}

class DefaultModelConfig:
    def __init__(self):
        self.mask = True
        self.mask_ratio_min = 0.5
        self.mlp_ratio = 4.0 # yl说这个ratio在3-4之间最好
        
        self.rope_theta = 10000.0  # RoPE frequency scaling
        # # LoRA
        # self.use_lora = False
        # # self.lora_rank = 8
        # # self.lora_alpha = 16.0
        # # self.lora_dropout = 0.1

    def as_dict(self):
        return {
            "mask": self.mask,
            "mask_ratio_min": self.mask_ratio_min,
            "mlp_ratio": self.mlp_ratio,
            "rope_theta": self.rope_theta,
            # "use_lora": self.use_lora,
        }

class DefaultHyperparamConfig:
    """
    配置类：用于保存不使用搜索时的默认超参数设置
    通过实例属性进行参数管理
    """
    def __init__(self):
        self.lr = 1e-4
        self.encoder_depth = int(os.getenv("VAE_ENCODER_DEPTH", "2"))
        self.encoder_num_heads = int(os.getenv("VAE_ENCODER_NUM_HEADS", "16")) # c_token=768, c_hidden=768/12=64 必须使得c_hidden<=64! 因此decoder_num_heads必须>=12!
        # self.decoder_embed_dim = 512
        self.decoder_depth = int(os.getenv("VAE_DECODER_DEPTH", "2"))
        self.decoder_num_heads = int(os.getenv("VAE_DECODER_NUM_HEADS", "16")) # c_token=768, c_hidden=768/12=64 必须使得c_hidden<=64! 因此decoder_num_heads必须>=12!
        self.buffer_size = int(os.getenv("VAE_BUFFER_SIZE", "16"))

        self.condition_drop_prob = float(os.getenv("VAE_CONDITION_DROP_PROB", "0.1"))
        self.proj_dropout = float(os.getenv("VAE_PROJ_DROPOUT", "0.1"))
        self.attn_dropout = float(os.getenv("VAE_ATTN_DROPOUT", "0.1"))

        self.warmup_epochs = 5
        self.min_lr = 1e-6
        self.warmup_start_lr = 1e-7
        self.plateau_patience = 8
        self.plateau_factor = 0.7
        self.beta = 1e-6

        # self.lora_rank = 8
        # self.lora_alpha = 16.0
        # self.lora_dropout = 0.1

    def as_dict(self):
        """返回字典形式用于兼容性传参"""    
        default_model_config = DefaultModelConfig()
        default_model_config = default_model_config.as_dict()

        default_hyperparam_config = {
            "lr": self.lr,
            "encoder_depth": self.encoder_depth,
            "encoder_num_heads": self.encoder_num_heads,
            # "decoder_embed_dim": self.decoder_embed_dim,
            "decoder_depth": self.decoder_depth,
            "decoder_num_heads": self.decoder_num_heads,
            "buffer_size": self.buffer_size,
            "condition_drop_prob": self.condition_drop_prob,
            "proj_dropout": self.proj_dropout,
            "attn_dropout": self.attn_dropout,
            "warmup_epochs": self.warmup_epochs,
            "min_lr": self.min_lr,
            "warmup_start_lr": self.warmup_start_lr,
            "plateau_patience": self.plateau_patience,
            "plateau_factor": self.plateau_factor,
            "beta": self.beta,
            # "lora_rank": self.lora_rank,
            # "lora_alpha": self.lora_alpha,
            # "lora_dropout": self.lora_dropout,
        }
        return {**default_model_config, **default_hyperparam_config}

def get_search_space() -> Dict[str, Any]:
    """ß
    定义超参数搜索空间
    
    Returns:
        搜索空间字典
    """
    # Ray is needed only when constructing a Tune search space.  Fixed-config
    # train/test runs import this module without paying Ray's startup cost.
    from ray import tune

    default_model_config = DefaultModelConfig()
    default_model_config = default_model_config.as_dict()

    search_space = {
        # 学习率搜索（对数均匀采样）
        "lr": tune.loguniform(1e-5, 1e-3),
        
        # 模型架构超参数
        "encoder_depth": tune.choice([2, 4, 6]),
        "encoder_num_heads": tune.choice([4, 8, 16]),
        # "decoder_embed_dim": tune.choice([256, 512, 768]),
        "decoder_depth": tune.choice([2, 4, 6]),
        "decoder_num_heads": tune.choice([16, 32, 64]), # 必须>=12
        
        # Buffer size
        "buffer_size": tune.choice([4, 8, 16, 32, 64]),
        #Conditional drop for CFG
        "condition_drop_prob": tune.uniform(0.0, 1.0),
        "proj_dropout": tune.uniform(0.0, 1.0),
        "attn_dropout": tune.uniform(0.0, 1.0),
        # "lora_rank": tune.choice([4, 8, 16]),
        # "lora_alpha": tune.choice([8.0, 16.0, 32.0]),
        # "lora_dropout": tune.uniform(0.0, 1.0),
        # 优化器参数
        "warmup_epochs": tune.choice([5, 10, 15]),
        "min_lr": tune.loguniform(1e-7, 1e-5),
        "warmup_start_lr": tune.loguniform(1e-8, 1e-6),
        "plateau_patience": tune.choice([5, 8, 10]),
        "plateau_factor": tune.choice([0.5, 0.7, 0.9]),
        
        # Beta参数（KL loss权重）
        "beta": tune.loguniform(1e-7, 1e-5),
    }
    
    return {**default_model_config, **search_space}
