import torch
import os
import atexit  # 导入atexit模块用于注册退出函数
from src.utils.logger import get_logger

logger = get_logger(__name__)

def cleanup():  
    """  
    清理分布式训练环境  
    """  
    if torch.cuda.is_available():
        try:
            from megatron.core import parallel_state
            has_megatron = True
            if (parallel_state.model_parallel_is_initialized() or parallel_state.sequence_parallel_is_initialized()):
                logger.info("正在清理Megatron并行状态...")
                parallel_state.destroy_model_parallel()
                logger.info("Megatron并行状态已清理")

        except (ImportError, RuntimeError) as e:
            has_megatron = False        
            logger.warning(f"Warning: Megatron-LM initialization failed: {e}")
    else:
        has_megatron = False
        logger.warning("No GPU available, skipping Megatron-LM import")

    # 如果使用了Megatron-LM，也需要清理其并行状态

def setup_megatron_environment(
    configs,
    dtype, 
    tensor_parallel_size=None, 
    sequence_parallel_size=None):
    """
    设置Megatron-LM环境
    
    Args:
        training_params: 训练参数
        dtype: 数据类型
        rank: 当前进程排名
        tensor_parallel_size: 张量并行大小
        sequence_parallel_size: 序列并行大小
    
    Returns:
        tuple: (是否成功初始化Megatron环境, Megatron配置)
    """
    if torch.cuda.is_available():
        try:
            from megatron.core import parallel_state
            has_megatron = True
        except (ImportError, RuntimeError) as e:
            has_megatron = False        
            logger.warning(f"Warning: Megatron-LM initialization failed: {e}")
    else:
        has_megatron = False
        logger.warning("No GPU available, skipping Megatron-LM import")

    if not has_megatron:
        logger.warning("Warning: Megatron-LM initialization failed: has_megatron is not in globals() or not has_megatron")
        return False, None
        
    # 如果tensor_parallel_size为1且sequence_parallel_size为1，则不启用张量并行
    if tensor_parallel_size <= 1 and sequence_parallel_size <= 1:
        logger.warning("Warning: Megatron-LM initialization failed: tensor_parallel_size or sequence_parallel_size is not greater than 1")
        return False, None
    
    # 第一次调用时真正初始化 Megatron；之后如果已经初始化，则跳过 initialize_model_parallel
    if not parallel_state.model_parallel_is_initialized():
        logger.info("Initializing Megatron-LM model parallel...")
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=tensor_parallel_size,
            sequence_parallel_size=sequence_parallel_size,
            pipeline_model_parallel_size=1,  # 暂不使用流水线并行
            virtual_pipeline_model_parallel_size=None,
            use_fp8=False,
            use_distributed_optimizer=False
            # distributed_timeout_minutes=360
        )
        # see https://github.com/deepspeedai/Megatron-DeepSpeed/blob/main/megatron/core/parallel_state.py
        # and https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/parallel_state.py
        
        # 初始化CUDA RNG状态追踪器（只在第一次初始化时添加）
        from megatron.core.tensor_parallel.random import get_cuda_rng_tracker, model_parallel_cuda_manual_seed

        # 设置随机种子以确保所有GPU获得一致的随机初始化
        model_parallel_cuda_manual_seed(configs.seed)
        
        # 添加模型并行RNG状态（使用try-except防止重复添加）
        try:
            cuda_rng_tracker = get_cuda_rng_tracker()
            if 'model-parallel-rng' not in cuda_rng_tracker.get_states().keys():
                cuda_rng_tracker.add('model-parallel-rng', configs.seed)
                logger.debug("已添加 model-parallel-rng 状态")
            else:
                logger.debug("model-parallel-rng 状态已存在，跳过添加")
        except Exception as e:
            # 如果RNG状态已经存在，捕获异常并继续（可能是其他进程已经添加了）
            if 'already exists' in str(e):
                logger.debug(f"RNG状态已存在（可能由其他进程添加）: {e}")
            else:
                logger.warning(f"添加RNG状态时出现异常: {e}")
        
        # 注册程序退出时的清理函数（使用atexit.register的idempotent特性，多次注册不会重复执行）
        try:
            atexit.register(cleanup)
        except Exception:
            pass  # 如果已经注册过，忽略异常

    # 无论是否是第一次调用，都构造一个 ModelParallelConfig 返回给上层使用
    # from megatron.core.model_parallel_config import ModelParallelConfig
    from configs.megatron_parallel_config import ModelParallelConfig # 导入一个我fix bugs后的版本

    config = ModelParallelConfig(
        # Model parallelism
        tensor_model_parallel_size=tensor_parallel_size, # 张量模型并行维度（设备数，默认为1）  
        pipeline_model_parallel_size=1, # 流水线并行维度 
        # virtual_pipeline_model_parallel_size is None by default
        sequence_parallel=(sequence_parallel_size > 1), # 是否使用序列并行

        # Initialization
        perform_initialization=True, # 是否执行权重初始化
        use_cpu_initialization=False, # 是否在CPU上初始化参数 

        # Training
        fp16=(dtype == torch.float16),
        bf16=(dtype == torch.bfloat16),
        params_dtype=dtype, # 模型参数数据类型

        # Optimizations
        gradient_accumulation_fusion=False,

        # Pipeline Parallelism
        pipeline_dtype=dtype,
        enable_autocast=True,

        # Other options
        # You may add or remove args as needed, no unsupported fields!
    )

    # 记录模型并行信息
    tensor_model_parallel_rank = parallel_state.get_tensor_model_parallel_rank()
    data_parallel_rank = parallel_state.get_data_parallel_rank()
    sequence_parallel_rank = parallel_state.get_sequence_parallel_rank()
    
    logger.info(f"模型并行信息: "
              f"tensor_parallel_rank={tensor_model_parallel_rank}, "
              f"data_parallel_rank={data_parallel_rank}, "
              f"sequence_parallel_rank={sequence_parallel_rank}")
    
    return True, config
