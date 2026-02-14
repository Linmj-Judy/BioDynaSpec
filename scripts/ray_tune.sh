#!/bin/bash
# Ray Tune超参数优化脚本

# ============================================================================
# 基础配置
# ============================================================================

# Python环境（根据实际情况修改）
# source /path/to/your/conda/bin/activate your_env_name
# 或者
# export PATH=/path/to/your/python:$PATH

# 进入项目目录
cd "$(dirname "$0")/.." || exit
export PYTHONPATH="${PWD}/src:${PYTHONPATH}"

# ============================================================================
# Ray Tune参数配置
# ============================================================================

# 运行模式: "tune" 或 "test"
MODE="tune"

# 超参数采样数量（试验次数）
NUM_SAMPLES=10

# 每个trial的最大训练epoch数
NUM_EPOCHS=50

# 资源配置
GPUS_PER_TRIAL=1
CPUS_PER_TRIAL=4
MAX_CONCURRENT_TRIALS=4

# ASHA调度器参数
GRACE_PERIOD=10
REDUCTION_FACTOR=2

# ============================================================================
# 快速配置模板
# ============================================================================

# 取消注释以使用不同的配置模板

# --- 快速测试配置 ---
# MODE="test"

# --- 小规模搜索配置 ---
# NUM_SAMPLES=5
# NUM_EPOCHS=30
# MAX_CONCURRENT_TRIALS=2

# --- 中规模搜索配置 ---
# NUM_SAMPLES=20
# NUM_EPOCHS=50
# MAX_CONCURRENT_TRIALS=4

# --- 大规模搜索配置 ---
# NUM_SAMPLES=50
# NUM_EPOCHS=100
# MAX_CONCURRENT_TRIALS=8

# ============================================================================
# 运行Ray Tune
# ============================================================================

echo "========================================"
echo "Ray Tune 超参数优化"
echo "========================================"
echo "运行模式: $MODE"
echo "采样数量: $NUM_SAMPLES"
echo "每个trial的epoch数: $NUM_EPOCHS"
echo "每个trial的GPU数: $GPUS_PER_TRIAL"
echo "每个trial的CPU数: $CPUS_PER_TRIAL"
echo "最大并发trials: $MAX_CONCURRENT_TRIALS"
echo "========================================"
echo ""

# 构建命令
CMD="python src/trainer_far_ray_tune.py \
    --mode $MODE \
    --num_samples $NUM_SAMPLES \
    --num_epochs $NUM_EPOCHS \
    --gpus_per_trial $GPUS_PER_TRIAL \
    --cpus_per_trial $CPUS_PER_TRIAL \
    --max_concurrent_trials $MAX_CONCURRENT_TRIALS \
    --grace_period $GRACE_PERIOD \
    --reduction_factor $REDUCTION_FACTOR"

# 可选：禁用ASHA调度器（如果需要）
# CMD="$CMD --no_asha"

echo "执行命令:"
echo "$CMD"
echo ""

# 运行
eval $CMD

# ============================================================================
# 后处理
# ============================================================================

echo ""
echo "========================================"
echo "Ray Tune运行完成"
echo "========================================"
echo ""
echo "查看结果："
echo "  结果目录: ./runs/"
echo "  最佳配置: ./runs/best_config.txt"
echo ""
echo "使用Ray Dashboard查看详细结果："
echo "  ray dashboard --port 8265"
echo ""

