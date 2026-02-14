#!/bin/bash
# ============================================================================
# BioDynaSpec Training Script
# ============================================================================
# Usage:
#   bash scripts/pretrain.sh [--config CONFIG_YAML] [--gpus NUM_GPUS]
#
# Environment variables (set before running):
#   PROJECT_ROOT    - Root directory of this project (default: parent of scripts/)
#   CONDA_ENV       - Conda environment name (default: biodynaspec)
#   NUM_GPUS        - Number of GPUs to use (default: auto-detect)
# ============================================================================

set -euo pipefail

# ============================================================================
# Path Configuration (edit these or set via environment variables)
# ============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

cd "${PROJECT_ROOT}/src"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# ============================================================================
# GPU Configuration
# ============================================================================
NUM_GPUS="${NUM_GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
if [ "${NUM_GPUS}" -eq 0 ]; then
    NUM_GPUS=1
fi

# Default config file
CONFIG_FILE="${PROJECT_ROOT}/configs/trainer_BioDynaSpec.yaml"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config) CONFIG_FILE="$2"; shift 2 ;;
        --gpus) NUM_GPUS="$2"; shift 2 ;;
        --resume) CHECKPOINT_PATH="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ============================================================================
# Environment Setup
# ============================================================================
nvidia-smi 2>/dev/null || true
echo "CUDA version: $(nvcc --version 2>/dev/null | grep release || echo 'N/A')"
echo "Number of GPUs: ${NUM_GPUS}"
echo "Config file: ${CONFIG_FILE}"
echo "Project root: ${PROJECT_ROOT}"
echo "Start time: $(date)"

# ============================================================================
# NCCL Configuration (adjust based on your hardware)
# ============================================================================
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:512}"

# ============================================================================
# Launch Training
# ============================================================================
MASTER_PORT="${MASTER_PORT:-29500}"

RESUME_ARGS=""
if [ -n "${CHECKPOINT_PATH:-}" ]; then
    RESUME_ARGS="--resume_ckpt ${CHECKPOINT_PATH}"
    echo "Resuming from checkpoint: ${CHECKPOINT_PATH}"
fi

if [ "${NUM_GPUS}" -gt 1 ]; then
    echo "Launching distributed training with ${NUM_GPUS} GPUs..."
    NCCL_NVLS_DISABLE=1 torchrun \
        --nproc_per_node=${NUM_GPUS} \
        --master_port=${MASTER_PORT} \
        trainer_far.py \
        --mode test \
        --config "${CONFIG_FILE}" \
        ${RESUME_ARGS} \
        2>&1 | tee "${PROJECT_ROOT}/logs/train_$(date +%Y%m%d_%H%M%S).log"
else
    echo "Launching single-GPU training..."
    python trainer_far.py \
        --mode test \
        --config "${CONFIG_FILE}" \
        ${RESUME_ARGS} \
        2>&1 | tee "${PROJECT_ROOT}/logs/train_$(date +%Y%m%d_%H%M%S).log"
fi

echo "Training completed at $(date)"
