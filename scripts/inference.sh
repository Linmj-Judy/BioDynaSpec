#!/bin/bash
# ============================================================================
# BioDynaSpec Inference Script
# ============================================================================
# Usage:
#   bash scripts/inference.sh \
#       --checkpoint /path/to/checkpoint.ckpt \
#       --data /path/to/test_data.pth \
#       --output_dir /path/to/output
#
# Environment variables (optional):
#   PROJECT_ROOT    - Root directory of this project
# ============================================================================

set -euo pipefail

# ============================================================================
# Path Configuration
# ============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

cd "${PROJECT_ROOT}/src"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# ============================================================================
# Default Parameters
# ============================================================================
CHECKPOINT_PATH=""
DATA_PATH="${PROJECT_ROOT}/dataset/fft_window2500/atlas_10_features_fouriered_list_test_normalized.pth"
OUTPUT_DIR="${PROJECT_ROOT}/runs/inference_output"
DATA_MODE="fft_window"
DEVICE="cuda"
NUM_SAMPLES=1
WINDOW_SIZE=2500
SEED=42
NUM_PDB_PROTEINS=100
NUM_FRAMES=2500
CFG=0.0
TEMPERATURE=1.0

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint) CHECKPOINT_PATH="$2"; shift 2 ;;
        --data) DATA_PATH="$2"; shift 2 ;;
        --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
        --data_mode) DATA_MODE="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        --num_samples) NUM_SAMPLES="$2"; shift 2 ;;
        --window_size) WINDOW_SIZE="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --num_frames) NUM_FRAMES="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [ -z "${CHECKPOINT_PATH}" ]; then
    echo "Error: --checkpoint is required."
    echo "Usage: bash scripts/inference.sh --checkpoint /path/to/checkpoint.ckpt"
    exit 1
fi

# ============================================================================
# Environment Setup
# ============================================================================
echo "============================================"
echo "BioDynaSpec Inference"
echo "============================================"
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "Data path: ${DATA_PATH}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Data mode: ${DATA_MODE}"
echo "Device: ${DEVICE}"
echo "Window size: ${WINDOW_SIZE}"
echo "Num samples: ${NUM_SAMPLES}"
echo "Start time: $(date)"
echo "============================================"

nvidia-smi 2>/dev/null || true

# ============================================================================
# Run Inference
# ============================================================================
python inferencer_far.py \
    --checkpoint_path "${CHECKPOINT_PATH}" \
    --data_path "${DATA_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --data_mode "${DATA_MODE}" \
    --device "${DEVICE}" \
    --model_size base \
    --cfg "${CFG}" \
    --temperature "${TEMPERATURE}" \
    --sample_method nomask \
    --num_samples "${NUM_SAMPLES}" \
    --window_size "${WINDOW_SIZE}" \
    --seed "${SEED}" \
    --save_pdb \
    --num_pdb_proteins "${NUM_PDB_PROTEINS}" \
    --pdb_output_dir "${OUTPUT_DIR}/reconstructed_trajs" \
    --num_frames "${NUM_FRAMES}" \
    --save_gt \
    2>&1 | tee "${OUTPUT_DIR}/inference_$(date +%Y%m%d_%H%M%S).log"

echo "Inference completed at $(date)"
