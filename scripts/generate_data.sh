#!/bin/bash
# ============================================================================
# BioDynaSpec Data Generation Script
# ============================================================================
# This script generates the windowed FFT dataset from raw ATLAS trajectories.
#
# Usage:
#   bash scripts/generate_data.sh
#
# Environment variables (set before running):
#   PROJECT_ROOT        - Root directory of this project
#   DATASET_DIR         - Where to store generated datasets (default: ./dataset)
#   FEATURE_CACHE_DIR   - Feature cache directory
#   BIHITO_REPR_DIR     - BiHiTo representation directory
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
DATASET_DIR="${DATASET_DIR:-${PROJECT_ROOT}/dataset}"
FEATURE_CACHE_DIR="${FEATURE_CACHE_DIR:-${DATASET_DIR}/feature_cache/atlas}"
BIHITO_REPR_DIR="${BIHITO_REPR_DIR:-${DATASET_DIR}/bihito_repr_cache}"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

echo "============================================"
echo "BioDynaSpec Data Generation"
echo "============================================"
echo "Project root: ${PROJECT_ROOT}"
echo "Dataset dir: ${DATASET_DIR}"
echo "============================================"

# ============================================================================
# Step 1: Generate windowed FFT data (window_size=2500)
# ============================================================================
echo "[Step 1] Generating windowed FFT data (window_size=2500)..."
python "${PROJECT_ROOT}/runner/data_prepare/replace_positions_full_fft_atlas.py" \
  --use_fft_windows \
  --window_size 2500 \
  --hop_ratio 1.0 \
  --save_new \
  --dt_str 10ps \
  --output_dir "${DATASET_DIR}/fft_window2500" \
  --feature_cache_dir "${FEATURE_CACHE_DIR}" \
  --bihito_repr_dir "${BIHITO_REPR_DIR}" \
  --truncate_to_expected_n \
  --verbose

# ============================================================================
# Step 2: Compute FFT window scaling factors & normalize
# ============================================================================
echo "[Step 2] Computing scaling factors and normalizing..."
python "${PROJECT_ROOT}/runner/data_prepare/compute_fft_window_scaling.py" \
  --data_dir "${DATASET_DIR}/fft_window2500" \
  --num_freq 1251 \
  --percentile 95.0 \
  --output_path "${DATASET_DIR}/fft_window2500/fft_window2500_scaling_factors.pth" \
  --normalize \
  --output_suffix _normalized

echo "Data generation completed successfully!"
echo "Generated files are in: ${DATASET_DIR}/fft_window2500/"
