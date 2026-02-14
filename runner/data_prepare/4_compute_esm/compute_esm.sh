#!/bin/bash
# ============================================================================
# Compute ESM embeddings for ATLAS proteins
# ============================================================================
# Usage:
#   bash runner/data_prepare/4_compute_esm/compute_esm.sh
#
# Environment variables:
#   PROJECT_ROOT      - Root directory of this project
#   ESM_MODEL_ROOT_DIR - Directory containing ESM model weights
#   ESM_SAVE_DIR      - Directory to save ESM embeddings
#   OUTPUT_SEQ_PATH   - Directory to save extracted sequences
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# Configure paths (edit these or set via environment variables)
ESM_MODEL_ROOT_DIR="${ESM_MODEL_ROOT_DIR:-./data/esm_models}"
ESM_SAVE_DIR="${ESM_SAVE_DIR:-./dataset/esm_cache/esm2_3b_embed}"
OUTPUT_SEQ_PATH="${OUTPUT_SEQ_PATH:-./dataset/sequences/}"

python src/data/basic/compute_esm.py \
  --esm_model_root_dir "${ESM_MODEL_ROOT_DIR}" \
  --esm_save_dir "${ESM_SAVE_DIR}" \
  --output_seq_path "${OUTPUT_SEQ_PATH}"
