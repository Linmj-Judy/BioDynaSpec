#!/bin/bash
# ============================================================================
# Compute ESM embeddings for ATLAS proteins (GPU version)
# ============================================================================
# Same as compute_esm.sh, but prints GPU/model diagnostics first.
# Prefer this on a GPU node.
#
# Usage:
#   bash runner/data_prepare/4_compute_esm/compute_esm_h200.sh
#
# Environment variables:
#   PROJECT_ROOT        - Root directory of this project
#   ESM_MODEL_NAME      - Key into configs/configs_model.py ESM_CONFIG
#                         (default: esm2-650m; the paper uses ESM-2-650M = 1280-d)
#   ESM_SAVE_DIR        - Directory to save ESM embeddings
#   ESM_MODEL_ROOT_DIR  - Directory containing ESM model weights
#                         (passed as --local_esm_model_dir)
#   PDB_SEQ_PATH        - Path to the PDB sequence CSV
#   PDB_SEQ_LABEL_PATH  - Path to the PDB sequence label CSV
#   GPU_ID              - GPU index to use (optional)
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

nvidia-smi 2>/dev/null || true
echo "CUDA version: $(nvcc --version 2>/dev/null | grep release || echo 'N/A')"

# Configure paths (edit these or set via environment variables)
ESM_MODEL_NAME="${ESM_MODEL_NAME:-esm2-650m}"
ESM_SAVE_DIR="${ESM_SAVE_DIR:-./dataset/esm_cache/esm2_650m_embed}"
ESM_MODEL_ROOT_DIR="${ESM_MODEL_ROOT_DIR:-./data/esm_models}"
PDB_SEQ_PATH="${PDB_SEQ_PATH:-./data/pdb_seqs/pdb_seq.csv}"
PDB_SEQ_LABEL_PATH="${PDB_SEQ_LABEL_PATH:-./data/pdb_seqs/pdb_labels_seqs.csv}"

echo "ESM Model Name: ${ESM_MODEL_NAME}"
echo "ESM Model Dir: ${ESM_MODEL_ROOT_DIR}"
echo "ESM Save Dir: ${ESM_SAVE_DIR}"

GPU_ARGS=""
if [ -n "${GPU_ID:-}" ]; then
  GPU_ARGS="--gpu_id ${GPU_ID}"
fi

python src/data/basic/compute_esm.py \
  --model_name "${ESM_MODEL_NAME}" \
  --esm_save_dir "${ESM_SAVE_DIR}" \
  --local_esm_model_dir "${ESM_MODEL_ROOT_DIR}" \
  --pdb_seq_path "${PDB_SEQ_PATH}" \
  --pdb_seq_label_path "${PDB_SEQ_LABEL_PATH}" \
  ${GPU_ARGS}
