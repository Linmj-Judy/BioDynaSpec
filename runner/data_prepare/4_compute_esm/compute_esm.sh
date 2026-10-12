#!/bin/bash
# ============================================================================
# Compute ESM embeddings for ATLAS proteins
# ============================================================================
# Usage:
#   bash runner/data_prepare/4_compute_esm/compute_esm.sh
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
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# Configure paths (edit these or set via environment variables)
ESM_MODEL_NAME="${ESM_MODEL_NAME:-esm2-650m}"
ESM_SAVE_DIR="${ESM_SAVE_DIR:-./dataset/esm_cache/esm2_650m_embed}"
ESM_MODEL_ROOT_DIR="${ESM_MODEL_ROOT_DIR:-./data/esm_models}"
PDB_SEQ_PATH="${PDB_SEQ_PATH:-./data/pdb_seqs/pdb_seq.csv}"
PDB_SEQ_LABEL_PATH="${PDB_SEQ_LABEL_PATH:-./data/pdb_seqs/pdb_labels_seqs.csv}"

python src/data/basic/compute_esm.py \
  --model_name "${ESM_MODEL_NAME}" \
  --esm_save_dir "${ESM_SAVE_DIR}" \
  --local_esm_model_dir "${ESM_MODEL_ROOT_DIR}" \
  --pdb_seq_path "${PDB_SEQ_PATH}" \
  --pdb_seq_label_path "${PDB_SEQ_LABEL_PATH}"

