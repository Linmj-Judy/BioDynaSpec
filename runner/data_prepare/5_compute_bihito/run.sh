#!/bin/bash
# ============================================================================
# Compute BiHiTo representations for MD trajectory data
# ============================================================================
# Prerequisites:
#   - BiHiTo repository cloned and installed
#   - BiHiTo checkpoint downloaded
#
# Environment variables:
#   BIHITO_ROOT           - Path to BiHiTo repository
#   BIHITO_CHECKPOINT_PATH - Path to BiHiTo model checkpoint
#   PROJECT_ROOT          - Path to BioDynaSpec project root
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"

BIHITO_ROOT="${BIHITO_ROOT:?Please set BIHITO_ROOT to the path of BiHiTo repository}"
BIHITO_CHECKPOINT_PATH="${BIHITO_CHECKPOINT_PATH:?Please set BIHITO_CHECKPOINT_PATH}"

export PYTHONPATH="${BIHITO_ROOT}/src:${PYTHONPATH:-}"

# Step 1: Extract first-frame PDBs for each window
python "${PROJECT_ROOT}/runner/data_prepare/5_compute_bihito/get_window_first_frame_pdbs.py" \
  --out "${PROJECT_ROOT}/dataset/first_frame_pdbs/" \
  -w 2500 -j 48

# Step 2: Generate BiHiTo embeddings
cd "${BIHITO_ROOT}"
python "${PROJECT_ROOT}/runner/data_prepare/5_compute_bihito/generate_bihito_keep_order.py" \
  --config test_pdb_dir_mddata.yaml \
  --ckpt_path "${BIHITO_CHECKPOINT_PATH}" \
  --num_gpus 8 --threads 24 --io_threads 8 2>&1 | tee -a "${PROJECT_ROOT}/logs/bihito_repr/compute_bihito.log"

# Step 3: Aggregate representations
python "${PROJECT_ROOT}/runner/data_prepare/5_compute_bihito/aggregate_repr.py" --overwrite \
  --bihito_cache_dir "${PROJECT_ROOT}/dataset/bihito_cache" \
  --bihito_repr_out_path "${PROJECT_ROOT}/dataset/bihito_repr_cache"
