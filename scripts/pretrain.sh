#!/bin/bash
# ============================================================================
# BioDynaSpec Training Script
# ============================================================================
# Usage:
#   bash scripts/pretrain.sh [--config CONFIG_YAML] [--gpus NUM_GPUS]
#
# Environment variables (set before running):
#   PROJECT_ROOT    - Root directory of this project (default: parent of scripts/)
#   CONDA_ENV_NAME  - Conda env to use (default: torch); auto-located, no need to activate
#   PYTHON_BIN      - Explicit interpreter path, overrides env lookup
#   NUM_GPUS        - Number of GPUs to use (default: auto-detect)
# ============================================================================

set -euo pipefail

# ============================================================================
# Path Configuration (edit these or set via environment variables)
# ============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

# ----------------------------------------------------------------------------
# Resolve the interpreter: use the "torch" conda env without requiring the
# caller to activate it. The default `python` here is conda *base*, whose torch
# is a CPU-only build and which has no pytorch_lightning — that mismatch is the
# single most common reason this script "does not work".
#
# The probe is a site-packages check (milliseconds); we deliberately do NOT
# import torch to test for it, since on a loaded box that costs ~50s.
# ----------------------------------------------------------------------------
CONDA_ENV_NAME="${CONDA_ENV_NAME:-torch}"
_looks_usable() {
  [ -n "$1" ] && [ -x "$1" ] || return 1
  local _root _d
  _root="$(dirname "$(dirname "$1")")"
  for _d in "$_root"/lib/python3.*/site-packages/pytorch_lightning; do
    [ -d "$_d" ] && return 0
  done
  return 1
}
if [ -z "${PYTHON_BIN:-}" ] && ! _looks_usable "$(command -v python 2>/dev/null)"; then
  for _cand in "$HOME/miniconda3/envs/${CONDA_ENV_NAME}/bin/python" \
               "$HOME/anaconda3/envs/${CONDA_ENV_NAME}/bin/python" \
               "/data/L202500336/miniconda3/envs/${CONDA_ENV_NAME}/bin/python" \
               "/opt/conda/envs/${CONDA_ENV_NAME}/bin/python"; do
    _looks_usable "${_cand}" && { PYTHON_BIN="${_cand}"; break; }
  done
fi
if [ -n "${PYTHON_BIN:-}" ]; then
  export PATH="$(dirname "${PYTHON_BIN}"):${PATH}"
else
  PYTHON_BIN="$(command -v python 2>/dev/null || echo python)"
fi
if ! _looks_usable "${PYTHON_BIN}"; then
  echo "ERROR: no python with pytorch_lightning found (env '${CONDA_ENV_NAME}')." >&2
  echo "  Fix: conda activate ${CONDA_ENV_NAME}" >&2
  echo "   or: PYTHON_BIN=/path/to/envs/${CONDA_ENV_NAME}/bin/python $0 ..." >&2
  exit 1
fi

# Run from PROJECT_ROOT, NOT from src/.
#
# Two different import roots are needed and both must resolve:
#   * `from main_model import ...`      -> resolved because we invoke
#                                          `python "${PROJECT_ROOT}/src/trainer_far.py"`,
#                                          so Python puts src/ itself on sys.path[0].
#   * `from src.openfold_local import ...` -> resolved via PYTHONPATH below.
#
# Staying at PROJECT_ROOT matters because every path in the configs/*.yaml is
# relative to it ("./dataset/...", "./runs"). `cd src` used to break those:
# "./dataset" became src/dataset, which does not exist, so dataset loading
# failed with FileNotFoundError.
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# ============================================================================
# GPU Configuration
# ============================================================================
# `nvidia-smi -L` reports ALL physical GPUs and ignores CUDA_VISIBLE_DEVICES, so
# using it unconditionally made `CUDA_VISIBLE_DEVICES=0 bash pretrain.sh` launch
# a 4-way DDP run. Count the visible devices instead.
_gpu_count() {
    if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
        # Count only the devices CUDA_VISIBLE_DEVICES exposes. Ask the resolved
        # interpreter (PYTHON_BIN), NOT bare `python` — the latter may be a
        # CPU-only build and would report 0.
        "${PYTHON_BIN:-python}" -c 'import torch;print(torch.cuda.device_count())' 2>/dev/null || echo 1
    else
        nvidia-smi -L 2>/dev/null | wc -l
    fi
}
NUM_GPUS="${NUM_GPUS:-$(_gpu_count)}"
if [ "${NUM_GPUS}" -eq 0 ] 2>/dev/null; then
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
        -h|--help) sed -n '2,14p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Resolve --config to an ABSOLUTE path.
# NOTE: we are already inside "${PROJECT_ROOT}/src" at this point (see the
# `cd` above), so a relative --config from the caller — e.g.
# `--config configs/trainer_smoke.yaml` — must be resolved against
# PROJECT_ROOT, not against the current directory, or it becomes src/configs/...
# and fails with "配置文件不存在".
if [ "${CONFIG_FILE#/}" = "${CONFIG_FILE}" ]; then
    for _base in "${PROJECT_ROOT}" "${PROJECT_ROOT}/src" "$(pwd)"; do
        if [ -f "${_base}/${CONFIG_FILE}" ]; then
            CONFIG_FILE="${_base}/${CONFIG_FILE}"
            break
        fi
    done
fi
if [ ! -f "${CONFIG_FILE}" ]; then
    echo "Error: config file not found: ${CONFIG_FILE}"
    echo "  (searched PROJECT_ROOT=${PROJECT_ROOT} and its src/)"
    exit 1
fi

# The training log is tee'd into ${PROJECT_ROOT}/logs; create it up front or
# the tee fails on a clean checkout (logs/ is gitignored).
mkdir -p "${PROJECT_ROOT}/logs"

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
        "${PROJECT_ROOT}/src/trainer_far.py" \
        --config "${CONFIG_FILE}" \
        ${RESUME_ARGS} \
        2>&1 | tee "${PROJECT_ROOT}/logs/train_$(date +%Y%m%d_%H%M%S).log"
else
    echo "Launching single-GPU training..."
    "${PYTHON_BIN}" "${PROJECT_ROOT}/src/trainer_far.py" \
        --config "${CONFIG_FILE}" \
        ${RESUME_ARGS} \
        2>&1 | tee "${PROJECT_ROOT}/logs/train_$(date +%Y%m%d_%H%M%S).log"
fi

echo "Training completed at $(date)"
