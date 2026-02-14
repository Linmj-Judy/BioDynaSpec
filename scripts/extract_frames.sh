#!/bin/bash
# ============================================================================
# BioDynaSpec Frame Extraction Script
# ============================================================================
# Extracts frames from raw MD trajectories and saves as numpy arrays.
#
# Usage:
#   bash scripts/extract_frames.sh
#
# Environment variables:
#   PROJECT_ROOT  - Root directory of this project
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

python "${PROJECT_ROOT}/runner/data_prepare/extract_frames/save_frames_to_npy_atlas.py"
