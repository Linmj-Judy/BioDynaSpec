#!/usr/bin/env bash
# ============================================================================
# BioDynaSpec — complete data-preparation pipeline
# ============================================================================
# Reproduces every stage that turns the RAW ATLAS molecular-dynamics data into
# the training tensors consumed by `scripts/pretrain.sh`.
#
# STARTING POINT
#   Raw ATLAS: per-trajectory GROMACS topology + trajectory files,
#       $ATLAS_RAW_DIR/<pdb>_<chain>/total/<pdb>_<chain>_prod_<R1|R2|R3>.{tpr,xtc}
#   Public source: https://www.dsimb.inserm.fr/ATLAS
#
# ENDING POINT
#   $DATASET_DIR/fft_window<W>/atlas_10_features_fouriered_list_{train,val,test}_normalized.pth
#   which is what configs/trainer_BioDynaSpec.yaml points at.
#
# STAGE MAP (see README.md "Data Preparation" for the full narrative)
#   0  download   ATLAS metadata + mmCIF              (optional, network)
#   1  extract    .tpr/.xtc           -> frames .npy  [Cα only]
#   2  features   mmCIF + frames      -> bioassembly .pkl.gz, seq indices, traj indices
#   3  esm        sequences           -> ESM-2 embeddings (.pt per PDB entity)
#   4  assemble   frames+assembly+esm -> atlas_10_features_fouriered_list_{split}.pth
#   5  normalize  split .pth          -> *_normalized.pth + scaling factors
#
# Stages 4 and 5 are the ones that were missing from the repository; the
# assembler (runner/data_prepare/assemble_training_dataset.py) was authored to
# fill the gap. Everything downstream of it already existed.
#
# USAGE
#   # whole pipeline with defaults
#   conda activate torch
#   bash scripts/generate_data.sh --all
#
#   # just re-run one stage (e.g. after changing the window size)
#   bash scripts/generate_data.sh --stage assemble --stage normalize
#
#   # see what would run, without running it
#   bash scripts/generate_data.sh --all --dry-run
#
# All paths are overridable by environment variable; see the CONFIG block.
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# ============================================================================
# CONFIG — every path/knob, all overridable from the environment
# ============================================================================
# ---- raw input -------------------------------------------------------------
ATLAS_RAW_DIR="${ATLAS_RAW_DIR:-${PROJECT_ROOT}/data/raw_data/ATLAS}"
# mmCIF structures (one per PDB id) — input to the feature stage.
MMCIF_DIR="${MMCIF_DIR:-${PROJECT_ROOT}/data/mmcifs}"
# ATLAS metadata CSV used by the (optional) download stage.
ATLAS_CSV_PATH="${ATLAS_CSV_PATH:-${PROJECT_ROOT}/data/raw_data/ATLAS.csv}"

# ---- intermediate products -------------------------------------------------
TRAJ_DIR="${TRAJ_DIR:-${PROJECT_ROOT}/data/trajectories/atlas}"        # stage 1
BIOASSEMBLY_DIR="${BIOASSEMBLY_DIR:-${PROJECT_ROOT}/data/bioassembly/atlas}"  # stage 2
SEQ_INDEX_DIR="${SEQ_INDEX_DIR:-${PROJECT_ROOT}/data/sequences}"       # stage 2
ESM_DIR="${ESM_DIR:-${PROJECT_ROOT}/data/esm_cache/esm2_650m_embed}"   # stage 3
ESM_MODEL_NAME="${ESM_MODEL_NAME:-esm2-650m}"                          # MUST be the 1280-d model

# ---- final dataset ---------------------------------------------------------
DATASET_DIR="${DATASET_DIR:-${PROJECT_ROOT}/dataset}"
WINDOW_SIZE="${WINDOW_SIZE:-2500}"
HOP_RATIO="${HOP_RATIO:-1.0}"
DT_STR="${DT_STR:-10ps}"
SPLIT_RATIOS="${SPLIT_RATIOS:-0.93 0.03 0.04}"
SEED="${SEED:-42}"
NUM_WORKERS="${NUM_WORKERS:-8}"

# ---- derived ---------------------------------------------------------------
FFT_SUBDIR="fft_window${WINDOW_SIZE}"
OUT_DIR="${DATASET_DIR}/${FFT_SUBDIR}"
# F = floor(W/2) + 1 non-negative rFFT bins (paper Eq. 2). W=2500 -> 1251.
NUM_FREQ="${NUM_FREQ:-$(( WINDOW_SIZE / 2 + 1 ))}"
SCALING_PCT="${SCALING_PCT:-95.0}"

# ============================================================================
# CLI
# ============================================================================
STAGES=()
DRY_RUN=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --all)      STAGES=(download extract features esm assemble normalize); shift ;;
    --stage)    STAGES+=("$2"); shift 2 ;;
    --dry-run)  DRY_RUN=1; shift ;;
    -h|--help)
      sed -n '2,45p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
      exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done
if [ ${#STAGES[@]} -eq 0 ]; then
  echo "No stages selected. Use --all, or --stage <name>." >&2
  echo "Stages: download extract features esm assemble normalize" >&2
  exit 2
fi

run() {
  if [ "${DRY_RUN}" = "1" ]; then
    printf '  [dry-run] %s\n' "$*"
  else
    printf '  $ %s\n' "$*"
    "$@"
  fi
}

# Hard error normally; during --dry-run just flag it and continue, so one dry
# run surfaces every missing input instead of stopping at the first.
fail() {
  if [ "${DRY_RUN}" = "1" ]; then
    echo "  [dry-run would FAIL] $1"
    return 0
  fi
  echo "ERROR: $1" >&2
  shift
  for line in "$@"; do echo "  $line" >&2; done
  exit 1
}

banner() {
  echo
  echo "================================================================================"
  echo "  $1"
  echo "================================================================================"
}

# ============================================================================
# Resolve the interpreter — use the "torch" conda env, don't just demand it.
# The default `python` on PATH is conda *base*, whose torch is a CPU-only build
# and which lacks pytorch_lightning. Rather than failing and telling the user to
# activate, locate the env ourselves; PATH is fixed up so any child process
# (and `python` inside the stage commands) gets the same interpreter.
# ============================================================================
CONDA_ENV_NAME="${CONDA_ENV_NAME:-torch}"
PYTHON_BIN="${PYTHON_BIN:-}"

# Two probes, deliberately:
#   _looks_usable — a filesystem check (milliseconds). Importing torch just to
#     test for torch costs ~11s on the base interpreter (it loads the whole
#     CPU build before failing), which made startup unusable.
#   _usable       — the authoritative import check, run ONCE on the winner.
_looks_usable() {  # 0 if $1's env has pytorch_lightning installed
  [ -n "$1" ] && [ -x "$1" ] || return 1
  local _root _d
  _root="$(dirname "$(dirname "$1")")"   # <env>/bin/python -> <env>
  for _d in "$_root"/lib/python3.*/site-packages/pytorch_lightning; do
    [ -d "$_d" ] && return 0
  done
  return 1
}

_usable() {  # authoritative: can $1 import the training stack?
  [ -n "$1" ] && [ -x "$1" ] && "$1" -c "import torch, pytorch_lightning" >/dev/null 2>&1
}

# 1) honour an explicit PYTHON_BIN, 2) an already-active suitable env,
# 3) probe the known install roots directly (cheap), and only then
#    4) ask conda — `conda info --base` costs ~6s because conda boots Python.
if [ -z "${PYTHON_BIN}" ] && ! _looks_usable "$(command -v python 2>/dev/null)"; then
  # 3) direct path probe — cheap
  for _cand in "$HOME/miniconda3/envs/${CONDA_ENV_NAME}/bin/python" \
               "$HOME/anaconda3/envs/${CONDA_ENV_NAME}/bin/python" \
               "/data/L202500336/miniconda3/envs/${CONDA_ENV_NAME}/bin/python" \
               "/opt/conda/envs/${CONDA_ENV_NAME}/bin/python" \
               "/usr/local/envs/${CONDA_ENV_NAME}/bin/python"; do
    _looks_usable "${_cand}" && { PYTHON_BIN="${_cand}"; break; }
  done

  # 4) fall back to conda, if we still have not found one
  if [ -z "${PYTHON_BIN}" ]; then
    if ! command -v conda >/dev/null 2>&1; then
      for _c in "${CONDA_EXE:-}" "$HOME/miniconda3/bin/conda" "$HOME/anaconda3/bin/conda" \
                "/data/L202500336/miniconda3/bin/conda" "/opt/conda/bin/conda"; do
        [ -n "${_c}" ] && [ -x "${_c}" ] && { export PATH="$(dirname "${_c}"):${PATH}"; break; }
      done
    fi
    if command -v conda >/dev/null 2>&1; then
      _base="$(conda info --base 2>/dev/null || true)"
      if [ -n "${_base}" ]; then
        for _cand in "${_base}/envs/${CONDA_ENV_NAME}/bin/python" \
                     "${_base}/envs/${CONDA_ENV_NAME}/bin/python3"; do
          _looks_usable "${_cand}" && { PYTHON_BIN="${_cand}"; break; }
        done
      fi
    fi
  fi
fi

if [ -n "${PYTHON_BIN}" ]; then
  export PATH="$(dirname "${PYTHON_BIN}"):${PATH}"
else
  PYTHON_BIN="$(command -v python 2>/dev/null || echo python)"
fi

# Final gate. NOTE: we deliberately do NOT re-import torch here to verify —
# `python -c "import torch, pytorch_lightning"` costs ~50s on a loaded machine,
# which would dominate startup. `_looks_usable` already checked the env's
# site-packages (74ms); importing is the job of the stages themselves, which
# would fail loudly anyway.
if ! _looks_usable "${PYTHON_BIN}"; then
  echo "ERROR: no python with pytorch_lightning found." >&2
  echo "  Tried: PYTHON_BIN, the active env, conda env '${CONDA_ENV_NAME}', and common roots." >&2
  echo "  Fix:  conda activate ${CONDA_ENV_NAME}" >&2
  echo "     or: PYTHON_BIN=/path/to/envs/${CONDA_ENV_NAME}/bin/python $0 ..." >&2
  exit 1
fi

echo "BioDynaSpec data preparation"
echo "  PROJECT_ROOT : ${PROJECT_ROOT}"
echo "  python       : ${PYTHON_BIN}"
echo "  raw ATLAS    : ${ATLAS_RAW_DIR}"
echo "  output dir   : ${OUT_DIR}"
echo "  window       : W=${WINDOW_SIZE}  hop_ratio=${HOP_RATIO}  F=${NUM_FREQ}  dt=${DT_STR}"
echo "  stages       : ${STAGES[*]}"
[ "${DRY_RUN}" = "1" ] && echo "  (dry run — nothing will be executed)"

# ============================================================================
# Stage 0 — download ATLAS metadata + mmCIF structures  (optional, network)
# ============================================================================
stage_download() {
  banner "Stage 0/5 — download ATLAS metadata + mmCIF"
  echo "  Env: ATLAS_CSV_PATH=${ATLAS_CSV_PATH}  CIF_SAVE_DIR=${MMCIF_DIR}"
  if [ ! -f "${ATLAS_CSV_PATH}" ]; then
    echo "  [skip] metadata CSV not found: ${ATLAS_CSV_PATH}"
    echo "         fetch it from https://www.dsimb.inserm.fr/ATLAS and set ATLAS_CSV_PATH,"
    echo "         or drop prebuilt mmCIFs into ${MMCIF_DIR} and skip this stage."
    return 0
  fi
  run env ATLAS_CSV_PATH="${ATLAS_CSV_PATH}" CIF_SAVE_DIR="${MMCIF_DIR}" \
    python runner/data_prepare/download_atlas_cifs.py
}

# ============================================================================
# Stage 1 — extract frames from raw trajectories   [Cα only, paper Sec. 3.1]
# ============================================================================
stage_extract() {
  banner "Stage 1/5 — extract Cα frames from .tpr/.xtc"
  if [ ! -d "${ATLAS_RAW_DIR}" ]; then
    fail "raw ATLAS dir not found: ${ATLAS_RAW_DIR}" \
         "Set ATLAS_RAW_DIR to the directory holding <pdb>_<chain>/total/*.xtc"
  fi
  echo "  NOTE: USE_CA_ONLY=True in save_frames_to_npy_atlas.py (residue-level model)."
  echo "        Set it to False only if you deliberately want multi-atom frames."
  run env ATLAS_RAW_DIR="${ATLAS_RAW_DIR}" ATLAS_TRAJ_DIR="${TRAJ_DIR}" \
    python runner/data_prepare/extract_frames/save_frames_to_npy_atlas.py
}

# ============================================================================
# Stage 2 — mmCIF + frames -> bioassembly, seq indices, trajectory indices
# ============================================================================
stage_features() {
  banner "Stage 2/5 — build bioassembly + indices (Protenix data pipeline)"
  if [ ! -d "${MMCIF_DIR}" ]; then
    fail "mmCIF dir not found: ${MMCIF_DIR}" "Run the download stage, or set MMCIF_DIR."
  fi
  run python runner/data_prepare/2_prepare_data/prepare_md_training_data.py \
    --input_path "${MMCIF_DIR}" \
    --output_pdb_indices_csv "${PROJECT_ROOT}/data/indices/atlas_pdb_indices.csv" \
    --output_traj_indices_csv "${PROJECT_ROOT}/data/indices/atlas_traj_indices.csv" \
    --bio_output_dir "${BIOASSEMBLY_DIR}" \
    --fourier_output_dir "${PROJECT_ROOT}/data/fourier" \
    --seq_index_dir "${SEQ_INDEX_DIR}" \
    --dt_str "${DT_STR}" \
    --n_cpu "${NUM_WORKERS}"
  echo "  -> produces the trajectory list used by stage 4:"
  echo "     ${PROJECT_ROOT}/data/indices/atlas_traj_indices.csv  (columns: traj_name,pdb_name)"
}

# ============================================================================
# Stage 3 — ESM-2 embeddings  (1280-d, i.e. ESM-2-650M — matches the paper)
# ============================================================================
stage_esm() {
  banner "Stage 3/5 — compute ESM-2 embeddings (${ESM_MODEL_NAME})"
  if [ ! -f "${SEQ_INDEX_DIR}/pdb_seq.csv" ]; then
    fail "${SEQ_INDEX_DIR}/pdb_seq.csv not found — run the features stage first."
  fi
  if [ "${ESM_MODEL_NAME}" = "esm2-3b" ]; then
    echo "  WARNING: esm2-3b is 2560-d, but the model/config expect 1280 (ESM-2-650M)." >&2
    echo "           Paper Sec. 4.2 specifies ESM-2-650M. Re-run with esm2-650m unless" >&2
    echo "           you also change esm_dim and featureextractor.esm_feats_protein." >&2
  fi
  run python src/data/basic/compute_esm.py \
    --model_name "${ESM_MODEL_NAME}" \
    --pdb_seq_path "${SEQ_INDEX_DIR}/pdb_seq.csv" \
    --pdb_seq_label_path "${SEQ_INDEX_DIR}/pdb_labels_seqs.csv" \
    --esm_save_dir "${ESM_DIR}"
  echo "  -> ${ESM_DIR}/<shard>/<pdb>_<entity>.pt   (float32, (N_res, 1280))"
}

# ============================================================================
# Stage 4 — assemble the split .pth   *** this was the missing stage ***
# ============================================================================
stage_assemble() {
  banner "Stage 4/5 — assemble atlas_10_features_fouriered_list_{split}.pth"
  TRAJ_INDICES="${PROJECT_ROOT}/data/indices/atlas_traj_indices.csv"
  if [ ! -f "${TRAJ_INDICES}" ]; then
    fail "$TRAJ_INDICES not found — run the features stage first."
  fi
  # Optional: restrict to a subset of ids from a text file (one per line).
  SUBSET_ARGS=()
  if [ -n "${TRAJ_ID_SUBSET:-}" ]; then
    SUBSET_ARGS=(--max_proteins "$(wc -l < "${TRAJ_ID_SUBSET}")")
    echo "  subset file given (${TRAJ_ID_SUBSET}); consider --shard_size for large runs"
  fi
  run python runner/data_prepare/assemble_training_dataset.py \
    --traj_dir "${TRAJ_DIR}" \
    --bioassembly_dir "${BIOASSEMBLY_DIR}" \
    --esm_dir "${ESM_DIR}" \
    --indices_csv "${TRAJ_INDICES}" \
    --output_dir "${OUT_DIR}" \
    --window_size "${WINDOW_SIZE}" \
    --hop_ratio "${HOP_RATIO}" \
    --dt_str "${DT_STR}" \
    --esm_dim 1280 \
    --ratios ${SPLIT_RATIOS} \
    --seed "${SEED}" \
    --num_workers "${NUM_WORKERS}" \
    "${SUBSET_ARGS[@]+"${SUBSET_ARGS[@]}"}"
  echo "  -> ${OUT_DIR}/atlas_10_features_fouriered_list_{train,val,test}.pth"
}

# ============================================================================
# Stage 5 — per-frequency scaling factors + normalisation
# ============================================================================
stage_normalize() {
  banner "Stage 5/5 — per-frequency scaling factors + normalise"
  for split in train val test; do
    if [ ! -f "${OUT_DIR}/atlas_10_features_fouriered_list_${split}.pth" ]; then
      fail "missing ${OUT_DIR}/atlas_10_features_fouriered_list_${split}.pth" \
           "Run the assemble stage first."
    fi
  done
  echo "  NOTE: scaling factors must come from the TRAIN split only (no test leakage);"
  echo "        --fit_split train enforces this in compute_fft_window_scaling.py."
  run python runner/data_prepare/compute_fft_window_scaling.py \
    --data_dir "${OUT_DIR}" \
    --num_freq "${NUM_FREQ}" \
    --percentile "${SCALING_PCT}" \
    --output_path "${OUT_DIR}/${FFT_SUBDIR}_scaling_factors.pth" \
    --fit_split train \
    --recompute_sf \
    --normalize \
    --output_suffix _normalized
  echo
  echo "  Final training inputs:"
  for split in train val test; do
    echo "    ${OUT_DIR}/atlas_10_features_fouriered_list_${split}_normalized.pth"
  done
}

# ============================================================================
# Dispatch
# ============================================================================
for s in "${STAGES[@]}"; do
  case "$s" in
    download)  stage_download ;;
    extract)   stage_extract ;;
    features)  stage_features ;;
    esm)       stage_esm ;;
    assemble)  stage_assemble ;;
    normalize) stage_normalize ;;
    *) echo "Unknown stage: $s" >&2; exit 2 ;;
  esac
done

banner "Data preparation complete"
echo "Train with:"
echo "  bash scripts/pretrain.sh --config configs/trainer_BioDynaSpec.yaml"
echo
echo "Config expects these paths (configs/trainer_BioDynaSpec.yaml):"
echo "  train_path: ./${FFT_SUBDIR}/atlas_10_features_fouriered_list_train_normalized.pth"
echo "  val_path  : ./${FFT_SUBDIR}/atlas_10_features_fouriered_list_val_normalized.pth"
echo "(relative to DATASET_DIR=${DATASET_DIR}; make sure dataset/ resolves there)"
