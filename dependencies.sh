#!/bin/bash
# ============================================================================
# BioDynaSpec environment setup
# ============================================================================
# Creates a conda env named "torch" (CUDA 12.4 / Python 3.11) with everything
# needed to run scripts/pretrain.sh and scripts/inference.sh.
#
# Usage:
#   bash dependencies.sh
#   conda activate torch
#
# After this, run training with the env on PATH (scripts/pretrain.sh calls a
# bare `python`, so the env must be active):
#   conda activate torch
#   bash scripts/pretrain.sh --config configs/trainer_BioDynaSpec.yaml
#
# NOTE ON THE ENV NAME: this used to be "A800_dl_cu124", a machine-specific
# name that did not match what the code/README assumed. It is now "torch".
# The pinned versions below mirror requirements.txt, which was read out of a
# known-working env (Python 3.11.15, torch 2.6.0+cu124).
# ============================================================================
set -euo pipefail
ENV_NAME="torch"
conda_channels=(
  "https://mirrors.ustc.edu.cn/anaconda/pkgs/main"
  "https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge"
  "nvidia/label/cuda-12.4.1"
  "defaults"
)

# ============================================================================
# Locate conda explicitly.
# ============================================================================
# `bash dependencies.sh` starts a NON-interactive shell, which does NOT source
# ~/.bashrc — so the conda-init block there never runs and a bare `conda` may
# not be on PATH. Resolve it here, from CONDA_EXE / known install roots, and
# fail with an actionable message instead of a bare "command not found".
if ! command -v conda >/dev/null 2>&1; then
  for _c in \
      "${CONDA_EXE:-}" \
      "$HOME/miniconda3/bin/conda" \
      "$HOME/anaconda3/bin/conda" \
      "/data/L202500336/miniconda3/bin/conda" \
      "/opt/conda/bin/conda"
  do
    if [ -n "${_c}" ] && [ -x "${_c}" ]; then
      export PATH="$(dirname "${_c}"):${PATH}"
      break
    fi
  done
fi
if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda not found on PATH."
  echo "  This script is run non-interactively, so ~/.bashrc (and any conda"
  echo "  init block in it) is NOT sourced."
  echo "  Fix by exporting PATH first, e.g.:"
  echo "      export PATH=/data/L202500336/miniconda3/bin:\$PATH   # adjust to your install"
  echo "      bash dependencies.sh"
  exit 1
fi
CONDA_BIN="$(command -v conda)"
CONDA_BASE="$(dirname "$(dirname "${CONDA_BIN}")")"
echo "Using conda: ${CONDA_BIN}"
echo "Conda base : ${CONDA_BASE}"
echo "Target env : ${ENV_NAME}  (will live at ${CONDA_BASE}/envs/${ENV_NAME})"

# ==============Step1: 下载gcc13.3.0==============
export GCC_ROOT=$HOME/software/gcc-13.3.0 # 选择一个你有权限的安装路径
mkdir -p $GCC_ROOT

#  ⚠️ gcc 13 不能用太老的 gcc 编译
cd ~/software/src
wget https://ftp.gnu.org/gnu/gcc/gcc-13.3.0/gcc-13.3.0.tar.xz
tar -xf gcc-13.3.0.tar.xz
cd gcc-13.3.0
# 下载依赖（官方脚本，强烈推荐）
./contrib/download_prerequisites
# 独立 build 目录（必须）
mkdir build && cd build
# 配置
../configure \
  --prefix=$GCC_ROOT \
  --enable-languages=c,c++,fortran \
  --disable-multilib
# 编译 & 安装（⏳ 30–90 分钟）
make -j$(nproc)
make install

export PATH=$GCC_ROOT/bin:$PATH
export LD_LIBRARY_PATH=$GCC_ROOT/lib64:$LD_LIBRARY_PATH

gcc --version
# gcc (GCC) 13.3.0

# ==============Step2: 创建conda环境==============

conda create -n "${ENV_NAME}" \
  python=3.11 pip=25.3 \
  -y \
  $(printf -- "--channel %s " "${conda_channels[@]}")
# 2. 安装系统级依赖（GCC/BLAS/CUDA）
conda install -n "${ENV_NAME}" -y \
  --channel nvidia/label/cuda-12.4.1 \
  openblas=0.3.30 \
  libgcc-ng=14.2.0 \
  libgfortran5=14.2.0
# 3. 安装常用科学/深度学习库
conda install -n "${ENV_NAME}" -y \
  --channel conda-forge \
  numpy=1.26.4 scipy=1.16.3 pandas=1.5.3 matplotlib=3.10.1 \
  pillow=10.4.0 seaborn=0.13.2 \
  pytorch=2.6.0 torchvision=0.21.0 torchaudio=2.6.0 \
  jax=0.4.29 jaxlib=0.4.29+cuda12.cudnn91 \
  transformers=4.51.3
# 4. 安装 CUDA 工具链、NVIDIA 库
conda install -n "${ENV_NAME}" -y \
  --channel nvidia/label/cuda-12.4.1 \
  cuda-toolkit=12.4 \
  cuda-cudart=12.9.37 \
  cuda-nvprof=12.9.19 cuda-cupti=12.9.19 \
  cuda-nvcc=12.4.131 cuda-nvrtc=12.9.41 \
  libcublas=12.9.0.13 libcudnn=9.1.0.70
# ==============Step3: Python 依赖（与 requirements.txt 对齐）==============
# 这些 pin 取自一个已验证可用的环境（Python 3.11.15 + torch 2.6.0+cu124），
# 不是估计值。torch 必须先从 CUDA 索引装，否则会拿到 CPU 版。
conda run -n "${ENV_NAME}" pip install --no-cache-dir \
  torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124

conda run -n "${ENV_NAME}" pip install --no-cache-dir -r requirements.txt

# ==============Step4: flash-attn（可选，但推荐）==============
# 需要 CUDA 工具链（见 Step1 的 gcc 与下面的 CUDA 说明）。编译较慢。
# 若编译失败，BioDynaSpec 会优雅降级为普通 attention（见
# src/protein_spectrum_far.py 的 try/except），不影响训练正确性。
conda run -n "${ENV_NAME}" pip install --no-build-isolation flash-attn==2.7.2.post1 || \
  echo "[warn] flash-attn 安装失败，将回退到标准 attention"

# ==============说明==============
# * 不需要 torchdrug：它原先只是 src/data/dataset_atlas.py 里一条死代码路径
#   （collate_fn / data.PackedGraph）的依赖，DataModule 实际用的是
#   collate_fn_single。该 import 已移除。
# * 不需要 openfold：仓库自带 src/openfold_local/，根目录的 openfold.py 是
#   fair-esm 的 esmfold 需要顶层 openfold 包时用的 shim（详见 requirements.txt
#   的 notes）。请勿 pip install openfold，会和 shim 冲突。
# * gemmi 可选：只用于 extract_frames 的 CIF 导出路径，训练不需要。
#
# 验证安装：
#   conda activate ${ENV_NAME}
#   python -c "import torch, pytorch_lightning; print(torch.__version__)"
#   python check_imports.py --chain     # 应输出 "BROKEN imports ... 0"
