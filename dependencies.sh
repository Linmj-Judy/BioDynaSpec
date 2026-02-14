#!/bin/bash
set -euo pipefail
ENV_NAME="A800_dl_cu124"
conda_channels=(
  "https://mirrors.ustc.edu.cn/anaconda/pkgs/main"
  "https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge"
  "nvidia/label/cuda-12.4.1"
  "defaults"
)

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
# 5. 通过 pip 安装剩余 python 包（将 requirements.txt 拆分为核心/额外两组）
pip install --no-cache-dir \
  absl-py==2.3.1 accelerate==1.5.2 \
  datasets==3.6.0 deepspeed==0.16.5 \
  jupyterlab==4.4.7 \
  fastapi==0.120.4 \
  tensorflow-cpu==2.19.0 \
  torchmetrics==1.7.1 \
  safetensors==0.5.3 \
  wandb==0.19.9 \
  pymatgen==2024.2.23 \
conda install -c conda-forge rdkit

CUDA_INSTALL_PATH=/share/software/cuda/local/cuda-12.4
export CUDACXX=${CUDA_INSTALL_PATH}/bin/nvcc
git clone https://github.com/NVIDIA/cutlass.git
cd cutlass
mkdir build && cd build
cmake .. -DCUTLASS_NVCC_ARCHS=80               # compiles for NVIDIA's Ampere Architecture

# pip install "git+https://github.com/MDAnalysis/mdanalysis@develop"
pip install --upgrade MDAnalysis

pip install git+https://github.com/TimoLassmann/kalign.git

# cd torchdrug
# # pip install -r requirements.txt
# python setup.py install
py_version=$(python -c "import torch;print(torch.__version__.split('+')[0])")
cuda_version=$(nvcc --version | grep release | awk '{print $6}' | awk -F '.' '{print $1"."$2}')
pip install torch-scatter -f https://data.pyg.org/whl/torch-${py_version}+cu${cuda_version}.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-${py_version}+cu${cuda_version}.html
pip install torch-cluster -f https://data.pyg.org/whl/torch-${py_version}+cu${cuda_version}.html

pip install deepspeed==0.16.5
pip install pytorch_lightning==2.5.1.post0
pip install ray==2.51.1
pip install "ray[tune]"
pip install ml_collections==1.1.0
pip install rdkit==2025.3.6
pip install optree==0.15.0
pip install biotite==1.2.0
pip install -U 'wandb>=0.12.10'
pip install tokenizers==0.21.1
pip install transformers==4.51.3
pip install fair-esm
pip install easydict
pip install lmdb
pip install torchdrug
pip install dm-tree
pip install modelcif