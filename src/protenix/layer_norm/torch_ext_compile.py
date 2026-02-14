# Copyright 2024 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

try:
    from src.utils.cuda_arch_setup import setup_cuda_arch, CUDA_ARCH_FLAGS
except:
    print("警告：无法导入CUDA架构设置，使用默认值")
    CUDA_ARCH_FLAGS = [
        "-gencode",
        "arch=compute_70,code=sm_70",
        "-gencode",
        "arch=compute_80,code=sm_80",
        "-gencode",
        "arch=compute_86,code=sm_86",
        "-gencode",
        "arch=compute_90,code=sm_90",
    ]
    os.environ["TORCH_CUDA_ARCH_LIST"] = "7.0;8.0"

from torch.utils.cpp_extension import load

def compile(name, sources, extra_include_paths, build_directory):
    # 不再手动设置TORCH_CUDA_ARCH_LIST，由cuda_arch_setup模块设置
    
    # 基本的CUDA标志
    basic_cuda_flags = [
        "-O3",
        "--use_fast_math",
        "-DVERSION_GE_1_1",
        "-DVERSION_GE_1_3",
        "-DVERSION_GE_1_5",
        "-std=c++17",
        "-maxrregcount=50",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
    ]   
    cuda_flags = basic_cuda_flags + CUDA_ARCH_FLAGS
    return load(
        name=name,
        sources=sources,
        extra_include_paths=extra_include_paths,
        extra_cflags=[
            "-O3",
            "-DVERSION_GE_1_1",
            "-DVERSION_GE_1_3",
            "-DVERSION_GE_1_5",
        ],
        extra_cuda_cflags=cuda_flags,
        verbose=True,
        build_directory=build_directory,
    )