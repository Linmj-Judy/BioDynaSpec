import os
import subprocess
import re

def get_gpu_compute_capability():
    """获取当前系统中的GPU计算能力"""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True
        )
        output = result.stdout.strip().split("\n")
        
        # 提取所有GPU的计算能力
        compute_caps = []
        for line in output:
            match = re.search(r'(\d+\.\d+)$', line.strip())
            if match:
                compute_caps.append(match.group(1))
        
        # 去重并排序
        unique_caps = sorted(set(compute_caps))
        
        if not unique_caps:
            print("警告：无法检测到GPU计算能力，使用默认值")
            return "8.0"
            
        return unique_caps
    except Exception as e:
        print(f"警告：自动检测GPU失败 ({str(e)})，使用默认值")
        return ["8.0"]

def setup_cuda_arch():
    """设置CUDA架构环境变量和返回编译标志"""
    compute_caps = get_gpu_compute_capability()
    
    # 设置环境变量
    os.environ["TORCH_CUDA_ARCH_LIST"] = ";".join(compute_caps)
    
    # 生成编译标志
    compile_flags = []
    for cap in compute_caps:
        cap_no_dot = cap.replace('.', '')
        compile_flags.extend([
            "-gencode", f"arch=compute_{cap_no_dot},code=sm_{cap_no_dot}"
        ])
    
    print(f"已设置CUDA架构为: {os.environ['TORCH_CUDA_ARCH_LIST']}")
    return compile_flags

# 自动设置CUDA架构
CUDA_ARCH_FLAGS = setup_cuda_arch()