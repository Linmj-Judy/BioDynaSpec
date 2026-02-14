import torch
import torch.nn as nn
import numpy as np
import random 
import os  
import sys  
import logging  
import sys  
import threading  
import multiprocessing  
from typing import Optional  
import platform
import torch.distributed as dist
import pickle


# 跨平台资源管理  
try:  
    import resource  
except:  
    resource = None  

try:  
    import psutil  
except:  
    psutil = None  


# def get_device():  
#     if torch.cuda.is_available():  
#         # 获取第一个可用GPU  
#         device = torch.device("cuda:0")  
#         print(f"Using GPU: {torch.cuda.get_device_name(0)}")  
#     else:  
#         device = torch.device("cpu")  
#         print("Using CPU")  
#     return device  
def get_device(local_rank: int = -1, is_distributed: bool = False):  
    """  
    根据训练模式选择设备  
    
    Args:  
        local_rank (int): 本地进程号，用于分布式训练  
        is_distributed (bool): 是否为分布式训练  
    
    Returns:  
        tuple: (device, rank, world_size, local_rank)
            - device: 选择的训练设备
            - rank: 全局进程号
            - world_size: 总进程数
            - local_rank: 本地进程号
    """  
    # 初始化分布式训练参数
    rank = 0
    world_size = 1
    
    # 分布式训练场景  
    if is_distributed:  
        if not torch.cuda.is_available():  
            raise RuntimeError("Distributed training requires CUDA")  
        
        # 从环境变量获取分布式训练参数
        if 'WORLD_SIZE' in os.environ:
            world_size = int(os.environ['WORLD_SIZE'])
            rank = int(os.environ['RANK'])
            local_rank = int(os.environ['LOCAL_RANK'])
        
        # 使用进程对应的 GPU  
        device = torch.device(f"cuda:{local_rank}")  
        print(f"Distributed Training on GPU {local_rank}: {torch.cuda.get_device_name(local_rank)}")  
        return device, rank, world_size, local_rank
    
    # 单机多卡训练  
    if torch.cuda.device_count() > 1:  
        print(f"Using {torch.cuda.device_count()} GPUs")  
        device = torch.device("cuda:0")  
        return device, rank, world_size, local_rank
    
    # 单卡或CPU训练  
    if torch.cuda.is_available():  
        device = torch.device("cuda:0")  
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")  
    else:  
        device = torch.device("cpu")  
        print("Using CPU")  
    
    return device, rank, world_size, local_rank

def setup_distributed(backend='nccl'):
    """初始化分布式训练环境"""
    if 'WORLD_SIZE' in os.environ:
        # 使用环境变量初始化
        world_size = int(os.environ['WORLD_SIZE'])
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        # 默认为单进程
        world_size = 1
        rank = 0
        local_rank = 0
    
    # 设置当前设备
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")
        
    # 初始化进程组
    if world_size > 1:
        dist.init_process_group(backend=backend)
    
    return device, rank, world_size, local_rank

def set_file_limit_linux():  
    if resource is None:  
        print("resource module is not available")  
        return
    try:  
        # 尝试获取当前软限制  
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)  
        
        # 尝试设置为当前的最大值  
        try:  
            resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))  
            print(f"File descriptor limit set to {hard}")  
        except ValueError:  
            print(f"Cannot set to hard limit. Using current soft limit of {soft}")  
            resource.setrlimit(resource.RLIMIT_NOFILE, (soft, soft))  
    except Exception as e:  
        print(f"Error setting file descriptor limit: {e}")  


def get_system_type():  
    """获取当前操作系统类型"""  
    return platform.system().lower()

def set_file_limit_windows():  
    """Windows系统资源管理"""  
    if psutil is None:  
        print("psutil模块不可用，无法进行资源管理")  
        return  

    try:  
        # 获取当前进程  
        current_process = psutil.Process()  
        
        # 设置最大打开文件数(Windows下是句柄数)  
        try:  
            current_process.as_dict(attrs=['num_handles'])  
            print(f"当前进程句柄数: {current_process.num_handles()}")  
        except Exception as e:  
            print(f"获取句柄数失败: {e}")  

    except Exception as e:  
        print(f"Windows资源管理设置错误: {e}")  

def set_file_limit():  
    """根据系统类型选择资源限制方案"""  
    system_type = get_system_type()  
    
    if system_type == 'linux':  
        set_file_limit_linux()  
    elif system_type == 'windows':  
        set_file_limit_windows()  
    else:  
        print(f"不支持的操作系统类型: {system_type}")  


def set_seed(seed: int = 42):  
    torch.manual_seed(seed)  
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = False  
    np.random.seed(seed)  
    random.seed(seed)

def save_object(obj, filename):
    with open(filename, 'wb') as file:
        pickle.dump(obj, file)

def load_object(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)
