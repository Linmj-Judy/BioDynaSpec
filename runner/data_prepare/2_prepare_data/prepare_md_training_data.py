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

import argparse
import csv
import os
import re
import sys
import logging
import time
import threading
import multiprocessing as mp
import signal
import traceback
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed, wait
import gc

import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
import torch
import numpy as np
import time

# 改进的`move_to_cpu`函数，添加缓存和优化
import functools

@functools.lru_cache(maxsize=128)
def get_cpu_converter(dtype, device):
    """缓存CPU转换器，避免重复创建"""
    pass

def move_to_cpu(obj, inplace=False):
    """优化版本的move_to_cpu"""
    if torch.is_tensor(obj):
        # 使用non_blocking异步传输
        cpu_tensor = obj.cpu() if not inplace else obj.cpu()
        # 如果不需要梯度，可以detach
        if not obj.requires_grad:
            cpu_tensor = cpu_tensor.detach()
        return cpu_tensor
    
    # 对于常见类型使用更高效的实现
    if isinstance(obj, dict):
        # 使用字典推导式而不是递归
        return {k: move_to_cpu(v, inplace) for k, v in obj.items()}
    
    if isinstance(obj, list):
        # 使用列表推导式
        return [move_to_cpu(v, inplace) for v in obj]
    
    if isinstance(obj, tuple):
        # 元组特殊处理
        return tuple(move_to_cpu(v, inplace) for v in obj)
    
    # 对于NumPy数组，不需要处理
    if isinstance(obj, np.ndarray):
        return obj
    
    return obj

from src.data.basic.data_pipeline import DataPipeline
from src.utils.file_io import dump_gzip_pickle, load_gzip_pickle
from src.data.fourier.utils import TrajectoryFFTProcessor
from src.data.basic.get_pdb_seqs import export_to_fasta, mapping_seqs_to_pdb_entity_id, mapping_seqs_to_integer_identifiers

# 配置日志，确保输出立即刷新
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# 设置信号处理，用于SLURM超时警告
def signal_handler(signum, frame):
    """处理SLURM发送的超时警告信号"""
    logger.warning(f"[WARNING] 收到信号 {signum}，任务即将超时，但仍在运行中...")
    sys.stdout.flush()

# 注册信号处理器（如果SLURM配置了--signal）
if hasattr(signal, 'SIGUSR1'):
    signal.signal(signal.SIGUSR1, signal_handler)
if hasattr(signal, 'SIGUSR2'):
    signal.signal(signal.SIGUSR2, signal_handler)

def extract_pdb_id_from_filename(filename: str) -> str:
    """
    从文件名中提取 PDB ID，与 gen_a_bioassembly_data 中的逻辑保持一致。
    
    Args:
        filename: 文件名（可以是完整路径或文件名）
    
    Returns:
        pdb_id: 提取的 PDB ID（小写）
    """
    # 获取文件名（不含扩展名）
    if '/' in filename or '\\' in filename:
        stem = Path(filename).stem
    else:
        stem = Path(filename).stem
    
    # 只使用前6个字符（与 gen_a_bioassembly_data 保持一致）
    filename_6 = stem[:6]
    
    # 匹配4字母代码 + 可选的链标识符
    match = re.match(r"([0-9a-zA-Z]{4})(?:[_\-].+)?", filename_6)
    if match:
        # 如果匹配，使用整个6字符（包含链标识符）
        pdb_id = filename_6.lower()
    else:
        # 否则只使用前4个字符
        pdb_id = filename_6.lower()[:4]
    
    return pdb_id

def find_bioassembly_dict(traj_filename: str, bioassembly_dict_DICT: dict) -> dict:
    """
    从轨迹文件名中提取 pdb_id，并在 bioassembly_dict_DICT 中查找对应的字典。
    如果找不到，尝试多种可能的格式。
    
    Args:
        traj_filename: 轨迹文件名（如 "5w82_b_r3.npy"）
        bioassembly_dict_DICT: bioassembly 字典的字典
    
    Returns:
        bioassembly_dict: 找到的字典，如果找不到则返回空字典
    """
    # 方法1: 提取前6个字符（标准方法）
    pdb_id_6 = extract_pdb_id_from_filename(traj_filename)
    if pdb_id_6 in bioassembly_dict_DICT:
        return bioassembly_dict_DICT[pdb_id_6]
    
    # 方法2: 尝试前4个字符（如果6字符找不到）
    pdb_id_4 = pdb_id_6[:4] if len(pdb_id_6) >= 4 else pdb_id_6
    if pdb_id_4 in bioassembly_dict_DICT:
        return bioassembly_dict_DICT[pdb_id_4]
    
    # 方法3: 尝试移除下划线（5w82_b -> 5w82b）
    pdb_id_no_underscore = pdb_id_6.replace('_', '')
    if pdb_id_no_underscore in bioassembly_dict_DICT:
        return bioassembly_dict_DICT[pdb_id_no_underscore]
    
    # 方法4: 尝试所有可能的变体
    # 例如：5w82_b -> 5w82b, 5w82_b, 5w82
    variants = [
        pdb_id_6,
        pdb_id_4,
        pdb_id_no_underscore,
        pdb_id_6.replace('-', ''),
        pdb_id_6.replace('_', '').replace('-', ''),
    ]
    
    for variant in variants:
        if variant in bioassembly_dict_DICT:
            logger.warning(f"[WARNING] 轨迹 {traj_filename} 使用变体 {variant} 找到 bioassembly_dict（原始尝试: {pdb_id_6}）")
            return bioassembly_dict_DICT[variant]
    
    # 如果都找不到，返回空字典并记录警告
    logger.warning(f"[WARNING] 轨迹 {traj_filename} 无法找到对应的 bioassembly_dict。尝试的 key: {variants}")
    logger.warning(f"[WARNING] 可用的 bioassembly_dict keys (前10个): {list(bioassembly_dict_DICT.keys())[:10]}")
    return {}

def gen_a_bioassembly_data(
    pdb: Path,
    bioassembly_output_dir: Path,
    cluster_file: Optional[Path],
) -> Optional[Tuple[Optional[list[dict]], Optional[dict], Optional[pd.DataFrame]]]:
    """
    Generates bioassembly data from a structure file and saves it to the specified output directory.

    Args:
        pdb (Path): Path to the structure (e.g. PDB or mmCIF) file.
        bioassembly_output_dir (Path): Directory where the bioassembly data will be saved.
        cluster_file (Optional[Path]): Path to the cluster file, if available.

    Returns:
        Optional[Tuple[list[dict], dict, pd.DataFrame]]: sample indices, bioassembly_dict,
            info_df；若生成失败则返回 None。
    """
    start_time = time.time()
    logger.info(f"[DEBUG] 开始处理: {pdb}")
    sys.stdout.flush()
    
    # Extract PDB ID from filename, support chain or custom IDs (e.g., 1ear_A, 1ear_chainA)
    filename = pdb.stem[:6] # only use the first 6 characters
    # Typical 4-letter code, e.g. 1ear or 1ear_A or custom
    # Use everything up to first non-alnum/underscore as base id, otherwise fallback to [:4]
    match = re.match(r"([0-9a-zA-Z]{4})(?:[_\-].+)?", filename)
    if match:
        # If filename starts with 4-char code, capture chain/extra info if present
        pdb_id = filename.lower()
    else:
        # Fallback to base, just the first 4 chars
        pdb_id = filename.lower()[:4]

    if os.path.exists(bioassembly_output_dir / f"{pdb_id}.pkl.gz"):
        try:
            bioassembly_dict = load_gzip_pickle(bioassembly_output_dir / f"{pdb_id}.pkl.gz")
            # 验证加载的数据是否有效
            if bioassembly_dict is None or not isinstance(bioassembly_dict, dict):
                logger.warning(f"[WARNING] 缓存文件 {pdb_id}.pkl.gz 数据无效，将重新处理")
            elif bioassembly_dict.get("atom_array") is None:
                logger.warning(f"[WARNING] 缓存文件 {pdb_id}.pkl.gz 中 atom_array 为 None，将重新处理")
            else:
                return (None, bioassembly_dict, None)
        except Exception as e:
            logger.warning(f"[WARNING] 加载缓存文件失败: {e}，将重新处理")
    
    try:
        sample_indices_list, bioassembly_dict, info_df = DataPipeline.get_data_from_pdb(
            pdb, pdb_cluster_file=cluster_file,
            apply_filters=True
        )
        elapsed = time.time() - start_time
        logger.info(f"[DEBUG] 处理完成: {pdb}, 耗时: {elapsed:.2f}秒")
        logger.info(f"[DEBUG] sample_indices_list长度: {len(sample_indices_list) if sample_indices_list else 0}")
        sys.stdout.flush()
        
        # 验证 bioassembly_dict 和 atom_array
        if bioassembly_dict is None:
            logger.warning(f"[WARNING] {pdb} bioassembly_dict 为 None")
            sys.stdout.flush()
            return None
        
        atom_array = bioassembly_dict.get("atom_array")
        if atom_array is None:
            logger.warning(f"[WARNING] {pdb} atom_array 为 None (PDB Code: {bioassembly_dict.get('pdb_id', 'unknown')})")
            logger.warning(f"[WARNING] 可能原因: 链数过多或为0, num_assembly_polymer_chains={bioassembly_dict.get('num_assembly_polymer_chains', 'unknown')}")
            sys.stdout.flush()
            return None
        
        cond_indices = sample_indices_list is not None and len(sample_indices_list) > 0
        cond_bio = bioassembly_dict is not None and atom_array is not None
        cond_info = (info_df is not None) and (not info_df.empty)
        
        if cond_indices and cond_bio and cond_info:
            pdb_id = bioassembly_dict["pdb_id"]
            # Save bioassembly_dict as a compressed pickle file
            dump_gzip_pickle(bioassembly_dict, bioassembly_output_dir / f"{pdb_id}.pkl.gz")
            return sample_indices_list, bioassembly_dict, info_df
        else:
            logger.warning(f"[WARNING] {pdb} 返回空结果: "
                  f"sample_indices_list_nonempty={cond_indices}, "
                  f"bioassembly_dict_exists={cond_bio}, "
                  f"atom_array_exists={atom_array is not None}, "
                  f"info_df_nonempty={cond_info}")
            sys.stdout.flush()
            return None
    except SystemExit:
        # 不要捕获 SystemExit，让它正常退出
        raise
    except KeyboardInterrupt:
        # 不要捕获 KeyboardInterrupt
        raise
    except Exception as e:
        elapsed = time.time() - start_time
        error_type = type(e).__name__
        error_msg = str(e)
        logger.error(f"[ERROR] 处理失败: {pdb}, 耗时: {elapsed:.2f}秒")
        logger.error(f"[ERROR] 错误类型: {error_type}, 错误信息: {error_msg}")
        
        # 对于段错误等严重错误，记录更多信息但不重新抛出，避免中断整个流程
        if "Segmentation fault" in error_msg or "segmentation" in error_msg.lower():
            logger.error(f"[ERROR] 检测到段错误，这可能是由于 PDB 文件损坏或 biotite 库的 bug")
            logger.error(f"[ERROR] 跳过此文件: {pdb}")
            import traceback
            traceback.print_exc()
            sys.stdout.flush()
            return None  # 返回 None 而不是 raise，避免中断整个流程
        else:
            import traceback
            traceback.print_exc()
            sys.stdout.flush()
            # 对于其他异常，仍然抛出，但会被 safe_run_func_with_exception 捕获
            raise

def gen_a_trajectory_data(
    trajectory_file: Path,
    traj_name: str,
    fourier_output_dir: Path,
    bioassembly_dict: dict,
    processor: Optional[TrajectoryFFTProcessor] = None,
    processor_config: Optional[dict] = None,
    dt_str: str = "10ps",
):
    """
    Generates trajectory data from a trajectory file and saves the results to a directory.
    
    Args:
        processor: If None, will create a new processor using processor_config (safer for multiprocessing)
        processor_config: Config dict for creating processor if processor is None
    """
    import time
    start_time = time.time()
    logger.info(f"[DEBUG] 开始处理轨迹: {traj_name}")
    sys.stdout.flush()
    
    # 检查 bioassembly_dict 是否为空
    if not bioassembly_dict or len(bioassembly_dict) == 0:
        elapsed = time.time() - start_time
        logger.warning(f"[WARNING] 轨迹 {traj_name} 跳过处理：找不到对应的 bioassembly_dict，耗时: {elapsed:.2f}秒")
        sys.stdout.flush()
        return None  # 返回 None 表示跳过
    
    # 如果 processor 为 None，在子进程中创建（避免 GPU 资源竞争）
    if processor is None:
        if processor_config is None:
            raise ValueError("Either processor or processor_config must be provided")
        processor = DataPipeline.init_trajectory_fft_processor(**processor_config)
    
    try:
        logger.info(f"[INFO] 开始GPU FFT计算: {traj_name}")
        sys.stdout.flush()
        
        fft_start_time = time.time()
        package = DataPipeline.get_trajectory_fft(
            trajectory_file,
            processor,
            bioassembly_dict,
            remove_com_for_alignment=True,
            align_rotation_for_alignment=False,
            dt_str=dt_str,
        )
        fft_elapsed = time.time() - fft_start_time
        logger.info(f"[INFO] GPU FFT计算完成: {traj_name}, FFT耗时: {fft_elapsed:.2f}秒")
        sys.stdout.flush()
        
        save_start_time = time.time()
        # 确保所有张量都在CPU上再保存
        cpu_package = move_to_cpu(package)
        dump_gzip_pickle(cpu_package, fourier_output_dir / f"{traj_name}.pkl.gz")
        save_elapsed = time.time() - save_start_time
        elapsed = time.time() - start_time
        logger.info(f"[INFO] 轨迹处理完成: {traj_name}, 总耗时: {elapsed:.2f}秒 (FFT: {fft_elapsed:.2f}秒, 保存: {save_elapsed:.2f}秒)")
        sys.stdout.flush()
        return traj_name
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"[ERROR] 轨迹处理失败: {traj_name}, 耗时: {elapsed:.2f}秒, 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        raise

def get_pdb_seqs(
    info_dfs: list[pd.DataFrame],
    out_dir: Path,
):
    """
    Get pdb sequence info from a list of info dataframes.
    如果文件已存在，会在原有基础上合并，而不是替换。
    """
    
    out_df = pd.concat(info_dfs)
    out_df = out_df.sort_values(["pdb_id", "entity_id"])

    if not out_dir.exists():
        out_dir.mkdir(parents=True)
    
    # 1. extract pdb sequence info (合并模式)
    seq_file = out_dir / "pdb_seq.csv"
    seq_df = out_df[out_df.mol_type != "other"]
    
    # 如果文件已存在，合并数据
    if seq_file.exists():
        existing_df = pd.read_csv(seq_file)
        # 合并并去重（基于 pdb_id 和 entity_id）
        combined_df = pd.concat([existing_df, seq_df])
        # 去重：保留最新的数据（如果同一 pdb_id + entity_id 出现多次）
        combined_df = combined_df.drop_duplicates(
            subset=["pdb_id", "entity_id"], 
            keep="last"
        )
        combined_df = combined_df.sort_values(["pdb_id", "entity_id"])
        seq_df = combined_df
        logger.info(f"[INFO] 合并现有序列数据，原有 {len(existing_df)} 条，新增 {len(out_df[out_df.mol_type != 'other'])} 条，合并后 {len(seq_df)} 条")
    else:
        logger.info(f"[INFO] 创建新的序列数据文件，包含 {len(seq_df)} 条记录")
    
    seq_df.to_csv(seq_file, index=False)
    
    # 2. generate protein fasta file as MSA input (追加模式)
    fasta_file = out_dir / "pdb_seq.fasta"
    export_to_fasta(seq_df, fasta_file, append=False)  # 重新生成整个文件，因为已经合并了 seq_df
    
    # 3. get seq_to_pdb_id_entity_id mapping (合并模式)
    seq_to_pdb_id_entity_id_json = out_dir / "seq_to_pdb_id_entity_id.json"
    sequence_mapping = mapping_seqs_to_pdb_entity_id(
        seq_df, seq_to_pdb_id_entity_id_json, merge_existing=True
    )

    # 4. mapping sequence with integers identifiers for saving MSA (合并模式)
    pdb_index_to_seq_path = out_dir / "pdb_index_to_seq.json"
    seq_to_pdb_index_path = out_dir / "seq_to_pdb_index.json"
    mapping_seqs_to_integer_identifiers(
        sequence_mapping, pdb_index_to_seq_path, seq_to_pdb_index_path, merge_existing=True
    )



def cpu_preprocess_trajectory(
    trajectory_file: Path,
    bioassembly_dict: dict,
    dt_str: str = "10ps",
) -> Optional[Tuple[str, np.ndarray, Dict[str, Any]]]:
    """
    CPU预处理轨迹：加载数据、过滤、计算质量等
    返回: (traj_name, preprocessed_traj, metadata) 或 None
    """
    try:
        traj_name = str(trajectory_file.name)[:9].lower()
        
        if not bioassembly_dict or len(bioassembly_dict) == 0:
            return None
        
        # 加载轨迹数据（CPU）
        arr = np.load(trajectory_file)  # (T, N, 3)
        traj = torch.from_numpy(arr).to(torch.float32)
        
        # 应用过滤
        filter_indices = bioassembly_dict.get("filter_indices", None)
        apply_filters = bioassembly_dict.get("apply_filters", True)
        
        T, N_traj, _ = traj.shape
        atom_array = bioassembly_dict["atom_array"]
        N_ref = len(atom_array)
        
        if apply_filters and filter_indices is not None:
            if N_traj >= len(filter_indices):
                traj = traj[:, filter_indices, :]
            else:
                return None
        elif N_traj != N_ref:
            return None
        
        # 计算质量（CPU）
        from src.data.basic.data_pipeline import Featurizer
        ref_element = Featurizer.elem_onehot_encoded(atom_array.element).long()
        ref_masses = Featurizer.masses_from_ref_element(ref_element, unknown_mass=12.011)
        
        # 准备元数据
        from src.data.fourier.utils import parse_time_str_to_seconds
        dt = parse_time_str_to_seconds(dt_str)
        
        metadata = {
            "traj_name": traj_name,
            "bioassembly_dict": bioassembly_dict,
            "masses": ref_masses,
            "dt": dt,
            "dt_str": dt_str,
        }
        
        return (traj_name, traj.numpy(), metadata)
    
    except Exception as e:
        logger.error(f"[ERROR] CPU预处理失败 {trajectory_file}: {e}")
        return None


def gpu_preprocess_trajectory(
    trajectory_file: Path,
    bioassembly_dict: dict,
    dt_str: str = "10ps",
    device: Optional[torch.device] = None,
) -> Optional[Tuple[str, torch.Tensor, Dict[str, Any]]]:
    """
    GPU预处理轨迹：直接在GPU上完成过滤和质量计算，避免CPU/GPU往返。
    返回: (traj_name, traj_tensor_on_gpu, metadata) 或 None
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{torch.cuda.current_device()}")
        else:
            device = torch.device("cpu")
    if device.type != "cuda":
        # 如果没有可用GPU，退回CPU预处理
        cpu_result = cpu_preprocess_trajectory(trajectory_file, bioassembly_dict, dt_str)
        if cpu_result is None:
            return None
        traj_name, traj_np, metadata = cpu_result
        metadata["masses"] = torch.as_tensor(metadata["masses"]).to(device)
        return traj_name, torch.from_numpy(traj_np).to(device, non_blocking=True), metadata

    try:
        traj_name = str(trajectory_file.name)[:9].lower()
        if not bioassembly_dict or len(bioassembly_dict) == 0:
            return None

        # 直接加载到 GPU（先转为 torch，再搬移）
        arr = np.load(trajectory_file)  # (T, N, 3) 仍由CPU从磁盘读取
        # 使用同步传输确保完全转移到 GPU
        # 先创建 tensor，然后显式转移到 GPU
        traj = torch.from_numpy(arr).float()
        traj = traj.to(device=device, non_blocking=False)
        # 同步 CUDA 操作，确保转移完成
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        # 显式验证 traj 在正确的设备上
        if traj.device != device:
            logger.warning(f"[WARNING] traj 设备不匹配，期望 {device}，实际 {traj.device}，强制转移")
            traj = traj.to(device=device, non_blocking=False)
            if device.type == "cuda":
                torch.cuda.synchronize(device)

        filter_indices = bioassembly_dict.get("filter_indices", None)
        apply_filters = bioassembly_dict.get("apply_filters", True)

        T, N_traj, _ = traj.shape
        atom_array = bioassembly_dict["atom_array"]
        N_ref = len(atom_array)

        if apply_filters and filter_indices is not None:
            if N_traj >= len(filter_indices):
                # 在GPU上索引，确保 indices 与 positions 在同一设备
                # 使用同步传输确保 filter_tensor 在 GPU 上
                if isinstance(filter_indices, torch.Tensor):
                    filter_tensor = filter_indices.clone().detach().to(device=device, dtype=torch.long, non_blocking=False)
                else:
                    # torch.tensor 会复制到指定 device，避免 as_tensor 在已有 tensor 时忽略 device
                    filter_tensor = torch.tensor(filter_indices, device=device, dtype=torch.long)
                # 同步 CUDA 操作
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                # 显式验证 filter_tensor 在正确的设备上
                if filter_tensor.device != device:
                    logger.warning(f"[WARNING] filter_tensor 设备不匹配，期望 {device}，实际 {filter_tensor.device}，强制转移")
                    filter_tensor = filter_tensor.to(device=device, non_blocking=False)
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                # 最终双重检查：确保两个张量都在同一设备上
                if traj.device != filter_tensor.device:
                    # 如果设备不匹配，将 filter_tensor 移到 traj 的设备上
                    logger.warning(f"[WARNING] 设备不匹配，traj在{traj.device}, filter_tensor在{filter_tensor.device}，将filter_tensor移到traj的设备")
                    filter_tensor = filter_tensor.to(traj.device, non_blocking=False)
                    if traj.device.type == "cuda":
                        torch.cuda.synchronize(traj.device)
                # 最后一次验证
                if traj.device != filter_tensor.device:
                    logger.error(f"[ERROR] 设备不匹配: traj在{traj.device}, filter_tensor在{filter_tensor.device}, 期望在{device}")
                    return None
                traj = traj.index_select(1, filter_tensor)
            else:
                return None
        elif N_traj != N_ref:
            return None

        # 质量也放在GPU
        from src.data.basic.data_pipeline import Featurizer
        ref_element = Featurizer.elem_onehot_encoded(atom_array.element).long().to(device, non_blocking=True)
        ref_masses = Featurizer.masses_from_ref_element(ref_element, unknown_mass=12.011).to(device, non_blocking=True)

        from src.data.fourier.utils import parse_time_str_to_seconds
        dt = parse_time_str_to_seconds(dt_str)

        metadata = {
            "traj_name": traj_name,
            "bioassembly_dict": bioassembly_dict,
            "masses": ref_masses,
            "dt": dt,
            "dt_str": dt_str,
        }
        return traj_name, traj, metadata
    except Exception as e:
        import traceback
        logger.error(f"[ERROR] GPU预处理失败 {trajectory_file}: {traceback.print_exc()}")
        return None


def preprocess_wrapper(traj_file: Path, bioassembly_dict: dict, dt_str: str) -> Optional[Tuple[str, np.ndarray, Dict[str, Any]]]:
    """
    CPU预处理包装函数，用于多进程处理。
    接受三个独立参数而不是元组，以便 ProcessPoolExecutor 可以正确 pickle。
    """
    return cpu_preprocess_trajectory(traj_file, bioassembly_dict, dt_str)


def prepare_batch_for_gpu(batch_data: List[Tuple[str, np.ndarray, Dict[str, Any]]], target_gpu_id: Optional[int], num_gpus: int) -> List[Tuple[str, torch.Tensor, Dict[str, Any]]]:
    """
    将batch数据（numpy数组）转换为GPU tensor。
    
    Args:
        batch_data: List of (traj_name, traj_np, metadata) tuples
        target_gpu_id: 目标GPU ID，如果为None则使用CPU
        num_gpus: 可用GPU数量
    
    Returns:
        List of (traj_name, traj_tensor, metadata) tuples with tensors on target device
    """
    prepared_batch = []
    for traj_name, traj_np, metadata in batch_data:
        if target_gpu_id is not None and num_gpus > 0:
            target_device = torch.device(f"cuda:{target_gpu_id}")
            traj_tensor = torch.from_numpy(traj_np).to(target_device, non_blocking=True)
            masses_tensor = torch.as_tensor(metadata["masses"]).to(target_device, non_blocking=True)
        else:
            traj_tensor = torch.from_numpy(traj_np)
            masses_tensor = torch.as_tensor(metadata["masses"])
        
        new_metadata = metadata.copy()
        new_metadata["masses"] = masses_tensor
        prepared_batch.append((traj_name, traj_tensor, new_metadata))
    return prepared_batch


def async_save_worker(save_queue: Queue, pbar: Optional[tqdm] = None, success_set: Optional[set] = None, max_queue_size: int = 50):
    """异步保存工作线程
    
    Args:
        save_queue: 保存任务队列（应该有大小限制）
        pbar: 进度条（可选）
        success_set: 成功保存的轨迹名称集合（用于跟踪）
        max_queue_size: 队列最大大小（用于监控）
    """
    while True:
        try:
            item = save_queue.get(timeout=1.0)
            if item is None:  # 结束信号
                break
            
            traj_name, pack, output_path = item
            try:
                dump_gzip_pickle(pack, output_path)
                
                # 立即释放 pack 的内存
                del pack
                gc.collect()
                
                # 验证文件是否成功保存
                if output_path.exists():
                    try:
                        # 尝试验证文件完整性
                        test_data = load_gzip_pickle(output_path)
                        if test_data is not None:
                            if success_set is not None:
                                success_set.add(traj_name)
                            if pbar:
                                pbar.update(1)
                            logger.debug(f"[DEBUG] 成功保存并验证: {traj_name}")
                            del test_data  # 释放验证数据的内存
                        else:
                            logger.error(f"[ERROR] 保存的文件 {traj_name} 内容为空")
                            if output_path.exists():
                                output_path.unlink()
                    except Exception as verify_error:
                        logger.error(f"[ERROR] 保存的文件 {traj_name} 损坏或无法读取: {verify_error}")
                        if output_path.exists():
                            try:
                                output_path.unlink()
                                logger.info(f"[INFO] 已删除损坏的文件: {output_path}")
                            except Exception:
                                pass
                else:
                    logger.error(f"[ERROR] 保存失败 {traj_name}: 文件不存在")
                    
            except Exception as e:
                logger.error(f"[ERROR] 保存失败 {traj_name}: {e}")
                logger.error(f"[ERROR] 异常详情: {traceback.format_exc()}")
                # 如果文件存在但可能损坏，尝试删除
                if output_path.exists():
                    try:
                        output_path.unlink()
                        logger.info(f"[INFO] 已删除可能损坏的文件: {output_path}")
                    except Exception:
                        pass
                # 确保释放内存
                del pack
                gc.collect()
            
            save_queue.task_done()
            
            # 监控队列大小
            queue_size = save_queue.qsize()
            if queue_size > max_queue_size * 0.8:
                logger.warning(f"[WARNING] 保存队列较大: {queue_size}/{max_queue_size}，可能影响内存使用")
        except Empty:
            continue
        except Exception as e:
            logger.error(f"[ERROR] 异步保存线程错误: {e}")
            logger.error(f"[ERROR] 异常详情: {traceback.format_exc()}")


def gen_data_from_pdbs(
    pdb_list: list[Path],
    trajectory_list: list[Path],
    output_pdb_indices_csv: Path,
    output_traj_indices_csv: Path,
    bioassembly_output_dir: Path,
    fourier_output_dir: Path,
    seq_index_dir: Path,
    cluster_file: Optional[Path],
    dt_str: str = "10ps",
    num_workers: int = 1,
    use_optimized: bool = False,
    num_cpu_workers: Optional[int] = None,
    batch_size: int = 4,
    fourier_save_minimal: bool = False,
    prefer_gpu_preprocess: bool = True,
):
    """
    Generates training data from a list of input structure files and saves the results to CSV files.

    Args:
        pdb_list (list[Path]): List of paths to structure files (.pdb/.cif).
        trajectory_list (list[Path]): List of paths to trajectory files.
        output_pdb_indices_csv (Path): Path to the output CSV file for structure indices.
        output_traj_indices_csv (Path): Path to the output CSV file for trajectory names.
        bioassembly_output_dir (Path): Directory where the bioassembly output will be stored.
        fourier_output_dir (Path): Directory where the trajectory Fourier data will be stored.
        cluster_file (Optional[Path]): Path to the cluster file. If None, clustering is not performed.
        dt_str (str, optional): Timestep string for trajectory FFT. Defaults to "10ps".
        num_workers (int, optional): Number of parallel workers to use. Defaults to 1.
        prefer_gpu_preprocess (bool): 尽量在GPU完成预处理，减少CPU/GPU数据搬运。
    """
    logger.info(f"[INFO] 开始处理 {len(pdb_list)} 个 PDB 文件...")
    sys.stdout.flush()

    from joblib import parallel
    import traceback

    def safe_run_func_with_exception(*args, func=None, **kwargs):
        try:
            assert func is not None, "safe_run_func_with_exception requires a 'func' keyword argument"
            return func(*args, **kwargs)
        except SystemExit:
            raise
        except KeyboardInterrupt:
            raise
        except Exception as e:
            error_type = type(e).__name__
            error_msg = str(e)
            logger.error(f"[ERROR] Exception in {func.__name__}: {error_type}: {error_msg}")
            if "Segmentation fault" in error_msg or "segmentation" in error_msg.lower():
                logger.error(f"[ERROR] 检测到段错误，这可能是由于文件损坏或库的 bug")
                logger.error(f"[ERROR] 跳过此文件，继续处理其他文件")
                traceback.print_exc()
                sys.stdout.flush()
                return None
            else:
                traceback.print_exc()
                sys.stdout.flush()
                raise

    results = list(
        tqdm(
            Parallel(n_jobs=num_workers, return_as="generator")(
                delayed(safe_run_func_with_exception)(
                    func=gen_a_bioassembly_data,
                    pdb=pdb,
                    bioassembly_output_dir=bioassembly_output_dir,
                    cluster_file=cluster_file,
                )
                for pdb in pdb_list
            ),
            total=len(pdb_list),
        )
    )

    bioassembly_dict_DICT = {}
    all_sample_indices_lists = []
    info_dfs = []
    for res in results:
        if res is not None:
            sample_indices_list, bioassembly_dict, info_df = res
            if sample_indices_list is not None and len(sample_indices_list) > 0:
                all_sample_indices_lists.append(sample_indices_list)
            if bioassembly_dict is not None:
                pdb_id = bioassembly_dict.get("pdb_id", None)
                if pdb_id is not None:
                    bioassembly_dict_DICT[pdb_id] = bioassembly_dict
            if (info_df is not None) and (not info_df.empty):
                info_dfs.append(info_df)
    logger.info(f"[DEBUG] bioassembly_dict_DICT keys: {list(bioassembly_dict_DICT.keys()) if bioassembly_dict_DICT else 'None'}")
    sys.stdout.flush()

    if info_dfs:
        logger.info(f"[INFO] 开始生成序列索引文件...")
        sys.stdout.flush()
        get_pdb_seqs(info_dfs=info_dfs, out_dir=seq_index_dir)
        logger.info(f"[INFO] 序列索引文件生成完成")
        sys.stdout.flush()
    
    if all_sample_indices_lists:
        merged_results = []
        for sample_indices_list in all_sample_indices_lists: 
            if sample_indices_list:
                merged_results += sample_indices_list
        df = pd.DataFrame(merged_results)
        df.to_csv(output_pdb_indices_csv, index=False, quoting=csv.QUOTE_NONNUMERIC)

    processor_config = {
        "store_dtype": torch.float32,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "remove_com_for_alignment": True,
        "align_rotation_for_alignment": False,
        "window_size": 200,
        "use_sliding_window": True,
        "fpath": fourier_output_dir / "processor.pth",
    }
    
    logger.info(f"[INFO] 开始处理轨迹数据，共 {len(trajectory_list)} 个轨迹文件...")
    logger.info(f"[INFO] 使用优化策略: {use_optimized}")
    sys.stdout.flush()
    
    # 找不到 bioassembly_dict（skipped）和配好（有效）的轨迹
    skipped_traj_names = []
    valid_trajectories = []
    for traj_file in trajectory_list:
        traj_name = str(traj_file.name)[:9].lower()
        bioassembly_dict = find_bioassembly_dict(str(traj_file.name), bioassembly_dict_DICT)
        if not bioassembly_dict or len(bioassembly_dict) == 0:
            skipped_traj_names.append(traj_name)
        else:
            valid_trajectories.append((traj_file, bioassembly_dict))

    # 打印找不到 bioassembly_dict 的轨迹列表
    if skipped_traj_names:
        logger.warning("=" * 80)
        logger.warning(f"[WARNING] 共找到 {len(skipped_traj_names)} 个轨迹无法匹配到 bioassembly_dict，将跳过处理：")
        logger.warning("=" * 80)
        for traj_name in skipped_traj_names[:10]:  # 只显示前10个
            logger.warning(f"  - {traj_name}")
        if len(skipped_traj_names) > 10:
            logger.warning(f"  ... 还有 {len(skipped_traj_names) - 10} 个")
        logger.warning("=" * 80)
        sys.stdout.flush()
    else:
        logger.info(f"[INFO] 所有 {len(trajectory_list)} 个轨迹都成功匹配到 bioassembly_dict")
        sys.stdout.flush()
    
    # 正式的成功轨迹名字列表
    traj_name_list = []
    actually_processed_traj_names = set()  # 记录实际FFT真正处理过的轨迹名，仅用于统计/监控

    if fourier_save_minimal:
        save_minimal = True
        save_full_spectrum = False
        save_windowed = False
        save_psd = False
    else:
        save_minimal = False
        save_full_spectrum = True
        save_windowed = True
        save_psd = True

    if use_optimized:
        if num_cpu_workers is None:
            num_cpu_workers = min(max(1, mp.cpu_count() // 4), 16)
        
        logger.info(f"[INFO] 使用优化处理策略: CPU工作进程={num_cpu_workers}, GPU批大小={batch_size}")
        sys.stdout.flush()
        
        use_gpu_prefetch = prefer_gpu_preprocess and torch.cuda.is_available()
        if use_gpu_prefetch:
            logger.info(f"[INFO] 启用GPU预处理，尽量减少CPU/GPU数据传输")
            sys.stdout.flush()
        
        trajectories_to_process = []
        skipped_existing = []
        
        for traj_file, bioassembly_dict in valid_trajectories:
            traj_name = str(traj_file.name)[:9].lower()
            output_path = fourier_output_dir / f"{traj_name}.pkl.gz"
            if output_path.exists():
                skipped_existing.append(traj_name)
                traj_name_list.append(traj_name)
            else:
                trajectories_to_process.append((traj_file, bioassembly_dict))
        
        if skipped_existing:
            logger.info(f"[INFO] 发现 {len(skipped_existing)} 个轨迹的输出文件已存在，将跳过处理")
            sys.stdout.flush()
        
        total_to_process = len(trajectories_to_process)
        if total_to_process == 0:
            logger.info(f"[INFO] 所有轨迹都已处理完成，无需重新处理")
            sys.stdout.flush()
        else:
            logger.info(f"[INFO] 需要处理 {total_to_process} 个轨迹")
            sys.stdout.flush()

            max_save_queue_size = 50
            save_queue = Queue(maxsize=max_save_queue_size)
            save_success_set = set()
            save_thread = threading.Thread(
                target=async_save_worker,
                args=(save_queue, None, save_success_set, max_save_queue_size),
                daemon=True
            )
            save_thread.start()

            # 检测可用GPU数量并设置多GPU并行
            num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
            if num_gpus > 1:
                logger.info(f"[INFO] 检测到 {num_gpus} 个GPU，启用多GPU并行处理")
                sys.stdout.flush()
                gpu_ids = list(range(num_gpus))
            else:
                logger.info(f"[INFO] 使用单GPU或CPU处理 (GPU数量: {num_gpus})")
                sys.stdout.flush()
                gpu_ids = [0] if num_gpus > 0 else []
            
            preprocessed_batch = []
            processed_count = 0
            last_heartbeat_time = time.time()
            heartbeat_interval = 60
            
            # 用于线程安全的锁
            batch_lock = threading.Lock()
            # GPU批处理并发数（根据GPU内存调整，通常2-4个）
            # 如果有多个GPU，每个GPU可以处理更多并发
            gpu_batch_concurrency = min(4, max(2, batch_size // 2)) if batch_size >= 4 else 1
            if num_gpus > 1:
                # 多GPU时，总并发数 = 每个GPU的并发数 * GPU数量
                total_concurrency = gpu_batch_concurrency * num_gpus
            else:
                total_concurrency = gpu_batch_concurrency

            def process_gpu_batch(batch, batch_id, gpu_id):
                """处理单个batch，每个batch使用独立的processor实例和指定的GPU设备"""
                if not batch:
                    return
                # 为指定的GPU创建processor实例
                gpu_processor_config = processor_config.copy()
                if gpu_id is not None and num_gpus > 0:
                    gpu_processor_config["device"] = f"cuda:{gpu_id}"
                else:
                    gpu_processor_config["device"] = processor_config.get("device", "cpu")
                
                thread_processor = DataPipeline.init_trajectory_fft_processor(**gpu_processor_config)
                gpu_device_str = f"GPU{gpu_id}" if gpu_id is not None and num_gpus > 0 else "CPU"
                logger.info(f"[INFO] 开始GPU FFT处理批次#{batch_id} (设备: {gpu_device_str})，包含 {len(batch)} 个轨迹")
                sys.stdout.flush()
                batch_processed = []
                for batch_idx, (batch_traj_name, batch_traj_tensor, batch_metadata) in enumerate(batch):
                    try:
                        batch_output_path = fourier_output_dir / f"{batch_traj_name}.pkl.gz"
                        if batch_output_path.exists():
                            logger.debug(f"[DEBUG] 轨迹 {batch_traj_name} 的输出文件已存在，跳过处理")
                            with batch_lock:
                                if batch_traj_name not in traj_name_list:
                                    traj_name_list.append(batch_traj_name)
                            continue
                        with batch_lock:
                            actually_processed_traj_names.add(batch_traj_name)
                        logger.info(f"[INFO] GPU FFT处理中(批次#{batch_id}, {gpu_device_str}): {batch_traj_name} ({batch_idx+1}/{len(batch)})")
                        sys.stdout.flush()
                        
                        # 确保张量在正确的GPU设备上
                        if gpu_id is not None and num_gpus > 0:
                            target_device = torch.device(f"cuda:{gpu_id}")
                            if batch_traj_tensor.device != target_device:
                                batch_traj_tensor = batch_traj_tensor.to(target_device, non_blocking=True)
                            if batch_metadata["masses"].device != target_device:
                                batch_metadata["masses"] = batch_metadata["masses"].to(target_device, non_blocking=True)
                        
                        traj = batch_traj_tensor.to(thread_processor.config.store_dtype, non_blocking=True)
                        fft_start_time = time.time()
                        pack = thread_processor.process_and_package(
                            positions=traj,
                            dt=batch_metadata["dt"],
                            masses=batch_metadata["masses"],
                            box_lengths=None,
                            missing_mask=None,
                            layout_hint="TN3",
                            remove_com=True,
                            align_rotation=True,
                            save_minimal=save_minimal,
                            save_psd=save_psd,
                        )
                        fft_elapsed = time.time() - fft_start_time
                        logger.info(f"[INFO] GPU FFT处理完成(批次#{batch_id}, {gpu_device_str}): {batch_traj_name}, 耗时: {fft_elapsed:.2f}秒")
                        sys.stdout.flush()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        cpu_pack = move_to_cpu(pack)
                        save_queue.put((batch_traj_name, cpu_pack, batch_output_path), block=True, timeout=300)
                        with batch_lock:
                            traj_name_list.append(batch_traj_name)
                        batch_processed.append(batch_traj_name)
                        # 释放显存
                        del batch_traj_tensor, traj, pack
                    except Exception as e:
                        logger.error(f"[ERROR] GPU FFT处理失败(批次#{batch_id}, {gpu_device_str}) {batch_traj_name}: {e}")
                        sys.stdout.flush()
                logger.info(f"[INFO] 批次#{batch_id} ({gpu_device_str}) GPU FFT处理完成，已处理 {len(batch_processed)} 个轨迹")
                sys.stdout.flush()
                batch.clear()
                del thread_processor
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            if use_gpu_prefetch:
                # 为每个GPU创建独立的线程池（如果有多GPU）
                if num_gpus > 1:
                    # 多GPU模式：为每个GPU创建线程池
                    gpu_executors = {}
                    for gpu_id in gpu_ids:
                        gpu_executors[gpu_id] = ThreadPoolExecutor(max_workers=gpu_batch_concurrency)
                    logger.info(f"[INFO] 为 {num_gpus} 个GPU各创建了 {gpu_batch_concurrency} 个工作线程")
                else:
                    # 单GPU或CPU模式：只创建一个线程池
                    gpu_executors = {0: ThreadPoolExecutor(max_workers=gpu_batch_concurrency)}
                
                batch_id_counter = 0
                pending_batch_futures = []
                gpu_round_robin_counter = 0  # 用于轮询分配GPU
                
                with tqdm(total=total_to_process, desc="处理轨迹(GPU预处理)") as pbar:
                    for traj_file, bioassembly_dict in trajectories_to_process:
                        try:
                            # 轮询选择GPU进行预处理
                            if num_gpus > 1:
                                current_gpu_id = gpu_ids[gpu_round_robin_counter % num_gpus]
                                gpu_round_robin_counter += 1
                                device = torch.device(f"cuda:{current_gpu_id}")
                            else:
                                device = torch.device(f"cuda:{gpu_ids[0]}" if gpu_ids else "cpu")
                            
                            result = gpu_preprocess_trajectory(traj_file, bioassembly_dict, dt_str, device=device)
                        except Exception as e:
                            logger.error(f"[ERROR] GPU预处理阶段异常: {e}")
                            traceback.print_exc()
                            pbar.update(1)
                            processed_count += 1
                            continue

                        if result is not None:
                            preprocessed_batch.append(result)
                            if len(preprocessed_batch) >= batch_size:
                                # 轮询选择GPU处理这个batch
                                if num_gpus > 1:
                                    assigned_gpu_id = gpu_ids[batch_id_counter % num_gpus]
                                else:
                                    assigned_gpu_id = gpu_ids[0] if gpu_ids else None
                                
                                # 提交batch到对应GPU的线程池
                                batch_id = batch_id_counter
                                batch_id_counter += 1
                                batch_to_process = preprocessed_batch.copy()
                                preprocessed_batch = []  # 清空，准备下一个batch
                                
                                executor = gpu_executors[assigned_gpu_id]
                                future = executor.submit(process_gpu_batch, batch_to_process, batch_id, assigned_gpu_id)
                                pending_batch_futures.append(future)
                                
                                # 限制并发数：等待一些batch完成
                                if len(pending_batch_futures) >= total_concurrency:
                                    # 等待至少一个batch完成
                                    done, not_done = wait(pending_batch_futures, return_when='FIRST_COMPLETED')
                                    pending_batch_futures = list(not_done)  # 更新为未完成的futures
                                    for f in done:
                                        try:
                                            f.result()  # 检查是否有异常
                                        except Exception as e:
                                            logger.error(f"[ERROR] 批次处理异常: {e}")
                                            traceback.print_exc()
                        pbar.update(1)
                        processed_count += 1

                        current_time = time.time()
                        if processed_count % 10 == 0 or (current_time - last_heartbeat_time) >= heartbeat_interval:
                            logger.info(f"[HEARTBEAT] GPU预处理进度: {processed_count}/{total_to_process} | FFT已处理(真实): {len(actually_processed_traj_names)} | 累计计入: {len(traj_name_list)} | 并发批次: {len(pending_batch_futures)}")
                            sys.stdout.flush()
                            last_heartbeat_time = current_time

                    # 处理剩余的batch
                    if preprocessed_batch:
                        if num_gpus > 1:
                            assigned_gpu_id = gpu_ids[batch_id_counter % num_gpus]
                        else:
                            assigned_gpu_id = gpu_ids[0] if gpu_ids else None
                        batch_id = batch_id_counter
                        batch_id_counter += 1
                        executor = gpu_executors[assigned_gpu_id]
                        future = executor.submit(process_gpu_batch, preprocessed_batch, batch_id, assigned_gpu_id)
                        pending_batch_futures.append(future)
                    
                    # 等待所有batch完成
                    logger.info(f"[INFO] 等待所有批次处理完成，共 {len(pending_batch_futures)} 个批次...")
                    sys.stdout.flush()
                    for future in as_completed(pending_batch_futures):
                        try:
                            future.result()
                        except Exception as e:
                            logger.error(f"[ERROR] 批次处理异常: {e}")
                            traceback.print_exc()
                
                # 关闭所有线程池
                for executor in gpu_executors.values():
                    executor.shutdown(wait=True)
            else:
                # CPU预处理模式：为每个GPU创建独立的线程池（如果有多GPU）
                if num_gpus > 1:
                    gpu_executors = {}
                    for gpu_id in gpu_ids:
                        gpu_executors[gpu_id] = ThreadPoolExecutor(max_workers=gpu_batch_concurrency)
                    logger.info(f"[INFO] CPU预处理模式：为 {num_gpus} 个GPU各创建了 {gpu_batch_concurrency} 个工作线程")
                else:
                    gpu_executors = {0: ThreadPoolExecutor(max_workers=gpu_batch_concurrency)} if gpu_ids else {}
                
                batch_id_counter = 0
                gpu_fft_futures = []
                
                with ProcessPoolExecutor(max_workers=num_cpu_workers) as cpu_executor:
                    max_pending_tasks = num_cpu_workers * 2
                    pending_futures = {}
                    task_iterator = iter(trajectories_to_process)
                    
                    with tqdm(total=total_to_process, desc="处理轨迹") as pbar:
                        for _ in range(min(max_pending_tasks, total_to_process)):
                            try:
                                traj_file, bioassembly_dict = next(task_iterator)
                                future = cpu_executor.submit(preprocess_wrapper, traj_file, bioassembly_dict, dt_str)
                                pending_futures[future] = traj_file
                            except StopIteration:
                                break

                        while pending_futures:
                            done, _ = wait(list(pending_futures.keys()), return_when='FIRST_COMPLETED')
                            for future in done:
                                try:
                                    result = future.result()
                                except Exception as e:
                                    logger.error(f"[ERROR] CPU预处理阶段异常: {e}")
                                    traceback.print_exc()
                                    del pending_futures[future]
                                    pbar.update(1)
                                    processed_count += 1
                                    continue
                                del pending_futures[future]

                                try:
                                    traj_file, bioassembly_dict = next(task_iterator)
                                    new_future = cpu_executor.submit(preprocess_wrapper, traj_file, bioassembly_dict, dt_str)
                                    pending_futures[new_future] = traj_file
                                except StopIteration:
                                    pass

                                if result is not None:
                                    traj_name, traj_np, metadata = result
                                    output_path = fourier_output_dir / f"{traj_name}.pkl.gz"
                                    if output_path.exists():
                                        logger.debug(f"[DEBUG] 轨迹 {traj_name} 的输出文件已存在，跳过处理")
                                        if traj_name not in traj_name_list:
                                            traj_name_list.append(traj_name)
                                        pbar.update(1)
                                        processed_count += 1
                                        del result, traj_np, metadata
                                        gc.collect()
                                        continue
                                    preprocessed_batch.append(result)
                                    if len(preprocessed_batch) >= batch_size:
                                        # 轮询选择GPU处理这个batch
                                        if num_gpus > 1:
                                            assigned_gpu_id = gpu_ids[batch_id_counter % num_gpus]
                                        else:
                                            assigned_gpu_id = gpu_ids[0] if gpu_ids else None
                                        
                                        batch_id = batch_id_counter
                                        batch_id_counter += 1
                                        batch_to_process = preprocessed_batch.copy()
                                        preprocessed_batch = []  # 清空，准备下一个batch
                                        
                                        # 将numpy数组转换为tensor并移到对应GPU
                                        prepared_batch = prepare_batch_for_gpu(batch_to_process, assigned_gpu_id, num_gpus)
                                        
                                        if assigned_gpu_id is not None and assigned_gpu_id in gpu_executors:
                                            executor = gpu_executors[assigned_gpu_id]
                                            future = executor.submit(process_gpu_batch, prepared_batch, batch_id, assigned_gpu_id)
                                            gpu_fft_futures.append(future)
                                            
                                            # 限制并发数
                                            if len(gpu_fft_futures) >= total_concurrency:
                                                done, not_done = wait(gpu_fft_futures, return_when='FIRST_COMPLETED')
                                                gpu_fft_futures = list(not_done)
                                                for f in done:
                                                    try:
                                                        f.result()
                                                    except Exception as e:
                                                        logger.error(f"[ERROR] 批次处理异常: {e}")
                                                        traceback.print_exc()
                                pbar.update(1)
                                processed_count += 1

                                current_time = time.time()
                                if processed_count % 10 == 0 or (current_time - last_heartbeat_time) >= heartbeat_interval:
                                    logger.info(f"[HEARTBEAT] 预处理进度: {processed_count}/{total_to_process} | FFT已处理(真实): {len(actually_processed_traj_names)} | 替输出/成功计入: {len(traj_name_list)} | GPU FFT并发: {len(gpu_fft_futures)}")
                                    sys.stdout.flush()
                                    last_heartbeat_time = current_time

                # 处理剩余的batch
                if preprocessed_batch:
                    if num_gpus > 1:
                        assigned_gpu_id = gpu_ids[batch_id_counter % num_gpus]
                    else:
                        assigned_gpu_id = gpu_ids[0] if gpu_ids else None
                    
                    prepared_batch = prepare_batch_for_gpu(preprocessed_batch, assigned_gpu_id, num_gpus)
                    batch_id = batch_id_counter
                    if assigned_gpu_id is not None and assigned_gpu_id in gpu_executors:
                        executor = gpu_executors[assigned_gpu_id]
                        future = executor.submit(process_gpu_batch, prepared_batch, batch_id, assigned_gpu_id)
                        gpu_fft_futures.append(future)
                
                # 等待所有GPU FFT任务完成
                if gpu_fft_futures:
                    logger.info(f"[INFO] 等待所有GPU FFT批次处理完成，共 {len(gpu_fft_futures)} 个批次...")
                    sys.stdout.flush()
                    for future in as_completed(gpu_fft_futures):
                        try:
                            future.result()
                        except Exception as e:
                            logger.error(f"[ERROR] 批次处理异常: {e}")
                            traceback.print_exc()
                
                # 关闭所有GPU线程池
                for executor in gpu_executors.values():
                    executor.shutdown(wait=True)

            save_queue.put(None)
            save_thread.join(timeout=300)

            if 'save_success_set' in locals():
                original_count = len(traj_name_list)
                traj_name_list = [name for name in traj_name_list if name in save_success_set or (fourier_output_dir / f"{name}.pkl.gz").exists()]
                removed_count = original_count - len(traj_name_list)
                if removed_count > 0:
                    logger.warning(f"[WARNING] 优化模式下发现 {removed_count} 个轨迹未成功保存，已从列表中移除")
                    logger.info(f"[INFO] 更新后成功数量: {len(traj_name_list)}")

            logger.info(f"[INFO] 优化模式下，GPU FFT真实处理过的轨迹数（不含已存在/跳过）：{len(actually_processed_traj_names)}")
            logger.info(f"[INFO] 优化模式下，最终成功轨迹(含已存在/成功写入总数)：{len(traj_name_list)}")
        
    else:
        processor = DataPipeline.init_trajectory_fft_processor(**processor_config)
        logger.info(f"[INFO] 使用原始处理策略: {num_workers} 个进程")
        sys.stdout.flush()
        
        trajectories_to_process = []
        skipped_existing = []
        
        for traj_file, bioassembly_dict in valid_trajectories:
            traj_name = str(traj_file.name)[:9].lower()
            output_path = fourier_output_dir / f"{traj_name}.pkl.gz"
            if output_path.exists():
                skipped_existing.append(traj_name)
                traj_name_list.append(traj_name)
            else:
                trajectories_to_process.append((traj_file, bioassembly_dict))
        
        if skipped_existing:
            logger.info(f"[INFO] 发现 {len(skipped_existing)} 个轨迹的输出文件已存在，将跳过处理")
            sys.stdout.flush()
        
        total_to_process = len(trajectories_to_process)
        if total_to_process:
            logger.info(f"[INFO] 需要处理 {total_to_process} 个轨迹")
            sys.stdout.flush()
            processed_count = 0
            failed_count = 0
            last_heartbeat_time = time.time()
            heartbeat_interval = 300

            actually_processed_traj_names = set()

            for traj_file, bioassembly_dict in tqdm(trajectories_to_process, total=total_to_process):
                traj_name = str(traj_file.name)[:9].lower()
                logger.info(f"[INFO] 开始处理轨迹: {traj_name} ({processed_count+1}/{total_to_process})")
                sys.stdout.flush()
                try:
                    r = safe_run_func_with_exception(
                        func=gen_a_trajectory_data,
                        trajectory_file=traj_file,
                        traj_name=traj_name,
                        fourier_output_dir=fourier_output_dir,
                        bioassembly_dict=bioassembly_dict,
                        processor=processor,
                        dt_str=dt_str,
                    )
                    output_path = fourier_output_dir / f"{traj_name}.pkl.gz"
                    if r is not None:
                        if output_path.exists():
                            try:
                                test_data = load_gzip_pickle(output_path)
                                if test_data is not None:
                                    traj_name_list.append(r)
                                    actually_processed_traj_names.add(traj_name)
                                    logger.info(f"[INFO] 轨迹处理完成: {traj_name}")
                                else:
                                    logger.error(f"[ERROR] 轨迹 {traj_name} 文件存在但内容为空，跳过")
                                    failed_count += 1
                            except Exception as verify_error:
                                logger.error(f"[ERROR] 轨迹 {traj_name} 文件损坏或无法读取: {verify_error}")
                                logger.error(f"[ERROR] 文件路径: {output_path}")
                                try:
                                    output_path.unlink()
                                    logger.info(f"[INFO] 已删除损坏的文件: {output_path}")
                                except Exception as del_error:
                                    logger.warning(f"[WARNING] 无法删除损坏的文件: {del_error}")
                                failed_count += 1
                        else:
                            logger.error(f"[ERROR] 轨迹 {traj_name} 处理返回成功但文件不存在: {output_path}")
                            failed_count += 1
                    else:
                        logger.warning(f"[WARNING] 轨迹处理返回None: {traj_name}")
                        failed_count += 1
                except Exception as e:
                    failed_count += 1
                    error_msg = f"{type(e).__name__}: {str(e)}"
                    logger.error(f"[ERROR] 处理轨迹 {traj_name} 时发生异常: {error_msg}")
                    logger.error(f"[ERROR] 异常详情: {traceback.format_exc()}")
                    output_path = fourier_output_dir / f"{traj_name}.pkl.gz"
                    if output_path.exists():
                        try:
                            test_data = load_gzip_pickle(output_path)
                            if test_data is None:
                                output_path.unlink()
                                logger.info(f"[INFO] 已删除不完整的文件: {output_path}")
                        except Exception:
                            try:
                                output_path.unlink()
                                logger.info(f"[INFO] 已删除损坏的文件: {output_path}")
                            except Exception as del_error:
                                logger.warning(f"[WARNING] 无法删除损坏的文件: {del_error}")
                    sys.stdout.flush()
                processed_count += 1
                current_time = time.time()
                if processed_count % 10 == 0 or (current_time - last_heartbeat_time) >= heartbeat_interval:
                    logger.info(f"[HEARTBEAT] 处理进度: {processed_count}/{total_to_process} 个轨迹已处理, "
                              f"成功FFT(真实): {len(actually_processed_traj_names)}, 累计计入: {len(traj_name_list)}, 失败: {failed_count}")
                    sys.stdout.flush()
                    last_heartbeat_time = current_time
        else:
            logger.info(f"[INFO] 所有轨迹都已处理完成，无需重新处理")
            sys.stdout.flush()
    
    # 最终输出轨迹列表写到csv，并再次确保列表里名字的文件真实存在
    missing_files = []
    filtered_traj_name_list = []
    for traj_name in traj_name_list:
        output_path = fourier_output_dir / f"{traj_name}.pkl.gz"
        if not output_path.exists():
            missing_files.append(traj_name)
        else:
            filtered_traj_name_list.append(traj_name)
    if missing_files:
        logger.warning(f"[WARNING] 发现 {len(missing_files)} 个轨迹在列表中但文件不存在（前10个）: {missing_files[:10]}")
        logger.warning(f"[WARNING] 已从列表中移除缺失的文件，更新后成功数量: {len(filtered_traj_name_list)}")
    traj_name_list = filtered_traj_name_list

    # 只输出最终成功轨迹列表到index csv
    df = pd.DataFrame({
        "traj_name": traj_name_list,
        "pdb_name": [str(name)[:6] for name in traj_name_list],
    })
    df.to_csv(output_traj_indices_csv, index=False, quoting=csv.QUOTE_NONNUMERIC)
    logger.info(f"[INFO] 轨迹索引已保存到: {output_traj_indices_csv}")
    
    # 结尾统计
    logger.info("=" * 80)
    logger.info("数据准备流程完成！")
    logger.info("=" * 80)
    logger.info(f"[INFO] 统计信息：")
    logger.info(f"  - 总轨迹数: {len(trajectory_list)}")
    logger.info(f"  - 成功(含已存在和处理成功): {len(traj_name_list)}")
    if skipped_traj_names:
        logger.info(f"  - 跳过（找不到 bioassembly_dict）: {len(skipped_traj_names)}")
    # 优化/原始，都统计实际FFT处理数
    if 'actually_processed_traj_names' in locals():
        logger.info(f"  - 真实被FFT处理过轨迹数: {len(actually_processed_traj_names)}")
    if 'failed_count' in locals():
        logger.info(f"  - 处理失败: {failed_count}")
    logger.info("=" * 80)
    sys.stdout.flush()

def run_gen_data(
    input_path: Path,
    output_pdb_indices_csv: Path,
    output_traj_indices_csv: Path,
    bioassembly_output_dir: Path,
    fourier_output_dir: Path,
    cluster_file: Optional[Path],
    seq_index_dir: Path,
    dt_str: str = "10ps",
    num_workers: int = 1,
    use_optimized: bool = False,
    num_cpu_workers: Optional[int] = None,
    batch_size: int = 4,
    fourier_save_minimal: bool = False,
    prefer_gpu_preprocess: bool = True,
):
    logger.info("=" * 80)
    logger.info("开始数据准备流程")
    logger.info("=" * 80)
    logger.info(f"输入路径: {input_path}")
    logger.info(f"输出目录: {bioassembly_output_dir}, {fourier_output_dir}")
    logger.info(f"工作进程数: {num_workers}")
    sys.stdout.flush()
    
    input_path = Path(input_path)
    bioassembly_output_dir = Path(bioassembly_output_dir)
    output_pdb_indices_csv = Path(output_pdb_indices_csv)
    output_traj_indices_csv = Path(output_traj_indices_csv)

    # create directory for output
    output_pdb_indices_csv.parent.mkdir(parents=True, exist_ok=True)
    output_traj_indices_csv.parent.mkdir(parents=True, exist_ok=True)
    bioassembly_output_dir.mkdir(parents=True, exist_ok=True)
    fourier_output_dir.mkdir(parents=True, exist_ok=True)
    seq_index_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"[INFO] 输出目录已创建")
    sys.stdout.flush()

    if input_path.is_dir():
        logger.info(f"[INFO] 扫描目录: {input_path}")
        sys.stdout.flush()
        # # 这样写只是获取input_path目录下的所有.cif和.cif.gz文件, 不会递归子目录
        # mmcif_list = list(input_path.glob("*.cif")) + list(input_path.glob("*.cif.gz"))
        # 如果要递归所有子目录下的.cif和.cif.gz文件，可以使用rglob:
        # Collect all direct subdirectories of input_path
        # pdb_list = list(input_path.rglob("*.pdb")) + list(input_path.rglob("*.pdb.gz"))
        # trajectory_list = list(input_path.rglob("*.npy"))

        # 忽略不在一级或者多级子目录下的文件（其实那些是完整的，没有做过切分的）
        subdirs = [d for d in input_path.iterdir() if d.is_dir()]
        pdb_list = []
        trajectory_list = []
        for subdir in subdirs:
            pdb_list += list(subdir.rglob("*.pdb"))
            pdb_list += list(subdir.rglob("*.pdb.gz"))
            trajectory_list += list(subdir.rglob("*.npy"))

        logger.info(f"[INFO] 找到 {len(pdb_list)} 个 PDB 文件和 {len(trajectory_list)} 个轨迹文件")
        sys.stdout.flush()
    
    elif input_path.suffix == ".txt":
        logger.info(f"[INFO] 从文件列表读取: {input_path}")
        sys.stdout.flush()
        with open(input_path) as f:
            pdb_list = [Path(i.strip()) for i in f.readlines() if i.strip()]
        trajectory_list = []  # 文本文件模式通常不包含轨迹列表
        logger.info(f"[INFO] 从文件列表读取到 {len(pdb_list)} 个 PDB 文件")
        sys.stdout.flush()
    else:
        raise NotImplementedError(f"Unsupported input path: {input_path}")

    gen_data_from_pdbs(
        pdb_list,
        trajectory_list,
        output_pdb_indices_csv,
        output_traj_indices_csv,
        bioassembly_output_dir,
        fourier_output_dir,
        seq_index_dir,
        cluster_file,
        dt_str,
        num_workers,
        use_optimized=use_optimized,
        num_cpu_workers=num_cpu_workers,
        batch_size=batch_size,
        fourier_save_minimal=fourier_save_minimal,
        prefer_gpu_preprocess=prefer_gpu_preprocess,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input_path",
        type=Path,
        default=None,
        help="Path to the input directory containing MMCIF files or a .txt file listing MMCIF file paths.",
    )
    parser.add_argument(
        "-opdb",
        "--output_pdb_indices_csv",
        type=Path,
        default=None,
        help="Path to the output CSV file where indices will be saved.",
    )
    parser.add_argument(
        "-otraj",
        "--output_traj_indices_csv",
        type=Path,
        default=None,
        help="Path to the output CSV file where trajectory indices will be saved.",
    )
    parser.add_argument(
        "-obio",
        "--bio_output_dir",
        type=Path,
        default=None,
        help="Directory where bioassembly outputs will be saved.",
    )
    parser.add_argument(
        "-ofo",
        "--fourier_output_dir",
        type=Path,
        default=None,
        help="Directory where fourier outputs will be saved.",
    )
    parser.add_argument(
        "-oseq",
        "--seq_index_dir",
        type=Path,
        default=None,
        help="Directory where sequence index files will be saved.",
    )
    parser.add_argument(
        "-c",
        "--cluster_file",
        type=Path,
        default=None,
        help="Path to the cluster txt file, if any",
    )

    parser.add_argument(
        "-dt",
        "--dt_str",
        type=str,
        default="10ps",
        help="Time step string, e.g. '10ps'.",
    )
    parser.add_argument(
        "-n",
        "--n_cpu",
        type=int,
        default=1,
        help="Number of worker processes to use. Defaults to 1.",
    )
    parser.add_argument(
        "--use_optimized",
        action="store_true",
        help="使用优化的混合处理策略（CPU预处理 + GPU FFT + 异步I/O）",
    )
    parser.add_argument(
        "--num_cpu_workers",
        type=int,
        default=None,
        help="CPU预处理工作进程数（优化模式，None则自动设置为CPU核心数的一半）",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="GPU批处理大小（优化模式，默认4）",
    )
    parser.add_argument(
        "--fourier_save_minimal",
        action="store_true",
        help="使用最小的傅里叶变换保存策略（默认保存完整的傅里叶变换）",
    )
    parser.add_argument(
        "--disable_gpu_preprocess",
        action="store_true",
        help="禁用GPU预处理（默认启用以减少CPU/GPU数据传输）",
    )
    args = parser.parse_args()

    run_gen_data(
        input_path=args.input_path,
        output_pdb_indices_csv=args.output_pdb_indices_csv,
        output_traj_indices_csv=args.output_traj_indices_csv,
        bioassembly_output_dir=args.bio_output_dir,
        fourier_output_dir=args.fourier_output_dir,
        cluster_file=args.cluster_file,
        seq_index_dir=args.seq_index_dir,
        dt_str=args.dt_str,
        num_workers=args.n_cpu,
        use_optimized=args.use_optimized,
        num_cpu_workers=args.num_cpu_workers,
        batch_size=args.batch_size,
        fourier_save_minimal=args.fourier_save_minimal,
        prefer_gpu_preprocess=not args.disable_gpu_preprocess,
    )
# TODO:
# 1. 获取每个轨迹的pdb id list, 保存在txt中（保存每个轨迹的pdb id + 链名）
# 2. 根据pdb id list，从PDB数据库中获取静态晶体结构（）
# 3. 如果是单链（例如ATLAS），根据链名从
# 4. 输入两个CIF文件(来自PDB的静态晶体结构&抽帧得到的第一帧结构)，检查这两个文件是否是同一个分子/复合物
