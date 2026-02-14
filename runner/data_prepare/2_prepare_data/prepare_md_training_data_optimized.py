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

"""
优化版本的轨迹数据处理：支持批处理、异步I/O和混合CPU/GPU处理策略
"""

import argparse
import csv
import os
import re
import sys
import logging
import time
import threading
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
from collections import deque
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing as mp

import pandas as pd
from tqdm import tqdm
import torch
import numpy as np

from src.data.basic.data_pipeline import DataPipeline
from src.utils.file_io import dump_gzip_pickle, load_gzip_pickle
from src.data.fourier.utils import TrajectoryFFTProcessor

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def extract_pdb_id_from_filename(filename: str) -> str:
    """从文件名中提取 PDB ID"""
    if '/' in filename or '\\' in filename:
        stem = Path(filename).stem
    else:
        stem = Path(filename).stem
    
    filename_6 = stem[:6]
    match = re.match(r"([0-9a-zA-Z]{4})(?:[_\-].+)?", filename_6)
    if match:
        pdb_id = filename_6.lower()
    else:
        pdb_id = filename_6.lower()[:4]
    
    return pdb_id


def find_bioassembly_dict(traj_filename: str, bioassembly_dict_DICT: dict) -> dict:
    """从轨迹文件名中提取 pdb_id，并在 bioassembly_dict_DICT 中查找对应的字典"""
    pdb_id_6 = extract_pdb_id_from_filename(traj_filename)
    if pdb_id_6 in bioassembly_dict_DICT:
        return bioassembly_dict_DICT[pdb_id_6]
    
    pdb_id_4 = pdb_id_6[:4] if len(pdb_id_6) >= 4 else pdb_id_6
    if pdb_id_4 in bioassembly_dict_DICT:
        return bioassembly_dict_DICT[pdb_id_4]
    
    pdb_id_no_underscore = pdb_id_6.replace('_', '')
    if pdb_id_no_underscore in bioassembly_dict_DICT:
        return bioassembly_dict_DICT[pdb_id_no_underscore]
    
    variants = [
        pdb_id_6, pdb_id_4, pdb_id_no_underscore,
        pdb_id_6.replace('-', ''),
        pdb_id_6.replace('_', '').replace('-', ''),
    ]
    
    for variant in variants:
        if variant in bioassembly_dict_DICT:
            logger.warning(f"[WARNING] 轨迹 {traj_filename} 使用变体 {variant} 找到 bioassembly_dict")
            return bioassembly_dict_DICT[variant]
    
    logger.warning(f"[WARNING] 轨迹 {traj_filename} 无法找到对应的 bioassembly_dict")
    return {}


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
        
        # 检查 bioassembly_dict
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
                logger.error(f"轨迹 {traj_name} 原子数不匹配")
                return None
        elif N_traj != N_ref:
            logger.error(f"轨迹 {traj_name} 原子数不匹配")
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


def gpu_process_fft_batch(
    preprocessed_batch: List[Tuple[str, np.ndarray, Dict[str, Any]]],
    processor_config: dict,
    fourier_output_dir: Path,
) -> List[str]:
    """
    GPU批量处理FFT：对一批预处理好的轨迹进行FFT处理
    返回: 成功处理的轨迹名称列表
    """
    processor = DataPipeline.init_trajectory_fft_processor(**processor_config)
    successful = []
    
    for traj_name, traj_np, metadata in preprocessed_batch:
        try:
            traj = torch.from_numpy(traj_np).to(processor.config.store_dtype)
            
            pack = processor.process_and_package(
                positions=traj,
                dt=metadata["dt"],
                masses=metadata["masses"],
                box_lengths=None,
                missing_mask=None,
                layout_hint="TN3",
                remove_com=True,
                align_rotation=False,
                save_minimal=True,
                save_psd=True,
            )
            
            # 异步保存（在后台线程中）
            output_path = fourier_output_dir / f"{traj_name}.pkl.gz"
            successful.append((traj_name, pack, output_path))
            
        except Exception as e:
            logger.error(f"[ERROR] GPU FFT处理失败 {traj_name}: {e}")
    
    return successful


def async_save_worker(save_queue: Queue, pbar: Optional[tqdm] = None):
    """异步保存工作线程"""
    while True:
        try:
            item = save_queue.get(timeout=1.0)
            if item is None:  # 结束信号
                break
            
            traj_name, pack, output_path = item
            try:
                dump_gzip_pickle(pack, output_path)
                if pbar:
                    pbar.update(1)
            except Exception as e:
                logger.error(f"[ERROR] 保存失败 {traj_name}: {e}")
            
            save_queue.task_done()
        except Empty:
            continue
        except Exception as e:
            logger.error(f"[ERROR] 异步保存线程错误: {e}")


def process_trajectories_optimized(
    trajectory_list: List[Path],
    bioassembly_dict_DICT: dict,
    fourier_output_dir: Path,
    processor_config: dict,
    dt_str: str = "10ps",
    num_cpu_workers: int = 4,
    batch_size: int = 4,
    max_gpu_queue_size: int = 8,
) -> List[str]:
    """
    优化的轨迹处理：混合CPU/GPU处理 + 异步I/O
    
    Args:
        trajectory_list: 轨迹文件列表
        bioassembly_dict_DICT: bioassembly字典
        fourier_output_dir: 输出目录
        processor_config: processor配置
        dt_str: 时间步长字符串
        num_cpu_workers: CPU预处理工作进程数
        batch_size: GPU批处理大小
        max_gpu_queue_size: GPU队列最大大小
    
    Returns:
        成功处理的轨迹名称列表
    """
    logger.info(f"[INFO] 开始优化处理 {len(trajectory_list)} 个轨迹")
    logger.info(f"[INFO] CPU工作进程: {num_cpu_workers}, GPU批大小: {batch_size}")
    
    # 过滤有效轨迹
    valid_trajectories = []
    for traj_file in trajectory_list:
        traj_name = str(traj_file.name)[:9].lower()
        bioassembly_dict = find_bioassembly_dict(str(traj_file.name), bioassembly_dict_DICT)
        if bioassembly_dict and len(bioassembly_dict) > 0:
            valid_trajectories.append((traj_file, bioassembly_dict))
    
    logger.info(f"[INFO] 有效轨迹数: {len(valid_trajectories)}")
    
    # 创建队列
    preprocess_queue = Queue(maxsize=max_gpu_queue_size * 2)  # CPU预处理结果队列
    save_queue = Queue()  # 异步保存队列
    
    # 启动异步保存线程
    save_thread = threading.Thread(
        target=async_save_worker,
        args=(save_queue,),
        daemon=True
    )
    save_thread.start()
    
    # CPU预处理：多进程并行
    def preprocess_wrapper(args):
        traj_file, bioassembly_dict = args
        return cpu_preprocess_trajectory(traj_file, bioassembly_dict, dt_str)
    
    # GPU处理：单进程（避免GPU资源竞争）
    successful_trajs = []
    preprocessed_batch = []
    
    with ProcessPoolExecutor(max_workers=num_cpu_workers) as cpu_executor:
        # 提交所有CPU预处理任务
        future_to_traj = {
            cpu_executor.submit(preprocess_wrapper, (traj_file, bioassembly_dict)): traj_file
            for traj_file, bioassembly_dict in valid_trajectories
        }
        
        # 处理完成的任务
        with tqdm(total=len(valid_trajectories), desc="处理轨迹") as pbar:
            for future in as_completed(future_to_traj):
                result = future.result()
                if result is not None:
                    preprocessed_batch.append(result)
                    
                    # 当批次达到指定大小时，提交GPU处理
                    if len(preprocessed_batch) >= batch_size:
                        gpu_results = gpu_process_fft_batch(
                            preprocessed_batch,
                            processor_config,
                            fourier_output_dir,
                        )
                        
                        # 提交异步保存
                        for traj_name, pack, output_path in gpu_results:
                            save_queue.put((traj_name, pack, output_path))
                            successful_trajs.append(traj_name)
                        
                        preprocessed_batch = []
                        pbar.update(batch_size)
                
                pbar.update(1)
    
    # 处理剩余的批次
    if preprocessed_batch:
        gpu_results = gpu_process_fft_batch(
            preprocessed_batch,
            processor_config,
            fourier_output_dir,
        )
        for traj_name, pack, output_path in gpu_results:
            save_queue.put((traj_name, pack, output_path))
            successful_trajs.append(traj_name)
    
    # 等待所有保存完成
    save_queue.put(None)  # 结束信号
    save_thread.join(timeout=300)  # 最多等待5分钟
    
    logger.info(f"[INFO] 成功处理 {len(successful_trajs)} 个轨迹")
    return successful_trajs


# 为了兼容性，保留原来的函数接口
def gen_a_trajectory_data(
    trajectory_file: Path,
    traj_name: str,
    fourier_output_dir: Path,
    bioassembly_dict: dict,
    processor: Optional[TrajectoryFFTProcessor] = None,
    processor_config: Optional[dict] = None,
    dt_str: str = "10ps",
):
    """原始的单轨迹处理函数（保持兼容性）"""
    import time
    start_time = time.time()
    
    if not bioassembly_dict or len(bioassembly_dict) == 0:
        return None
    
    if processor is None:
        if processor_config is None:
            raise ValueError("Either processor or processor_config must be provided")
        processor = DataPipeline.init_trajectory_fft_processor(**processor_config)
    
    try:
        package = DataPipeline.get_trajectory_fft(
            trajectory_file,
            processor,
            bioassembly_dict,
            remove_com_for_alignment=True,
            align_rotation_for_alignment=False,
            dt_str=dt_str,
        )
        dump_gzip_pickle(package, fourier_output_dir / f"{traj_name}.pkl.gz")
        elapsed = time.time() - start_time
        logger.info(f"[DEBUG] 轨迹处理完成: {traj_name}, 耗时: {elapsed:.2f}秒")
        return traj_name
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"[ERROR] 轨迹处理失败: {traj_name}, 耗时: {elapsed:.2f}秒, 错误: {e}")
        raise


# 导入其他必要的函数
from prepare_md_training_data import (
    gen_a_bioassembly_data,
    get_pdb_seqs,
    run_gen_data as run_gen_data_original,
)


def gen_data_from_pdbs_optimized(
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
    use_optimized: bool = True,
    num_cpu_workers: Optional[int] = None,
    batch_size: int = 4,
):
    """
    优化的数据生成函数
    
    Args:
        use_optimized: 是否使用优化处理策略
        num_cpu_workers: CPU预处理工作进程数（None则自动设置）
        batch_size: GPU批处理大小
    """
    logger.info(f"[INFO] 开始处理 {len(pdb_list)} 个 PDB 文件...")
    
    # 处理PDB文件（与原版相同）
    from joblib import Parallel, delayed
    import traceback
    
    def safe_run_func_with_exception(*args, func=None, **kwargs):
        try:
            assert func is not None
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Exception in {func.__name__}: {e}")
            traceback.print_exc()
            raise
    
    results = list(
        tqdm(
            Parallel(n_jobs=num_workers, return_as="generator_unordered")(
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
    
    if info_dfs:
        get_pdb_seqs(info_dfs=info_dfs, out_dir=seq_index_dir)
    
    if all_sample_indices_lists:
        merged_results = []
        for sample_indices_list in all_sample_indices_lists:
            if sample_indices_list:
                merged_results += sample_indices_list
        df = pd.DataFrame(merged_results)
        df.to_csv(output_pdb_indices_csv, index=False, quoting=csv.QUOTE_NONNUMERIC)
    
    # 处理轨迹数据
    processor_config = {
        "store_dtype": torch.float32,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "remove_com_for_alignment": True,
        "align_rotation_for_alignment": False,
        "fpath": fourier_output_dir / "processor.pth",
    }
    
    if use_optimized:
        # 使用优化处理策略
        if num_cpu_workers is None:
            num_cpu_workers = max(1, mp.cpu_count() // 2)  # 使用一半CPU核心
        
        traj_name_list = process_trajectories_optimized(
            trajectory_list=trajectory_list,
            bioassembly_dict_DICT=bioassembly_dict_DICT,
            fourier_output_dir=fourier_output_dir,
            processor_config=processor_config,
            dt_str=dt_str,
            num_cpu_workers=num_cpu_workers,
            batch_size=batch_size,
        )
    else:
        # 使用原始处理策略
        processor = DataPipeline.init_trajectory_fft_processor(**processor_config)
        traj_name_list = []
        for traj_file in tqdm(trajectory_list, total=len(trajectory_list)):
            traj_name = str(traj_file.name)[:9].lower()
            bioassembly_dict = find_bioassembly_dict(str(traj_file.name), bioassembly_dict_DICT)
            if bioassembly_dict and len(bioassembly_dict) > 0:
                r = gen_a_trajectory_data(
                    trajectory_file=traj_file,
                    traj_name=traj_name,
                    fourier_output_dir=fourier_output_dir,
                    bioassembly_dict=bioassembly_dict,
                    processor=processor,
                    dt_str=dt_str,
                )
                if r is not None:
                    traj_name_list.append(r)
    
    df = pd.DataFrame({
        "traj_name": traj_name_list,
        "pdb_name": [str(name)[:6] for name in traj_name_list],
    })
    df.to_csv(output_traj_indices_csv, index=False, quoting=csv.QUOTE_NONNUMERIC)
    logger.info(f"[INFO] 轨迹索引已保存到: {output_traj_indices_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_path", type=Path, default=None)
    parser.add_argument("-opdb", "--output_pdb_indices_csv", type=Path, default=None)
    parser.add_argument("-otraj", "--output_traj_indices_csv", type=Path, default=None)
    parser.add_argument("-obio", "--bio_output_dir", type=Path, default=None)
    parser.add_argument("-ofo", "--fourier_output_dir", type=Path, default=None)
    parser.add_argument("-oseq", "--seq_index_dir", type=Path, default=None)
    parser.add_argument("-c", "--cluster_file", type=Path, default=None)
    parser.add_argument("-dt", "--dt_str", type=str, default="10ps")
    parser.add_argument("-n", "--n_cpu", type=int, default=1)
    parser.add_argument("--use_optimized", action="store_true", help="使用优化处理策略")
    parser.add_argument("--num_cpu_workers", type=int, default=None, help="CPU预处理工作进程数")
    parser.add_argument("--batch_size", type=int, default=4, help="GPU批处理大小")
    
    args = parser.parse_args()
    
    # 这里需要实现run_gen_data_optimized函数
    # 暂时使用原版，但可以添加优化选项

