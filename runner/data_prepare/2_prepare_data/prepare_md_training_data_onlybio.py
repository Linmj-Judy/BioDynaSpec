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
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
import torch
import numpy as np

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
) -> Optional[tuple[list[dict], dict]]:
    """
    Generates bioassembly data from a structure file and saves it to the specified output directory.

    Args:
        pdb (Path): Path to the structure (e.g. PDB or mmCIF) file.
        bioassembly_output_dir (Path): Directory where the bioassembly data will be saved.
        cluster_file (Optional[Path]): Path to the cluster file, if available.

    Returns:
        Optional[tuple[list[dict], dict]]: A tuple containing the list of sample indices and the 
            bioassembly_dict if data is successfully generated, otherwise None.
    """
    import time
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

    # if os.path.exists(bioassembly_output_dir / f"{pdb_id}.pkl.gz"):
    #     bioassembly_dict = load_gzip_pickle(bioassembly_output_dir / f"{pdb_id}.pkl.gz")
    #     return None, bioassembly_dict, None
    try:
        sample_indices_list, bioassembly_dict, info_df = DataPipeline.get_data_from_pdb(
            pdb, pdb_cluster_file=cluster_file,
            apply_filters=True
        )
        elapsed = time.time() - start_time
        logger.info(f"[DEBUG] 处理完成: {pdb}, 耗时: {elapsed:.2f}秒")
        logger.info(f"[DEBUG] sample_indices_list长度: {len(sample_indices_list) if sample_indices_list else 0}")
        sys.stdout.flush()
        
        cond_indices = sample_indices_list is not None and len(sample_indices_list) > 0
        cond_bio = bioassembly_dict is not None
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
                  f"info_df_nonempty={cond_info}")
            sys.stdout.flush()
            return None
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"[ERROR] 处理失败: {pdb}, 耗时: {elapsed:.2f}秒, 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
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


def preprocess_wrapper(traj_file: Path, bioassembly_dict: dict, dt_str: str) -> Optional[Tuple[str, np.ndarray, Dict[str, Any]]]:
    """
    CPU预处理包装函数，用于多进程处理。
    接受三个独立参数而不是元组，以便 ProcessPoolExecutor 可以正确 pickle。
    """
    return cpu_preprocess_trajectory(traj_file, bioassembly_dict, dt_str)


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
    """
    logger.info(f"[INFO] 开始处理 {len(pdb_list)} 个 PDB 文件...")
    sys.stdout.flush()

    # Parallel process structure files to generate bioassembly data
    # 为了暴露多线程中的异常，可以添加一个try/except包装，并在发生异常时直接raise或者打印完整堆栈后raise
    from joblib import parallel
    import traceback

    def safe_run_func_with_exception(*args, func=None, **kwargs):
        """
        通用的多线程安全函数调用，捕获并打印异常，便于调试。
        Args:
            func: 需要执行的函数
            *args, **kwargs: 传递给函数的参数
        """
        try:
            assert func is not None, "safe_run_func_with_exception requires a 'func' keyword argument"
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Exception in {func.__name__}: {e}")
            traceback.print_exc()
            sys.stdout.flush()
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
    # Separate valid results: filter out Nones, unpack tuples
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
        # 更新/生成序列索引文件
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
        "fpath": fourier_output_dir / "processor.pth",
    }
    
    logger.info(f"[INFO] 开始处理轨迹数据，共 {len(trajectory_list)} 个轨迹文件...")
    logger.info(f"[INFO] 使用优化策略: {use_optimized}")
    sys.stdout.flush()
    
    # 预先检查哪些轨迹找不到 bioassembly_dict
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
    
    # # 处理轨迹数据
    # traj_name_list = []
    # if fourier_save_minimal:
    #     save_minimal = True
    #     save_full_spectrum = False
    #     save_windowed = False
    #     save_psd = False
    # else:
    #     save_minimal = False
    #     save_full_spectrum = True
    #     save_windowed = True
    #     save_psd = True

    # if use_optimized:
    #     # 使用优化的混合处理策略
    #     if num_cpu_workers is None:
    #         num_cpu_workers = max(1, mp.cpu_count() // 2)
        
    #     logger.info(f"[INFO] 使用优化处理策略: CPU工作进程={num_cpu_workers}, GPU批大小={batch_size}")
    #     sys.stdout.flush()
        
    #     # 预先检查哪些轨迹已经处理过（输出文件已存在）
    #     trajectories_to_process = []
    #     skipped_existing = []
        
    #     for traj_file, bioassembly_dict in valid_trajectories:
    #         traj_name = str(traj_file.name)[:9].lower()
    #         output_path = fourier_output_dir / f"{traj_name}.pkl.gz"
            
    #         if output_path.exists():
    #             # 文件已存在，跳过处理
    #             skipped_existing.append(traj_name)
    #             traj_name_list.append(traj_name)  # 仍然添加到成功列表
    #         else:
    #             # 需要处理的轨迹
    #             trajectories_to_process.append((traj_file, bioassembly_dict))
        
    #     if skipped_existing:
    #         logger.info(f"[INFO] 发现 {len(skipped_existing)} 个轨迹的输出文件已存在，将跳过处理")
    #         sys.stdout.flush()
        
    #     if not trajectories_to_process:
    #         logger.info(f"[INFO] 所有轨迹都已处理完成，无需重新处理")
    #         sys.stdout.flush()
    #     else:
    #         logger.info(f"[INFO] 需要处理 {len(trajectories_to_process)} 个轨迹")
    #         sys.stdout.flush()
            
    #         # 创建异步保存队列
    #         save_queue = Queue()
    #         save_thread = threading.Thread(
    #             target=async_save_worker,
    #             args=(save_queue,),
    #             daemon=True
    #         )
    #         save_thread.start()
            
    #         # GPU处理函数
    #         processor = DataPipeline.init_trajectory_fft_processor(**processor_config)
    #         preprocessed_batch = []
            
    #         with ProcessPoolExecutor(max_workers=num_cpu_workers) as cpu_executor:
    #             # 提交所有CPU预处理任务
    #             # 直接传递三个参数，而不是元组，以便 ProcessPoolExecutor 可以正确 pickle
    #             future_to_traj = {
    #                 cpu_executor.submit(preprocess_wrapper, traj_file, bioassembly_dict, dt_str): traj_file
    #                 for traj_file, bioassembly_dict in trajectories_to_process
    #             }
            
    #             # 处理完成的任务
    #             with tqdm(total=len(trajectories_to_process), desc="处理轨迹") as pbar:
    #                 for future in as_completed(future_to_traj):
    #                     result = future.result()
    #                     if result is not None:
    #                         traj_name, traj_np, metadata = result
                            
    #                         # 再次检查文件是否存在（防止并发处理时重复）
    #                         output_path = fourier_output_dir / f"{traj_name}.pkl.gz"
    #                         if output_path.exists():
    #                             logger.debug(f"[DEBUG] 轨迹 {traj_name} 的输出文件已存在，跳过处理")
    #                             if traj_name not in traj_name_list:
    #                                 traj_name_list.append(traj_name)
    #                             pbar.update(1)
    #                             continue
                            
    #                         preprocessed_batch.append(result)
                            
    #                         # 当批次达到指定大小时，提交GPU处理
    #                         if len(preprocessed_batch) >= batch_size:
    #                             for batch_traj_name, batch_traj_np, batch_metadata in preprocessed_batch:
    #                                 try:
    #                                     # 再次检查（防止批次处理时文件已存在）
    #                                     batch_output_path = fourier_output_dir / f"{batch_traj_name}.pkl.gz"
    #                                     if batch_output_path.exists():
    #                                         logger.debug(f"[DEBUG] 轨迹 {batch_traj_name} 的输出文件已存在，跳过处理")
    #                                         if batch_traj_name not in traj_name_list:
    #                                             traj_name_list.append(batch_traj_name)
    #                                         continue
                                        
    #                                     traj = torch.from_numpy(batch_traj_np).to(processor.config.store_dtype)
                                        
    #                                     pack = processor.process_and_package(
    #                                         positions=traj,
    #                                         dt=batch_metadata["dt"],
    #                                         masses=batch_metadata["masses"],
    #                                         box_lengths=None,
    #                                         missing_mask=None,
    #                                         layout_hint="TN3",
    #                                         remove_com=True,
    #                                         align_rotation=False,
    #                                         save_minimal=save_minimal,
    #                                         save_full_spectrum=save_full_spectrum,
    #                                         save_windowed=save_windowed,
    #                                         save_psd=save_psd,
    #                                     )
    #                                     # 提交异步保存
    #                                     save_queue.put((batch_traj_name, pack, batch_output_path))
    #                                     traj_name_list.append(batch_traj_name)
                                        
    #                                 except Exception as e:
    #                                     logger.error(f"[ERROR] GPU FFT处理失败 {batch_traj_name}: {e}")
                                
    #                             preprocessed_batch = []
                        
    #                     pbar.update(1)
            
    #         # 处理剩余的批次
    #         if preprocessed_batch:
    #             for traj_name, traj_np, metadata in preprocessed_batch:
    #                 try:
    #                     # 再次检查文件是否存在
    #                     output_path = fourier_output_dir / f"{traj_name}.pkl.gz"
    #                     if output_path.exists():
    #                         logger.debug(f"[DEBUG] 轨迹 {traj_name} 的输出文件已存在，跳过处理")
    #                         if traj_name not in traj_name_list:
    #                             traj_name_list.append(traj_name)
    #                         continue
                        
    #                     traj = torch.from_numpy(traj_np).to(processor.config.store_dtype)
                        
    #                     pack = processor.process_and_package(
    #                         positions=traj,
    #                         dt=metadata["dt"],
    #                         masses=metadata["masses"],
    #                         box_lengths=None,
    #                         missing_mask=None,
    #                         layout_hint="TN3",
    #                         remove_com=True,
    #                         align_rotation=False,
    #                         save_minimal=True,
    #                         save_full_spectrum=False,
    #                         save_windowed=False,
    #                         save_psd=False,
    #                     )
                        
    #                     save_queue.put((traj_name, pack, output_path))
    #                     traj_name_list.append(traj_name)
                        
    #                 except Exception as e:
    #                     logger.error(f"[ERROR] GPU FFT处理失败 {traj_name}: {e}")
            
    #         # 等待所有保存完成
    #         save_queue.put(None)  # 结束信号
    #         save_thread.join(timeout=300)  # 最多等待5分钟
        
    # else:
    #     # 使用原始处理策略
    #     processor = DataPipeline.init_trajectory_fft_processor(**processor_config)
    #     logger.info(f"[INFO] 使用原始处理策略: {num_workers} 个进程")
    #     sys.stdout.flush()
        
    #     # 预先检查哪些轨迹已经处理过
    #     trajectories_to_process = []
    #     skipped_existing = []
        
    #     for traj_file, bioassembly_dict in valid_trajectories:
    #         traj_name = str(traj_file.name)[:9].lower()
    #         output_path = fourier_output_dir / f"{traj_name}.pkl.gz"
            
    #         if output_path.exists():
    #             skipped_existing.append(traj_name)
    #             traj_name_list.append(traj_name)
    #         else:
    #             trajectories_to_process.append((traj_file, bioassembly_dict))
        
    #     if skipped_existing:
    #         logger.info(f"[INFO] 发现 {len(skipped_existing)} 个轨迹的输出文件已存在，将跳过处理")
    #         sys.stdout.flush()
        
    #     if trajectories_to_process:
    #         logger.info(f"[INFO] 需要处理 {len(trajectories_to_process)} 个轨迹")
    #         sys.stdout.flush()
            
    #         for traj_file, bioassembly_dict in tqdm(trajectories_to_process, total=len(trajectories_to_process)):
    #             r = safe_run_func_with_exception(
    #                 func=gen_a_trajectory_data,
    #                 trajectory_file=traj_file,
    #                 traj_name=str(traj_file.name)[:9].lower(),
    #                 fourier_output_dir=fourier_output_dir,
    #                 bioassembly_dict=bioassembly_dict,
    #                 processor=processor,
    #                 dt_str=dt_str,
    #             )
    #             if r is not None:
    #                 traj_name_list.append(r)
    #     else:
    #         logger.info(f"[INFO] 所有轨迹都已处理完成，无需重新处理")
    #         sys.stdout.flush()
    # # 将输出改成两列，第一列"traj_name"，第二列"pdb_name"，后者为traj_name的前6位
    # df = pd.DataFrame(
    #     {
    #         "traj_name": traj_name_list,
    #         "pdb_name": [str(name)[:6] for name in traj_name_list],
    #     }
    # )
    # df.to_csv(output_traj_indices_csv, index=False, quoting=csv.QUOTE_NONNUMERIC)
    # logger.info(f"[INFO] 轨迹索引已保存到: {output_traj_indices_csv}")
    
    # 最终统计信息
    logger.info("=" * 80)
    logger.info("数据准备流程完成！")
    logger.info("=" * 80)
    logger.info(f"[INFO] 统计信息：")
    logger.info(f"  - 总轨迹数: {len(trajectory_list)}")
    # logger.info(f"  - 成功处理/已存在: {len(traj_name_list)}")
    if skipped_traj_names:
        logger.info(f"  - 跳过（找不到 bioassembly_dict）: {len(skipped_traj_names)}")
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
        pdb_list = list(input_path.rglob("*.pdb")) + list(input_path.rglob("*.pdb.gz"))
        trajectory_list = list(input_path.rglob("*.npy"))
        logger.info(f"[INFO] 找到 {len(pdb_list)} 个 PDB 文件和 {len(trajectory_list)} 个轨迹文件")
        sys.stdout.flush()
    
    elif input_path.suffix == ".txt":
        logger.info(f"[INFO] 从文件列表读取: {input_path}")
        sys.stdout.flush()
        with open(input_path) as f:
            pdb_list = [i.strip() for i in f.readlines()]
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
    )
# TODO:
# 1. 获取每个轨迹的pdb id list, 保存在txt中（保存每个轨迹的pdb id + 链名）
# 2. 根据pdb id list，从PDB数据库中获取静态晶体结构（）
# 3. 如果是单链（例如ATLAS），根据链名从
# 4. 输入两个CIF文件(来自PDB的静态晶体结构&抽帧得到的第一帧结构)，检查这两个文件是否是同一个分子/复合物
