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

import json
import logging
import os
import time
import traceback
import warnings
import signal
from typing import Any, Mapping
import threading  
from tqdm import tqdm
import pickle

import torch
import torch.multiprocessing
from biotite.structure import AtomArray
from ml_collections.config_dict import ConfigDict
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from src.data.basic.data_pipeline import DataPipeline
from src.data.basic.esm_featurizer import ESMFeaturizer
from configs.configs_model import ESM_CONFIG

from src.data.basic.json_to_feature import SampleDictToFeatures
from src.data.basic.msa_featurizer import InferenceMSAFeaturizer
from src.data.basic.utils import data_type_transform, make_dummy_feature
from src.utils.distributed import DIST_WRAPPER
from src.utils.torch_utils import dict_to_tensor

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", module="biotite")

class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("处理样本超时")

def process_with_timeout(func, single_sample_dict=None, timeout=360):  
    result = [None]  
    error = [None]  
    stop_event = threading.Event()

    def worker():  
        try:  
            result[0] = func(single_sample_dict=single_sample_dict)  
        except Exception as e:  
            error[0] = e  

    thread = threading.Thread(target=worker)  
    thread.daemon = True  # 设置为守护线程，这样主线程结束时它也会结束
    thread.start()  
    thread.join(timeout)  

    if thread.is_alive():  
        # 我们不能强制终止线程，但设置daemon=True并继续执行
        # 不再等待线程完成
        logger.warning("处理线程超时，但无法强制终止。设置为守护线程并继续")
        raise TimeoutError("处理样本超时")  
    
    if error[0]:  
        raise error[0]  # Re-raise the captured exception  

    return result[0]  

def get_inference_dataloader(configs: Any) -> DataLoader:
    """
    Creates and returns a DataLoader for inference using the InferenceDataset.

    Args:
        configs: A configuration object containing the necessary parameters for the DataLoader.

    Returns:
        A DataLoader object configured for inference.
    """
    # 设置 PyTorch 多进程共享策略，解决共享内存问题
    try:
        torch.multiprocessing.set_sharing_strategy("file_system")
    except RuntimeError:
        # 如果已经设置过，忽略错误
        pass
    
    # 设置共享内存目录环境变量（如果未设置）
    if "TORCH_SHM_DIR" not in os.environ:
        # 尝试使用临时目录或用户目录
        import tempfile
        shm_dir = os.path.join(tempfile.gettempdir(), "torch_shm")
        os.makedirs(shm_dir, exist_ok=True)
        os.environ["TORCH_SHM_DIR"] = shm_dir
        logger.info(f"设置 TORCH_SHM_DIR={shm_dir}")
    
    sample_timeout = getattr(configs, "sample_timeout", 360)
    cache_dir = os.path.join(configs.dump_dir, "feature_cache")
    os.makedirs(cache_dir, exist_ok=True)
    if configs.data.esm.embedding_dir is None:
        raise ValueError("embedding_dir is not set")
    cfg_esm = ConfigDict({
        "enable": configs.data.esm.enable,
        "model_name": configs.data.esm.model_name,
        "embedding_dir": configs.data.esm.embedding_dir,
        "embedding_dim": ESM_CONFIG[configs.data.esm.model_name]["emb_dim"],  # 从全局配置获取
        "sequence_fpath": configs.data.esm.sequence_fpath
    })

    inference_dataset = InferenceDataset(
        input_json_path=configs.input_json_path,
        dump_dir=configs.dump_dir,
        cfg_esm=cfg_esm,
        use_msa=configs.use_msa,
        use_esm=configs.use_esm,
        sample_timeout=sample_timeout,
        cache_dir=cache_dir
    )
    filtered_indices = inference_dataset.filter_problematic_samples()
    logger.info(f"筛选后保留 {len(filtered_indices)}/{len(inference_dataset)} 个有效样本")
    sampler = DistributedSampler(
        dataset=inference_dataset,
        num_replicas=DIST_WRAPPER.world_size,
        rank=DIST_WRAPPER.rank,
        shuffle=False,
    )
    preprocess_samples_distributed(inference_dataset, sampler, configs)
    cached_dataset = CachedInferenceDataset(inference_dataset)
    logger.info(f"Successfully created cached dataset")
    
    # 根据 num_workers 调整 pin_memory 设置
    num_workers = getattr(configs, "num_workers", 0)
    pin_memory = getattr(configs, "pin_memory", True) if num_workers > 0 else False
    
    # 如果共享内存有问题，可以禁用 pin_memory 或减少 num_workers
    if num_workers > 0:
        try:
            dataloader = DataLoader(
                dataset=cached_dataset,
                batch_size=1,
                sampler=sampler,
                collate_fn=lambda batch: batch,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=True if num_workers > 0 else False,
                multiprocessing_context='spawn' if num_workers > 0 else None
            )
        except RuntimeError as e:
            if "torch_shm_manager" in str(e) or "sharing" in str(e).lower():
                logger.warning(f"共享内存错误，尝试禁用 pin_memory: {e}")
                dataloader = DataLoader(
                    dataset=cached_dataset,
                    batch_size=1,
                    sampler=sampler,
                    collate_fn=lambda batch: batch,
                    num_workers=num_workers,
                    pin_memory=False,
                    persistent_workers=True if num_workers > 0 else False,
                    multiprocessing_context='spawn' if num_workers > 0 else None
                )
            else:
                raise
    else:
        dataloader = DataLoader(
            dataset=cached_dataset,
            batch_size=1,
            sampler=sampler,
            collate_fn=lambda batch: batch,
            num_workers=0,
            pin_memory=False
        )
    
    logging.info(f"Successfully created dataloader with num_workers={num_workers}, pin_memory={pin_memory}")
    return dataloader


class InferenceDataset(Dataset):
    def __init__(
        self,
        input_json_path: str,
        dump_dir: str,
        cfg_esm: ConfigDict,
        use_msa: bool = True,
        use_esm: bool = True,
        sample_timeout: int = 360,
        cache_dir: str = None,
    ) -> None:

        self.input_json_path = input_json_path
        self.dump_dir = dump_dir
        self.use_msa = use_msa
        self.use_esm = use_esm
        self.cfg_esm = cfg_esm
        self.sample_timeout = sample_timeout
        assert use_msa or use_esm, "Neither MSA feautre nor ESM feature is used."
        self.cache_dir = cache_dir or os.path.join(dump_dir, "feature_cache")
        
        with open(self.input_json_path, "r") as f:
            self.inputs = json.load(f)
        self.valid_indices = list(range(len(self.inputs)))
        self.name_to_index = {sample.get("name", f"unnamed_{i}"): i 
                              for i, sample in enumerate(self.inputs)}
        self.problem_samples_log = os.path.join(self.dump_dir, "problem_samples.json")
        
        # If ESM embedding doesn't exist, pre-compute them
        if use_esm:
            os.makedirs(cfg_esm.embedding_dir, exist_ok=True)
            os.makedirs(os.path.dirname(cfg_esm.sequence_fpath), exist_ok=True)
            
            if not os.listdir(cfg_esm.embedding_dir) or os.getenv("FORCE_COMPUTE_ESM_EMBEDDING") == "true":
                logger.info(f"ESM embedding目录为空，开始预计算ESM嵌入")
                ESMFeaturizer.precompute_esm_embedding(
                    self.inputs,
                    cfg_esm.model_name,
                    cfg_esm.embedding_dir,
                    cfg_esm.sequence_fpath,
                )
            else:
                logger.info(f"ESM embedding目录非空，跳过预计算步骤")
                
            self.esm_featurizer = ESMFeaturizer(
                embedding_dir=cfg_esm.embedding_dir,
                sequence_fpath=cfg_esm.sequence_fpath,
                embedding_dim=cfg_esm.embedding_dim,
            )
    def process_one(
        self,
        single_sample_dict: Mapping[str, Any],
    ) -> tuple[dict[str, torch.Tensor], AtomArray, dict[str, float]]:
        """
        Processes a single sample from the input JSON to generate features and statistics.

        Args:
            single_sample_dict: A dictionary containing the sample data.

        Returns:
            A tuple containing:
                - A dictionary of features.
                - An AtomArray object.
                - A dictionary of time tracking statistics.
        """
        # general features
        t0 = time.time()
        sample2feat = SampleDictToFeatures(
            single_sample_dict,
        )
        features_dict, atom_array, token_array = sample2feat.get_feature_dict()
        features_dict["distogram_rep_atom_mask"] = torch.Tensor(
            atom_array.distogram_rep_atom_mask
        ).long()
        entity_poly_type = sample2feat.entity_poly_type
        t1 = time.time()

        # MSA features
        if self.use_msa:
            entity_to_asym_id = DataPipeline.get_label_entity_id_to_asym_id_int(
                atom_array
            )
            msa_features = (
                InferenceMSAFeaturizer.make_msa_feature(
                    bioassembly=single_sample_dict["sequences"],
                    entity_to_asym_id=entity_to_asym_id,
                    token_array=token_array,
                    atom_array=atom_array,
                )
                if self.use_msa
                else {}
            )
        else:
            msa_features = []

        # ESM features
        dummy_feats = []  # 在此处初始化dummy_feats变量
        if self.use_esm:
            x_esm = self.esm_featurizer(
                token_array=token_array,
                atom_array=atom_array,
                bioassembly_dict=single_sample_dict,
                inference_mode=True,
            )
            features_dict["esm_token_embedding"] = x_esm["esm_token_embedding"]
            features_dict["esm_embedding_dim"] = self.cfg_esm.embedding_dim
            dummy_feats.append("esm")
        else:
            features_dict["esm_token_embedding"] = torch.zeros(
                [len(token_array), self.cfg_esm.embedding_dim]
            )
            features_dict["esm_embedding_dim"] = self.cfg_esm.embedding_dim
            dummy_feats.append("esm")

        # Make dummy features for not implemented features
        dummy_feats.append("template")  # 将template添加到已有的dummy_feats列表，而不是重新初始化
        if len(msa_features) == 0:
            dummy_feats.append("msa")
        else:
            msa_features = dict_to_tensor(msa_features)
            features_dict.update(msa_features)
        features_dict = make_dummy_feature(
            features_dict=features_dict,
            dummy_feats=dummy_feats,
        )

        # Transform to right data type
        feat = data_type_transform(feat_or_label_dict=features_dict)

        t2 = time.time()

        data = {}
        data["input_feature_dict"] = feat

        # Add dimension related items
        N_token = feat["token_index"].shape[0]
        N_atom = feat["atom_to_token_idx"].shape[0]
        N_msa = feat["msa"].shape[0]

        stats = {}
        for mol_type in ["ligand", "protein", "dna", "rna"]:
            mol_type_mask = feat[f"is_{mol_type}"].bool()
            stats[f"{mol_type}/atom"] = int(mol_type_mask.sum(dim=-1).item())
            stats[f"{mol_type}/token"] = len(
                torch.unique(feat["atom_to_token_idx"][mol_type_mask])
            )

        N_asym = len(torch.unique(data["input_feature_dict"]["asym_id"]))
        data.update(
            {
                "N_asym": torch.tensor([N_asym]),
                "N_token": torch.tensor([N_token]),
                "N_atom": torch.tensor([N_atom]),
                "N_msa": torch.tensor([N_msa]),
            }
        )

        def formatted_key(key):
            type_, unit = key.split("/")
            if type_ == "protein":
                type_ = "prot"
            elif type_ == "ligand":
                type_ = "lig"
            else:
                pass
            return f"N_{type_}_{unit}"

        data.update(
            {
                formatted_key(k): torch.tensor([stats[k]])
                for k in [
                    "protein/atom",
                    "ligand/atom",
                    "dna/atom",
                    "rna/atom",
                    "protein/token",
                    "ligand/token",
                    "dna/token",
                    "rna/token",
                ]
            }
        )
        data.update({"entity_poly_type": entity_poly_type})
        t3 = time.time()
        time_tracker = {
            "crop": t1 - t0,
            "featurizer": t2 - t1,
            "added_feature": t3 - t2,
        }

        return data, atom_array, time_tracker

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, index: int) -> tuple[dict[str, torch.Tensor], AtomArray, str]:
        error_message = ""
        try:
            single_sample_dict = self.inputs[index]
            sample_name = single_sample_dict["name"]
            logger.info(f"Featurizing sample {index}: {sample_name}...")
            
            # 在Windows系统中不使用信号机制
            if os.name != 'nt':
                # 设置超时处理
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(self.sample_timeout)
            
            try:
                start_time = time.time()
                data, atom_array, time_tracker = process_with_timeout(
                    self.process_one,
                    single_sample_dict=single_sample_dict, 
                    timeout=self.sample_timeout
                )
                if os.name != 'nt':
                    signal.alarm(0)  # 取消信号超时
                process_duration = time.time() - start_time
                logger.info(f"样本 {sample_name} 处理完成，耗时 {process_duration:.2f} 秒")
                logger.info(f"样本 {sample_name} 详细时间统计: {time_tracker}")
                
            except TimeoutError as te:
                logger.error(f"样本 {sample_name} 处理超时 (>{self.sample_timeout}秒)，跳过该样本")
                data, atom_array = {}, None
                error_message = f"处理超时: {str(te)}"
                # 确保取消超时
                if os.name != 'nt':
                    signal.alarm(0)
                
        except Exception as e:
            if os.name != 'nt':
                signal.alarm(0)
            data, atom_array = {}, None
            error_message = f"{e}:\n{traceback.format_exc()}"
            logger.error(f"处理样本 {index}:{sample_name} 时出错: {error_message}, 跳过该样本")

        data["sample_name"] = single_sample_dict["name"]
        data["sample_index"] = index
        return data, atom_array, error_message

    def filter_problematic_samples(self):
        """预处理并筛选出有问题的样本            
        Returns:
            list: 筛选后的有效样本索引列表
        """
        known_problem_samples = set()
        if os.path.exists(self.problem_samples_log):
            try:
                with open(self.problem_samples_log, "r") as f:
                    problem_data = json.load(f)
                    known_problem_samples = set(problem_data.get("problem_samples", []))
                    logger.info(f"加载了 {len(known_problem_samples)} 个已知问题样本")
            except Exception as e:
                logger.warning(f"读取问题样本日志失败: {e}")
        
        new_problem_samples = []
        
        valid_indices = []
        for i, sample in enumerate(self.inputs):
            sample_name = sample.get("name", f"unnamed_{i}")
            
            if sample_name in known_problem_samples:
                logger.info(f"跳过已知问题样本: {sample_name}")
                continue
        
            # 检查有问题的PDB代码
            known_problematic_pdbs = []  # 根据您的观察添加有问题的PDB
            if sample_name in known_problematic_pdbs:
                logger.warning(f"跳过已知有问题的PDB: {sample_name}")
                new_problem_samples.append(sample_name)
                continue
            
            # 检查样本结构，排除可能导致问题的样本
            try:
                if "sequences" not in sample or not sample["sequences"]:
                    logger.warning(f"样本 {sample_name} 缺少必要字段 'sequences'，跳过")
                    new_problem_samples.append(sample_name)
                    continue
                    
                # 可以添加更多检查...
                
            except Exception as e:
                logger.warning(f"检查样本 {sample_name} 时出错: {e}，跳过此样本")
                new_problem_samples.append(sample_name)
                continue
            
            valid_indices.append(i)
        
        self.valid_indices = valid_indices
        filtered_inputs = [self.inputs[i] for i in valid_indices]
        
        if new_problem_samples:
            all_problem_samples = list(known_problem_samples) + new_problem_samples
            try:
                with open(self.problem_samples_log, "w") as f:
                    json.dump({"problem_samples": all_problem_samples}, f, indent=2)
                logger.info(f"更新问题样本日志，添加了 {len(new_problem_samples)} 个新问题样本")
            except Exception as e:
                logger.error(f"保存问题样本日志失败: {e}")
                
        self.inputs = filtered_inputs
        logger.info(f"筛选后保留 {len(valid_indices)}/{len(self.inputs)} 个有效样本")
        return valid_indices

def preprocess_samples_distributed(dataset, sampler, configs):
    """分布式预处理并缓存样本
    
    Args:
        dataset: 数据集实例
        sampler: 分布式采样器
        configs: 配置对象
    """
    indices = list(sampler)
    total_samples = len(indices)
    rank = DIST_WRAPPER.rank
    world_size = DIST_WRAPPER.world_size
    
    logger.info(f"进程 {rank}/{world_size} 开始预处理 {total_samples} 个样本")
    
    processed_count = 0
    
    for i, idx in tqdm(enumerate(indices), 
                    total=total_samples, 
                    desc=f"进程 {rank} 预处理进度", 
                    ncols=100,
                    position=rank):
        sample_name = dataset.inputs[idx].get("name", f"unnamed_{idx}")
        cache_path = os.path.join(dataset.cache_dir, f"{sample_name}.pt")
        
        if os.path.exists(cache_path):
            processed_count += 1
            if i % 10 == 0:
                logger.info(f"进程 {rank}: 样本 {sample_name} 已缓存 ({processed_count}/{total_samples})")
            continue
            
        try:
            logger.info(f"进程 {rank}: 处理样本 {idx}: {sample_name}...")
            data, atom_array, error_msg = dataset.__getitem__(idx)
            
            if error_msg == "":  # 只缓存成功处理的样本
                save_data = {"data": data, "atom_array_exists": atom_array is not None}
                if atom_array is not None:
                    try:
                        save_data["atom_array_pickle"] = pickle.dumps(atom_array)
                        logger.info(f"成功序列化原子数组")
                    except Exception as e:
                        logger.warning(f"序列化原子数组失败: {e}")
                torch.save(save_data, cache_path)
                logger.info(f"进程 {rank}: 样本 {sample_name} 已缓存")
            else:
                logger.warning(f"进程 {rank}: 样本 {sample_name} 处理失败: {error_msg}")
                
            processed_count += 1
            if processed_count % 5 == 0:
                logger.info(f"进程 {rank}: 已处理 {processed_count}/{total_samples} 个样本")
                
        except Exception as e:
            logger.error(f"进程 {rank}: 预处理样本 {sample_name} 时出错: {e}")
    
    # 同步所有进程，确保所有预处理完成
    import torch.distributed as dist
    if world_size > 1:
        dist.barrier()
    
    logger.info(f"进程 {rank}: 预处理完成，共处理 {processed_count}/{total_samples} 个样本")


class CachedInferenceDataset(InferenceDataset):
    """包装InferenceDataset，优先从缓存加载数据"""
    
    def __init__(self, inference_dataset):
        """
        Args:
            inference_dataset: 原始的InferenceDataset实例
        """
        self.dataset = inference_dataset
        self.cache_dir = inference_dataset.cache_dir
        self.cache_hits = 0
        self.total_requests = 0
        
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, index):
        self.total_requests += 1
        sample = self.dataset.inputs[index]
        sample_name = sample.get("name", f"unnamed_{index}")
        cache_path = os.path.join(self.cache_dir, f"{sample_name}.pt")
        
        if os.path.exists(cache_path):
            try:
                cached_data = torch.load(cache_path, weights_only=False)
                self.cache_hits += 1
                hit_rate = self.cache_hits / self.total_requests * 100
                
                if index % 20 == 0:  # 减少日志输出频率
                    logger.info(f"从缓存加载样本 {sample_name}，缓存命中率: {hit_rate:.1f}%")
                    
                # 从缓存恢复数据
                data = cached_data["data"]
                if cached_data["atom_array_exists"]:
                    if "atom_array_pickle" in cached_data:
                        atom_array = pickle.loads(cached_data["atom_array_pickle"])
                    else:
                        logger.warning(f"样本 {sample_name} 缓存中缺少原子数组，将重新处理")
                        data, atom_array, _ = self.dataset.process_one(self.dataset.inputs[index])
                return data, atom_array, ""
                
            except Exception as e:
                logger.warning(f"加载样本 {sample_name} 缓存失败: {e}，将重新处理")
        
        return self.dataset.__getitem__(index)
    
    def __getattr__(self, name):
        """
        对于在此类中未定义的属性，尝试从原始数据集获取
        
        Args:
            name: 属性名称
            
        Returns:
            从原始数据集获取的属性值
            
        Raises:
            AttributeError: 如果原始数据集也没有该属性
        """
        if hasattr(self.dataset, name):
            return getattr(self.dataset, name)
        raise AttributeError(f"'{self.dataset.__class__.__name__}' object has no attribute '{name}'")
