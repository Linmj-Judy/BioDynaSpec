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
import os
import random
import traceback
import time
import signal
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Optional, Union

import numpy as np
import pandas as pd
import torch
from biotite.structure.atoms import AtomArray
from ml_collections.config_dict import ConfigDict
from torch.utils.data import Dataset

from protenix.data.constants import EvaluationChainInterface
from protenix.data.data_pipeline import DataPipeline, TimeoutError, timeout_handler, process_with_timeout
from protenix.data.featurizer import Featurizer
from protenix.data.msa_featurizer import MSAFeaturizer
from protenix.data.esm_featurizer import ESMFeaturizer
from protenix.data.tokenizer import TokenArray
from protenix.data.utils import data_type_transform, make_dummy_feature
from protenix.utils.cropping import CropData
from protenix.utils.file_io import read_indices_csv
from protenix.utils.logger import get_logger
from protenix.utils.torch_utils import dict_to_tensor
from protenix.data.compute_esm import ESM_CONFIG

logger = get_logger(__name__)


class BaseSingleDataset(Dataset):
    """
    dataset for a single data source
    data = self.__item__(idx)
    return a dict of features and labels, the keys and the shape are defined in protenix.data.utils
    """

    def __init__(
        self,
        mmcif_dir: Union[str, Path],
        bioassembly_dict_dir: Optional[Union[str, Path]],
        indices_fpath: Union[str, Path],
        cropping_configs: dict[str, Any],
        msa_featurizer: Optional[MSAFeaturizer] = None,
        esm_featurizer: Optional[ESMFeaturizer] = None,
        template_featurizer: Optional[Any] = None,
        name: str = None,
        use_msa: bool = False,
        use_esm: bool = False,
        cfg_esm: dict[str, Any] = {},
        **kwargs,
    ) -> None:
        super(BaseSingleDataset, self).__init__()

        # Configs
        self.mmcif_dir = mmcif_dir
        self.bioassembly_dict_dir = bioassembly_dict_dir
        self.indices_fpath = indices_fpath
        self.cropping_configs = cropping_configs
        self.name = name
        # General dataset configs
        self.ref_pos_augment = kwargs.get("ref_pos_augment", True)
        self.lig_atom_rename = kwargs.get("lig_atom_rename", False)
        self.reassign_continuous_chain_ids = kwargs.get(
            "reassign_continuous_chain_ids", False
        )
        self.shuffle_mols = kwargs.get("shuffle_mols", False)
        self.shuffle_sym_ids = kwargs.get("shuffle_sym_ids", False)

        # Typically used for test sets
        self.find_pocket = kwargs.get("find_pocket", False)
        self.find_all_pockets = kwargs.get("find_all_pockets", False)  # for dev
        self.find_eval_chain_interface = kwargs.get("find_eval_chain_interface", False)
        self.group_by_pdb_id = kwargs.get("group_by_pdb_id", False)  # for test set
        self.sort_by_n_token = kwargs.get("sort_by_n_token", False)

        # Typically used for training set
        self.random_sample_if_failed = kwargs.get("random_sample_if_failed", False)
        self.use_reference_chains_only = kwargs.get("use_reference_chains_only", False)
        self.is_distillation = kwargs.get("is_distillation", False)
        self.esm_fusion_rate = kwargs.get(
            "esm_fusion_rate", {"msa": 0.0, "esm": 1.0, "both": 0.0, "neither": 0.0}
        )
        # Configs for data filters
        self.max_n_token = kwargs.get("max_n_token", -1)
        self.pdb_list = kwargs.get("pdb_list", None)
        if len(self.pdb_list) == 0:
            self.pdb_list = None
        # Used for removing rows in the indices list. Column names and excluded values are specified in this dict.
        self.exclusion_dict = kwargs.get("exclusion", {})
        self.limits = kwargs.get(
            "limits", -1
        )  # Limit number of indices rows, mainly for test

        self.error_dir = kwargs.get("error_dir", None)
        if self.error_dir is not None:
            os.makedirs(self.error_dir, exist_ok=True)

        self.return_atom_token_array = kwargs.get("return_atom_token_array", False)

        self.msa_featurizer = msa_featurizer
        self.esm_featurizer = esm_featurizer
        self.template_featurizer = template_featurizer
        self.use_msa = use_msa
        # self.use_esm = use_esm # TODO: 测试用,暂时改
        self.use_esm = use_esm
        self.cfg_esm = cfg_esm
        assert self.use_msa or self.use_esm, "Neither MSA feautre nor ESM feature is used."

        self.cache_dir = kwargs.get("cache_dir", os.path.join(os.path.dirname(indices_fpath), "feature_cache"))
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self.use_cache = kwargs.get("use_cache", True)
        self.sample_timeout = kwargs.get("sample_timeout", 360)
        
        # 问题样本日志
        self.problem_samples_log = os.path.join(os.path.dirname(self.cache_dir), "problem_samples.json")

        # Read data
        self.known_problem_samples = self._load_known_problem_samples()
        self.indices_list = self.read_indices_list(indices_fpath)
        if kwargs.get("filter_problematic_samples", True):
            self.filter_problematic_samples()

        self.invalid_count = 0
        self.cache_hits = 0
        self.total_requests = 0

    def _load_known_problem_samples(self) -> set:
        """
        加载已知问题样本
        
        Returns:
            set: 问题样本名称集合
        """
        known_problem_samples = set()
        if os.path.exists(self.problem_samples_log):
            try:
                with open(self.problem_samples_log, "r") as f:
                    problem_data = json.load(f)
                    for sample_name in problem_data.get("problem_samples", {}):
                        known_problem_samples.add(sample_name)
                    logger.info(f"加载了 {len(known_problem_samples)} 个已知问题样本")
            except Exception as e:
                logger.warning(f"读取问题样本日志失败: {e}")
        return known_problem_samples

    def filter_problematic_samples(self):
        """
        预处理并筛选出有问题的样本
        
        Returns:
            list: 筛选后的有效样本索引列表
        """
        new_problem_samples = []
        
        if isinstance(self.indices_list, list):
            # 针对group_by_pdb_id=True的情况
            filtered_indices_list = []
            valid_indices = []
            
            for i, df in enumerate(self.indices_list):
                if len(df) == 0:
                    continue
                    
                sample_name = df["pdb_id"].iloc[0]
                if sample_name in self.known_problem_samples:
                    logger.info(f"跳过已知问题样本: {sample_name}")
                    continue
                    
                # 检查有问题的PDB代码
                known_problematic_pdbs = []  # 根据观察添加有问题的PDB
                if sample_name in known_problematic_pdbs:
                    logger.warning(f"跳过已知有问题的PDB: {sample_name}")
                    new_problem_samples.append(sample_name)
                    continue
                    
                filtered_indices_list.append(df)
                valid_indices.append(i)
                
            self.indices_list = filtered_indices_list
            
        else:
            # 针对DataFrame的情况
            filtered_indices = []
            for i, row in self.indices_list.iterrows():
                sample_name = row["pdb_id"]
                
                if sample_name in self.known_problem_samples:
                    logger.info(f"跳过已知问题样本: {sample_name}")
                    continue
                    
                # 检查有问题的PDB代码
                known_problematic_pdbs = []  # 根据观察添加有问题的PDB
                if sample_name in known_problematic_pdbs:
                    logger.warning(f"跳过已知有问题的PDB: {sample_name}")
                    new_problem_samples.append(sample_name)
                    continue
                    
                filtered_indices.append(i)
                
            self.indices_list = self.indices_list.iloc[filtered_indices].reset_index(drop=True)
            valid_indices = filtered_indices
            
        # 更新问题样本日志
        if new_problem_samples:
            try:
                # 读取现有问题样本记录
                if os.path.exists(self.problem_samples_log):
                    with open(self.problem_samples_log, "r") as f:
                        log_data = json.load(f)
                else:
                    log_data = {"problem_samples": {}}
                    
                # 添加新的问题样本
                for sample_name in new_problem_samples:
                    if sample_name not in log_data["problem_samples"]:
                        log_data["problem_samples"][sample_name] = []
                    
                    log_data["problem_samples"][sample_name].append({
                        "error": "预过滤识别为问题样本",
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                    })
                self.known_problem_samples.update(new_problem_samples)
                
                # 写回文件
                with open(self.problem_samples_log, "w") as f:
                    json.dump(log_data, f, indent=2)
                    
                logger.info(f"更新问题样本日志，添加了 {len(new_problem_samples)} 个新问题样本")
            except Exception as e:
                logger.error(f"保存问题样本日志失败: {e}")
                
        logger.info(f"筛选后保留 {len(self.indices_list)}/{len(valid_indices)} 个有效样本")
        return valid_indices
    
    @staticmethod
    def read_pdb_list(pdb_list: Union[list, str]) -> Optional[list]:
        """
        Reads a list of PDB IDs from a file or directly from a list.

        Args:
            pdb_list: A list of PDB IDs or a file path containing PDB IDs.

        Returns:
            A list of PDB IDs if the input is valid, otherwise None.
        """
        if pdb_list is None:
            return None

        if isinstance(pdb_list, list):
            return pdb_list

        with open(pdb_list, "r") as f:
            pdb_filter_list = []
            for l in f.readlines():
                l = l.strip()
                if l:
                    pdb_filter_list.append(l)
        return pdb_filter_list

    def read_indices_list(self, indices_fpath: Union[str, Path]) -> pd.DataFrame:
        """
        Reads and processes a list of indices from a CSV file.

        Args:
            indices_fpath: Path to the CSV file containing the indices.

        Returns:
            A DataFrame containing the processed indices.
        """
        indices_list = read_indices_csv(indices_fpath)
        num_data = len(indices_list)
        logger.info(f"#Rows in indices list: {num_data}")
        # Filter by pdb_list
        if self.pdb_list is not None:
            pdb_filter_list = set(self.read_pdb_list(pdb_list=self.pdb_list))
            indices_list = indices_list[indices_list["pdb_id"].isin(pdb_filter_list)]
            logger.info(f"[filtered by pdb_list] #Rows: {len(indices_list)}")

        # Filter by max_n_token
        if self.max_n_token > 0:
            valid_mask = indices_list["num_tokens"].astype(int) <= self.max_n_token
            removed_list = indices_list[~valid_mask]
            indices_list = indices_list[valid_mask]
            logger.info(f"[removed] #Rows: {len(removed_list)}")
            logger.info(f"[removed] #PDB: {removed_list['pdb_id'].nunique()}")
            logger.info(
                f"[filtered by n_token ({self.max_n_token})] #Rows: {len(indices_list)}"
            )

        # Filter by exclusion_dict
        for col_name, exclusion_list in self.exclusion_dict.items():
            cols = col_name.split("|")
            exclusion_set = {tuple(excl.split("|")) for excl in exclusion_list}

            def is_valid(row):
                return tuple(row[col] for col in cols) not in exclusion_set

            valid_mask = indices_list.apply(is_valid, axis=1)
            indices_list = indices_list[valid_mask].reset_index(drop=True)
            logger.info(
                f"[Excluded by {col_name} -- {exclusion_list}] #Rows: {len(indices_list)}"
            )
        self.print_data_stats(indices_list)
        indices_list = indices_list.sort_values(by="pdb_id").reset_index(drop=True)

        # Group by pdb_id
        # A list of dataframe. Each contains one pdb with multiple rows.
        if self.group_by_pdb_id:
            indices_list = [
                df.reset_index() for _, df in indices_list.groupby("pdb_id", sort=True)
            ]

        if self.sort_by_n_token:
            # Sort the dataset in a descending order, so that if OOM it will raise Error at an early stage.
            if self.group_by_pdb_id:
                indices_list = sorted(
                    indices_list,
                    key=lambda df: int(df["num_tokens"].iloc[0]),
                    reverse=True,
                )
            else:
                indices_list = indices_list.sort_values(
                    by="num_tokens", key=lambda x: x.astype(int), ascending=False
                ).reset_index(drop=True)

        if self.find_eval_chain_interface:
            # Remove data that does not contain eval_type in the EvaluationChainInterface list
            if self.group_by_pdb_id:
                indices_list = [
                    df
                    for df in indices_list
                    if len(
                        set(df["eval_type"].to_list()).intersection(
                            set(EvaluationChainInterface)
                        )
                    )
                    > 0
                ]
            else:
                indices_list = indices_list[
                    indices_list["eval_type"].apply(
                        lambda x: x in EvaluationChainInterface
                    )
                ]
        if self.limits > 0 and len(indices_list) > self.limits:
            logger.info(
                f"Limit indices list size from {len(indices_list)} to {self.limits}"
            )
            indices_list = indices_list[: self.limits]
        return indices_list

    def print_data_stats(self, df: pd.DataFrame) -> None:
        """
        Prints statistics about the dataset, including the distribution of molecular group types.

        Args:
            df: A DataFrame containing the indices list.
        """
        if self.name:
            logger.info("-" * 10 + f" Dataset {self.name}" + "-" * 10)
        if df.empty:  
            raise ValueError("DataFrame is empty in line 380 of dataset.py!") 
        df["mol_group_type"] = df.apply(
            lambda row: "_".join(
                sorted(
                    [
                        str(row["mol_1_type"]),
                        str(row["mol_2_type"]).replace("nan", "intra"),
                    ]
                )
            ),
            axis=1,
        )

        group_size_dict = dict(df["mol_group_type"].value_counts())
        for i, n_i in group_size_dict.items():
            logger.info(f"{i}: {n_i}/{len(df)}({round(n_i*100/len(df), 2)}%)")

        logger.info("-" * 30)
        if "cluster_id" in df.columns:
            n_cluster = df["cluster_id"].nunique()
            for i in group_size_dict:
                n_i = df[df["mol_group_type"] == i]["cluster_id"].nunique()
                logger.info(f"{i}: {n_i}/{n_cluster}({round(n_i*100/n_cluster, 2)}%)")
            logger.info("-" * 30)

        logger.info(f"Final pdb ids: {len(set(df.pdb_id.tolist()))}")
        logger.info("-" * 30)

    def __len__(self) -> int:
        return len(self.indices_list)

    def save_error_data(self, idx: int, error_message: str) -> None:
        """
        Saves the error data for a specific index to a JSON file in the error directory.

        Args:
            idx: The index of the data sample that caused the error.
            error_message: The error message to be saved.
        """
        if self.error_dir is not None:
            sample_indice = self._get_sample_indice(idx=idx)
            data = sample_indice.to_dict()
            data["error"] = error_message

            filename = f"{sample_indice.pdb_id}-{sample_indice.chain_1_id}-{sample_indice.chain_2_id}.json"
            fpath = os.path.join(self.error_dir, filename)
            if not os.path.exists(fpath):
                with open(fpath, "w") as f:
                    json.dump(data, f)

    def __getitem__(self, idx: int):
        """
        Retrieves a data sample by processing the given index.
        If an error occurs, it attempts to handle it by either saving the error data or randomly sampling another index.

        Args:
            idx: The index of the data sample to retrieve.

        Returns:
            A dictionary containing the processed data sample.
        """
        self.total_requests += 1
        sample_indice = self._get_sample_indice(idx=idx)
        sample_name = sample_indice["pdb_id"]
        cache_path = os.path.join(self.cache_dir, f"{sample_name}.pt")
        if self.use_cache and os.path.exists(cache_path):
            try:
                data, atom_array, token_array, success_flag = DataPipeline.load_cached_feature(cache_dir=self.cache_dir, sample_name=sample_name)
                if success_flag:
                    self.cache_hits += 1
                    hit_rate = self.cache_hits / self.total_requests * 100
                
                if idx % 20 == 0:  # 减少日志输出频率
                    logger.info(f"从缓存加载样本 {sample_name}，缓存命中率: {hit_rate:.1f}%")
                    
                return data
                
            except Exception as e:
                logger.warning(f"加载样本 {sample_name} 缓存失败: {e}，将重新处理")

        # Try at most 10 times
        data = None
        last_exception = None
        for attempt in range(10):
            try:
                # 使用超时机制处理样本
                if os.name != 'nt':
                    # 设置超时处理（不适用于Windows）
                    signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(self.sample_timeout)
                
                try:
                    start_time = time.time()
                    data = process_with_timeout(
                        func=self.process_one,
                        args=(idx,),
                        kwargs={"return_atom_token_array": self.return_atom_token_array},
                        timeout=self.sample_timeout
                    )
                    
                    if os.name != 'nt':
                        signal.alarm(0)  # 取消信号超时
                        
                    process_duration = time.time() - start_time
                    logger.info(f"样本 {sample_name} 处理完成，耗时 {process_duration:.2f} 秒")
                    
                    # 缓存成功处理的结果
                    if self.use_cache and data:
                        DataPipeline.cache_feature(
                            cache_dir=self.cache_dir,
                            sample_name=sample_name,
                            data=data,
                            atom_array=data.get("cropped_atom_array") if isinstance(data, dict) and "cropped_atom_array" in data else None,
                            token_array=data.get("cropped_token_array") if isinstance(data, dict) and "cropped_token_array" in data else None
                        )
                    
                    return data
                    
                except TimeoutError as te:
                    if os.name != 'nt':
                        signal.alarm(0)
                    logger.error(f"样本 {sample_name} 处理超时 (>{self.sample_timeout}秒)，尝试 {attempt+1}/10")
                    last_exception = te
                    # 记录问题样本
                    DataPipeline.log_problem_sample(
                        problem_samples_log=self.problem_samples_log,
                        sample_name=sample_name,
                        error_message=f"处理超时: {str(te)}"
                    )
                    continue
                
            except Exception as e:
                if os.name != 'nt':
                    signal.alarm(0)
                error_message = f"{e} at idx {idx}:\n{traceback.format_exc()}"
                logger.error(f"处理样本 {sample_name} 时出错: {error_message}")
                self.save_error_data(idx, error_message)
                last_exception = e
                
                # 记录问题样本
                DataPipeline.log_problem_sample(
                    problem_samples_log=self.problem_samples_log,
                    sample_name=sample_name,
                    error_message=error_message
                )

                if self.random_sample_if_failed:
                    logger.info(f"[skip data {idx}: {sample_name}] 随机选择新样本...")
                    # 随机选择另一个索引重试
                    idx = random.choice(range(len(self.indices_list)))
                    sample_indice = self._get_sample_indice(idx=idx)
                    sample_name = sample_indice["pdb_id"]
                    continue
                else:
                    break
        
        if data is None:
            self.invalid_count += 1
            logger.warning(f"样本 {sample_name} 处理失败，返回空数据")
            # 创建一个最小可用的数据结构
            data = {
                "input_feature_dict": {},
                "label_dict": {},
                "label_full_dict": {},
                "basic": {"error": str(last_exception), "pdb_id": sample_name}
            }
            
        return data

    def _get_bioassembly_data(
        self, idx: int
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        sample_indice = self._get_sample_indice(idx=idx)
        if self.bioassembly_dict_dir is not None:
            bioassembly_dict_fpath = os.path.join(
                self.bioassembly_dict_dir, sample_indice.pdb_id + ".pkl.gz"
            )
        else:
            bioassembly_dict_fpath = None

        bioassembly_dict = DataPipeline.get_data_bioassembly(
            bioassembly_dict_fpath=bioassembly_dict_fpath
        )
        
        bioassembly_dict["pdb_id"] = sample_indice.pdb_id
        return sample_indice, bioassembly_dict, bioassembly_dict_fpath

    @staticmethod
    def _reassign_atom_array_chain_id(atom_array: AtomArray):
        """
        In experiments conducted to observe overfitting effects using training sets,
        the pre-stored AtomArray in the training set may experience issues with discontinuous chain IDs due to filtering.
        Consequently, a temporary patch has been implemented to resolve this issue.

        e.g. 3x6u asym_id_int = [0, 1, 2, ... 18, 20] -> reassigned_asym_id_int [0, 1, 2, ..., 18, 19]
        """

        def _get_contiguous_array(array):
            array_uniq = np.sort(np.unique(array))
            map_dict = {i: idx for idx, i in enumerate(array_uniq)}
            new_array = np.vectorize(map_dict.get)(array)
            return new_array

        atom_array.asym_id_int = _get_contiguous_array(atom_array.asym_id_int)
        atom_array.entity_id_int = _get_contiguous_array(atom_array.entity_id_int)
        atom_array.sym_id_int = _get_contiguous_array(atom_array.sym_id_int)
        return atom_array

    @staticmethod
    def _shuffle_array_based_on_mol_id(token_array: TokenArray, atom_array: AtomArray):
        """
        Shuffle both token_array and atom_array.
        Atoms/tokens with the same mol_id will be shuffled as a integrated component.
        """

        # Get token mol_id
        centre_atom_indices = token_array.get_annotation("centre_atom_index")
        token_mol_id = atom_array[centre_atom_indices].mol_id

        # Get unique molecule IDs and shuffle them in place
        shuffled_mol_ids = np.unique(token_mol_id).copy()
        np.random.shuffle(shuffled_mol_ids)

        # Get shuffled token indices
        original_token_indices = np.arange(len(token_mol_id))
        shuffled_token_indices = []
        for mol_id in shuffled_mol_ids:
            mol_token_indices = original_token_indices[token_mol_id == mol_id]
            shuffled_token_indices.append(mol_token_indices)
        shuffled_token_indices = np.concatenate(shuffled_token_indices)

        # Get shuffled token and atom array
        # Use `CropData.select_by_token_indices` to shuffle safely
        token_array, atom_array, _, _, _ = CropData.select_by_token_indices(
            token_array=token_array,
            atom_array=atom_array,
            selected_token_indices=shuffled_token_indices,
        )

        return token_array, atom_array

    @staticmethod
    def _assign_random_sym_id(atom_array: AtomArray):
        """
        Assign random sym_id for chains of the same entity_id
        e.g.
        when entity_id = 0
            sym_id_int = [0, 1, 2] -> random_sym_id_int = [2, 0, 1]
        when entity_id = 1
            sym_id_int = [0, 1, 2, 3] -> random_sym_id_int = [3, 0, 1, 2]
        """

        def _shuffle(x):
            x_unique = np.sort(np.unique(x))
            x_shuffled = x_unique.copy()
            np.random.shuffle(x_shuffled)  # shuffle in-place
            map_dict = dict(zip(x_unique, x_shuffled))
            new_x = np.vectorize(map_dict.get)(x)
            return new_x.copy()

        for entity_id in np.unique(atom_array.label_entity_id):
            mask = atom_array.label_entity_id == entity_id
            atom_array.sym_id_int[mask] = _shuffle(atom_array.sym_id_int[mask])
        return atom_array

    def process_one(
        self, idx: int, return_atom_token_array: bool = False, inference_mode: bool = False
    ) -> dict[str, dict]:
        """
        Processes a single data sample by retrieving bioassembly data, applying various transformations, and cropping the data.
        It then extracts features and labels, and optionally returns the processed atom and token arrays.

        Args:
            idx: The index of the data sample to process.
            return_atom_token_array: Whether to return the processed atom and token arrays.

        Returns:
            A dict containing the input features, labels, basic_info and optionally the processed atom and token arrays.
        """

        sample_indice, bioassembly_dict, bioassembly_dict_fpath = (
            self._get_bioassembly_data(idx=idx)
        )

        if self.use_reference_chains_only:
            # Get the reference chains
            ref_chain_ids = [sample_indice.chain_1_id, sample_indice.chain_2_id]
            if sample_indice.type == "chain":
                ref_chain_ids.pop(-1)
            # Remove other chains from the bioassembly_dict
            # Remove them safely using the crop method
            token_centre_atom_indices = bioassembly_dict["token_array"].get_annotation(
                "centre_atom_index"
            )
            token_chain_id = bioassembly_dict["atom_array"][
                token_centre_atom_indices
            ].chain_id
            is_ref_chain = np.isin(token_chain_id, ref_chain_ids)
            bioassembly_dict["token_array"], bioassembly_dict["atom_array"], _, _ = (
                CropData.select_by_token_indices(
                    token_array=bioassembly_dict["token_array"],
                    atom_array=bioassembly_dict["atom_array"],
                    selected_token_indices=np.arange(len(is_ref_chain))[is_ref_chain],
                )
            )

        if self.shuffle_mols:
            bioassembly_dict["token_array"], bioassembly_dict["atom_array"] = (
                self._shuffle_array_based_on_mol_id(
                    token_array=bioassembly_dict["token_array"],
                    atom_array=bioassembly_dict["atom_array"],
                )
            )

        if self.shuffle_sym_ids:
            bioassembly_dict["atom_array"] = self._assign_random_sym_id(
                bioassembly_dict["atom_array"]
            )

        if self.reassign_continuous_chain_ids:
            bioassembly_dict["atom_array"] = self._reassign_atom_array_chain_id(
                bioassembly_dict["atom_array"]
            )

        # Crop
        (
            crop_method,
            cropped_token_array,
            cropped_atom_array,
            cropped_msa_features,
            cropped_esm_features,
            cropped_template_features,
            reference_token_index,
        ) = self.crop(
            sample_indice=sample_indice,
            bioassembly_dict=bioassembly_dict,
            use_msa=self.use_msa,
            use_esm=self.use_esm,
            cfg_esm=self.cfg_esm,
            **self.cropping_configs,
        )
        # Randomly keep msa or esm features
        assert np.isclose(sum(self.esm_fusion_rate.values()), 1.0, rtol=1e-9)
        drop_msa, drop_esm = False, False
        p = np.random.rand()
        if p < self.esm_fusion_rate["msa"]:
            # only keep msa
            drop_esm = True
        elif p < self.esm_fusion_rate["msa"] + self.esm_fusion_rate["esm"]:
            # only keep esm
            drop_msa = True
        elif p > (
            self.esm_fusion_rate["msa"]
            + self.esm_fusion_rate["esm"]
            + self.esm_fusion_rate["both"]
        ):
            # do not use msa and esm
            drop_esm = True
            drop_msa = True

        feat, label, label_full = self.get_feature_and_label(
            idx=idx,
            token_array=cropped_token_array,
            atom_array=cropped_atom_array,
            msa_features=cropped_msa_features,
            esm_features=cropped_esm_features,
            template_features=cropped_template_features,
            full_atom_array=bioassembly_dict["atom_array"],
            is_spatial_crop="spatial" in crop_method.lower(),
        )

        # ESM feature
        if self.use_esm and self.esm_featurizer is not None:
            x_esm = self.esm_featurizer(
                token_array=cropped_token_array,
                atom_array=cropped_atom_array,
                bioassembly_dict=bioassembly_dict,
                inference_mode=inference_mode
            )
            if drop_esm:
                feat["esm_token_embedding"] = torch.zeros_like(x_esm["esm_token_embedding"])
            else:
                feat["esm_token_embedding"] = x_esm["esm_token_embedding"]
        if drop_msa and not drop_esm:
            feat["esm_fusion_mode"] = "esm"
        elif not drop_msa and drop_esm:
            feat["esm_fusion_mode"] = "msa"
        elif (not drop_msa) and (not drop_esm):
            feat["esm_fusion_mode"] = "both"
        else:
            feat["esm_fusion_mode"] = "neither"

        # Basic info, e.g. dimension related items
        basic_info = {
            "pdb_id": (
                bioassembly_dict["pdb_id"]
                if self.is_distillation is False
                else sample_indice["pdb_id"]
            ),
            "N_asym": torch.tensor([len(torch.unique(feat["asym_id"]))]),
            "N_token": torch.tensor([feat["token_index"].shape[0]]),
            "N_atom": torch.tensor([feat["atom_to_token_idx"].shape[0]]),
            "N_msa": torch.tensor([feat["msa"].shape[0]]),
            "bioassembly_dict_fpath": bioassembly_dict_fpath,
            "N_msa_prot_pair": torch.tensor([feat["prot_pair_num_alignments"]]),
            "N_msa_prot_unpair": torch.tensor([feat["prot_unpair_num_alignments"]]),
            "N_msa_rna_pair": torch.tensor([feat["rna_pair_num_alignments"]]),
            "N_msa_rna_unpair": torch.tensor([feat["rna_unpair_num_alignments"]]),
        }

        for mol_type in ("protein", "ligand", "rna", "dna"):
            abbr = {"protein": "prot", "ligand": "lig"}
            abbr_type = abbr.get(mol_type, mol_type)
            mol_type_mask = feat[f"is_{mol_type}"].bool()
            n_atom = int(mol_type_mask.sum(dim=-1).item())
            n_token = len(torch.unique(feat["atom_to_token_idx"][mol_type_mask]))
            basic_info[f"N_{abbr_type}_atom"] = torch.tensor([n_atom])
            basic_info[f"N_{abbr_type}_token"] = torch.tensor([n_token])

        # Add chain level chain_id
        asymn_id_to_chain_id = {
            atom.asym_id_int: atom.chain_id for atom in cropped_atom_array
        }
        chain_id_list = [
            asymn_id_to_chain_id[asymn_id_int]
            for asymn_id_int in sorted(asymn_id_to_chain_id.keys())
        ]
        basic_info["chain_id"] = chain_id_list

        data = {
            "input_feature_dict": feat,
            "label_dict": label,
            "label_full_dict": label_full,
            "basic": basic_info,
        }

        if return_atom_token_array:
            data["cropped_atom_array"] = cropped_atom_array
            data["cropped_token_array"] = cropped_token_array
        return data

    def crop(
        self,
        sample_indice: pd.Series,
        bioassembly_dict: dict[str, Any],
        crop_size: int,
        method_weights: list[float],
        contiguous_crop_complete_lig: bool = True,
        spatial_crop_complete_lig: bool = True,
        drop_last: bool = True,
        remove_metal: bool = True,
        use_msa: bool = False,
        use_esm: bool = False,
        cfg_esm: dict[str, Any] = {},
    ) -> tuple[str, TokenArray, AtomArray, dict[str, Any], dict[str, Any]]:
        """
        Crops the bioassembly data based on the specified configurations.

        Returns:
            A tuple containing the cropping method, cropped token array, cropped atom array,
                cropped MSA features, and cropped template features.
        """
        return DataPipeline.crop(
            one_sample=sample_indice,
            bioassembly_dict=bioassembly_dict,
            crop_size=crop_size,
            msa_featurizer=self.msa_featurizer,
            esm_featurizer=self.esm_featurizer,
            template_featurizer=self.template_featurizer,
            method_weights=method_weights,
            contiguous_crop_complete_lig=contiguous_crop_complete_lig,
            spatial_crop_complete_lig=spatial_crop_complete_lig,
            drop_last=drop_last,
            remove_metal=remove_metal,
            use_msa=use_msa,
            use_esm=use_esm,
            cfg_esm=cfg_esm,
        )

    def _get_sample_indice(self, idx: int) -> pd.Series:
        """
        Retrieves the sample indice for a given index. If the dataset is grouped by PDB ID, it returns the first row of the PDB-idx.
        Otherwise, it returns the row at the specified index.

        Args:
            idx: The index of the data sample to retrieve.

        Returns:
            A pandas Series containing the sample indice.
        """
        if self.group_by_pdb_id:
            # Row-0 of PDB-idx
            sample_indice = self.indices_list[idx].iloc[0]
        else:
            sample_indice = self.indices_list.iloc[idx]
        return sample_indice
    
    def _get_idx_by_pdb_id(self, pdb_id: str) -> Optional[int]:
        """
        Retrieves the index of a data sample by PDB ID.
        
        Args:
            pdb_id: The PDB ID of the data sample to retrieve.

        Returns:
            The index of the data sample with the specified PDB ID, or None if no match is found.
        """
        if self.group_by_pdb_id:
            for idx, group_df in enumerate(self.indices_list):
                if len(group_df) > 0 and group_df.iloc[0]["pdb_id"] == pdb_id:
                    return idx
        else:
            matches = self.indices_list[self.indices_list["pdb_id"] == pdb_id]
            if not matches.empty:
                return matches.index[0]
        
        return None
    
    def _get_eval_chain_interface_mask(
        self, idx: int, atom_array_chain_id: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, torch.Tensor, torch.Tensor]:
        """
        Retrieves the evaluation chain/interface mask for a given index.

        Args:
            idx: The index of the data sample.
            atom_array_chain_id: An array containing the chain IDs of the atom array.

        Returns:
            A tuple containing the evaluation type, cluster ID, chain 1 mask, and chain 2 mask.
        """
        if self.group_by_pdb_id:
            df = self.indices_list[idx]
        else:
            df = self.indices_list.iloc[idx : idx + 1]

        # Only consider chain/interfaces defined in EvaluationChainInterface
        df = df[df["eval_type"].apply(lambda x: x in EvaluationChainInterface)].copy()
        if len(df) < 1:
            raise ValueError(
                f"Cannot find a chain/interface for evaluation in the PDB."
            )

        def get_atom_mask(row):
            chain_1_mask = atom_array_chain_id == row["chain_1_id"]
            if row["type"] == "chain":
                chain_2_mask = chain_1_mask
            else:
                chain_2_mask = atom_array_chain_id == row["chain_2_id"]
            chain_1_mask = torch.tensor(chain_1_mask).bool()
            chain_2_mask = torch.tensor(chain_2_mask).bool()
            if chain_1_mask.sum() == 0 or chain_2_mask.sum() == 0:
                return None, None
            return chain_1_mask, chain_2_mask

        df["chain_1_mask"], df["chain_2_mask"] = zip(*df.apply(get_atom_mask, axis=1))
        df = df[df["chain_1_mask"].notna()]  # drop NaN

        if len(df) < 1:
            raise ValueError(
                f"Cannot find a chain/interface for evaluation in the atom_array."
            )

        eval_type = np.array(df["eval_type"].tolist())
        cluster_id = np.array(df["cluster_id"].tolist())
        # [N_eval, N_atom]
        chain_1_mask = torch.stack(df["chain_1_mask"].tolist())
        # [N_eval, N_atom]
        chain_2_mask = torch.stack(df["chain_2_mask"].tolist())

        return eval_type, cluster_id, chain_1_mask, chain_2_mask

    def get_feature_and_label(
        self,
        idx: int,
        token_array: TokenArray,
        atom_array: AtomArray,
        msa_features: dict[str, Any],
        esm_features: dict[str, Any],
        template_features: dict[str, Any],
        full_atom_array: AtomArray,
        is_spatial_crop: bool = True,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """
        Get feature and label information for a given data point.
        It uses a Featurizer object to obtain input features and labels, and applies several
        steps to add other features and labels. Finally, it returns the feature dictionary, label
        dictionary, and a full label dictionary.

        Args:
            idx: Index of the data point.
            token_array: Token array representing the amino acid sequence.
            atom_array: Atom array containing atomic information.
            msa_features: Dictionary of MSA features.
            template_features: Dictionary of template features.
            full_atom_array: Full atom array containing all atoms.
            is_spatial_crop: Flag indicating whether spatial cropping is applied, by default True.

        Returns:
            A tuple containing the feature dictionary and the label dictionary.

        Raises:
            ValueError: If the ligand cannot be found in the data point.
        """
        # Get feature and labels from Featurizer
        feat = Featurizer(
            cropped_token_array=token_array,
            cropped_atom_array=atom_array,
            ref_pos_augment=self.ref_pos_augment,
            lig_atom_rename=self.lig_atom_rename,
        )
        features_dict = feat.get_all_input_features()
        labels_dict = feat.get_labels()

        # Permutation list for atom permutation
        features_dict["atom_perm_list"] = feat.get_atom_permutation_list()

        # Labels for multi-chain permutation
        # Note: the returned full_atom_array may contain fewer atoms than the input
        label_full_dict, full_atom_array = Featurizer.get_gt_full_complex_features(
            atom_array=full_atom_array,
            cropped_atom_array=atom_array,
            get_cropped_asym_only=is_spatial_crop,
        )

        # Masks for Pocket Metrics
        if self.find_pocket:
            # Get entity_id of the interested ligand
            sample_indice = self._get_sample_indice(idx=idx)
            if sample_indice.mol_1_type == "ligand":
                lig_entity_id = str(sample_indice.entity_1_id)
                lig_chain_id = str(sample_indice.chain_1_id)
            elif sample_indice.mol_2_type == "ligand":
                lig_entity_id = str(sample_indice.entity_2_id)
                lig_chain_id = str(sample_indice.chain_2_id)
            else:
                raise ValueError(f"Cannot find ligand from this data point.")
            # Make sure the cropped array contains interested ligand
            assert lig_entity_id in set(atom_array.label_entity_id)
            assert lig_chain_id in set(atom_array.chain_id)

            # Get asym ID of the specific ligand in the `main` pocket
            lig_asym_id = atom_array.label_asym_id[atom_array.chain_id == lig_chain_id]
            assert len(np.unique(lig_asym_id)) == 1
            lig_asym_id = lig_asym_id[0]
            ligands = [lig_asym_id]

            if self.find_all_pockets:
                # Get asym ID of other ligands with the same entity_id
                all_lig_asym_ids = set(
                    full_atom_array[
                        full_atom_array.label_entity_id == lig_entity_id
                    ].label_asym_id
                )
                ligands.extend(list(all_lig_asym_ids - set([lig_asym_id])))

            # Note: the `main` pocket is the 0-indexed one.
            # [N_pocket, N_atom], [N_pocket, N_atom].
            # If not find_all_pockets, then N_pocket = 1.
            interested_ligand_mask, pocket_mask = feat.get_lig_pocket_mask(
                atom_array=full_atom_array, lig_label_asym_id=ligands
            )

            label_full_dict["pocket_mask"] = pocket_mask
            label_full_dict["interested_ligand_mask"] = interested_ligand_mask

        # Masks for Chain/Interface Metrics
        if self.find_eval_chain_interface:
            eval_type, cluster_id, chain_1_mask, chain_2_mask = (
                self._get_eval_chain_interface_mask(
                    idx=idx, atom_array_chain_id=full_atom_array.chain_id
                )
            )
            labels_dict["eval_type"] = eval_type  # [N_eval]
            labels_dict["cluster_id"] = cluster_id  # [N_eval]
            labels_dict["chain_1_mask"] = chain_1_mask  # [N_eval, N_atom]
            labels_dict["chain_2_mask"] = chain_2_mask  # [N_eval, N_atom]

        # Make dummy features for not implemented features
        dummy_feats = []
        if len(msa_features) == 0:
            dummy_feats.append("msa")
        else:
            msa_features = dict_to_tensor(msa_features)
            features_dict.update(msa_features)

        features_dict["esm_embedding_dim"] = self.cfg_esm.embedding_dim
        if len(esm_features) == 0:
            dummy_feats.append("esm")
        else:
            esm_features = dict_to_tensor(esm_features)
            features_dict.update(esm_features)

        if len(template_features) == 0:
            dummy_feats.append("template")
        else:
            template_features = dict_to_tensor(template_features)
            features_dict.update(template_features)

        features_dict = make_dummy_feature(
            features_dict=features_dict, dummy_feats=dummy_feats
        )
        # Transform to right data type
        features_dict = data_type_transform(feat_or_label_dict=features_dict)
        labels_dict = data_type_transform(feat_or_label_dict=labels_dict)

        # Is_distillation
        features_dict["is_distillation"] = torch.tensor([self.is_distillation])
        if self.is_distillation is True:
            features_dict["resolution"] = torch.tensor([-1.0])
        return features_dict, labels_dict, label_full_dict


def get_msa_featurizer(configs, dataset_name: str, stage: str) -> Optional[Callable]:
    """
    Creates and returns an MSAFeaturizer object based on the provided configurations.

    Args:
        configs: A dictionary containing the configurations for the MSAFeaturizer.
        dataset_name: The name of the dataset.
        stage: The stage of the dataset (e.g., 'train', 'test').

    Returns:
        An MSAFeaturizer object if MSA is enabled in the configurations, otherwise None.
    """
    if "msa" in configs["data"] and configs["data"]["msa"]["enable"]:
        msa_info = configs["data"]["msa"]
        msa_args = deepcopy(msa_info)

        if "msa" in (dataset_config := configs["data"][dataset_name]):
            for k, v in dataset_config["msa"].items():
                if k not in ["prot", "rna"]:
                    msa_args[k] = v
                else:
                    for kk, vv in dataset_config["msa"][k].items():
                        msa_args[k][kk] = vv

        prot_msa_args = msa_args["prot"]
        prot_msa_args.update(
            {
                "dataset_name": dataset_name,
                "merge_method": msa_args["merge_method"],
                "max_size": msa_args["max_size"][stage],
            }
        )

        rna_msa_args = msa_args["rna"]
        rna_msa_args.update(
            {
                "dataset_name": dataset_name,
                "merge_method": msa_args["merge_method"],
                "max_size": msa_args["max_size"][stage],
            }
        )

        return MSAFeaturizer(
            prot_msa_args=prot_msa_args,
            rna_msa_args=rna_msa_args,
            enable_rna_msa=configs.data.msa.enable_rna_msa,
        )

    else:
        return None

def get_esm_featurizer(configs, embedding_dir=None, sequence_fpath=None, error_dir=None) -> Optional[Callable]:
    """
    Creates and returns an ESMFeaturizer object based on the provided configurations.

    Args:
        configs: A dictionary containing the configurations for the MSAFeaturizer.
        dataset_name: The name of the dataset.
        stage: The stage of the dataset (e.g., 'train', 'test').

    Returns:
        An MSAFeaturizer object if MSA is enabled in the configurations, otherwise None.
    """
    if configs.data.esm.embedding_dir is None:
        raise ValueError("embedding_dir is not set")
    if embedding_dir is None:
        embedding_dir = configs.data.esm.embedding_dir
    if sequence_fpath is None:
        sequence_fpath = configs.data.esm.train_sequence_fpath
    cfg_esm = ConfigDict({
        "enable": configs.data.esm.enable,
        "model_name": configs.data.esm.model_name,
        "embedding_dir": embedding_dir,
        "embedding_dim": ESM_CONFIG[configs.data.esm.model_name]["emb_dim"],  # 从全局配置获取
        "sequence_fpath": sequence_fpath
    })
    use_esm = configs.data.use_esm
    os.makedirs(cfg_esm.embedding_dir, exist_ok=True)
    os.makedirs(os.path.dirname(cfg_esm.sequence_fpath), exist_ok=True)
    # ESMFeaturizer.precompute_esm_embedding(
    #     self.inputs,
    #     cfg_esm.model_name,
    #     cfg_esm.embedding_dir,
    #     cfg_esm.sequence_fpath,
    # ) # 目前这个只支持inference的格式
    if cfg_esm.get("enable", False):
        return ESMFeaturizer(
            embedding_dir=cfg_esm.embedding_dir,
            sequence_fpath=cfg_esm.sequence_fpath,
            embedding_dim=cfg_esm.embedding_dim,
            error_dir=error_dir,
        )
    else:
        return None
        if use_esm:
            os.makedirs(cfg_esm.embedding_dir, exist_ok=True)
            os.makedirs(os.path.dirname(cfg_esm.sequence_fpath), exist_ok=True)
            ESMFeaturizer.precompute_esm_embedding(
                self.inputs,
                cfg_esm.model_name,
                cfg_esm.embedding_dir,
                cfg_esm.sequence_fpath,
            )
            self.esm_featurizer = ESMFeaturizer(
                embedding_dir=cfg_esm.embedding_dir,
                sequence_fpath=cfg_esm.sequence_fpath,
                embedding_dim=cfg_esm.embedding_dim,
            )

class CachedDataset(Dataset):
    """优先从缓存加载数据的数据集封装器"""
    
    def __init__(self, dataset: BaseSingleDataset):
        """
        初始化缓存数据集
        
        Args:
            dataset: 原始数据集实例
        """
        self.dataset = dataset
        self.cache_dir = dataset.cache_dir
        self.cache_hits = 0
        self.total_requests = 0
        
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, index):
        """
        获取数据项，优先从缓存加载
        
        Args:
            index: 数据索引
            
        Returns:
            处理后的数据样本
        """
        self.total_requests += 1
        sample_indice = self.dataset._get_sample_indice(index)
        sample_name = sample_indice["pdb_id"]
        cache_path = os.path.join(self.cache_dir, f"{sample_name}.pt")
        
        # 尝试从缓存加载
        if os.path.exists(cache_path):
            try:
                cached_data = torch.load(cache_path, weights_only=False)
                self.cache_hits += 1
                hit_rate = self.cache_hits / self.total_requests * 100
                
                logger.info(f"从缓存加载样本 {sample_name}，缓存命中率: {hit_rate:.1f}%")
                    
                # 从缓存恢复数据
                data = cached_data["data"]
                # 如果需要恢复atom_array，可以在这里处理
                return data
                
            except Exception as e:
                logger.warning(f"加载样本 {sample_name} 缓存失败: {e}，将删除损坏的缓存文件并重新处理")
                try:
                    os.remove(cache_path)  # 删除损坏的缓存文件
                    logger.info(f"已删除损坏的缓存文件: {cache_path}")
                except Exception as del_err:
                    logger.warning(f"删除损坏的缓存文件失败: {del_err}")
                        
        # 如果缓存加载失败，调用原始数据集的处理方法
        return self.dataset.__getitem__(index)
    
    def __getattr__(self, name):
        """
        对于未定义的属性，从原始数据集获取
        
        Args:
            name: 属性名称
            
        Returns:
            属性值
        """
        if hasattr(self.dataset, name):
            return getattr(self.dataset, name)
        raise AttributeError(f"'{self.dataset.__class__.__name__}' object has no attribute '{name}'")


class WeightedMultiDataset(Dataset):
    """
    A weighted dataset composed of multiple datasets with weights.
    支持基于CachedDataset封装的数据集。
    """

    def __init__(
        self,
        datasets: list[Dataset],
        dataset_names: list[str],
        datapoint_weights: list[list[float]],
        dataset_sample_weights: list[torch.tensor],
    ):
        """
        Initializes the WeightedMultiDataset.
        Args:
            datasets: A list of Dataset objects (可以是CachedDataset封装过的).
            dataset_names: A list of dataset names corresponding to the datasets.
            datapoint_weights: A list of lists containing sampling weights for each datapoint in the datasets.
            dataset_sample_weights: A list of torch tensors containing sampling weights for each dataset.
        """
        self.datasets = datasets
        self.dataset_names = dataset_names
        self.datapoint_weights = datapoint_weights
        self.dataset_sample_weights = torch.Tensor(dataset_sample_weights)
        self.iteration = 0
        self.offset = 0
        
        # 缓存相关指标
        self.cache_hits = 0
        self.total_requests = 0
        
        self.init_datasets()

    def init_datasets(self):
        """Calculate global weights of each datapoint in datasets for future sampling."""
        self.merged_datapoint_weights = []
        self.weight = 0.0
        self.dataset_indices = []
        self.within_dataset_indices = []
        for dataset_index, (
            dataset,
            datapoint_weight_list,
            dataset_weight,
        ) in enumerate(
            zip(self.datasets, self.datapoint_weights, self.dataset_sample_weights)
        ):
            # normalize each dataset weights
            weight_sum = sum(datapoint_weight_list)
            datapoint_weight_list = [
                dataset_weight * w / weight_sum for w in datapoint_weight_list
            ]
            self.merged_datapoint_weights.extend(datapoint_weight_list)
            self.weight += dataset_weight
            self.dataset_indices.extend([dataset_index] * len(datapoint_weight_list))
            self.within_dataset_indices.extend(list(range(len(datapoint_weight_list))))
        self.merged_datapoint_weights = torch.tensor(
            self.merged_datapoint_weights, dtype=torch.float64
        )

    def __len__(self) -> int:
        return len(self.merged_datapoint_weights)

    def __getitem__(self, index: int) -> dict[str, dict]:
        """获取指定索引的数据，并更新缓存统计"""
        self.total_requests += 1
        dataset_idx = self.dataset_indices[index]
        within_idx = self.within_dataset_indices[index]
        
        dataset = self.datasets[dataset_idx]
        result = dataset[within_idx]
        
        # 更新缓存命中统计
        if hasattr(dataset, 'cache_hits') and hasattr(dataset, 'total_requests'):
            self.cache_hits = sum(ds.cache_hits for ds in self.datasets if hasattr(ds, 'cache_hits'))
            
        return result
        
    def get_cache_stats(self):
        """获取所有数据集的缓存统计信息"""
        total_hits = sum(ds.cache_hits for ds in self.datasets if hasattr(ds, 'cache_hits'))
        total_requests = sum(ds.total_requests for ds in self.datasets if hasattr(ds, 'total_requests'))
        hit_rate = (total_hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "total_hits": total_hits,
            "total_requests": total_requests,
            "hit_rate": hit_rate
        }

def get_weighted_pdb_weight(
    data_type: str,
    cluster_size: int,
    chain_count: dict,
    eps: float = 1e-9,
    beta_dict: Optional[dict] = None,
    alpha_dict: Optional[dict] = None,
) -> float:
    """
    Get sample weight for each example in a weighted PDB dataset.

        data_type (str): Type of data, either 'chain' or 'interface'.
        cluster_size (int): Cluster size of this chain/interface.
        chain_count (dict): Count of each kind of chains, e.g., {"prot": int, "nuc": int, "ligand": int}.
        eps (float, optional): A small epsilon value to avoid division by zero. Default is 1e-9.
        beta_dict (Optional[dict], optional): Dictionary containing beta values for 'chain' and 'interface'.
        alpha_dict (Optional[dict], optional): Dictionary containing alpha values for different chain types.

    Returns:
         float: Calculated weight for the given chain/interface.
    """
    if not beta_dict:
        beta_dict = {
            "chain": 0.5,
            "interface": 1,
        }
    if not alpha_dict:
        alpha_dict = {
            "prot": 3,
            "nuc": 3,
            "ligand": 1,
        }

    assert cluster_size > 0
    assert data_type in ["chain", "interface"]
    beta = beta_dict[data_type]
    assert set(chain_count.keys()).issubset(set(alpha_dict.keys()))
    weight = (
        beta
        * sum(
            [alpha * chain_count[data_mode] for data_mode, alpha in alpha_dict.items()]
        )
        / (cluster_size + eps)
    )
    return weight


def calc_weights_for_df(
    indices_df: pd.DataFrame, beta_dict: dict[str, Any], alpha_dict: dict[str, Any]
) -> pd.DataFrame:
    """
    Calculate weights for each example in the dataframe.

    Args:
        indices_df: A pandas DataFrame containing the indices.
        beta_dict: A dictionary containing beta values for different data types.
        alpha_dict: A dictionary containing alpha values for different data types.

    Returns:
        A pandas DataFrame with an column 'weights' containing the calculated weights.
    """
    # Specific to assembly, and entities (chain or interface)
    indices_df["pdb_sorted_entity_id"] = indices_df.apply(
        lambda x: f"{x['pdb_id']}_{x['assembly_id']}_{'_'.join(sorted([str(x['entity_1_id']), str(x['entity_2_id'])]))}",
        axis=1,
    )

    entity_member_num_dict = {}
    for pdb_sorted_entity_id, sub_df in indices_df.groupby("pdb_sorted_entity_id"):
        # Number of repeatative entities in the same assembly
        entity_member_num_dict[pdb_sorted_entity_id] = len(sub_df)
    indices_df["pdb_sorted_entity_id_member_num"] = indices_df.apply(
        lambda x: entity_member_num_dict[x["pdb_sorted_entity_id"]], axis=1
    )

    cluster_size_record = {}
    for cluster_id, sub_df in indices_df.groupby("cluster_id"):
        cluster_size_record[cluster_id] = len(set(sub_df["pdb_sorted_entity_id"]))

    weights = []
    for _, row in indices_df.iterrows():
        data_type = row["type"]
        cluster_size = cluster_size_record[row["cluster_id"]]
        chain_count = {"prot": 0, "nuc": 0, "ligand": 0}
        for mol_type in [row["mol_1_type"], row["mol_2_type"]]:
            if chain_count.get(mol_type) is None:
                continue
            chain_count[mol_type] += 1
        # Weight specific to (assembly, entity(chain/interface))
        weight = get_weighted_pdb_weight(
            data_type=data_type,
            cluster_size=cluster_size,
            chain_count=chain_count,
            beta_dict=beta_dict,
            alpha_dict=alpha_dict,
        )
        weights.append(weight)
    indices_df["weights"] = weights / indices_df["pdb_sorted_entity_id_member_num"]
    return indices_df


def get_sample_weights(
    sampler_type: str,
    indices_df: pd.DataFrame = None,
    beta_dict: dict = {
        "chain": 0.5,
        "interface": 1,
    },
    alpha_dict: dict = {
        "prot": 3,
        "nuc": 3,
        "ligand": 1,
    },
    force_recompute_weight: bool = False,
) -> Union[pd.Series, list[float]]:
    """
    Computes sample weights based on the specified sampler type.

    Args:
        sampler_type: The type of sampler to use ('weighted' or 'uniform').
        indices_df: A pandas DataFrame containing the indices.
        beta_dict: A dictionary containing beta values for different data types.
        alpha_dict: A dictionary containing alpha values for different data types.
        force_recompute_weight: Whether to force recomputation of weights even if they already exist.

    Returns:
        A list of sample weights.

    Raises:
        ValueError: If an unknown sampler type is provided.
    """
    if sampler_type == "weighted":
        assert indices_df is not None
        if "weights" not in indices_df.columns or force_recompute_weight:
            indices_df = calc_weights_for_df(
                indices_df=indices_df,
                beta_dict=beta_dict,
                alpha_dict=alpha_dict,
            )
        return indices_df["weights"].astype("float32")
    elif sampler_type == "uniform":
        assert indices_df is not None
        return [1 / len(indices_df) for _ in range(len(indices_df))]
    else:
        raise ValueError(f"Unknown sampler type: {sampler_type}")


def get_datasets(
    configs: ConfigDict, error_dir: Optional[str]
) -> tuple[WeightedMultiDataset, dict[str, BaseSingleDataset]]:
    """
    Get training and testing datasets given configs

    Args:
        configs: A ConfigDict containing the dataset configurations.
        error_dir: The directory where error logs will be saved.

    Returns:
        A tuple containing the training dataset and a dictionary of testing datasets.
    """
    if configs.data.esm.embedding_dir is None:
        raise ValueError("embedding_dir is not set")
    cfg_esm = ConfigDict({
        "model_name": configs.data.esm.model_name,
        "embedding_dir": configs.data.esm.embedding_dir,
        "embedding_dim": ESM_CONFIG[configs.data.esm.model_name]["emb_dim"]  # 从全局配置获取
    })
    def _get_dataset_param(config_dict, dataset_name: str, stage: str):
        # Template_featurizer is under development
        # Lig_atom_rename/shuffle_mols/shuffle_sym_ids do not affect the performance very much
        use_cache = config_dict.get("use_cache", False)
        cache_dir = config_dict.get("cache_dir", None)
        if use_cache and cache_dir is None:
            raise ValueError("use_cache is True but cache_dir is not set, please set it for every dataset in dataset_configs in the config_data.py file")
        os.makedirs(cache_dir, exist_ok=True)

        return {
            "name": dataset_name,
            **config_dict["base_info"],
            "cropping_configs": config_dict["cropping_configs"],
            "error_dir": error_dir,
            "msa_featurizer": get_msa_featurizer(configs, dataset_name, stage),
            "esm_featurizer": get_esm_featurizer(configs, error_dir=error_dir, 
                                                embedding_dir=config_dict.get("esm_embedding_dir", None), 
                                                sequence_fpath=config_dict.get("sequence_fpath", None)) if configs.data.esm.enable else None,
            "template_featurizer": None,
            "use_msa": configs.data.use_msa,
            "use_esm": configs.data.use_esm,
            "cfg_esm": cfg_esm,
            "lig_atom_rename": config_dict.get("lig_atom_rename", False),
            "shuffle_mols": config_dict.get("shuffle_mols", False),
            "shuffle_sym_ids": config_dict.get("shuffle_sym_ids", False),
            "use_cache": use_cache,
            "sample_timeout": config_dict.get("sample_timeout", 360),
            "filter_problematic_samples": config_dict.get("filter_problematic_samples", True),
            "cache_dir": cache_dir,
            "return_atom_token_array": config_dict.get("return_atom_token_array", False),
        }

    data_config = configs.data
    logger.info(f"Using train sets {data_config.train_sets}")
    assert len(data_config.train_sets) == len(
        data_config.train_sampler.train_sample_weights
    )
    train_datasets = []
    datapoint_weights = []
    for train_name in data_config.train_sets:
        config_dict = data_config[train_name].to_dict()
        dataset_param = _get_dataset_param(
            config_dict, dataset_name=train_name, stage="train"
        )
        dataset_param["ref_pos_augment"] = data_config.get(
            "train_ref_pos_augment", True
        )
        dataset_param["limits"] = data_config.get("limits", -1)
        train_dataset = BaseSingleDataset(**dataset_param)
        
        if str(configs.data.get("preprocess_distributed", "False")).lower() == "true" and torch.distributed.is_initialized():
            from torch.utils.data import DistributedSampler
            from protenix.utils.distributed import DIST_WRAPPER
            
            # 创建分布式采样器
            sampler = DistributedSampler(
                dataset=train_dataset,
                num_replicas=DIST_WRAPPER.world_size,
                rank=DIST_WRAPPER.rank,
                shuffle=False,
            )
            
            # 预处理样本
            preprocess_samples_distributed(train_dataset, sampler, configs)
            
        cached_dataset = CachedDataset(train_dataset)
        train_datasets.append(cached_dataset)
        
        datapoint_weights.append(
            get_sample_weights(
                **data_config[train_name]["sampler_configs"],
                indices_df=train_dataset.indices_list,
            )
        )
        
    train_dataset = WeightedMultiDataset(
        datasets=train_datasets,
        dataset_names=data_config.train_sets,
        datapoint_weights=datapoint_weights,
        dataset_sample_weights=data_config.train_sampler.train_sample_weights,
    )
    logger.info(f"Successfully created WeightedMultiDataset with {len(train_datasets)} datasets")

    test_datasets = {}
    test_sets = data_config.test_sets
    for test_name in test_sets:
        config_dict = data_config[test_name].to_dict()
        dataset_param = _get_dataset_param(
            config_dict, dataset_name=test_name, stage="test"
        )
        dataset_param["ref_pos_augment"] = data_config.get("test_ref_pos_augment", True)
        
        test_dataset = BaseSingleDataset(**dataset_param)
        
        cached_test_dataset = CachedDataset(test_dataset)
        test_datasets[test_name] = cached_test_dataset
        
        logger.info(f"Successfully created test dataset: {test_name}")
        
    return train_dataset, test_datasets

def preprocess_samples_distributed(dataset, sampler, configs):
    """
    分布式预处理并缓存样本
    
    Args:
        dataset: 数据集实例
        sampler: 分布式采样器
        configs: 配置对象
    """
    from protenix.utils.distributed import DIST_WRAPPER
    from tqdm import tqdm
    import time
    
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
        try:
            sample_indice = dataset._get_sample_indice(idx)
            sample_name = sample_indice["pdb_id"]
            cache_path = os.path.join(dataset.cache_dir, f"{sample_name}.pt")
            
            if os.path.exists(cache_path):
                processed_count += 1
                if i % 10 == 0:
                    logger.info(f"进程 {rank}: 样本 {sample_name} 已缓存 ({processed_count}/{total_samples})")
                continue
                
            logger.info(f"进程 {rank}: 处理样本 {idx}: {sample_name}...")
            start_time = time.time()
            
            # 处理样本并将结果缓存
            data = dataset.__getitem__(idx)
            
            process_duration = time.time() - start_time
            logger.info(f"进程 {rank}: 样本 {sample_name} 处理完成，耗时 {process_duration:.2f} 秒")
            
            processed_count += 1
            if processed_count % 5 == 0:
                logger.info(f"进程 {rank}: 已处理 {processed_count}/{total_samples} 个样本")
                
        except Exception as e:
            logger.error(f"进程 {rank}: 预处理样本 {idx} 时出错: {e}")
    
    # 同步所有进程
    import torch.distributed as dist
    if world_size > 1:
        dist.barrier()
    
    logger.info(f"进程 {rank}: 预处理完成，共处理 {processed_count}/{total_samples} 个样本")
