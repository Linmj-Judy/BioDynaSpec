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

import logging
import os
import time
import traceback
import warnings
import signal
import threading
import pickle
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional, Union, Callable, Dict
import sys
import biotite.structure.io as strucio
import numpy as np
import pandas as pd
import torch
from biotite.structure import AtomArray
from src.data.basic.featurizer import Featurizer
from src.data.basic.msa_featurizer import MSAFeaturizer
from src.data.basic.esm_featurizer import ESMFeaturizer
from src.data.basic.parser import DistillationMMCIFParser, MMCIFParser, PDBParser
from src.data.basic.tokenizer import AtomArrayTokenizer, TokenArray
from src.data.basic.cropping import CropData
from src.utils.file_io import load_gzip_pickle
from src.data.fourier.utils import TrajectoryFFTProcessor, TrajectoryFFTConfig, parse_time_str_to_seconds

if sys.platform != "darwin":
    torch.multiprocessing.set_sharing_strategy("file_descriptor")
else:
    torch.multiprocessing.set_sharing_strategy("file_system")
    
logger = logging.getLogger(__name__)

class TimeoutError(Exception):
    """处理样本超时异常"""
    pass


def timeout_handler(signum, frame):
    """信号处理函数，用于处理超时情况"""
    raise TimeoutError("处理样本超时")


def process_with_timeout(func, args=None, kwargs=None, timeout=360):  
    """
    使用超时机制执行函数
    
    Args:
        func: 要执行的函数
        args: 函数参数
        kwargs: 函数关键字参数
        timeout: 超时时间（秒）
    
    Returns:
        函数执行结果
    
    Raises:
        TimeoutError: 当函数执行超时
    """
    args = args or ()
    kwargs = kwargs or {}
    result = [None]  
    error = [None]  
    
    def worker():  
        try:  
            result[0] = func(*args, **kwargs)  
        except Exception as e:  
            error[0] = e  

    thread = threading.Thread(target=worker)  
    thread.daemon = True
    thread.start()  
    thread.join(timeout)  

    if thread.is_alive():
        logging.warning("处理线程超时，但无法强制终止。设置为守护线程并继续")
        raise TimeoutError("处理样本超时")  
    
    if error[0]:  
        raise error[0]
        
    return result[0]


class DataPipeline(object):
    """
    DataPipeline class provides static methods to handle various data processing tasks related to bioassembly structures.
    """
    @staticmethod
    def get_data_from_mmcif(
        mmcif: Union[str, Path],
        pdb_cluster_file: Union[str, Path, None] = None,
        dataset: str = "WeightedPDB",
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """
        Get raw data from mmcif with tokenizer and a list of chains and interfaces for sampling.

        Args:
            mmcif (Union[str, Path]): The raw mmcif file.
            pdb_cluster_file (Union[str, Path, None], optional): Cluster info txt file. Defaults to None.
            dataset (str, optional): The dataset type, either "WeightedPDB" or "Distillation". Defaults to "WeightedPDB".

        Returns:
            tuple[list[dict[str, Any]], dict[str, Any]]:
                sample_indices_list (list[dict[str, Any]]): The sample indices list (each one is a chain or an interface).
                bioassembly_dict (dict[str, Any]): The bioassembly dict with sequence, atom_array, and token_array.
        """
        try:
            if dataset == "WeightedPDB":
                parser = MMCIFParser(mmcif_file=mmcif)
                bioassembly_dict = parser.get_bioassembly()
            elif dataset == "Distillation":
                parser = DistillationMMCIFParser(mmcif_file=mmcif)
                bioassembly_dict = parser.get_structure_dict()
            else:
                raise NotImplementedError(
                    'Unsupported "dataset", please input either "WeightedPDB" or "Distillation".'
                )

            sample_indices_list = parser.make_indices(
                bioassembly_dict=bioassembly_dict, pdb_cluster_file=pdb_cluster_file
            )
            if len(sample_indices_list) == 0:
                # empty indices and AtomArray
                return [], bioassembly_dict

            atom_array = bioassembly_dict["atom_array"]
            atom_array.set_annotation(
                "resolution", [parser.resolution] * len(atom_array)
            )

            tokenizer = AtomArrayTokenizer(atom_array)
            token_array = tokenizer.get_token_array()
            bioassembly_dict["msa_features"] = None
            bioassembly_dict["template_features"] = None

            bioassembly_dict["token_array"] = token_array
            return sample_indices_list, bioassembly_dict

        except Exception as e:
            logging.warning("Gen data failed for %s due to %s", mmcif, e)
            return [], {}

    @staticmethod
    def get_data_from_pdb(
        pdb: Union[str, Path],
        pdb_cluster_file: Union[str, Path, None] = None,
        apply_filters: bool = True,
    ) -> tuple[list[dict[str, Any]], dict[str, Any], Any]:
        """
        Get raw data from PDB file with tokenizer and a list of chains and interfaces for sampling.
        Equivalent to get_data_from_mmcif but for PDB format files.

        Args:
            pdb (Union[str, Path]): The raw PDB file.
            pdb_cluster_file (Union[str, Path, None], optional): Cluster info txt file. Defaults to None.
            apply_filters (bool, optional): Whether to apply filtering operations in PDBParser.
                                          If False, all atoms are kept. Defaults to True.

        Returns:
            tuple[list[dict[str, Any]], dict[str, Any], Any]:
                sample_indices_list (list[dict[str, Any]]): The sample indices (each one is a chain or an interface).
                bioassembly_dict (dict[str, Any]): The bioassembly dict with sequence, atom_array, and token_array.
                                                   If apply_filters=True, includes 'filter_indices' for trajectory filtering.
                info_df (DataFrame or None): PDB sequences info dataframe for downstream usage.
        """
        try:
            parser = PDBParser(pdb_file=pdb)
            bioassembly_dict = parser.get_bioassembly(apply_filters=apply_filters)

            # 检查 atom_array 是否为 None（可能由于链数过多或为0）
            atom_array = bioassembly_dict.get("atom_array")
            if atom_array is None:
                pdb_id = bioassembly_dict.get("pdb_id", "unknown")
                num_chains = bioassembly_dict.get("num_assembly_polymer_chains", "unknown")
                logging.warning(f"atom_array is None for {pdb} (PDB Code: {pdb_id}, num_chains: {num_chains})")
                logging.warning("This usually happens when num_assembly_polymer_chains is 0 or exceeds max_assembly_chains")
                return [], bioassembly_dict, None

            sample_indices_list = parser.make_indices(
                bioassembly_dict=bioassembly_dict, pdb_cluster_file=pdb_cluster_file
            )
            if len(sample_indices_list) == 0:
                # empty indices but AtomArray exists
                return [], bioassembly_dict, None

            # 现在可以安全地使用 atom_array
            atom_array.set_annotation(
                "resolution", [parser.resolution] * len(atom_array)
            )

            tokenizer = AtomArrayTokenizer(atom_array)
            token_array = tokenizer.get_token_array()
            bioassembly_dict["msa_features"] = None
            bioassembly_dict["template_features"] = None

            bioassembly_dict["token_array"] = token_array
            pdb_id, info_df = parser.get_seqs_from_atom_array(atom_array)
            return sample_indices_list, bioassembly_dict, info_df

        except SystemExit:
            # 不要捕获 SystemExit
            raise
        except KeyboardInterrupt:
            # 不要捕获 KeyboardInterrupt
            raise
        except Exception as e:
            import traceback
            error_type = type(e).__name__
            logging.warning("Gen data failed for %s due to %s: %s", pdb, error_type, e)
            # 对于段错误等严重错误，记录但不重新抛出，避免中断整个流程
            if "Segmentation fault" in str(e) or "segmentation" in str(e).lower():
                logging.error("Segmentation fault detected, this may be due to corrupted PDB file or biotite library bug")
                traceback.print_exc()
                return [], {}, pd.DataFrame(columns=["pdb_id", "entity_id", "mol_type", "pdbx_type", "length", "seq", "diff_seq_mmcif_vs_atom_site", "has_alt_res", "diff_alt_res_seq_vs_atom_site", "pdbx_description", "auth_asym_id", "release_date", "release_date_retrace_obsolete"])
            else:
                traceback.print_exc()
                return [], {}, pd.DataFrame(columns=["pdb_id", "entity_id", "mol_type", "pdbx_type", "length", "seq", "diff_seq_mmcif_vs_atom_site", "has_alt_res", "diff_alt_res_seq_vs_atom_site", "pdbx_description", "auth_asym_id", "release_date", "release_date_retrace_obsolete"])

    @staticmethod
    def init_trajectory_fft_processor(
        store_dtype: torch.dtype = torch.float32,
        device: str = "cpu",
        remove_com_for_alignment: bool = False,
        align_rotation_for_alignment: bool = False,
        window_size: int = None,
        use_sliding_window: bool = True,
        fpath: Union[str, Path] = "trajectory_fft_processor.pth",
    ) -> TrajectoryFFTProcessor:
        """
        Initialize the trajectory FFT processor.
        """
        cfg = TrajectoryFFTConfig(
            norm="ortho", use_double=True,
            remove_com=remove_com_for_alignment, align_rotation=align_rotation_for_alignment, detrend_order=None,  # 不在预处理中做，在对齐中做
            window="hamming", window_size=window_size, use_sliding_window=use_sliding_window,
            device=device, store_dtype=store_dtype, store_complex_as_float2=True
        )
        processor = TrajectoryFFTProcessor(cfg)
        processor.save(fpath)
        return processor

    @staticmethod
    def load_trajectory_fft_processor(
        fpath: Union[str, Path],
    ) -> TrajectoryFFTProcessor:
        """
        Load the trajectory FFT processor from file.
        If a GPU is available, set its device to 'cuda'.
        """
        # NOTE:
        # - TrajectoryFFTProcessor.load(fpath) already returns a TrajectoryFFTProcessor instance,
        #   not a TrajectoryFFTConfig.
        # - Earlier versions mistakenly treated the returned object as a config and wrapped it
        #   again via TrajectoryFFTProcessor(cfg), which caused self.config to become a
        #   TrajectoryFFTProcessor instead of TrajectoryFFTConfig. This led to confusing
        #   AttributeError like:
        #       'TrajectoryFFTProcessor' object has no attribute 'norm'
        #   when accessing self.config.norm inside TrajectoryFFTProcessor methods.
        processor = TrajectoryFFTProcessor.load(fpath)

        # Check if torch.cuda is available and update the underlying config.device
        if torch.cuda.is_available():
            processor.config.device = "cuda"
        else:
            processor.config.device = "cpu"

        return processor

    @staticmethod
    def get_trajectory_fft(
        traj_coord_path: str,
        processor: TrajectoryFFTProcessor,
        bioassembly_dict: dict[str, Any],
        remove_com_for_alignment: bool = False,
        align_rotation_for_alignment: bool = False,
        dt_str: str = "10ps",
    ) -> Dict[str, Any]:
        """
        处理轨迹并提取频谱，可选进行对齐和分析。
        
        Args:
            traj_coord_path: 轨迹文件路径，numpy array shape (T, N, 3)
            remove_com_for_alignment: 是否去除质心用于对齐
            align_rotation_for_alignment: 是否旋转对齐用于对齐
            dt_str: 时间步长字符串，如"10ps"
        
        Returns:
            Dict[str, Any]: 包含所有结果的字典
        """
        # Check if bioassembly_dict is valid
        if not bioassembly_dict:
            raise ValueError(f"Invalid bioassembly_dict: bioassembly_dict is empty or None. {bioassembly_dict}")
        if "atom_array" not in bioassembly_dict:
            raise ValueError(f"Invalid bioassembly_dict: atom_array key is missing. {bioassembly_dict.keys()}")
        if bioassembly_dict["atom_array"] is None:
            raise ValueError(f"Invalid bioassembly_dict: atom_array is None. {bioassembly_dict['atom_array']}")
        
        arr = np.load(traj_coord_path)  # (T, N, 3) numpy数组
        traj = torch.from_numpy(arr).to(processor.config.store_dtype)
        
        # Check if filters were applied and filter_indices are available
        filter_indices = bioassembly_dict.get("filter_indices", None)
        apply_filters = bioassembly_dict.get("apply_filters", True)
        
        # Get trajectory dimensions
        T, N_traj, _ = traj.shape
        
        # Get reference structure atom count
        atom_array = bioassembly_dict["atom_array"]
        N_ref = len(atom_array)
        
        # Apply same filtering to trajectory if filters were applied
        if apply_filters and filter_indices is not None:
            # Filter trajectory to match filtered structure
            if N_traj != len(filter_indices):
                # Check if trajectory has more atoms (likely includes all atoms from PDB)
                if N_traj >= len(filter_indices):
                    # Apply filter_indices to trajectory
                    traj = traj[:, filter_indices, :]
                    logging.info(
                        f"应用过滤索引到轨迹: 原始轨迹有 {N_traj} 个原子，"
                        f"过滤后轨迹有 {len(filter_indices)} 个原子，"
                        f"与参考结构原子数 {N_ref} 匹配。"
                    )
                else:
                    raise ValueError(
                        f"轨迹原子数 ({N_traj}) 小于过滤索引数 ({len(filter_indices)})。"
                        f"这通常意味着轨迹已经被部分过滤，或者过滤索引不正确。"
                    )
            elif N_traj != N_ref:
                raise ValueError(
                    f"轨迹原子数 ({N_traj}) 与参考结构原子数 ({N_ref}) 不匹配，"
                    f"但过滤索引数 ({len(filter_indices)}) 也不匹配。"
                    f"这可能是数据不一致的问题。"
                )
        elif not apply_filters:
            # No filters applied, trajectory should match structure exactly
            if N_traj != N_ref:
                raise ValueError(
                    f"未应用过滤，但轨迹原子数 ({N_traj}) 与参考结构原子数 ({N_ref}) 不匹配。"
                    f"请确保轨迹和结构文件一致。"
                )
        else:
            # Filters were applied but filter_indices not available (backward compatibility)
            if N_traj != N_ref:
                raise ValueError(
                    f"轨迹原子数 ({N_traj}) 与参考结构原子数 ({N_ref}) 不匹配，"
                    f"且无法获取过滤索引。请使用 apply_filters=True 重新生成 bioassembly_dict。"
                )
        
        # Now traj should match atom_array in atom count
        ref_element = Featurizer.elem_onehot_encoded(
            atom_array.element
        ).long()
        ref_masses = Featurizer.masses_from_ref_element(
            ref_element,
            unknown_mass=12.011,  # TODO: 也可以改成 0.0 或 14.007 等
        )

        dt = parse_time_str_to_seconds(dt_str) # seconds

        pack = processor.process_and_package(
            positions=traj,
            dt=dt,
            masses=ref_masses,
            box_lengths=None,
            missing_mask=None,
            layout_hint="TN3",
            remove_com=remove_com_for_alignment,
            align_rotation=align_rotation_for_alignment,
            save_minimal=True,  # 只保存必要数据以节省空间和显存
            save_psd=True,  # 保存PSD
        )
        return pack

    @staticmethod
    def get_trajectory_ifft(
        pack: Dict[str, Any],
        processor: TrajectoryFFTProcessor,
        reconstruct_from: str = "X", # "X", "V", "A"
    ) -> torch.Tensor:
        """
        Get trajectory IFFT from pack.

        Args:
            pack: Dict[str, Any]: The pack from processor.process_and_package.
            processor: TrajectoryFFTProcessor: The trajectory FFT processor.
            reconstruct_from: str: The reconstruct from, either "X", "V", or "A".

        Returns:
            torch.Tensor: The trajectory IFFT.
        """
        return processor.reconstruct_and_unpack(pack, reconstruct_from=reconstruct_from)


    @staticmethod
    def get_label_entity_id_to_asym_id_int(atom_array: AtomArray) -> dict[str, int]:
        """
        Get a dictionary that associates each label_entity_id with its corresponding asym_id_int.

        Args:
            atom_array (AtomArray): AtomArray object

        Returns:
            dict[str, int]: label_entity_id to its asym_id_int
        """
        entity_to_asym_id = defaultdict(set)
        for atom in atom_array:
            entity_id = atom.label_entity_id
            entity_to_asym_id[entity_id].add(atom.asym_id_int)
        return entity_to_asym_id

    @staticmethod
    def get_data_bioassembly(
        bioassembly_dict_fpath: Union[str, Path],
    ) -> dict[str, Any]:
        """
        Get the bioassembly dict.

        Args:
            bioassembly_dict_fpath (Union[str, Path]): The path to the bioassembly dictionary file.

        Returns:
            dict[str, Any]: The bioassembly dict with sequence, atom_array and token_array.

        Raises:
            AssertionError: If the bioassembly dictionary file does not exist.
        """
        assert os.path.exists(
            bioassembly_dict_fpath
        ), f"File not exists {bioassembly_dict_fpath}"
        bioassembly_dict = load_gzip_pickle(bioassembly_dict_fpath)

        return bioassembly_dict

    @staticmethod
    def get_data_fourier(
        fourier_fpath: Union[str, Path],
    ) -> dict[str, Any]:
        """
        Get the fourier data.
        
        Args:
            fourier_fpath: Path to the fourier data file.
            
        Returns:
            dict: The fourier data dictionary.
            
        Raises:
            FileNotFoundError: If the file does not exist.
            EOFError: If the file is corrupted or incomplete.
            pickle.UnpicklingError: If the file cannot be unpickled.
        """
        if not os.path.exists(fourier_fpath):
            raise FileNotFoundError(f"Fourier data file not found: {fourier_fpath}")
        
        try:
            fourier_data = load_gzip_pickle(fourier_fpath)
            return fourier_data
        except (EOFError, pickle.UnpicklingError) as e:
            logger.error(f"Failed to load fourier data from {fourier_fpath}: {e}")
            logger.error(f"This file appears to be corrupted. "
                        f"Please regenerate it or remove it to skip this sample.")
            raise

    @staticmethod
    def _map_ref_chain(
        one_sample: pd.Series, bioassembly_dict: dict[str, Any]
    ) -> list[int]:
        """
        Map the chain or interface chain_x_id to the reference chain asym_id.

        Args:
            one_sample (pd.Series): A dict of one chain or interface from indices list.
            bioassembly_dict (dict[str, Any]): The bioassembly dict with sequence, atom_array and token_array.

        Returns:
            list[int]: A list of asym_id_lnt of the chosen chain or interface, length 1 or 2.
        """
        atom_array = bioassembly_dict["atom_array"]
        ref_chain_indices = []
        for chain_id_field in ["chain_1_id", "chain_2_id"]:
            chain_id = one_sample[chain_id_field]
            assert np.isin(
                chain_id, np.unique(atom_array.chain_id)
            ), f"PDB {bioassembly_dict['pdb_id']} {chain_id_field}:{chain_id} not in atom_array"
            chain_asym_id = atom_array[atom_array.chain_id == chain_id].asym_id_int[0]
            ref_chain_indices.append(chain_asym_id)
            if one_sample["type"] == "chain":
                break
        return ref_chain_indices

    @staticmethod
    def get_msa_raw_features(
        bioassembly_dict: dict[str, Any],
        selected_indices: np.ndarray,
        msa_featurizer: Optional[MSAFeaturizer],
    ) -> Optional[dict[str, np.ndarray]]:
        """
        Get tokenized MSA features of the bioassembly

        Args:
            bioassembly_dict (Mapping[str, Any]): The bioassembly dict with sequence, atom_array and token_array.
            selected_indices (torch.Tensor): Cropped token indices.
            msa_featurizer (MSAFeaturizer): MSAFeaturizer instance.

        Returns:
            Optional[dict[str, np.ndarray]]: The tokenized MSA features of the bioassembly.
        """
        if msa_featurizer is None:
            return None

        entity_to_asym_id_int = dict(
            DataPipeline.get_label_entity_id_to_asym_id_int(
                bioassembly_dict["atom_array"]
            )
        )

        msa_feats = msa_featurizer(
            bioassembly_dict=bioassembly_dict,
            selected_indices=selected_indices,
            entity_to_asym_id_int=entity_to_asym_id_int,
        )

        return msa_feats

    @staticmethod
    def get_esm_raw_features(
        bioassembly_dict: dict[str, Any],
        esm_featurizer: Optional[ESMFeaturizer],
    ) -> Optional[dict[str, np.ndarray]]:
        if esm_featurizer is None:
            return None
        x_esm = esm_featurizer(
            token_array=bioassembly_dict["token_array"],
            atom_array=bioassembly_dict["atom_array"],
            bioassembly_dict=bioassembly_dict,
            inference_mode=False,
        )
        return x_esm
    

    @staticmethod
    def get_template_raw_features(
        bioassembly_dict: dict[str, Any],
        selected_indices: np.ndarray,
        template_featurizer: None,
    ) -> Optional[dict[str, np.ndarray]]:
        """
        Get tokenized template features of the bioassembly.

        Args:
            bioassembly_dict (dict[str, Any]): The bioassembly dict with sequence, atom_array and token_array.
            selected_indices (np.ndarray): Cropped token indices.
            template_featurizer (None): Placeholder for the template featurizer.

        Returns:
            Optional[dict[str, np.ndarray]]: The tokenized template features of the bioassembly,
                or None if the template featurizer is not provided.
        """
        if template_featurizer is None:
            return None

        entity_to_asym_id_int = dict(
            DataPipeline.get_label_entity_id_to_asym_id_int(
                bioassembly_dict["atom_array"]
            )
        )

        template_feats = template_featurizer(
            bioassembly_dict=bioassembly_dict,
            selected_indices=selected_indices,
            entity_to_asym_id_int=entity_to_asym_id_int,
        )
        return template_feats

    @staticmethod
    def crop(
        one_sample: pd.Series,
        bioassembly_dict: dict[str, Any],
        fourier_package: dict[str, Any],
        crop_size: int,
        msa_featurizer: Optional[MSAFeaturizer],
        esm_featurizer: Optional[ESMFeaturizer],
        template_featurizer: None,
        method_weights: list[float] = [0.2, 0.4, 0.4],
        contiguous_crop_complete_lig: bool = False,
        spatial_crop_complete_lig: bool = False,
        drop_last: bool = False,
        remove_metal: bool = False,
        use_msa: bool = False,
        use_esm: bool = False,
        cfg_esm: dict[str, Any] = {},
    ) -> tuple[str, TokenArray, AtomArray, dict[str, Any], dict[str, Any]]:
        """
        Crop data based on the crop size and reference chain indices.

        Args:
            one_sample (pd.Series): A dict of one chain or interface from indices list.
            bioassembly_dict (dict[str, Any]): A dict of bioassembly dict with sequence, atom_array and token_array.
            crop_size (int): the crop size.
            msa_featurizer (MSAFeaturizer): Default to an empty replacement for msa featurizer.
            template_featurizer (None): Placeholder for the template featurizer.
            method_weights (list[float]): The weights corresponding to these three cropping methods:
                                          ["ContiguousCropping", "SpatialCropping", "SpatialInterfaceCropping"].
            contiguous_crop_complete_lig (bool): Whether to crop the complete ligand in ContiguousCropping method.
            spatial_crop_complete_lig (bool): Whether to crop the complete ligand in SpatialCropping method.
            drop_last (bool): Whether to drop the last fragment in ContiguousCropping.
            remove_metal (bool): Whether to remove metal atoms from the crop.

        Returns:
            tuple[str, TokenArray, AtomArray, dict[str, Any], dict[str, Any]]:
                crop_method (str): The crop method.
                cropped_token_array (TokenArray): TokenArray after cropping.
                cropped_atom_array (AtomArray): AtomArray after cropping.
                cropped_msa_features (dict[str, Any]): The cropped msa features.
                cropped_template_features (dict[str, Any]): The cropped template features.
        """
        if crop_size <= 0:
            selected_indices = None
            if use_msa:
                # Prepare msa
                msa_features = DataPipeline.get_msa_raw_features(
                    bioassembly_dict=bioassembly_dict,
                    selected_indices=selected_indices,
                    msa_featurizer=msa_featurizer,
                )
            else:
                msa_features = {}
            # ESM features
            if use_esm:
                esm_features = DataPipeline.get_esm_raw_features(
                    bioassembly_dict=bioassembly_dict,
                    esm_featurizer=esm_featurizer,
                )
            else:
                esm_features = {}

            # features_dict["esm_token_embedding"] = x_esm
            # Prepare template
            template_features = DataPipeline.get_template_raw_features(
                bioassembly_dict=bioassembly_dict,
                selected_indices=selected_indices,
                template_featurizer=template_featurizer,
            )
            return (
                "no_crop",
                bioassembly_dict["token_array"],
                bioassembly_dict["atom_array"],
                msa_features or {},
                esm_features or {},
                template_features or {},
                -1,
            )

        ref_chain_indices = DataPipeline._map_ref_chain(
            one_sample=one_sample, bioassembly_dict=bioassembly_dict
        )

        crop = CropData(
            crop_size=crop_size,
            ref_chain_indices=ref_chain_indices,
            token_array=bioassembly_dict["token_array"],
            atom_array=bioassembly_dict["atom_array"],
            method_weights=method_weights,
            contiguous_crop_complete_lig=contiguous_crop_complete_lig,
            spatial_crop_complete_lig=spatial_crop_complete_lig,
            drop_last=drop_last,
            remove_metal=remove_metal,
        )
        # Get crop method
        crop_method = crop.random_crop_method()
        # Get crop indices based crop method
        selected_indices, reference_token_index = crop.get_crop_indices(
            crop_method=crop_method
        )
        if use_msa:
            # Prepare msa
            msa_features = DataPipeline.get_msa_raw_features(
                bioassembly_dict=bioassembly_dict,
                selected_indices=selected_indices,
                msa_featurizer=msa_featurizer,
            )
        else:
            msa_features = {}
        # ESM features
        if use_esm:
            esm_features = DataPipeline.get_esm_raw_features(
                bioassembly_dict=bioassembly_dict,
                esm_featurizer=esm_featurizer,
            )
        else:
            esm_features = {}
        # Prepare template
        template_features = DataPipeline.get_template_raw_features(
            bioassembly_dict=bioassembly_dict,
            selected_indices=selected_indices,
            template_featurizer=template_featurizer,
        )

        (
            cropped_token_array,
            cropped_atom_array,
            cropped_msa_features,
            cropped_esm_features,
            cropped_template_features,
            cropped_atom_indices,
        ) = crop.crop_by_indices(
            selected_token_indices=selected_indices,
            msa_features=msa_features,
            esm_features=esm_features,
            template_features=template_features,
        )

        if crop_method == "ContiguousCropping":
            resovled_atom_num = cropped_atom_array.is_resolved.sum()
            # The criterion of "more than 4 atoms" is chosen arbitrarily.
            assert (
                resovled_atom_num > 4
            ), f"resovled_atom_num={resovled_atom_num} <= 4 after ContiguousCropping"

        if fourier_package is not None:
            cropped_fourier_package = DataPipeline._crop_fourier_package(
                fourier_package=fourier_package,
                cropped_atom_indices=cropped_atom_indices,
            )
        else:
            cropped_fourier_package = None

        return (
            crop_method,
            cropped_fourier_package,
            cropped_token_array,
            cropped_atom_array,
            cropped_msa_features,
            cropped_esm_features,
            cropped_template_features,
            reference_token_index,
        )

    @staticmethod
    def _crop_fourier_package(
        fourier_package: dict[str, Any],
        cropped_atom_indices: list[int],
    ) -> dict[str, Any]:
        """
        Crop fourier package data based on cropped atom indices.
        
        Args:
            fourier_package: The fourier package dictionary containing spectrum data.
            cropped_atom_indices: List of atom indices to keep (indices in original atom array).
        
        Returns:
            Cropped fourier package dictionary.
        """
        if not cropped_atom_indices:
            return fourier_package
        
        cropped_package = fourier_package.copy()
        cropped_atom_indices_array = np.array(cropped_atom_indices)
        
        # Convert to torch tensor if needed for indexing
        if isinstance(cropped_atom_indices_array, np.ndarray):
            cropped_atom_indices_torch = torch.from_numpy(cropped_atom_indices_array)
        else:
            cropped_atom_indices_torch = cropped_atom_indices_array
        
        # List of keys that need cropping along the first dimension (atom dimension)
        # These have shape (N, 3, ...) or (N, ...) where N is the number of atoms
        atom_dimension_keys = [
            # Positive frequency half spectrum (real/imag parts)
            "X_positive_real", "X_positive_imag",
            "V_positive_real", "V_positive_imag",
            "A_positive_real", "A_positive_imag",
            # Full spectrum (real/imag parts)
            "X_real", "X_imag",
            "V_real", "V_imag",
            "A_real", "A_imag",
            # Windowed spectrum
            "Xw_real", "Xw_imag",
            # Complex tensors (if store_complex_as_float2=False)
            "X_positive", "V_positive", "A_positive",
            "X", "V", "A", "X_windowed",
            # Metrics
            "psd_two_sided",
            # Coordinates
            "first_frame_coords",  # (N, 3)
            "coords_t",  # (N, 3, T)
        ]
        
        for key in atom_dimension_keys:
            if key in cropped_package:
                value = cropped_package[key]
                # Convert to tensor if it's numpy array
                if isinstance(value, np.ndarray):
                    value = torch.from_numpy(value)
                
                if isinstance(value, torch.Tensor):
                    # Crop along first dimension (atom dimension)
                    if value.ndim >= 1:
                        cropped_package[key] = value[cropped_atom_indices_torch]
                    else:
                        # Keep as is if it's a scalar
                        cropped_package[key] = value
                elif isinstance(value, np.ndarray):
                    # Handle numpy arrays
                    cropped_package[key] = value[cropped_atom_indices_array]
        
        # Log the cropping operation
        original_atom_count = None
        if "first_frame_coords" in fourier_package:
            original_value = fourier_package["first_frame_coords"]
            if isinstance(original_value, (torch.Tensor, np.ndarray)):
                original_atom_count = original_value.shape[0]
        
        cropped_atom_count = len(cropped_atom_indices)
        if original_atom_count is not None:
            logger.info(
                f"裁剪Fourier package: 原始原子数={original_atom_count}, "
                f"裁剪后原子数={cropped_atom_count}"
            )
        
        return cropped_package

    @staticmethod
    def save_atoms_to_cif(
        output_cif_file: str, atom_array: AtomArray, include_bonds: bool = False
    ) -> None:
        """
        Save atom array data to a CIF file.

        Args:
            output_cif_file (str): The output path for saving atom array in cif
            atom_array (AtomArray): The atom array to be saved
            include_bonds (bool): Whether to include bond information in the CIF file. Default is False.

        """
        strucio.save_structure(
            file_path=output_cif_file,
            array=atom_array,
            data_block=os.path.basename(output_cif_file).replace(".cif", ""),
            include_bonds=include_bonds,
        )
    
    @staticmethod
    def cache_feature(cache_dir: str, sample_name: str, data, atom_array: AtomArray = None, token_array: TokenArray = None):
        """
        缓存特征数据到磁盘
        
        Args:
            cache_dir: 缓存目录
            sample_name: 样本名称
            data: 要缓存的数据
            atom_array: 原子数组数据
        """
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f"{sample_name}.pt")
        
        save_data = {"data": data, "atom_array_exists": atom_array is not None, "token_array_exists": token_array is not None}
        if atom_array is not None:
            try:
                save_data["atom_array_pickle"] = pickle.dumps(atom_array)
                logging.info(f"成功序列化保存原子数组")
            except Exception as e:
                logging.warning(f"序列化保存原子数组失败: {e}")
        if token_array is not None:
            try:
                save_data["token_array_pickle"] = pickle.dumps(token_array)
                logging.info(f"成功序列化保存token数组")
            except Exception as e:
                logging.warning(f"序列化保存token数组失败: {e}")
        
        torch.save(save_data, cache_path)
        logging.info(f"样本 {sample_name} 已缓存")
        
    @staticmethod
    def load_cached_feature(cache_dir: str, sample_name: str):
        """
        从缓存加载特征数据
        
        Args:
            cache_dir: 缓存目录
            sample_name: 样本名称
            
        Returns:
            tuple: (data, atom_array, success_flag)
        """
        cache_path = os.path.join(cache_dir, f"{sample_name}.pt")
        
        if not os.path.exists(cache_path):
            return None, None, False
            
        try:
            cached_data = torch.load(cache_path, weights_only=False)
            data = cached_data["data"]
            
            atom_array = None
            if cached_data["atom_array_exists"]:
                if "atom_array_pickle" in cached_data:
                    atom_array = pickle.loads(cached_data["atom_array_pickle"])
                else:
                    logging.warning(f"样本 {sample_name} 缓存中缺少原子数组")

            token_array = None
            if cached_data["token_array_exists"]:
                if "token_array_pickle" in cached_data:
                    token_array = pickle.loads(cached_data["token_array_pickle"])
                else:
                    logging.warning(f"样本 {sample_name} 缓存中缺少token数组")
                    
            if atom_array is None or token_array is None:
                flag = False
            else:
                flag = True
            
            return data, atom_array, token_array, flag
            
        except Exception as e:
            logging.warning(f"加载样本 {sample_name} 缓存失败: {e}")
            return None, None, None, False
            
    @staticmethod
    def log_problem_sample(problem_samples_log: str, sample_name: str, error_message: str):
        """
        记录问题样本
        
        Args:
            problem_samples_log: 问题样本日志文件路径
            sample_name: 样本名称
            error_message: 错误信息
        """
        try:
            # 读取现有问题样本记录
            if os.path.exists(problem_samples_log):
                with open(problem_samples_log, "r") as f:
                    log_data = json.load(f)
            else:
                log_data = {"problem_samples": {}}
                
            # 添加新的问题样本信息
            if sample_name not in log_data["problem_samples"]:
                log_data["problem_samples"][sample_name] = []
            
            log_data["problem_samples"][sample_name].append({
                "error": error_message,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            })
            
            with open(problem_samples_log, "w") as f:
                json.dump(log_data, f, indent=2)
        except json.JSONDecodeError as jde:
            logging.error(f"记录问题样本失败: 可能是 {problem_samples_log} 已损坏或格式不正确: {jde}")
            # 可以考虑重置日志文件
            try:
                log_data = {"problem_samples": {}}
                log_data["problem_samples"][sample_name] = [{
                    "error": error_message + "（首次写入，原日志损坏）",
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }]
                with open(problem_samples_log, "w") as f:
                    json.dump(log_data, f, indent=2)
                logging.warning(f"已重置损坏的日志文件: {problem_samples_log}")
            except Exception as inner_e:
                logging.error(f"重置日志文件仍失败: {inner_e}")
        except Exception as e:
            logging.error(f"记录问题样本失败: {e}")
