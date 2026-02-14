import os
import traceback

import pandas as pd
import torch

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ESMFeaturizer:

    def __init__(
        self,
        embedding_dir: str,
        sequence_fpath: str,
        embedding_dim: int = 1028,
        error_dir: str = None,
    ):
        self.embedding_dir = embedding_dir
        self.sequence_fpath = sequence_fpath
        self.seq_to_filename = self.get_seq_to_filename(sequence_fpath)
        self.embedding_dim = embedding_dim
        self.error_dir = error_dir
        if self.error_dir is not None:
            self.error_dir = os.path.join(self.error_dir, "esm_error")
            os.makedirs(self.error_dir, exist_ok=True)

    def get_seq_to_filename(self, sequence_fpath: str) -> dict[str, str]:
        df = pd.read_csv(sequence_fpath)
        df["filename"] = (
            df["part_id"].astype(str) + "/" + df["seq_label"].astype(str) + ".pt"
        )
        return df.set_index("seq")["filename"].to_dict()

    def load_esm_embedding(self, sequence: str):
        try:
            if sequence in self.seq_to_filename:
                x = torch.load(os.path.join(self.embedding_dir, self.seq_to_filename[sequence]), weights_only=False)
                assert x.size(0) == len(sequence)
                return x
            else:
                print(f"警告: 找不到序列 '{sequence}' 的ESM嵌入，将使用零矩阵代替")
                return torch.zeros((len(sequence), self.embedding_dim))
        except Exception as e:
            logger.warning(f"[{sequence}] ESM error: {e}")
            return None

    def save_error(self, error_sequences, pdb_id):
        if (self.error_dir is None) or (len(error_sequences) == 0):
            return
        for error_data in error_sequences:
            fpath = os.path.join(
                self.error_dir, f"{pdb_id}_{error_data['entity_id']}.txt"
            )
            if os.path.exists(fpath):
                continue
            with open(fpath, "w") as f:
                f.write(error_data["error"])

    def __call__(self, token_array, atom_array, bioassembly_dict, inference_mode=False):

        # init as zeros
        N_token = len(token_array)
        x = torch.zeros([N_token, self.embedding_dim])

        # get one atom per token
        centre_atoms_indices = token_array.get_annotation("centre_atom_index")
        centre_atom_array = atom_array[centre_atoms_indices]

        # protein entities
        is_protein = centre_atom_array.chain_mol_type == "protein"
        protein_entity_ids = set(centre_atom_array.label_entity_id[is_protein])

        if inference_mode:
            entity_id_to_sequence = (
                {}
            )  # Only contains protein entity, many-to-one mapping
            for i, entity_info_wrapper in enumerate(bioassembly_dict["sequences"]):
                entity_id = str(i + 1)
                entity_type = list(entity_info_wrapper.keys())[0]
                entity_info = entity_info_wrapper[entity_type]
                if entity_type == "proteinChain":
                    entity_id_to_sequence[entity_id] = entity_info["sequence"]

        # enumerate over the entities
        error_sequences = []
        for entity_id in protein_entity_ids:
            try:
                # Get sequence
                if inference_mode:
                    sequence = entity_id_to_sequence.get(entity_id)
                    if sequence is None:
                        logger.warning(f"Entity ID {entity_id} not found in entity_id_to_sequence, using empty sequence")
                        sequence = ""
                        error_message = f"Entity ID {entity_id} not found in entity_id_to_sequence:\n{traceback.format_exc()}"
                        error_sequences.append(
                            {
                                "entity_id": entity_id,
                                "error": error_message,
                            }
                        )
                        continue
                else:
                    # 检查sequences键是否存在
                    if "sequences" not in bioassembly_dict:
                        logger.warning(f"'sequences' key not found in bioassembly_dict for entity ID {entity_id}, using empty sequence")
                        sequence = ""
                        error_message = f"'sequences' key not found in bioassembly_dict:\n{traceback.format_exc()}"
                        error_sequences.append(
                            {
                                "entity_id": entity_id,
                                "error": error_message,
                            }
                        )
                        continue
                    
                    # 检查entity_id是否在sequences字典中
                    if str(entity_id) not in bioassembly_dict["sequences"]:
                        logger.warning(f"Entity ID {entity_id} not found in bioassembly_dict['sequences'], using empty sequence")
                        sequence = ""
                        error_message = f"Entity ID {entity_id} not found in bioassembly_dict['sequences']:\n{traceback.format_exc()}"
                        error_sequences.append(
                            {
                                "entity_id": entity_id,
                                "error": error_message,
                            }
                        )
                        continue
                    
                    sequence = bioassembly_dict["sequences"][str(entity_id)]
                    
                # 确保有序列
                if not sequence:
                    raise ValueError(f"Empty sequence for entity ID {entity_id}")
                    
                x_esm = self.load_esm_embedding(sequence)
                # 处理x_esm为None的情况
                if x_esm is None:
                    logger.warning(f"x_esm为None, 使用零矩阵替代序列 '{sequence}' 的ESM嵌入")
                    x_esm = torch.zeros((len(sequence), self.embedding_dim))
                
                # Get residue indices of the cropped tokens
                entity_mask = centre_atom_array.label_entity_id == entity_id
                res_index = (
                    centre_atom_array.res_id[entity_mask] - 1
                )  # res_id starts with 1
                # Get esm embeddding according to residue indices
                x[entity_mask] = x_esm[res_index]
            except Exception as e:
                error_message = f"{e}:\n{traceback.format_exc()}"
                error_sequences.append(
                    {
                        "entity_id": entity_id,
                        "error": error_message,
                    }
                )
                # 获取标识符时优先使用当前模式下的正确键
                identifier = bioassembly_dict.get('name' if inference_mode else 'pdb_id', 'unknown')
                logger.warning(
                    f"[{identifier}] ESM error: {error_message}"
                )

        # 使用当前模式下正确的键
        id_key = "name" if inference_mode else "pdb_id"
        self.save_error(error_sequences, pdb_id=bioassembly_dict.get(id_key, "unknown"))

        # 返回字典格式而不是直接返回张量
        esm_features = {
            "esm_token_embedding": x,
            "esm_coverage": torch.tensor([1.0 if len(protein_entity_ids) > 0 else 0.0]),
            "esm_error_count": torch.tensor([len(error_sequences)]),
        }
        
        return esm_features
# TODO: 待开发，开发一个可用于traing阶段的esm_embedding计算函数
    # def compute_esm_embedding(self, inputs: list, model_name, embedding_dir, sequence_fpath):
    #     df_seq = pd.read_csv(sequence_fpath)

    #     model, alphabet = load_esm_model(model_name)
    #     error_parts = []
    #     part_counts = dict(df_seq["part_id"].value_counts())
    #     for part_id, count in part_counts.items():
    #         df_part = df_seq[df_seq["part_id"] == part_id]
    #         print(f"Part {part_id}: {len(df_part)} sequences.")
    #         labels = df_part["seq_label"].tolist()
    #         sequences = df_part["seq"].tolist()
    #         try:
    #             save_dir = os.path.join(embedding_dir, part_id)
    #             if not os.path.exists(save_dir):
    #                 os.makedirs(save_dir)
    #             lm_embeddings = compute_ESM_embeddings(
    #                 model_name,
    #                 model,
    #                 alphabet,
    #                 labels,
    #                 sequences,
    #                 save_dir,
    #                 truncation_seq_length=4094,
    #                 toks_per_batch=16384,
    #             )
    #             print(
    #                 f"[{part_id}] Processed {len(lm_embeddings)} sequences in total. Done!"
    #             )
    #         except Exception as e:
    #             print(f"[{part_id}] {e}")
    #             error_parts.append(part_id)
    #     print("Error parts: ", error_parts)

    @staticmethod
    def precompute_esm_embedding(
        inputs: list, model_name, embedding_dir, sequence_fpath
    ):
        print("Precompute ESM embeddings")
        from src.data.basic.compute_esm import compute_ESM_embeddings, load_esm_model
        # prepare seq_label
        all_seq_dict = []
        for sample_dict in inputs:
            sample_name = sample_dict["name"]
            for i, entity_info_wrapper in enumerate(sample_dict["sequences"]):
                pdb_entity_id = sample_name + "_" + str(i + 1)
                entity_type = list(entity_info_wrapper.keys())[0]
                entity_info = entity_info_wrapper[entity_type]
                if entity_type == "proteinChain":
                    all_seq_dict.append(
                        {
                            "seq": entity_info["sequence"],
                            "pdb_entity_id": pdb_entity_id,
                            "seq_label": pdb_entity_id,
                            "part_id": pdb_entity_id[1:3],
                        }
                    )
        df_seq = pd.DataFrame(
            all_seq_dict, columns=["seq", "pdb_entity_id", "seq_label", "part_id"]
        )
        df_seq.to_csv(sequence_fpath)
        print(f"Save sequence file to {sequence_fpath}")

        model, alphabet = load_esm_model(model_name)
        error_parts = []
        part_counts = dict(df_seq["part_id"].value_counts())
        for part_id, count in part_counts.items():
            df_part = df_seq[df_seq["part_id"] == part_id]
            print(f"Part {part_id}: {len(df_part)} sequences.")
            labels = df_part["seq_label"].tolist()
            sequences = df_part["seq"].tolist()
            try:
                save_dir = os.path.join(embedding_dir, part_id)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                lm_embeddings = compute_ESM_embeddings(
                    model_name,
                    model,
                    alphabet,
                    labels,
                    sequences,
                    save_dir,
                    truncation_seq_length=4094,
                    toks_per_batch=16384,
                )
                print(
                    f"[{part_id}] Processed {len(lm_embeddings)} sequences in total. Done!"
                )
            except Exception as e:
                print(f"[{part_id}] {e}")
                error_parts.append(part_id)
        print("Error parts: ", error_parts)