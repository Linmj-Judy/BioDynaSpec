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

import copy
from collections import defaultdict
from typing import Union, Any, Optional

import numpy as np
import torch
from biotite.structure import Atom, AtomArray, get_residue_starts
from scipy.spatial import cKDTree as KDTree
from rdkit.Chem import GetPeriodicTable

from src.data.basic.constants import STD_RESIDUES, get_all_elems
from src.data.basic.tokenizer import Token, TokenArray
from src.data.basic.utils import get_ligand_polymer_bond_mask
from src.data.basic.lj_parameters import get_lj_parameters_from_atom_name
from src.utils.geometry import angle_3p, random_transform


class Featurizer(object):
    def __init__(
        self,
        cropped_token_array: TokenArray,
        cropped_atom_array: AtomArray,
        cropped_fourier_package: dict[str, Any],
        ref_pos_augment: bool = True,
        lig_atom_rename: bool = False,
    ) -> None:
        """
        Args:
            cropped_token_array (TokenArray): TokenArray object after cropping
            cropped_atom_array (AtomArray): AtomArray object after cropping
            cropped_fourier_package (dict[str, Any]): Fourier package object after cropping
            ref_pos_augment (bool): Boolean indicating whether apply random rotation and translation on ref_pos
            lig_atom_rename (bool): Boolean indicating whether rename atom name for ligand atoms
        """
        self.cropped_token_array = cropped_token_array

        self.cropped_atom_array = cropped_atom_array
        self.cropped_fourier_package = cropped_fourier_package
        self.ref_pos_augment = ref_pos_augment
        self.lig_atom_rename = lig_atom_rename

    @staticmethod
    def encoder(encode_def_list: list[str], input_list: list[str]) -> torch.Tensor:
        """
        Encode a list of input values into a binary format using a specified encoding definition list.

        Args:
            encode_def_list (list): A list of encoding definitions.
            input_list (list): A list of input values to be encoded.

        Returns:
            torch.Tensor: A tensor representing the binary encoding of the input values.
        """
        onehot_dict = {}
        num_keys = len(encode_def_list)
        for index, key in enumerate(encode_def_list):
            onehot = [0] * num_keys
            onehot[index] = 1
            onehot_dict[key] = onehot

        onehot_encoded_data = [onehot_dict[item] for item in input_list]
        onehot_tensor = torch.Tensor(onehot_encoded_data)
        return onehot_tensor

    @staticmethod
    def restype_onehot_encoded(restype_list: list[str]) -> torch.Tensor:
        """
        Ref: AlphaFold3 SI Table 5 "restype"
        One-hot encoding of the sequence. 32 possible values: 20 amino acids + unknown,
        4 RNA nucleotides + unknown, 4 DNA nucleotides + unknown, and gap.
        Ligands represented as “unknown amino acid”.

        Args:
            restype_list (List[str]): A list of residue types.
                                      The residue type of ligand should be "UNK" in the input list.

        Returns:
            torch.Tensor:  A Tensor of one-hot encoded residue types
        """

        return Featurizer.encoder(list(STD_RESIDUES.keys()) + ["-"], restype_list)

    @staticmethod
    def elem_onehot_encoded(elem_list: list[str]) -> torch.Tensor:
        """
        Ref: AlphaFold3 SI Table 5 "ref_element"
        One-hot encoding of the element atomic number for each atom
        in the reference conformer, up to atomic number 128.

        Args:
            elem_list (List[str]): A list of element symbols.

        Returns:
            torch.Tensor:  A Tensor of one-hot encoded elements
        """
        return Featurizer.encoder(get_all_elems(), elem_list)

    @staticmethod
    def ref_atom_name_chars_encoded(atom_names: list[str]) -> torch.Tensor:
        """
        Ref: AlphaFold3 SI Table 5 "ref_atom_name_chars"
        One-hot encoding of the unique atom names in the reference conformer.
        Each character is encoded as ord(c) − 32, and names are padded to length 4.

        Args:
            atom_name_list (List[str]): A list of atom names.

        Returns:
            torch.Tensor:  A Tensor of character encoded atom names
        """
        onehot_dict = {}
        for index, key in enumerate(range(64)):
            onehot = [0] * 64
            onehot[index] = 1
            onehot_dict[key] = onehot
        # [N_atom, 4, 64]
        mol_encode = []
        for atom_name in atom_names:
            # [4, 64]
            atom_encode = []
            for name_str in atom_name.ljust(4):
                atom_encode.append(onehot_dict[ord(name_str) - 32])
            mol_encode.append(atom_encode)
        onehot_tensor = torch.Tensor(mol_encode)
        return onehot_tensor

    @staticmethod
    def masses_from_ref_element(
        ref_element: torch.Tensor,
        *,
        unknown_mass: float = 12.011,     # fallback 质量（默认按碳）
        # out_dtype: torch.dtype = torch.float32,
        # out_device: Optional[torch.device] = None,
        strict_onehot: bool = False       # True 则遇到非 one-hot 行直接报错
    ) -> torch.Tensor:
        """
        根据 AlphaFold3 风格的 ref_element one-hot (N_atom, 128) 自动推导 masses (N_atom,).

        规则:
          - 1..118 使用 RDKit 平均原子量 (GetAtomicWeight)
          - 119..128 (UNK_ELEM) 与异常行使用 unknown_mass
        参数:
          ref_element: (N_atom, 128), dtype=bool/int，one-hot（行向量仅一个1）
          unknown_mass: 未知或非法行的回退质量
          strict_onehot: 若为 True，遇到行和为0或>1时抛出异常；否则回退为 unknown_mass
        返回:
          masses: (N_atom,), out_dtype/out_device
        """
        if ref_element.ndim != 2 or ref_element.shape[-1] != 128:
            raise ValueError(f"ref_element must be of shape (N_atom, 128). Got {tuple(ref_element.shape)}")

        # 行和用于健壮性检查
        row_sum = ref_element.sum(dim=-1)
        if strict_onehot:
            if (row_sum != 1).any():
                bad_idx = torch.nonzero(row_sum != 1, as_tuple=False).squeeze(-1).tolist()
                raise ValueError(f"ref_element rows must be strictly one-hot. Bad rows: {bad_idx[:10]}{'...' if len(bad_idx)>10 else ''}")

        # 计算 one-hot 索引 (0..127)
        idx = ref_element.argmax(dim=-1).to(torch.long)  # 即使不是严格 one-hot，argmax 也能给出某一列

        # 构造原子序号 Z：1..118 对应真实元素；119..128 为 UNK；行和为0 也视为 UNK
        # 对于行和为0的行，设定一个 UNK 的占位 Z=119
        has_any = (row_sum > 0)
        Z = torch.where(has_any, idx + 1, torch.full_like(idx, 119))  # (N_atom,), Z in [1..128]

        # 用 RDKit 构建权重查找表（按 Z 索引）
        try:
            pt = GetPeriodicTable()

            # table[0]占位用不上；table[1..118] 填平均原子量；table[119..128] 用 unknown_mass
            table = [0.0] + [float(pt.GetAtomicWeight(z)) for z in range(1, 119)]
            table += [float(unknown_mass) for _ in range(119, 129)]  # 119..128
            # weight_lut = torch.Tensor(table, dtype=torch.float64)  # 先用高精度
            # 确保 weight_lut 在 ref_element 相同的设备上
            weight_lut = torch.tensor(table, device=ref_element.device, dtype=torch.float32)
        except Exception as e:
            raise ImportError(
                "RDKit is required to compute atomic weights. Please install RDKit "
                "(conda install -c conda-forge rdkit). Original error: " + str(e)
            )

        # 映射得到质量
        masses = weight_lut[Z.clamp_min(1).clamp_max(128)]

        # 非严格 one-hot 下，额外把 Z>118 视为 unknown；严格 one-hot 时已经通过
        unknown_mask = (Z > 118) | (row_sum == 0)
        if unknown_mask.any():
            masses = masses.clone()
            masses[unknown_mask] = float(unknown_mass)
        return masses
        # if out_device is None:
        #     out_device = ref_element.device
        # return masses.to(device=out_device, dtype=out_dtype)

    @staticmethod
    def get_prot_nuc_frame(token: Token, centre_atom: Atom) -> tuple[int, list[int]]:
        """
        Ref: AlphaFold3 SI Chapter 4.3.2
        For proteins/DNA/RNA, we use the three atoms [N, CA, C] / [C1', C3', C4']

        Args:
            token (Token): Token object.
            centre_atom (Atom): Biotite Atom object of Token centre atom.

        Returns:
            has_frame (int): 1 if the token has frame, 0 otherwise.
            frame_atom_index (List[int]): The index of the atoms used to construct the frame.
        """
        if centre_atom.mol_type == "protein":
            # For protein
            abc_atom_name = ["N", "CA", "C"]
        else:
            # For DNA and RNA
            abc_atom_name = [r"C1'", r"C3'", r"C4'"]

        idx_in_atom_indices = []
        for i in abc_atom_name:
            if centre_atom.mol_type == "protein" and "N" not in token.atom_names:
                return 0, [-1, -1, -1]
            elif centre_atom.mol_type != "protein" and "C1'" not in token.atom_names:
                return 0, [-1, -1, -1]
            idx_in_atom_indices.append(token.atom_names.index(i))
        # Protein/DNA/RNA always has frame
        has_frame = 1
        frame_atom_index = [token.atom_indices[i] for i in idx_in_atom_indices]
        return has_frame, frame_atom_index

    @staticmethod
    def get_lig_frame(
        token: Token,
        centre_atom: Atom,
        lig_res_ref_conf_kdtree: dict[str, tuple[KDTree, list[int]]],
        ref_pos: torch.Tensor,
        ref_mask: torch.Tensor,
    ) -> tuple[int, list[int]]:
        """
        Ref: AlphaFold3 SI Chapter 4.3.2
        For ligands, we use the reference conformer of the ligand to construct the frame.

        Args:
            token (Token): Token object.
            centre_atom (Atom): Biotite Atom object of Token centre atom.
            lig_res_ref_conf_kdtree (Dict[str, Tuple[KDTree, List[int]]]): A dictionary of KDTree objects and atom indices.
            ref_pos (torch.Tensor): Atom positions in the reference conformer. Size=[N_atom, 3]
            ref_mask (torch.Tensor): Mask indicating which atom slots are used in the reference conformer. Size=[N_atom]

        Returns:
            tuple[int, List[int]]:
                has_frame (int): 1 if the token has frame, 0 otherwise.
                frame_atom_index (List[int]): The index of the atoms used to construct the frame.
        """
        kdtree, atom_ids = lig_res_ref_conf_kdtree[centre_atom.ref_space_uid]
        b_ref_pos = ref_pos[token.centre_atom_index]
        b_idx = token.centre_atom_index
        if kdtree is None:
            # Atom num < 3
            frame_atom_index = [-1, b_idx, -1]
            has_frame = 0
        else:
            _dist, ind = kdtree.query([b_ref_pos], k=3)
            a_idx, c_idx = atom_ids[ind[0][1]], atom_ids[ind[0][2]]
            frame_atom_index = [a_idx, b_idx, c_idx]

            # Check if reference confomrer vaild
            has_frame = all([ref_mask[idx] for idx in frame_atom_index])

            # Colinear check
            if has_frame:
                theta_degrees = angle_3p(*[ref_pos[idx] for idx in frame_atom_index])
                if theta_degrees <= 25 or theta_degrees >= 155:
                    has_frame = 0
        return has_frame, frame_atom_index

    @staticmethod
    def get_token_frame(
        token_array: TokenArray,
        atom_array: AtomArray,
        ref_pos: torch.Tensor,
        ref_mask: torch.Tensor,
    ) -> TokenArray:
        """
        Ref: AlphaFold3 SI Chapter 4.3.2
        The atoms (a_i, b_i, c_i) used to construct token i’s frame depend on the chain type of i:
        Protein tokens use their residue’s backbone (N, Cα, C),
        while DNA and RNA tokens use (C1′, C3′, C4′) atoms of their residue.
        All other tokens (small molecules, glycans, ions) contain only one atom per token.
        The token atom is assigned to b_i, the closest atom to the token atom is a_i,
        and the second closest atom to the token atom is c_i.
        If this set of three atoms is close to colinear (less than 25 degree deviation),
        or if three atoms do not exist in the chain (e.g. a sodium ion),
        then the frame is marked as invalid.

        Note: frames constucted from reference conformer

        Args:
            token_array (TokenArray): A list of tokens.
            atom_array (AtomArray): An atom array.
            ref_pos (torch.Tensor): Atom positions in the reference conformer. Size=[N_atom, 3]
            ref_mask (torch.Tensor): Mask indicating which atom slots are used in the reference conformer. Size=[N_atom]

        Returns:
            TokenArray: A TokenArray with updated frame annotations.
                        - has_frame: 1 if the token has frame, 0 otherwise.
                        - frame_atom_index: The index of the atoms used to construct the frame.
        """
        token_array_w_frame = copy.deepcopy(token_array)

        # Construct a KDTree for queries to avoid redundant distance calculations
        lig_res_ref_conf_kdtree = {}
        # Ligand and non-standard residues need to use ref to identify frames
        lig_atom_array = atom_array[
            (atom_array.mol_type == "ligand")
            | (~np.isin(atom_array.res_name, list(STD_RESIDUES.keys())))
        ]
        for ref_space_uid in np.unique(lig_atom_array.ref_space_uid):
            # The ref_space_uid is the unique identifier ID for each residue.
            atom_ids = np.where(atom_array.ref_space_uid == ref_space_uid)[0]
            if len(atom_ids) >= 3:
                # # sklearn.neighbors.KDTree
                # kdtree = KDTree(ref_pos[atom_ids], metric="euclidean")
                # scipy.spatial.cKDTree
                kdtree = KDTree(ref_pos[atom_ids])
            else:
                # Invalid frame
                kdtree = None
            lig_res_ref_conf_kdtree[ref_space_uid] = (kdtree, atom_ids)

        has_frame = []
        for token in token_array_w_frame:
            centre_atom = atom_array[token.centre_atom_index]
            if (
                centre_atom.mol_type != "ligand"
                and centre_atom.res_name in STD_RESIDUES
            ):
                has_frame, frame_atom_index = Featurizer.get_prot_nuc_frame(
                    token, centre_atom
                )

            else:
                has_frame, frame_atom_index = Featurizer.get_lig_frame(
                    token, centre_atom, lig_res_ref_conf_kdtree, ref_pos, ref_mask
                )

            token.has_frame = has_frame
            token.frame_atom_index = frame_atom_index
        return token_array_w_frame

    def get_token_features(self) -> dict[str, torch.Tensor]:
        """
        Ref: AlphaFold3 SI Chapter 2.8

        Get token features.
        The size of these features is [N_token].

        Returns:
            Dict[str, torch.Tensor]: A dict of token features.
        """
        token_features = {}

        centre_atoms_indices = self.cropped_token_array.get_annotation(
            "centre_atom_index"
        )
        centre_atoms = self.cropped_atom_array[centre_atoms_indices]

        restype = centre_atoms.cano_seq_resname
        restype_onehot = self.restype_onehot_encoded(restype)

        token_features["token_index"] = torch.arange(0, len(self.cropped_token_array))
        token_features["residue_index"] = torch.Tensor(
            centre_atoms.res_id.astype(int)
        ).long()
        token_features["asym_id"] = torch.Tensor(centre_atoms.asym_id_int).long()
        token_features["entity_id"] = torch.Tensor(centre_atoms.entity_id_int).long()
        token_features["sym_id"] = torch.Tensor(centre_atoms.sym_id_int).long()
        token_features["restype"] = restype_onehot

        return token_features

    def get_chain_perm_features(self) -> dict[str, torch.Tensor]:
        """
        The chain permutation use "entity_mol_id", "mol_id" and "mol_atom_index"
        instead of the "entity_id", "asym_id" and "residue_index".

        The shape of these features is [N_atom].

        Returns:
            Dict[str, torch.Tensor]: A dict of chain permutation features.
        """

        chain_perm_features = {}
        chain_perm_features["mol_id"] = torch.Tensor(
            self.cropped_atom_array.mol_id
        ).long()
        chain_perm_features["mol_atom_index"] = torch.Tensor(
            self.cropped_atom_array.mol_atom_index
        ).long()
        chain_perm_features["entity_mol_id"] = torch.Tensor(
            self.cropped_atom_array.entity_mol_id
        ).long()
        return chain_perm_features

    def get_renamed_atom_names(self) -> np.ndarray:
        """
        Rename the atom names of ligands to avioid information leakage.

        Returns:
            np.ndarray: A numpy array of renamed atom names.
        """
        res_starts = get_residue_starts(
            self.cropped_atom_array, add_exclusive_stop=True
        )
        new_atom_names = copy.deepcopy(self.cropped_atom_array.atom_name)
        for start, stop in zip(res_starts[:-1], res_starts[1:]):
            res_mol_type = self.cropped_atom_array.mol_type[start]
            if res_mol_type != "ligand":
                continue

            elem_count = defaultdict(int)
            new_res_atom_names = []
            for elem in self.cropped_atom_array.element[start:stop]:
                elem_count[elem] += 1
                new_res_atom_names.append(f"{elem.upper()}{elem_count[elem]}")
            new_atom_names[start:stop] = new_res_atom_names
        return new_atom_names

    def get_reference_features(self) -> dict[str, torch.Tensor]:
        """
        Ref: AlphaFold3 SI Chapter 2.8

        Get reference features.
        The size of these features is [N_atom].

        Returns:
            Dict[str, torch.Tensor]: a dict of reference features.
        """
        ref_pos = []
        for ref_space_uid in np.unique(self.cropped_atom_array.ref_space_uid):
            res_ref_pos = random_transform(
                self.cropped_atom_array.ref_pos[
                    self.cropped_atom_array.ref_space_uid == ref_space_uid,
                ],
                apply_augmentation=self.ref_pos_augment,
                centralize=True,
            )
            ref_pos.append(res_ref_pos)
        ref_pos = np.concatenate(ref_pos)

        ref_features = {}
        ref_features["ref_pos"] = torch.Tensor(ref_pos)
        ref_features["ref_mask"] = torch.Tensor(self.cropped_atom_array.ref_mask).long()
        ref_features["ref_element"] = Featurizer.elem_onehot_encoded(
            self.cropped_atom_array.element
        ).long()
        ref_features["ref_masses"] = Featurizer.masses_from_ref_element(
            ref_features["ref_element"],
            unknown_mass=12.011,      # TODO: 也可以改成 0.0 或 14.007 等
        )
        ref_features["ref_charge"] = torch.Tensor(
            self.cropped_atom_array.ref_charge
        ).long()
        
        if self.lig_atom_rename:
            atom_names = self.get_renamed_atom_names()
        else:
            atom_names = self.cropped_atom_array.atom_name
        
        # ===== Add LJ parameters =====
        lj_sigma, lj_epsilon = get_lj_parameters_from_atom_name(atom_names)
        ref_features["lj_sigma"] = torch.Tensor(lj_sigma)
        ref_features["lj_epsilon"] = torch.Tensor(lj_epsilon)
        # =============================
                
        ref_features["ref_atom_name_chars"] = Featurizer.ref_atom_name_chars_encoded(
            atom_names
        ).long()
        ref_features["ref_space_uid"] = torch.Tensor(
            self.cropped_atom_array.ref_space_uid
        ).long()

        token_array_with_frame = self.get_token_frame(
            token_array=self.cropped_token_array,
            atom_array=self.cropped_atom_array,
            ref_pos=ref_features["ref_pos"],
            ref_mask=ref_features["ref_mask"],
        )
        ref_features["has_frame"] = torch.Tensor(
            token_array_with_frame.get_annotation("has_frame")
        ).long()  # [N_token]
        ref_features["frame_atom_index"] = torch.Tensor(
            token_array_with_frame.get_annotation("frame_atom_index")
        ).long()  # [N_token, 3]
        return ref_features

    def get_bond_features(self) -> dict[str, torch.Tensor]:
        """
        Ref: AlphaFold3 SI Chapter 2.8
        A 2D matrix indicating if there is a bond between any atom in token i and token j,
        restricted to just polymer-ligand and ligand-ligand bonds and bonds less than 2.4 Å during training.
        The size of bond feature is [N_token, N_token].

        Returns:
            Dict[str, torch.Tensor]: A dict of bond features.
        """
        bond_features = {}
        num_tokens = len(self.cropped_token_array)
        adj_matrix = self.cropped_atom_array.bonds.adjacency_matrix().astype(int)

        token_adj_matrix = np.zeros((num_tokens, num_tokens), dtype=int)
        atom_bond_mask = adj_matrix > 0

        for i in range(num_tokens):
            atoms_i = self.cropped_token_array[i].atom_indices
            token_i_mol_type = self.cropped_atom_array.mol_type[atoms_i[0]]
            token_i_res_name = self.cropped_atom_array.res_name[atoms_i[0]]
            token_i_ref_space_uid = self.cropped_atom_array.ref_space_uid[atoms_i[0]]
            unstd_res_token_i = (
                token_i_res_name not in STD_RESIDUES and token_i_mol_type != "ligand"
            )
            is_polymer_i = token_i_mol_type in ["protein", "dna", "rna"]

            for j in range(i + 1, num_tokens):
                atoms_j = self.cropped_token_array[j].atom_indices
                token_j_mol_type = self.cropped_atom_array.mol_type[atoms_j[0]]
                token_j_res_name = self.cropped_atom_array.res_name[atoms_j[0]]
                token_j_ref_space_uid = self.cropped_atom_array.ref_space_uid[
                    atoms_j[0]
                ]
                unstd_res_token_j = (
                    token_j_res_name not in STD_RESIDUES
                    and token_j_mol_type != "ligand"
                )
                is_polymer_j = token_j_mol_type in ["protein", "dna", "rna"]

                # The polymer-polymer (std-std, std-unstd, and inter-unstd) bond will not be included in token_bonds.
                if is_polymer_i and is_polymer_j:
                    is_same_res = token_i_ref_space_uid == token_j_ref_space_uid
                    unstd_res_bonds = unstd_res_token_i and unstd_res_token_j
                    if not (is_same_res and unstd_res_bonds):
                        continue

                sub_matrix = atom_bond_mask[np.ix_(atoms_i, atoms_j)]
                if np.any(sub_matrix):
                    token_adj_matrix[i, j] = 1
                    token_adj_matrix[j, i] = 1
        bond_features["token_bonds"] = torch.Tensor(token_adj_matrix)
        return bond_features

    def get_extra_features(self) -> dict[str, torch.Tensor]:
        """
        Get other features not listed in AlphaFold3 SI Chapter 2.8 Table 5.
        The size of these features is [N_atom].

        Returns:
            Dict[str, torch.Tensor]: a dict of extra features.
        """
        atom_to_token_idx_dict = {}
        for idx, token in enumerate(self.cropped_token_array.tokens):
            for atom_idx in token.atom_indices:
                atom_to_token_idx_dict[atom_idx] = idx

        # Ensure the order of the atom_to_token_idx is the same as the atom_array
        atom_to_token_idx = [
            atom_to_token_idx_dict[atom_idx]
            for atom_idx in range(len(self.cropped_atom_array))
        ]

        extra_features = {}
        extra_features["atom_to_token_idx"] = torch.Tensor(atom_to_token_idx).long()
        extra_features["atom_to_tokatom_idx"] = torch.Tensor(
            self.cropped_atom_array.tokatom_idx
        ).long()

        extra_features["is_protein"] = torch.Tensor(
            self.cropped_atom_array.is_protein
        ).long()
        extra_features["is_ligand"] = torch.Tensor(
            self.cropped_atom_array.is_ligand
        ).long()
        extra_features["is_dna"] = torch.Tensor(self.cropped_atom_array.is_dna).long()
        extra_features["is_rna"] = torch.Tensor(self.cropped_atom_array.is_rna).long()
        if "resolution" in self.cropped_atom_array._annot:
            extra_features["resolution"] = torch.Tensor(
                [self.cropped_atom_array.resolution[0]]
            )
        else:
            extra_features["resolution"] = torch.Tensor([-1])
        return extra_features

    @staticmethod
    def get_lig_pocket_mask(
        atom_array: AtomArray, lig_label_asym_id: Union[str, list]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Ref: AlphaFold3 Chapter Methods.Metrics

        the pocket is defined as all heavy atoms within 10 Å of any heavy atom of the ligand,
        restricted to the primary polymer chain for the ligand or modified residue being scored,
        and further restricted to only backbone atoms for proteins. The primary polymer chain is defined variously:
        for PoseBusters it is the protein chain with the most atoms within 10 Å of the ligand,
        for bonded ligand scores it is the bonded polymer chain and for modified residues it
        is the chain that the residue is contained in (minus that residue).

        Args:
            atom_array (AtomArray): atoms in the complex.
            lig_label_asym_id (Union[str, List]): The label_asym_id of the ligand of interest.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple of ligand pocket mask and pocket mask.
        """

        if isinstance(lig_label_asym_id, str):
            lig_label_asym_ids = [lig_label_asym_id]
        else:
            lig_label_asym_ids = list(lig_label_asym_id)

        # Get backbone mask
        prot_backbone = (
            atom_array.is_protein & np.isin(atom_array.atom_name, ["C", "N", "CA"])
        ).astype(bool)

        kdtree = KDTree(atom_array.coord)

        ligand_mask_list = []
        pocket_mask_list = []
        for lig_label_asym_id in lig_label_asym_ids:
            assert np.isin(
                lig_label_asym_id, atom_array.label_asym_id
            ), f"{lig_label_asym_id} is not in the label_asym_id of the cropped atom array."

            ligand_mask = atom_array.label_asym_id == lig_label_asym_id
            lig_pos = atom_array.coord[ligand_mask]

            # Get atoms in 10 Angstrom radius
            near_atom_indices = np.unique(
                # np.concatenate(kdtree.query_radius(lig_pos, 10.0)) # sklearn.neighbors.KDTree
                np.concatenate(kdtree.query_ball_point(lig_pos, 10.0)) # scipy.spatial.cKDTree
            )
            near_atoms = [
                True if i in near_atom_indices else False
                for i in range(len(atom_array))
            ]

            # Get primary chain (protein backone in 10 Angstrom radius)
            primary_chain_candidates = near_atoms & prot_backbone
            primary_chain_candidates_atoms = atom_array[primary_chain_candidates]

            max_atom = 0
            primary_chain_asym_id_int = None
            for asym_id_int in np.unique(primary_chain_candidates_atoms.asym_id_int):
                n_atoms = np.sum(
                    primary_chain_candidates_atoms.asym_id_int == asym_id_int
                )
                if n_atoms > max_atom:
                    max_atom = n_atoms
                    primary_chain_asym_id_int = asym_id_int
            assert (
                primary_chain_asym_id_int is not None
            ), f"No primary chain found for ligand ({lig_label_asym_id=})."

            pocket_mask = primary_chain_candidates & (
                atom_array.asym_id_int == primary_chain_asym_id_int
            )
            ligand_mask_list.append(ligand_mask)
            pocket_mask_list.append(pocket_mask)

        ligand_mask_by_pockets = torch.Tensor(
            np.array(ligand_mask_list).astype(int)
        ).long()
        pocket_mask_by_pockets = torch.Tensor(
            np.array(pocket_mask_list).astype(int)
        ).long()
        return ligand_mask_by_pockets, pocket_mask_by_pockets

    def get_mask_features(self) -> dict[str, torch.Tensor]:
        """
        Generate mask features for the cropped atom array.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing various mask features.
        """
        mask_features = {}

        mask_features["pae_rep_atom_mask"] = torch.Tensor(
            self.cropped_atom_array.centre_atom_mask
        ).long()

        mask_features["plddt_m_rep_atom_mask"] = torch.Tensor(
            self.cropped_atom_array.plddt_m_rep_atom_mask
        ).long()  # [N_atom]

        mask_features["distogram_rep_atom_mask"] = torch.Tensor(
            self.cropped_atom_array.distogram_rep_atom_mask
        ).long()  # [N_atom]

        mask_features["modified_res_mask"] = torch.Tensor(
            self.cropped_atom_array.modified_res_mask
        ).long()

        lig_polymer_bonds = get_ligand_polymer_bond_mask(self.cropped_atom_array)
        num_atoms = len(self.cropped_atom_array)
        bond_mask_mat = np.zeros((num_atoms, num_atoms))
        for i, j, _ in lig_polymer_bonds:
            bond_mask_mat[i, j] = 1
            bond_mask_mat[j, i] = 1
        mask_features["bond_mask"] = torch.Tensor(
            bond_mask_mat
        ).long()  # [N_atom, N_atom]
        return mask_features

    def get_all_input_features(self):
        """
        Get input features from cropped data.

        Returns:
            Dict[str, torch.Tensor]: a dict of features.
        """
        features = {}
        token_features = self.get_token_features()
        features.update(token_features)

        bond_features = self.get_bond_features()
        features.update(bond_features)

        reference_features = self.get_reference_features()
        features.update(reference_features)

        extra_features = self.get_extra_features()
        features.update(extra_features)

        chain_perm_features = self.get_chain_perm_features()
        features.update(chain_perm_features)

        mask_features = self.get_mask_features()
        features.update(mask_features)
        return features

    def get_labels(self) -> dict[str, torch.Tensor]:
        """
        Get the input labels required for the training phase.

        Returns:
            Dict[str, torch.Tensor]: a dict of labels.
        """

        labels = {}

        labels["coordinate"] = torch.Tensor(
            self.cropped_atom_array.coord
        )  # [N_atom, 3]

        labels["coordinate_mask"] = torch.Tensor(
            self.cropped_atom_array.is_resolved.astype(int)
        ).long()  # [N_atom]

        # Ensure all torch.tensor values in cropped_fourier_package have the same dtype as other tensors (coordinate).
        fourier_package = copy.deepcopy(self.cropped_fourier_package)
        # Use dtype from the coordinate label
        target_dtype = torch.Tensor(self.cropped_atom_array.coord).dtype
        if fourier_package is not None:
            for k, v in fourier_package.items():
                if isinstance(v, torch.Tensor) and v.dtype != target_dtype:
                    fourier_package[k] = v.to(dtype=target_dtype)
        labels["fourier_package"] = fourier_package
        
        return labels

    def get_atom_permutation_list(
        self,
    ) -> list[list[int]]:
        """
        Generate info of permutations.

        Returns:
            List[List[int]]: a list of atom permutations.
        """
        atom_perm_list = []
        for i in self.cropped_atom_array.res_perm:
            # Decode list[str] -> list[list[int]]
            atom_perm_list.append([int(j) for j in i.split("_")])

        # Atoms connected to different residue are fixed.
        # Bonds array: [[atom_idx_i, atom_idx_j, bond_type]]
        idx_i = self.cropped_atom_array.bonds._bonds[:, 0]
        idx_j = self.cropped_atom_array.bonds._bonds[:, 1]
        diff_mask = (
            self.cropped_atom_array.ref_space_uid[idx_i]
            != self.cropped_atom_array.ref_space_uid[idx_j]
        )
        inter_residue_bonds = self.cropped_atom_array.bonds._bonds[diff_mask]
        fixed_atom_mask = np.isin(
            np.arange(len(self.cropped_atom_array)),
            np.unique(inter_residue_bonds[:, :2]),
        )

        # Get fixed atom permutation for each residue.
        fixed_atom_perm_list = []
        res_starts = get_residue_starts(
            self.cropped_atom_array, add_exclusive_stop=True
        )
        for r_start, r_stop in zip(res_starts[:-1], res_starts[1:]):
            atom_res_perm = np.array(
                atom_perm_list[r_start:r_stop]
            )  # [N_res_atoms, N_res_perm]
            res_fixed_atom_mask = fixed_atom_mask[r_start:r_stop]

            if np.sum(res_fixed_atom_mask) == 0:
                # If all atoms in the residue are not fixed, e.g. ions
                fixed_atom_perm_list.extend(atom_res_perm.tolist())
                continue

            # Create a [N_res_atoms, N_res_perm] template of indices
            n_res_atoms, n_perm = atom_res_perm.shape
            indices_template = (
                atom_res_perm[:, 0].reshape(n_res_atoms, 1).repeat(n_perm, axis=1)
            )

            # Identify the column where the positions of the fixed atoms remain unchanged
            fixed_atom_perm = atom_res_perm[
                res_fixed_atom_mask
            ]  # [N_fixed_res_atoms, N_res_perm]
            fixed_indices_template = indices_template[
                res_fixed_atom_mask
            ]  # [N_fixed_res_atoms, N_res_perm]
            unchanged_columns_mask = np.all(
                fixed_atom_perm == fixed_indices_template, axis=0
            )

            # Remove the columns related to the position changes of fixed atoms.
            fiedx_atom_res_perm = atom_res_perm[:, unchanged_columns_mask]
            fixed_atom_perm_list.extend(fiedx_atom_res_perm.tolist())
        return fixed_atom_perm_list

    @staticmethod
    def get_gt_full_complex_features(
        atom_array: AtomArray,
        cropped_atom_array: AtomArray = None,
        get_cropped_asym_only: bool = True,
    ) -> tuple[dict[str, torch.Tensor], AtomArray, np.ndarray]:
        """Get full ground truth complex features.
        It is used for multi-chain permutation alignment.

        Args:
            atom_array (AtomArray): all atoms in the complex.
            cropped_atom_array (AtomArray, optional): cropped atoms. Defaults to None.
            get_cropped_asym_only (bool, optional): Defaults to True.
                - If true, a chain is returned only if its asym_id (mol_id) appears in the
                cropped_atom_array. It should be a favored setting for the spatial cropping.
                - If false, a chain is returned if its entity_id (entity_mol_id) appears in
                the cropped_atom_array.

        Returns:
            Tuple[
                Dict[str, torch.Tensor],  # features
                AtomArray,                # filtered atom array
                np.ndarray,               # indices selected from the original atom_array
            ]
        """
        gt_features = {}

        if cropped_atom_array is not None:
            # Get the cropped part of gt entities
            entity_atom_set = set(
                zip(
                    cropped_atom_array.entity_mol_id,
                    cropped_atom_array.mol_atom_index,
                )
            )
            mask = np.array(
                [
                    (entity, atom) in entity_atom_set
                    for (entity, atom) in zip(
                        atom_array.entity_mol_id, atom_array.mol_atom_index
                    )
                ],
                dtype=bool,
            )

            if get_cropped_asym_only:
                # Restrict to asym chains appeared in cropped_atom_array
                asyms = np.unique(cropped_atom_array.mol_id)
                mask = mask & np.isin(atom_array.mol_id, asyms)

            selected_atom_indices = np.nonzero(mask)[0]
            atom_array = atom_array[mask]
        else:
            selected_atom_indices = np.arange(len(atom_array))

        gt_features["coordinate"] = torch.Tensor(atom_array.coord)
        gt_features["coordinate_mask"] = torch.Tensor(atom_array.is_resolved).long()
        gt_features["entity_mol_id"] = torch.Tensor(atom_array.entity_mol_id).long()
        gt_features["mol_id"] = torch.Tensor(atom_array.mol_id).long()
        gt_features["mol_atom_index"] = torch.Tensor(atom_array.mol_atom_index).long()
        gt_features["pae_rep_atom_mask"] = torch.Tensor(
            atom_array.centre_atom_mask
        ).long()
        return gt_features, atom_array, selected_atom_indices

    @staticmethod
    def get_gt_full_fourier_package(
        fourier_package: dict[str, Any],
        selected_atom_indices: np.ndarray,
    ) -> Optional[dict[str, Any]]:
        """
        Filter the full fourier package to keep the atoms corresponding to the
        selected indices (consistent with the filtered atom array).

        Args:
            fourier_package: The original fourier package before cropping.
            selected_atom_indices: Indices selected from the original atom array.

        Returns:
            A filtered fourier package dictionary, or None if not applicable.
        """
        if fourier_package is None or selected_atom_indices is None:
            return None

        if len(selected_atom_indices) == 0:
            return None

        pack = copy.deepcopy(fourier_package)
        indices_np = np.asarray(selected_atom_indices, dtype=int)
        indices_torch = torch.from_numpy(indices_np)

        atom_dimension_keys = [
            "X_positive_real",
            "X_positive_imag",
            "V_positive_real",
            "V_positive_imag",
            "A_positive_real",
            "A_positive_imag",
            "X_real",
            "X_imag",
            "V_real",
            "V_imag",
            "A_real",
            "A_imag",
            "Xw_real",
            "Xw_imag",
            "X_positive",
            "V_positive",
            "A_positive",
            "X",
            "V",
            "A",
            "X_windowed",
            "psd_two_sided",
            "first_frame_coords",
            "coords_t",
        ]

        for key in atom_dimension_keys:
            if key not in pack:
                continue

            value = pack[key]

            if isinstance(value, torch.Tensor):
                if value.ndim >= 1:
                    pack[key] = value.index_select(
                        0, indices_torch.to(value.device, non_blocking=False)
                    )
            elif isinstance(value, np.ndarray):
                pack[key] = value[indices_np]

        return pack
