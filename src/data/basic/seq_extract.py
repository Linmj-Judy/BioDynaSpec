import os
import pandas as pd
from src.data.basic.constants import mmcif_restype_3to1


current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)

def read_ccd_code_mapping(ccd_code_mapping_file: str):
    df = pd.read_csv(ccd_code_mapping_file)
    chem_codes_dict = dict(zip(df["three_letter_code"], df["one_letter_code"]))
    return chem_codes_dict


ptm_mapping_dict = read_ccd_code_mapping(
    os.path.join(
        os.path.dirname(os.path.dirname(current_directory)),
        "configs",
        "ccd_cache",
        "chem_ccd_code.csv",
    )
)

def parse_pdb_line(line):
    return {
        "atom_num": int(line[6:11].strip()),
        "atom_name": line[12:16].strip(),
        "residue_name": line[17:21].strip(),
        "chain": line[21].strip(),
        "residue_num": int(line[22:26].strip()),
        "insertion_code": line[26].strip(),
        "x": float(line[30:38]),
        "y": float(line[38:46]),
        "z": float(line[46:54]),
    }

def extract_sequences_and_ptm(pdb_lines):
    # 按chain分组存储序列和PTM
    sequences_by_chain = {}
    ptms_by_chain = {}
    residue_numbers_by_chain = {}

    # 第一遍遍历,按chain收集所有残基
    for line in pdb_lines:
        if line.startswith("ATOM") or line.startswith("HETATM"):
            residue_info = parse_pdb_line(line)
            chain_id = residue_info["chain"]
            residue_name = residue_info["residue_name"]
            residue_number = residue_info["residue_num"]
            insertion_code = residue_info["insertion_code"]

            if chain_id not in residue_numbers_by_chain:
                residue_numbers_by_chain[chain_id] = set()

            residue_key = (residue_number, insertion_code, residue_name)
            residue_numbers_by_chain[chain_id].add(residue_key)

    for chain_id in residue_numbers_by_chain:
        sorted_residues = sorted(
            list(residue_numbers_by_chain[chain_id]), key=lambda x: (x[0], x[1])
        )
        residue_to_position = {res: i for i, res in enumerate(sorted_residues)}
        sequence = ["X"] * len(sorted_residues)
        ptms = []

        for line in pdb_lines:
            if (
                line.startswith("ATOM") or line.startswith("HETATM")
            ) and parse_pdb_line(line)["chain"] == chain_id:
                residue_info = parse_pdb_line(line)
                residue_number = residue_info["residue_num"]
                insertion_code = residue_info["insertion_code"]
                letter_res_type = residue_info["residue_name"]

                residue_key = (residue_number, insertion_code, letter_res_type)
                position = residue_to_position[residue_key]

                residue = mmcif_restype_3to1.get(letter_res_type, None)
                if not residue:
                    # 尝试从PTM映射中获取
                    residue = ptm_mapping_dict.get(letter_res_type, None)
                    ptm = {
                        "ptmType": "CCD_" + str(letter_res_type),
                        "ptmPosition": position + 1,
                        "chain": chain_id,
                    }
                    if ptm not in ptms:
                        ptms.append(ptm)
                    if not residue:
                        residue = "X"
                        print(
                            f"Warning: {letter_res_type} is not a standard amino acid or PTM, use X instead."
                        )

                sequence[position] = residue

        assert len(sequence) == len(
            sorted_residues
        ), f"Chain {chain_id}: sequence length {len(sequence)} does not match sorted_residues length {len(sorted_residues)}"

        sequences_by_chain[chain_id] = "".join(sequence)
        ptms_by_chain[chain_id] = ptms
        print(f"Chain {chain_id} sequence:", sequences_by_chain[chain_id])

    return sequences_by_chain, ptms_by_chain