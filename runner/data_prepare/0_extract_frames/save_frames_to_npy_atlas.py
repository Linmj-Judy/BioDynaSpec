import csv
import multiprocessing as mp
import os
import numpy as np

import MDAnalysis as mda
from MDAnalysis.coordinates.PDB import PDBWriter
from tqdm import tqdm


try:
    import gemmi  # pip install gemmi
    HAS_GEMMI = True
except Exception:
    gemmi = None
    HAS_GEMMI = False

# ===================== User-configurable switches =====================
# 只提取 Cα 原子（BioDynaSpec 是 residue-level / Cα-only 建模，见论文 Sec. 3.1）。
# 设为 False 才会退回按 IGNORE_WATER / IGNORE_IONS 选择多原子。
USE_CA_ONLY = True

# 是否在保存时忽略水分子（使用 MDAnalysis 选择字符串 "water"）
# 仅在 USE_CA_ONLY = False 时生效。
IGNORE_WATER = True

# 是否在保存时忽略常见离子
IGNORE_IONS = True

# 如你的体系里还有特殊的水残基名不在默认 "water" 关键字内，可在此补充
# 一般 water 已包含：WAT, SOL, HOH, TIP3P/TIP4P/TIP5P, SPC/SPCE 等
EXTRA_WATER_RESNAMES = []  # 例如：["W", "H2O"]

# 如你的体系里有不常见的离子(RESNAME)，可在此补充
DEFAULT_ION_RESNAMES = ["NA", "K", "CL", "CA", "MG", "ZN", "FE", "MN", "CU", "CO", "LI", "F", "BR", "IOD", "CS", "SR", "BA", "RB"]
EXTRA_ION_RESNAMES = []   # 例如：["SR", "BA"]

# base_dir = "/home/ymao_pkuhpc/gpfs3/kaiwencheng/MD_raw_data/Atlas"
# base_save_dir = "/home/ymao_pkuhpc/gpfs3/kaiwencheng/MD_data/Atlas"
import argparse

DEFAULT_BASE_DIR = os.environ.get("ATLAS_RAW_DIR", "./data/raw_data/ATLAS")
DEFAULT_BASE_SAVE_DIR = os.environ.get("ATLAS_TRAJ_DIR", "./data/trajectories/atlas")

parser = argparse.ArgumentParser(description="Extract frames and save to npy for Atlas dataset")
parser.add_argument("--base_dir", type=str, default=DEFAULT_BASE_DIR, help="Raw data directory")
parser.add_argument("--base_save_dir", type=str, default=DEFAULT_BASE_SAVE_DIR, help="Processed data save directory")
parser.add_argument("--chunk_size", type=int, default=10000, help="Number of frames per chunk when splitting trajectories")
args = parser.parse_args()

base_dir = args.base_dir
base_save_dir = args.base_save_dir
chunk_size = max(1, args.chunk_size)

suffixes = ["R1", "R2", "R3"]
# ======================================================================

def _build_selection_string():
    """根据开关构建选择字符串。

    BioDynaSpec models residue-level dynamics on Cα coordinates (paper
    Sec. 3.1, Eq. 1), so the default selection is CA-only, matching
    ``replace_positions_full_fft_atlas.py --atom_selection "name CA"``.
    """
    if USE_CA_ONLY:
        return "name CA"
    if not IGNORE_WATER and not IGNORE_IONS:
        return None  # 选择全部原子
    sel = None
    water_sel = ""
    ion_sel = ""
    # water处理
    if IGNORE_WATER:
        if EXTRA_WATER_RESNAMES:
            extras = " ".join(EXTRA_WATER_RESNAMES)
            water_sel = f"(water or resname {extras})"
        else:
            water_sel = "water"
    # ion处理
    if IGNORE_IONS:
        all_ions = set(DEFAULT_ION_RESNAMES + EXTRA_ION_RESNAMES)
        if all_ions:
            ions_str = " ".join(all_ions)
            ion_sel = f"resname {ions_str}"
    # 组合
    exclusion_list = []
    if water_sel:
        exclusion_list.append(water_sel)
    if ion_sel:
        exclusion_list.append(ion_sel)
    if exclusion_list:
        sel = "not (" + " or ".join(exclusion_list) + ")"
    return sel


def process_trajectory(task):
    name, suffix = task
    # tpr_file = os.path.join(base_dir, name, f"{name}.pdb")
    tpr_file = os.path.join(base_dir, name, "total",f"{name}_prod_{suffix}.tpr")
    xtc_file = os.path.join(base_dir, name, "total",f"{name}_prod_{suffix}.xtc")
    save_dir = os.path.join(base_save_dir, f"{name}_{suffix}")
    os.makedirs(save_dir, exist_ok=True)

    if not os.path.exists(xtc_file):
        return f"{name}_{suffix}: MISSING xtc file."
    # preprocess pdb files
    # split_chains(tpr_file, tpr_file)
    
    try:
        u = mda.Universe(tpr_file, xtc_file)
        sel_str = _build_selection_string()
        if sel_str:
            ag = u.select_atoms(sel_str)
            if len(ag) == 0:
                return f"{name}_{suffix}: selection '{sel_str}' is empty."
            sel_tag = "no_water"
        else:
            ag = u.atoms
            sel_tag = "all_atoms"
        ag.chainIDs = np.array([name[-1]] * len(ag))

        chunk_dir = os.path.join(save_dir, f"{name}_{suffix}_{sel_tag}")
        os.makedirs(chunk_dir, exist_ok=True)

        n_frames = len(u.trajectory)
        
        # 预先计算所有chunk的边界，确保衔接：
        # chunk 0: [0, chunk_size-1]
        # chunk 1: [chunk_size-1, 2*chunk_size-1]  (首帧与chunk 0的最后一帧相同)
        # chunk 2: [2*chunk_size-1, 3*chunk_size-1]  (首帧与chunk 1的最后一帧相同)
        # ...
        chunk_boundaries = []
        chunk_start = 0
        while chunk_start < n_frames:
            # chunk_end是当前chunk的最后一帧（包含）
            chunk_end = min(chunk_start + chunk_size - 1, n_frames - 1)
            chunk_boundaries.append((chunk_start, chunk_end))
            # 下一个chunk从当前chunk的最后一帧开始（重叠一帧用于衔接）
            chunk_start = chunk_end
        
        # 如果最后一个chunk帧数太少（少于2帧无法做FFT），合并到前一个chunk
        if len(chunk_boundaries) > 1:
            last_start, last_end = chunk_boundaries[-1]
            last_length = last_end - last_start + 1
            if last_length < 2:  # 至少需要2帧才能做FFT
                # 合并到前一个chunk
                prev_start, prev_end = chunk_boundaries[-2]
                chunk_boundaries[-2] = (prev_start, last_end)
                chunk_boundaries.pop()
        
        # 读取所有帧的坐标到内存（用于后续chunk间衔接）
        all_coords = []
        for ts in u.trajectory:
            all_coords.append(ag.positions.astype(np.float32))
        all_coords = np.stack(all_coords, axis=0)  # (n_frames, n_atoms, 3)
        
        meta_rows = []
        
        # 处理每个chunk
        for chunk_idx, (start_frame, end_frame) in enumerate(chunk_boundaries):
            # chunk的首帧PDB：跳转到start_frame并写入
            u.trajectory[start_frame]
            first_pdb_path = os.path.join(chunk_dir, f"chunk_{chunk_idx}_first_frame.pdb")
            with PDBWriter(first_pdb_path) as writer:
                writer.write(ag)
            
            # 提取该chunk的坐标（包含start_frame到end_frame的所有帧，包含两端）
            chunk_coords = all_coords[start_frame:end_frame+1]  # (length, n_atoms, 3)
            
            # 保存坐标
            coords_path = os.path.join(chunk_dir, f"chunk_{chunk_idx}_frames_coords.npy")
            np.save(coords_path, chunk_coords)
            
            chunk_length = end_frame - start_frame + 1
            meta_rows.append((chunk_idx, start_frame, end_frame, chunk_length))
        
        # 写出块信息，便于后续频谱与逆变换串联
        meta_path = os.path.join(chunk_dir, "chunks_meta.csv")
        with open(meta_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["chunk_idx", "start_frame", "end_frame", "length"])
            writer.writerows(meta_rows)

        return f"{name}_{suffix} done. chunks={len(meta_rows)}, saved in {chunk_dir}"
    except Exception as e:
        return f"{name}_{suffix} error: {str(e)}"

        # for ts in u.trajectory:
        #     frame_number = ts.frame
        #     pdb_path = os.path.join(
        #         save_dir, f"{name}_{suffix}_frame{frame_number:04d}.pdb"
        #     )
        #     with PDBWriter(pdb_path) as writer:
        #         writer.write(u.atoms)
    #     return f"{name}_{suffix} done."
    # except Exception as e:
    #     return f"{name}_{suffix} error: {str(e)}"


if __name__ == "__main__":
    subfolders = [
        name
        for name in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, name))
    ]
    tasks = [(name, suffix) for name in subfolders for suffix in suffixes]

    num_processes = mp.cpu_count()

    with mp.Pool(processes=num_processes) as pool:
        for result in tqdm(
            pool.imap_unordered(process_trajectory, tasks), total=len(tasks)
        ):
            tqdm.write(result)

