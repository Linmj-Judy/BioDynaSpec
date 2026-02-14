import multiprocessing as mp
import os

import MDAnalysis as mda
from MDAnalysis.coordinates.PDB import PDBWriter
from tqdm import tqdm
import numpy as np

# ===================== User-configurable switches =====================
# 是否在保存时忽略水分子（使用 MDAnalysis 选择字符串 "water"）
IGNORE_WATER = True

# 如你的体系里还有特殊的水残基名不在默认 "water" 关键字内，可在此补充
# 一般 water 已包含：WAT, SOL, HOH, TIP3P/TIP4P/TIP5P, SPC/SPCE 等
EXTRA_WATER_RESNAMES = []  # 例如：["W", "H2O"]

# base_dir = "/home/ymao_pkuhpc/gpfs3/kaiwencheng/MD_raw_data/Atlas"
# base_save_dir = "/home/ymao_pkuhpc/gpfs3/kaiwencheng/MD_data/Atlas"
base_dir = os.environ.get("ATLAS_RAW_DIR", "./data/raw_data/ATLAS")
base_save_dir = os.environ.get("ATLAS_TRAJ_DIR", "./data/trajectories/atlas")

suffixes = ["R1", "R2", "R3"]
# ======================================================================

def _build_selection_string():
    """根据开关构建选择字符串。"""
    if not IGNORE_WATER:
        return None  # 选择全部原子
    sel = "not water"
    if EXTRA_WATER_RESNAMES:
        # 将额外水残基也排除
        extras = " ".join(EXTRA_WATER_RESNAMES)
        sel = f"not (water or resname {extras})"
    return sel



def process_trajectory(task):
    name, suffix = task
    # tpr_file = os.path.join(base_dir, name, f"{name}.pdb")
    tpr_file = os.path.join(base_dir, name, "total",f"{name}_prod_{suffix}.tpr")
    xtc_file = os.path.join(base_dir, name, "total",f"{name}_prod_{suffix}.xtc")
    save_dir = os.path.join(base_save_dir, f"{name}_{suffix}", "complex")
    os.makedirs(save_dir, exist_ok=True)

    if not os.path.exists(xtc_file):
        return f"{name}_{suffix}: MISSING xtc file."
    
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
        u.atoms.chainIDs = np.array([name[-1]] * len(u.atoms))

        for ts in u.trajectory:
            frame_number = ts.frame
            pdb_path = os.path.join(
                save_dir, f"{name}_{suffix}_frame{frame_number:04d}.pdb"
            )
            with PDBWriter(pdb_path) as writer:
                writer.write(u.atoms)
        return f"{name}_{suffix} done."
    except Exception as e:
        return f"{name}_{suffix} error: {str(e)}"


if __name__ == "__main__":
    subfolders = [
        name
        for name in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, name))
    ]
    tasks = [(name, suffix) for name in subfolders for suffix in suffixes]

    num_processes = mp.cpu_count() // 2

    with mp.Pool(processes=num_processes) as pool:
        for result in tqdm(
            pool.imap_unordered(process_trajectory, tasks), total=len(tasks)
        ):
            tqdm.write(result)

