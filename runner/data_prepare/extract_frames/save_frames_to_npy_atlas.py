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

        n_frames = len(u.trajectory)
        n_atoms = len(ag)

        # Pre-allocate coordinates array: (n_frames, n_atoms, 3) in float32
        coords = np.empty((n_frames, n_atoms, 3), dtype=np.float32)

        # Output paths
        first_cif_path = os.path.join(save_dir, f"{name}_{suffix}_{sel_tag}_first_frame.cif")
        first_pdb_path = os.path.join(save_dir, f"{name}_{suffix}_{sel_tag}_first_frame.pdb")
        coords_npy_path = os.path.join(save_dir, f"{name}_{suffix}_{sel_tag}_frames_coords.npy")
        for i, ts in enumerate(u.trajectory):
            # store coordinates
            coords[i] = ag.positions  # cast happens when assigning into float32 array
            # handle first frame CIF once
            if i == 0:
                # if not HAS_GEMMI:
                #     tqdm.write(
                #         f"[WARN] {name}_{suffix}: gemmi not installed; skip CIF. "
                #         f"Install with: pip install gemmi"
                #     )
                # Write a temporary PDB for frame 0, then convert to mmCIF with gemmi
                # PDBWriter (MDAnalysis) does not have a direct argument to specify chain names at write time,
                # because it writes the chain IDs that are present in u.atoms (AtomGroup). 
                # If you want to change the chain names, you must modify u.atoms.chainIDs or u.atoms.segids before writing.
                
                with PDBWriter(first_pdb_path) as writer:
                    writer.write(ag)
                structure = gemmi.read_structure(first_pdb_path)
                structure.make_mmcif_document().write_file(first_cif_path)
                # # clean up temporary PDB
                # try:
                #     os.remove(first_pdb_path)
                # except Exception:
                #     pass
        # Save the coordinates array to .npy
        np.save(coords_npy_path, coords)

        return (
            f"{name}_{suffix} done. "
            f"CIF: {os.path.basename(first_cif_path)}, "
            f"coords: {os.path.basename(coords_npy_path)} "
            f"shape={coords.shape}"
        )
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

