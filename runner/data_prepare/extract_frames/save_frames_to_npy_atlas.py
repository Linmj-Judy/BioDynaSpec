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
# 设为 False 才会退回按 IGNORE_WATER 选择多原子。
USE_CA_ONLY = True

# 是否在保存时忽略水分子（使用 MDAnalysis 选择字符串 "water"）
# 仅在 USE_CA_ONLY = False 时生效。
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
    """根据开关构建选择字符串。

    BioDynaSpec models *residue-level* dynamics on Cα coordinates (paper
    Sec. 3.1, Eq. 1), so the default selection is CA-only. Set
    ``USE_CA_ONLY = False`` only if you deliberately want more atoms — the
    downstream FFT/training pipeline assumes one representative atom per
    residue, and `replace_positions_full_fft_atlas.py` defaults to ``name CA``
    for the same reason.
    """
    if USE_CA_ONLY:
        return "name CA"
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
    # ATLAS ships `<name>/<name>_prod_R{1,2,3}_fit.xtc` -- already PBC-removed and
    # frame-aligned (see the dataset README: "with PBC removed and frames
    # aligned"). The old layout here was `<name>/total/<name>_prod_R{n}.xtc`,
    # which does not exist in the raw data and would have picked up a
    # PBC-uncorrected trajectory if one were ever present.
    #
    # The `_fit` file is REQUIRED: an un-unwrapped trajectory puts a step of
    # ~box length into the time series, which the rFFT smears across every
    # frequency bin.
    tpr_file = os.path.join(base_dir, name, f"{name}_prod_{suffix}.tpr")
    xtc_file = os.path.join(base_dir, name, f"{name}_prod_{suffix}_fit.xtc")
    save_dir = os.path.join(base_save_dir, f"{name}_{suffix}")
    os.makedirs(save_dir, exist_ok=True)

    if not os.path.exists(xtc_file):
        return f"{name}_{suffix}: MISSING xtc file: {xtc_file}"
    # preprocess pdb files
    # split_chains(tpr_file, tpr_file)
    
    try:
        u = mda.Universe(tpr_file, xtc_file)
        sel_str = _build_selection_string()
        if sel_str:
            ag = u.select_atoms(sel_str)
            if len(ag) == 0:
                return f"{name}_{suffix}: selection '{sel_str}' is empty."
            # Tag the output with what was actually selected, so CA-only data is
            # not mislabelled as merely "no_water".
            sel_tag = "ca" if USE_CA_ONLY else "no_water"
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

