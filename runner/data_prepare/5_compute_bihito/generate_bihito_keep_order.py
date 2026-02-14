#!/usr/bin/env python3
import os
import argparse
import json
import time
import warnings
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp
import numpy as np

import torch
from tqdm import tqdm
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from Bio.PDB import PDBParser
from Bio.PDB.MMCIFParser import MMCIFParser

from bio2token.models.autoencoder import Autoencoder, AutoencoderConfig
from bio2token.utils.configs import utilsyaml_to_dict, pi_instantiate
from bio2token.data.utils.utils import compute_masks
from bio2token.data.utils.molecule_conventions import ABBRS
from bio2token.data.utils.tokens import (
    AA_TO_TOKEN,
    RNA_TO_TOKEN,
    BB_CLASS,
    C_REF_CLASS,
    SC_CLASS,
)
from bio2token.data.utils.molecule_conventions import (
    SC_ATOMS_AA,
    SC_ATOMS_RNA,
    AMINO_ACID_ABBRS,
    AA_ABRV_REVERSED,
    RNA_ABRV_REVERSED,
    BB_ATOMS_AA,
    BB_ATOMS_RNA,
    AA_C_REF,
    RNA_C_REF,
    AA_BB_REF,
    RNA_BB_REF,
)

warnings.filterwarnings("ignore", category=PDBConstructionWarning)


# ---------- helpers ----------
def list_protein_dirs(pdb_root_dir: Path, only_one: str | None):
    if only_one:
        d = pdb_root_dir / only_one
        if not d.exists():
            raise FileNotFoundError(f"--processing_protein not found: {d}")
        if not d.is_dir():
            raise NotADirectoryError(f"--processing_protein is not a dir: {d}")
        return [d]
    return sorted([p for p in pdb_root_dir.iterdir() if p.is_dir()])


def partition_round_robin(items, n):
    buckets = [[] for _ in range(n)]
    for i, x in enumerate(items):
        buckets[i % n].append(x)
    return buckets


def monitor_progress(total, shared_val, lock, pos):
    pbar = tqdm(total=total, desc="TOTAL", position=pos, leave=True)
    try:
        last = 0
        while True:
            with lock:
                current = shared_val.value
            if current != last:
                pbar.n = current
                pbar.refresh()
                last = current
            if current >= total:
                break
            time.sleep(0.5)
    finally:
        pbar.n = total
        pbar.refresh()
        pbar.close()


def load_model_on_device(global_configs, ckpt_path: str, device: torch.device):
    model_config = pi_instantiate(AutoencoderConfig, yaml_dict=global_configs["model"])
    model = Autoencoder(model_config)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt["state_dict"]
    state_dict_bis = {k.replace("model.", ""): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict_bis)
    model.eval()
    model.to(device)
    return model


# ---------- scheme B: keep original atom order/count ----------
def pdb_2_dict_keep_all(pdb_path: str, chains: list[str] | None = None):
    if Path(pdb_path).name.endswith(".cif"):
        parser = MMCIFParser()
    else:
        parser = PDBParser()
    pdb_ids = []
    pdb_id = os.path.basename(pdb_path).split(".")[0]
    structure = parser.get_structure(pdb_id, pdb_path)
    seq = ""
    coords = []
    atom_names = []
    atom_types = []
    res_names = []
    chain_ids = []
    res_ids = []
    res_types = []
    continuous_res_ids = []
    res_atom_start = []
    res_atom_end = []
    full_res_names = []
    # if no chains are provided we read in everything
    if chains is None:
        chains = [chain.id for chain in structure.get_chains()]
    print(f"PDB loading. Chains: {chains}")
    continuous_res_id = 0
    continuous_atom_id = 0
    for chain in structure.get_chains():
        if chain.id not in chains:
            continue
        res_id = 0
        for residue in chain:
            resname = residue.get_resname()
            if len(resname) == 3 and resname in AA_ABRV_REVERSED:
                ABBRS_REVERSED = AA_ABRV_REVERSED
                res_type = "aa"
            elif resname in RNA_ABRV_REVERSED:
                ABBRS_REVERSED = RNA_ABRV_REVERSED
                res_type = "rna"
            else:
                ABBRS_REVERSED = None
                res_type = "other"

            res = ABBRS_REVERSED[resname] if ABBRS_REVERSED else "X"
            seq += res
            res_atom_start.append(continuous_atom_id)
            res_types.append(res_type)
            for atom in residue:
                atom_names.append(atom.get_name())
                atom_types.append(atom.get_name()[0])
                chain_ids.append(chain.id)
                pdb_ids.append(pdb_id)
                coords.append(np.array(atom.get_coord()))
                res_names.append(res)
                full_res_names.append(resname)
                res_ids.append(res_id)
                continuous_res_ids.append(continuous_res_id)
                continuous_atom_id += 1
                res_id += 1
            res_atom_end.append(continuous_atom_id)
            continuous_res_id += 1
    coords = np.vstack(coords)
    print(f"PDB loading. N residues: {len(res_atom_start)}")
    print(f"PDB loading. N atoms: {len(atom_names)}")
    pdb_dict = {
        "pdb_id": pdb_ids,
        "seq": seq,
        "res_names": res_names,
        "coords_groundtruth": coords,
        "atom_names": atom_names,
        "atom_types": atom_types,
        "seq_length": len(seq),
        "atom_length": len(atom_names),
        "chains": chain_ids,
        "res_ids": res_ids,
        "continuous_res_ids": continuous_res_ids,
        "res_types": res_types,
        "res_atom_start": res_atom_start,
        "res_atom_end": res_atom_end,
        "full_res_names": full_res_names,
    }
    return pdb_dict


def uniform_dataframe_keep_order(
    seq,
    res_types,
    atom_coords,
    atom_names,
    res_atom_start,
    res_atom_end,
    full_res_names,
):
    # Keep original atom order; assign token classes based on atom name matches.
    atom_coords = np.array(atom_coords)
    atom_coords = [atom_coords[res_atom_start[i] : res_atom_end[i]].tolist() for i in range(len(res_atom_start))]
    atom_names = np.array(atom_names)
    atom_names = [atom_names[res_atom_start[i] : res_atom_end[i]].tolist() for i in range(len(res_atom_start))]

    residue_name = []
    mask_bb = []
    mask_c_ref = []
    mask_sc = []
    atom_names_reordered = []
    for i in range(len(atom_names)):
        res_type = res_types[i]
        if res_type == "aa":
            bb_list = BB_ATOMS_AA
            c_ref_list = AA_C_REF
            sc_list = SC_ATOMS_AA[AMINO_ACID_ABBRS[AA_ABRV_REVERSED[full_res_names[i]]]]
            res_token = AA_TO_TOKEN[AA_ABRV_REVERSED[full_res_names[i]]]
        elif res_type == "rna":
            bb_list = BB_ATOMS_RNA
            c_ref_list = RNA_C_REF
            sc_list = SC_ATOMS_RNA[RNA_ABRV_REVERSED[full_res_names[i]]]
            res_token = RNA_TO_TOKEN[RNA_ABRV_REVERSED[full_res_names[i]]]
        else:
            bb_list = []
            c_ref_list = []
            sc_list = []
            res_token = "UNK"

        atom_names_reordered.append(atom_names[i])
        residue_name.append([res_token] * len(atom_names[i]))
        mask_bb.append([name in bb_list for name in atom_names[i]])
        mask_c_ref.append([name in c_ref_list for name in atom_names[i]])
        mask_sc.append([name in sc_list or (name not in bb_list and name not in c_ref_list) for name in atom_names[i]])

    structure = np.array([item for sublist in atom_coords for item in sublist])
    unknown_structure = np.isnan(structure).sum(axis=-1) > 0

    residue_name = [item for sublist in residue_name for item in sublist]
    residue_ids = []
    for i, atoms in enumerate(atom_names):
        residue_ids.extend([i] * len(atoms))
    residue_ids = np.array(residue_ids)

    atom_names_reordered = [item for sublist in atom_names_reordered for item in sublist]
    mask_bb = np.array([item for sublist in mask_bb for item in sublist])
    mask_c_ref = np.array([item for sublist in mask_c_ref for item in sublist])
    mask_sc = np.array([item for sublist in mask_sc for item in sublist])

    token_class = np.zeros(mask_bb.shape).astype(int)
    token_class[mask_bb] = BB_CLASS
    token_class[mask_c_ref] = C_REF_CLASS
    token_class[mask_sc] = SC_CLASS

    barycenter = np.mean(structure[~unknown_structure], axis=0)
    structure[~unknown_structure] = structure[~unknown_structure] - barycenter

    return structure, unknown_structure, residue_name, residue_ids, token_class, atom_names_reordered


def preprocess_pdb(pdb_path: Path, chains_cfg):
    pdb_dict = pdb_2_dict_keep_all(str(pdb_path), chains=chains_cfg)
    chains_mark = pdb_dict["chains"]

    structure, unknown_structure, residue_name, residue_ids, token_class, atom_names_reordered = (
        uniform_dataframe_keep_order(
            pdb_dict["seq"],
            pdb_dict["res_types"],
            pdb_dict["coords_groundtruth"],
            pdb_dict["atom_names"],
            pdb_dict["res_atom_start"],
            pdb_dict["res_atom_end"],
            pdb_dict["full_res_names"],
        )
    )

    structure = np.nan_to_num(structure, nan=0.0)
    batch = {
        "structure": torch.tensor(structure).float(),
        "unknown_structure": torch.tensor(unknown_structure).bool(),
        "residue_ids": torch.tensor(residue_ids).long(),
        "token_class": torch.tensor(token_class).long(),
    }
    batch = compute_masks(batch, structure_track=True)

    is_mol = torch.zeros(1, dtype=torch.bool)

    meta = {
        "chains_mark": chains_mark,
        "unknown_structure": unknown_structure,
        "residue_name": residue_name,
        "residue_ids": residue_ids,
        "atom_names_reordered": atom_names_reordered,
    }
    return batch, is_mol, meta


def move_batch_to_device(batch_cpu: dict, is_mol_cpu: torch.Tensor, device: torch.device):
    batch = {k: v[None].to(device, non_blocking=True) for k, v in batch_cpu.items()}
    batch["is_mol"] = is_mol_cpu.to(device, non_blocking=True)
    return batch


def batch_to_cpu(batch: dict):
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.detach().cpu()
        else:
            out[k] = v
    return out


def save_outputs(
    batch_cpu: dict,
    meta: dict,
    global_configs: dict,
    results_root_dir: Path,
    protein_name: str,
    pdb_name: str,
    ckpt_path_name: str,
):
    protein_results_dir = results_root_dir / protein_name

    residue_name = meta["residue_name"]
    residue_ids = meta["residue_ids"]
    atom_types = meta["atom_names_reordered"]
    unknown_structure = meta["unknown_structure"]
    chains_mark = meta["chains_mark"]

    # fallback to residue_name itself if ABBRS lookup fails
    residue_types = []
    for res in residue_name:
        try:
            residue_types.append(ABBRS[res.split("_")[0]][res.split("_")[1]])
        except Exception:
            residue_types.append(str(res)[:3].upper())

    if "chains" in global_configs["data"] and global_configs["data"]["chains"] is not None:
        chains = "_".join(global_configs["data"]["chains"])
    else:
        chains = "all"

    output_dir = (
        protein_results_dir
        / pdb_name
        / global_configs["infer"]["run_id"]
        / ckpt_path_name
        / chains
    )
    os.makedirs(output_dir, exist_ok=True)

    gt_coords = batch_cpu["structure"][0].numpy()
    rec_coords = batch_cpu["all_atom_coords"][0].numpy()

    from bio2token.data.utils.utils import write_pdb
    write_pdb(
        gt_coords,
        atom_types,
        residue_types,
        residue_ids,
        str(output_dir / "gt.pdb"),
        unknown_structure=unknown_structure,
        chains=chains_mark,
    )
    torch.save(batch_cpu, str(output_dir / "code_layers.pth"))
    write_pdb(
        rec_coords,
        atom_types,
        residue_types,
        residue_ids,
        str(output_dir / "rec.pdb"),
        unknown_structure=unknown_structure,
        chains=chains_mark,
    )

    drop_keys = {"encoding_layers", "encoding", "all_atom_coords", "structure", "losses"}
    keys = []
    keys += [k for k in global_configs["infer"].get("keys_to_summarize", []) if k not in drop_keys]
    keys += [k for k in global_configs["infer"].get("keys_to_save", []) if k not in drop_keys]
    seen = set()
    keys = [k for k in keys if not (k in seen or seen.add(k))]

    test_outputs = {}
    for k in keys:
        if k not in batch_cpu:
            continue
        v = batch_cpu[k]
        if isinstance(v, torch.Tensor):
            if v.ndim > 1:
                eos_mask = batch_cpu["eos_pad_mask"][0]
                outputs_list = v[0][eos_mask == 0].numpy().tolist()
            else:
                outputs_list = v[0].item()
        else:
            outputs_list = v
        test_outputs[k] = outputs_list

    with open(str(output_dir / "outputs.json"), "w") as f:
        json.dump(test_outputs, f, indent=4)


# ---------- worker ----------
def worker_main(
    gpu_id: int,
    protein_dirs: list[str],
    config_path: str,
    ckpt_path: str,
    threads: int,
    io_threads: int,
    shared_progress=None,
    progress_lock=None,
):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    torch.set_num_threads(1)

    global_configs = utilsyaml_to_dict(config_path)

    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(gpu_id)

    pdb_root_dir = Path(global_configs["data"]["pdb_path_dir"])
    results_root_dir = Path(global_configs["infer"]["results_dir"])
    chains_cfg = global_configs["data"]["chains"] if "chains" in global_configs["data"] else None

    ckpt_path_name = ckpt_path.split("/")[-1].strip(".ckpt")
    model = load_model_on_device(global_configs, ckpt_path=ckpt_path, device=device)

    preprocess_pool = ThreadPoolExecutor(max_workers=threads)
    io_pool = ThreadPoolExecutor(max_workers=io_threads)

    try:
        for protein_dir_s in protein_dirs:
            protein_dir = Path(protein_dir_s)
            protein_name = protein_dir.name

            pdb_paths = sorted(protein_dir.glob("*.pdb"))
            if not pdb_paths:
                continue

            protein_results_dir = results_root_dir / protein_name
            print(f"[GPU{gpu_id}] ==> [{protein_name}] pdbs: {len(pdb_paths)}", flush=True)

            futures = {}
            for pdb_path in pdb_paths:
                marker_dir = protein_results_dir / pdb_path.name
                if marker_dir.exists():
                    continue
                futures[preprocess_pool.submit(preprocess_pdb, pdb_path, chains_cfg)] = pdb_path
            if not futures:
                continue

            pbar = tqdm(
                total=len(futures),
                desc=f"[GPU{gpu_id}] {protein_name}",
                position=gpu_id,
                leave=False,
            )
            try:
                for fut in as_completed(futures):
                    pdb_path = futures[fut]
                    try:
                        batch_cpu, is_mol_cpu, meta = fut.result()
                    except Exception as e:
                        print(f"[GPU{gpu_id}] preprocess failed: {protein_name}/{pdb_path.name}: {e}", flush=True)
                        pbar.update(1)
                        if shared_progress is not None and progress_lock is not None:
                            with progress_lock:
                                shared_progress.value += 1
                        continue

                    try:
                        batch_gpu = move_batch_to_device(batch_cpu, is_mol_cpu, device=device)
                        with torch.no_grad():
                            out = model(batch_gpu)

                        for k, v in out.get("losses", {}).items():
                            out[k] = v.item()

                        out_cpu = batch_to_cpu(out)
                    except Exception as e:
                        print(f"[GPU{gpu_id}] inference failed: {protein_name}/{pdb_path.name}: {e}", flush=True)
                        pbar.update(1)
                        if shared_progress is not None and progress_lock is not None:
                            with progress_lock:
                                shared_progress.value += 1
                        continue

                    io_pool.submit(
                        save_outputs,
                        out_cpu,
                        meta,
                        global_configs,
                        results_root_dir,
                        protein_name,
                        pdb_path.name,
                        ckpt_path_name,
                    )
                    pbar.update(1)
                    if shared_progress is not None and progress_lock is not None:
                        with progress_lock:
                            shared_progress.value += 1
            finally:
                pbar.close()
    finally:
        preprocess_pool.shutdown(wait=True)
        io_pool.shutdown(wait=True)


# ---------- entry ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--processing_protein", type=str, default=None)
    parser.add_argument("--ckpt_path", type=str, required=True)

    parser.add_argument("--num_gpus", type=int, default=None, help="默认自动检测")
    parser.add_argument("--threads", type=int, default=24, help="每个GPU进程：预处理线程数")
    parser.add_argument("--io_threads", type=int, default=8, help="每个GPU进程：保存线程数")
    args = parser.parse_args()

    global_configs = utilsyaml_to_dict(args.config)
    pdb_root_dir = Path(global_configs["data"]["pdb_path_dir"])
    results_root_dir = Path(global_configs["infer"]["results_dir"])

    protein_dirs = list_protein_dirs(pdb_root_dir, args.processing_protein)
    if not protein_dirs:
        print("No protein dirs found.")
        return

    ngpu = args.num_gpus
    if ngpu is None:
        ngpu = torch.cuda.device_count() if torch.cuda.is_available() else 1
    ngpu = max(1, ngpu)

    buckets = partition_round_robin([str(p) for p in protein_dirs], ngpu)

    total_tasks = 0
    for protein_dir in protein_dirs:
        protein_results_dir = results_root_dir / protein_dir.name
        pdb_paths = sorted(protein_dir.glob("*.pdb"))
        for pdb_path in pdb_paths:
            marker_dir = protein_results_dir / pdb_path.name
            if marker_dir.exists():
                continue
            total_tasks += 1

    ctx = mp.get_context("spawn")
    shared_progress = ctx.Value("i", 0)
    progress_lock = ctx.Lock()
    monitor_proc = None
    if total_tasks > 0:
        monitor_proc = ctx.Process(
            target=monitor_progress,
            args=(total_tasks, shared_progress, progress_lock, ngpu),
        )
        monitor_proc.start()
    procs = []
    for gpu_id in range(ngpu):
        if not buckets[gpu_id]:
            continue
        p = ctx.Process(
            target=worker_main,
            args=(
                gpu_id,
                buckets[gpu_id],
                args.config,
                args.ckpt_path,
                args.threads,
                args.io_threads,
                shared_progress,
                progress_lock,
            ),
        )
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

    if monitor_proc is not None:
        monitor_proc.join()

    for p in procs:
        if p.exitcode != 0:
            raise SystemExit(p.exitcode)


if __name__ == "__main__":
    main()
