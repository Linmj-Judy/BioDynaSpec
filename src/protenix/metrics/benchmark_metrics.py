
#!/usr/bin/env python3
"""
compute_af3_metrics.py

Compute per‐CIF metrics (RMSD, ligand_pLDDT, ipTM, ipAE) for AlphaFold3 predicted complexes,
aggregate statistics per sample (5 ranks) and overall across all samples.

Dependencies:
  - biopython
  - numpy
  - pandas
  - subprocess (for TMscore)
  - json
  - pathlib
"""

import os
import sys
import json
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from Bio.PDB import PDBParser, Superimposer

current_dir_path = os.path.dirname(os.path.abspath(__file__))

def load_structure(path):
    parser = PDBParser(QUIET=True)
    return parser.get_structure(path.stem, str(path))

def extract_plddt(struct):
    # AF3 writes per‐atom pLDDT in the B‐factor field
    scores = []
    for atom in struct.get_atoms():
        scores.append(atom.get_bfactor())
    return np.array(scores)

def average_plddt_on_chain(struct, chain_id):
    # flatten all atoms in that chain
    scores = [atom.get_bfactor()
              for chain in struct.get_chains() if chain.id == chain_id
              for atom in chain.get_atoms()]
    return np.mean(scores)

def compute_rmsd(pred, ref, chain_id_pred, chain_id_ref, atom_name='CA'):
    # superimpose predicted vs reference on given chain
    pred_chain = pred[0][chain_id_pred]
    ref_chain = ref[0][chain_id_ref]
    coords_pred = []
    coords_ref = []
    for res_p, res_r in zip(pred_chain, ref_chain):
        try:
            coords_pred.append(res_p[atom_name].get_coord())
            coords_ref.append(res_r[atom_name].get_coord())
        except KeyError:
            continue
    coords_pred = np.array(coords_pred)
    coords_ref = np.array(coords_ref)
    sup = Superimposer()
    sup.set_atoms(coords_ref, coords_pred)
    sup.apply(coords_pred)
    return sup.rms

def run_tmscore(pdb1, pdb2):
    """Run external TMscore binary; return TM-score float."""
    proc = subprocess.run([os.path.join(current_dir_path, 'TMscore'), str(pdb1), str(pdb2)],
                          stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                          text=True)
    for line in proc.stdout.splitlines():
        if line.strip().startswith("TM-score"):
            parts = line.split('=')
            return float(parts[1].strip().split()[0])
    return np.nan

def compute_ipTM(pred_cif, ref_pdb, interface_cutoff=5.0):
    """
    Extract interface residues from ref and pred (residues within cutoff).
    Save to temp PDBs and run TMscore.
    """
    # This is a placeholder implementation;
    # for production, implement selection of interface residues properly.
    return run_tmscore(pred_cif, ref_pdb)

def compute_ipAE(pae_json, interface_pairs):
    """
    Given predicted_aligned_error.json and list of interface residue index pairs,
    average the predicted aligned error.
    """
    with open(pae_json) as f:
        data = json.load(f)
    pae = np.array(data['predicted_aligned_error'])  # N×N matrix
    errs = [pae[i, j] for i, j in interface_pairs]
    return np.mean(errs)

def find_interface_pairs(ref_struct, chain_prot, chain_lig, cutoff=5.0):
    """Return list of (i, j) residue indices across interface within cutoff."""
    prot = ref_struct[0][chain_prot]
    lig = ref_struct[0][chain_lig]
    pairs = []
    for res_p in prot:
        for res_l in lig:
            for atom_p in res_p:
                for atom_l in res_l:
                    if (atom_p - atom_l) < cutoff:
                        i = res_p.get_id()[1] - 1
                        j = res_l.get_id()[1] - 1
                        pairs.append((i, j))
                        break
    return list(set(pairs))

def main(pred_dir, ref_dir, output_csv):
    """
    pred_dir: root of predictions, organized as pred_dir/sample_id/rank_?.cif (+ .json for pae)
    ref_dir: directory of native PDBs named sample_id.pdb
    """
    records = []
    for sample in os.listdir(pred_dir):
        samp_dir = Path(pred_dir) / sample
        ref_pdb = Path(ref_dir) / f"{sample}.pdb"
        if not ref_pdb.exists(): continue
        # pre‐load reference
        ref_struct = load_structure(ref_pdb)
        # find interface pairs once
        interface_pairs = find_interface_pairs(ref_struct, chain_prot='A', chain_lig='L')
        for cif in sorted(samp_dir.glob("*.cif")):
            rank = cif.stem.split('_')[-1]
            pred_struct = load_structure(cif)
            # metric: RMSD on ligand (chain L)
            rmsd = compute_rmsd(pred_struct, ref_struct, chain_id_pred='L', chain_id_ref='L')
            # ligand_pLDDT
            l_plddt = average_plddt_on_chain(pred_struct, chain_id='L')
            # ipTM (placeholder using full-complex TMscore)
            iptm = compute_ipTM(cif, ref_pdb)
            # ipAE (requires .json)
            pae_json = cif.with_suffix('.predicted_aligned_error.json')
            if pae_json.exists():
                ipae = compute_ipAE(pae_json, interface_pairs)
            else:
                ipae = np.nan
            records.append({
                'sample': sample,
                'rank': rank,
                'rmsd': rmsd,
                'ligand_pLDDT': l_plddt,
                'ipTM': iptm,
                'ipAE': ipae
            })
    # create DataFrame
    df = pd.DataFrame(records)
    # per-sample stats
    per_sample = df.groupby('sample').agg(['mean','std','min','max'])
    per_sample.to_csv(output_csv.replace('.csv','_per_sample.csv'))
    # overall stats
    overall = df.agg(['mean','std','min','max'])
    overall.to_csv(output_csv.replace('.csv','_overall.csv'))
    # save raw
    df.to_csv(output_csv, index=False)
    print(f"Saved per-CIF data to {output_csv}, per-sample stats and overall stats.")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python compute_af3_metrics.py <pred_dir> <ref_dir> <output.csv>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2], sys.argv[3])
