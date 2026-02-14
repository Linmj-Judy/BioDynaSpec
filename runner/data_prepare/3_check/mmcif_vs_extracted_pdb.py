
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from collections import namedtuple
from typing import List, Tuple, Dict, Optional

from Bio.PDB import MMCIFParser, PDBParser, is_aa

ResidueInfo = namedtuple("ResidueInfo", ["resname", "resseq", "icode", "atom_names"])


def load_chain_from_mmcif(
    mmcif_path: str,
    chain_id: str = "A",
    ignore_h: bool = True,
    ignore_water: bool = True,
    ignore_hetero: bool = True,
    altloc_preference: str = "A",
) -> List[ResidueInfo]:
    """
    使用 BioPython 读取 mmCIF，抽取指定链的残基与原子信息。
    - 默认为蛋白聚合物残基（忽略水与异源物），忽略氢，优先选 altloc=A。
    """
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("ref", mmcif_path)
    model = structure[0]
    # 尝试直接拿链 ID；如果失败，列出可用链
    if chain_id not in model:
        avail = ",".join(ch.id for ch in model)
        raise ValueError(f"Chain '{chain_id}' not found in mmCIF. Available: {avail}")
    chain = model[chain_id]
    residues: List[ResidueInfo] = []
    for res in chain.get_residues():
        hetflag, resseq, icode = res.id  # (' ', 12, 'A') for standard; ('W', ... ) water; ('H_', ...) hetero
        name3 = res.get_resname().strip()
        # 过滤策略
        if ignore_water and hetflag == "W":
            continue
        if ignore_hetero and hetflag != " ":
            # 异源物（包含水与配体），已在 ignore_water 上面处理水，这里处理其它异源物
            continue
        # 仅保留标准氨基酸（可进一步收紧）
        if not is_aa(res, standard=True):
            # 如果需要包含非标准氨基酸，可改为 standard=False 或放宽逻辑
            continue
        atom_names = []
        for atom in res.get_atoms():
            # altloc 处理：优先 'A' 其后空格
            altloc = getattr(atom, "altloc", " ")
            if altloc not in (altloc_preference, " "):
                continue
            aname = atom.get_id().strip()
            # 氢过滤：优先使用元素，缺失时用命名首字母
            element = getattr(atom, "element", None)
            if ignore_h and ((element == "H") or aname.startswith("H")):
                continue
            atom_names.append(aname)
        residues.append(ResidueInfo(name3, int(resseq), (icode or " ").strip(), atom_names))
    return residues


def load_chain_from_pdb(
    pdb_path: str,
    chain_id: str = "A",
    ignore_h: bool = True,
    ignore_water: bool = True,
    ignore_hetero: bool = True,
    altloc_preference: str = "A",
) -> List[ResidueInfo]:
    """
    使用 BioPython 读取 PDB（模拟第一帧），抽取指定链的残基与原子信息。
    过滤策略与 mmCIF 一致以保证可比性。
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("mob", pdb_path)
    model = structure[0]
    if chain_id not in model:
        avail = ",".join(ch.id for ch in model)
        raise ValueError(f"Chain '{chain_id}' not found in PDB. Available: {avail}")
    chain = model[chain_id]
    residues: List[ResidueInfo] = []
    for res in chain.get_residues():
        hetflag, resseq, icode = res.id
        name3 = res.get_resname().strip()
        if ignore_water and hetflag == "W":
            continue
        if ignore_hetero and hetflag != " ":
            continue
        if not is_aa(res, standard=True):
            continue
        atom_names = []
        for atom in res.get_atoms():
            altloc = getattr(atom, "altloc", " ")
            if altloc not in (altloc_preference, " "):
                continue
            aname = atom.get_id().strip()
            element = getattr(atom, "element", None)
            if ignore_h and ((element == "H") or aname.startswith("H")):
                continue
            atom_names.append(aname)
        residues.append(ResidueInfo(name3, int(resseq), (icode or " ").strip(), atom_names))
    return residues


def compare_residue_sequences(ref: List[ResidueInfo], mob: List[ResidueInfo]) -> None:
    """
    对比残基序列（按文件顺序）：长度、名称序列、编号与插入码。
    """
    print("=== Residue-level comparison ===")
    print(f"Reference (mmCIF) residues: {len(ref)}")
    print(f"Mobile   (PDB)   residues: {len(mob)}")
    min_len = min(len(ref), len(mob))
    mismatch_count = 0
    for i in range(min_len):
        r, m = ref[i], mob[i]
        if (r.resname != m.resname) or (r.resseq != m.resseq) or (r.icode != m.icode):
            if mismatch_count == 0:
                print("First 10 residue mismatches (index: ref vs mob):")
            if mismatch_count < 10:
                print(
                    f"  idx {i}: "
                    f"ref={r.resname} {r.resseq}{r.icode or ''} "
                    f"vs mob={m.resname} {m.resseq}{m.icode or ''}"
                )
            mismatch_count += 1
    if mismatch_count == 0 and len(ref) == len(mob):
        print("Residue order and numbering match for all positions.")
    else:
        if len(ref) != len(mob):
            print(f"Residue list length differs: ref={len(ref)} vs mob={len(mob)}")
        print(f"Total residue mismatches (by position): {mismatch_count}")


def compare_atomlists_per_residue(ref: List[ResidueInfo], mob: List[ResidueInfo]) -> None:
    """
    对比逐残基的原子名称列表：数量、集合差异，以及顺序差异。
    """
    print("\n=== Atom-level comparison per residue (ignoring H if configured) ===")
    min_len = min(len(ref), len(mob))
    atom_count_mismatch = 0
    atom_set_mismatch = 0
    atom_order_mismatch = 0
    for i in range(min_len):
        r, m = ref[i], mob[i]
        # 仅在残基名一致时对比原子（否则意义不大）
        if r.resname != m.resname:
            continue
        r_list, m_list = r.atom_names, m.atom_names
        if len(r_list) != len(m_list):
            if atom_count_mismatch == 0:
                print("First 10 atom-count mismatches:")
            if atom_count_mismatch < 10:
                print(
                    f"  idx {i} {r.resname} {r.resseq}{r.icode or ''}: "
                    f"ref atoms={len(r_list)} vs mob atoms={len(m_list)}"
                )
            atom_count_mismatch += 1
        r_set, m_set = set(r_list), set(m_list)
        if r_set != m_set:
            if atom_set_mismatch == 0:
                print("First 10 atom-set mismatches (missing/extra atom names):")
            if atom_set_mismatch < 10:
                missing = sorted(list(r_set - m_set))
                extra = sorted(list(m_set - r_set))
                print(
                    f"  idx {i} {r.resname} {r.resseq}{r.icode or ''}: "
                    f"missing_in_mob={missing} extra_in_mob={extra}"
                )
            atom_set_mismatch += 1
        else:
            # 集合相同，检查顺序是否一致
            if r_list != m_list:
                if atom_order_mismatch == 0:
                    print("First 10 atom-order mismatches (sets equal but order differs):")
                if atom_order_mismatch < 10:
                    print(
                        f"  idx {i} {r.resname} {r.resseq}{r.icode or ''}: "
                        f"ref order={r_list} vs mob order={m_list}"
                    )
                atom_order_mismatch += 1
    print(
        f"Summary: atom-count mismatches={atom_count_mismatch}, "
        f"atom-set mismatches={atom_set_mismatch}, "
        f"atom-order mismatches={atom_order_mismatch}"
    )


def summarize_totals(ref: List[ResidueInfo], mob: List[ResidueInfo]) -> None:
    """
    总结原子总数、残基总数。
    """
    n_atoms_ref = sum(len(r.atom_names) for r in ref)
    n_atoms_mob = sum(len(m.atom_names) for m in mob)
    print("\n=== Totals ===")
    print(f"Ref (mmCIF) total residues: {len(ref)}, atoms: {n_atoms_ref}")
    print(f"Mob (PDB)   total residues: {len(mob)}, atoms: {n_atoms_mob}")


def main():
    ap = argparse.ArgumentParser(
        description="Compare chain A between an MD snapshot PDB and the reference mmCIF (residues, order, atoms)."
    )
    ap.add_argument("--mmcif", required=True, help="Path to the reference full crystal structure .cif (mmCIF)")
    ap.add_argument("--pdb", required=True, help="Path to the MD first-frame PDB (chain A snapshot)")
    ap.add_argument("--chain", default="A", help="Chain ID to compare (default: A)")
    ap.add_argument("--keep-het", action="store_true", help="Keep hetero residues/ligands (default: off)")
    ap.add_argument("--keep-water", action="store_true", help="Keep waters (default: off)")
    ap.add_argument("--keep-h", action="store_true", help="Keep hydrogens (default: off)")
    ap.add_argument("--altloc", default="A", help="AltLoc preference (default: A; fallback is ' ')")

    args = ap.parse_args()

    ref = load_chain_from_mmcif(
        args.mmcif,
        chain_id=args.chain,
        ignore_h=not args.keep_h,
        ignore_water=not args.keep_water,
        ignore_hetero=not args.keep_het,
        altloc_preference=args.altloc,
    )
    mob = load_chain_from_pdb(
        args.pdb,
        chain_id=args.chain,
        ignore_h=not args.keep_h,
        ignore_water=not args.keep_water,
        ignore_hetero=not args.keep_het,
        altloc_preference=args.altloc,
    )

    print(f"Loaded chain '{args.chain}': ref residues={len(ref)}, mob residues={len(mob)}")
    compare_residue_sequences(ref, mob)
    compare_atomlists_per_residue(ref, mob)
    summarize_totals(ref, mob)


if __name__ == "__main__":
    main()
