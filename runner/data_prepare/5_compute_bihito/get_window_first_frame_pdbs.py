#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

def index_files(root: Path):
    """更快的递归索引：用 os.scandir，返回 stem -> [Path,...]"""
    xtc_map = {}
    pdb_map = {}

    root = Path(root)

    def walk(dir_path: Path):
        try:
            with os.scandir(dir_path) as it:
                for e in it:
                    # 跳过符号链接目录，避免循环
                    try:
                        if e.is_dir(follow_symlinks=False):
                            walk(Path(e.path))
                            continue
                        if not e.is_file(follow_symlinks=False):
                            continue
                    except OSError:
                        continue

                    name = e.name
                    # 只处理 .xtc / .pdb
                    if name.endswith(".xtc"):
                        stem = name[:-4]
                        xtc_map.setdefault(stem, []).append(Path(e.path))
                    elif name.endswith(".pdb"):
                        stem = name[:-4]
                        pdb_map.setdefault(stem, []).append(Path(e.path))
        except (PermissionError, FileNotFoundError, NotADirectoryError):
            return

    walk(root)
    return xtc_map, pdb_map

def match_pairs(xtc_map, pdb_map, prefer_top_contains=None):
    """
    通过 stem 匹配 xtc/pdb。
    若同 stem 下有多个候选，做一个简单选择策略：
      - 若 prefer_top_contains 给定，则优先选择路径包含该字符串的
      - 否则选择“路径最短”的（通常更像主文件）
    返回: list of (stem, xtc_path, pdb_path, note)
    """
    pairs = []
    stems = sorted(set(xtc_map) & set(pdb_map))

    for stem in stems:
        xtcs = xtc_map[stem]
        pdbs = pdb_map[stem]

        def pick(cands):
            if prefer_top_contains:
                hit = [c for c in cands if prefer_top_contains in str(c)]
                if hit:
                    cands = hit
            # 选路径最短的；若仍多，选字典序最小
            cands = sorted(cands, key=lambda x: (len(str(x)), str(x)))
            return cands[0]

        xtc = pick(xtcs)
        pdb = pick(pdbs)

        note = ""
        if len(xtcs) > 1 or len(pdbs) > 1:
            note = f"multi-candidates: xtc={len(xtcs)}, pdb={len(pdbs)} (picked one)"
        pairs.append((stem, xtc, pdb, note))

    return pairs


def extract_windows_one(stem, xtc_path, pdb_path, out_root, window_size,
                        sele="all", prefix="window", include_last=False):
    """
    对一条轨迹执行抽取：帧 0, window_size, 2*window_size... 写PDB（按原始帧号命名）
    """
    import MDAnalysis as mda  # 放到子进程内导入

    out_dir = Path(out_root) / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    u = mda.Universe(str(pdb_path), str(xtc_path))
    ag = u.select_atoms(sele)

    n_frames = len(u.trajectory)
    frames = list(range(0, n_frames, window_size))
    if include_last and (n_frames - 1) not in frames:
        frames.append(n_frames - 1)

    written = 0
    for fi in frames:
        u.trajectory[fi]
        out_pdb = out_dir / f"{prefix}_{fi:05d}.pdb"
        ag.write(str(out_pdb))
        written += 1

    return {
        "stem": stem,
        "xtc": str(xtc_path),
        "pdb": str(pdb_path),
        "n_frames": n_frames,
        "written": written,
        "out_dir": str(out_dir),
    }


def main():
    ap = argparse.ArgumentParser(
        description="从atlas目录批量匹配 xtc/pdb，并并行抽取每条轨迹每window_size的起始帧保存为PDB（按原始帧号命名）"
    )
    ap.add_argument(
        "--root",
        default="./data/trajectories_xtcs/atlas",
        help="atlas 根目录（递归搜索）"
    )
    ap.add_argument("--out", required=True, help="输出根目录（每条轨迹一个子目录）")
    ap.add_argument("-w", "--window_size", type=int, default=2500, help="window 间隔（帧数）")
    ap.add_argument("--sele", default="all", help='MDAnalysis selection，默认"all"')
    ap.add_argument("--prefix", default="window", help="输出PDB文件名前缀")
    ap.add_argument("--include_last", action="store_true", help="额外输出最后一帧（即使不是整倍数）")
    ap.add_argument("-j", "--jobs", type=int, default=max(1, os.cpu_count() // 2),
                    help="并行进程数（默认CPU一半）")
    ap.add_argument("--prefer_top_contains", default=None,
                    help="当同stem存在多个pdb/xtc候选时，优先选择路径包含该字符串的文件")
    ap.add_argument("--dry_run", action="store_true", help="只打印匹配结果，不实际处理")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"ERROR: root not exists: {root}", file=sys.stderr)
        sys.exit(2)

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    xtc_map, pdb_map = index_files(root)
    pairs = match_pairs(xtc_map, pdb_map, prefer_top_contains=args.prefer_top_contains)

    missing_xtc = sorted(set(pdb_map) - set(xtc_map))
    missing_pdb = sorted(set(xtc_map) - set(pdb_map))

    print(f"Found xtc stems: {len(xtc_map)}")
    print(f"Found pdb stems: {len(pdb_map)}")
    print(f"Matched pairs:   {len(pairs)}")
    if missing_xtc:
        print(f"Stems with pdb but no xtc: {len(missing_xtc)}")
    if missing_pdb:
        print(f"Stems with xtc but no pdb: {len(missing_pdb)}")

    if args.dry_run:
        for stem, xtc, pdb, note in pairs[:50]:
            print(f"[PAIR] {stem}\n  xtc: {xtc}\n  pdb: {pdb}" + (f"\n  note: {note}" if note else ""))
        if len(pairs) > 50:
            print(f"... ({len(pairs)-50} more)")
        return

    # 并行处理
    print(f"Running with jobs={args.jobs}, window_size={args.window_size}")
    futures = []
    ok = 0
    fail = 0

    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        for stem, xtc, pdb, note in pairs:
            futures.append(ex.submit(
                extract_windows_one,
                stem, xtc, pdb,
                str(out_root), args.window_size,
                args.sele, args.prefix, args.include_last
            ))

        for fu in as_completed(futures):
            try:
                res = fu.result()
                ok += 1
                print(f"[OK] {res['stem']}  frames={res['n_frames']}  written={res['written']}  out={res['out_dir']}")
            except Exception as e:
                fail += 1
                print(f"[FAIL] {repr(e)}", file=sys.stderr)

    print(f"Done. OK={ok}, FAIL={fail}, total={len(pairs)}")


if __name__ == "__main__":
    main()