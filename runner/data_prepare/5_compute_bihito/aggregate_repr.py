
import os
import re
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import torch


# ---------- utils ----------
_FRAME_RE = re.compile(r"_(?:frame)?(\d{5})\.pdb$")

def parse_frame_idx(p: str | Path) -> int:
    s = str(p)
    m = _FRAME_RE.search(s)
    if not m:
        raise ValueError(f"Bad path (expect *_(frame)?#####.pdb): {s}")
    return int(m.group(1))


def _load_one_frame(pdb_root: Path):
    """
    返回: (frame_idx:int, out:dict[str, Tensor])
    """
    frame_idx = parse_frame_idx(pdb_root.name)

    p = next(pdb_root.rglob("code_layers.pth"), None)
    if p is None:
        raise FileNotFoundError(f"No code_layers.pth under {pdb_root}")

    bihito_batch_dict = torch.load(str(p), map_location="cpu", weights_only=False)

    # 统一去掉梯度信息
    for k, v in list(bihito_batch_dict.items()):
        if isinstance(v, torch.Tensor):
            bihito_batch_dict[k] = v.detach()

    encoding = bihito_batch_dict["encoding"]                 # (1, n_atom, 128)
    encoding_layers = bihito_batch_dict["encoding_layers"]   # list[tensor], each (1, ..., 128)

    out = {
        "final_encoding": encoding.squeeze(0),
        "level1_encoding": encoding_layers[0].squeeze(0),  # all atom
        "level2_encoding": encoding_layers[1].squeeze(0),  # backbone
        "level3_encoding": encoding_layers[2].squeeze(0),  # alpha c
        "level4_encoding": encoding_layers[3].squeeze(0),  # alpha c 间隔采样
        "level5_encoding": encoding_layers[4].squeeze(0),  # global
    }
    return frame_idx, out


def _process_one_traj(
    traj_subdir_str: str,
    out_dir_str: str,
    frame_threads: int = 4,
    overwrite: bool = False,
):
    """
    子进程入口：处理一条trajectory并直接写出 .pt
    返回: 输出文件路径(str)
    """
    # 防止每个进程内部再开太多算子线程导致CPU超卖
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    torch.set_num_threads(1)

    traj_subdir = Path(traj_subdir_str)
    out_dir = Path(out_dir_str)
    out_dir.mkdir(parents=True, exist_ok=True)

    traj_name = traj_subdir.name
    pdb_subdirs = sorted([p for p in traj_subdir.iterdir() if p.is_dir()])
    if not pdb_subdirs:
        return str(out_dir / f"{traj_name.lower()}.pt")

    bihito_out_dict = {}

    # trajectory 内部并行加载 frame（IO/反序列化为主；线程数别开太大）
    if frame_threads and frame_threads > 1:
        with ThreadPoolExecutor(max_workers=frame_threads) as tp:
            futs = {tp.submit(_load_one_frame, pdb_root): pdb_root for pdb_root in pdb_subdirs}
            for fut in as_completed(futs):
                frame_idx, out = fut.result()
                bihito_out_dict[frame_idx] = out
    else:
        for pdb_root in pdb_subdirs:
            frame_idx, out = _load_one_frame(pdb_root)
            bihito_out_dict[frame_idx] = out

    out_path = out_dir / f"{traj_name.lower()}.pt"
    if overwrite or not os.path.exists(str(out_path)):
        torch.save(bihito_out_dict, str(out_path))
    return str(out_path)


# ---------- main ----------
def export_all_trajs_parallel(
    bihito_cache_dir: str | Path,
    bihito_repr_out_path: str | Path,
    traj_workers: int | None = None,
    frame_threads: int = 4,
    overwrite: bool = False,
):
    """
    - 不同 traj_subdir：多进程并行
    - 同一 traj_subdir 内：可选多线程并行读取/torch.load
    """
    bihito_cache_dir = Path(bihito_cache_dir)
    bihito_repr_out_path = Path(bihito_repr_out_path)
    bihito_repr_out_path.mkdir(parents=True, exist_ok=True)

    traj_subdirs = sorted([p for p in bihito_cache_dir.iterdir() if p.is_dir()])
    print(f"Found {len(traj_subdirs)} traj_subdirs under {bihito_cache_dir}")

    if traj_workers is None:
        # 经验值：IO+pickle为主，别把192核全打满；先用 32~64 试
        traj_workers = min(64, (os.cpu_count() or 64))

    # 重要：不要把大量Tensor返回主进程；子进程内直接torch.save
    with ProcessPoolExecutor(max_workers=traj_workers) as pp:
        futs = []
        for traj_subdir in traj_subdirs:
            futs.append(
                pp.submit(
                    _process_one_traj,
                    str(traj_subdir),
                    str(bihito_repr_out_path),
                    int(frame_threads),
                    overwrite,
                )
            )

        for fut in as_completed(futs):
            out_path = fut.result()
            print(out_path, flush=True)

# ---------- example usage ----------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="atlas")
    parser.add_argument("--bihito_cache_dir", type=str, default=None)
    parser.add_argument("--bihito_repr_out_path", type=str, default=None)
    parser.add_argument("--traj_workers", type=int, default=64)
    parser.add_argument("--frame_threads", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    dataset_name = args.dataset_name
    bihito_cache_dir = (
        args.bihito_cache_dir
        if args.bihito_cache_dir
        else f"./dataset/bihito_cache/{dataset_name}"
    )
    bihito_repr_out_path = (
        args.bihito_repr_out_path
        if args.bihito_repr_out_path
        else f"./dataset/bihito_repr_cache/{dataset_name}"
    )
    export_all_trajs_parallel(
        bihito_cache_dir=bihito_cache_dir,
        bihito_repr_out_path=bihito_repr_out_path,
        traj_workers=args.traj_workers,     # 建议先 32~64 之间找最优
        frame_threads=args.frame_threads,   # 每条轨迹内部加载frame的线程数；不建议>8
        overwrite=args.overwrite,
    )