import gzip
import pickle
from pathlib import Path
import shutil
import os
def is_valid_pkl_gz(path: Path) -> bool:
    """
    尝试读取 gzip 压缩的 pickle 文件的第一个对象。
    读取成功则认为是有效文件，失败则视为无效/损坏。
    """
    try:
        with gzip.open(path, "rb") as f:
            _ = pickle.load(f)  # 只读第一个对象就行
        return True
    except Exception as e:
        print(f"[BAD] {path} -> {type(e).__name__}: {e}")
        return False


def clean_bad_pkl_gz(
    root_dir,
    backup_dir=None,
    recursive: bool = False,
):
    """
    扫描 root_dir 下的所有 .pkl.gz 文件，删除/转移那些无法读取的文件。

    参数:
        root_dir: 要扫描的根目录
        backup_dir: 坏文件备份目录。如果为 None，则直接删除坏文件。
        recursive: 是否递归子目录
    """
    root_dir = Path(root_dir)
    if backup_dir is not None:
        backup_dir = Path(backup_dir)
        backup_dir.mkdir(parents=True, exist_ok=True)

    pattern = "**/*.pkl.gz" if recursive else "*.pkl.gz"
    files = list(root_dir.glob(pattern))

    print(f"在 {root_dir} 中找到 {len(files)} 个 .pkl.gz 文件，开始检查...\n")

    bad_count = 0
    for p in files:
        if not is_valid_pkl_gz(p):
            bad_count += 1
            if backup_dir is not None:
                # 移动到备份目录
                target = backup_dir / p.name
                print(f"  -> 移动到 {target}")
                shutil.move(str(p), str(target))
            else:
                # 直接删除
                print(f"  -> 删除 {p}")
                p.unlink()

    print(f"\n检查完成，无效/损坏文件数量: {bad_count}")


if __name__ == "__main__":
    ROOT = os.environ.get("FOURIER_DATA_DIR", "./data/fourier/")
    BACKUP = os.environ.get("FOURIER_BACKUP_DIR", "./data/fourier/bad")
    os.makedirs(BACKUP, exist_ok=True)
    # recursive=False 表示只扫这一层目录；如果有子目录，改为 True
    clean_bad_pkl_gz(ROOT, backup_dir=BACKUP, recursive=True)