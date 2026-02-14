import argparse
import os
import random

def read_ids(path):
    with open(path, "r", encoding="utf-8") as f:
        ids = [line.strip() for line in f if line.strip()]
    # 保持唯一并保持原始顺序
    seen = set()
    uniq = []
    for x in ids:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq

def write_ids(path, ids):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for x in ids:
            f.write(x + "\n")

def split_list(items, train_frac, seed=None):
    rng = random.Random(seed)
    idxs = list(range(len(items)))
    rng.shuffle(idxs)
    cut = int(len(items) * float(train_frac))
    train_idxs = set(idxs[:cut])
    train = [items[i] for i in range(len(items)) if i in train_idxs]
    test = [items[i] for i in range(len(items)) if i not in train_idxs]
    return train, test

def main():
    parser = argparse.ArgumentParser(description="Split PDB id list into train/test.")
    parser.add_argument("--input", "-i",
                        default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "examples", "finetune_example_pdb_ids.txt"),
                        help="Input txt file with one PDB id per line.")
    parser.add_argument("--train_out", "-tr",
                        default="data/train_pdb_ids.txt",
                        help="Output path for training set.")
    parser.add_argument("--test_out", "-te",
                        default="data/test_pdb_ids.txt",
                        help="Output path for test set.")
    parser.add_argument("--train_frac", "-r",
                        type=float, default=0.8,
                        help="Fraction of ids assigned to train (0.0-1.0). Default 0.8")
    parser.add_argument("--seed", "-s",
                        type=int, default=42,
                        help="Random seed.")
    args = parser.parse_args()

    if not 0.0 < args.train_frac < 1.0:
        raise SystemExit("train_frac must be between 0 and 1 (exclusive).")

    ids = read_ids(args.input)
    if not ids:
        raise SystemExit(f"No ids found in {args.input}")

    train, test = split_list(ids, args.train_frac, seed=args.seed)

    write_ids(args.train_out, train)
    write_ids(args.test_out, test)

    print(f"Input: {args.input}  total={len(ids)}")
    print(f"Train ({len(train)}): saved to {args.train_out}")
    print(f"Test  ({len(test)}): saved to {args.test_out}")

if __name__ == "__main__":
    main()