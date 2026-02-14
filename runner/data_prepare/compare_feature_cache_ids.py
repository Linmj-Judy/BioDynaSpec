#!/usr/bin/env python3
import os
from typing import Set

import torch

DATA_LIST_PATH = os.environ.get("DATA_LIST_PATH", "./dataset/atlas_10_features_fouriered_list_train.pth")
FEATURE_CACHE_DIR = os.environ.get("FEATURE_CACHE_DIR", "./dataset/feature_cache/atlas")
FEATURE_CACHE_AFTER_BIHITO_DIR = os.environ.get("FEATURE_CACHE_AFTER_BIHITO_DIR", "./dataset/feature_cache_after_bihito/atlas")


def list_cache_ids_lower(path: str) -> Set[str]:
    try:
        names = set()
        for name in os.listdir(path):
            full = os.path.join(path, name)
            if os.path.isdir(full):
                names.add(name.lower())
            elif name.endswith(".pt"):
                names.add(os.path.splitext(name)[0].lower())
        return names
    except FileNotFoundError:
        return set()


def main() -> None:
    data_list = torch.load(DATA_LIST_PATH, weights_only=False)
    if not isinstance(data_list, list):
        raise ValueError(f"Expected list from {DATA_LIST_PATH}, got {type(data_list)}")

    ids = set()
    for item in data_list:
        if isinstance(item, dict) and "id" in item:
            ids.add(str(item["id"]).lower())

    cache_ids = list_cache_ids_lower(FEATURE_CACHE_DIR)
    cache_after_ids = list_cache_ids_lower(FEATURE_CACHE_AFTER_BIHITO_DIR)

    in_cache = sorted(ids & cache_ids)
    in_cache_after = sorted(ids & cache_after_ids)
    in_both = sorted(ids & cache_ids & cache_after_ids)
    in_neither = sorted(ids - (cache_ids | cache_after_ids))

    print(f"Total ids in data list: {len(ids)}")
    print(f"Found in feature_cache/atlas: {len(in_cache)}")
    print(f"Found in feature_cache_after_bihito/atlas: {len(in_cache_after)}")
    print(f"Found in both: {len(in_both)}")
    print(f"Found in neither: {len(in_neither)}")

    # Print examples for quick inspection
    if in_neither:
        print("\nSample ids not found in either (up to 20):")
        for _id in in_neither[:20]:
            print(f"  {_id}")


if __name__ == "__main__":
    main()
