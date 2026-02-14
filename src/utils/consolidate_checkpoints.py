from pathlib import Path
import torch

# 1) allowlist：必须发生在 torch.load 之前、且在 consolidate 的同一进程里
from src.config.unified_config import ConfigAdapter
from ml_collections.config_dict import ConfigDict

if hasattr(torch.serialization, "add_safe_globals"):
    torch.serialization.add_safe_globals([ConfigAdapter, ConfigDict])

# 2) 调 Lightning Fabric 的 loader（PL consolidate 脚本内部也是走这套）
from lightning_fabric.utilities.load import _load_distributed_checkpoint

import sys

if len(sys.argv) < 2:
    print(f"Usage: {sys.argv[0]} <checkpoint_path>")
    exit(1)
ckpt = Path(sys.argv[1])

obj = _load_distributed_checkpoint(ckpt)
# 关键：Lightning 需要 state_dict
obj["state_dict"] = obj.pop("model")

out = ckpt.with_name("last_consolidated.ckpt")
torch.save(obj, out)
print("saved:", out)
print("keys:", list(out.keys())[:20])
