from .utils import TrajectoryFFTProcessor, PreprocessStateStruct, _restore_layout
from typing import Dict, Any
import torch


def reconstruct_trajectory_from_label_full(
    processor: TrajectoryFFTProcessor,
    label_full_dict: Dict[str, Any],
    spectrum_key: str = "trajectory_spectrum",
    fold_back_to_box: bool = False,
) -> torch.Tensor:
    """
    使用保存于 label_full_dict 的完整频谱与状态，重建完整轨迹的原始坐标时域信号。
    返回 shape 为保存时 original_layout 的张量。
    """
    pack = label_full_dict[spectrum_key]
    dt = float(pack["dt"])
    T = int(pack["T"])
    n_fft = int(pack["n_fft"])
    state = PreprocessStateStruct.from_dict(pack["preprocess_state"])

    # 恢复频谱
    if "X" in pack:
        X = pack["X"]
    else:
        X = pack["X_real"].to(torch.float64 if state.dtype_name == "float64" else torch.float32)
        X = X + 1j * pack["X_imag"].to(torch.float64 if state.dtype_name == "float64" else torch.float32)

    # 逆变换得到预处理域信号
    x_proc = processor.reconstruct_processed(X, original_T=T)

    # 反向还原预处理
    x_orig_n3t = processor.invert_preprocessing(x_proc, state, fold_back_to_box=fold_back_to_box)

    # 恢复为保存时的布局
    x_orig = _restore_layout(x_orig_n3t, state.original_layout)
    return x_orig

# TODO: 
# 1. 从恢复的坐标写入cif文件（把一堆轨迹写入一个？），构成抽帧后的结构序列
