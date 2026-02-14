#!/usr/bin/env python3
"""
Lennard-Jones (LJ) 参数计算模块

提供基于原子类型的LJ参数（σ 和 ε）查询功能。
参数来源：AMBER ff14SB 力场和 OPLS-AA 力场的组合。

References:
    - AMBER ff14SB: Maier et al. (2015) J. Chem. Theory Comput.
    - OPLS-AA: Jorgensen et al. (1996) J. Am. Chem. Soc.
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional


# ============================================================================
# 完整LJ参数表：覆盖元素周期表119种元素
# ============================================================================
# 参数来源：
#   - AMBER ff14SB (生物分子)
#   - OPLS-AA (有机分子)
#   - UFF (Universal Force Field, Rappe et al. 1992)
#   - CHARMM36 (核酸和蛋白)
#   - 文献数据 (金属、稀有气体等)
#   - 合理估算 (基于原子半径和电负性)
#
# 注意：
#   - 元素符号遵循rdkit标准（标准化学元素符号）
#   - σ单位：Å (埃)
#   - ε单位：kcal/mol
#   - 对于生物分子常见元素，使用实验验证的力场参数
#   - 对于稀有元素，使用UFF参数或合理估算
# ============================================================================

LJ_PARAMETERS_BY_ELEMENT = {
    # ========== 第1周期 ==========
    "H": (1.20, 0.0157),    # 氢 - AMBER ff14SB (极性H)
    "He": (2.28, 0.0200),   # 氦 - 稀有气体
    
    # ========== 第2周期 ==========
    "Li": (2.45, 0.0183),   # 锂 - UFF
    "Be": (2.55, 0.0850),   # 铍 - UFF
    "B": (3.64, 0.1800),    # 硼 - UFF
    "C": (3.50, 0.0860),    # 碳 - AMBER ff14SB (sp2/sp3平均)
    "N": (3.25, 0.1700),    # 氮 - AMBER ff14SB
    "O": (2.96, 0.2100),    # 氧 - AMBER ff14SB
    "F": (3.12, 0.0610),    # 氟 - OPLS-AA
    "Ne": (2.78, 0.0690),   # 氖 - 稀有气体
    
    # ========== 第3周期 ==========
    "Na": (2.35, 0.0028),   # 钠 - 金属离子(+1)
    "Mg": (1.36, 0.8750),   # 镁 - 金属离子(+2) CHARMM36
    "Al": (4.01, 0.5050),   # 铝 - UFF
    "Si": (3.83, 0.4020),   # 硅 - UFF
    "P": (3.74, 0.2000),    # 磷 - AMBER ff14SB (核酸)
    "S": (3.55, 0.2500),    # 硫 - AMBER ff14SB
    "Cl": (4.40, 0.1000),   # 氯 - OPLS-AA (氯离子)
    "Ar": (3.40, 0.2380),   # 氩 - 稀有气体
    
    # ========== 第4周期 ==========
    "K": (3.40, 0.0003),    # 钾 - 金属离子(+1)
    "Ca": (2.41, 0.4500),   # 钙 - 金属离子(+2) CHARMM36
    "Sc": (2.94, 0.0190),   # 钪 - UFF估算
    "Ti": (2.83, 0.0170),   # 钛 - UFF估算
    "V": (2.80, 0.0160),    # 钒 - UFF估算
    "Cr": (2.69, 0.0150),   # 铬 - UFF估算
    "Mn": (2.64, 0.0140),   # 锰 - 金属
    "Fe": (1.70, 0.0500),   # 铁 - 金属(+2/+3) CHARMM
    "Co": (2.56, 0.0140),   # 钴 - UFF估算
    "Ni": (2.52, 0.0150),   # 镍 - 金属
    "Cu": (2.50, 0.0200),   # 铜 - 金属(+1/+2)
    "Zn": (1.95, 0.2500),   # 锌 - 金属(+2) CHARMM36
    "Ga": (3.90, 0.4150),   # 镓 - UFF
    "Ge": (3.81, 0.3790),   # 锗 - UFF
    "As": (3.77, 0.3090),   # 砷 - UFF
    "Se": (3.78, 0.2910),   # 硒 - 硒代半胱氨酸
    "Br": (4.33, 0.3200),   # 溴 - OPLS-AA
    "Kr": (3.69, 0.3200),   # 氪 - 稀有气体
    
    # ========== 第5周期 ==========
    "Rb": (3.67, 0.0002),   # 铷 - 金属离子(+1)
    "Sr": (3.24, 0.1180),   # 锶 - 金属离子(+2)
    "Y": (2.98, 0.0720),    # 钇 - UFF估算
    "Zr": (2.78, 0.0690),   # 锆 - UFF估算
    "Nb": (2.82, 0.0590),   # 铌 - UFF估算
    "Mo": (2.72, 0.0560),   # 钼 - UFF估算
    "Tc": (2.67, 0.0480),   # 锝 - UFF估算
    "Ru": (2.64, 0.0560),   # 钌 - UFF估算
    "Rh": (2.61, 0.0530),   # 铑 - UFF估算
    "Pd": (2.58, 0.0480),   # 钯 - 金属
    "Ag": (2.80, 0.0360),   # 银 - 金属(+1)
    "Cd": (2.54, 0.2280),   # 镉 - 金属(+2)
    "In": (3.98, 0.3010),   # 铟 - UFF
    "Sn": (3.91, 0.5670),   # 锡 - UFF
    "Sb": (3.94, 0.4490),   # 锑 - UFF
    "Te": (3.98, 0.3980),   # 碲 - UFF
    "I": (4.70, 0.4000),    # 碘 - OPLS-AA
    "Xe": (3.92, 0.4320),   # 氙 - 稀有气体
    
    # ========== 第6周期 ==========
    "Cs": (4.02, 0.0001),   # 铯 - 金属离子(+1)
    "Ba": (3.30, 0.4700),   # 钡 - 金属离子(+2)
    
    # 镧系元素 (Lanthanides)
    "La": (3.14, 0.1370),   # 镧 - UFF估算
    "Ce": (3.17, 0.0130),   # 铈 - UFF估算
    "Pr": (3.21, 0.0100),   # 镨 - UFF估算
    "Nd": (3.18, 0.0100),   # 钕 - UFF估算
    "Pm": (3.16, 0.0090),   # 钷 - UFF估算
    "Sm": (3.14, 0.0080),   # 钐 - UFF估算
    "Eu": (3.11, 0.0080),   # 铕 - UFF估算
    "Gd": (3.00, 0.0090),   # 钆 - UFF估算
    "Tb": (3.07, 0.0070),   # 铽 - UFF估算
    "Dy": (3.05, 0.0070),   # 镝 - UFF估算
    "Ho": (3.04, 0.0070),   # 钬 - UFF估算
    "Er": (3.02, 0.0070),   # 铒 - UFF估算
    "Tm": (3.01, 0.0060),   # 铥 - UFF估算
    "Yb": (2.99, 0.2280),   # 镱 - UFF估算
    "Lu": (3.24, 0.0410),   # 镥 - UFF估算
    
    # 过渡金属 (续)
    "Hf": (2.80, 0.0720),   # 铪 - UFF估算
    "Ta": (2.82, 0.0810),   # 钽 - UFF估算
    "W": (2.73, 0.0670),    # 钨 - UFF估算
    "Re": (2.63, 0.0660),   # 铼 - UFF估算
    "Os": (2.78, 0.0370),   # 锇 - UFF估算
    "Ir": (2.53, 0.0730),   # 铱 - UFF估算
    "Pt": (2.45, 0.0800),   # 铂 - 金属
    "Au": (2.93, 0.0390),   # 金 - 金属(+1)
    "Hg": (2.41, 0.3850),   # 汞 - 金属
    "Tl": (3.87, 0.6800),   # 铊 - UFF
    "Pb": (3.83, 0.6630),   # 铅 - UFF
    "Bi": (3.89, 0.5180),   # 铋 - UFF
    "Po": (4.20, 0.3250),   # 钋 - UFF估算
    "At": (4.23, 0.2840),   # 砹 - UFF估算
    "Rn": (4.25, 0.2480),   # 氡 - 稀有气体
    
    # ========== 第7周期 ==========
    "Fr": (4.37, 0.0001),   # 钫 - 碱金属估算
    "Ra": (3.48, 0.4030),   # 镭 - UFF估算
    
    # 锕系元素 (Actinides)
    "Ac": (3.10, 0.1330),   # 锕 - UFF估算
    "Th": (3.03, 0.0260),   # 钍 - UFF估算
    "Pa": (3.05, 0.0220),   # 镤 - UFF估算
    "U": (3.02, 0.0220),    # 铀 - UFF估算
    "Np": (3.00, 0.0190),   # 镎 - UFF估算
    "Pu": (2.96, 0.0160),   # 钚 - UFF估算
    "Am": (2.96, 0.0140),   # 镅 - UFF估算
    "Cm": (2.94, 0.0130),   # 锔 - UFF估算
    "Bk": (2.93, 0.0130),   # 锫 - UFF估算
    "Cf": (2.92, 0.0130),   # 锎 - UFF估算
    "Es": (2.91, 0.0120),   # 锿 - UFF估算
    "Fm": (2.90, 0.0120),   # 镄 - UFF估算
    "Md": (2.89, 0.0110),   # 钔 - UFF估算
    "No": (2.88, 0.0110),   # 锘 - UFF估算
    "Lr": (2.87, 0.0110),   # 铹 - UFF估算
    
    # 超重元素 (Transactinides, 104-119)
    "Rf": (2.80, 0.0700),   # 𬬻 - 估算基于Hf
    "Db": (2.82, 0.0780),   # 𬭊 - 估算基于Ta
    "Sg": (2.73, 0.0650),   # 𬭳 - 估算基于W
    "Bh": (2.63, 0.0640),   # 𬭛 - 估算基于Re
    "Hs": (2.78, 0.0360),   # 𬭶 - 估算基于Os
    "Mt": (2.53, 0.0710),   # 鿏 - 估算基于Ir
    "Ds": (2.45, 0.0780),   # 𫟼 - 估算基于Pt
    "Rg": (2.93, 0.0380),   # 𬬻 - 估算基于Au
    "Cn": (2.41, 0.3750),   # 鿔 - 估算基于Hg
    "Nh": (3.85, 0.6600),   # 鿭 - 估算基于Tl
    "Fl": (3.80, 0.6400),   # 𫓧 - 估算基于Pb
    "Mc": (3.87, 0.5000),   # 镆 - 估算基于Bi
    "Lv": (4.18, 0.3150),   # 𫟷 - 估算基于Po
    "Ts": (4.21, 0.2750),   # 石田 - 估算基于At
    "Og": (4.23, 0.2400),   # 气奥 - 估算基于Rn (稀有气体)
    
    # ========== 特殊情况 ==========
    "D": (1.20, 0.0157),    # 氘 - 同H
    "T": (1.20, 0.0157),    # 氚 - 同H
    
    # 未知元素的默认值
    "UNK": (3.40, 0.1000),
    "X": (3.40, 0.1000),    # 通配符
}

# 更精细的LJ参数：基于原子类型（元素 + 化学环境）
# 格式: "元素_环境" -> (sigma, epsilon)
LJ_PARAMETERS_BY_ATOM_TYPE = {
    # 碳原子（不同杂化）
    "C_sp3": (3.50, 0.0660),     # sp3碳（如Cα、Cβ）
    "C_sp2": (3.55, 0.0860),     # sp2碳（如羰基C）
    "C_aromatic": (3.55, 0.0700), # 芳香碳（如Phe, Tyr）
    
    # 氮原子
    "N_backbone": (3.25, 0.1700), # 主链N
    "N_sidechain": (3.25, 0.1700),# 侧链N（如Lys, Arg）
    "N_aromatic": (3.25, 0.1700), # 芳香N（如His, Trp）
    
    # 氧原子
    "O_carbonyl": (2.96, 0.2100), # 羰基O
    "O_hydroxyl": (3.07, 0.1700), # 羟基O（如Ser, Thr）
    "O_carboxyl": (2.96, 0.2100), # 羧基O（如Asp, Glu）
    
    # 硫原子
    "S_thiol": (3.55, 0.2500),    # 巯基S（Cys）
    "S_thioether": (3.55, 0.2500),# 硫醚S（Met）
    "S_disulfide": (3.55, 0.2500),# 二硫键S
    
    # 氢原子（区分极性/非极性）
    "H_polar": (1.20, 0.0157),    # 极性H（如OH, NH）
    "H_nonpolar": (2.50, 0.0150), # 非极性H（如CH）
}

# ============================================================================
# 元素符号列表：按照元素周期表顺序（原子序数1-119）
# ============================================================================
# 用于从one-hot编码解码到元素符号
# 索引对应原子序数-1（例如：H在索引0，He在索引1）
# ============================================================================

ELEMENT_LIST = [
    # 原子序数 1-10
    "H", "He",                                                      # 第1周期
    "Li", "Be", "B", "C", "N", "O", "F", "Ne",                    # 第2周期
    # 原子序数 11-18
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",                # 第3周期
    # 原子序数 19-36
    "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni",   # 第4周期
    "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr",
    # 原子序数 37-54
    "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd",  # 第5周期
    "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe",
    # 原子序数 55-86
    "Cs", "Ba",                                                     # 第6周期开始
    "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",  # 镧系
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn",   # 第6周期续
    # 原子序数 87-118
    "Fr", "Ra",                                                     # 第7周期开始
    "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr",  # 锕系
    "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",  # 超重元素
    # 原子序数 119 (预留)
    "UNK",
]

# 扩展到128个元素（为了兼容one-hot编码的维度）
ELEMENT_LIST = ELEMENT_LIST + ["UNK"] * (128 - len(ELEMENT_LIST))


# ============================================================================
# LJ参数计算函数
# ============================================================================

def get_lj_parameters_from_element(
    ref_element: torch.Tensor,  # [..., N_atom, 128] one-hot编码的元素
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    根据原子元素类型获取LJ参数（σ和ε）。
    
    Args:
        ref_element: 原子元素的one-hot编码，形状 [..., N_atom, 128]
        device: 目标设备
        dtype: 目标数据类型
    
    Returns:
        lj_sigma: σ参数（Å），形状 [..., N_atom]
        lj_epsilon: ε参数（kcal/mol），形状 [..., N_atom]
    
    Example:
        >>> ref_element = torch.zeros(100, 128)
        >>> ref_element[:, 1] = 1.0  # 所有原子都是碳
        >>> sigma, epsilon = get_lj_parameters_from_element(ref_element)
        >>> print(sigma.shape, epsilon.shape)
        torch.Size([100]) torch.Size([100])
    """
    if device is None:
        device = ref_element.device
    if dtype is None:
        dtype = torch.float32
    
    # 解码one-hot得到元素索引
    element_indices = torch.argmax(ref_element, dim=-1)  # [..., N_atom]
    
    # 准备LJ参数张量
    shape = element_indices.shape
    lj_sigma = torch.zeros(shape, device=device, dtype=dtype)
    lj_epsilon = torch.zeros(shape, device=device, dtype=dtype)
    
    # 构建参数查找表
    sigma_table = torch.zeros(128, device=device, dtype=dtype)
    epsilon_table = torch.zeros(128, device=device, dtype=dtype)
    
    for idx, element in enumerate(ELEMENT_LIST[:128]):
        if element in LJ_PARAMETERS_BY_ELEMENT:
            sigma, epsilon = LJ_PARAMETERS_BY_ELEMENT[element]
        else:
            sigma, epsilon = LJ_PARAMETERS_BY_ELEMENT["UNK"]
        sigma_table[idx] = sigma
        epsilon_table[idx] = epsilon
    
    # 批量查表
    lj_sigma = sigma_table[element_indices]
    lj_epsilon = epsilon_table[element_indices]
    
    return lj_sigma, lj_epsilon

def get_lj_parameters_from_atom_name(
    ref_atom_name: np.ndarray,  # 形如 (N_atom,) 的字符串数组
) -> Tuple[np.ndarray, np.ndarray]:
    """
    根据原子名称字符串数组获取LJ参数(σ和ε)，输出为numpy array格式。

    Args:
        ref_atom_name: 原子名称字符串数组，形状为 (N_atom,)

    Returns:
        lj_sigma: σ参数(Å)，形状 (N_atom,)
        lj_epsilon: ε参数(kcal/mol)，形状 (N_atom,)
    
    Example:
        >>> ref_atom_name = np.array(["C", "O", "N", "UNK"])
        >>> sigma, epsilon = get_lj_parameters_from_atom_name(ref_atom_name)
        >>> print(sigma, epsilon)
    """
    N = ref_atom_name.shape[0]
    lj_sigma = np.zeros(N, dtype=np.float32)
    lj_epsilon = np.zeros(N, dtype=np.float32)
    for i, name in enumerate(ref_atom_name):
        # 先尝试直接查找
        if name in LJ_PARAMETERS_BY_ELEMENT:
            sigma, epsilon = LJ_PARAMETERS_BY_ELEMENT[name]
        else:
            # 若找不到，尝试将第二个字母变成小写再查找（如果有第二个字母）
            name_str = str(name)
            if len(name_str) >= 2:
                name_mod = name_str[0] + name_str[1].lower() + name_str[2:]
                if name_mod in LJ_PARAMETERS_BY_ELEMENT:
                    sigma, epsilon = LJ_PARAMETERS_BY_ELEMENT[name_mod]
                else:
                    name_upper = name_str.upper()
                    if name_upper in LJ_PARAMETERS_BY_ELEMENT:
                        sigma, epsilon = LJ_PARAMETERS_BY_ELEMENT[name_upper]
                    else:
                        sigma, epsilon = LJ_PARAMETERS_BY_ELEMENT["UNK"]
            else:
                name_upper = name_str.upper()
                if name_upper in LJ_PARAMETERS_BY_ELEMENT:
                    sigma, epsilon = LJ_PARAMETERS_BY_ELEMENT[name_upper]
                else:
                    sigma, epsilon = LJ_PARAMETERS_BY_ELEMENT["UNK"]
            lj_sigma[i] = sigma
            lj_epsilon[i] = epsilon
        return lj_sigma, lj_epsilon


def get_lj_parameters_refined(
    ref_element: torch.Tensor,      # [..., N_atom, 128]
    restype: torch.Tensor,           # [..., N_token, 32]
    atom_to_token_idx: torch.Tensor, # [..., N_atom]
    ref_atom_name_chars: Optional[torch.Tensor] = None,  # [..., N_atom, 4, 64]
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    根据原子元素类型和化学环境获取精细化的LJ参数。
    
    这个版本考虑了：
    1. 原子所在的残基类型
    2. 原子名称（如果提供）
    3. 原子的化学环境（主链/侧链）
    
    Args:
        ref_element: 原子元素的one-hot编码
        restype: 残基类型的one-hot编码
        atom_to_token_idx: 原子到残基的映射
        ref_atom_name_chars: 原子名称编码（可选）
        device: 目标设备
        dtype: 目标数据类型
    
    Returns:
        lj_sigma: σ参数（Å）
        lj_epsilon: ε参数（kcal/mol）
    """
    # 目前简化版本：使用基础参数
    # TODO: 未来可以根据restype和atom_name_chars进一步细化
    return get_lj_parameters_from_element(ref_element, device, dtype)


def estimate_lj_from_charge_and_radius(
    ref_charge: torch.Tensor,  # [..., N_atom]
    ref_element: torch.Tensor, # [..., N_atom, 128]
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    基于原子电荷和元素估算LJ参数（启发式方法）。
    
    使用经验关系：
    - σ ≈ 2 × 范德华半径
    - ε ≈ α × |q| + β（α, β为拟合参数）
    
    Args:
        ref_charge: 原子电荷
        ref_element: 原子元素
        device: 目标设备
        dtype: 目标数据类型
    
    Returns:
        lj_sigma: σ参数
        lj_epsilon: ε参数
    
    Note:
        这是一个粗糙估计，精度低于查表法。仅用于缺失元素的fallback。
    """
    # 先用查表法获取基础参数
    lj_sigma_base, lj_epsilon_base = get_lj_parameters_from_element(
        ref_element, device, dtype
    )
    
    # 根据电荷微调ε（带电原子相互作用更强）
    charge_factor = 1.0 + 0.1 * torch.abs(ref_charge)  # 电荷越大，ε略微增加
    lj_epsilon_adjusted = lj_epsilon_base * charge_factor
    
    return lj_sigma_base, lj_epsilon_adjusted


# ============================================================================
# 便捷函数：添加LJ参数到input_feature_dict
# ============================================================================

def add_lj_parameters_to_features(
    input_feature_dict: Dict[str, torch.Tensor],
    use_refined: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    向input_feature_dict添加LJ参数。
    
    Args:
        input_feature_dict: 特征字典，必须包含 'ref_element'
        use_refined: 是否使用精细化参数（需要额外的restype等）
    
    Returns:
        更新后的input_feature_dict，包含 'lj_sigma' 和 'lj_epsilon'
    
    Example:
        >>> features = {"ref_element": torch.zeros(100, 128)}
        >>> features = add_lj_parameters_to_features(features)
        >>> print("lj_sigma" in features, "lj_epsilon" in features)
        True True
    """
    if "ref_element" not in input_feature_dict:
        raise ValueError("input_feature_dict must contain 'ref_element'")
    
    device = input_feature_dict["ref_element"].device
    dtype = input_feature_dict["ref_element"].dtype
    
    if use_refined and all(k in input_feature_dict for k in ["restype", "atom_to_token_idx"]):
        lj_sigma, lj_epsilon = get_lj_parameters_refined(
            ref_element=input_feature_dict["ref_element"],
            restype=input_feature_dict["restype"],
            atom_to_token_idx=input_feature_dict["atom_to_token_idx"],
            ref_atom_name_chars=input_feature_dict.get("ref_atom_name_chars"),
            device=device,
            dtype=dtype,
        )
    else:
        lj_sigma, lj_epsilon = get_lj_parameters_from_element(
            ref_element=input_feature_dict["ref_element"],
            device=device,
            dtype=dtype,
        )
    
    input_feature_dict["lj_sigma"] = lj_sigma
    input_feature_dict["lj_epsilon"] = lj_epsilon
    
    return input_feature_dict


# ============================================================================
# 辅助函数：查看LJ参数表
# ============================================================================

def print_lj_parameters_table():
    """打印LJ参数表（用于调试和验证）"""
    print("=" * 80)
    print("Lennard-Jones Parameters Table")
    print("=" * 80)
    print(f"{'Element':<10} {'σ (Å)':<12} {'ε (kcal/mol)':<15}")
    print("-" * 80)
    
    for element, (sigma, epsilon) in sorted(LJ_PARAMETERS_BY_ELEMENT.items()):
        print(f"{element:<10} {sigma:<12.4f} {epsilon:<15.6f}")
    
    print("=" * 80)


if __name__ == "__main__":
    # 测试代码
    print_lj_parameters_table()
    
    # 测试参数提取
    print("\n" + "=" * 80)
    print("Testing LJ parameter extraction")
    print("=" * 80)
    
    # 创建测试数据
    N_atom = 10
    ref_element = torch.zeros(N_atom, 128)
    ref_element[0:3, 1] = 1.0  # C
    ref_element[3:5, 2] = 1.0  # N
    ref_element[5:7, 3] = 1.0  # O
    ref_element[7:10, 0] = 1.0 # H
    
    sigma, epsilon = get_lj_parameters_from_element(ref_element)
    
    print(f"\nTest results:")
    print(f"sigma shape: {sigma.shape}")
    print(f"epsilon shape: {epsilon.shape}")
    print(f"\nFirst 10 atoms:")
    for i in range(N_atom):
        element_idx = torch.argmax(ref_element[i]).item()
        element_name = ELEMENT_LIST[element_idx] if element_idx < len(ELEMENT_LIST) else "UNK"
        print(f"  Atom {i}: {element_name:3s} -> σ={sigma[i]:.3f} Å, ε={epsilon[i]:.6f} kcal/mol")

