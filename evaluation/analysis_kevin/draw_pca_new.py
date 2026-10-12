import os
import argparse
import mdtraj as md
import numpy as np
import joblib
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
import multiprocessing
import pandas as pd
import time
import sys
from functools import partial
import matplotlib as mpl

try:
    import seaborn as sns
except ModuleNotFoundError as exc:  # declared in requirements.txt
    raise ModuleNotFoundError(
        "draw_pca_new.py needs seaborn for its KDE plots; install it with "
        "`pip install seaborn` (it is pinned in requirements.txt)."
    ) from exc
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# ==========================================
# 全局绘图风格设置
# ==========================================
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
mpl.rcParams["font.size"] = 20
mpl.rcParams["axes.linewidth"] = 1.5
mpl.rcParams["xtick.major.width"] = 1.5
mpl.rcParams["ytick.major.width"] = 1.5
mpl.rcParams["figure.dpi"] = 150
mpl.rcParams["savefig.dpi"] = 300


def load_pca_model(protein_name, pca_dir, feature_type):
    """加载PCA模型和真实MD轨迹的投影坐标"""
    model_path = os.path.join(pca_dir, protein_name, f"{protein_name}_pca_model.pkl")
    coords_path = os.path.join(pca_dir, protein_name, f"{protein_name}_pca_coords.npy")

    if not os.path.exists(model_path) or not os.path.exists(coords_path):
        print(f"Missing PCA data for {protein_name} ({feature_type})")
        return None, None

    pca_model = joblib.load(model_path)
    real_pca_coords = np.load(coords_path)
    return pca_model, real_pca_coords


def process_generated_trajectory(pdb_path, pca_model, feature_type):
    """处理单个生成轨迹：叠加、特征提取、PCA投影"""
    try:
        traj = md.load_pdb(pdb_path)

        # 简单检查帧数，不强制退出，避免个别文件错误中断整个流程
        if traj.n_frames != 250:
            print(f"Note: {pdb_path} has {traj.n_frames} frames (expected 250)")

        ca_indices = traj.topology.select("name CA")
        if len(ca_indices) == 0:
            print(f"Error: No C-alpha atoms found in {pdb_path}")
            return None

        # 叠加到首帧
        traj_super = traj.superpose(traj, frame=0, atom_indices=ca_indices)
        ca_coords = traj_super.xyz[:, ca_indices, :]

        if feature_type == "coord":
            n_frames, n_ca, _ = ca_coords.shape
            features = ca_coords.reshape(n_frames, -1)
        elif feature_type == "pairwise":
            features = []
            for frame_coords in ca_coords:
                dist_matrix = squareform(pdist(frame_coords))
                upper_triangle = dist_matrix[np.triu_indices_from(dist_matrix, k=1)]
                features.append(upper_triangle)
            features = np.array(features)
        else:
            raise ValueError(f"Unknown feature type: {feature_type}")

        pca_coords = pca_model.transform(features)
        return pca_coords

    except Exception as e:
        print(f"Error processing {pdb_path}: {str(e)}")
        return None


def plot_combined_pca_results(
    protein_name,
    rep_id,
    real_pca_coords,
    net1_coords,
    net2_coords,
    feature_type,
    output_dir,
):
    """
    绘制核心可视化图：使用等高线 (KDE) 对比分布
    net1: MDGen (对比组)
    net2: FrameAR (Ours, ProAR)
    """
    if net1_coords is None or net2_coords is None:
        return None

    # 设置 Seaborn 风格
    sns.set_style(
        "whitegrid", {"axes.grid": True, "grid.linestyle": "--", "grid.alpha": 0.5}
    )

    # 创建画布
    fig, ax = plt.subplots(figsize=(9, 8))

    # =========================================================
    # 1. 绘制 GT (Ground Truth) 背景 - 蓝色填充
    # =========================================================
    # 使用较浅的蓝色填充，避免干扰前景
    try:
        sns.kdeplot(
            x=real_pca_coords[:, 0],
            y=real_pca_coords[:, 1],
            fill=True,
            cmap="Blues",
            levels=10,
            thresh=0.05,  # 过滤掉低密度区域，保持画面干净
            alpha=0.7,
            ax=ax,
        )
        # 添加一条极细的边界线增强轮廓
        sns.kdeplot(
            x=real_pca_coords[:, 0],
            y=real_pca_coords[:, 1],
            fill=False,
            color="#1f77b4",
            levels=10,
            thresh=0.05,
            linewidths=0.5,
            alpha=0.5,
            ax=ax,
        )
    except Exception as e:
        print(f"Warning: Failed to plot GT KDE for {protein_name}: {e}")

    # =========================================================
    # 2. 绘制 MDGEN (对比模型) - 橙色虚线
    # =========================================================
    # 颜色：橙色 (Orange)，代表局限性或对比
    c_mdgen = "#ff7f0e"
    try:
        sns.kdeplot(
            x=net1_coords[:, 0],
            y=net1_coords[:, 1],
            fill=False,
            color=c_mdgen,
            levels=5,  # 层数少一点，清晰
            thresh=0.1,
            linewidths=2,
            linestyles="--",  # 虚线区分
            ax=ax,
        )
    except ValueError:
        # 如果模型生成的点过于集中导致方差为0，KDE会报错，此时改画散点
        ax.scatter(
            net1_coords[:, 0],
            net1_coords[:, 1],
            c=c_mdgen,
            s=10,
            alpha=0.5,
            label="MDGen (Collapsed)",
        )

    # =========================================================
    # 3. 绘制 FrameAR/ProAR (Ours) - 红色/洋红实线
    # =========================================================
    # 颜色：红色 (Red) 或 深品红，代表 Ours，强调覆盖广
    c_proar = "#d62728"
    try:
        sns.kdeplot(
            x=net2_coords[:, 0],
            y=net2_coords[:, 1],
            fill=False,
            color=c_proar,
            levels=6,
            thresh=0.1,
            linewidths=2.5,
            ax=ax,
        )
    except ValueError:
        ax.scatter(net2_coords[:, 0], net2_coords[:, 1], c=c_proar, s=10, alpha=0.5)

    # =========================================================
    # 4. 标记起始结构 (Initial Structure)
    # =========================================================
    # 假设所有生成轨迹都从同一个晶体结构或首帧出发
    start_x, start_y = net1_coords[0, 0], net1_coords[0, 1]
    ax.scatter(
        start_x,
        start_y,
        c="black",
        s=240,
        marker="X",
        edgecolors="white",
        linewidth=1.5,
        zorder=10,
    )

    # =========================================================
    # 5. 美化与图例
    # =========================================================
    # 设置坐标轴标签
    ax.set_xlabel("PC1", fontsize=36, fontweight="bold")
    ax.set_ylabel("PC2", fontsize=36, fontweight="bold")

    ax.set_xticks([])
    ax.set_yticks([])

    # 动态调整坐标轴范围 (紧凑不留白模式)
    # =========================================================
    all_x = np.concatenate(
        [real_pca_coords[:, 0], net1_coords[:, 0], net2_coords[:, 0]]
    )
    all_y = np.concatenate(
        [real_pca_coords[:, 1], net1_coords[:, 1], net2_coords[:, 1]]
    )

    # --- 方案 A (推荐)：严格贴合数据边缘 ---
    # 直接使用数据的最小值和最大值，不做任何缓冲
    # 这样图的边缘会正好切在最远的那个点上
    ax.set_xlim(all_x.min(), all_x.max())
    ax.set_ylim(all_y.min(), all_y.max())

    # 标题 (可选，正式论文通常不需要图内标题，这里先加上方便查看)
    # ax.set_title(f"{protein_name}", fontsize=18, pad=10)

    # 自定义图例句柄
    legend_elements = [
        # GT
        Patch(
            facecolor=sns.color_palette("Blues")[3],
            edgecolor="none",
            alpha=0.6,
            label="GT MD Simulation",
        ),
        # ProAR (Ours)
        Line2D([0], [0], color=c_proar, lw=2.5, label="ProAR (Ours)"),
        # MDGen
        Line2D([0], [0], color=c_mdgen, lw=2, linestyle="--", label="MDGEN"),
        # Start
        Line2D(
            [0],
            [0],
            marker="X",
            color="w",
            markerfacecolor="black",
            markersize=20,
            label="Initial Structure",
        ),
    ]

    # 添加图例
    ax.legend(
        handles=legend_elements,
        loc="upper right",
        frameon=True,
        framealpha=0.9,
        fontsize=34,
    )

    # 保存图片
    plot_dir = os.path.join(output_dir, protein_name)
    os.makedirs(plot_dir, exist_ok=True)

    filename = f"{protein_name}_{rep_id}_{feature_type}.png"
    # 同时保存 PDF 以获得矢量图质量
    filename_pdf = f"{protein_name}_{rep_id}_{feature_type}.pdf"

    plot_path = os.path.join(plot_dir, filename)
    plt.tight_layout()
    plt.savefig(plot_path, bbox_inches="tight")
    plt.savefig(os.path.join(plot_dir, filename_pdf), bbox_inches="tight")
    plt.close(fig)

    return plot_path


def process_protein_rep(protein_name, rep_id, pca_dirs, net_dirs, output_dir):
    """处理单个蛋白质-重复的组合"""
    results = []

    for feature_type, pca_dir in pca_dirs.items():
        pca_model, real_pca_coords = load_pca_model(protein_name, pca_dir, feature_type)
        if pca_model is None or real_pca_coords is None:
            continue

        net1_path = os.path.join(net_dirs["mdgen"][rep_id], f"{protein_name}.pdb")
        net2_path = os.path.join(net_dirs["framear"][rep_id], f"{protein_name}.pdb")

        # 检查文件是否存在
        if not (os.path.exists(net1_path) and os.path.exists(net2_path)):
            # print(f"Missing PDB files for {protein_name}")
            continue

        net1_coords = process_generated_trajectory(net1_path, pca_model, feature_type)
        net2_coords = process_generated_trajectory(net2_path, pca_model, feature_type)

        plot_path = plot_combined_pca_results(
            protein_name,
            rep_id,
            real_pca_coords,
            net1_coords,
            net2_coords,
            feature_type,
            output_dir,
        )

        if plot_path:
            results.append(
                {
                    "protein": protein_name,
                    "rep": rep_id,
                    "feature": feature_type,
                    "plot_path": plot_path,
                }
            )

    return results


def main_visualization(pca_dirs, net_dirs, output_dir, max_processes=None):
    """主可视化流程"""
    # 此处你可以修改为你需要处理的蛋白质列表
    # protein_names = pd.read_csv(
    #     "/public/home/lingwang/lmj/kevin/MD_data/split/atlas_test.csv"
    # ).iloc[:, 0]
    protein_names = [
        "7nmq_A",
        "6iah_A",
        "5znj_A",
        "7mf4_A",
        "6tgk_C",
        "6pnv_A",
        "6ovk_R",
        "6e7e_A",
        "6q9c_A",
        "6h86_A",
        "7jfl_C",
        "7c45_A",
        "7asg_A",
        "7aqx_A",
        "7a66_B",
        "6jpt_A",
    ]

    print(f"Found {len(protein_names)} proteins to visualize")
    os.makedirs(output_dir, exist_ok=True)

    tasks = []
    for protein_name in protein_names:
        for rep_id in ["R1",]: # "R2", "R3"]:
            tasks.append((protein_name, rep_id))

    num_processes = max_processes or multiprocessing.cpu_count()
    # 限制最大进程数，防止绘图时内存溢出
    num_processes = min(num_processes, 10)

    pool = multiprocessing.Pool(processes=num_processes)
    process_func = partial(
        process_protein_rep, pca_dirs=pca_dirs, net_dirs=net_dirs, output_dir=output_dir
    )

    results = []
    start_time = time.time()

    print(f"Starting visualization with {num_processes} processes...")
    try:
        for i, task_results in enumerate(pool.starmap(process_func, tasks)):
            if task_results:
                results.extend(task_results)

            if (i + 1) % 5 == 0:
                print(f"Progress: {i + 1}/{len(tasks)} tasks completed.")

    except KeyboardInterrupt:
        pool.terminate()
    finally:
        pool.close()
        pool.join()

    # 保存简单统计
    if results:
        df = pd.DataFrame(results)
        df.to_csv(os.path.join(output_dir, "visualization_summary.csv"), index=False)
        print(f"Done. Generated {len(df)} plots in {output_dir}")


DEFAULT_PCA_DIRS = {
    "coord": "/public/home/lingwang/lmj/kevin/MD_data/analysis_code/Atlas/pca_models",
}

DEFAULT_NET_DIRS = {
    "mdgen": {
        "R1": "/public/home/lingwang/lmj/kevin/file_trans/mdgen_results/atlas_R1",
        "R2": "/public/home/lingwang/lmj/kevin/file_trans/mdgen_results/atlas_R2",
        "R3": "/public/home/lingwang/lmj/kevin/file_trans/mdgen_results/atlas_R3",
    },
    "framear": {
        "R1": "/public/home/lingwang/lmj/kevin/file_trans/2025-07-18-p2dflow-R1",
        "R2": "/public/home/lingwang/lmj/kevin/file_trans/2025-07-21-p2dflow-R2",
        "R3": "/public/home/lingwang/lmj/kevin/file_trans/2025-07-21-p2dflow-R3",
    },
}


def build_parser():
    p = argparse.ArgumentParser(
        description=(
            "Draw KDE visualizations of the 2D PCA projection for each protein, "
            "comparing the reference MD ensemble against each generated ensemble."
        )
    )
    p.add_argument("--pca-dir", action="append", default=None,
                   metavar="FEATURE=PATH",
                   help=("Per-protein PCA model dir, repeatable. Formats: "
                         "'coord=/path' (Cartesian) or 'pairwise=/path' (C-alpha pairwise "
                         "distance). Defaults to the Cartesian models used for the paper."))
    p.add_argument("--net", action="append", default=None,
                   metavar="NAME=R1:/path[,R2:/path]",
                   help=("Generated ensemble to visualize, repeatable. Defaults to the "
                         "mdgen and framear layouts used for the paper figures."))
    p.add_argument("--output-dir", default="pca_visualizations_new",
                   help="Where to write the figures and visualization_summary.csv")
    p.add_argument("--max-processes", type=int, default=2,
                   help="Worker processes")
    return p


def _parse_pca_dirs(specs):
    """Turn ['coord=/a', 'pairwise=/b'] into {'coord': '/a', 'pairwise': '/b'}."""
    if not specs:
        return DEFAULT_PCA_DIRS
    out = {}
    for spec in specs:
        feature, _, path = spec.partition("=")
        if not path:
            raise ValueError(f"--pca-dir expects FEATURE=PATH, got {spec!r}")
        out[feature] = path
    return out


def _parse_net_dirs(specs):
    """Turn ['mdgen=R1:/a,R2:/b', ...] into {'mdgen': {'R1': '/a', 'R2': '/b'}}."""
    if not specs:
        return DEFAULT_NET_DIRS
    net_dirs = {}
    for spec in specs:
        name, _, rest = spec.partition("=")
        if not rest:
            raise ValueError(f"--net expects NAME=R1:/path[,R2:/path], got {spec!r}")
        reps = {}
        for item in rest.split(","):
            rep, _, path = item.partition(":")
            if not path:
                raise ValueError(f"replica entry must be R1:/path, got {item!r}")
            reps[rep] = path
        net_dirs[name] = reps
    return net_dirs


if __name__ == "__main__":
    args = build_parser().parse_args()
    main_visualization(
        _parse_pca_dirs(args.pca_dir),
        _parse_net_dirs(args.net),
        args.output_dir,
        args.max_processes,
    )
