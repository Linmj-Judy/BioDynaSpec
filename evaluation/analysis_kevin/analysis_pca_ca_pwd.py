import os
import argparse
import mdtraj as md
import numpy as np
from sklearn.decomposition import PCA
import joblib
from scipy.stats import pearsonr
from scipy.spatial.distance import directed_hausdorff
import multiprocessing
import pandas as pd
import time
import sys
from functools import partial
from scipy.spatial.distance import pdist, squareform


def calculate_bidirectional_hausdorff(u, v):
    """计算两个点集之间的双向Hausdorff距离"""
    forward = directed_hausdorff(u, v)[0]
    backward = directed_hausdorff(v, u)[0]
    return max(forward, backward)

def process_generated_trajectory(pdb_path, pca_model, protein_name, net_name, rep_id):
    """
    处理单个生成轨迹：叠加、PCA投影、计算两种距离指标
    """
    try:
        # 加载生成轨迹 (包含250个MODEL)
        traj = md.load_pdb(pdb_path)
        
        if traj.n_frames != 250:
            print(f"Warning: {pdb_path} has {traj.n_frames} frames, expected 250")
        
        # 获取C-alpha原子索引
        ca_indices = traj.topology.select("name CA")
        if len(ca_indices) == 0:
            print(f"Error: No C-alpha atoms found in {pdb_path}")
            return None
        
        # 叠加到首帧
        traj_super = traj.superpose(traj, frame=0, atom_indices=ca_indices)
        
        # 提取C-alpha坐标并展平
        ca_coords = traj_super.xyz[:, ca_indices, :]
        
        # 计算成对距离特征
        pairwise_features = []
        for frame_coords in ca_coords:
            # 计算所有C-alpha原子之间的成对距离
            dist_matrix = squareform(pdist(frame_coords))
            
            # 取上三角部分（不包括对角线）
            upper_triangle = dist_matrix[np.triu_indices_from(dist_matrix, k=1)]
            pairwise_features.append(upper_triangle)
        
        pairwise_features = np.array(pairwise_features)
        
        # 使用PCA模型投影
        pca_coords = pca_model.transform(pairwise_features)
        
        # 计算相邻帧距离 (欧氏距离)
        stepwise_distances = np.sqrt(np.sum(np.diff(pca_coords, axis=0)**2, axis=1))
        
        # 计算每帧与第一帧的距离
        first_frame = pca_coords[0]
        to_first_distances = np.sqrt(np.sum((pca_coords - first_frame)**2, axis=1))
        
        return {
            'protein': protein_name,
            'net': net_name,
            'rep': rep_id,
            'stepwise_distances': stepwise_distances,
            'to_first_distances': to_first_distances,
            'pca_coords': pca_coords
        }
    
    except Exception as e:
        print(f"Error processing {pdb_path}: {str(e)}")
        return None

def process_real_md_trajectory(real_pca_coords, downsample_factor=40):
    """
    处理真实MD轨迹：下采样、计算两种距离指标
    """
    # 拆分三条重复轨迹
    rep1 = real_pca_coords[:10001]
    rep2 = real_pca_coords[10001:20002]
    rep3 = real_pca_coords[20002:]
    
    # 下采样 (400ps间隔)
    rep1_ds = rep1[::downsample_factor][:250]  # 取前250帧
    rep2_ds = rep2[::downsample_factor][:250]
    rep3_ds = rep3[::downsample_factor][:250]
    
    # 计算相邻帧距离
    stepwise_dist1 = np.sqrt(np.sum(np.diff(rep1_ds, axis=0)**2, axis=1))
    stepwise_dist2 = np.sqrt(np.sum(np.diff(rep2_ds, axis=0)**2, axis=1))
    stepwise_dist3 = np.sqrt(np.sum(np.diff(rep3_ds, axis=0)**2, axis=1))
    
    # 计算每帧与第一帧的距离
    to_first_dist1 = np.sqrt(np.sum((rep1_ds - rep1_ds[0])**2, axis=1))
    to_first_dist2 = np.sqrt(np.sum((rep2_ds - rep2_ds[0])**2, axis=1))
    to_first_dist3 = np.sqrt(np.sum((rep3_ds - rep3_ds[0])**2, axis=1))
    
    # 为Hausdorff距离合并所有距离值
    all_stepwise = np.concatenate([stepwise_dist1, stepwise_dist2, stepwise_dist3])
    all_to_first = np.concatenate([to_first_dist1, to_first_dist2, to_first_dist3])
    
    return {
        'per_rep': {
            'R1': {'stepwise': stepwise_dist1, 'to_first': to_first_dist1},
            'R2': {'stepwise': stepwise_dist2, 'to_first': to_first_dist2},
            'R3': {'stepwise': stepwise_dist3, 'to_first': to_first_dist3}
        },
        'combined': {
            'stepwise': all_stepwise,
            'to_first': all_to_first
        }
    }

def calculate_metrics(gen_distances, real_distances):
    """计算皮尔逊相关系数（要求有序序列）"""
    # 确保长度匹配
    min_len = min(len(gen_distances), len(real_distances))
    gen_distances = gen_distances[:min_len]
    real_distances = real_distances[:min_len]
    
    # 计算皮尔逊相关系数
    pearson_corr, _ = pearsonr(gen_distances, real_distances)
    
    return pearson_corr

def calculate_hausdorff_metrics(gen_distances, real_distances):
    """计算双向Hausdorff距离（不要求有序）"""
    # 准备Hausdorff距离计算的数据
    u = gen_distances.reshape(-1, 1)
    v = real_distances.reshape(-1, 1)
    
    # 计算双向Hausdorff距离
    hausdorff_dist = calculate_bidirectional_hausdorff(u, v)
    
    return hausdorff_dist

def analyze_protein(protein_name, pca_models_dir, real_data_dir, net_dirs):
    """
    分析单个蛋白质：加载PCA模型，处理所有生成轨迹，计算两种指标
    """
    try:
        # 加载PCA模型和真实MD的PCA坐标
        model_path = os.path.join(pca_models_dir, protein_name, f"{protein_name}_pca_model.pkl")
        real_coords_path = os.path.join(real_data_dir, protein_name, f"{protein_name}_pca_coords.npy")
        
        if not os.path.exists(model_path) or not os.path.exists(real_coords_path):
            print(f"Missing data for {protein_name}")
            return None
        
        pca_model = joblib.load(model_path)
        real_pca_coords = np.load(real_coords_path)
        
        # 处理真实MD轨迹
        real_results = process_real_md_trajectory(real_pca_coords)
        
        # 存储所有结果
        all_results = []
        
        # 处理每个网络的生成轨迹
        for net_name, net_dir in net_dirs.items():
            # 存储当前网络的指标
            net_stepwise_pearsons = []
            net_to_first_pearsons = []
            net_stepwise_distances = []
            net_to_first_distances = []
            
            for rep_id in ['R1',]: # 'R2', 'R3']:
                if net_name == "specpd":
                    pdb_path = os.path.join(net_dir[rep_id], f"protein_{protein_name}_R1_reconstructed.pdb")  # protein_5znj_A_R1_reconstructed.pdb
                elif net_name == "alphafolding":
                    pdb_path = os.path.join(net_dir[rep_id], f"{protein_name}_time_3.pdb")
                else:
                    pdb_path = os.path.join(net_dir[rep_id], f"{protein_name}.pdb")

                if not os.path.exists(pdb_path):
                    print(f"Missing generated trajectory: {pdb_path}")
                    continue
                
                # 处理生成轨迹
                gen_result = process_generated_trajectory(
                    pdb_path, pca_model, protein_name, net_name, rep_id
                )
                
                if gen_result is None:
                    continue
                
                # 获取对应的真实MD距离序列（相同重复）
                real_rep = real_results['per_rep'][rep_id]
                
                # 计算相邻帧距离的皮尔逊相关系数
                stepwise_pearson = calculate_metrics(
                    gen_result['stepwise_distances'], 
                    real_rep['stepwise']
                )
                
                # 计算与第一帧距离的皮尔逊相关系数
                to_first_pearson = calculate_metrics(
                    gen_result['to_first_distances'], 
                    real_rep['to_first']
                )
                
                # 收集结果
                net_stepwise_pearsons.append(stepwise_pearson)
                net_to_first_pearsons.append(to_first_pearson)
                
                # 收集距离值用于Hausdorff计算
                net_stepwise_distances.append(gen_result['stepwise_distances'])
                net_to_first_distances.append(gen_result['to_first_distances'])
            
            # 计算平均皮尔逊相关系数
            avg_stepwise_pearson = np.mean(net_stepwise_pearsons) if net_stepwise_pearsons else np.nan
            avg_to_first_pearson = np.mean(net_to_first_pearsons) if net_to_first_pearsons else np.nan
            
            # 计算Hausdorff距离（合并所有重复）
            if net_stepwise_distances:
                combined_stepwise = np.concatenate(net_stepwise_distances)
                stepwise_hausdorff = calculate_hausdorff_metrics(
                    combined_stepwise,
                    real_results['combined']['stepwise']
                )
                
                combined_to_first = np.concatenate(net_to_first_distances)
                to_first_hausdorff = calculate_hausdorff_metrics(
                    combined_to_first,
                    real_results['combined']['to_first']
                )
            else:
                stepwise_hausdorff = np.nan
                to_first_hausdorff = np.nan
            
            # 添加结果（每个蛋白质每个网络一个条目）
            result_entry = {
                'protein': protein_name,
                'net': net_name,
                'stepwise_pearson': avg_stepwise_pearson,
                'stepwise_hausdorff': stepwise_hausdorff,
                'to_first_pearson': avg_to_first_pearson,
                'to_first_hausdorff': to_first_hausdorff
            }
            
            all_results.append(result_entry)
        
        return all_results
    
    except Exception as e:
        print(f"Error analyzing {protein_name}: {str(e)}")
        return None

def main_analysis(pca_models_dir, real_data_dir, net_dirs, output_dir, max_processes=None,
                  split_csv="/public/home/lingwang/lmj/kevin/MD_data/split/atlas_test.csv"):
    """
    主分析流程：并行处理所有蛋白质
    """
    protein_names = pd.read_csv(split_csv).iloc[:, 0]
    
    print(f"Found {len(protein_names)} proteins to analyze")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 准备参数
    analyze_func = partial(
        analyze_protein,
        pca_models_dir=pca_models_dir,
        real_data_dir=real_data_dir,
        net_dirs=net_dirs
    )
    
    # 设置并行处理
    num_processes = max_processes or multiprocessing.cpu_count() // 2
    pool = multiprocessing.Pool(processes=num_processes)
    
    # 进度跟踪
    results = []
    completed = 0
    total = len(protein_names)
    start_time = time.time()
    
    print(f"Starting analysis with {num_processes} processes...")
    
    try:
        # 处理所有蛋白质
        for i, protein_results in enumerate(pool.imap_unordered(analyze_func, protein_names)):
            completed = i + 1
            
            if protein_results is not None:
                results.extend(protein_results)
                print(f"Completed {completed}/{total}: {protein_results[0]['protein']} "
                      f"({len(protein_results)} networks processed)")
            else:
                print(f"Completed {completed}/{total}: (no results)")
            
            # 定期保存中间结果
            if completed % 5 == 0 or completed == total:
                save_results(results, output_dir)
                
            # 打印进度
            elapsed = time.time() - start_time
            avg_time_per_protein = elapsed / completed if completed > 0 else 0
            remaining_time = avg_time_per_protein * (total - completed)
            print(f"Progress: {completed}/{total} ({completed/total*100:.1f}%) | "
                  f"Elapsed: {elapsed/60:.1f} min | Remaining: {remaining_time/60:.1f} min")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user! Terminating processes...")
        pool.terminate()
    finally:
        pool.close()
        pool.join()
    
    # 保存最终结果
    save_results(results, output_dir)
    print_analysis_summary(results, output_dir)

def save_results(results, output_dir):
    """保存结果到CSV文件"""
    if not results:
        return
    
    # 转换为DataFrame
    df = pd.DataFrame(results)
    
    # 保存完整结果
    full_path = os.path.join(output_dir, "full_analysis_results.csv")
    df.to_csv(full_path, index=False)
    
    # 保存简化结果
    if not df.empty:
        summary_path = os.path.join(output_dir, "summary_results.csv")
        df.to_csv(summary_path, index=False)

def print_analysis_summary(results, output_dir):
    """打印并保存分析摘要"""
    if not results:
        print("No results to summarize")
        return
    
    df = pd.DataFrame(results)
    
    # 按网络计算总体平均
    net_summary = df.groupby('net').agg({
        'stepwise_pearson': 'mean',
        'stepwise_hausdorff': 'mean',
        'to_first_pearson': 'mean',
        'to_first_hausdorff': 'mean'
    }).reset_index()
    
    # 按蛋白质和网络分组计算平均指标
    protein_net_summary = df.groupby(['protein', 'net']).agg({
        'stepwise_pearson': 'mean',
        'stepwise_hausdorff': 'mean',
        'to_first_pearson': 'mean',
        'to_first_hausdorff': 'mean'
    }).reset_index()
    
    # 按蛋白质计算平均 (跨网络)
    protein_summary = df.groupby('protein').agg({
        'stepwise_pearson': 'mean',
        'stepwise_hausdorff': 'mean',
        'to_first_pearson': 'mean',
        'to_first_hausdorff': 'mean'
    }).reset_index()
    
    # 保存所有摘要
    summary_path = os.path.join(output_dir, "analysis_summary.txt")
    with open(summary_path, "w") as f:
        # 按网络的结果
        f.write("=== Overall Network Performance ===\n")
        f.write(net_summary.to_string(index=False))
        f.write("\n\n")
        
        # 按蛋白质和网络的结果
        f.write("=== Per Protein and Network Metrics ===\n")
        f.write(protein_net_summary.to_string(index=False))
        f.write("\n\n")
        
        # 按蛋白质的结果
        f.write("=== Per Protein Metrics (Average across nets) ===\n")
        f.write(protein_summary.to_string(index=False))
        f.write("\n")
    
    # 打印关键结果
    print("\n" + "="*60)
    print("Analysis Summary:")
    
    # 按网络总结
    print("\nOverall Network Performance:")
    for _, row in net_summary.iterrows():
        net = row['net']
        print(f"\nNetwork: {net}")
        print(f"  Stepwise Pearson: {row['stepwise_pearson']:.4f} (average over reps)")
        print(f"  Stepwise Hausdorff: {row['stepwise_hausdorff']:.4f}")
        print(f"  To-First Pearson: {row['to_first_pearson']:.4f} (average over reps)")
        print(f"  To-First Hausdorff: {row['to_first_hausdorff']:.4f}")
    
    # 按指标排序的蛋白质表现
    print("\nTop 5 Proteins by Stepwise Pearson:")
    top_stepwise = protein_summary.sort_values('stepwise_pearson', ascending=False).head(5)
    print(top_stepwise[['protein', 'stepwise_pearson']])
    
    print("\nTop 5 Proteins by To-First Pearson:")
    top_to_first = protein_summary.sort_values('to_first_pearson', ascending=False).head(5)
    print(top_to_first[['protein', 'to_first_pearson']])
    
    print("\nTop 5 Proteins by Stepwise Hausdorff (lower is better):")
    top_hausdorff_stepwise = protein_summary.sort_values('stepwise_hausdorff').head(5)
    print(top_hausdorff_stepwise[['protein', 'stepwise_hausdorff']])
    
    print("\nTop 5 Proteins by To-First Hausdorff (lower is better):")
    top_hausdorff_to_first = protein_summary.sort_values('to_first_hausdorff').head(5)
    print(top_hausdorff_to_first[['protein', 'to_first_hausdorff']])
    
    print(f"\nFull results saved to: {output_dir}")


DEFAULT_NET_DIRS = {
    "alphafolding": {
        "R1": "/public/home/lingwang/lmj/kevin/file_trans/SpecPD/alphafolding_atlas_extension_pdb",
    },
    "specpd": {
        "R1": "/public/home/lingwang/lmj/kevin/file_trans/SpecPD/reconstructed_pdbs_atlas",
    },
    "mdgen": {
        "R1": "/public/home/lingwang/lmj/kevin/file_trans/mdgen_results/atlas_R1",
        "R2": "/public/home/lingwang/lmj/kevin/file_trans/mdgen_results/atlas_R2",
        "R3": "/public/home/lingwang/lmj/kevin/file_trans/mdgen_results/atlas_R3",
    },
}


def build_parser():
    p = argparse.ArgumentParser(
        description=(
            "Pearson correlation and bidirectional Hausdorff distance between the "
            "reference MD ensemble and each generated ensemble, computed in the 2D "
            "PCA projection of the C-alpha pairwise-distance feature space. Writes "
            "CSVs into --output-dir."
        )
    )
    p.add_argument("--pca-models-dir",
                   default="/public/home/lingwang/lmj/kevin/MD_data/analysis_code/Atlas/pca_models_pwd",
                   help="Directory of per-protein pwd PCA models (from fit_pca_ca_pwd.py)")
    p.add_argument("--real-data-dir",
                   default="/public/home/lingwang/lmj/kevin/MD_data/analysis_code/Atlas/pca_models_pwd",
                   help="Directory holding the per-protein reference PCA coordinates")
    p.add_argument("--output-dir", default="pca_analysis_results_yihengtao_ca_pwd",
                   help="Where to write the analysis CSVs")
    p.add_argument("--split-csv", default="/public/home/lingwang/lmj/kevin/MD_data/split/atlas_test.csv",
                   help="CSV whose first column lists the protein names to analyze")
    p.add_argument("--max-processes", type=int, default=None,
                   help="Worker processes (default: all CPU cores)")
    p.add_argument("--net", action="append", default=None,
                   metavar="NAME=PATH[,REP=PATH...]",
                   help=("Generated ensemble to evaluate, repeatable. Format: "
                         "'mdgen=R1:/path,R2:/path'. Defaults to the alphafolding and "
                         "specpd layouts used for the paper figures."))
    return p


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

    # ===== 执行分析 =====
    main_analysis(
        pca_models_dir=args.pca_models_dir,
        real_data_dir=args.real_data_dir,
        net_dirs=_parse_net_dirs(args.net),
        output_dir=args.output_dir,
        max_processes=args.max_processes,
        split_csv=args.split_csv,
    )