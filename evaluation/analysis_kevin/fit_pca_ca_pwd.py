import os
import argparse
import mdtraj as md
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import joblib
import multiprocessing
import time
import sys
from functools import partial
from scipy.spatial.distance import pdist, squareform

def process_protein(protein_name, root_dir="root_reference", output_dir="pca_results"):
    """
    处理单个蛋白质的轨迹数据：叠加轨迹、计算并保存PCA模型
    """
    try:
        start_time = time.time()
        print(f"[{protein_name}] Starting processing...")
        
        # 定义文件路径
        pdb_path = os.path.join(root_dir, protein_name, f"{protein_name}.pdb")
        traj_paths = [
            os.path.join(root_dir, protein_name, f"{protein_name}_prod_R1_fit.xtc"),
            os.path.join(root_dir, protein_name, f"{protein_name}_prod_R2_fit.xtc"),
            os.path.join(root_dir, protein_name, f"{protein_name}_prod_R3_fit.xtc")
        ]
        
        # 检查文件是否存在
        missing_files = [p for p in [pdb_path] + traj_paths if not os.path.exists(p)]
        if missing_files:
            print(f"[{protein_name}] Missing files: {', '.join(missing_files)}")
            return protein_name, "failed - missing files", 0
        
        # 创建输出目录
        protein_output_dir = os.path.join(output_dir, protein_name)
        os.makedirs(protein_output_dir, exist_ok=True)
        
        # 加载拓扑文件
        top = md.load(pdb_path).topology
        
        # 获取C-alpha原子索引
        ca_indices = top.select("name CA")
        if len(ca_indices) == 0:
            print(f"[{protein_name}] No C-alpha atoms found")
            return protein_name, "failed - no CA atoms", 0
        
        # 存储所有轨迹的成对距离特征
        all_pairwise_features = []
        
        for i, traj_path in enumerate(traj_paths):
            print(f"[{protein_name}] Processing trajectory {i+1}/3: {os.path.basename(traj_path)}")
            
            # 加载轨迹
            traj = md.load(traj_path, top=pdb_path)
            
            # 检查帧数
            if traj.n_frames != 10001:
                print(f"[{protein_name}] Warning: Expected 10001 frames, found {traj.n_frames}")
            
            # 叠加到首帧 (使用C-alpha原子)
            traj_super = traj.superpose(traj, frame=0, atom_indices=ca_indices)
            
            # 提取C-alpha坐标
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
            all_pairwise_features.append(pairwise_features)
        
        # 合并所有轨迹数据
        combined_features = np.vstack(all_pairwise_features)
        print(f"[{protein_name}] Combined data shape: {combined_features.shape}")
        print(f"[{protein_name}] Feature dimension: {combined_features.shape[1]} pairwise distances")
        
        # 创建并训练PCA模型
        pca = PCA(n_components=2)
        pca.fit(combined_features)
        
        # 保存模型
        model_path = os.path.join(protein_output_dir, f"{protein_name}_pca_model.pkl")
        joblib.dump(pca, model_path)
        
        # 保存转换后的坐标
        transformed = pca.transform(combined_features)
        np.save(os.path.join(protein_output_dir, f"{protein_name}_pca_coords.npy"), transformed)
        
        # 保存解释方差
        np.save(os.path.join(protein_output_dir, f"{protein_name}_explained_variance.npy"), 
                pca.explained_variance_ratio_)
        
        # 保存元数据
        metadata = {
            "protein": protein_name,
            "num_frames": combined_features.shape[0],
            "num_ca": len(ca_indices),
            "num_features": combined_features.shape[1],
            "processing_time": time.time() - start_time
        }
        joblib.dump(metadata, os.path.join(protein_output_dir, f"{protein_name}_metadata.pkl"))
        
        print(f"[{protein_name}] Completed in {time.time() - start_time:.2f} seconds")
        return protein_name, "success", combined_features.shape[0]
    
    except Exception as e:
        print(f"[{protein_name}] Error: {str(e)}")
        return protein_name, f"failed - {str(e)}", 0

def process_all_proteins(protein_names, root_dir="root_reference", output_dir="pca_results", max_processes=None):
    """
    使用多进程处理所有蛋白质
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Found {len(protein_names)} proteins to process")
    print(f"Using {max_processes or multiprocessing.cpu_count() // 32} parallel processes")
    
    # 创建进程池
    num_processes = max_processes or multiprocessing.cpu_count() // 32
    pool = multiprocessing.Pool(processes=num_processes)
    
    # 准备参数
    process_func = partial(process_protein, root_dir=root_dir, output_dir=output_dir)
    
    # 进度跟踪
    start_time = time.time()
    results = []
    completed = 0
    total = len(protein_names)
    
    try:
        # 使用imap_unordered获取结果（按完成顺序）
        for i, result in enumerate(pool.imap_unordered(process_func, protein_names)):
            protein_name, status, num_frames = result
            results.append((protein_name, status, num_frames))
            completed = i + 1
            
            # 打印进度
            elapsed = time.time() - start_time
            remaining = (elapsed / completed) * (total - completed) if completed > 0 else 0
            print(f"\nProgress: {completed}/{total} ({completed/total*100:.1f}%) "
                  f"| Elapsed: {elapsed/60:.1f} min | Remaining: {remaining/60:.1f} min")
            print(f"Last completed: {protein_name} - {status}")
            
            # 定期保存进度
            if completed % 5 == 0 or completed == total:
                with open(os.path.join(output_dir, "processing_summary.txt"), "w") as f:
                    f.write("Protein\tStatus\tFrames\n")
                    for r in results:
                        f.write(f"{r[0]}\t{r[1]}\t{r[2]}\n")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user! Terminating processes...")
        pool.terminate()
        sys.exit(1)
    finally:
        pool.close()
        pool.join()
    
    # 最终总结
    success = [r for r in results if r[1] == "success"]
    failed = [r for r in results if r[1] != "success"]
    
    print("\n" + "="*50)
    print(f"Processing completed for {len(protein_names)} proteins")
    print(f"Success: {len(success)}, Failed: {len(failed)}")
    print(f"Total processing time: {(time.time() - start_time)/60:.1f} minutes")
    
    # 保存最终摘要
    summary_path = os.path.join(output_dir, "final_summary.txt")
    with open(summary_path, "w") as f:
        f.write("Protein\tStatus\tFrames\n")
        for r in results:
            f.write(f"{r[0]}\t{r[1]}\t{r[2]}\n")
    
    # 打印失败案例
    if failed:
        print("\nFailed proteins:")
        for protein, status, _ in failed:
            print(f"  {protein}: {status}")

def build_parser():
    p = argparse.ArgumentParser(
        description=(
            "Fit a C-alpha pairwise-distance PCA model per protein from MD "
            "trajectories (the 'pwd' feature space). Writes the same per-protein "
            "artefacts as fit_pca.py into --output-dir."
        )
    )
    p.add_argument("--root-dir", default="/public/home/lingwang/lmj/kevin/alphaflow/atlas",
                   help="Root holding <name>/<name>.pdb and <name>_prod_R{1,2,3}_fit.xtc")
    p.add_argument("--output-dir", default="/public/home/lingwang/lmj/kevin/MD_data/analysis_code/Atlas/pca_models_pwd",
                   help="Where to write the fitted PCA models")
    p.add_argument("--split-csv", default=None,
                   help="CSV whose first column lists the protein names. Overrides --proteins.")
    p.add_argument("--proteins", nargs="*", default=["6l3r_E", "6lrd_A", "6oz1_A"],
                   help="Explicit protein names (default: the three used for the paper figures)")
    p.add_argument("--max-processes", type=int, default=None,
                   help="Worker processes (default: all CPU cores)")
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()

    if args.split_csv is not None:
        protein_names = pd.read_csv(args.split_csv).iloc[:, 0].tolist()
    else:
        protein_names = args.proteins

    # 开始处理
    process_all_proteins(
        protein_names,
        root_dir=args.root_dir,
        output_dir=args.output_dir,
        max_processes=args.max_processes,
    )