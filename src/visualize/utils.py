import os
import matplotlib.pyplot as plt
import numpy as np
import re
from pathlib import Path


def plot_far_results(results, save_dir):
    """Plot FAR results charts"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Create charts
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. FAR RMSE distribution
    axes[0, 0].hist(results['far_rmse'], bins=30, alpha=0.7, color='blue')
    axes[0, 0].set_title('FAR Reconstruction RMSE Distribution')
    axes[0, 0].set_xlabel('RMSE (Å)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].axvline(np.mean(results['far_rmse']), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(results["far_rmse"]):.4f}Å')
    axes[0, 0].legend()
    
    # 2. FAR MAE distribution
    axes[0, 1].hist(results['far_mae'], bins=30, alpha=0.7, color='green')
    axes[0, 1].set_title('FAR Reconstruction MAE Distribution')
    axes[0, 1].set_xlabel('MAE (Å)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].axvline(np.mean(results['far_mae']), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(results["far_mae"]):.4f}Å')
    axes[0, 1].legend()
    
    # 3. FAR correlation coefficient distribution
    axes[0, 2].hist(results['far_correlation'], bins=30, alpha=0.7, color='orange')
    axes[0, 2].set_title('FAR Reconstruction Correlation Distribution')
    axes[0, 2].set_xlabel('Correlation')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].axvline(np.mean(results['far_correlation']), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(results["far_correlation"]):.4f}')
    axes[0, 2].legend()
    
    # 4. Protein size vs reconstruction RMSE scatter plot
    num_residues = results['per_protein_metrics']['num_residues']
    protein_rmse = results['per_protein_metrics']['rmse']
    axes[1, 0].scatter(num_residues, protein_rmse, alpha=0.6, color='purple')
    axes[1, 0].set_title('Protein Size vs FAR Reconstruction RMSE')
    axes[1, 0].set_xlabel('Number of Residues')
    axes[1, 0].set_ylabel('RMSE (Å)')
    
    # 5. Time distribution histogram
    axes[1, 1].hist(results['timing']['total_times'], bins=20, alpha=0.7, color='red')
    axes[1, 1].set_title('Total Processing Time Distribution')
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].axvline(np.mean(results['timing']['total_times']), color='blue', linestyle='--', 
                       label=f'Mean: {np.mean(results["timing"]["total_times"]):.3f}s')
    axes[1, 1].legend()
    
    # 6. Time distribution pie chart by processing step
    far_time = np.mean(results['timing']['far_sampling_times'])
    trajectory_time = np.mean(results['timing']['trajectory_reconstruction_times'])
    metrics_time = np.mean(results['timing']['metrics_computation_times'])
    
    times = [far_time, trajectory_time, metrics_time]
    labels = ['FAR Sampling', 'Trajectory Reconstruction', 'Metrics Computation']
    colors = ['#ff9999', '#99ff99', '#ffcc99']
    
    axes[1, 2].pie(times, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    axes[1, 2].set_title('Time Distribution by Processing Step')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'far_reconstruction_results.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"FAR results chart saved to {os.path.join(save_dir, 'far_reconstruction_results.png')}")

def plot_loss_curves_from_ckpt_dir(DIR):
    """Plot loss curves from checkpoint directory, the directory should be like this: 
    /path/to/runs/checkpoints/protein_far-epoch=02-train_loss=0.0000-val_loss=0.0000.ckpt
    /path/to/runs/checkpoints/protein_far-epoch=03-train_loss=0.0000-val_loss=0.0000.ckpt
    ...
    """
    # 匹配文件名的正则：
    # 允许 epoch 为整数，loss 为浮点数；文件名前后缀保持灵活一些
    pattern = re.compile(
        r"protein_far-(\d+)-([0-9]+(?:\.[0-9]+)?)-val([0-9]+(?:\.[0-9]+)?|none)\.ckpt$"
    )

    epochs = []
    train_losses = []
    val_losses = []

    p = Path(DIR)
    for f in p.iterdir():
        if not f.is_file():
            continue
        m = pattern.search(f.name)
        if m:
            epoch = int(m.group(1))
            train = float(m.group(2))
            val_str = m.group(3)
            val = float("nan") if val_str == "none" else float(val_str)
            epochs.append(epoch)
            train_losses.append(train)
            val_losses.append(val)

    # 检查是否找到数据
    if not epochs:
        print(f"⚠️  在目录 {DIR} 下没有匹配到符合命名规则的 .ckpt 文件，跳过绘图")
        return

    # 按 epoch 排序
    data = sorted(zip(epochs, train_losses, val_losses), key=lambda x: x[0])
    epochs, train_losses, val_losses = zip(*data)

    # 画图
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, marker='o', label='Train Loss')
    plt.plot(epochs, val_losses, marker='o', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss vs. Epoch')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    # 保存与显示
    out_png = os.path.join(os.path.dirname(DIR), "plots", "loss_curves.png")
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    
    plt.savefig(out_png, dpi=600, bbox_inches='tight')
    print(f"曲线图已保存到: {out_png}")
