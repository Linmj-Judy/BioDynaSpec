
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

# ============================================================
# 1. 加载你的数据
# ============================================================
print("="*70)
print("加载频谱数据")
print("="*70)

data = np.load('./my_protein_spectra_global.npz')

frequencies = data['frequencies']  # Hz
D_true = data['displacement_spectrum']  # Å
V_true = data['velocity_spectrum']  # Å/ps

# 转换为合适的频率范围（只取前一半，避免Nyquist以上）
n_freq = len(frequencies) // 2
frequencies = frequencies[:n_freq]
D_true = D_true[:n_freq]
V_true = V_true[:n_freq]

print(f"✅ 数据shape: {D_true.shape}")
print(f"   频率范围: {frequencies[0]/1e9:.4f} - {frequencies[-1]/1e9:.2f} GHz")

# ============================================================
# 2. 构造虚拟输入特征（模拟真实场景）
# ============================================================
# 在真实应用中，这可能是蛋白质的结构特征、氨基酸序列等
# 这里我们用频率本身作为特征（只是为了测试网络）

# 构造特征：频率 + 一些简单的变换
def create_features(freqs):
    """构造输入特征"""
    freqs_normalized = freqs / freqs.max()
    features = np.column_stack([
        freqs_normalized,
        np.log10(freqs + 1),
        np.sin(2 * np.pi * freqs_normalized),
        np.cos(2 * np.pi * freqs_normalized),
    ])
    return features

features = create_features(frequencies)
print(f"✅ 特征shape: {features.shape}")

# ============================================================
# 3. 定义数据集
# ============================================================
class SpectrumDataset(Dataset):
    def __init__(self, features, displacement, velocity):
        self.features = torch.FloatTensor(features)
        self.displacement = torch.FloatTensor(displacement)
        self.velocity = torch.FloatTensor(velocity)
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.displacement[idx], self.velocity[idx]

dataset = SpectrumDataset(features, D_true, V_true)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

# ============================================================
# 4. 定义神经网络（简单的3层MLP）
# ============================================================
class SimpleSpectrumPredictor(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64):
        super(SimpleSpectrumPredictor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        return self.network(x).squeeze()

# ============================================================
# 5. 训练函数
# ============================================================
def train_model(model, dataloader, target_type='displacement', n_epochs=100, lr=0.001):
    """
    训练模型
    
    参数:
        model: 神经网络
        dataloader: 数据加载器
        target_type: 'displacement' 或 'velocity'
        n_epochs: 训练轮数
        lr: 学习率
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    losses = []
    
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        for features_batch, disp_batch, vel_batch in dataloader:
            optimizer.zero_grad()
            
            # 前向传播
            pred = model(features_batch)
            
            # 选择目标
            if target_type == 'displacement':
                target = disp_batch
            else:
                target = vel_batch
            
            # 计算损失
            loss = criterion(pred, target)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{n_epochs}], Loss: {avg_loss:.6e}")
    
    return losses

# ============================================================
# 6. 评估函数
# ============================================================
def evaluate_model(model, features, D_true, V_true, frequencies, target_type='displacement'):
    """
    评估模型性能
    
    参数:
        model: 训练好的模型
        features: 输入特征
        D_true: 真实位移谱
        V_true: 真实速度谱
        frequencies: 频率
        target_type: 训练时的目标类型
    
    返回:
        D_pred: 预测的位移谱
        errors: 各种误差指标
    """
    model.eval()
    with torch.no_grad():
        features_tensor = torch.FloatTensor(features)
        pred = model(features_tensor).numpy()
    
    if target_type == 'displacement':
        # 方案B: 直接预测位移
        D_pred = pred
    else:
        # 方案A: 预测速度，然后还原位移
        V_pred = pred
        omega = 2 * np.pi * frequencies
        # 避免除以零
        omega = np.where(omega < 1e-6, 1e-6, omega)
        D_pred = V_pred / omega
    
    # 计算各种误差
    errors = {}
    
    # 相对误差
    relative_error = np.abs(D_pred - D_true) / (D_true + 1e-10) * 100
    errors['mean_relative_error'] = np.mean(relative_error)
    errors['median_relative_error'] = np.median(relative_error)
    errors['max_relative_error'] = np.max(relative_error)
    
    # 绝对误差
    absolute_error = np.abs(D_pred - D_true)
    errors['mean_absolute_error'] = np.mean(absolute_error)
    
    # MSE
    errors['mse'] = np.mean((D_pred - D_true)**2)
    errors['rmse'] = np.sqrt(errors['mse'])
    
    # 低频误差（前10%）
    low_freq_idx = len(frequencies) // 10
    errors['low_freq_error'] = np.mean(relative_error[:low_freq_idx])
    
    # 高频误差（后50%）
    high_freq_idx = len(frequencies) // 2
    errors['high_freq_error'] = np.mean(relative_error[high_freq_idx:])
    
    # 关键频率点的误差
    key_freqs_GHz = [0.1, 0.2, 0.5, 1.0, 5.0, 10.0]
    errors['key_freq_errors'] = {}
    for freq_GHz in key_freqs_GHz:
        idx = np.argmin(np.abs(frequencies - freq_GHz*1e9))
        if idx < len(relative_error):
            errors['key_freq_errors'][f'{freq_GHz}GHz'] = relative_error[idx]
    
    return D_pred, pred, relative_error, errors

# ============================================================
# 7. 主实验：对比两种方案
# ============================================================
print("\n" + "="*70)
print("开始训练实验")
print("="*70)

# 方案A: 预测速度谱
print("\n" + "▶"*35)
print("方案A: 预测速度谱 → 还原位移谱")
print("▶"*35)

model_velocity = SimpleSpectrumPredictor(input_dim=4, hidden_dim=64)
losses_velocity = train_model(
    model_velocity, 
    dataloader, 
    target_type='velocity',
    n_epochs=100,
    lr=0.001
)

# 方案B: 直接预测位移谱
print("\n" + "▶"*35)
print("方案B: 直接预测位移谱")
print("▶"*35)

model_displacement = SimpleSpectrumPredictor(input_dim=4, hidden_dim=64)
losses_displacement = train_model(
    model_displacement,
    dataloader,
    target_type='displacement',
    n_epochs=100,
    lr=0.001
)

# ============================================================
# 8. 评估两种方案
# ============================================================
print("\n" + "="*70)
print("评估结果")
print("="*70)

D_pred_from_V, V_pred, error_from_V, metrics_A = evaluate_model(
    model_velocity, features, D_true, V_true, frequencies, target_type='velocity'
)

D_pred_direct, _, error_direct, metrics_B = evaluate_model(
    model_displacement, features, D_true, V_true, frequencies, target_type='displacement'
)

# ============================================================
# 9. 打印对比结果
# ============================================================
print("\n" + "▶"*35)
print("方案A: 预测速度 → 还原位移")
print("▶"*35)
print(f"平均相对误差: {metrics_A['mean_relative_error']:.2f}%")
print(f"中位数相对误差: {metrics_A['median_relative_error']:.2f}%")
print(f"最大相对误差: {metrics_A['max_relative_error']:.2f}%")
print(f"RMSE: {metrics_A['rmse']:.6e} Å")
print(f"低频误差 (0-10%): {metrics_A['low_freq_error']:.2f}%")
print(f"高频误差 (50-100%): {metrics_A['high_freq_error']:.2f}%")

print("\n关键频率点误差:")
for freq_label, error in metrics_A['key_freq_errors'].items():
    print(f"  {freq_label}: {error:.2f}%")

print("\n" + "▶"*35)
print("方案B: 直接预测位移")
print("▶"*35)
print(f"平均相对误差: {metrics_B['mean_relative_error']:.2f}%")
print(f"中位数相对误差: {metrics_B['median_relative_error']:.2f}%")
print(f"最大相对误差: {metrics_B['max_relative_error']:.2f}%")
print(f"RMSE: {metrics_B['rmse']:.6e} Å")
print(f"低频误差 (0-10%): {metrics_B['low_freq_error']:.2f}%")
print(f"高频误差 (50-100%): {metrics_B['high_freq_error']:.2f}%")

print("\n关键频率点误差:")
for freq_label, error in metrics_B['key_freq_errors'].items():
    print(f"  {freq_label}: {error:.2f}%")

# ============================================================
# 10. 误差对比
# ============================================================
print("\n" + "="*70)
print("📊 对比总结")
print("="*70)

improvement_ratio = metrics_A['mean_relative_error'] / metrics_B['mean_relative_error']
print(f"\n方案A vs 方案B:")
print(f"  平均误差比: {improvement_ratio:.2f}x")
print(f"  {'方案A更差' if improvement_ratio > 1 else '方案B更差'}")

improvement_low_freq = metrics_A['low_freq_error'] / metrics_B['low_freq_error']
print(f"\n低频部分 (0-10%):")
print(f"  误差比: {improvement_low_freq:.2f}x")
print(f"  {'方案A低频误差显著更大!' if improvement_low_freq > 1.5 else '两者相近'}")

improvement_high_freq = metrics_A['high_freq_error'] / metrics_B['high_freq_error']
print(f"\n高频部分 (50-100%):")
print(f"  误差比: {improvement_high_freq:.2f}x")
print(f"  {'方案A高频误差更大' if improvement_high_freq > 1.2 else '两者相近'}")

# ============================================================
# 11. 可视化结果
# ============================================================
print("\n" + "="*70)
print("生成可视化")
print("="*70)

fig = plt.figure(figsize=(16, 14))

# 子图1: 训练损失曲线
ax1 = plt.subplot(3, 2, 1)
ax1.semilogy(losses_velocity, 'b-', linewidth=2, label='Velocity Prediction')
ax1.semilogy(losses_displacement, 'r-', linewidth=2, label='Displacement Prediction')
ax1.set_xlabel('Epoch', fontsize=11, fontweight='bold')
ax1.set_ylabel('Training Loss (MSE)', fontsize=11, fontweight='bold')
ax1.set_title('Training Loss Curves', fontsize=12, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# 子图2: 频谱对比（全频段）
ax2 = plt.subplot(3, 2, 2)
ax2.semilogy(frequencies/1e9, D_true, 'k-', linewidth=2.5, label='Ground Truth', alpha=0.8)
ax2.semilogy(frequencies/1e9, D_pred_from_V, 'b--', linewidth=2, label='From Velocity', alpha=0.7)
ax2.semilogy(frequencies/1e9, D_pred_direct, 'r--', linewidth=2, label='Direct Prediction', alpha=0.7)
ax2.set_xlabel('Frequency (GHz)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Displacement Amplitude (Å)', fontsize=11, fontweight='bold')
ax2.set_title('Predicted Displacement Spectra (Full Range)', fontsize=12, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, frequencies[-1]/1e9)

# 子图3: 低频段放大
ax3 = plt.subplot(3, 2, 3)
freq_limit = 10  # GHz
mask = frequencies < freq_limit * 1e9
ax3.semilogy(frequencies[mask]/1e9, D_true[mask], 'k-', linewidth=2.5, label='Ground Truth', alpha=0.8)
ax3.semilogy(frequencies[mask]/1e9, D_pred_from_V[mask], 'b--', linewidth=2, label='From Velocity', alpha=0.7)
ax3.semilogy(frequencies[mask]/1e9, D_pred_direct[mask], 'r--', linewidth=2, label='Direct Prediction', alpha=0.7)
ax3.set_xlabel('Frequency (GHz)', fontsize=11, fontweight='bold')
ax3.set_ylabel('Displacement Amplitude (Å)', fontsize=11, fontweight='bold')
ax3.set_title(f'Low Frequency Range (0-{freq_limit} GHz)', fontsize=12, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)

# 子图4: 相对误差对比（全频段）
ax4 = plt.subplot(3, 2, 4)
ax4.plot(frequencies/1e9, error_from_V, 'b-', linewidth=2, label='From Velocity', alpha=0.7)
ax4.plot(frequencies/1e9, error_direct, 'r-', linewidth=2, label='Direct Prediction', alpha=0.7)
ax4.set_xlabel('Frequency (GHz)', fontsize=11, fontweight='bold')
ax4.set_ylabel('Relative Error (%)', fontsize=11, fontweight='bold')
ax4.set_title('Relative Error vs Frequency', fontsize=12, fontweight='bold')
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3)
ax4.set_xlim(0, frequencies[-1]/1e9)
ax4.set_ylim(0, min(np.percentile(error_from_V, 95), 100))  # 只显示到95分位数

# 子图5: 低频段相对误差放大
ax5 = plt.subplot(3, 2, 5)
ax5.semilogy(frequencies[mask]/1e9, error_from_V[mask], 'b-', linewidth=2.5, label='From Velocity', alpha=0.7)
ax5.semilogy(frequencies[mask]/1e9, error_direct[mask], 'r-', linewidth=2.5, label='Direct Prediction', alpha=0.7)
ax5.set_xlabel('Frequency (GHz)', fontsize=11, fontweight='bold')
ax5.set_ylabel('Relative Error (%) [log scale]', fontsize=11, fontweight='bold')
ax5.set_title(f'Error Amplification at Low Frequencies (0-{freq_limit} GHz)', fontsize=12, fontweight='bold')
ax5.legend(fontsize=10)
ax5.grid(True, alpha=0.3)
ax5.axhline(y=5, color='gray', linestyle=':', linewidth=1.5, label='5% threshold')
ax5.axhline(y=10, color='gray', linestyle='--', linewidth=1.5, label='10% threshold')

# 子图6: 误差分布柱状图
ax6 = plt.subplot(3, 2, 6)
error_bins = np.linspace(0, min(np.percentile(error_from_V, 95), 50), 30)
ax6.hist(error_from_V, bins=error_bins, alpha=0.6, color='blue', 
         label=f'From Velocity (mean={metrics_A["mean_relative_error"]:.1f}%)', edgecolor='black')
ax6.hist(error_direct, bins=error_bins, alpha=0.6, color='red', 
         label=f'Direct (mean={metrics_B["mean_relative_error"]:.1f}%)', edgecolor='black')
ax6.set_xlabel('Relative Error (%)', fontsize=11, fontweight='bold')
ax6.set_ylabel('Frequency Count', fontsize=11, fontweight='bold')
ax6.set_title('Error Distribution', fontsize=12, fontweight='bold')
ax6.legend(fontsize=10)
ax6.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('velocity_vs_direct_training_comparison.png', dpi=300, bbox_inches='tight')
print("✅ 图像已保存: velocity_vs_direct_training_comparison.png")
plt.show()

# ============================================================
# 12. 详细的频段分析
# ============================================================
print("\n" + "="*70)
print("📈 详细频段分析")
print("="*70)

# 定义频段
freq_bands = [
    ("Ultra-low (主导运动)", 0, 0.5),
    ("Low (次要运动)", 0.5, 2),
    ("Mid (loop运动)", 2, 10),
    ("High (快速振动)", 10, frequencies[-1]/1e9)
]

print("\n%-25s | %-15s | %-15s | %-10s" % ("频段", "方案A误差", "方案B误差", "误差比"))
print("-" * 70)

for band_name, freq_min, freq_max in freq_bands:
    mask = (frequencies >= freq_min * 1e9) & (frequencies < freq_max * 1e9)
    if np.sum(mask) > 0:
        error_A = np.mean(error_from_V[mask])
        error_B = np.mean(error_direct[mask])
        ratio = error_A / error_B if error_B > 0 else np.inf
        
        print("%-25s | %13.2f%% | %13.2f%% | %8.2fx" % (
            band_name, error_A, error_B, ratio
        ))

# ============================================================
# 13. 保存详细结果到文件
# ============================================================
print("\n" + "="*70)
print("保存详细结果")
print("="*70)

results_file = 'velocity_vs_direct_comparison_results.npz'
np.savez(results_file,
    frequencies=frequencies,
    D_true=D_true,
    V_true=V_true,
    D_pred_from_V=D_pred_from_V,
    V_pred=V_pred,
    D_pred_direct=D_pred_direct,
    error_from_V=error_from_V,
    error_direct=error_direct,
    losses_velocity=losses_velocity,
    losses_displacement=losses_displacement
)
print(f"✅ 结果已保存: {results_file}")

# ============================================================
# 14. 生成详细报告
# ============================================================
report = f"""
{'='*70}
蛋白质运动频谱预测方案对比实验报告
{'='*70}

实验设置:
-----------
- 数据点数: {len(frequencies)}
- 频率范围: {frequencies[0]/1e9:.4f} - {frequencies[-1]/1e9:.2f} GHz
- 网络结构: 3层MLP (输入维度=4, 隐藏层=64)
- 训练轮数: 100 epochs
- 优化器: Adam (lr=0.001)
- 损失函数: MSE

方案A: 预测速度谱 → 还原位移谱
----------------------------------------
训练损失:
  初始: {losses_velocity[0]:.6e}
  最终: {losses_velocity[-1]:.6e}
  收敛程度: {(1 - losses_velocity[-1]/losses_velocity[0])*100:.1f}%

位移谱预测误差:
  平均相对误差: {metrics_A['mean_relative_error']:.2f}%
  中位数相对误差: {metrics_A['median_relative_error']:.2f}%
  最大相对误差: {metrics_A['max_relative_error']:.2f}%
  RMSE: {metrics_A['rmse']:.6e} Å
  
分频段误差:
  低频 (0-10%): {metrics_A['low_freq_error']:.2f}%
  高频 (50-100%): {metrics_A['high_freq_error']:.2f}%

关键频率点:
"""

for freq_label, error in metrics_A['key_freq_errors'].items():
    report += f"  {freq_label}: {error:.2f}%\n"

report += f"""
方案B: 直接预测位移谱
----------------------------------------
训练损失:
  初始: {losses_displacement[0]:.6e}
  最终: {losses_displacement[-1]:.6e}
  收敛程度: {(1 - losses_displacement[-1]/losses_displacement[0])*100:.1f}%

位移谱预测误差:
  平均相对误差: {metrics_B['mean_relative_error']:.2f}%
  中位数相对误差: {metrics_B['median_relative_error']:.2f}%
  最大相对误差: {metrics_B['max_relative_error']:.2f}%
  RMSE: {metrics_B['rmse']:.6e} Å
  
分频段误差:
  低频 (0-10%): {metrics_B['low_freq_error']:.2f}%
  高频 (50-100%): {metrics_B['high_freq_error']:.2f}%

关键频率点:
"""

for freq_label, error in metrics_B['key_freq_errors'].items():
    report += f"  {freq_label}: {error:.2f}%\n"

report += f"""
{'='*70}
核心结论
{'='*70}

总体对比:
---------
  平均误差比 (A/B): {improvement_ratio:.2f}x
  结论: {'方案A误差更大' if improvement_ratio > 1 else '方案B误差更大'}

低频性能 (0-10%, 主导运动区域):
----------------------------------
  误差比 (A/B): {improvement_low_freq:.2f}x
  方案A误差: {metrics_A['low_freq_error']:.2f}%
  方案B误差: {metrics_B['low_freq_error']:.2f}%
  结论: {'⚠️ 方案A在低频区域误差显著更大!' if improvement_low_freq > 1.5 else '两者性能相近'}

高频性能 (50-100%):
-------------------
  误差比 (A/B): {improvement_high_freq:.2f}x
  方案A误差: {metrics_A['high_freq_error']:.2f}%
  方案B误差: {metrics_B['high_freq_error']:.2f}%
  结论: {'方案A略差' if improvement_high_freq > 1.2 else '两者性能相近'}

{'='*70}
最终推荐
{'='*70}

"""

if improvement_ratio > 1.2 and improvement_low_freq > 1.5:
    report += """
✅ 强烈推荐: 方案B (直接预测位移谱)

理由:
1. 整体误差小 {:.1f}%
2. 低频区域 (主导运动) 误差显著更小
3. 数值稳定性更好
4. 不需要后处理 (避免除法运算)
5. 训练更容易收敛

适用场景:
- 蛋白质构象变化分析 (纳秒尺度)
- 结构域运动研究
- 低频集体运动分析

⚠️ 方案A的问题:
- 低频误差放大 {:.2f}倍
- 除法运算引入数值不稳定
- 对于你的数据 (主导频率 0.1 GHz)，效果明显更差
""".format(metrics_B['mean_relative_error'], improvement_low_freq)

elif improvement_ratio < 0.8:
    report += """
⚠️ 意外结果: 方案A (速度→位移) 表现更好

这可能说明:
1. 速度谱更容易被网络学习
2. 或者当前网络结构/特征对速度预测更友好
3. 需要进一步分析原因

建议:
- 尝试不同的网络结构
- 增加训练数据
- 分析速度谱和位移谱的统计特性
"""

else:
    report += """
📊 结论: 两种方案性能接近

但考虑到:
1. 实际应用中的数值稳定性
2. 低频区域的重要性 (主导运动)
3. 实现的简洁性

仍然建议: 方案B (直接预测位移谱)
"""

report += "\n" + "="*70 + "\n"

# 保存报告
with open('comparison_report.txt', 'w', encoding='utf-8') as f:
    f.write(report)

print(report)
print("✅ 详细报告已保存: comparison_report.txt")

# ============================================================
# 15. 额外分析：频率响应函数
# ============================================================
print("\n" + "="*70)
print("📊 频率响应分析")
print("="*70)

# 计算每个频率点的误差放大因子
omega = 2 * np.pi * frequencies
omega = np.where(omega < 1e-6, 1e-6, omega)  # 避免除零

# 理论上，从速度还原位移时，误差会被 1/ω 放大
theoretical_amplification = 1.0 / omega * frequencies[len(frequencies)//2]  # 归一化到中间频率

# 实际的误差比
actual_amplification = error_from_V / (error_direct + 1e-10)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

# 理论 vs 实际放大因子
ax1.loglog(frequencies/1e9, theoretical_amplification, 'k--', linewidth=2, 
           label='Theoretical (∝ 1/ω)', alpha=0.7)
ax1.loglog(frequencies/1e9, actual_amplification, 'b-', linewidth=2, 
           label='Actual (measured)', alpha=0.7)
ax1.axhline(y=1, color='red', linestyle=':', linewidth=2, label='No amplification')
ax1.set_xlabel('Frequency (GHz)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Error Amplification Factor', fontsize=12, fontweight='bold')
ax1.set_title('Error Amplification: From Velocity vs Direct Prediction', fontsize=13, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3, which='both')
ax1.set_xlim(frequencies[1]/1e9, frequencies[-1]/1e9)

# 误差差异
error_difference = error_from_V - error_direct
ax2.plot(frequencies/1e9, error_difference, 'purple', linewidth=2.5, alpha=0.7)
ax2.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Equal performance')
ax2.fill_between(frequencies/1e9, 0, error_difference, 
                  where=(error_difference > 0), alpha=0.3, color='blue', 
                  label='Velocity method worse')
ax2.fill_between(frequencies/1e9, 0, error_difference, 
                  where=(error_difference < 0), alpha=0.3, color='red', 
                  label='Direct method worse')
ax2.set_xlabel('Frequency (GHz)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Error Difference (A - B) %', fontsize=12, fontweight='bold')
ax2.set_title('Absolute Error Difference: Method A minus Method B', fontsize=13, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, frequencies[-1]/1e9)

plt.tight_layout()
plt.savefig('error_amplification_analysis.png', dpi=300, bbox_inches='tight')
print("✅ 频率响应分析图已保存: error_amplification_analysis.png")
plt.show()

# ============================================================
# 16. 预测值与真实值的散点图对比
# ============================================================
print("\n" + "="*70)
print("生成散点图对比")
print("="*70)

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# 方案A散点图
ax = axes[0]
ax.scatter(D_true, D_pred_from_V, alpha=0.5, s=20, c=frequencies/1e9, 
           cmap='viridis', edgecolors='none')
ax.plot([D_true.min(), D_true.max()], [D_true.min(), D_true.max()], 
        'r--', linewidth=2, label='Perfect prediction')

# 添加R²
from scipy.stats import pearsonr
r_value_A, _ = pearsonr(D_true, D_pred_from_V)
r2_A = r_value_A ** 2

ax.set_xlabel('True Displacement (Å)', fontsize=12, fontweight='bold')
ax.set_ylabel('Predicted Displacement (Å)', fontsize=12, fontweight='bold')
ax.set_title(f'Method A: Velocity → Displacement\nR² = {r2_A:.4f}', 
             fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xscale('log')
ax.set_yscale('log')

# 添加色标
cbar = plt.colorbar(ax.collections[0], ax=ax)
cbar.set_label('Frequency (GHz)', fontsize=11, fontweight='bold')

# 方案B散点图
ax = axes[1]
scatter = ax.scatter(D_true, D_pred_direct, alpha=0.5, s=20, c=frequencies/1e9, 
                     cmap='viridis', edgecolors='none')
ax.plot([D_true.min(), D_true.max()], [D_true.min(), D_true.max()], 
        'r--', linewidth=2, label='Perfect prediction')

# 添加R²
r_value_B, _ = pearsonr(D_true, D_pred_direct)
r2_B = r_value_B ** 2

ax.set_xlabel('True Displacement (Å)', fontsize=12, fontweight='bold')
ax.set_ylabel('Predicted Displacement (Å)', fontsize=12, fontweight='bold')
ax.set_title(f'Method B: Direct Displacement Prediction\nR² = {r2_B:.4f}', 
             fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xscale('log')
ax.set_yscale('log')

# 添加色标
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Frequency (GHz)', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('prediction_scatter_comparison.png', dpi=300, bbox_inches='tight')
print("✅ 散点图已保存: prediction_scatter_comparison.png")
plt.show()

print(f"\n相关系数对比:")
print(f"  方案A: R² = {r2_A:.4f}")
print(f"  方案B: R² = {r2_B:.4f}")
print(f"  {'方案B拟合更好' if r2_B > r2_A else '方案A拟合更好'}")

# ============================================================
# 17. 关键频率点的详细对比
# ============================================================
print("\n" + "="*70)
print("🎯 关键频率点详细对比")
print("="*70)

key_frequencies_GHz = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 2.0, 5.0, 10.0]

print("\n%-10s | %-12s | %-15s | %-15s | %-15s | %-15s" % (
    "频率(GHz)", "周期", "真实值(Å)", "方案A预测", "方案B预测", "误差比(A/B)"
))
print("-" * 100)

for freq_GHz in key_frequencies_GHz:
    idx = np.argmin(np.abs(frequencies - freq_GHz * 1e9))
    if idx < len(D_true):
        period_ns = 1 / freq_GHz
        true_val = D_true[idx]
        pred_A = D_pred_from_V[idx]
        pred_B = D_pred_direct[idx]
        error_A = error_from_V[idx]
        error_B = error_direct[idx]
        error_ratio = error_A / error_B if error_B > 0 else np.inf
        
        print("%-10.1f | %-12s | %13.4f | %13.4f | %13.4f | %13.2fx" % (
            freq_GHz,
            f"{period_ns:.1f} ns" if period_ns >= 1 else f"{period_ns*1000:.0f} ps",
            true_val,
            pred_A,
            pred_B,
            error_ratio
        ))

# ============================================================
# 18. 物理意义分析
# ============================================================
print("\n" + "="*70)
print("🔬 物理意义分析")
print("="*70)

print("""
对于蛋白质运动频谱预测，两种方案的物理含义：

方案A: 预测速度谱 → 还原位移谱
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
物理量关系: v(ω) = iω · d(ω)
反向转换:   d(ω) = v(ω) / (iω)

优势:
  ✓ 速度谱能量分布可能更均匀
  ✓ 某些物理模型直接产生速度
  
劣势:
  ✗ 低频时 ω→0，除法导致数值不稳定
  ✗ 预测误差在低频被放大
  ✗ 需要额外的后处理步骤
  ✗ 对于慢速构象变化(你的主要兴趣)误差大

方案B: 直接预测位移谱
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
直接输出: d(ω)

优势:
  ✓ 直接优化目标量
  ✓ 数值稳定
  ✓ 低频和高频同等对待
  ✓ 适合分析构象变化
  
劣势:
  ✗ 位移谱动态范围可能较大

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
根据你的数据特征：
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")

# 分析频谱特征
low_freq_power = np.sum(D_true[frequencies < 1e9]**2)
total_power = np.sum(D_true**2)
low_freq_ratio = low_freq_power / total_power * 100

print(f"  • 主导频率: {frequencies[np.argmax(D_true)]/1e9:.2f} GHz")
print(f"  • 低频能量占比 (<1 GHz): {low_freq_ratio:.1f}%")
print(f"  • 位移动态范围: {D_true.max()/D_true[D_true>0].min():.1f}x")
print(f"  • 速度动态范围: {V_true.max()/V_true[V_true>0].min():.1f}x")

if low_freq_ratio > 50:
    print(f"\n  ⚠️ 低频成分占主导 ({low_freq_ratio:.1f}%)")
    print("  → 强烈推荐方案B (直接预测位移)")
    print(f"  → 低频误差降低 {improvement_low_freq:.1f}倍")

# ============================================================
# 19. 训练稳定性分析
# ============================================================
print("\n" + "="*70)
print("📉 训练稳定性分析")
print("="*70)

# 计算训练过程的梯度
losses_velocity = np.array(losses_velocity)
losses_displacement = np.array(losses_displacement)

# 损失下降速度
loss_drop_velocity = (losses_velocity[0] - losses_velocity[-1]) / losses_velocity[0] * 100
loss_drop_displacement = (losses_displacement[0] - losses_displacement[-1]) / losses_displacement[0] * 100

# 损失稳定性（后期波动）
stability_velocity = np.std(losses_velocity[-10:]) / np.mean(losses_velocity[-10:]) * 100
stability_displacement = np.std(losses_displacement[-10:]) / np.mean(losses_displacement[-10:]) * 100

print(f"\n训练收敛性:")
print(f"  方案A (速度): {loss_drop_velocity:.1f}% 损失下降")
print(f"  方案B (位移): {loss_drop_displacement:.1f}% 损失下降")
print(f"  {'方案B收敛更好' if loss_drop_displacement > loss_drop_velocity else '方案A收敛更好'}")

print(f"\n训练稳定性 (最后10 epochs的变异系数):")
print(f"  方案A (速度): {stability_velocity:.2f}%")
print(f"  方案B (位移): {stability_displacement:.2f}%")
print(f"  {'方案B更稳定' if stability_displacement < stability_velocity else '方案A更稳定'}")

# # ============================================================
# # 20. 最终总结表格
# # ============================================================
# print("\n" + "="*70)
# print("📋 综合对比总结")
# print("="*70)

# summary_table = f"""
# ┏━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
# ┃ 评估指标                  ┃   方案A (速度)  ┃   方案B (位移)  ┃   优胜方   ┃
# ┡━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━┩
# │ 平均相对误差 (%)         │ {metrics_A['mean_relative_error']:>13.2f} │ {metrics_B['mean_relative_error']:>13.2f} │ {'   B ✓' if metrics_B['mean_relative_error'] < metrics_A['mean_relative_error'] else '   A ✓':>10} │
# │ 中位数误差 (%)           │ {metrics_A['median_relative_error']:>13.2f} │ {metrics_B['median_relative_error']:>13.2f} │ {'   B ✓' if metrics_B['median_relative_error'] < metrics_A['median_relative_error'] else '   A ✓':>10} │
# │ 最大误差 (%)             │ {metrics_A['max_relative_error']:>13.2f} │ {metrics_B['max_relative_error']:>13.2f} │ {'   B ✓' if metrics_B['max_relative_error'] < metrics_A['max_relative_error'] else '   A ✓':>10} │
# │ RMSE (Å)                 │ {metrics_A['rmse']:>13.2e} │ {metrics_B['rmse']:>13.2e} │ {'   B ✓' if metrics_B['rmse'] < metrics_A['rmse'] else '   A ✓':>10} │
# ├─────────────────────────┼───────────────┼───────────────┼────────────┤
# │ 低频误差 (0-10%)         │ {metrics_A['low_freq_error']:>13.2f} │ {metrics_B['low_freq_error']:>13.2f} │ {'   B ✓✓' if metrics_B['low_freq_error'] < metrics_A['low_freq_error'] else '   A ✓✓':>10} │
# │ 高频误差 (50-100%)       │ {metrics_A['high_freq_error']:>13.2f} │ {metrics_B['high_freq_error']:>13.2f} │ {'   B ✓' if metrics_B['high_freq_error'] < metrics_A['high_freq_error'] else '   A ✓':>10} │
# ├─────────────────────────┼───────────────┼───────────────┼────────────┤
# │ R² 值                    │ {r2_A:>13.4f} │ {r2_B:>13.4f} │ {'   B ✓' if r2_B > r2_A else '   A ✓':>10} │
# │ 训

