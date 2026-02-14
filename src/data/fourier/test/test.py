import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from pathlib import Path
import seaborn as sns

class ProteinMotionSpectrumAnalyzer:
    """蛋白质运动频谱分析器"""
    
    def __init__(self, trajectory_path, dt=1.0):
        """
        参数:
            trajectory_path: npy文件路径
            dt: 时间步长 (ps, ns, 或其他单位)
        """
        # 加载轨迹
        self.coords = np.load(trajectory_path)  # (frames, num_residues, 3)
        self.frames, self.num_residues, _ = self.coords.shape
        self.dt = dt
        self.fs = 1.0 / dt  # 采样频率
        
        print(f"✅ 加载轨迹: {self.frames} frames, {self.num_residues} residues")
        print(f"   时间步长: {dt}, 采样频率: {self.fs:.4f}")
        print(f"   总时长: {self.frames * dt:.2f} 时间单位")
        
        # 存储结果
        self.displacement_spectrum = None
        self.velocity_spectrum = None
        self.acceleration_spectrum = None
        self.frequencies = None
        
    def center_trajectory(self):
        """中心化轨迹（去除整体平移）"""
        # 计算每帧的质心
        centroid = np.mean(self.coords, axis=1, keepdims=True)  # (frames, 1, 3)
        self.coords = self.coords - centroid
        print("✅ 轨迹已中心化")
    
    def align_trajectory(self):
        """对齐轨迹（去除整体旋转）- 简化版"""
        # 使用第一帧作为参考
        reference = self.coords[0]
        
        for i in range(1, self.frames):
            # 计算最优旋转矩阵（Kabsch算法）
            H = self.coords[i].T @ reference
            U, S, Vt = np.linalg.svd(H)
            R = Vt.T @ U.T
            
            # 确保是旋转矩阵（det=1）
            if np.linalg.det(R) < 0:
                Vt[-1, :] *= -1
                R = Vt.T @ U.T
            
            self.coords[i] = self.coords[i] @ R
        
        print("✅ 轨迹已对齐")
    
    def compute_spectra(self, per_residue=False):
        """
        计算位移谱、速度谱、加速度谱
        
        参数:
            per_residue: 是否计算每个残基的频谱（False则计算整体平均）
        """
        print("\n" + "="*60)
        print("计算频谱...")
        print("="*60)
        
        if per_residue:
            self._compute_per_residue_spectra()
        else:
            self._compute_global_spectra()
    
    def _compute_global_spectra(self):
        """计算全局平均频谱"""
        # 计算每个原子的位移幅度
        displacement = np.sqrt(np.sum(self.coords**2, axis=2))  # (frames, num_residues)
        
        # 平均所有残基
        avg_displacement = np.mean(displacement, axis=1)  # (frames,)
        
        # FFT
        fft_result = np.fft.fft(avg_displacement)
        self.frequencies = np.fft.fftfreq(self.frames, self.dt)
        omega = 2 * np.pi * self.frequencies
        
        # 只保留正频率
        positive_freq_idx = self.frequencies > 0
        self.frequencies = self.frequencies[positive_freq_idx]
        omega = omega[positive_freq_idx]
        
        # 位移谱
        self.displacement_spectrum = np.abs(fft_result[positive_freq_idx]) * 2 / self.frames
        
        # 速度谱和加速度谱（频域微分）
        fft_vel = 1j * 2*np.pi*self.frequencies * fft_result[positive_freq_idx]
        fft_acc = -(2*np.pi*self.frequencies)**2 * fft_result[positive_freq_idx]
        
        self.velocity_spectrum = np.abs(fft_vel) * 2 / self.frames
        self.acceleration_spectrum = np.abs(fft_acc) * 2 / self.frames
        
        print(f"✅ 全局频谱计算完成")
        print(f"   频率范围: {self.frequencies[0]:.6f} ~ {self.frequencies[-1]:.6f} Hz")
        print(f"   频率分辨率: {self.frequencies[1] - self.frequencies[0]:.6f} Hz")
    
    def _compute_per_residue_spectra(self):
        """计算每个残基的频谱"""
        # 计算每个残基的位移幅度
        displacement = np.sqrt(np.sum(self.coords**2, axis=2))  # (frames, num_residues)
        
        # 为每个残基计算FFT
        self.displacement_spectrum_per_res = []
        self.velocity_spectrum_per_res = []
        self.acceleration_spectrum_per_res = []
        
        for res_idx in range(self.num_residues):
            res_disp = displacement[:, res_idx]
            
            fft_result = np.fft.fft(res_disp)
            frequencies = np.fft.fftfreq(self.frames, self.dt)
            
            positive_freq_idx = frequencies > 0
            freq = frequencies[positive_freq_idx]
            omega = 2 * np.pi * freq
            
            # 位移谱
            disp_spec = np.abs(fft_result[positive_freq_idx]) * 2 / self.frames
            
            # 速度谱和加速度谱
            fft_vel = 1j * omega * fft_result[positive_freq_idx]
            fft_acc = -omega**2 * fft_result[positive_freq_idx]
            
            vel_spec = np.abs(fft_vel) * 2 / self.frames
            acc_spec = np.abs(fft_acc) * 2 / self.frames
            
            self.displacement_spectrum_per_res.append(disp_spec)
            self.velocity_spectrum_per_res.append(vel_spec)
            self.acceleration_spectrum_per_res.append(acc_spec)
        
        # 转换为数组
        self.displacement_spectrum_per_res = np.array(self.displacement_spectrum_per_res)
        self.velocity_spectrum_per_res = np.array(self.velocity_spectrum_per_res)
        self.acceleration_spectrum_per_res = np.array(self.acceleration_spectrum_per_res)
        
        # 计算平均频谱
        self.displacement_spectrum = np.mean(self.displacement_spectrum_per_res, axis=0)
        self.velocity_spectrum = np.mean(self.velocity_spectrum_per_res, axis=0)
        self.acceleration_spectrum = np.mean(self.acceleration_spectrum_per_res, axis=0)
        
        self.frequencies = freq
        
        print(f"✅ 每残基频谱计算完成")
        print(f"   频谱形状: ({self.num_residues}, {len(freq)})")
    
    def analyze_snr(self, noise_freq_threshold=None):
        """
        分析信噪比
        
        参数:
            noise_freq_threshold: 高于此频率视为噪声 (Hz)
        """
        print("\n" + "="*60)
        print("信噪比分析")
        print("="*60)
        
        if noise_freq_threshold is None:
            # 自动确定：使用Nyquist频率的一半
            noise_freq_threshold = self.frequencies[-1] / 2
        
        # 分离信号和噪声频段
        signal_idx = self.frequencies < noise_freq_threshold
        noise_idx = self.frequencies >= noise_freq_threshold
        
        # 计算信号和噪声功率
        def compute_snr(spectrum, signal_idx, noise_idx):
            signal_power = np.sum(spectrum[signal_idx]**2)
            noise_power = np.sum(spectrum[noise_idx]**2)
            
            if noise_power == 0:
                return np.inf
            
            snr = signal_power / noise_power
            snr_db = 10 * np.log10(snr)
            return snr_db
        
        snr_disp = compute_snr(self.displacement_spectrum, signal_idx, noise_idx)
        snr_vel = compute_snr(self.velocity_spectrum, signal_idx, noise_idx)
        snr_acc = compute_snr(self.acceleration_spectrum, signal_idx, noise_idx)
        
        print(f"\n噪声阈值频率: {noise_freq_threshold:.4f} Hz")
        print(f"\n信噪比 (dB):")
        print(f"  位移谱: {snr_disp:.2f} dB")
        print(f"  速度谱: {snr_vel:.2f} dB")
        print(f"  加速度谱: {snr_acc:.2f} dB")
        
        # 找出主要频率成分
        print(f"\n主要频率成分 (位移谱):")
        peak_indices = np.argsort(self.displacement_spectrum)[-5:][::-1]
        for i, idx in enumerate(peak_indices):
            print(f"  #{i+1}: {self.frequencies[idx]:.4f} Hz, "
                  f"幅值: {self.displacement_spectrum[idx]:.6f}")
        
        return {
            'snr_displacement': snr_disp,
            'snr_velocity': snr_vel,
            'snr_acceleration': snr_acc,
            'noise_threshold': noise_freq_threshold
        }
    
    def analyze_frequency_bands(self):
        """分析不同频段的能量分布"""
        print("\n" + "="*60)
        print("频段能量分析")
        print("="*60)
        
        # 定义频段
        max_freq = self.frequencies[-1]
        bands = {
            'Ultra-low (0-10%)': (0, 0.1 * max_freq),
            'Low (10-30%)': (0.1 * max_freq, 0.3 * max_freq),
            'Mid (30-60%)': (0.3 * max_freq, 0.6 * max_freq),
            'High (60-100%)': (0.6 * max_freq, max_freq)
        }
        
        results = {}
        
        for band_name, (f_low, f_high) in bands.items():
            band_idx = (self.frequencies >= f_low) & (self.frequencies < f_high)
            
            energy_disp = np.sum(self.displacement_spectrum[band_idx]**2)
            energy_vel = np.sum(self.velocity_spectrum[band_idx]**2)
            energy_acc = np.sum(self.acceleration_spectrum[band_idx]**2)
            
            results[band_name] = {
                'displacement': energy_disp,
                'velocity': energy_vel,
                'acceleration': energy_acc
            }
            
            print(f"\n{band_name} [{f_low:.4f} - {f_high:.4f} Hz]:")
            print(f"  位移能量: {energy_disp:.6e}")
            print(f"  速度能量: {energy_vel:.6e}")
            print(f"  加速度能量: {energy_acc:.6e}")
        
        return results
    
    def plot_spectra(self, save_path=None, freq_range=None):
        """绘制频谱图"""
        fig, axes = plt.subplots(3, 1, figsize=(14, 12))
        
        # 频率范围
        if freq_range is not None:
            freq_mask = (self.frequencies >= freq_range[0]) & (self.frequencies <= freq_range[1])
            freq = self.frequencies[freq_mask]
            disp = self.displacement_spectrum[freq_mask]
            vel = self.velocity_spectrum[freq_mask]
            acc = self.acceleration_spectrum[freq_mask]
        else:
            freq = self.frequencies
            disp = self.displacement_spectrum
            vel = self.velocity_spectrum
            acc = self.acceleration_spectrum
        
        # 位移谱
        axes[0].semilogy(freq, disp, 'b-', linewidth=2)
        axes[0].set_ylabel('Displacement Amplitude', fontsize=12, fontweight='bold')
        axes[0].set_title('Protein Motion Spectra', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xlim(freq[0], freq[-1])
        
        # 速度谱
        axes[1].semilogy(freq, vel, 'g-', linewidth=2)
        axes[1].set_ylabel('Velocity Amplitude', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xlim(freq[0], freq[-1])
        
        # 加速度谱
        axes[2].semilogy(freq, acc, 'r-', linewidth=2)
        axes[2].set_xlabel('Frequency (Hz)', fontsize=12, fontweight='bold')
        axes[2].set_ylabel('Acceleration Amplitude', fontsize=12, fontweight='bold')
        axes[2].grid(True, alpha=0.3)
        axes[2].set_xlim(freq[0], freq[-1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ 图像已保存: {save_path}")
        
        plt.show()
    
    def plot_comparison(self, save_path=None):
        """对比三种频谱（归一化）"""
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # 归一化
        disp_norm = self.displacement_spectrum / np.max(self.displacement_spectrum)
        vel_norm = self.velocity_spectrum / np.max(self.velocity_spectrum)
        acc_norm = self.acceleration_spectrum / np.max(self.acceleration_spectrum)
        
        # 线性尺度
        axes[0].plot(self.frequencies, disp_norm, 'b-', label='Displacement', linewidth=2)
        axes[0].plot(self.frequencies, vel_norm, 'g-', label='Velocity', linewidth=2)
        axes[0].plot(self.frequencies, acc_norm, 'r-', label='Acceleration', linewidth=2)
        axes[0].set_ylabel('Normalized Amplitude', fontsize=12, fontweight='bold')
        axes[0].set_title('Normalized Spectrum Comparison', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=11)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xlim(self.frequencies[0], self.frequencies[-1])
        
        # 对数尺度
        axes[1].semilogy(self.frequencies, disp_norm, 'b-', label='Displacement', linewidth=2)
        axes[1].semilogy(self.frequencies, vel_norm, 'g-', label='Velocity', linewidth=2)
        axes[1].semilogy(self.frequencies, acc_norm, 'r-', label='Acceleration', linewidth=2)
        axes[1].set_xlabel('Frequency (Hz)', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Normalized Amplitude (log)', fontsize=12, fontweight='bold')
        axes[1].legend(fontsize=11)
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xlim(self.frequencies[0], self.frequencies[-1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ 图像已保存: {save_path}")
        
        plt.show()
    
    def plot_per_residue_heatmap(self, spectrum_type='displacement', save_path=None):
        """绘制每个残基的频谱热图"""
        if not hasattr(self, 'displacement_spectrum_per_res'):
            print("❌ 请先使用 per_residue=True 计算频谱")
            return
        
        if spectrum_type == 'displacement':
            data = self.displacement_spectrum_per_res
            title = 'Displacement Spectrum per Residue'
            cbar_label = 'Displacement Amplitude'
        elif spectrum_type == 'velocity':
            data = self.velocity_spectrum_per_res
            title = 'Velocity Spectrum per Residue'
            cbar_label = 'Velocity Amplitude'
        elif spectrum_type == 'acceleration':
            data = self.acceleration_spectrum_per_res
            title = 'Acceleration Spectrum per Residue'
            cbar_label = 'Acceleration Amplitude'
        else:
            print("❌ spectrum_type 必须是 'displacement', 'velocity', 或 'acceleration'")
            return
        
        fig, ax = plt.subplots(figsize=(16, 8))
        
        # 使用对数尺度
        data_log = np.log10(data + 1e-10)
        
        im = ax.imshow(data_log, aspect='auto', cmap='viridis', 
                      extent=[self.frequencies[0], self.frequencies[-1], 
                             self.num_residues, 0])
        
        ax.set_xlabel('Frequency (Hz)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Residue Index', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(f'log10({cbar_label})', fontsize=11)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ 图像已保存: {save_path}")
        
        plt.show()
    
    def identify_flexible_regions(self, top_n=10):
        """识别最灵活的残基区域"""
        if not hasattr(self, 'displacement_spectrum_per_res'):
            print("❌ 请先使用 per_residue=True 计算频谱")
            return
        
        print("\n" + "="*60)
        print("识别灵活区域")
        print("="*60)
        
        # 计算每个残基的总能量
        residue_energy = np.sum(self.displacement_spectrum_per_res**2, axis=1)
        
        # 找到最灵活的残基
        top_indices = np.argsort(residue_energy)[-top_n:][::-1]
        
        print(f"\n最灵活的 {top_n} 个残基:")
        for i, idx in enumerate(top_indices):
            print(f"  #{i+1}: 残基 {idx}, 能量: {residue_energy[idx]:.6e}")
        
        # 绘制能量分布
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # 能量沿残基的分布
        axes[0].plot(residue_energy, 'b-', linewidth=2)
        axes[0].scatter(top_indices, residue_energy[top_indices], 
                       c='r', s=100, zorder=5, label=f'Top {top_n}')
        axes[0].set_xlabel('Residue Index', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Total Spectral Energy', fontsize=12, fontweight='bold')
        axes[0].set_title('Residue Flexibility Profile', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=11)
        axes[0].grid(True, alpha=0.3)
        
        # 柱状图显示top残基
        axes[1].bar(range(top_n), residue_energy[top_indices], color='steelblue')
        axes[1].set_xlabel('Rank', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Spectral Energy', fontsize=12, fontweight='bold')
        axes[1].set_title(f'Top {top_n} Most Flexible Residues', fontsize=14, fontweight='bold')
        axes[1].set_xticks(range(top_n))
        axes[1].set_xticklabels([f'Res {idx}' for idx in top_indices], rotation=45)
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.show()
        
        return top_indices, residue_energy
    
    def export_spectra(self, output_prefix='protein_spectra'):
        """导出频谱数据"""
        # 保存全局频谱
        np.savez(f'{output_prefix}_global.npz',
                frequencies=self.frequencies,
                displacement_spectrum=self.displacement_spectrum,
                velocity_spectrum=self.velocity_spectrum,
                acceleration_spectrum=self.acceleration_spectrum)
        
        print(f"✅ 全局频谱已保存: {output_prefix}_global.npz")
        
        # 如果有每残基数据，也保存
        if hasattr(self, 'displacement_spectrum_per_res'):
            np.savez(f'{output_prefix}_per_residue.npz',
                    frequencies=self.frequencies,
                    displacement_spectrum_per_res=self.displacement_spectrum_per_res,
                    velocity_spectrum_per_res=self.velocity_spectrum_per_res,
                    acceleration_spectrum_per_res=self.acceleration_spectrum_per_res)
            
            print(f"✅ 每残基频谱已保存: {output_prefix}_per_residue.npz")
    
    def compute_effective_modes(self, n_modes=5):
        """计算有效振动模式"""
        print("\n" + "="*60)
        print(f"提取前 {n_modes} 个有效振动模式")
        print("="*60)
        
        # 找到能量最大的频率
        energy = self.displacement_spectrum**2
        top_freq_indices = np.argsort(energy)[-n_modes:][::-1]
        
        print("\n主要振动模式:")
        for i, idx in enumerate(top_freq_indices):
            freq = self.frequencies[idx]
            period = 1.0 / freq if freq > 0 else np.inf
            amplitude = self.displacement_spectrum[idx]
            energy_pct = energy[idx] / np.sum(energy) * 100
            
            print(f"\n模式 #{i+1}:")
            print(f"  频率: {freq:.6f} Hz")
            print(f"  周期: {period:.6f} 时间单位")
            print(f"  幅值: {amplitude:.6e}")
            print(f"  能量占比: {energy_pct:.2f}%")
        
        # 累积能量分析
        sorted_energy = np.sort(energy)[::-1]
        cumulative_energy = np.cumsum(sorted_energy) / np.sum(sorted_energy)
        
        # 找到90%能量需要多少模式
        n_modes_90 = np.argmax(cumulative_energy >= 0.90) + 1
        n_modes_95 = np.argmax(cumulative_energy >= 0.95) + 1
        n_modes_99 = np.argmax(cumulative_energy >= 0.99) + 1
        n_modes_999 = np.argmax(cumulative_energy >= 0.999) + 1
        
        print(f"\n累积能量分析:")
        print(f"  90% 能量需要: {n_modes_90} 个模式")
        print(f"  95% 能量需要: {n_modes_95} 个模式")
        print(f"  99% 能量需要: {n_modes_99} 个模式")
        print(f"  99.9% 能量需要: {n_modes_999} 个模式")
        # 绘制累积能量曲线
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(range(1, len(cumulative_energy)+1), cumulative_energy * 100, 
               'b-', linewidth=2)
        ax.axhline(90, color='r', linestyle='--', alpha=0.7, label='90%')
        ax.axhline(95, color='g', linestyle='--', alpha=0.7, label='95%')
        ax.axhline(99, color='orange', linestyle='--', alpha=0.7, label='99%')
        ax.axhline(99.9, color='purple', linestyle='--', alpha=0.7, label='99.9%')
        ax.set_xlabel('Number of Modes', fontsize=12, fontweight='bold')
        ax.set_ylabel('Cumulative Energy (%)', fontsize=12, fontweight='bold')
        ax.set_title('Cumulative Energy Distribution', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(1, min(100, len(cumulative_energy)))
        plt.tight_layout()
        plt.show()
        
        return top_freq_indices


# ============================================================
# 使用示例
# ============================================================

def example_usage():
    """完整使用示例"""
    
    # 1. 创建分析器
    analyzer = ProteinMotionSpectrumAnalyzer(
        trajectory_path='./test/data/1ear_A_R1_no_water_frames_coords.npy',  # Replace with your file path
        dt=1e-12  # 时间步长,单位s
    )
    
    # 2. 预处理（可选但推荐）
    analyzer.center_trajectory()  # 去除整体平移
    # analyzer.align_trajectory()  # 去除整体旋转（较慢，可选）
    
    # 3. 计算全局频谱（快速）
    print("\n" + "="*70)
    print("方案1: 计算全局平均频谱")
    print("="*70)
    analyzer.compute_spectra(per_residue=False)
    
    # 4. 分析信噪比
    snr_results = analyzer.analyze_snr(noise_freq_threshold=None)  # 自动确定阈值
    
    # 5. 分析频段能量
    band_results = analyzer.analyze_frequency_bands()
    
    # 6. 绘制频谱
    analyzer.plot_spectra(save_path='protein_spectra.png', freq_range=None)
    
    # 7. 对比三种频谱
    analyzer.plot_comparison(save_path='spectra_comparison.png')
    
    # 8. 计算有效振动模式
    analyzer.compute_effective_modes(n_modes=100)
    
    # 9. 导出数据
    analyzer.export_spectra(output_prefix='my_protein_spectra')
    
    # ============================================================
    # 如果需要每个残基的详细分析（较慢，适合小蛋白）
    # ============================================================
    if analyzer.num_residues < 500:  # 只对较小的蛋白质执行
        print("\n" + "="*70)
        print("方案2: 计算每个残基的频谱（可能较慢）")
        print("="*70)
        
        analyzer.compute_spectra(per_residue=True)
        
        # 识别灵活区域
        flexible_residues, energies = analyzer.identify_flexible_regions(top_n=10)
        
        # 绘制热图
        analyzer.plot_per_residue_heatmap(spectrum_type='displacement', 
                                         save_path='residue_heatmap.png')
example_usage()

# ============================================================
# 快速分析函数
# ============================================================

def quick_analysis(trajectory_path, dt=1.0, preprocess=True):
    """
    快速分析函数 - 一键运行所有分析
    
    参数:
        trajectory_path: npy文件路径
        dt: 时间步长
        preprocess: 是否进行预处理（中心化和对齐）
    """
    print("="*70)
    print("蛋白质运动频谱分析")
    print("="*70)
    
    # 创建分析器
    analyzer = ProteinMotionSpectrumAnalyzer(trajectory_path, dt=dt)
    
    # 预处理
    if preprocess:
        analyzer.center_trajectory()
        # 注意：对齐可能很慢，根据需要取消注释
        # analyzer.align_trajectory()
    
    # 计算频谱
    analyzer.compute_spectra(per_residue=False)
    
    # 分析
    print("\n" + "▶"*35)
    snr_results = analyzer.analyze_snr()
    band_results = analyzer.analyze_frequency_bands()
    analyzer.compute_effective_modes(n_modes=5)
    
    # 绘图
    print("\n" + "▶"*35)
    print("生成可视化...")
    analyzer.plot_spectra(save_path='spectra_analysis.png')
    analyzer.plot_comparison(save_path='spectra_comparison.png')
    
    # 导出
    analyzer.export_spectra(output_prefix='protein_spectra')
    
    print("\n" + "="*70)
    print("✅ 分析完成！")
    print("="*70)
    
    return

