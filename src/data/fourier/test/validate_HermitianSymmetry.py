import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple
import seaborn as sns

class HermitianSymmetryAnalyzer:
    """
    分析蛋白质动力学轨迹FFT结果的厄米对称性
    """
    
    def __init__(self, trajectory: torch.Tensor, dt: float):
        """
        Parameters:
        -----------
        trajectory : torch.Tensor
            形状 (N_atoms, 3, T) 或 (T,) 的时间序列
            单位：Å
        dt : float
            时间步长，单位：ps
        """
        self.trajectory = trajectory
        self.dt = dt
        self.T = trajectory.shape[-1]
        
        # 确保是实值
        assert torch.is_floating_point(trajectory) or trajectory.dtype == torch.complex64
        
    def compute_fft(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算FFT并返回频率
        """
        # FFT
        X = torch.fft.fft(self.trajectory, dim=-1)
        
        # 完整频率数组（包括负频率）
        freqs = torch.fft.fftfreq(self.T, d=self.dt * 1e-12)  # Hz
        
        return X, freqs
    
    def check_hermitian_symmetry(self, X: torch.Tensor, 
                                 tolerance: float = 1e-6) -> Dict:
        """
        检查厄米对称性
        
        Returns:
        --------
        结果字典包含：
        - is_hermitian: bool
        - max_violation: float (最大违反程度)
        - symmetry_scores: 各频率点的对称性分数
        """
        T = X.shape[-1]
        
        # 获取正负频率对
        if T % 2 == 0:
            # 偶数长度
            n_pairs = T // 2 - 1
            pos_indices = torch.arange(1, T // 2)
            neg_indices = torch.arange(T - 1, T // 2, -1)
        else:
            # 奇数长度
            n_pairs = T // 2
            pos_indices = torch.arange(1, T // 2 + 1)
            neg_indices = torch.arange(T - 1, T // 2, -1)
        
        # 提取正负频率对应的值
        X_pos = X[..., pos_indices]
        X_neg = X[..., neg_indices]
        
        # 检查 X(-f) = X*(f)
        X_neg_conj = torch.conj(X_neg)
        
        # 计算差异
        diff = X_pos - X_neg_conj
        abs_diff = torch.abs(diff)
        
        # 相对误差
        X_pos_mag = torch.abs(X_pos)
        relative_error = abs_diff / (X_pos_mag + 1e-10)
        
        # 统计
        max_abs_diff = abs_diff.max().item()
        max_rel_error = relative_error.max().item()
        mean_rel_error = relative_error.mean().item()
        
        is_hermitian = max_rel_error < tolerance
        
        # DC分量和Nyquist频率应该是实数
        dc_is_real = torch.abs(X[..., 0].imag).max().item() < tolerance
        
        nyquist_is_real = True
        if T % 2 == 0:
            nyquist_is_real = torch.abs(X[..., T // 2].imag).max().item() < tolerance
        
        return {
            'is_hermitian': is_hermitian,
            'dc_is_real': dc_is_real,
            'nyquist_is_real': nyquist_is_real,
            'max_absolute_violation': max_abs_diff,
            'max_relative_violation': max_rel_error,
            'mean_relative_violation': mean_rel_error,
            'symmetry_scores': relative_error.cpu().numpy(),
            'pos_indices': pos_indices.numpy(),
            'neg_indices': neg_indices.numpy()
        }
    
    def analyze_amplitude_phase_symmetry(self, X: torch.Tensor, 
                                         freqs: torch.Tensor) -> Dict:
        """
        分别分析幅值和相位的对称性
        """
        T = X.shape[-1]
        
        # 幅值和相位
        amplitude = torch.abs(X)
        phase = torch.angle(X)
        
        # 正负频率配对
        if T % 2 == 0:
            pos_idx = torch.arange(1, T // 2)
            neg_idx = torch.arange(T - 1, T // 2, -1)
        else:
            pos_idx = torch.arange(1, T // 2 + 1)
            neg_idx = torch.arange(T - 1, T // 2, -1)
        
        amp_pos = amplitude[..., pos_idx]
        amp_neg = amplitude[..., neg_idx]
        phase_pos = phase[..., pos_idx]
        phase_neg = phase[..., neg_idx]
        
        # 幅值应该相等
        amp_diff = torch.abs(amp_pos - amp_neg)
        amp_rel_error = amp_diff / (amp_pos + 1e-10)
        
        # 相位应该相反
        phase_sum = phase_pos + phase_neg
        # 处理相位的周期性 [-π, π]
        phase_sum = torch.atan2(torch.sin(phase_sum), torch.cos(phase_sum))
        phase_antisym_error = torch.abs(phase_sum)
        
        return {
            'amplitude': {
                'max_relative_error': amp_rel_error.max().item(),
                'mean_relative_error': amp_rel_error.mean().item(),
                'errors': amp_rel_error.cpu().numpy()
            },
            'phase': {
                'max_antisymmetry_violation': phase_antisym_error.max().item(),
                'mean_antisymmetry_violation': phase_antisym_error.mean().item(),
                'violations': phase_antisym_error.cpu().numpy()
            },
            'frequencies_GHz': freqs[pos_idx].cpu().numpy() / 1e9
        }
    
    def visualize_symmetry(self, X: torch.Tensor, freqs: torch.Tensor, 
                          save_path: str = 'hermitian_symmetry_analysis.png'):
        """
        可视化对称性分析
        """
        # 选择一个原子/坐标进行分析
        if X.ndim > 1:
            X_plot = X[0] if X.shape[0] <= 3 else X.reshape(-1, X.shape[-1])[0]
        else:
            X_plot = X
        
        T = X_plot.shape[-1]
        freqs_GHz = freqs.cpu().numpy() / 1e9
        
        # 幅值和相位
        amplitude = torch.abs(X_plot).cpu().numpy()
        phase = torch.angle(X_plot).cpu().numpy()
        
        # 实部和虚部
        real_part = X_plot.real.cpu().numpy()
        imag_part = X_plot.imag.cpu().numpy()
        
        fig, axes = plt.subplots(3, 2, figsize=(16, 12))
        
        # 1. 幅值谱（完整）
        axes[0, 0].plot(freqs_GHz, amplitude, 'b-', linewidth=1, alpha=0.7)
        axes[0, 0].axvline(0, color='r', linestyle='--', alpha=0.5)
        axes[0, 0].set_xlabel('Frequency (GHz)', fontweight='bold')
        axes[0, 0].set_ylabel('|X(f)|', fontweight='bold')
        axes[0, 0].set_title('Amplitude Spectrum (Full Range)', fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 相位谱（完整）
        axes[0, 1].plot(freqs_GHz, phase, 'g-', linewidth=1, alpha=0.7)
        axes[0, 1].axvline(0, color='r', linestyle='--', alpha=0.5)
        axes[0, 1].axhline(0, color='k', linestyle=':', alpha=0.3)
        axes[0, 1].set_xlabel('Frequency (GHz)', fontweight='bold')
        axes[0, 1].set_ylabel('Phase (rad)', fontweight='bold')
        axes[0, 1].set_title('Phase Spectrum (Full Range)', fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 幅值对称性验证
        pos_mask = freqs_GHz > 0
        neg_mask = freqs_GHz < 0
        
        freqs_pos = freqs_GHz[pos_mask]
        amp_pos = amplitude[pos_mask]
        
        freqs_neg_abs = np.abs(freqs_GHz[neg_mask][::-1])
        amp_neg = amplitude[neg_mask][::-1]
        
        # 只取公共范围
        min_len = min(len(freqs_pos), len(freqs_neg_abs))
        freqs_pos = freqs_pos[:min_len]
        amp_pos = amp_pos[:min_len]
        freqs_neg_abs = freqs_neg_abs[:min_len]
        amp_neg = amp_neg[:min_len]
        
        axes[1, 0].plot(freqs_pos, amp_pos, 'b-', linewidth=2, label='|X(+f)|', alpha=0.7)
        axes[1, 0].plot(freqs_neg_abs, amp_neg, 'r--', linewidth=2, label='|X(-f)|', alpha=0.7)
        axes[1, 0].set_xlabel('Frequency (GHz)', fontweight='bold')
        axes[1, 0].set_ylabel('Amplitude', fontweight='bold')
        axes[1, 0].set_title('Amplitude Symmetry: |X(+f)| vs |X(-f)|', fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 幅值差异
        amp_diff = np.abs(amp_pos - amp_neg)
        axes[1, 1].semilogy(freqs_pos, amp_diff, 'purple', linewidth=1.5)
        axes[1, 1].set_xlabel('Frequency (GHz)', fontweight='bold')
        axes[1, 1].set_ylabel('||X(+f)| - |X(-f)||', fontweight='bold')
        axes[1, 1].set_title('Amplitude Symmetry Violation', fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 5. 相位反对称性
        phase_pos = phase[pos_mask][:min_len]
        phase_neg = phase[neg_mask][::-1][:min_len]
        
        axes[2, 0].plot(freqs_pos, phase_pos, 'b-', linewidth=2, label='arg(X(+f))', alpha=0.7)
        axes[2, 0].plot(freqs_neg_abs, -phase_neg, 'r--', linewidth=2, label='-arg(X(-f))', alpha=0.7)
        axes[2, 0].set_xlabel('Frequency (GHz)', fontweight='bold')
        axes[2, 0].set_ylabel('Phase (rad)', fontweight='bold')
        axes[2, 0].set_title('Phase Anti-symmetry: arg(X(+f)) vs -arg(X(-f))', fontweight='bold')
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)
        
        # 6. 相位反对称违反度
        phase_antisym_violation = np.abs(phase_pos + phase_neg)
        axes[2, 1].plot(freqs_pos, phase_antisym_violation, 'orange', linewidth=1.5)
        axes[2, 1].set_xlabel('Frequency (GHz)', fontweight='bold')
        axes[2, 1].set_ylabel('|arg(X(+f)) + arg(X(-f))|', fontweight='bold')
        axes[2, 1].set_title('Phase Anti-symmetry Violation', fontweight='bold')
        axes[2, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ 对称性分析图已保存: {save_path}")
        plt.show()
    
    def full_analysis(self, visualize: bool = True) -> Dict:
        """
        完整分析流程
        """
        print("="*70)
        print("🔬 蛋白质动力学轨迹厄米对称性分析")
        print("="*70)
        
        # 1. 检查输入是否为实值
        if torch.is_complex(self.trajectory):
            print("\n⚠️  输入信号包含复数值！")
            is_real_input = False
        else:
            print("\n✅ 输入信号为实值")
            is_real_input = True
        
        # 2. 计算FFT
        print(f"\n计算FFT...")
        print(f"  时间序列长度: {self.T}")
        print(f"  时间步长: {self.dt} ps")
        print(f"  采样频率: {1/(self.dt*1e-12)/1e9:.2f} GHz")
        print(f"  Nyquist频率: {1/(2*self.dt*1e-12)/1e9:.2f} GHz")
        
        X, freqs = self.compute_fft()
        
        # 3. 厄米对称性检查
        print(f"\n" + "="*70)
        print("📊 厄米对称性检查 (X(-f) = X*(f))")
        print("="*70)
        
        hermitian_results = self.check_hermitian_symmetry(X)
        
        print(f"\n✓ DC分量 (f=0) 是实数: {hermitian_results['dc_is_real']}")
        if self.T % 2 == 0:
            print(f"✓ Nyquist频率 是实数: {hermitian_results['nyquist_is_real']}")
        
        print(f"\n对称性统计:")
        print(f"  最大绝对违反: {hermitian_results['max_absolute_violation']:.6e}")
        print(f"  最大相对违反: {hermitian_results['max_relative_violation']:.6e}")
        print(f"  平均相对违反: {hermitian_results['mean_relative_violation']:.6e}")
        
        if hermitian_results['is_hermitian']:
            print(f"\n✅ 满足厄米对称性 (相对误差 < 1e-6)")
        else:
            print(f"\n❌ 不满足厄米对称性!")
        
        # 4. 幅值和相位分析
        print(f"\n" + "="*70)
        print("📊 幅值和相位对称性分析")
        print("="*70)
        
        amp_phase_results = self.analyze_amplitude_phase_symmetry(X, freqs)
        
        print(f"\n幅值对称性 (|X(+f)| = |X(-f)|):")
        print(f"  最大相对误差: {amp_phase_results['amplitude']['max_relative_error']:.6e}")
        print(f"  平均相对误差: {amp_phase_results['amplitude']['mean_relative_error']:.6e}")
        
        print(f"\n相位反对称性 (arg(X(+f)) = -arg(X(-f))):")
        print(f"  最大违反 (rad): {amp_phase_results['phase']['max_antisymmetry_violation']:.6e}")
        print(f"  平均违反 (rad): {amp_phase_results['phase']['mean_antisymmetry_violation']:.6e}")
        
        # 5. 验证IFFT是否返回实值
        print(f"\n" + "="*70)
        print("📊 逆FFT验证")
        print("="*70)
        
        x_reconstructed = torch.fft.ifft(X, dim=-1)
        max_imag = torch.abs(x_reconstructed.imag).max().item()
        
        print(f"\nIFFT后虚部最大值: {max_imag:.6e}")
        
        if max_imag < 1e-6:
            print(f"✅ IFFT返回实值信号 (虚部 < 1e-6)")
        else:
            print(f"⚠️  IFFT包含显著虚部")
        
        # 6. 重建误差
        reconstruction_error = torch.sqrt(torch.mean((x_reconstructed.real - self.trajectory) ** 2))
        relative_error = reconstruction_error / torch.sqrt(torch.mean(self.trajectory ** 2))
        
        print(f"\n重建误差:")
        print(f"  RMSE: {reconstruction_error.item():.6e}")
        print(f"  相对误差: {relative_error.item():.6e}")
        
        # 7. 可视化
        if visualize:
            print(f"\n" + "="*70)
            print("📊 生成可视化...")
            self.visualize_symmetry(X, freqs)
        
        # 汇总结果
        summary = {
            'is_real_input': is_real_input,
            'is_hermitian': hermitian_results['is_hermitian'],
            'hermitian_details': hermitian_results,
            'amplitude_phase_details': amp_phase_results,
            'ifft_max_imaginary': max_imag,
            'reconstruction_error': reconstruction_error.item(),
            'relative_reconstruction_error': relative_error.item()
        }
        
        print(f"\n" + "="*70)
        print("✅ 分析完成")
        print("="*70)
        
        return summary


# ============================================================
# 测试函数
# ============================================================

def test_with_synthetic_signal():
    """
    使用合成信号测试
    """
    print("\n" + "🧪"*35)
    print("测试1: 合成实值信号 (sin波)")
    print("🧪"*35 + "\n")
    
    # 参数
    T = 1000
    dt = 0.001  # ps
    t = torch.arange(T, dtype=torch.float32) * dt
    
    # 生成混合频率信号
    f1, f2, f3 = 1e9, 5e9, 10e9  # Hz
    t_seconds = t * 1e-12
    
    x = (torch.sin(2 * np.pi * f1 * t_seconds) + 
         0.5 * torch.sin(2 * np.pi * f2 * t_seconds) +
         0.3 * torch.sin(2 * np.pi * f3 * t_seconds))
    
    # 分析
    analyzer = HermitianSymmetryAnalyzer(x, dt)
    results = analyzer.full_analysis(visualize=True)
    
    return results


def test_with_complex_signal():
    """
    测试复数信号（应该不满足厄米对称）
    """
    print("\n" + "🧪"*35)
    print("测试2: 复数信号")
    print("🧪"*35 + "\n")
    
    T = 1000
    dt = 0.001
    t = torch.arange(T, dtype=torch.float32) * dt
    
    f = 5e9
    t_seconds = t * 1e-12
    
    # 复数信号：不对称
    x = torch.sin(2 * np.pi * f * t_seconds) + 1j * torch.cos(2 * np.pi * f * t_seconds)
    
    analyzer = HermitianSymmetryAnalyzer(x, dt)
    results = analyzer.full_analysis(visualize=True)
    
    return results


def test_with_noisy_signal():
    """
    测试带噪声的实值信号
    """
    print("\n" + "🧪"*35)
    print("测试3: 带噪声的实值信号")
    print("🧪"*35 + "\n")
    
    T = 2000
    dt = 0.0001  # ps
    t = torch.arange(T, dtype=torch.float32) * dt
    t_seconds = t * 1e-12
    
    # 信号 + 高斯噪声
    f = 10e9
    signal = torch.sin(2 * np.pi * f * t_seconds)
    noise = 0.1 * torch.randn_like(signal)
    x = signal + noise
    
    analyzer = HermitianSymmetryAnalyzer(x, dt)
    results = analyzer.full_analysis(visualize=True)
    
    return results


def test_with_protein_trajectory(trajectory_path: str = None):
    """
    测试真实蛋白质轨迹
    
    Parameters:
    -----------
    trajectory_path : str
        轨迹文件路径，如果为None则生成模拟数据
    """
    print("\n" + "🧪"*35)
    print("测试4: 蛋白质MD轨迹")
    print("🧪"*35 + "\n")
    
    if trajectory_path is None:
        # 生成模拟的蛋白质运动
        print("使用模拟数据...")
        
        N_atoms = 100
        T = 5000
        dt = 0.001  # 1 fs
        
        t = torch.arange(T, dtype=torch.float32) * dt
        t_seconds = t * 1e-12
        
        # 模拟多个振动模式
        trajectory = torch.zeros(N_atoms, 3, T)
        
        for i in range(N_atoms):
            # 每个原子有不同的振动频率
            freq = (1 + 0.1 * i) * 1e9  # 1-11 GHz
            amplitude = 0.1 * (1 + 0.01 * i)  # 0.1-0.2 Å
            
            for dim in range(3):
                phase = np.random.rand() * 2 * np.pi
                trajectory[i, dim, :] = amplitude * torch.sin(
                    2 * np.pi * freq * t_seconds + phase
                )
        
        # 添加低频漂移
        drift_freq = 0.1e9  # 100 MHz
        drift = 0.5 * torch.sin(2 * np.pi * drift_freq * t_seconds)
        trajectory = trajectory + drift.view(1, 1, -1)
        
    else:
        # 加载真实轨迹
        print(f"从文件加载: {trajectory_path}")
        # 这里需要根据实际文件格式实现
        # trajectory = load_trajectory(trajectory_path)
        raise NotImplementedError("请实现轨迹加载函数")
    
    # 分析
    analyzer = HermitianSymmetryAnalyzer(trajectory, dt)
    results = analyzer.full_analysis(visualize=True)
    
    return results


def analyze_single_atom_trajectory(trajectory: torch.Tensor, 
                                   atom_idx: int = 0,
                                   dt: float = 0.001):
    """
    分析单个原子的轨迹
    
    Parameters:
    -----------
    trajectory : torch.Tensor
        形状 (N_atoms, 3, T)
    atom_idx : int
        要分析的原子索引
    dt : float
        时间步长 (ps)
    """
    print(f"\n分析原子 {atom_idx}")
    print("-" * 50)
    
    # 分析三个坐标分量
    results = {}
    for dim, coord in enumerate(['x', 'y', 'z']):
        print(f"\n坐标分量: {coord}")
        x = trajectory[atom_idx, dim, :]
        
        analyzer = HermitianSymmetryAnalyzer(x, dt)
        res = analyzer.full_analysis(visualize=False)
        results[coord] = res
    
    return results


def compare_fft_implementations():
    """
    比较不同FFT实现的厄米对称性
    """
    print("\n" + "🧪"*35)
    print("测试5: 比较不同FFT实现")
    print("🧪"*35 + "\n")
    
    T = 1000
    dt = 0.001
    t = torch.arange(T, dtype=torch.float32) * dt
    t_seconds = t * 1e-12
    
    f = 5e9
    x = torch.sin(2 * np.pi * f * t_seconds)
    
    print("信号参数:")
    print(f"  长度: {T}")
    print(f"  频率: {f/1e9} GHz")
    
    # 1. torch.fft.fft
    print("\n" + "="*50)
    print("1. torch.fft.fft")
    print("="*50)
    X_torch = torch.fft.fft(x)
    analyzer1 = HermitianSymmetryAnalyzer(x, dt)
    results1 = analyzer1.check_hermitian_symmetry(X_torch.unsqueeze(0))
    print(f"最大相对违反: {results1['max_relative_violation']:.6e}")
    
    # 2. numpy.fft.fft
    print("\n" + "="*50)
    print("2. numpy.fft.fft")
    print("="*50)
    x_np = x.numpy()
    X_numpy = np.fft.fft(x_np)
    X_numpy_torch = torch.from_numpy(X_numpy)
    results2 = analyzer1.check_hermitian_symmetry(X_numpy_torch.unsqueeze(0))
    print(f"最大相对违反: {results2['max_relative_violation']:.6e}")
    
    # 3. rfft (实数专用)
    print("\n" + "="*50)
    print("3. torch.fft.rfft (仅正频率)")
    print("="*50)
    X_rfft = torch.fft.rfft(x)
    print(f"输出长度: {len(X_rfft)} (仅包含非负频率)")
    print(f"完整FFT长度: {len(X_torch)}")
    print(f"差异: rfft自动利用厄米对称性，只存储一半数据")
    
    # 验证rfft和fft的一致性
    X_torch_positive = X_torch[:len(X_rfft)]
    diff = torch.abs(X_torch_positive - X_rfft)
    print(f"\nrfft vs fft (正频率部分):")
    print(f"  最大差异: {diff.max().item():.6e}")
    print(f"  平均差异: {diff.mean().item():.6e}")
    
    return {
        'torch_fft': results1,
        'numpy_fft': results2,
        'rfft_length': len(X_rfft),
        'fft_length': len(X_torch)
    }


def detect_non_hermitian_frequencies(X: torch.Tensor, freqs: torch.Tensor,
                                     threshold: float = 1e-4) -> Dict:
    """
    检测哪些频率违反了厄米对称性
    """
    T = X.shape[-1]
    
    if T % 2 == 0:
        pos_idx = torch.arange(1, T // 2)
        neg_idx = torch.arange(T - 1, T // 2, -1)
    else:
        pos_idx = torch.arange(1, T // 2 + 1)
        neg_idx = torch.arange(T - 1, T // 2 - 1, -1)
    
    X_pos = X[pos_idx]
    X_neg = X[neg_idx]
    X_neg_conj = torch.conj(X_neg)
    
    # 相对误差
    diff = torch.abs(X_pos - X_neg_conj)
    rel_error = diff / (torch.abs(X_pos) + 1e-10)
    
    # 找到违反阈值的频率
    violation_mask = rel_error > threshold
    violation_freqs = freqs[pos_idx][violation_mask].cpu().numpy()
    violation_errors = rel_error[violation_mask].cpu().numpy()
    
    print(f"\n检测到 {violation_mask.sum().item()} 个频率违反厄米对称性 (阈值={threshold})")
    
    if len(violation_freqs) > 0:
        print("\n违反厄米对称性的频率:")
        for freq, error in zip(violation_freqs, violation_errors):
            print(f"  {freq/1e9:.4f} GHz: 相对误差 = {error:.6e}")
    
    return {
        'n_violations': violation_mask.sum().item(),
        'violation_frequencies_Hz': violation_freqs,
        'violation_errors': violation_errors,
        'all_relative_errors': rel_error.cpu().numpy(),
        'frequencies_GHz': freqs[pos_idx].cpu().numpy() / 1e9
    }


def visualize_hermitian_heatmap(trajectory: torch.Tensor, dt: float,
                                save_path: str = 'hermitian_heatmap.png'):
    """
    可视化多个原子/坐标的厄米对称性热图
    
    Parameters:
    -----------
    trajectory : torch.Tensor
        形状 (N_atoms, 3, T) 或 (N, T)
    """
    print("\n" + "="*70)
    print("📊 生成厄米对称性热图")
    print("="*70)
    
    if trajectory.ndim == 3:
        N_atoms, N_dims, T = trajectory.shape
        traj_flat = trajectory.reshape(N_atoms * N_dims, T)
        labels = [f"Atom{i//3}_{['x','y','z'][i%3]}" for i in range(N_atoms * N_dims)]
    else:
        N, T = trajectory.shape
        traj_flat = trajectory
        labels = [f"Series_{i}" for i in range(N)]
    
    N_series = traj_flat.shape[0]
    
    # 计算每个序列的对称性指标
    symmetry_scores = []
    
    for i in range(min(N_series, 100)):  # 最多100个序列
        x = traj_flat[i, :]
        analyzer = HermitianSymmetryAnalyzer(x, dt)
        X, freqs = analyzer.compute_fft()
        results = analyzer.check_hermitian_symmetry(X.unsqueeze(0))
        
        symmetry_scores.append({
            'max_violation': results['max_relative_violation'],
            'mean_violation': results['mean_relative_violation'],
            'is_hermitian': results['is_hermitian']
        })
    
    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    n_plot = len(symmetry_scores)
    indices = np.arange(n_plot)
    
    max_violations = [s['max_violation'] for s in symmetry_scores]
    mean_violations = [s['mean_violation'] for s in symmetry_scores]
    colors = ['green' if s['is_hermitian'] else 'red' for s in symmetry_scores]
    
    # 最大违反度
    axes[0].bar(indices, max_violations, color=colors, alpha=0.7)
    axes[0].axhline(1e-6, color='blue', linestyle='--', linewidth=2, label='Threshold (1e-6)')
    axes[0].set_xlabel('Series Index', fontweight='bold')
    axes[0].set_ylabel('Max Relative Violation', fontweight='bold')
    axes[0].set_title('Hermitian Symmetry Violation (Max)', fontweight='bold')
    axes[0].set_yscale('log')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 平均违反度
    axes[1].bar(indices, mean_violations, color=colors, alpha=0.7)
    axes[1].axhline(1e-6, color='blue', linestyle='--', linewidth=2, label='Threshold (1e-6)')
    axes[1].set_xlabel('Series Index', fontweight='bold')
    axes[1].set_ylabel('Mean Relative Violation', fontweight='bold')
    axes[1].set_title('Hermitian Symmetry Violation (Mean)', fontweight='bold')
    axes[1].set_yscale('log')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 热图已保存: {save_path}")
    plt.show()
    
    # 统计
    n_hermitian = sum(1 for s in symmetry_scores if s['is_hermitian'])
    print(f"\n统计结果:")
    print(f"  总序列数: {n_plot}")
    print(f"  满足厄米对称: {n_hermitian} ({n_hermitian/n_plot*100:.1f}%)")
    print(f"  不满足: {n_plot - n_hermitian} ({(n_plot-n_hermitian)/n_plot*100:.1f}%)")
    
    return symmetry_scores


def advanced_symmetry_diagnostics(trajectory: torch.Tensor, dt: float):
    """
    高级诊断：深入分析为什么可能不满足厄米对称性
    """
    print("\n" + "="*70)
    print("🔍 高级厄米对称性诊断")
    print("="*70)
    
    # 如果是多维，只取第一个
    if trajectory.ndim > 1:
        x = trajectory.reshape(-1, trajectory.shape[-1])[0]
    else:
        x = trajectory
    
    T = x.shape[-1]
    
    # 1. 检查数值精度
    print("\n1️⃣ 数值精度检查")
    print("-" * 50)
    print(f"数据类型: {x.dtype}")
    print(f"数值范围: [{x.min().item():.6e}, {x.max().item():.6e}]")
    print(f"标准差: {x.std().item():.6e}")
    
    # 2. 检查是否包含NaN或Inf
    has_nan = torch.isnan(x).any()
    has_inf = torch.isinf(x).any()
    print(f"包含NaN: {has_nan}")
    print(f"包含Inf: {has_inf}")
    
    # 3. 检查长度（奇偶性）
    print(f"\n2️⃣ 序列长度分析")
    print("-" * 50)
    print(f"序列长度: {T} ({'偶数' if T % 2 == 0 else '奇数'})")
    
    # 4. FFT并分析
    print(f"\n3️⃣ FFT计算")
    print("-" * 50)
    X = torch.fft.fft(x)
    
    # DC分量
    dc = X[0]
    print(f"DC分量 (f=0):")
    print(f"  实部: {dc.real.item():.6e}")
    print(f"  虚部: {dc.imag.item():.6e}")
    print(f"  是否为实数: {torch.abs(dc.imag).item() < 1e-6}")
    
    # Nyquist分量
    if T % 2 == 0:
        nyquist = X[T // 2]
        print(f"\nNyquist分量 (f_max):")
        print(f"  实部: {nyquist.real.item():.6e}")
        print(f"  虚部: {nyquist.imag.item():.6e}")
        print(f"  是否为实数: {torch.abs(nyquist.imag).item() < 1e-6}")
    
    # 5. 对称性详细检查
    print(f"\n4️⃣ 对称性详细分析")
    print("-" * 50)
    
    analyzer = HermitianSymmetryAnalyzer(x, dt)
    results = analyzer.check_hermitian_symmetry(X.unsqueeze(0))
    
    print(f"对称性检查结果:")
    for key, value in results.items():
        if key not in ['symmetry_scores', 'pos_indices', 'neg_indices']:
            print(f"  {key}: {value}")
    
    # 6. 频率域详细分析
    print(f"\n5️⃣ 频域能量分布")
    print("-" * 50)
    
    power = torch.abs(X) ** 2
    total_power = power.sum().item()
    
    # 找到主要频率
    top_k = 5
    top_powers, top_indices = torch.topk(power, k=min(top_k, T))
    
    freqs = torch.fft.fftfreq(T, d=dt * 1e-12)
    
    print(f"主要频率成分 (Top {top_k}):")
    for i, (idx, pow) in enumerate(zip(top_indices, top_powers)):
        freq_Hz = freqs[idx].item()
        percentage = pow.item() / total_power * 100
        print(f"  #{i+1}: {freq_Hz/1e9:.4f} GHz, 功率占比: {percentage:.2f}%")
    
    # 7. 数值误差来源分析
    print(f"\n6️⃣ 潜在误差来源")
    print("-" * 50)
    
    issues = []
    
    if x.dtype != torch.float64:
        issues.append("• 使用float32而非float64，精度较低")
    
    if T < 100:
        issues.append("• 序列过短，可能导致频率分辨率不足")
    
    if has_nan or has_inf:
        issues.append("• 数据包含NaN或Inf，需要预处理")
    
    if torch.abs(dc.imag).item() > 1e-6:
        issues.append("• DC分量虚部不为零，输入可能不是实信号")
    
    if results['max_relative_violation'] > 1e-6:
        issues.append(f"• 对称性违反度 {results['max_relative_violation']:.2e} 超过阈值")
    
    if len(issues) == 0:
        print("✅ 未发现明显问题")
    else:
        print("⚠️  发现以下问题:")
        for issue in issues:
            print(f"  {issue}")
    
    # 8. 建议
    print(f"\n7️⃣ 改进建议")
    print("-" * 50)
    
    suggestions = []
    
    if x.dtype != torch.float64:
        suggestions.append("• 使用 torch.float64 提高精度")
    
    if T < 1000:
        suggestions.append("• 增加采样点数以提高频率分辨率")
    
    if results['max_relative_violation'] > 1e-6:
        suggestions.append("• 检查输入数据是否包含复数或数值错误")
        suggestions.append("• 考虑使用 torch.fft.rfft() 专门处理实信号")
    
    if len(suggestions) > 0:
        for suggestion in suggestions:
            print(f"  {suggestion}")
    else:
        print("✅ 当前设置已经很好")
    
    return {
        'data_info': {
            'dtype': str(x.dtype),
            'length': T,
            'has_nan': has_nan.item(),
            'has_inf': has_inf.item()
        },
        'symmetry_results': results,
        'issues': issues,
        'suggestions': suggestions
    }


def compare_fft_with_rfft(x: torch.Tensor, dt: float):
    """
    比较fft和rfft的结果（支持多维输入，每次画1个trace）
    """
    print("\n" + "="*70)
    print("📊 FFT vs RFFT 对比")
    print("="*70)
    
    # 展平仅保留最后一个维度
    # 支持如 (N_atoms, 3, T) 或 (T,)
    # 画图时只展示第一个trace，聚合用全体
    if x.ndim > 1:
        x_plot = x.reshape(-1, x.shape[-1])[0].detach().cpu()
    else:
        x_plot = x.detach().cpu()

    T = x.shape[-1]
    
    # 完整FFT
    X_full = torch.fft.fft(x)
    print(f"X_full.shape: {X_full.shape}")
    # 实数FFT
    X_rfft = torch.fft.rfft(x)
    print(f"X_rfft.shape: {X_rfft.shape}")

    print(f"\n序列长度: {T}")
    print(f"完整FFT输出长度: {X_full.shape[-1]}")
    print(f"RFFT输出长度: {X_rfft.shape[-1]}")
    print(f"存储效率: {X_rfft.shape[-1]/X_full.shape[-1]*100:.1f}%")

    # 比较正频率部分，对所有轨迹做最大值, 但显示时只画 x_plot 对应的那条
    X_full_positive = X_full[..., :X_rfft.shape[-1]]
    diff = torch.abs(X_full_positive - X_rfft)
    
    print(f"\n正频率部分对比:")
    print(f"  最大绝对差异: {diff.max().item():.6e}")
    print(f"  平均绝对差异: {diff.mean().item():.6e}")
    print(f"  最大相对差异: {(diff / (torch.abs(X_rfft) + 1e-10)).max().item():.6e}")
    
    # 验证逆变换
    x_ifft = torch.fft.ifft(X_full).real
    x_irfft = torch.fft.irfft(X_rfft, n=T)
    
    reconstruction_error_fft = torch.sqrt(torch.mean((x_ifft.real - x) ** 2))
    reconstruction_error_rfft = torch.sqrt(torch.mean((x_irfft - x) ** 2))
    
    print(f"\n逆变换重建误差:")
    print(f"  IFFT: {reconstruction_error_fft.item():.6e}")
    print(f"  IRFFT: {reconstruction_error_rfft.item():.6e}")
    
    # 可视化(只取x_plot相关)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    freqs_full = torch.fft.fftfreq(T, d=dt * 1e-12).detach().cpu().numpy() / 1e9
    freqs_rfft = torch.fft.rfftfreq(T, d=dt * 1e-12).detach().cpu().numpy() / 1e9

    # 用来提取作图信号(1D), fft的输出为 (..., T), x_plot为(T,)
    if x.ndim > 1:
        # 选择同样方式 reshape 以一致性
        X_full_plot = torch.fft.fft(x_plot)
        X_rfft_plot = torch.fft.rfft(x_plot)
        diff_plot = torch.abs(X_full_plot[:X_rfft_plot.shape[-1]] - X_rfft_plot)
    else:
        X_full_plot = X_full
        X_rfft_plot = X_rfft
        diff_plot = diff

    # 幅值谱对比
    axes[0, 0].plot(freqs_full, torch.abs(X_full_plot).detach().cpu().numpy(), 'b-', 
                    linewidth=1.5, alpha=0.7, label='FFT (full)')
    axes[0, 0].plot(freqs_rfft, torch.abs(X_rfft_plot).detach().cpu().numpy(), 'r--', 
                    linewidth=2, alpha=0.8, label='RFFT (positive only)')
    axes[0, 0].set_xlabel('Frequency (GHz)', fontweight='bold')
    axes[0, 0].set_ylabel('|X(f)|', fontweight='bold')
    axes[0, 0].set_title('Amplitude Spectrum Comparison', fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 仅正频率放大
    axes[0, 1].plot(freqs_rfft, torch.abs(X_rfft_plot).detach().cpu().numpy(), 'g-', linewidth=2)
    axes[0, 1].set_xlabel('Frequency (GHz)', fontweight='bold')
    axes[0, 1].set_ylabel('|X(f)|', fontweight='bold')
    axes[0, 1].set_title('RFFT: Positive Frequencies Only', fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 差异分析
    axes[1, 0].semilogy(freqs_rfft, diff_plot.detach().cpu().numpy(), 'purple', linewidth=1.5)
    axes[1, 0].set_xlabel('Frequency (GHz)', fontweight='bold')
    axes[1, 0].set_ylabel('|FFT - RFFT|', fontweight='bold')
    axes[1, 0].set_title('Absolute Difference (Positive Frequencies)', fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 重建信号对比(注意重建也是一维)
    t_plot = np.arange(min(500, T)) * dt
    if x.ndim > 1:
        # 匹配
        x_ifft_plot = torch.fft.ifft(torch.fft.fft(x_plot)).real
        x_irfft_plot = torch.fft.irfft(torch.fft.rfft(x_plot), n=T)
    else:
        x_ifft_plot = x_ifft
        x_irfft_plot = x_irfft

    axes[1, 1].plot(t_plot, x_plot[:len(t_plot)].detach().cpu().numpy(), 'b-', 
                    linewidth=1.5, alpha=0.7, label='Original')
    axes[1, 1].plot(t_plot, x_ifft_plot[:len(t_plot)].detach().cpu().numpy(), 'r--', 
                    linewidth=1, alpha=0.8, label='IFFT')
    axes[1, 1].plot(t_plot, x_irfft_plot[:len(t_plot)].detach().cpu().numpy(), 'g:', 
                    linewidth=2, alpha=0.6, label='IRFFT')
    axes[1, 1].set_xlabel('Time (ps)', fontweight='bold')
    axes[1, 1].set_ylabel('Amplitude', fontweight='bold')
    axes[1, 1].set_title('Signal Reconstruction', fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('fft_vs_rfft_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\n✅ 对比图已保存: fft_vs_rfft_comparison.png")
    plt.show()
    
    return {
        'storage_efficiency': X_rfft.shape[-1] / X_full.shape[-1],
        'max_difference': diff.max().item(),
        'mean_difference': diff.mean().item(),
        'reconstruction_error_fft': reconstruction_error_fft.item(),
        'reconstruction_error_rfft': reconstruction_error_rfft.item()
    }


# ============================================================
# 主测试脚本
# ============================================================

def run_all_tests():
    """
    运行所有测试
    """
    print("\n" + "🚀"*35)
    print(" " * 20 + "完整测试套件")
    print("🚀"*35 + "\n")
    
    results = {}
    
    # 测试1: 合成实值信号
    try:
        results['test1_synthetic'] = test_with_synthetic_signal()
        print("\n✅ 测试1完成\n")
    except Exception as e:
        print(f"\n❌ 测试1失败: {e}\n")
    
    # 测试2: 复数信号
    try:
        results['test2_complex'] = test_with_complex_signal()
        print("\n✅ 测试2完成\n")
    except Exception as e:
        print(f"\n❌ 测试2失败: {e}\n")
    
    # 测试3: 带噪声信号
    try:
        results['test3_noisy'] = test_with_noisy_signal()
        print("\n✅ 测试3完成\n")
    except Exception as e:
        print(f"\n❌ 测试3失败: {e}\n")
    
    # 测试4: 蛋白质轨迹
    try:
        results['test4_protein'] = test_with_protein_trajectory()
        print("\n✅ 测试4完成\n")
    except Exception as e:
        print(f"\n❌ 测试4失败: {e}\n")
    
    # 测试5: FFT实现对比
    try:
        results['test5_implementations'] = compare_fft_implementations()
        print("\n✅ 测试5完成\n")
    except Exception as e:
        print(f"\n❌ 测试5失败: {e}\n")
    
    # 总结
    print("\n" + "="*70)
    print("📊 总结")
    print("="*70)
    
    for test_name, result in results.items():
        if isinstance(result, dict) and 'is_hermitian' in result:
            status = "✅" if result['is_hermitian'] else "❌"
            print(f"{status} {test_name}: {'满足' if result['is_hermitian'] else '不满足'}厄米对称性")
    
    return results


# ============================================================
# 实用工具函数
# ============================================================

def quick_check_hermitian(x: torch.Tensor, tolerance: float = 1e-6) -> bool:
    """
    快速检查信号FFT是否满足厄米对称性
    
    Parameters:
    -----------
    x : torch.Tensor
        时间域信号
    tolerance : float
        容差阈值
        
    Returns:
    --------
    bool : 是否满足厄米对称性
    """
    X = torch.fft.fft(x, dim=-1)
    T = X.shape[-1]
    
    if T % 2 == 0:
        pos_idx = torch.arange(1, T // 2)
        neg_idx = torch.arange(T - 1, T // 2, -1)
    else:
        pos_idx = torch.arange(1, T // 2 + 1)
        neg_idx = torch.arange(T - 1, T // 2 - 1, -1)
    
    X_pos = X[..., pos_idx]
    X_neg = X[..., neg_idx]
    
    diff = torch.abs(X_pos - torch.conj(X_neg))
    rel_error = diff / (torch.abs(X_pos) + 1e-10)
    
    return rel_error.max().item() < tolerance


def ensure_real_signal(x: torch.Tensor, method: str = 'real') -> torch.Tensor:
    """
    确保信号为实值
    
    Parameters:
    -----------
    x : torch.Tensor
        输入信号（可能包含微小虚部）
    method : str
        'real': 只取实部
        'abs': 取模
        'ifft': 通过FFT->IFFT强制厄米对称
        
    Returns:
    --------
    torch.Tensor : 实值信号
    """
    if method == 'real':
        return x.real if torch.is_complex(x) else x
    
    elif method == 'abs':
        return torch.abs(x)
    
    elif method == 'ifft':
        # 使用rfft->irfft确保实值
        X = torch.fft.rfft(x.real if torch.is_complex(x) else x, dim=-1)
        return torch.fft.irfft(X, n=x.shape[-1], dim=-1)
    
    else:
        raise ValueError(f"Unknown method: {method}")


def batch_hermitian_check(trajectories: torch.Tensor, 
                          dt: float,
                          verbose: bool = True) -> Dict:
    """
    批量检查多条轨迹的厄米对称性
    
    Parameters:
    -----------
    trajectories : torch.Tensor
        形状 (N_batch, T) 或 (N_batch, N_atoms, 3, T)
    dt : float
        时间步长 (ps)
    verbose : bool
        是否打印详细信息
        
    Returns:
    --------
    Dict : 统计结果
    """
    if trajectories.ndim > 2:
        # 展平为 (N, T)
        original_shape = trajectories.shape
        trajectories = trajectories.reshape(-1, trajectories.shape[-1])
        if verbose:
            print(f"展平形状: {original_shape} -> {trajectories.shape}")
    
    N_batch = trajectories.shape[0]
    
    hermitian_count = 0
    max_violations = []
    mean_violations = []
    
    if verbose:
        print(f"\n检查 {N_batch} 条轨迹...")
    
    for i in range(N_batch):
        x = trajectories[i]
        analyzer = HermitianSymmetryAnalyzer(x, dt)
        X, _ = analyzer.compute_fft()
        results = analyzer.check_hermitian_symmetry(X.unsqueeze(0))
        
        if results['is_hermitian']:
            hermitian_count += 1
        
        max_violations.append(results['max_relative_violation'])
        mean_violations.append(results['mean_relative_violation'])
        
        if verbose and (i + 1) % max(1, N_batch // 10) == 0:
            print(f"  进度: {i+1}/{N_batch} ({(i+1)/N_batch*100:.1f}%)")
    
    max_violations = np.array(max_violations)
    mean_violations = np.array(mean_violations)
    
    results = {
        'total': N_batch,
        'hermitian_count': hermitian_count,
        'hermitian_percentage': hermitian_count / N_batch * 100,
        'max_violations': {
            'mean': max_violations.mean(),
            'std': max_violations.std(),
            'min': max_violations.min(),
            'max': max_violations.max()
        },
        'mean_violations': {
            'mean': mean_violations.mean(),
            'std': mean_violations.std(),
            'min': mean_violations.min(),
            'max': mean_violations.max()
        }
    }
    
    if verbose:
        print(f"\n" + "="*70)
        print(f"批量检查结果:")
        print(f"  总数: {results['total']}")
        print(f"  满足厄米对称: {results['hermitian_count']} ({results['hermitian_percentage']:.1f}%)")
        print(f"  最大违反度统计:")
        print(f"    平均: {results['max_violations']['mean']:.6e}")
        print(f"    标准差: {results['max_violations']['std']:.6e}")
        print(f"    范围: [{results['max_violations']['min']:.6e}, {results['max_violations']['max']:.6e}]")
        print("="*70)
    
    return results


def save_analysis_report(results: Dict, filename: str = 'hermitian_analysis_report.txt'):
    """
    保存分析报告到文件
    """
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("蛋白质动力学FFT厄米对称性分析报告\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"生成时间: {np.datetime64('now')}\n\n")
        
        f.write("分析结果:\n")
        f.write("-"*70 + "\n")
        
        def write_dict(d, indent=0):
            for key, value in d.items():
                if isinstance(value, dict):
                    f.write("  " * indent + f"{key}:\n")
                    write_dict(value, indent + 1)
                else:
                    f.write("  " * indent + f"{key}: {value}\n")
        
        write_dict(results)
        
        f.write("\n" + "="*70 + "\n")
        f.write("报告结束\n")
        f.write("="*70 + "\n")
    
    print(f"✅ 报告已保存: {filename}")


# ============================================================
# 使用示例
# ============================================================

if __name__ == "__main__":
    
    print("""
    ╔═══════════════════════════════════════════════════════════════════╗
    ║                                                                   ║
    ║    蛋白质动力学FFT厄米对称性分析工具                               ║
    ║    Hermitian Symmetry Analyzer for Protein Dynamics              ║
    ║                                                                   ║
    ╚═══════════════════════════════════════════════════════════════════╝
    """)
    traj_coord_path = "./test/data/1ear_A_R1_no_water_frames_coords.npy"
    arr = np.load(traj_coord_path)  # (T, N, 3) numpy数组
    traj = torch.from_numpy(arr).to(torch.float64)
    print(f"traj.shape: {traj.shape}")
    T, N, _ = traj.shape
    first_frame_positions = traj[0]  # shape is (N, 3)
    masses = torch.ones(N)
    dt = 10
    x = traj.permute(1, 2, 0) # (N, 3, T)
    # # 示例1: 快速检查单个信号
    # print("\n" + "📍"*35)
    # print("示例1: 快速检查")
    # print("📍"*35)
    
    # T = 1000
    # dt = 0.001
    # t = torch.arange(T, dtype=torch.float32) * dt
    # t_seconds = t * 1e-12
    # x = torch.sin(2 * np.pi * 5e9 * t_seconds)
    
    # is_hermitian = quick_check_hermitian(x)
    # print(f"\n信号是否满足厄米对称性: {is_hermitian}")
    
    # 示例2: 详细分析
    print("\n" + "📍"*35)
    print("示例2: 详细分析")
    print("📍"*35)
    
    analyzer = HermitianSymmetryAnalyzer(x, dt)
    results = analyzer.full_analysis(visualize=True)
    
    # 示例3: 高级诊断
    print("\n" + "📍"*35)
    print("示例3: 高级诊断")
    print("📍"*35)
    
    diagnostics = advanced_symmetry_diagnostics(x, dt)
    
    # 示例4: FFT vs RFFT
    print("\n" + "📍"*35)
    print("示例4: FFT vs RFFT对比")
    print("📍"*35)
    
    comparison = compare_fft_with_rfft(x, dt)
    
    # # 示例5: 批量检查
    # print("\n" + "📍"*35)
    # print("示例5: 批量检查")
    # print("📍"*35)
    
    # # 生成多条轨迹
    # N_batch = 50
    # T = 1000
    # dt = 0.001
    # t = torch.arange(T, dtype=torch.float32) * dt
    # t_seconds = t * 1e-12
    
    # trajectories = torch.zeros(N_batch, T)
    # for i in range(N_batch):
    #     freq = (1 + 0.1 * i) * 1e9
    #     trajectories[i] = torch
