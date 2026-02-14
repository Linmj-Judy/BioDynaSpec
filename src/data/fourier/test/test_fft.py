import torch
import math
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Literal
import numpy as np

def _fft_full(x: torch.Tensor, dt: float, norm: Literal["backward", "ortho", "forward"], 
              n_fft: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Full complex FFT along last dimension (time). Returns spectrum, freq (Hz), ang_freq (rad/s).
    x: real-valued (N,3,T)
    dt: time step in ps
    """
    T = x.shape[-1]
    if n_fft is None:
        n_fft = T
    X = torch.fft.fft(x, n=n_fft, dim=-1, norm=norm)
    
    # 频率单位：Hz (注意dt单位是ps，需要转换)
    freqs = torch.fft.fftfreq(n_fft, d=dt * 1e-12).to(device=x.device, dtype=x.real.dtype)  # Hz
    omega = 2.0 * math.pi * freqs  # rad/s
    return X, freqs, omega


def compute_velocity_acceleration_spectra_comparison(
    x_for_spectrum: torch.Tensor,  # (N, 3, T)
    dt: float,  # ps
    norm: Literal["backward", "ortho", "forward"] = "ortho"
) -> dict:
    """
    完整对比频域微分 vs 时域差分，包含时域重建验证
    """
    T = x_for_spectrum.shape[-1]
    N = x_for_spectrum.shape[0]
    
    results = {}
    
    # ============================================================
    # 方法1：频域微分（理论正确）
    # ============================================================
    X, freqs, omega = _fft_full(x_for_spectrum, dt=dt, norm=norm, n_fft=T)
    omega_expanded = omega.view(1, 1, -1)
    
    # 速度谱 = iω·X(ω)，单位：Å·rad/s
    V_freq = 1j * omega_expanded * X
    
    # 加速度谱 = -ω²·X(ω)，单位：Å·(rad/s)²
    A_freq = -(omega_expanded ** 2) * X
    
    # IFFT回到时域
    v_freq_time = torch.fft.ifft(V_freq, dim=-1, norm=norm).real  # Å/s
    a_freq_time = torch.fft.ifft(A_freq, dim=-1, norm=norm).real  # Å/s²
    
    # 转换单位：Å/s -> Å/ps, Å/s² -> Å/ps²
    v_freq_time = v_freq_time * 1e-12  # Å/ps
    a_freq_time = a_freq_time * 1e-24  # Å/ps²
    
    results['method1'] = {
        'X': X, 'V': V_freq, 'A': A_freq,
        'v_time': v_freq_time, 'a_time': a_freq_time
    }
    
    # ============================================================
    # 方法2：时域差分
    # ============================================================
    # 计算速度：中心差分，单位 Å/ps
    v_diff = torch.zeros_like(x_for_spectrum)
    v_diff[..., 0] = (x_for_spectrum[..., 1] - x_for_spectrum[..., 0]) / dt
    v_diff[..., 1:-1] = (x_for_spectrum[..., 2:] - x_for_spectrum[..., :-2]) / (2 * dt)
    v_diff[..., -1] = (x_for_spectrum[..., -1] - x_for_spectrum[..., -2]) / dt
    
    # 计算加速度：中心差分，单位 Å/ps²
    a_diff = torch.zeros_like(x_for_spectrum)
    a_diff[..., 0] = (v_diff[..., 1] - v_diff[..., 0]) / dt
    a_diff[..., 1:-1] = (v_diff[..., 2:] - v_diff[..., :-2]) / (2 * dt)
    a_diff[..., -1] = (v_diff[..., -1] - v_diff[..., -2]) / dt
    
    # 对时域速度和加速度做FFT
    V_diff, _, _ = _fft_full(v_diff * 1e12, dt=dt, norm=norm, n_fft=T)  # 转换为Å/s
    A_diff, _, _ = _fft_full(a_diff * 1e24, dt=dt, norm=norm, n_fft=T)  # 转换为Å/s²
    
    results['method2'] = {
        'V': V_diff, 'A': A_diff,
        'v_time': v_diff, 'a_time': a_diff
    }
    
    results['meta'] = {
        'freqs': freqs,
        'omega': omega,
        'dt': dt,
        'T': T,
        'N': N
    }
    
    return results


def verify_with_analytical_signal(
    freq_test_GHz: float = 1.0,
    T: int = 1000,
    dt: float = 0.001,  # ps
    N: int = 10,
    plot: bool = True
):
    """
    使用解析信号验证：x(t) = A*sin(2πft)
    """
    print("="*70)
    print(f"🧪 解析信号验证 (频率={freq_test_GHz} GHz)")
    print("="*70)
    
    # 生成信号
    t = torch.arange(T, dtype=torch.float32) * dt  # ps
    freq_Hz = freq_test_GHz * 1e9  # Hz
    omega_rad = 2 * math.pi * freq_Hz  # rad/s
    
    # 计算周期数
    period_s = 1 / freq_Hz  # 秒
    period_ps = period_s * 1e12  # ps
    n_periods = t[-1].item() / period_ps
    
    print(f"\n信号参数:")
    print(f"  时间范围: 0 - {t[-1].item():.3f} ps")
    print(f"  时间步长: {dt} ps = {dt*1000:.3f} fs")
    print(f"  采样频率: {1/(dt*1e-12)/1e9:.2f} GHz")
    print(f"  信号频率: {freq_Hz/1e9:.2f} GHz")
    print(f"  信号周期: {period_ps:.6f} ps")
    print(f"  周期数: {n_periods:.2f}")
    
    A_amplitude = 1.0  # Å
    
    # ✅ 正确的信号生成
    # x(t) = A*sin(2πf*t) where t in seconds
    t_seconds = t * 1e-12  # 转换为秒
    x_true = A_amplitude * torch.sin(2 * math.pi * freq_Hz * t_seconds)  # Å
    x_test = x_true.view(1, 1, T).expand(N, 3, T)
    
    # 理论的速度和加速度
    # v(t) = dx/dt = A*2πf*cos(2πft), 单位 Å/s
    v_true_si = A_amplitude * 2 * math.pi * freq_Hz * torch.cos(2 * math.pi * freq_Hz * t_seconds)
    v_true = v_true_si * 1e-12  # 转换为 Å/ps
    
    # a(t) = dv/dt = -A*(2πf)²*sin(2πft), 单位 Å/s²
    a_true_si = -A_amplitude * (2 * math.pi * freq_Hz) ** 2 * torch.sin(2 * math.pi * freq_Hz * t_seconds)
    a_true = a_true_si * 1e-24  # 转换为 Å/ps²
    
    # 运行对比
    results = compute_velocity_acceleration_spectra_comparison(x_test, dt=dt, norm='ortho')
    
    # 提取结果
    v_freq_time = results['method1']['v_time'][0, 0, :]
    v_diff_time = results['method2']['v_time'][0, 0, :]
    
    a_freq_time = results['method1']['a_time'][0, 0, :]
    a_diff_time = results['method2']['a_time'][0, 0, :]
    
    # ============================================================
    # 误差分析
    # ============================================================
    print("\n" + "="*70)
    print("📊 时域重建误差分析")
    print("="*70)
    
    # 速度误差
    v_error_freq = torch.sqrt(torch.mean((v_freq_time - v_true) ** 2))
    v_error_diff = torch.sqrt(torch.mean((v_diff_time - v_true) ** 2))
    v_rms_true = torch.sqrt(torch.mean(v_true ** 2))
    
    print(f"\n速度 (Å/ps):")
    print(f"  理论值 RMS: {v_rms_true.item():.6e}")
    print(f"  频域方法 RMSE: {v_error_freq.item():.6e} ({v_error_freq/v_rms_true*100:.4f}%)")
    print(f"  时域差分 RMSE: {v_error_diff.item():.6e} ({v_error_diff/v_rms_true*100:.4f}%)")
    
    if v_error_freq > 0:
        print(f"  误差比 (差分/频域): {v_error_diff/v_error_freq:.2f}x")
    
    # 加速度误差
    a_error_freq = torch.sqrt(torch.mean((a_freq_time - a_true) ** 2))
    a_error_diff = torch.sqrt(torch.mean((a_diff_time - a_true) ** 2))
    a_rms_true = torch.sqrt(torch.mean(a_true ** 2))
    
    print(f"\n加速度 (Å/ps²):")
    print(f"  理论值 RMS: {a_rms_true.item():.6e}")
    print(f"  频域方法 RMSE: {a_error_freq.item():.6e} ({a_error_freq/a_rms_true*100:.4f}%)")
    print(f"  时域差分 RMSE: {a_error_diff.item():.6e} ({a_error_diff/a_rms_true*100:.4f}%)")
    
    if a_error_freq > 0:
        print(f"  误差比 (差分/频域): {a_error_diff/a_error_freq:.2f}x")
    
    # ============================================================
    # 频域验证
    # ============================================================
    print("\n" + "="*70)
    print("📊 频域幅值验证")
    print("="*70)
    
    freqs = results['meta']['freqs']
    
    # 找到测试频率的索引
    freq_idx = torch.argmin(torch.abs(freqs - freq_Hz))
    actual_freq = freqs[freq_idx].item()
    
    print(f"\n目标频率: {freq_Hz/1e9:.6f} GHz")
    print(f"实际频率: {actual_freq/1e9:.6f} GHz")
    print(f"频率误差: {abs(actual_freq - freq_Hz)/freq_Hz*100:.4f}%")
    
    # 频域幅值
    X_amp = torch.abs(results['method1']['X'][0, 0, freq_idx])
    V_freq_amp = torch.abs(results['method1']['V'][0, 0, freq_idx])
    V_diff_amp = torch.abs(results['method2']['V'][0, 0, freq_idx])
    A_freq_amp = torch.abs(results['method1']['A'][0, 0, freq_idx])
    A_diff_amp = torch.abs(results['method2']['A'][0, 0, freq_idx])    
    # 理论幅值
    # 对于 x(t) = A*sin(ωt)，FFT后在 ±f 处各有幅值 A/2（如果用ortho归一化）
    # 实际需要考虑归一化方式
    X_theory = A_amplitude * T / np.sqrt(T) / 2  # ortho norm
    V_theory = X_theory * omega_rad
    A_theory = X_theory * (omega_rad ** 2)
    
    print(f"\n位移谱 (在 {freq_Hz/1e9:.2f} GHz):")
    print(f"  实际幅值: {X_amp.item():.6e} Å")
    print(f"  理论幅值: {X_theory:.6e} Å")
    print(f"  相对误差: {abs(X_amp.item() - X_theory) / X_theory * 100:.2f}%")
    
    print(f"\n速度谱 (在 {freq_Hz/1e9:.2f} GHz):")
    print(f"  频域方法幅值: {V_freq_amp.item():.6e} Å·rad/s")
    print(f"  时域差分幅值: {V_diff_amp.item():.6e} Å·rad/s")
    print(f"  理论幅值: {V_theory:.6e} Å·rad/s")
    print(f"  频域方法误差: {abs(V_freq_amp.item() - V_theory) / V_theory * 100:.2f}%")
    print(f"  时域差分误差: {abs(V_diff_amp.item() - V_theory) / V_theory * 100:.2f}%")
    
    print(f"\n加速度谱 (在 {freq_Hz/1e9:.2f} GHz):")
    print(f"  频域方法幅值: {A_freq_amp.item():.6e} Å·(rad/s)²")
    print(f"  时域差分幅值: {A_diff_amp.item():.6e} Å·(rad/s)²")
    print(f"  理论幅值: {A_theory:.6e} Å·(rad/s)²")
    print(f"  频域方法误差: {abs(A_freq_amp.item() - A_theory) / A_theory * 100:.2f}%")
    print(f"  时域差分误差: {abs(A_diff_amp.item() - A_theory) / A_theory * 100:.2f}%")
    
    # ============================================================
    # 可视化
    # ============================================================
    if plot:
        fig, axes = plt.subplots(3, 2, figsize=(16, 12))
        
        # 选择显示的时间窗口（前几个周期）
        n_cycles = 5
        t_window = int(min(T, n_cycles / (freq_Hz * dt * 1e-12)))
        t_plot = t[:t_window].numpy()
        
        # 位移
        axes[0, 0].plot(t_plot, x_true[:t_window].numpy(), 'k-', linewidth=2, label='True', alpha=0.7)
        axes[0, 0].set_ylabel('Displacement (Å)', fontweight='bold')
        axes[0, 0].set_title('Position x(t)', fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 速度
        axes[1, 0].plot(t_plot, v_true[:t_window].numpy(), 'k-', linewidth=2, label='True', alpha=0.7)
        axes[1, 0].plot(t_plot, v_freq_time[:t_window].numpy(), 'b--', linewidth=1.5, label='Frequency Domain', alpha=0.8)
        axes[1, 0].plot(t_plot, v_diff_time[:t_window].numpy(), 'r:', linewidth=1.5, label='Time Difference', alpha=0.8)
        axes[1, 0].set_ylabel('Velocity (Å/ps)', fontweight='bold')
        axes[1, 0].set_title('Velocity v(t)', fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 加速度
        axes[2, 0].plot(t_plot, a_true[:t_window].numpy(), 'k-', linewidth=2, label='True', alpha=0.7)
        axes[2, 0].plot(t_plot, a_freq_time[:t_window].numpy(), 'b--', linewidth=1.5, label='Frequency Domain', alpha=0.8)
        axes[2, 0].plot(t_plot, a_diff_time[:t_window].numpy(), 'r:', linewidth=1.5, label='Time Difference', alpha=0.8)
        axes[2, 0].set_xlabel('Time (ps)', fontweight='bold')
        axes[2, 0].set_ylabel('Acceleration (Å/ps²)', fontweight='bold')
        axes[2, 0].set_title('Acceleration a(t)', fontweight='bold')
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)
        
        # 速度误差
        axes[1, 1].plot(t_plot, np.abs(v_freq_time[:t_window].numpy() - v_true[:t_window].numpy()), 
                       'b-', linewidth=1.5, label='Frequency Domain')
        axes[1, 1].plot(t_plot, np.abs(v_diff_time[:t_window].numpy() - v_true[:t_window].numpy()), 
                       'r-', linewidth=1.5, label='Time Difference')
        axes[1, 1].set_ylabel('Absolute Error', fontweight='bold')
        axes[1, 1].set_title('Velocity Error |v_pred - v_true|', fontweight='bold')
        axes[1, 1].set_yscale('log')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # 加速度误差
        axes[2, 1].plot(t_plot, np.abs(a_freq_time[:t_window].numpy() - a_true[:t_window].numpy()), 
                       'b-', linewidth=1.5, label='Frequency Domain')
        axes[2, 1].plot(t_plot, np.abs(a_diff_time[:t_window].numpy() - a_true[:t_window].numpy()), 
                       'r-', linewidth=1.5, label='Time Difference')
        axes[2, 1].set_xlabel('Time (ps)', fontweight='bold')
        axes[2, 1].set_ylabel('Absolute Error', fontweight='bold')
        axes[2, 1].set_title('Acceleration Error |a_pred - a_true|', fontweight='bold')
        axes[2, 1].set_yscale('log')
        axes[2, 1].legend()
        axes[2, 1].grid(True, alpha=0.3)
        
        # 频谱对比
        freqs_GHz = freqs.numpy() / 1e9
        # 只显示正频率
        pos_freq_mask = freqs_GHz > 0
        freqs_pos = freqs_GHz[pos_freq_mask]
        
        V_freq_spectrum = torch.abs(results['method1']['V'][0, 0, :]).numpy()[pos_freq_mask]
        V_diff_spectrum = torch.abs(results['method2']['V'][0, 0, :]).numpy()[pos_freq_mask]
        
        axes[0, 1].semilogy(freqs_pos, V_freq_spectrum, 'b-', linewidth=1.5, label='Frequency Domain', alpha=0.8)
        axes[0, 1].semilogy(freqs_pos, V_diff_spectrum, 'r--', linewidth=1.5, label='Time Difference', alpha=0.8)
        axes[0, 1].axvline(freq_test_GHz, color='k', linestyle=':', linewidth=2, alpha=0.5, label=f'Test freq = {freq_test_GHz} GHz')
        axes[0, 1].set_xlabel('Frequency (GHz)', fontweight='bold')
        axes[0, 1].set_ylabel('|V(f)|', fontweight='bold')
        axes[0, 1].set_title('Velocity Spectrum', fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_xlim([0, min(10*freq_test_GHz, freqs_pos.max())])
        
        plt.tight_layout()
        plt.savefig('velocity_acceleration_verification.png', dpi=300, bbox_inches='tight')
        print(f"\n✅ 图表已保存: velocity_acceleration_verification.png")
        plt.show()
    
    return results


def test_multiple_frequencies(plot: bool = True):
    """
    测试多个频率下的误差
    """
    print("\n" + "="*70)
    print("🧪 多频率测试")
    print("="*70)
    
    freq_list_GHz = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    T = 2000
    dt = 0.0001  # 0.1 fs
    N = 1
    
    results_summary = {
        'freq': [],
        'v_error_freq': [],
        'v_error_diff': [],
        'a_error_freq': [],
        'a_error_diff': []
    }
    
    for freq_GHz in freq_list_GHz:
        print(f"\n测试频率: {freq_GHz} GHz")
        
        # 生成信号
        t = torch.arange(T, dtype=torch.float32) * dt
        freq_Hz = freq_GHz * 1e9
        omega_rad = 2 * math.pi * freq_Hz
        A_amplitude = 1.0
        
        x_true = A_amplitude * torch.sin(omega_rad * t * 1e-12)
        x_test = x_true.view(1, 1, T).expand(N, 3, T)
        
        v_true = A_amplitude * omega_rad * torch.cos(omega_rad * t * 1e-12) * 1e-12
        a_true = -A_amplitude * (omega_rad ** 2) * torch.sin(omega_rad * t * 1e-12) * 1e-24
        
        # 计算
        res = compute_velocity_acceleration_spectra_comparison(x_test, dt=dt, norm='ortho')
        
        v_freq_time = res['method1']['v_time'][0, 0, :]
        v_diff_time = res['method2']['v_time'][0, 0, :]
        a_freq_time = res['method1']['a_time'][0, 0, :]
        a_diff_time = res['method2']['a_time'][0, 0, :]
        
        # 误差
        v_error_freq = torch.sqrt(torch.mean((v_freq_time - v_true) ** 2)) / torch.sqrt(torch.mean(v_true ** 2))
        v_error_diff = torch.sqrt(torch.mean((v_diff_time - v_true) ** 2)) / torch.sqrt(torch.mean(v_true ** 2))
        a_error_freq = torch.sqrt(torch.mean((a_freq_time - a_true) ** 2)) / torch.sqrt(torch.mean(a_true ** 2))
        a_error_diff = torch.sqrt(torch.mean((a_diff_time - a_true) ** 2)) / torch.sqrt(torch.mean(a_true ** 2))
        
        results_summary['freq'].append(freq_GHz)
        results_summary['v_error_freq'].append(v_error_freq.item() * 100)
        results_summary['v_error_diff'].append(v_error_diff.item() * 100)
        results_summary['a_error_freq'].append(a_error_freq.item() * 100)
        results_summary['a_error_diff'].append(a_error_diff.item() * 100)
        
        print(f"  速度误差: 频域={v_error_freq*100:.2f}%, 差分={v_error_diff*100:.2f}%")
        print(f"  加速度误差: 频域={a_error_freq*100:.2f}%, 差分={a_error_diff*100:.2f}%")
    
    # 可视化
    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 速度误差
        axes[0].semilogx(results_summary['freq'], results_summary['v_error_freq'], 
                        'bo-', linewidth=2, markersize=8, label='Frequency Domain')
        axes[0].semilogx(results_summary['freq'], results_summary['v_error_diff'], 
                        'rs-', linewidth=2, markersize=8, label='Time Difference')
        axes[0].set_xlabel('Frequency (GHz)', fontweight='bold', fontsize=12)
        axes[0].set_ylabel('Relative Error (%)', fontweight='bold', fontsize=12)
        axes[0].set_title('Velocity Reconstruction Error', fontweight='bold', fontsize=13)
        axes[0].legend(fontsize=11)
        axes[0].grid(True, alpha=0.3)
        
        # 加速度误差
        axes[1].semilogx(results_summary['freq'], results_summary['a_error_freq'], 
                        'bo-', linewidth=2, markersize=8, label='Frequency Domain')
        axes[1].semilogx(results_summary['freq'], results_summary['a_error_diff'], 
                        'rs-', linewidth=2, markersize=8, label='Time Difference')
        axes[1].set_xlabel('Frequency (GHz)', fontweight='bold', fontsize=12)
        axes[1].set_ylabel('Relative Error (%)', fontweight='bold', fontsize=12)
        axes[1].set_title('Acceleration Reconstruction Error', fontweight='bold', fontsize=13)
        axes[1].legend(fontsize=11)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('multi_frequency_error_comparison.png', dpi=300, bbox_inches='tight')
        print(f"\n✅ 图表已保存: multi_frequency_error_comparison.png")
        plt.show()
    
    return results_summary


def test_edge_cases():
    """
    测试边界情况
    """
    print("\n" + "="*70)
    print("🧪 边界情况测试")
    print("="*70)
    
    # 1. 低频信号（慢运动）
    print("\n1️⃣ 低频信号 (0.01 GHz = 100 MHz)")
    verify_with_analytical_signal(freq_test_GHz=0.01, T=10000, dt=0.001, plot=False)
    
    # 2. 高频信号（快速振动）
    print("\n2️⃣ 高频信号 (50 GHz)")
    verify_with_analytical_signal(freq_test_GHz=50.0, T=1000, dt=0.00001, plot=False)
    
    # 3. 混合频率
    print("\n3️⃣ 混合频率信号")
    T = 2000
    dt = 0.0001
    t = torch.arange(T, dtype=torch.float32) * dt
    
    # x(t) = A1*sin(ω1*t) + A2*sin(ω2*t)
    f1, f2 = 1e9, 5e9  # 1 GHz, 5 GHz
    A1, A2 = 1.0, 0.5
    omega1 = 2 * math.pi * f1
    omega2 = 2 * math.pi * f2
    
    x_mixed = A1 * torch.sin(omega1 * t * 1e-12) + A2 * torch.sin(omega2 * t * 1e-12)
    x_test = x_mixed.view(1, 1, T).expand(1, 3, T)
    
    v_true = (A1 * omega1 * torch.cos(omega1 * t * 1e-12) + 
              A2 * omega2 * torch.cos(omega2 * t * 1e-12)) * 1e-12
    
    res = compute_velocity_acceleration_spectra_comparison(x_test, dt=dt, norm='ortho')
    
    v_freq_time = res['method1']['v_time'][0, 0, :]
    v_diff_time = res['method2']['v_time'][0, 0, :]
    
    v_error_freq = torch.sqrt(torch.mean((v_freq_time - v_true) ** 2)) / torch.sqrt(torch.mean(v_true ** 2))
    v_error_diff = torch.sqrt(torch.mean((v_diff_time - v_true) ** 2)) / torch.sqrt(torch.mean(v_true ** 2))
    
    print(f"  速度误差: 频域={v_error_freq*100:.2f}%, 差分={v_error_diff*100:.2f}%")


# ============================================================
# 主测试
# ============================================================
if __name__ == "__main__":
    print("\n" + "🚀"*35)
    print("完整验证：频域微分 vs 时域差分")
    print("🚀"*35 + "\n")
    
    # 1. 单频率详细测试
    results = verify_with_analytical_signal(
        freq_test_GHz=1.0,
        T=1000,
        dt=0.001,
        N=10,
        plot=True
    )
    
    # 2. 多频率测试
    results_multi = test_multiple_frequencies(plot=True)
    
    # 3. 边界情况
    test_edge_cases()
