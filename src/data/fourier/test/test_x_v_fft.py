import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq

def generate_test_trajectory(T, N_atoms, dt):
    """
    生成测试用的原子轨迹数据
    使用多个频率的正弦波叠加，这样我们知道真实的速度
    """
    t = np.arange(T) * dt
    
    # 定义几个测试频率 (Hz)
    frequencies = [0.5, 1.0, 2.0, 3.5]  
    amplitudes = [1.0, 0.5, 0.3, 0.2]
    phases = [0, np.pi/4, np.pi/2, np.pi/3]
    
    # 初始化位置和真实速度
    positions = np.zeros((T, N_atoms, 3))
    true_velocities = np.zeros((T, N_atoms, 3))
    
    # 为每个原子和每个维度生成轨迹
    for atom in range(N_atoms):
        for dim in range(3):
            # 每个原子每个维度使用稍微不同的参数
            offset = (atom * 3 + dim) * 0.1
            
            for freq, amp, phase in zip(frequencies, amplitudes, phases):
                omega = 2 * np.pi * freq
                # 位置: x(t) = A * sin(ωt + φ)
                positions[:, atom, dim] += amp * np.sin(omega * t + phase + offset)
                # 真实速度: v(t) = A * ω * cos(ωt + φ)
                true_velocities[:, atom, dim] += amp * omega * np.cos(omega * t + phase + offset)
    
    return positions, true_velocities, t

def method1_frequency_domain_diff(positions, dt):
    """
    方法1: 频域差分
    先FFT位移 -> 频域乘以 iω -> IFFT得到速度
    """
    T = positions.shape[0]
    N_atoms = positions.shape[1]
    
    # 计算频率
    freqs = fftfreq(T, dt)
    omega = 2j * np.pi * freqs  # iω
    
    velocities = np.zeros_like(positions)
    velocity_spectrum = np.zeros((T, N_atoms, 3), dtype=complex)
    
    for atom in range(N_atoms):
        for dim in range(3):
            # FFT位移得到位移谱
            pos_spectrum = fft(positions[:, atom, dim])
            
            # 频域差分: V(ω) = iω * X(ω)
            vel_spectrum = omega * pos_spectrum
            velocity_spectrum[:, atom, dim] = vel_spectrum
            
            # IFFT回时域
            velocities[:, atom, dim] = np.real(ifft(vel_spectrum))
    
    return velocities, velocity_spectrum

def method2_time_domain_diff(positions, dt):
    """
    方法2: 时域差分
    先用中心差分计算速度 -> FFT得到速度谱
    """
    T = positions.shape[0]
    N_atoms = positions.shape[1]
    
    velocities = np.zeros_like(positions)
    velocity_spectrum = np.zeros((T, N_atoms, 3), dtype=complex)
    
    for atom in range(N_atoms):
        for dim in range(3):
            pos = positions[:, atom, dim]
            
            # 中心差分: v(t) = [x(t+dt) - x(t-dt)] / (2*dt)
            velocities[1:-1, atom, dim] = (pos[2:] - pos[:-2]) / (2 * dt)
            
            # 边界处理：前向/后向差分
            velocities[0, atom, dim] = (pos[1] - pos[0]) / dt
            velocities[-1, atom, dim] = (pos[-1] - pos[-2]) / dt
            
            # FFT得到速度谱
            velocity_spectrum[:, atom, dim] = fft(velocities[:, atom, dim])
    
    return velocities, velocity_spectrum

def calculate_true_velocity_spectrum(true_velocities):
    """计算真实速度的频谱"""
    T, N_atoms, _ = true_velocities.shape
    velocity_spectrum = np.zeros((T, N_atoms, 3), dtype=complex)
    
    for atom in range(N_atoms):
        for dim in range(3):
            velocity_spectrum[:, atom, dim] = fft(true_velocities[:, atom, dim])
    
    return velocity_spectrum

def calculate_errors(method_vel, method_spectrum, true_vel, true_spectrum):
    """计算时域和频域的误差"""
    # 时域误差 (RMSE)
    time_domain_rmse = np.sqrt(np.mean((method_vel - true_vel)**2))
    
    # 频域误差 (平均绝对误差的幅度)
    spectrum_magnitude_error = np.mean(np.abs(np.abs(method_spectrum) - np.abs(true_spectrum)))
    
    # 相对误差
    relative_error = np.sqrt(np.sum((method_vel - true_vel)**2)) / np.sqrt(np.sum(true_vel**2))
    
    return time_domain_rmse, spectrum_magnitude_error, relative_error

def plot_results(t, freqs, positions, true_vel, vel1, vel2, 
                 true_spectrum, spectrum1, spectrum2, atom_idx=0, dim_idx=0):
    """可视化结果"""
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    
    # 位置随时间变化
    axes[0, 0].plot(t, positions[:, atom_idx, dim_idx], 'b-', linewidth=1.5)
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Position')
    axes[0, 0].set_title(f'Position vs Time (Atom {atom_idx}, Dim {dim_idx})')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 速度时域比较
    axes[0, 1].plot(t, true_vel[:, atom_idx, dim_idx], 'k-', label='True', linewidth=2, alpha=0.7)
    axes[0, 1].plot(t, vel1[:, atom_idx, dim_idx], 'r--', label='Method 1 (Freq domain)', linewidth=1.5)
    axes[0, 1].plot(t, vel2[:, atom_idx, dim_idx], 'b:', label='Method 2 (Time domain)', linewidth=1.5)
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Velocity')
    axes[0, 1].set_title('Velocity Comparison (Time Domain)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 速度频谱幅度比较
    half_len = len(freqs) // 2
    axes[1, 0].semilogy(freqs[:half_len], np.abs(true_spectrum[:half_len, atom_idx, dim_idx]), 
                        'k-', label='True', linewidth=2, alpha=0.7)
    axes[1, 0].semilogy(freqs[:half_len], np.abs(spectrum1[:half_len, atom_idx, dim_idx]), 
                        'r--', label='Method 1', linewidth=1.5)
    axes[1, 0].semilogy(freqs[:half_len], np.abs(spectrum2[:half_len, atom_idx, dim_idx]), 
                        'b:', label='Method 2', linewidth=1.5)
    axes[1, 0].set_xlabel('Frequency (Hz)')
    axes[1, 0].set_ylabel('Magnitude')
    axes[1, 0].set_title('Velocity Spectrum Magnitude')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 方法1的误差
    error1_time = vel1[:, atom_idx, dim_idx] - true_vel[:, atom_idx, dim_idx]
    axes[1, 1].plot(t, error1_time, 'r-', linewidth=1)
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Error')
    axes[1, 1].set_title('Method 1 Error (Time Domain)')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 方法2的误差
    error2_time = vel2[:, atom_idx, dim_idx] - true_vel[:, atom_idx, dim_idx]
    axes[2, 0].plot(t, error2_time, 'b-', linewidth=1)
    axes[2, 0].set_xlabel('Time (s)')
    axes[2, 0].set_ylabel('Error')
    axes[2, 0].set_title('Method 2 Error (Time Domain)')
    axes[2, 0].grid(True, alpha=0.3)
    
    # 频谱误差比较
    error1_freq = np.abs(spectrum1[:half_len, atom_idx, dim_idx]) - np.abs(true_spectrum[:half_len, atom_idx, dim_idx])
    error2_freq = np.abs(spectrum2[:half_len, atom_idx, dim_idx]) - np.abs(true_spectrum[:half_len, atom_idx, dim_idx])
    axes[2, 1].plot(freqs[:half_len], error1_freq, 'r-', label='Method 1', linewidth=1.5)
    axes[2, 1].plot(freqs[:half_len], error2_freq, 'b-', label='Method 2', linewidth=1.5)
    axes[2, 1].set_xlabel('Frequency (Hz)')
    axes[2, 1].set_ylabel('Magnitude Error')
    axes[2, 1].set_title('Spectrum Magnitude Error')
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('velocity_spectrum_comparison.png', dpi=300, bbox_inches='tight')
    print("图像已保存为 velocity_spectrum_comparison.png")
    plt.show()

def main():
    """主测试函数"""
    print("=" * 60)
    print("傅里叶变换计算速度谱精度测试")
    print("=" * 60)
    
    # 设置参数
    T = 1024  # 时间步数
    N_atoms = 5  # 原子数
    dt = 0.01  # 时间步长 (秒)
    
    print(f"\n测试参数:")
    print(f"  时间步数: {T}")
    print(f"  原子数量: {N_atoms}")
    print(f"  时间步长: {dt} s")
    print(f"  总时长: {T*dt} s")
    print(f"  采样频率: {1/dt} Hz")
    
    # 生成测试数据
    print("\n生成测试轨迹...")
    positions, true_velocities, t = generate_test_trajectory(T, N_atoms, dt)
    true_spectrum = calculate_true_velocity_spectrum(true_velocities)
    freqs = fftfreq(T, dt)
    
    # 方法1: 频域差分
    print("\n方法1: 频域差分 (FFT位移 -> iω乘法 -> IFFT)")
    vel1, spectrum1 = method1_frequency_domain_diff(positions, dt)
    rmse1, spec_error1, rel_error1 = calculate_errors(vel1, spectrum1, true_velocities, true_spectrum)
    
    # 方法2: 时域差分
    print("方法2: 时域差分 (中心差分速度 -> FFT)")
    vel2, spectrum2 = method2_time_domain_diff(positions, dt)
    rmse2, spec_error2, rel_error2 = calculate_errors(vel2, spectrum2, true_velocities, true_spectrum)
    
    # 输出结果
    print("\n" + "=" * 60)
    print("精度比较结果:")
    print("=" * 60)
    
    print(f"\n方法1 (频域差分):")
    print(f"  时域RMSE: {rmse1:.6e}")
    print(f"  频谱误差: {spec_error1:.6e}")
    print(f"  相对误差: {rel_error1:.6e}")
    
    print(f"\n方法2 (时域差分):")
    print(f"  时域RMSE: {rmse2:.6e}")
    print(f"  频谱误差: {spec_error2:.6e}")
    print(f"  相对误差: {rel_error2:.6e}")
    
    print("\n" + "=" * 60)
    print("结论:")
    print("=" * 60)
    
    if rmse1 < rmse2:
        improvement = (rmse2 - rmse1) / rmse2 * 100
        print(f"✓ 频域差分的时域RMSE更低 (改善 {improvement:.2f}%)")
    else:
        improvement = (rmse1 - rmse2) / rmse1 * 100
        print(f"✓ 时域差分的时域RMSE更低 (改善 {improvement:.2f}%)")
    
    if spec_error1 < spec_error2:
        improvement = (spec_error2 - spec_error1) / spec_error2 * 100
        print(f"✓ 频域差分的频谱误差更低 (改善 {improvement:.2f}%)")
    else:
        improvement = (spec_error1 - spec_error2) / spec_error1 * 100
        print(f"✓ 时域差分的频谱误差更低 (改善 {improvement:.2f}%)")
    
    print("\n理论分析:")
    print("  - 频域差分: 理论上精确,因为 d/dt <-> iω 是精确的傅里叶变换性质")
    print("  - 时域差分: 存在离散化误差,精度取决于差分格式和时间步长")
    print("  - 实际应用中还需考虑边界效应、数值稳定性等因素")
    
    # 可视化
    print("\n生成对比图...")
    plot_results(t, freqs, positions, true_velocities, vel1, vel2, 
                 true_spectrum, spectrum1, spectrum2)
    
    print("\n测试完成!")

if __name__ == "__main__":
    main()
