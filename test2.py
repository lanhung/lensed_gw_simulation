# -*- coding: utf-8 -*-
"""
模拟强引力透镜下的双星系统引力波信号，并注入至ET噪声中
包括四个步骤：
1. 天体物理人群生成
2. 注入透镜效应（点透镜或SIE）
3. 注入第三代引力波探测器ET的噪声中
4. 检测性能评估
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve

# 以下是占位导入，实际应用中可使用PyCBC或LALSuite等真实波形工具
# from pycbc.waveform import get_td_waveform
# from pycbc.noise import noise_from_psd

##############################
# 第一步：生成天体物理源参数
##############################
def sample_astro_population(N):
    """ 采样双黑洞天体物理分布 """
    masses_1 = np.random.power(2.3, N) * (50 - 5) + 5  # 主星质量
    q = np.random.uniform(0.3, 1.0, N)  # 质量比
    masses_2 = masses_1 * q  # 次星质量
    spins = np.random.uniform(-0.5, 0.5, N)  # 自旋
    redshifts = np.random.exponential(0.5, N)  # 红移分布
    return masses_1, masses_2, spins, redshifts

#################################
# 第二步：注入透镜效应
#################################
def lensing_amplification_point_lens(y, f, M_L):
    """ 点透镜模型的放大因子（简化形式） """
    mu = 1 + 1/(y**2 + 1e-3)  # 简化近似的放大率
    return mu

def lensing_amplification_sie(f, y, sigma_v):
    """
    SIE（Singular Isothermal Ellipsoid）模型下的简化放大因子函数：
    本函数为近似模型，仅用于模拟不同放大趋势（非实际光学计算）。
    参数：
        f : ndarray 频率轴
        y : impact parameter（简化）
        sigma_v : 星系的速度弥散（控制质量）
    """
    velocity_term = (sigma_v / 200.0)**2  # 归一化速度项
    amplification = 1 + velocity_term / (1 + y**2) * np.exp(-f / 300.0)  # 模拟在低频增强、高频趋于1
    return amplification

def apply_lensing_effects(h_f, mu_f):
    """ 在频域中施加透镜放大因子 """
    return h_f * mu_f

#########################################
# 第三步：注入至ET-D噪声背景
#########################################
def simulate_et_noise(length, delta_t):
    """ 生成模拟ET噪声（此处为白噪声占位） """
    return np.random.normal(0, 1, int(length / delta_t))

def inject_signal_into_noise(signal, noise, snr_target):
    """ 按目标SNR将信号注入噪声中 """
    signal_power = np.sqrt(np.sum(signal**2))
    scaled_signal = signal * (snr_target / signal_power)
    return noise + scaled_signal

#########################################
# 第四步：检测与特征提取
#########################################
def compute_snr(signal, noise):
    """ 计算信号在噪声中的SNR（简化版本） """
    snr = np.dot(signal, signal) / np.std(noise)
    return snr

def extract_features(signal):
    """ 特征提取：可提取频谱特征、带宽等 """
    return np.fft.rfft(signal)

def main():
    N = 1000000  # 总信号数量
    N_lensed = 100  # 被透镜化的信号数量
    delta_t = 1.0 / 4096  # 采样率（s）
    length = 4.0  # 信号长度（秒）

    # 第一步：生成天体物理源参数
    m1, m2, spin, z = sample_astro_population(N)
    is_lensed = np.zeros(N, dtype=bool)
    is_lensed[np.random.choice(N, N_lensed, replace=False)] = True  # 标记被透镜的样本

    signals, snrs = [], []
    for i in range(N_lensed):
        # 第二步：构造假设波形（真实应用中应替换为get_td_waveform）
        t = np.linspace(0, length, int(length / delta_t))
        h = np.sin(200 * 2 * np.pi * t) * np.exp(-t)  # 占位波形（类chirp）

        # 透镜模型选择：点透镜 或 SIE
        use_sie = (i % 2 == 0)  # 偶数编号使用SIE，奇数使用点透镜

        # 注入透镜效应（频域放大）
        f = np.fft.rfftfreq(len(h), d=delta_t)  # 频率轴
        if use_sie:
            mu_f = lensing_amplification_sie(f, y=0.6, sigma_v=220)
        else:
            mu_f = lensing_amplification_point_lens(0.5, f, 1e6)

        h_f = np.fft.rfft(h)
        h_f_lensed = apply_lensing_effects(h_f, mu_f)
        h_lensed = np.fft.irfft(h_f_lensed)

        # 第三步：加入ET噪声
        noise = simulate_et_noise(length, delta_t)
        h_injected = inject_signal_into_noise(h_lensed, noise, snr_target=10)

        # 第四步：评估检测性能（SNR）
        snr = compute_snr(h_lensed, noise)
        snrs.append(snr)
        signals.append(h_injected)

    print(f"已模拟 {N_lensed} 个透镜信号（含SIE和点透镜），平均SNR: {np.mean(snrs):.2f}")

if __name__ == '__main__':
    main()
