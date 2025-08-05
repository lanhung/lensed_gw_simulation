# -*- coding: utf-8 -*-
"""
模拟强引力透镜下的双星系统引力波信号，并注入至ET噪声中
包括四个步骤：
1. 天体物理人群生成
2. 注入透镜效应（点透镜或SIE）
3. 注入第三代引力波探测器ET的噪声中
4. 检测性能评估，并输出模拟数据用于训练/可视化
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve
import pandas as pd
import seaborn as sns
from pycbc.waveform import get_td_waveform
from pycbc.detector import get_detector

##############################
# 第一步：生成天体物理源参数
##############################
def sample_astro_population(N):
    masses_1 = np.random.power(2.3, N) * (50 - 5) + 5
    q = np.random.uniform(0.3, 1.0, N)
    masses_2 = masses_1 * q
    spins = np.random.uniform(-0.5, 0.5, N)
    redshifts = np.random.exponential(0.5, N)
    return masses_1, masses_2, spins, redshifts

#################################
# 第二步：注入透镜效应
#################################
def lensing_amplification_point_lens(y, f, M_L):
    """
    基于波动光学理论，使用近似点透镜模型计算频率相关放大因子。
    y: 无量纲冲击参数（impact parameter）
    f: 频率数组
    M_L: 透镜质量（单位：太阳质量）
    返回: 每个频率下的复数放大因子
    """
    from scipy.special import gamma, hyp1f1
    omega = 2 * np.pi * f * 4.9254909e-6 * M_L  # 将 M_sun 转换为秒，f 转为自然单位频率
    w = omega + 1e-6
    phi = np.pi * w / 4 + w * (np.log(w / 2) - 1)
    F_w = np.exp(np.pi * w / 4) * gamma(1 - 1j * w) * hyp1f1(1j * w, 1, 1j * w * y**2 / 2) * np.exp(1j * phi)
    return np.abs(F_w)

def lensing_amplification_sie(f, y, sigma_v):
    """
    使用 SIE 星系透镜模型的近似频率响应
    f: 频率数组
    y: impact parameter
    sigma_v: 星系速度弥散（km/s）
    返回: 模拟放大因子（幅值）
    """
    velocity_term = (sigma_v / 200.0)**2
    amplification = 1 + velocity_term / (1 + y**2) * np.exp(-f / 300.0)
    return amplification

def apply_lensing_effects(h_f, mu_f):
    return h_f * mu_f

#########################################
# 第三步：注入至ET-D噪声背景
#########################################
def simulate_et_noise(length, delta_t):
    return np.random.normal(0, 1, int(length / delta_t))

def inject_signal_into_noise(signal, noise, snr_target):
    signal_power = np.sqrt(np.sum(signal**2))
    scaled_signal = signal * (snr_target / signal_power)
    return noise + scaled_signal

#########################################
# 第四步：检测与输出特征
#########################################
def compute_snr(signal, noise):
    snr = np.dot(signal, signal) / np.std(noise)
    return snr

def extract_features(signal):
    spec = np.abs(np.fft.rfft(signal))
    peak_freq = np.argmax(spec)
    mean_amp = np.mean(spec)
    std_amp = np.std(spec)
    return peak_freq, mean_amp, std_amp

def visualize_summary(df):
    plt.figure(figsize=(8, 4))
    sns.histplot(data=df, x="snr", hue="lens_type", bins=20, kde=True)
    plt.title("SNR 分布对比")
    plt.xlabel("信噪比")
    plt.savefig("snr_distribution.png")

    plt.figure(figsize=(8, 4))
    sns.scatterplot(data=df, x="peak_freq_idx", y="mean_amp", hue="lens_type")
    plt.title("频谱特征散点图")
    plt.xlabel("主频索引")
    plt.ylabel("平均幅值")
    plt.savefig("frequency_feature_scatter.png")

    lens_groups = df.groupby("lens_type")
    for name, group in lens_groups:
        det_rate = (group['snr'] > 8).sum() / len(group)
        print(f"透镜类型 {name} 的探测率：{det_rate:.2%}")
