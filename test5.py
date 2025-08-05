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
    from scipy.special import gamma, hyp1f1
    omega = 2 * np.pi * f * 4.9254909e-6 * M_L
    w = omega + 1e-6
    phi = np.pi * w / 4 + w * (np.log(w / 2) - 1)
    F_w = np.exp(np.pi * w / 4) * gamma(1 - 1j * w) * hyp1f1(1j * w, 1, 1j * w * y**2 / 2) * np.exp(1j * phi)
    return np.abs(F_w)

def lensing_amplification_sie(f, y, sigma_v):
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

#########################################
# 主函数：整合各阶段调用逻辑
#########################################
def main():
    N = 1000000
    N_lensed = 100
    delta_t = 1.0 / 4096
    length = 4.0

    m1, m2, spin, z = sample_astro_population(N)
    is_lensed = np.zeros(N, dtype=bool)
    is_lensed[np.random.choice(N, N_lensed, replace=False)] = True

    records = []
    for i in range(N_lensed):
        hp, _ = get_td_waveform(approximant="IMRPhenomPv2",
                                mass1=m1[i],
                                mass2=m2[i],
                                delta_t=delta_t,
                                f_lower=20.0)

        h = hp[:int(length / delta_t)]
        if len(h) < int(length / delta_t):
            h = np.pad(h, (0, int(length / delta_t) - len(h)))

        use_sie = (i % 2 == 0)
        f = np.fft.rfftfreq(len(h), d=delta_t)
        if use_sie:
            mu_f = lensing_amplification_sie(f, y=0.6, sigma_v=220)
            lens_type = "SIE"
        else:
            mu_f = lensing_amplification_point_lens(0.5, f, 1e6)
            lens_type = "Point"

        h_f = np.fft.rfft(h)
        h_f_lensed = apply_lensing_effects(h_f, mu_f)
        h_lensed = np.fft.irfft(h_f_lensed)

        noise = simulate_et_noise(length, delta_t)
        h_injected = inject_signal_into_noise(h_lensed, noise, snr_target=10)

        snr = compute_snr(h_lensed, noise)
        peak_f, mean_amp, std_amp = extract_features(h_injected)

        records.append({
            "id": i,
            "lens_type": lens_type,
            "snr": snr,
            "peak_freq_idx": peak_f,
            "mean_amp": mean_amp,
            "std_amp": std_amp,
            "mass_1": m1[i],
            "mass_2": m2[i],
            "spin": spin[i],
            "redshift": z[i]
        })

    df = pd.DataFrame(records)
    df.to_csv("lensed_signals_summary.csv", index=False)
    print(f"已输出特征摘要至 lensed_signals_summary.csv，平均SNR: {np.mean(df['snr']):.2f}")
    visualize_summary(df)

if __name__ == '__main__':
    main()
