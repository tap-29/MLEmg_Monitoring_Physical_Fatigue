"""
本部分主要实现对数据提取MNF值等
"""
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

# 生成模拟的肌肉信号数据
np.random.seed(42)
fs = 1000  # 采样率
t = np.arange(0, 5, 1/fs)  # 5秒钟的信号
f = 20  # 信号频率为20Hz
muscle_signal = 0.5 * np.sin(2 * np.pi * f * t) + np.random.normal(scale=0.1, size=len(t))

# 绘制原始信号
plt.figure(figsize=(10, 4))
plt.plot(t, muscle_signal)
plt.title('Original Muscle Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.show()

# 对信号进行傅里叶变换和频谱分析
frequencies, psd = signal.welch(muscle_signal, fs, nperseg=1024)

# 计算MNF
mnf = np.sum(frequencies * psd) / np.sum(psd)

# 绘制频谱
plt.figure(figsize=(10, 4))
plt.semilogy(frequencies, psd)
plt.title('Power Spectral Density')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power/Frequency (dB/Hz)')
plt.show()

# 输出MNF
print(f'Mean Frequency (MNF): {mnf:.2f} Hz')
