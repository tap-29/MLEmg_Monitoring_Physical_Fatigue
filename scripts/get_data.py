import pandas as pd
from scipy.signal import cheby1, filtfilt, freqz
import numpy as np
import os
from scipy import signal
from scipy.signal import medfilt
import pywt
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
from scipy.signal import iirnotch, lfilter

def remove_baseline(RawSignal, windows_size):
    '''
    基于中值滤波来实现去基线，window_size推荐为fs
    :param RawSignal:原始数据
    :param windows_size:中值滤波的窗口
    :return: 去除基线后数据
    '''

    # 转换为NumPy数组
    RawEmg = np.array(RawSignal)

    # 计算基线
    Baseline = median_filter_1d(RawEmg, windows_size)

    # 去除基线
    ReBlEmg = RawEmg - Baseline

    return ReBlEmg.tolist()  # 转换回列表形式，如果需要返回列表


def waveFilter(semg_data):#TODO:调整合适参数，检验小波滤波效果
    '''
    小波变换滤波
    :param semg_data:原始数据
    :return: 滤波后的结果
    '''
    # 执行小波分解:
    # 使用 'db4' 小波基函数对 semg_data 进行二级小波分解，得到分解系数 coeffs。
    coeffs = pywt.wavedec(semg_data, 'db4', level=3)

    # 估算标准差：
    # 最后一层细节系数的绝对值的中位数，然后除以调整因子 0.6745，估算信号的标准差。
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745

    # 计算 Stein 阈值
    threshold = sigma * np.sqrt(2 * np.log(len(semg_data)))

    # 应用软阈值处理
    coeffs_thresholded = [pywt.threshold(c, value=threshold, mode='soft') for c in coeffs]

    # 重构信号
    denoised_signal = pywt.waverec(coeffs_thresholded, 'db4')

    return denoised_signal


def median_filter_1d(data, window_size):
    """
    进行中值滤波
    :param data: 原始数据
    :param window_size:窗口大小
    :return: 滤波完成的数据
    """
    data_length = len(data)
    result = np.zeros(data_length)

    padded_data = np.pad(data, (window_size // 2, window_size // 2), mode='constant')

    for i in range(0, len(data), 1):
        window = padded_data[i:i + window_size]
        result[i] = np.median(window)
    return result


def chebyshev_bandpass_filter(data, lowcut, highcut, fs, rp=3, rs=40, order=3):
    """
    带通滤波器：切比雪夫滤波器，强调高频的效果，对低频不那么追求
    :param data:原始数据
    :param lowcut:低通截止频率
    :param highcut:高通截止频率
    :param fs:采样率
    :param rp:通带内的最大允许波纹（dB）
    :param rs:阻带内的最小衰减（dB）
    :param order:滤波器阶数，阶数越高，滤波器的性能和复杂性通常也越高。
    :return:滤波完成后的数据
    """
    # 计算归一化频率
    low = lowcut / (0.5 * fs)
    high = highcut / (0.5 * fs)

    # 使用cheby1设计滤波器
    b, a = cheby1(N=order, rp=rp, Wn=[low, high], btype='band', analog=False, fs=fs)

    # 使用设计好的滤波器对信号进行滤波
    filtered_data = filtfilt(b, a, data)

    return filtered_data


def plot_spectrogram(semg_data, filtered_data):
    """
    画出前后频谱图，单独用的话传入同一个值就行
    :param semg_data: 原数据
    :param filtered_data:滤波后数据
    :return:
    """
    # 绘制原始信号和滤波后信号的频谱分布
    plt.figure(figsize=(12, 6))

    # 原始信号频谱
    plt.subplot(2, 1, 1)
    plt.specgram(semg_data, Fs=fs, NFFT=256, cmap='viridis')
    plt.title('Original Signal Spectrogram')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')

    # 滤波后信号频谱
    plt.subplot(2, 1, 2)
    plt.specgram(filtered_data, Fs=fs, NFFT=256, cmap='viridis')
    plt.title('Filtered Signal Spectrogram')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')

    plt.tight_layout()
    plt.show()


def plot_spectrogram1(data, filtered_data):
    """
    画出前后频谱图，横坐标是频率，纵坐标是幅度，单独用的话传入同一个值就行
    :param semg_data:原始数据
    :param filtered_data:滤波后数据
    :return:
    """

    # 绘制原始信号频谱图
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.magnitude_spectrum(data, Fs=fs, scale='dB', color='blue')
    plt.title('Original Signal Spectrum')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude (dB)')

    # 绘制带通滤波后信号频谱图
    plt.subplot(2, 1, 2)
    plt.magnitude_spectrum(filtered_data, Fs=fs, scale='dB', color='green')
    plt.title('Bandpass Filtered Signal Spectrum')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude (dB)')
    plt.show()


def notch_filter(data, fs, center_freq, Q):
    """
    陷波滤波器
    :param data:原始数据
    :param fs: 采样率
    :param center_freq: 陷波对应的频率
    :param Q: 品质参数，越大则陷波周围的带宽越小
    :return:返回的数据
    """
    # 设计陷波滤波器
    b, a = iirnotch(center_freq, Q, fs)

    # 应用滤波器
    filtered_data = lfilter(b, a, data)

    return filtered_data

# 获取数据
csv_path = "../origin_data/Study1/EMG/U1Ex1Rep1.csv"
df = pd.read_csv(csv_path)
timestamps = df.iloc[:, 0]  # 假设时间戳在第一列
semg_data = df.iloc[:, 1]  # 假设sEMG数据在第二列
fs = 1000
t = np.arange(0, 1, 1/fs)
np.random.seed(40)
data = 0.5 * np.sin(2 * np.pi * 50 * t) + 0.2 * np.sin(2 * np.pi * 150 * t) + 0.1 * np.random.randn(len(t))



# 参数设置
fs = 1926.0
lowcut = 50.0
highcut = 150.0
rp = 3  # 通带内的最大允许波纹（dB）
rs = 40  # 阻带内的最小衰减（dB）
center_freq = 96  # 陷波中心频率
Q = 10  # 陷波滤波器的品质因数

