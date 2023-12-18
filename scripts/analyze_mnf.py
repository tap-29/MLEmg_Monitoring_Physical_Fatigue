import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.stats import linregress


# 读取CSV文件
def read_csv(file_name):
    return pd.read_csv(file_name)


# 计算每个时间窗口的MNF
def calculate_windowed_mnf(data, fs, window_size):
    num_windows = len(data) // window_size
    mnf_values = []
    times = []

    for i in range(num_windows):
        start = i * window_size
        end = (i + 1) * window_size
        window_data = data[start:end]
        f, Pxx = welch(window_data, fs=fs)
        mnf = np.sum(f * Pxx) / np.sum(Pxx)
        mnf_values.append(mnf)
        # 每个窗口的中点时间
        times.append((start + end) / (2 * fs))

    return times, mnf_values


# 归一化MNF值
def normalize_mnf(mnf_values):
    max_mnf = max(mnf_values)
    normalized_mnf = [x / max_mnf for x in mnf_values]
    return normalized_mnf


# 绘制图表
def plot_mnf(times, normalized_mnf):
    plt.scatter(times, normalized_mnf)

    # 拟合趋势线并计算斜率
    slope, intercept, r_value, p_value, std_err = linregress(times, normalized_mnf)
    plt.plot(times, intercept + slope * np.array(times), color='magenta', label=f'Km = {slope:.4f}')

    plt.xlabel('Time(s)')
    plt.ylabel('Normalized MNF')
    plt.title('Normalized MNF over Time')
    plt.legend()
    plt.show()

    return slope


# 主函数
def process_and_plot(file_name, fs, window_size):
    data = read_csv(file_name).iloc[:, 1]
    times, mnf_values = calculate_windowed_mnf(data, fs, window_size)
    normalized_mnf = normalize_mnf(mnf_values)
    slope = plot_mnf(times, normalized_mnf)
    print(f"Slope (Km): {slope}")


# 调用函数，采样频率为1967Hz，窗口大小为1967个样本（即1秒数据）
# process_and_plot('U1Ex1Rep1.csv', fs=1967, window_size=1967)

# 调用函数
process_and_plot('../original_emg_Study1/U1Ex1Rep1.csv', fs=1926, window_size=500)  # 假设采样频率为1000Hz，窗口大小为1000个样本

# 调用函数
# process_and_plot('../original_emg_Study1/U1Ex1Rep1.csv')
