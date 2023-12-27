import numpy as np
import pywt
import matplotlib.pyplot as plt

# Generate a signal with noise
np.random.seed(42)
t = np.linspace(0, 1, 1000, endpoint=False)
original_signal = np.sin(2 * np.pi * 7 * t)
noisy_signal = original_signal + 0.5 * np.random.normal(size=len(t))

# Perform wavelet decomposition
coeffs = pywt.wavedec(noisy_signal, 'db4', level=4)

# Estimate standard deviation
sigma = np.median(np.abs(coeffs[-1])) / 0.6745

# Calculate Stein threshold
threshold = sigma * np.sqrt(2 * np.log(len(noisy_signal)))

# Apply soft thresholding
coeffs_thresholded = [pywt.threshold(c, value=threshold, mode='soft') for c in coeffs]

# Reconstruct the signal
denoised_signal = pywt.waverec(coeffs_thresholded, 'db4')

# Simple low-pass filtering using moving average
window_size = 20
low_pass_filtered_signal = np.convolve(denoised_signal, np.ones(window_size) / window_size, mode='valid')

# Plot the results
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(t, original_signal, label='Original Signal', linestyle='--', linewidth=2)
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(t, noisy_signal, label='Noisy Signal')
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(t, denoised_signal, label='Wavelet Denoised Signal', linestyle='--', linewidth=2)
plt.legend()



plt.suptitle('Wavelet Denoising and Low-pass Filtering')
plt.show()
