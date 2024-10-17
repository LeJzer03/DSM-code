# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 10:21:29 2024

@author: colli
"""
#tst
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.fft import fft, fftfreq

# Function to smooth the data using a moving average filter
def smooth_data(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='same')

# 1. Load time response data
time_data = np.loadtxt('P2024_irf_acc.txt', delimiter='\t')
time = time_data[:, 0]
acceleration = time_data[:, 1]

# 2. Smooth the acceleration data (to reduce noise)
window_size = 5  # Adjust window size as necessary for your data
acceleration_smooth = smooth_data(acceleration, window_size)

# 3. Plot smoothed vs original data for comparison
plt.figure(figsize=(8, 6))  # Adjust size to match the example format
plt.plot(time, acceleration, label='Original Data', color='b')
plt.plot(time, acceleration_smooth, label='Smoothed Data', linestyle='--', color='r')
plt.title(r'Acceleration Data $a(t)$', fontsize=16)
plt.xlabel(r'Time $t$ [s]', fontsize=14)
plt.ylabel(r'Acceleration $a(t)$ [m/s²]', fontsize=14)
plt.legend(loc='best', fontsize=12)
plt.grid(True, linestyle='--', linewidth=0.5, color='black', alpha=0.5)  # Main grid
plt.minorticks_on()  # Minor ticks
plt.grid(True, which='minor', linestyle=':', linewidth=0.3, color='gray', alpha=0.7)  # Minor grid
plt.show()
#plt.savefig("acceleration smoothed.pdf", format="pdf")

# 4. Detect peaks in the smoothed acceleration data
# Define threshold and minimum distance to filter out noise
peak_threshold = 0.005  # Adjust based on your data (can be set as a fraction of the max value)
min_distance = 20  # Adjust this based on your time resolution and expected peak spacing

# Detect all peaks
peaks, _ = find_peaks(acceleration_smooth, height=peak_threshold, distance=min_distance)

# 5. Filter peaks after t > 0.02
filtered_peaks = peaks[time[peaks] > 0.02]

# 6. Plot the detected peaks after t > 0.02
plt.figure(figsize=(8, 6))
plt.plot(time, acceleration_smooth, label='Smoothed Data', color='b')
plt.plot(time[filtered_peaks], acceleration_smooth[filtered_peaks], 'rx', label='Detected Peaks (t > 0.02s)')
plt.title(r'Detected Peaks in Smoothed Acceleration Data (After $t > 0.02$)', fontsize=16)
plt.xlabel(r'Time $t$ [s]', fontsize=14)
plt.ylabel(r'Acceleration $a(t)$ [m/s²]', fontsize=14)
plt.legend(loc='best', fontsize=12)
plt.grid(True, linestyle='--', linewidth=0.5, color='black', alpha=0.5)
plt.minorticks_on()
plt.grid(True, which='minor', linestyle=':', linewidth=0.3, color='gray', alpha=0.7)
plt.show()
#plt.savefig("Detected Peaks in Smoothed Acceleration Data.pdf", format="pdf")

# 7. Estimate damped natural frequency using the cleaned peaks
peak_times = time[filtered_peaks]
T = np.mean(np.diff(peak_times))  # Average period between peaks
f_damped = 1 / T  # Damped natural frequency

# 8. Estimate damping ratio using logarithmic decrement
# Only work with valid, positive acceleration peaks
valid_peaks = acceleration_smooth[filtered_peaks] > 0
log_decrements = np.log(acceleration_smooth[filtered_peaks[:-1]][valid_peaks[:-1]] / acceleration_smooth[filtered_peaks[1:]][valid_peaks[1:]])

# Ensure only finite values (no NaN, Inf)
log_decrements = log_decrements[np.isfinite(log_decrements)]
zeta = np.mean(log_decrements) / (2 * np.pi)

print(f'Damped Natural Frequency: {f_damped} Hz')
print(f'Damping Ratio: {zeta}')

# 9. Load FRF data (3 columns: frequency, real part of FRF, imaginary part of FRF)
frf_data = np.loadtxt('P2024_frf_acc.txt', delimiter='\t')
frequency = frf_data[:, 0]
Re_FRF = frf_data[:, 1]
Im_FRF = frf_data[:, 2]
FRF = Re_FRF + 1j * Im_FRF

# 10. Bode plot
mag = np.abs(FRF)
mag[mag == 0] = 1e-12  # Replace zero magnitudes with a small value to avoid log10(0)
phase = np.angle(FRF, deg=True)

plt.figure(figsize=(8, 6))

# Magnitude plot
plt.subplot(2, 1, 1)
plt.plot(frequency, 20 * np.log10(mag), color='b')  # Magnitude in dB
plt.title(r'Bode Diagram', fontsize=16)
plt.ylabel(r'Magnitude [dB]', fontsize=14)
plt.grid(True, linestyle='--', linewidth=0.5, color='black', alpha=0.5)
plt.minorticks_on()
plt.grid(True, which='minor', linestyle=':', linewidth=0.3, color='gray', alpha=0.7)

# Phase plot
plt.subplot(2, 1, 2)
plt.plot(frequency, phase, color='r')  # Phase in degrees
plt.xlabel(r'Frequency [Hz]', fontsize=14)
plt.ylabel(r'Phase [degrees]', fontsize=14)
plt.grid(True, linestyle='--', linewidth=0.5, color='black', alpha=0.5)
plt.minorticks_on()
plt.grid(True, which='minor', linestyle=':', linewidth=0.3, color='gray', alpha=0.7)

plt.show()
#plt.savefig("Bode Diagram.pdf", format="pdf")

# 11. Nyquist plot
plt.figure(figsize=(8, 6))
plt.plot(Re_FRF, Im_FRF, color='b')
plt.title(r'Nyquist Diagram', fontsize=16)
plt.xlabel(r'Real Part', fontsize=14)
plt.ylabel(r'Imaginary Part', fontsize=14)
plt.grid(True, linestyle='--', linewidth=0.5, color='black', alpha=0.5)
plt.axis('equal')  # Ensure orthonormal plot
plt.show()
#plt.savefig("Bode Diagram.pdf", format="pdf")

# 12. Compute damping ratio using half-power method from FRF
half_power_points = np.where(20 * np.log10(mag) >= np.max(20 * np.log10(mag)) - 3)[0]
f_natural = frequency[np.argmax(mag)]  # Frequency at the peak
bandwidth = frequency[half_power_points[-1]] - frequency[half_power_points[0]]
zeta_half_power = bandwidth / (2 * f_natural)

print(f'Natural Frequency from Bode Plot: {f_natural} Hz')
print(f'Damping Ratio from Half-Power Method: {zeta_half_power}')


