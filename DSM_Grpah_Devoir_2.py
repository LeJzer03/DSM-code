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
#plt.savefig("acceleration smoothed.pdf", format="pdf")
plt.show()


# 4. Detect peaks in the smoothed acceleration data
# Define threshold and minimum distance to filter out noise
peak_threshold = 0.0025  # Adjust based on your data (can be set as a fraction of the max value)
min_distance = 20  # Adjust this based on your time resolution and expected peak spacing

# Detect all peaks
peaks, _ = find_peaks(acceleration_smooth, height=peak_threshold, distance=min_distance)

# 5. Filter peaks after t > 0.02
filtered_peaks = peaks[time[peaks] > 0.02]

# 6. Plot the detected peaks after t > 0.02
plt.figure(figsize=(8, 6))
plt.plot(time, acceleration_smooth, label='Smoothed Data', color='b')
plt.plot(time[filtered_peaks], acceleration_smooth[filtered_peaks], 'rx', label='Detected Peaks (t > 0.02s)')
plt.axvline(x=0.02, color='green', linestyle='--', linewidth=1, label=r'$t = 0.02 \, \text{s}$')  # Ligne verticale verte pointillée
plt.title(r'Detected Peaks in Smoothed Acceleration Data (After $t > 0.02$)', fontsize=16)
plt.xlabel(r'Time $t$ [s]', fontsize=14)
plt.ylabel(r'Acceleration $a(t)$ [m/s²]', fontsize=14)
plt.legend(loc='best', fontsize=12)
plt.grid(True, linestyle='--', linewidth=0.5, color='black', alpha=0.5)
plt.minorticks_on()
plt.grid(True, which='minor', linestyle=':', linewidth=0.3, color='gray', alpha=0.7)
#plt.savefig("Detected Peaks in Smoothed Acceleration Data.pdf", format="pdf")
plt.show()



# 7. Estimate damped natural frequency using the cleaned peaks
peak_times = time[filtered_peaks]
T = np.mean(np.diff(peak_times))  # Average period between peaks
"""
comme on a epsilon <<1 (a verif apres) donc  w_0=w_d (+-)
"""
f_damped = 1 / T  # Damped natural frequency


# 8. Calculate the damping ratio using log method
if len(filtered_peaks) >= 2:  # Ensure there are at least two peaks
    a_1 = acceleration_smooth[filtered_peaks[0]]  # Amplitude of the first peak
    a_Nk = acceleration_smooth[filtered_peaks[-1]]  # Amplitude of the last peak

    k = len(filtered_peaks) - 1  # Number of periods between the first and last peak
    zeta = np.log(a_1 / a_Nk) / (k * 2 * np.pi)  # Calculate damping ratio using the new formula

    print(f'Damped Natural Frequency from time data: {f_damped} Hz')
    print(f'Damping Ratio from time data: {zeta}')
else:
    print("Not enough peaks detected to compute the damping ratio.")



# 9. Load FRF data (3 columns: frequency, real part of FRF, imaginary part of FRF)
frf_data = np.loadtxt('P2024_frf_acc.txt', delimiter='\t')
frequency = frf_data[:, 0]
Re_FRF = frf_data[:, 1]
Im_FRF = frf_data[:, 2]
FRF = Re_FRF + 1j * Im_FRF

# 10. Bode plot
# Calcul de la magnitude et de la phase
mag = np.abs(FRF)
phase = np.angle(FRF, deg=True)

# Filtrer les valeurs de magnitude égales à zéro
non_zero_indices = mag > 0  # Créer un masque booléen où les valeurs de mag sont strictement positives

# Appliquer ce masque pour filtrer les valeurs de mag et phase
mag = mag[non_zero_indices]  # Garde seulement les valeurs non nulles de mag
phase = phase[non_zero_indices]  # Filtrer également la phase correspondante
frequency = frequency[non_zero_indices]

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

#plt.savefig("Bode Diagram.pdf", format="pdf")
plt.show()


# 10. Half-power method - improved based on Q-factor
max_magnitude = np.max(mag)
half_power_level = max_magnitude / np.sqrt(2)

# Find the frequencies at half-power points
half_power_points = np.where(mag >= half_power_level)[0]
f1 = frequency[half_power_points[0]]  # First half-power point
f2 = frequency[half_power_points[-1]]  # Second half-power point

# Convert frequencies to angular frequencies (omega)
omega1 = 2 * np.pi * f1
omega2 = 2 * np.pi * f2
omega_natural = 2 * np.pi * frequency[np.argmax(mag)]  # Angular frequency at peak

# Calculate delta omega (width between half-power points)
delta_omega = omega2 - omega1

Q = omega_natural / delta_omega


#zeta avec le systeme 2 eq a 2 inconnues 
#on note w_a = max_magnitude (pour la formule)
zeta = np.sqrt(delta_omega**2/((4*(omega_natural)**2)+delta_omega**2))

# Calculate the Q factor and damping ratio zeta (assuptions Charles)
#zeta = 1 / (2 * Q)

print(f'Natural Frequency (f_natural from Bode diag.): {frequency[np.argmax(mag)]} Hz')
print(f'Q-factor: {Q}')
print(f'Damping Ratio (zeta) with bode diagram: {zeta}')

# 11. Print the phase at the natural frequency
phase_at_natural = phase[np.argmax(mag)]
print(f'Phase at Natural Frequency: {phase_at_natural} degrees')



# Valeur donnée de M_eq
M_eq = 87.5

# 1. Load FRF data (3 columns: frequency, real part of FRF, imaginary part of FRF)
frf_data = np.loadtxt('P2024_frf_acc.txt', delimiter='\t')
Re_FRF = frf_data[:, 1]  # Real part of FRF
Im_FRF = frf_data[:, 2]  # Imaginary part of FRF

# 2. Trouver les indices où Re_FRF = 0

mask = np.logical_and(FRF.real >= -0.001, FRF.real <= 0.001)
result = FRF[mask]

# 4. Sélectionner la plus grande valeur de la partie imaginaire
if len(result) > 0:
    max_imaginary = np.max(result.imag)  # La plus grande partie imaginaire
    
    # 5. Calculer le damping ratio (epsilon)
    epsilon = 1 / (2 * max_imaginary * M_eq)
    
    print(f"Damping ratio with Nyquist diagram: {epsilon}")
else:
    print("No valid y value found where Re_FRF=0.")

# 6. Tracer le diagramme de Nyquist avec une ligne verticale pointillée partielle
plt.figure(figsize=(8, 6))
plt.plot(Re_FRF, Im_FRF, color='b')

# Tracer la ligne verticale pointillée allant de (0, 0) à (0, max_imaginary)
if len(result) > 0:
    plt.plot([0, 0], [0, max_imaginary], color='red', linestyle='--', linewidth=1.5, label = r'$\frac{1}{2 \epsilon M_{\text{eq}}}$')

# Ajouter le titre et les labels
plt.title(r'Nyquist Diagram', fontsize=16)
plt.xlabel(r'Real Part', fontsize=14)
plt.ylabel(r'Imaginary Part', fontsize=14)
plt.grid(True, linestyle='--', linewidth=0.5, color='black', alpha=0.5)
plt.minorticks_on()
plt.grid(True, which='minor', linestyle=':', linewidth=0.3, color='gray', alpha=0.7)
plt.axis('equal')  # Assure un repère orthonormé
plt.legend()
#plt.savefig("Nyquist Diagram.pdf", format="pdf")
plt.show()
