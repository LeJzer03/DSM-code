# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 17:58:42 2024

@author: colli
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Number of modes and points
nbr_mode = 4
nbr_points = 14

# Load data from files
frf_data = np.loadtxt('P2024_frf_Part3_f_ds.txt')
# Extracting FRF data for comparison
freq_frf = frf_data[:, 0]
re_frf = frf_data[:, 1]
im_frf = frf_data[:, 2]

# Load frequencies and damping factors for the modes
frequencies_data = np.loadtxt('P2024_f_eps_Part3.txt')
frequencies = frequencies_data[:, 0]  # Natural frequencies in Hz
damping_factors = frequencies_data[:, 1]  # Damping factors

# Convert natural frequencies to pulsations (rad/s)
pulsations_propres = 2 * np.pi * frequencies  # ω_r = 2πf

# Load mode shapes
modes = np.loadtxt('P2024_modes_Part3.txt')/1000  # Shape: (14 locations, 4 modes)

# Define the Transfer Function H_calcul
def H_calcul(omega, modes, pulsations_propres, amortissement):
    """
    Calculates a specific element of the transfer function matrix H(ω).
    
    Parameters:
    - omega: Angular frequency (rad/s)
    - modes: Mode shape matrix (14 x 4)
    - pulsations_propres: Natural pulsations of modes (rad/s)
    - amortissement: Damping factors for modes
    
    Returns:
    - H: Transfer function matrix element (complex)
    """
    H = 0 + 0j
    for r in range(len(pulsations_propres)):
        # Numerator: -ω² * (mode shape outer product)
        num = -omega**2 * np.outer(modes[:, r], modes[:, r].T)
        
        # Denominator: (ω_r² - ω² + 2i * ε * ω_r * ω)
        denom = (pulsations_propres[r]**2 - omega**2 + 2j * amortissement[r] * pulsations_propres[r] * omega)
        
        H += num / denom
    return H

# Frequency range for the analysis (same as simulation, 0-1500 Hz)
freq_range = np.linspace(0, 1500, 4500)
omega_range = 2 * np.pi * freq_range

# Compute the transfer function over the frequency range for a specific element
H_values = np.array([H_calcul(w, modes, pulsations_propres, damping_factors)[11, 0] for w in omega_range])

# Bode Plot: Amplitude
magnitude = 20 * np.log10(np.abs(H_values))

# Plot the Bode Plot
plt.figure(figsize=(8, 8))

# Bode Amplitude
plt.subplot(2, 1, 1)
plt.semilogx(freq_range, magnitude, label="Calculated H(ω)")
plt.semilogx(freq_frf, 20 * np.log10(np.sqrt(re_frf**2 + im_frf**2)), '--', label="Simulated FRF Data")
plt.xlabel('Frequency [Hz]',size = 11)
plt.ylabel('Magnitude [dB]',size = 11)
plt.legend()
plt.title('Diagramme de Bode en amplitude' ,size = 14)
plt.xlim([1, 1500])  # Avoid the issue with starting from 0 Hz
plt.ylim([-70, -30])


# Highlight resonance and anti-resonance frequencies
resonance_frequencies = frequencies  # Use natural frequencies for resonance
for freq in resonance_frequencies:
    plt.axvline(freq, color='r', linestyle='--', alpha=0.5)  # Vertical line at each resonance
    plt.annotate(f'Res: {freq:.1f} Hz', (freq, -35), textcoords="offset points", xytext=(0, 10),
                 ha='center', color='red')

# Highlight antiresonance frequencies
antiresonance_indices, _ = find_peaks(-magnitude)
antiresonance_frequencies = freq_range[antiresonance_indices]
for freq in antiresonance_frequencies:
    plt.axvline(freq, color='b', linestyle='--', alpha=0.5)  # Vertical line at each antiresonance
    plt.annotate(f'Anti: {freq:.1f} Hz', (freq, -55), textcoords="offset points", xytext=(0, -15),
                 ha='center', color='blue')    
 
# Custom legend with a box
legend_handles = [
    plt.Line2D([0], [0], color='r', linestyle='--', label='Resonance Frequencies'),
    plt.Line2D([0], [0], color='b', linestyle='--', label='Antiresonance Frequencies')
]
plt.legend(handles=legend_handles, loc='lower left', fontsize=10, frameon=True, framealpha=0.8, edgecolor='black')


# Nyquist Plot: Real vs Imaginary
real_part = np.real(H_values)
imag_part = np.imag(H_values)

plt.figure(figsize=(8, 8))
plt.plot(real_part, imag_part, label="Calculated H(ω)")
plt.plot(re_frf, im_frf, '--', label="Simulated FRF Data")
plt.xlabel('Partie Réelle',size = 14)
plt.ylabel('Partie Imaginaire', size = 14)
plt.title('Diagramme de Nyquist',size= 18)
plt.legend()
plt.grid(True)
plt.axis('equal')

plt.show()