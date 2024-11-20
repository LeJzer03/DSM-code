# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 17:58:42 2024

@author: colli
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.fft import ifft

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

epsilon = 1e-10  # Petite valeur pour éviter la division par zéro

# Bode Plot: Amplitude
magnitude = 20 * np.log10(np.abs(H_values) + epsilon)


# Plot the Bode Plot
plt.figure(figsize=(8, 8))
# Bode Amplitude
plt.subplot(2, 1, 1)
plt.semilogx(freq_range, magnitude, label=r'Calculated $H(\omega)$')
plt.semilogx(freq_frf, 20 * np.log10(np.sqrt(re_frf**2 + im_frf**2)), '--', label=r'Simulated FRF Data')
plt.xlabel(r'Frequency [Hz]', size=11)
plt.ylabel(r'Magnitude [dB]', size=11)
plt.legend()
plt.title(r'Diagramme de Bode en amplitude', size=14)
plt.xlim([1, 1500])  # Avoid the issue with starting from 0 Hz
plt.ylim([-70, -30])
plt.grid(True, linestyle='--', linewidth=0.5, color='black', alpha=0.5)
plt.minorticks_on()
plt.grid(True, which='minor', linestyle=':', linewidth=0.3, color='gray', alpha=0.7)

# Highlight resonance and anti-resonance frequencies
resonance_frequencies = frequencies  # Use natural frequencies for resonance
for freq in resonance_frequencies:
    plt.axvline(freq, color='r', linestyle='--', alpha=0.5)  # Vertical line at each resonance
    plt.annotate(rf'Res: {freq:.1f} Hz', (freq, -35), textcoords="offset points", xytext=(0, 10),
                 ha='center', color='red')

# Highlight antiresonance frequencies
antiresonance_indices, _ = find_peaks(-magnitude)
antiresonance_frequencies = freq_range[antiresonance_indices]
for freq in antiresonance_frequencies:
    plt.axvline(freq, color='b', linestyle='--', alpha=0.5)  # Vertical line at each antiresonance
    plt.annotate(rf'Anti: {freq:.1f} Hz', (freq, -55), textcoords="offset points", xytext=(0, -15),
                 ha='center', color='blue')    

# Custom legend with a box
legend_handles = [
    plt.Line2D([0], [0], color='r', linestyle='--', label=r'Resonance Frequencies'),
    plt.Line2D([0], [0], color='b', linestyle='--', label=r'Antiresonance Frequencies')
]
plt.legend(handles=legend_handles, loc='lower left', fontsize=10, frameon=True, framealpha=0.8, edgecolor='black')



# Nyquist Plot: Real vs Imaginary
real_part = np.real(H_values)
imag_part = np.imag(H_values)


plt.figure(figsize=(8, 8))
plt.plot(real_part, imag_part, label=r'Calculated $H(\omega)$')
plt.plot(re_frf, im_frf, '--', label=r'Simulated FRF Data')
plt.xlabel(r'Partie Réelle', size=14)
plt.ylabel(r'Partie Imaginaire', size=14)
plt.title(r'Diagramme de Nyquist', size=18)
plt.legend()
plt.grid(True, linestyle='--', linewidth=0.5, color='black', alpha=0.5)
plt.minorticks_on()
plt.grid(True, which='minor', linestyle=':', linewidth=0.3, color='gray', alpha=0.7)
plt.axis('equal')

plt.show()



# Reference points based on the image and distances provided
reference_points = [
    {"point": "P1", "distance_mm": 0, "description": "Fork (A)"},
    {"point": "P2", "distance_mm": 100, "description": ""},
    {"point": "P3", "distance_mm": 200, "description": ""},
    {"point": "P4", "distance_mm": 300, "description": ""},
    {"point": "P5", "distance_mm": 400, "description": "Engine (B)"},
    {"point": "P6", "distance_mm": 400, "description": ""},
    {"point": "P7", "distance_mm": 500, "description": ""},
    {"point": "P8", "distance_mm": 600, "description": ""},
    {"point": "P9", "distance_mm": 700, "description": ""},
    {"point": "P10", "distance_mm": 800, "description": "Swingarm Connection (C)"},
    {"point": "P11", "distance_mm": 901, "description": ""},
    {"point": "P12", "distance_mm": 1000, "description": "Driver Seat (D)"},
    {"point": "P13", "distance_mm": 1101, "description": ""},
    {"point": "P14", "distance_mm": 1200, "description": "Passenger Seat (E)"}
]

# Compute the maximum amplitude of the response (acceleration) for each reference point
max_amplitudes = np.zeros(len(reference_points))

for i in range(len(reference_points)):
    H_values_point = np.array([H_calcul(w, modes, pulsations_propres, damping_factors)[i, 0] for w in omega_range])
    max_amplitudes[i] = np.max(np.abs(H_values_point))

# Extract distances for x-axis
distances_mm = [point["distance_mm"] for point in reference_points]



# Tracer l'évolution de l'amplitude maximale pour chaque point de référence
plt.figure(figsize=(10, 6))
plt.plot(distances_mm, max_amplitudes, marker='o', linestyle='-', color='b')
plt.xlabel(r'Distance depuis la fourche le long du cadre [mm]', size=14)
plt.ylabel(r'Amplitude Maximale de la Réponse (Accélération) $[m/s^2]$', size=11)
plt.title(r'Évolution de l\'Amplitude Maximale de la Réponse pour Chaque Point de Référence', size=16)
plt.xticks(distances_mm)  # Définir les graduations de l'axe des x à chaque distance
plt.xlim(0, 1200)  # Définir les limites de l'axe des x de 0 à 1200 mm
plt.grid(True, linestyle='--', linewidth=0.5, color='black', alpha=0.5)
plt.minorticks_on()
plt.grid(True, which='minor', linestyle=':', linewidth=0.3, color='gray', alpha=0.7)

# Ajouter les noms des points au-dessus des points d'amplitude maximale avec un décalage vertical
offset = 0.003  # Ajustez cette valeur selon vos besoins
for i, point in enumerate(reference_points):
    if point["point"] == "P1":
        plt.text(distances_mm[i] + 15, max_amplitudes[i] + offset - 0.001, point["point"], ha='center', va='bottom', fontsize=10)
    elif point["point"] == "P5":
        plt.text(distances_mm[i], max_amplitudes[i] + offset + 0.003, "P5-P6", ha='center', va='bottom', fontsize=10)
    elif point["point"] == "P6":
        continue  # Skip P6 as it is grouped with P5
    elif point["point"] == "P12":
        plt.text(distances_mm[i], max_amplitudes[i] + offset + 0.003, point["point"], ha='center', va='bottom', fontsize=10)
    elif point["point"] == "P13":
        plt.text(distances_mm[i] - 15, max_amplitudes[i] + offset + 0.003, point["point"], ha='center', va='bottom', fontsize=10)
    elif point["point"] == "P14":
        plt.text(distances_mm[i] - 25, max_amplitudes[i] + offset, point["point"], ha='center', va='bottom', fontsize=10)
    else:
        plt.text(distances_mm[i], max_amplitudes[i] + offset, point["point"], ha='center', va='bottom', fontsize=10)

plt.show()




