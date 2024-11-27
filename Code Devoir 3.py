# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 17:58:42 2024

@author: colli
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.fft import fft, ifft
from scipy.interpolate import interp1d


# Set the working directory to the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

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

def force(t, F0, Omega):
    """
    Computes the force as a function of time.
    """
    return F0 * np.sin(Omega * t)

# Frequency range for the analysis (same as simulation, 0-1500 Hz)
freq_range = np.linspace(0, 1500, 4500)
omega_range = 2 * np.pi * freq_range

# Compute the transfer function over the frequency range for a specific element
H_values = np.array([H_calcul(w, modes, pulsations_propres, damping_factors)[11, 0] for w in omega_range])

epsilon = 1e-10  # Petite valeur pour éviter la division par zéro

# Bode Plot: Amplitude
magnitude = 20 * np.log10(np.abs(H_values) + epsilon)

# Plot the Bode Plot without annotations
plt.figure(figsize=(8, 6))
# Bode Amplitude
plt.subplot(2, 1, 1)
plt.plot(freq_range, magnitude, label=r'Calculated $H(\omega)$')
plt.plot(freq_frf, 20 * np.log10(np.sqrt(re_frf**2 + im_frf**2)), '--', label=r'Simulated FRF Data')
plt.xlabel(r'Frequency [Hz]', size=11)
plt.ylabel(r'Magnitude [dB]', size=11)
plt.legend()
plt.title(r'Diagramme de Bode en amplitude', size=14)
plt.xlim([-100, 1500])  # Avoid the issue with starting from 0 Hz
plt.ylim([-70, -30])
plt.grid(True, linestyle='--', linewidth=0.5, color='black', alpha=0.5)
plt.minorticks_on()
plt.grid(True, which='minor', linestyle=':', linewidth=0.3, color='gray', alpha=0.7)

# Custom legend with a box
legend_handles = [
    plt.Line2D([0], [0], color='r', linestyle='--', label=r'Resonance Frequencies'),
    plt.Line2D([0], [0], color='b', linestyle='--', label=r'Antiresonance Frequencies')
]
#plt.legend(handles=legend_handles, loc='lower left', fontsize=10, frameon=True, framealpha=0.8, edgecolor='black')

# Save the Bode plot without annotations as PDF
plt.savefig('bode_plot_no_annotations.pdf')

# Plot the Bode Plot with annotations
plt.figure(figsize=(8, 6))
# Bode Amplitude
plt.subplot(2, 1, 1)
plt.plot(freq_range, magnitude, label=r'Calculated $H(\omega)$')
plt.plot(freq_frf, 20 * np.log10(np.sqrt(re_frf**2 + im_frf**2)), '--', label=r'Simulated FRF Data')
plt.xlabel(r'Frequency [Hz]', size=11)
plt.ylabel(r'Magnitude [dB]', size=11)
plt.legend()
plt.title(r'Diagramme de Bode en amplitude (avec les pics de résonance et anti-résonance)', size=14)
plt.xlim([-100, 1500])  # Avoid the issue with starting from 0 Hz
plt.ylim([-70, -30])
plt.grid(True, linestyle='--', linewidth=0.5, color='black', alpha=0.5)
plt.minorticks_on()
plt.grid(True, which='minor', linestyle=':', linewidth=0.3, color='gray', alpha=0.7)

# Highlight resonance and anti-resonance frequencies
resonance_frequencies = frequencies  # Use natural frequencies for resonance
for i, freq in enumerate(resonance_frequencies):
    plt.axvline(freq, color='r', linestyle='--', alpha=0.5)  # Vertical line at each resonance
    if i == 0:
        plt.annotate(rf'Res: {freq:.1f} Hz', (freq, -55), textcoords="offset points", xytext=(0, 10),
                     ha='center', color='red')  # Lower y-coordinate for the first resonance
    else:
        plt.annotate(rf'Res: {freq:.1f} Hz', (freq, -36), textcoords="offset points", xytext=(0, 10),
                     ha='center', color='red')

# Highlight antiresonance frequencies
antiresonance_indices, _ = find_peaks(-magnitude)
antiresonance_frequencies = freq_range[antiresonance_indices]
for freq in antiresonance_frequencies:
    plt.axvline(freq, color='b', linestyle='--', alpha=0.5)  # Vertical line at each antiresonance
    plt.annotate(rf'Anti: {freq:.1f} Hz', (freq, -50), textcoords="offset points", xytext=(0, -15),
                 ha='center', color='blue')    

# Custom legend with a box
legend_handles = [
    plt.Line2D([0], [0], color='r', linestyle='--', label=r'Fréquence de Resonance'),
    plt.Line2D([0], [0], color='b', linestyle='--', label='Fréquence d\'Antiresonance')
]
plt.legend(handles=legend_handles, loc='lower left', fontsize=10, frameon=True, framealpha=0.8, edgecolor='black')

# Save the Bode plot with annotations as PDF
plt.savefig('bode_plot_with_annotations.pdf')
plt.show()




# Nyquist Plot: Real vs Imaginary
real_part = np.real(H_values)
imag_part = np.imag(H_values)

plt.figure(figsize=(8, 8))
plt.plot(real_part, imag_part, label=r'Calculated $H(\omega)$')
plt.plot(re_frf, im_frf, '--', label=r'Simulated FRF Data')

# Récupérer les parties réelles et imaginaires pour les fréquences de résonance
resonance_real = np.real([H_calcul(2 * np.pi * freq, modes, pulsations_propres, damping_factors)[11, 0] for freq in frequencies])
resonance_imag = np.imag([H_calcul(2 * np.pi * freq, modes, pulsations_propres, damping_factors)[11, 0] for freq in frequencies])

# Récupérer les parties réelles et imaginaires pour les fréquences d'anti-résonance
antiresonance_real = np.real([H_calcul(2 * np.pi * freq, modes, pulsations_propres, damping_factors)[11, 0] for freq in antiresonance_frequencies])
antiresonance_imag = np.imag([H_calcul(2 * np.pi * freq, modes, pulsations_propres, damping_factors)[11, 0] for freq in antiresonance_frequencies])

# Calculer la dérivée du tracé
delta_real = np.diff(real_part)
delta_imag = np.diff(imag_part)

# Détecter les changements de signe dans la dérivée
sign_changes = np.where(np.diff(np.sign(delta_real)))[0]

# Ajouter la dernière fréquence de résonance détectée
last_resonance_index = sign_changes[-1]
last_resonance_freq = freq_range[last_resonance_index]

# Tracer les points de résonance et d'anti-résonance sur le diagramme de Nyquist
plt.scatter(resonance_real, resonance_imag, color='r', label=r'Points de Résonance')
plt.scatter(antiresonance_real, antiresonance_imag, color='b', label='Points d\'Antiresonance')


plt.xlabel(r'Partie Réelle', size=14)
plt.ylabel(r'Partie Imaginaire', size=14)
plt.title(r'Diagramme de Nyquist', size=18)
plt.legend()
plt.grid(True, linestyle='--', linewidth=0.5, color='black', alpha=0.5)
plt.minorticks_on()
plt.grid(True, which='minor', linestyle=':', linewidth=0.3, color='gray', alpha=0.7)
plt.axis('equal')

plt.savefig('nyquist_plot.pdf')
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

#set the w for a 50 kmh speed
w = 436 #rad/s
#w = 550.22 #rad/s

# Paramètres de la force
F0 = 450  # Amplitude de la force en Newtons
Omega = 436  # Fréquence d'excitation en rad/s
T_max = 0.15  # Durée de la simulation en secondes

# Compute the maximum amplitude of the response (acceleration) for each reference point
max_amplitudes = np.zeros(len(reference_points))

# Temps de simulation
t_values = np.linspace(0, T_max, num=1000)  # Ajustez le nombre de points si nécessaire

for i in range(len(reference_points)):
    H_values_point = H_calcul(w, modes, pulsations_propres, damping_factors)[i, 0]
    accelerations = np.array([H_values_point * force(t,F0,w) for t in t_values])
    max_amplitudes[i] = np.max(np.abs(accelerations))
    


distances_mm = [point["distance_mm"] for point in reference_points]

# Interpolation des points
interp_func = interp1d(distances_mm, max_amplitudes, kind='linear')
distances_interp = np.linspace(min(distances_mm), max(distances_mm), num=500)
max_amplitudes_interp = interp_func(distances_interp)

# Tracer l'évolution de l'amplitude maximale pour chaque point de référence
plt.figure(figsize=(10, 6))
plt.plot(distances_interp, max_amplitudes_interp, linestyle='-', color='b', label='Interpolation')  # Ligne interpolée bleue
plt.scatter(distances_mm, max_amplitudes, color='red', label='Valeurs au capteurs')  # Points rouges
plt.xlabel(r'Distance depuis la fourche le long du cadre [mm]', size=14)
plt.ylabel(r'Amplitude Maximale de la Réponse (Accélération) $[m/s^2]$', size=11)
plt.title(r"Évolution de l'Amplitude Maximale de la Réponse pour Chaque Point de Référence", size=16)
plt.xticks(distances_mm)  # Définir les graduations de l'axe des x à chaque distance
plt.xlim(0, 1200)  # Définir les limites de l'axe des x de 0 à 1200 mm
plt.grid(True, linestyle='--', linewidth=0.5, color='black', alpha=0.5)
plt.minorticks_on()
plt.grid(True, which='minor', linestyle=':', linewidth=0.3, color='gray', alpha=0.7)

# Ajouter les noms des points au-dessus des points d'amplitude maximale avec un décalage vertical
offset = 0.2  # Ajustez cette valeur selon vos besoins
for i, point in enumerate(reference_points):
    if point["point"] == "P1":
        plt.text(distances_mm[i] + 15, max_amplitudes[i] + offset - 0.0015, point["point"], ha='center', va='bottom', fontsize=10)
    elif point["point"] == "P5":
        plt.text(distances_mm[i], max_amplitudes[i] + offset , "P5-P6", ha='center', va='bottom', fontsize=10)
    elif point["point"] == "P6":
        continue  # Skip P6 as it is grouped with P5
    elif point["point"] == "P12":
        plt.text(distances_mm[i], max_amplitudes[i] + offset + 0.0002, point["point"], ha='center', va='bottom', fontsize=10)
    elif point["point"] == "P13":
        plt.text(distances_mm[i] - 15, max_amplitudes[i] + offset + 0.0001, point["point"], ha='center', va='bottom', fontsize=10)
    elif point["point"] == "P14":
        plt.text(distances_mm[i] - 25, max_amplitudes[i] , point["point"], ha='center', va='bottom', fontsize=10)
    else:
        plt.text(distances_mm[i], max_amplitudes[i] + offset, point["point"], ha='center', va='bottom', fontsize=10)

# Ajouter la légende
plt.legend()

plt.savefig('amplitude_plot.pdf')
plt.show()




# Définition des paramètres de simulation
F0 = 450  # Amplitude de la force en Newtons
Omega = 436  # Fréquence d'excitation en rad/s
T_max = 0.15  # Durée de la simulation en secondes

# Temps de simulation
t_values = np.linspace(0, T_max, num=1000)


# Calcul de la réponse temporelle pour le point P12
H_values_point_P12 = H_calcul(Omega, modes, pulsations_propres, damping_factors)[11, 0]
accelerations = np.array([H_values_point_P12 * force(t, F0, Omega) for t in t_values])


# Partie réelle de l'accélération
acc_real = np.real(accelerations)

# Magnitude (module) de l'accélération avec signe
acc_magnitude_signed = np.abs(accelerations) * np.sign(acc_real)

# Tracer la réponse temporelle avec le module signé en utilisant le même template
plt.figure(figsize=(10, 6))
plt.plot(t_values, acc_magnitude_signed, linestyle='-', color='b', label=r"Réponse temporelle (accélération) avec module signé pour P12")
plt.xlabel(r"Temps [s]", size=14)
plt.ylabel(r"Accélération $[m/s^2]$", size=14)
plt.title(r"Réponse temporelle de l'accélération au siège du pilote en fonction du temps", size=16)
plt.grid(True, linestyle='--', linewidth=0.5, color='black', alpha=0.5)
plt.minorticks_on()
plt.grid(True, which='minor', linestyle=':', linewidth=0.3, color='gray', alpha=0.7)
plt.savefig('time_response_plot.pdf')
plt.show()






# Update damping factors to 0.02 for all modes
damping_factors = np.full_like(damping_factors, 0.02)

F0 = 450  # Amplitude de la force en Newtons
wavelength = 0.2  # Longueur d'onde en mètres (longueur d'un pavé)
T_max = 0.15  # Durée de la simulation en secondes
speeds_kmh = np.linspace(50, 70, num=200).astype(float)  # Vitesse en km/h

# Conversion des vitesses en m/s
speeds_m_s = speeds_kmh * 1000 / 3600

# Calcul de la période d'excitation pour chaque vitesse
T = wavelength / speeds_m_s

# Calcul de la fréquence d'excitation (Omega) pour chaque vitesse
speeds_rad_s = 2 * np.pi / T
#print(speeds_rad_s)

# Compute the maximum amplitude of the response for each speed
max_amplitudes_speeds = np.zeros(len(speeds_rad_s))

# Temps de simulation
t_values = np.linspace(0, T_max, num=1000)



for i, w in enumerate(speeds_rad_s):
    H_values_point_P12 = H_calcul(w, modes, pulsations_propres, damping_factors)[11, 0]
    accelerations = np.array([H_values_point_P12 * force(t, F0, w) for t in t_values])

    acc_real = np.real(accelerations)
    acc_magnitude_signed = np.abs(accelerations) * np.sign(acc_real)
    max_amplitudes_speeds[i] = np.max(acc_magnitude_signed)

# Find the speed with the maximum amplitude
max_index = np.argmax(max_amplitudes_speeds)
max_speed_kmh = speeds_kmh[max_index]
max_amplitude = max_amplitudes_speeds[max_index]

print(f"La vitesse à laquelle il y a le maximum d'amplitude est {max_speed_kmh:.2f} km/h avec une amplitude maximale de {max_amplitude:.4f} m/s^2")

"""
#de la merde pour test un truc 
# Compute the maximum amplitude of the response for each speed and each point
max_amplitudes_speeds = np.zeros(len(speeds_rad_s))

# Temps de simulation
t_values = np.linspace(0, T_max, num=1000)

for i, w in enumerate(speeds_rad_s):
    for j in range(len(reference_points)):
        H_values_point = H_calcul(w, modes, pulsations_propres, damping_factors)[i, 0]
        accelerations = np.array([H_values_point * force(t,F0,w) for t in t_values])
        max_amplitudes[j] = np.max(np.abs(accelerations))
    print('max_amplitudes',max_amplitudes)
    
"""




# Tracer l'amplitude maximale de la réponse en fonction de la vitesse
plt.figure(figsize=(10, 6))
plt.plot(speeds_kmh, max_amplitudes_speeds, linestyle='-', color='b')
plt.xlabel(r'Vitesse de la moto [km/h]', size=14)
plt.ylabel(r'Amplitude Maximale de la Réponse (Accélération) $[m/s^2]$', size=12)
plt.title(r"Amplitude Maximale de la Réponse en Fonction de la Vitesse de la Moto", size=16)
plt.grid(True, linestyle='--', linewidth=0.5, color='black', alpha=0.5)
plt.minorticks_on()
plt.grid(True, which='minor', linestyle=':', linewidth=0.3, color='gray', alpha=0.7)

# Highlight the maximum amplitude point
plt.plot(max_speed_kmh, max_amplitude, 'ro')  # Red dot at the maximum point
plt.annotate(f'Max: {max_speed_kmh:.2f} km/h', xy=(max_speed_kmh, max_amplitude), xytext=(max_speed_kmh + 1, max_amplitude + 0.1), fontsize=12, color='red')

# Save the speed response plot as PDF
#plt.savefig('speed_response_plot.pdf')

plt.show()


# Update damping factors to 0.02 for all modes
damping_factors = np.full_like(damping_factors, 0.02)

# Compute the transfer function over the frequency range for a specific element
H_values = np.array([H_calcul(w, modes, pulsations_propres, damping_factors)[11, 0] for w in omega_range])

# Bode Plot: Amplitude
magnitude = 20 * np.log10(np.abs(H_values) + epsilon)

# Plot the Bode Plot with updated damping factors
plt.figure(figsize=(8, 6))
# Bode Amplitude
plt.subplot(2, 1, 1)
plt.plot(freq_range, magnitude, label=r'Calculated $H(\omega)$ with $\zeta = 0.02$')
plt.plot(freq_frf, 20 * np.log10(np.sqrt(re_frf**2 + im_frf**2)), '--', label=r'Simulated FRF Data')
plt.xlabel(r'Frequency [Hz]', size=11)
plt.ylabel(r'Magnitude [dB]', size=11)
plt.legend()
plt.title(r'Diagramme de Bode en amplitude avec $\zeta = 0.02$', size=14)
plt.xlim([-100, 1500])  # Avoid the issue with starting from 0 Hz
plt.ylim([-70, -30])
plt.grid(True, linestyle='--', linewidth=0.5, color='black', alpha=0.5)
plt.minorticks_on()
plt.grid(True, which='minor', linestyle=':', linewidth=0.3, color='gray', alpha=0.7)

# Recalculate antiresonance frequencies
antiresonance_indices, _ = find_peaks(-magnitude)
antiresonance_frequencies = freq_range[antiresonance_indices]

# Tracer les lignes verticales et les annotations pour les fréquences d'anti-résonance
for freq in antiresonance_frequencies:
    plt.axvline(freq, color='b', linestyle='--', alpha=0.5)  # Ligne verticale à chaque anti-résonance
    plt.annotate(rf'Anti: {freq:.1f} Hz', (freq, -50), textcoords="offset points", xytext=(0, -15),
                 ha='center', color='blue')

# Save the Bode plot with updated damping factors as PDF
plt.savefig('bode_plot_damping_0_02.pdf')
plt.show()

# Print the antiresonance frequencies
print("Antiresonance Frequencies (Hz):")
for freq in antiresonance_frequencies:
    print(f"{freq:.2f}")