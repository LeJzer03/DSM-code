# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 17:58:42 2024

@author: colli
"""


# Importation des bibliothèques nécessaires
import numpy as np
import matplotlib.pyplot as plt

# Données de base
F0 = 450  # Force d'excitation en Newton
omega = 436  # Fréquence d'excitation en rad/s
L = 0.2  # Longueur d'onde en mètres (distance entre pavés)

# Paramètres temporels pour l'excitation
time_interval = 0.15  # Intervalle de temps pour le tracé
time_steps = np.linspace(0, time_interval, 1000)
F_z = F0 * np.sin(omega * time_steps)

# Tracé de l'évolution temporelle de la force d'excitation
plt.figure(figsize=(10, 5))
plt.plot(time_steps, F_z, label="Force d'excitation Fz")
plt.xlabel("Temps (s)")
plt.ylabel("Force (N)")
plt.title("Évolution temporelle de la force d'excitation")
plt.grid(True)
plt.legend()
plt.show()

# Chargement des données modales et de la FRF
# Remplacer les chemins par les fichiers fournis
frequencies, damping_ratios = np.loadtxt("P2024_f_eps_Part3.txt", unpack=True)
mode_shapes = np.loadtxt("P2024_modes_Part3.txt")
frf_data = np.loadtxt("P2024_frf_Part3_f_ds.txt")

# Construction de la matrice FRF (en termes d'accélération)
mass_matrix = np.diag(frequencies**2)  # Masses normalisées
damping_matrix = np.diag(2 * damping_ratios * frequencies)  # Amortissement

# Tracé du diagramme de Bode (Amplitude en échelle semi-logarithmique)
freq_range = np.linspace(0, 1500, 1000)
FRF_amplitude = np.zeros_like(freq_range, dtype=float)

for i, freq in enumerate(freq_range):
    frf_sum = 0
    for mode in range(len(frequencies)):
        frf_mode = F0 / np.sqrt((mass_matrix[mode, mode] - freq**2)**2 + (damping_matrix[mode, mode] * freq)**2)
        frf_sum += frf_mode
    FRF_amplitude[i] = frf_sum

plt.figure(figsize=(10, 5))
plt.semilogy(freq_range, FRF_amplitude)
plt.xlabel("Fréquence (Hz)")
plt.ylabel("Amplitude FRF (m/s²/N)")
plt.title("Diagramme de Bode en amplitude")
plt.grid(True)
plt.show()

# Diagramme de Nyquist
FRF_real = np.zeros_like(freq_range, dtype=float)
FRF_imag = np.zeros_like(freq_range, dtype=float)

for mode in range(len(frequencies)):
    real_part = F0 * (frequencies[mode]**2 - freq_range**2) / \
                ((frequencies[mode]**2 - freq_range**2)**2 + (2 * damping_ratios[mode] * frequencies[mode] * freq_range)**2)
    imag_part = F0 * (2 * damping_ratios[mode] * frequencies[mode] * freq_range) / \
                ((frequencies[mode]**2 - freq_range**2)**2 + (2 * damping_ratios[mode] * frequencies[mode] * freq_range)**2)
    FRF_real += real_part
    FRF_imag += imag_part

plt.figure(figsize=(10, 5))
plt.plot(FRF_real, FRF_imag, label="Nyquist")
plt.xlabel("Partie réelle de FRF")
plt.ylabel("Partie imaginaire de FRF")
plt.title("Diagramme de Nyquist")
plt.grid(True)
plt.legend()
plt.show()

# Comparaison avec les données de FRF fournies
plt.figure(figsize=(10, 5))
plt.plot(frf_data[:, 0], frf_data[:, 1], label="FRF Réelle", color="red")
plt.plot(freq_range, FRF_amplitude, label="FRF Calculée", linestyle="--")
plt.xlabel("Fréquence (Hz)")
plt.ylabel("FRF (m/s²/N)")
plt.title("Comparaison des FRF (réelle vs calculée)")
plt.legend()
plt.grid(True)
plt.show()

# Calcul de l'amplitude maximale de la réponse pour chaque point de référence
max_amplitudes = []
for shape in mode_shapes.T:  # Chaque colonne correspond à un mode
    response = np.abs(shape * F0)  # Réponse pour la force appliquée
    max_amplitudes.append(response.max())

# Tracé de l'évolution des amplitudes maximales
points = np.arange(1, len(max_amplitudes) + 1)
plt.figure(figsize=(10, 5))
plt.plot(points, max_amplitudes, marker="o")
plt.xlabel("Points de référence")
plt.ylabel("Amplitude maximale de réponse (m/s²)")
plt.title("Évolution des amplitudes maximales aux points de référence")
plt.grid(True)
plt.show()

# Tracé de la réponse temporelle en accélération au siège du conducteur
driver_response = np.fft.ifft(FRF_amplitude * np.sin(omega * time_steps))

plt.figure(figsize=(10, 5))
plt.plot(time_steps, np.real(driver_response), label="Réponse au siège du conducteur")
plt.xlabel("Temps (s)")
plt.ylabel("Accélération (m/s²)")
plt.title("Réponse temporelle en accélération au siège du conducteur")
plt.grid(True)
plt.legend()
plt.show()

# Variation du coefficient d'amortissement à epsilon=0.02 pour vitesses entre 50 et 70 km/h
damping_ratio_adjusted = 0.02
speeds = np.linspace(50, 70, 5)  # Vitesse entre 50 et 70 km/h
omega_speeds = 2 * np.pi * speeds / 3.6 / L  # Calcul de omega pour chaque vitesse

amplitudes_by_speed = []
for omega_v in omega_speeds:
    amplitude_v = F0 / np.sqrt((mass_matrix.diagonal() - omega_v**2)**2 + (2 * damping_ratio_adjusted * frequencies * omega_v)**2)
    amplitudes_by_speed.append(amplitude_v)

# Tracé de l'amplitude en fonction de la vitesse
plt.figure(figsize=(10, 5))
plt.plot(speeds, [max(amp) for amp in amplitudes_by_speed], marker="o", color="purple")
plt.xlabel("Vitesse (km/h)")
plt.ylabel("Amplitude maximale de réponse (m/s²)")
plt.title("Influence de la vitesse sur la réponse d'accélération au siège du conducteur")
plt.grid(True)
plt.show()
