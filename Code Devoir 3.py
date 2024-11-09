# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 17:58:42 2024

@author: colli
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal

# 1. Import des données
def load_modal_data():
    # Charger les données modales depuis les fichiers
    frequencies_damping = np.loadtxt("P2024_f_eps_Part3.txt")
    mode_shapes = np.loadtxt("P2024_modes_Part3.txt")
    frf_data = np.loadtxt("P2024_frf_Part3_f_ds.txt")
    return frequencies_damping, mode_shapes, frf_data

# 2. Génération du signal d'excitation
def excitation_force(F0=450, omega=436, duration=0.15, sampling_rate=10000):
    t = np.linspace(0, duration, int(duration * sampling_rate))
    force = F0 * np.sin(omega * t)
    return t, force

# 3. Calcul de la matrice FRF (en utilisant les données modales)
def calculate_frf(frequencies_damping, mode_shapes):
    freqs = frequencies_damping[:, 0]  # Fréquences naturelles
    damp_ratios = frequencies_damping[:, 1]  # Coefficients d'amortissement
    num_points = mode_shapes.shape[0]
    frf_matrix = np.zeros((num_points, num_points, len(freqs)), dtype=complex)
    
    for i, (freq, damp) in enumerate(zip(freqs, damp_ratios)):
        omega_n = 2 * np.pi * freq
        for j in range(num_points):
            for k in range(num_points):
                # Calcul de la FRF pour chaque mode et chaque point de mesure
                frf_matrix[j, k, i] = mode_shapes[j, i] * mode_shapes[k, i] / \
                    (omega_n**2 * (1 - (damp ** 2)))
    return frf_matrix

# 4. Tracé des diagrammes de Bode et Nyquist
def plot_bode_nyquist(frf_data):
    freq = frf_data[:, 0]
    re_frf = frf_data[:, 1]
    im_frf = frf_data[:, 2]
    frf_complex = re_frf + 1j * im_frf
    
    # Diagramme de Bode
    plt.figure()
    plt.semilogx(freq, 20 * np.log10(np.abs(frf_complex)))
    plt.title("Diagramme de Bode")
    plt.xlabel("Fréquence (Hz)")
    plt.ylabel("Amplitude (dB)")
    plt.grid()

    # Diagramme de Nyquist
    plt.figure()
    plt.plot(re_frf, im_frf)
    plt.title("Diagramme de Nyquist")
    plt.xlabel("Re(FRF)")
    plt.ylabel("Im(FRF)")
    plt.grid()
    plt.show()

# 5. Calcul et tracé de la réponse temporelle au niveau du siège du conducteur
def time_response(t, force, frf_driver_seat):
    # Convolution du signal d'excitation avec la FRF du siège du conducteur
    response = np.convolve(force, frf_driver_seat, mode='same')
    plt.plot(t, response)
    plt.title("Réponse temporelle en accélération au siège du conducteur")
    plt.xlabel("Temps (s)")
    plt.ylabel("Accélération (m/s²)")
    plt.grid()
    plt.show()
    return response

# 6. Étude de l'effet de l'amortissement
def damping_effect_analysis(frequencies_damping, mode_shapes, frf_data, speeds):
    for speed in speeds:
        # Conversion de la vitesse en fréquence d'excitation
        omega = 436 * (speed / 50)
        t, force = excitation_force(omega=omega)
        
        # Calcul de la réponse en fonction du nouveau coefficient d’amortissement
        modified_damping = np.copy(frequencies_damping)
        modified_damping[:, 1] = 0.02  # Imposer un amortissement de 2%
        
        frf_matrix = calculate_frf(modified_damping, mode_shapes)
        frf_driver_seat = frf_data[:, 1] + 1j * frf_data[:, 2]  # Approximation pour la FRF du siège du conducteur
        
        response = time_response(t, force, frf_driver_seat)
        plt.plot(speed, np.max(response), 'o', label=f'{speed} km/h')
    
    plt.title("Amplitude maximale de la réponse en fonction de la vitesse")
    plt.xlabel("Vitesse (km/h)")
    plt.ylabel("Amplitude maximale de la réponse (m/s²)")
    plt.legend()
    plt.grid()
    plt.show()

# Exécution des fonctions
frequencies_damping, mode_shapes, frf_data = load_modal_data()
t, force = excitation_force()
frf_matrix = calculate_frf(frequencies_damping, mode_shapes)
plot_bode_nyquist(frf_data)
time_response(t, force, frf_data[:, 1])  # Hypothèse sur la FRF pour le siège du conducteur
damping_effect_analysis(frequencies_damping, mode_shapes, frf_data, speeds=[50, 60, 70])
