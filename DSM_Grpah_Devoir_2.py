# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 10:21:29 2024

@author: colli
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.fft import fft, fftfreq

# Function to smooth the data using a moving average filter
def smooth_data(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='same')

# 1. Charger les données temporelles
time_data = np.loadtxt('P2024_irf_acc.txt', delimiter='\t')
time = time_data[:, 0]
acceleration = time_data[:, 1]

# 2. Lisser les données d'accélération (pour réduire le bruit)
window_size = 5  # Ajuster la taille de la fenêtre si nécessaire pour vos données
acceleration_smooth = smooth_data(acceleration, window_size)

# 3. Tracer les données lissées et originales pour comparaison
plt.figure(figsize=(8, 6))  # Ajuster la taille pour correspondre au format d'exemple
plt.plot(time, acceleration, label='Données originales', color='b')
plt.plot(time, acceleration_smooth, label='Données lissées', linestyle='--', color='r')
plt.title(r"Données d'accélération $a(t)$", fontsize=16)
plt.xlabel(r'Temps $t$ [s]', fontsize=14)
plt.ylabel(r'Accélération $a(t)$ [m/s²]', fontsize=14)
plt.legend(loc='best', fontsize=12)
plt.grid(True, linestyle='--', linewidth=0.5, color='black', alpha=0.5)  # Grille principale
plt.minorticks_on()  # Petits graduations
plt.grid(True, which='minor', linestyle=':', linewidth=0.3, color='gray', alpha=0.7)  # Grille mineure
#plt.savefig("acceleration_smoothed.pdf", format="pdf")
plt.show()


# 4. Détecter les pics dans les données lissées
peak_threshold = 0.0025  # Ajuster en fonction de vos données (peut être défini comme une fraction de la valeur max)
min_distance = 20  # Ajuster cela en fonction de la résolution temporelle et de l'espacement attendu des pics

# Détecter tous les pics
peaks, _ = find_peaks(acceleration_smooth, height=peak_threshold, distance=min_distance)

# 5. Filtrer les pics après t > 0.02
filtered_peaks = peaks[time[peaks] > 0.02]

# 6. Tracer les pics détectés après t > 0.02
plt.figure(figsize=(8, 6))
plt.plot(time, acceleration_smooth, label='Données lissées', color='b')
plt.plot(time[filtered_peaks], acceleration_smooth[filtered_peaks], 'rx', label='Pics détectés (t > 0.02s)')
plt.axvline(x=0.02, color='green', linestyle='--', linewidth=1, label=r'$t = 0.02 \, \text{s}$')  # Ligne verticale verte pointillée
plt.title(r'Pics détectés dans les données lissées d\'accélération (après $t > 0.02$)', fontsize=16)
plt.xlabel(r'Temps $t$ [s]', fontsize=14)
plt.ylabel(r'Accélération $a(t)$ [m/s²]', fontsize=14)
plt.legend(loc='best', fontsize=12)
plt.grid(True, linestyle='--', linewidth=0.5, color='black', alpha=0.5)
plt.minorticks_on()
plt.grid(True, which='minor', linestyle=':', linewidth=0.3, color='gray', alpha=0.7)
#plt.savefig("Detected_Peaks_in_Smoothed_Acceleration_Data.pdf", format="pdf")
plt.show()


# 7. Estimer la fréquence naturelle amortie à partir des pics nettoyés
peak_times = time[filtered_peaks]
T = np.mean(np.diff(peak_times))  # Période moyenne entre les pics
f_damped = 1 / T  # Fréquence naturelle amortie


# 8. Calculer le taux d'amortissement en utilisant la méthode logarithmique
if len(filtered_peaks) >= 2:  # Vérifier qu'il y a au moins deux pics
    a_1 = acceleration_smooth[filtered_peaks[0]]  # Amplitude du premier pic
    a_Nk = acceleration_smooth[filtered_peaks[-1]]  # Amplitude du dernier pic

    k = len(filtered_peaks) - 1  # Nombre de périodes entre le premier et le dernier pic
    zeta = np.log(a_1 / a_Nk) / (k * 2 * np.pi)  # Calculer le taux d'amortissement avec la nouvelle formule

    print(f'Fréquence naturelle amortie à partir des données temporelles : {f_damped} Hz')
    print(f'Taux d\'amortissement à partir des données temporelles : {zeta}')
else:
    print("Pas assez de pics détectés pour calculer le taux d'amortissement.")


# 9. Charger les données FRF (3 colonnes : fréquence, partie réelle du FRF, partie imaginaire du FRF)
frf_data = np.loadtxt('P2024_frf_acc.txt', delimiter='\t')
frequency = frf_data[:, 0]
Re_FRF = frf_data[:, 1]
Im_FRF = frf_data[:, 2]
FRF = Re_FRF + 1j * Im_FRF

# 10. Diagramme de Bode
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

# Tracé de la magnitude
plt.subplot(2, 1, 1)
plt.plot(frequency, 20 * np.log10(mag), color='b')  # Magnitude en dB
plt.title(r'Diagramme de Bode', fontsize=16)
plt.ylabel(r'Magnitude [dB]', fontsize=14)
plt.grid(True, linestyle='--', linewidth=0.5, color='black', alpha=0.5)
plt.minorticks_on()
plt.grid(True, which='minor', linestyle=':', linewidth=0.3, color='gray', alpha=0.7)

# Tracé de la phase
plt.subplot(2, 1, 2)
plt.plot(frequency, phase, color='r')  # Phase en degrés
plt.xlabel(r'Fréquence [Hz]', fontsize=14)
plt.ylabel(r'Phase [degrés]', fontsize=14)
plt.grid(True, linestyle='--', linewidth=0.5, color='black', alpha=0.5)
plt.minorticks_on()
plt.grid(True, which='minor', linestyle=':', linewidth=0.3, color='gray', alpha=0.7)

#plt.savefig("Bode_Diagram.pdf", format="pdf")
plt.show()


# 10. Méthode des demi-puissances - améliorée en fonction du facteur de qualité
max_magnitude = np.max(mag)
half_power_level = max_magnitude / np.sqrt(2)

# Trouver les fréquences aux points de demi-puissance
half_power_points = np.where(mag >= half_power_level)[0]
f1 = frequency[half_power_points[0]]  # Premier point de demi-puissance
f2 = frequency[half_power_points[-1]]  # Deuxième point de demi-puissance

# Convertir les fréquences en fréquences angulaires (omega)
omega1 = 2 * np.pi * f1
omega2 = 2 * np.pi * f2
f_a = frequency[np.argmax(mag)]
omega_a = 2 * np.pi * f_a  # Fréquence angulaire au pic

# Calculer delta omega (largeur entre les points de demi-puissance)
delta_omega = omega2 - omega1

Q = omega_a / delta_omega


# zeta avec le système 2 eq à 2 inconnues 
# on note w_a = max_magnitude (pour la formule)
zeta = np.sqrt(delta_omega**2/((4*(omega_a)**2)+delta_omega**2))

# Calculer le facteur de qualité et le taux d'amortissement zeta (hypothèses Charles)
# zeta = 1 / (2 * Q)

w_0_a =( omega_a / np.sqrt(1-(2*(zeta)**2)) )  
w_0_b =( delta_omega / (2 * zeta) )  

f_0_a = w_0_a  / (2*np.pi)
f_0_b = w_0_b  / (2*np.pi)

print(f'Fréquence naturelle (f_naturelle du diagramme de Bode) : {f_0_b} Hz')
print(f'Facteur de qualité : {Q}')
print(f'Taux d\'amortissement (zeta) avec le diagramme de Bode : {zeta}')

# 11. Imprimer la phase à la fréquence naturelle
phase_at_natural = phase[np.argmax(mag)]
#print(f'Phase à la fréquence naturelle : {phase_at_natural} degrés')





# Valeur donnée de M_eq
M_eq = 87.5

# 1. Load FRF data (3 columns: frequency, real part of FRF, imaginary part of FRF)
frf_data = np.loadtxt('P2024_frf_acc.txt', delimiter='\t')
Re_FRF = frf_data[:, 1]  # Real part of FRF
Im_FRF = frf_data[:, 2]  # Imaginary part of FRF

# 2. Trouver les indices où Re_FRF = 0

mask = np.logical_and(FRF.real >= -0.001, FRF.real <= 0.001)
result = FRF[mask]

if len(result) > 0:
    max_imaginary = np.max(result.imag)  # La plus grande partie imaginaire du mask 
    
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
plt.title(r'Diagramme de Nyquist', fontsize=16)
plt.xlabel(r'Partie réelle', fontsize=14)
plt.ylabel(r'Partie imaginaire', fontsize=14)
plt.grid(True, linestyle='--', linewidth=0.5, color='black', alpha=0.5)
plt.minorticks_on()
plt.grid(True, which='minor', linestyle=':', linewidth=0.3, color='gray', alpha=0.7)
plt.axis('equal')  # Assurer un repère orthonormé
plt.legend()
#plt.savefig("Nyquist_Diagram.pdf", format="pdf")
plt.show()

"""
#trouver la freqence naturelle f_0 avec Nyquist 

imaginary_parts = FRF.imag
max_im_index = np.argmax(imaginary_parts)

real_part_of_max_imaginary = FRF[max_im_index].real
print(f"La partie réelle correspondant à la plus grande partie imaginaire est : {real_part_of_max_imaginary}")
"""













