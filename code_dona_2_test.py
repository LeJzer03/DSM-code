# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 17:37:16 2023

@author: sdona
"""
import os
import numpy as np 
import scipy
from matplotlib import pyplot as plt 
from scipy.linalg import eigh
from scipy.linalg import eig
import sympy 
import seaborn as sns
from sympy import symbols, integrate


script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

L = 1.2  # Longueur du cadre [m]
E = 2.1e11  # Module d'Young [Pa]
I = 1.83866 * 10**-5  # Moment quadratique [m^4]
rho = 7850  # Densité du matériau [kg/m^3]
A = 0.15*0.15 - (0.15-0.02)**2 #0.0225  # Section transversale [m^2]


mB = 75.2  # Masse à la position B [kg]
xB = 0.4  # Position de la masse B sur x [m]
#jB = 1  # Moment d'inertie de la masse B [kg*m^2]
lB = 0.25  # Position de la masse B sur y [m]
jb = 1 + mB * lB**2  # Moment d'inertie de la masse B avec le th du transport[kg*m^2]

mDv = 75  # Masse à la position D (driver) [kg]
xDv = 1  # Position de la masse D sur x [m]
jDv = 10  # Moment d'inertie de la masse D [kg*m^2]

# Paramètres des ressorts
k1l = 10**4  # [N/m]
k1r = 10**9  # [Nm/rad]
x1 = 0 # Position des ressort 1 sur x [m]
k2l = 10**5  # [N/m]
k2r = 10**4  # [Nm/rad]
x2 = 0.8  # Position des ressort 2 sur x [m]



x = sympy.symbols("x")
#x, L = symbols('x L')


#functions for Rayleigh-Ritz (wi)
def phi(x, L, n):
    if n==0:
        return 1
    else:
        return (x/L)**n

"""
#derivatives of wi
def phi_deriv(x, L, n):
    if n == 0:
        return 0
    if n == 1:
        return 1/L
    else:
        return (n/L) * (x/L)**(n-1)
"""
def phi_deriv(x, L, n):
    x_sym = sympy.symbols("x")
    phi_expr = phi(x_sym, L, n)
    phi_deriv_expr = sympy.diff(phi_expr, x_sym)
    return phi_deriv_expr.subs(x_sym, x)

#matrix of stiffness
def stiff(N) : 
    """
    Constructs the stiffness matrix K for N degrees of freedom using symbolic integration.

    Parameters:
    - N: The number of degrees of freedom.

    Returns:
    - K: The stiffness matrix of size NxN.
    """
    K = np.zeros([N,N])
    for i in range(N):
        for j in range(N):
            fi = phi(x, L, i)
            fj = phi(x, L, j)
            d2fi = sympy.diff(fi, x, x)
            d2fj = sympy.diff(fj, x, x)

            # Calculate the stiffness contributions from material properties and spring constants
            k_frame = sympy.integrate(E*I*d2fi*d2fj, (x, 0, L))  # Bending stiffness

            # Paramètres des ressorts
            #k1l = 10**4  # [N/m]
            #k1r = 10**9  # [Nm/rad]
            #x1 = 0  # Position des ressort 1 sur x [m]
            #k2l = 10**5  # [N/m]
            #k2r = 10**5  # [Nm/rad]
            #x2 = 0.8  # Position des ressort 2 sur x [m]
            
            # Calcul des contributions des ressorts linéaires et rotationnels
            fi1l = phi(x1, L, i)
            fj1l = phi(x1, L, j)
            fi2l = phi(x2, L, i)
            fj2l = phi(x2, L, j)
            
            dfi1r = phi_deriv(x1, L, i)
            dfj1r = phi_deriv(x1, L, j)
            dfi2r = phi_deriv(x2, L, i)
            dfj2r = phi_deriv(x2, L, j)
            
            # Contribution des ressorts linéaires
            k_k1l = k1l * fi1l * fj1l
            k_k2l = k2l * fi2l * fj2l
            
            # Contribution des ressorts rotationnels
            k_k1r = k1r * dfi1r * dfj1r
            k_k2r = k2r * dfi2r * dfj2r
            
            # Calcul de la matrice de raideur totale
            K[i][j] = k_frame + k_k1l + k_k2l + k_k1r + k_k2r
            

    #print("matrice de raideur :", K)
            
    return K

"""
def T_max(w):
    M = np.zeros((len(w),len(w)))
    for i in range (len(w)):
        for j in range(len(w)):
            M[i][j] = m * integrate(w[i]*w[j], (x, 0, l))
                    + M_mot * w[i].subs(x,2*a).evalf() * w[j].subs(x,2*a).evalf()
                    + M_dv * w[i].subs(x,L+a).evalf() * w[j].subs(x,L+a).evalf()
                    + (J_B + M_mot*h**2) * w[i].diff(x).subs(x,2*a).evalf() * w[j].diff(x).subs(x,2*a).evalf()
                    + J_D * w[i].diff(x).subs(x,L+a).evalf() * w[j].diff(x).subs(x,L+a).evalf()
            
    return M

"""




#matrix of mass
def mass(N) :
    """
    #Constructs the mass matrix M for N degrees of freedom using symbolic integration.

    #Parameters:
    #- N: The number of degrees of freedom.
    
    #Returns:
    #- M: The mass matrix of size NxN.
    """

    M = np.zeros([N, N])
    for i in range(N):
        for j in range(N):
            fi = phi(x, L, i)
            fj = phi(x, L, j)
            m_frame = sympy.integrate(rho*A*fi*fj, (x, 0, L))

            
            # Add concentrated mass contributions at specific points
            fi_B = phi(xB, L, i)
            fj_B = phi(xB, L, j)
            fi_Dv = phi(xDv, L, i)
            fj_Dv = phi(xDv, L, j)

            m_M_B = mB * fi_B * fj_B
            m_M_Dv = mDv * fi_Dv * fj_Dv
            
            # Add inertia contributions from rotational effects
            dfi_B = phi_deriv(xB, L, i)
            dfj_B = phi_deriv(xB, L, j)
            dfi_Dv = phi_deriv(xDv, L, i)
            dfj_Dv = phi_deriv(xDv, L, j)

            m_J_B = jb * dfi_B * dfj_B
            m_J_Dv = jDv * dfi_Dv * dfj_Dv
            
            M[i][j] = m_frame + m_M_B + m_M_Dv + m_J_B + m_J_Dv

    #print("matrice de masse :", M)
    
    return M


#computing n natural fréquencies with Rayleigh Ritz
def n_w(n, nb_approx):
    """
    Computes and returns the first n natural frequencies and their corresponding mode shapes
    using the Rayleigh-Ritz method.

    Parameters:
    - n: The number of natural frequencies to compute.
    - nb_approx: The number of approximations to use in the calculation.

    Returns:
    - w: A sorted array of the first n natural frequencies in Hz.
    - modes: The corresponding mode shapes.
    """
    K = stiff(n)
    M = mass(n)
    
    w_sq, modes = eigh(K,M)
    w = np.sort(np.real(np.sqrt(w_sq)))
    w = w / (2*np.pi) #[Hz] pour comparer
    
    return w, modes


#computing 6 first frequencies for n =  1 to 13
def convergence():
    w1 = np.zeros(13)
    w2 = np.zeros(13)
    w3 = np.zeros(13)
    w4 = np.zeros(13)
    w5 = np.zeros(13)

    for n in range(1,14):
        K = stiff(n)
        M = mass(n)
        
        w_sq, _ = eigh(K,M)
        w = np.sort(np.real(np.sqrt(w_sq)))
        w = w/(2*np.pi)
        
        w1[n-1] = w[0]
        if len(w) > 1:
            w2[n-1] = w[1]
        
        if len(w) > 2 :
            w3[n-1] = w[2]
        
        if len(w) > 3 :
            w4[n-1] = w[3]
        
        if len(w) > 4 :
            w5[n-1] = w[4]
        

    x = np.arange(1, 14, 1)
    
    
    fig, ax = plt.subplots()
    plt.grid(True, linestyle='--', color='0.80')
    
    # Ajout des courbes
    plt.plot(x, w1, color='red', label='f1')
    plt.plot(x[1:], w2[1:], color='orange', label='f2')
    plt.plot(x[2:], w3[2:], color='yellow', label='f3')
    plt.plot(x[3:], w4[3:], color='limegreen', label='f4')
    
    # Ajout de la ligne verticale violette en pointillé
    ax.axvline(x=10, color='purple', linestyle='--', label='Ordre de convergence choisi')

    # Ajout des légendes
    #plt.legend(loc='upper right', ncol=2)
    plt.legend(loc='upper right')

    # Limites de l'axe y
    plt.ylim(-100, 5000)

    # Étiquettes des axes
    plt.xlabel('Ordre de convergence')
    plt.ylabel('Fréquences propres [Hz]')

    # Ajout des graduations mineures
    ax.minorticks_on()

    # Ajout de la grille pour les graduations principales et mineures
    ax.grid(which='both', linestyle='--', linewidth='0.5', color='gray')
    ax.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')

    # Sauvegarde et affichage du graphique
    plt.savefig('1-convergence.pdf')
    plt.show()
        
# Computing relative errors on frequencies with n = 10 (good convergence)
def relative_errors(nb_approx):
    """
    Computes and returns the relative errors between the analytical and numerical frequencies.

    Parameters:
    - nb_approx: The number of approximations to use in the calculation.

    Returns:
    - errors: An array of relative errors (in percentage) for the first 4 frequencies.
    """
    w_anal, _ = n_w(10, nb_approx)
    w_num = np.loadtxt('P2024_f_Part4.txt')
    print("w_num:", w_num)
    errors = np.zeros(4)
    for i in range(4):
        errors[i] = (np.abs(w_anal[i] - w_num[i]) / w_num[i]) * 100
    
    return errors

def y(x, mode, n):
    """
    Computes the deflection at position x for given mode shape coefficients.
    
    Parameters:
    - x : array-like, positions along the beam where deflection is calculated.
    - mode : array, coefficients of the mode shape.
    - n : int, number of coefficients to consider (should match length of mode).

    Returns:
    - f : array, calculated deflection at each position x.
    """
    x = np.array(x)  # Ensure x is a numpy array for element-wise operations
    x = np.sort(x)
    f = 0
    for i in range(n):
        f += (x / L)**i * mode[i]
    return f
        
#computing theoritical and numerical modes with m being the number of the mode (beginning with 0)
def mode_shapes(m, nb_approx):
    print("m:", m)  
    _, modes_anal = n_w(10,nb_approx)
    modes_num = np.loadtxt('P2024_Modes_Part4.txt') * 0.001

    #print("Shape of mode_anal:", modes_anal.shape)
    
    x = np.linspace(0, 1.2, 1000)

    x_point = [0, 100, 200, 300, 400, 400, 500, 600, 700, 800, 901, 1000, 1101, 1200]
    x_point = [x / 1000 for x in x_point] # Convertir en mètres
    

    mode_anal = modes_anal[:, m]
    mode_num = modes_num[:, m]
    #print("mode dans mode_shapes\n")
    #print("mode_anal avant norm.:", mode_anal)
    #print("mode_num avant norm.:", mode_num)

    #normalization because modes are defined up to a constant so to have the same ones : putting q0 at 1
    mode_anal /= mode_anal[0]
    mode_num /= mode_num[0]
    #print("mode_anal après norm.:", mode_anal)
    #print("mode_num après norm.:", mode_num)
    
    
    f = y(x, mode_anal,10)
    
    fig, ax = plt.subplots()
    plt.grid(True, linestyle='--', color='0.80')
    
    
    plt.plot(x, f, color='blue', label='Mode théorique (Rayleigh-Ritz)')
    
    
    # Ajout des légendes
    plt.legend()
    
    # Étiquettes des axes
    plt.xlabel('x [m]')
    plt.ylabel('Déflection z(x) [m]')
    
    # Ajout des graduations mineures
    ax.minorticks_on()
    
    # Ajout de la grille pour les graduations principales et mineures
    ax.grid(which='both', linestyle='--', linewidth='0.5', color='gray')
    ax.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')
    
    # Sauvegarde et affichage du graphique avec un nom de fichier basé sur le mode
    filename = f'mode_{m + 1}.pdf'
    plt.savefig(filename)
    plt.show()
    
 
#computing the MAC matrice with n being the number of modes
def MAC(n, nb_approx):
    M = np.zeros([n,n])
    
    freq_th, modes_anal = n_w(10, nb_approx)
    freq_exp = np.loadtxt('P2024_f_Part4.txt')
    
    modes_num = np.loadtxt('P2024_Modes_Part4.txt') * 0.001
    
    x_point = [0, 100, 200, 300, 400, 400, 500, 600, 700, 800, 901, 1000, 1101, 1200]
    x = [x_calc / 1000 for x_calc in x_point]  # Convertir en mètres
    
    # Initialiser la matrice M
    n = len(freq_exp)
    M = np.zeros((n, n))
    
    # i = experimental modes
    for i in range(n):
        shape_num = modes_num[:, i]
        shape_num /= shape_num[0]
    
        # mode into matrix
        vec_num = np.zeros([len(shape_num), 1])
        for k in range(len(shape_num)):
            vec_num[k][0] = shape_num[k]
    
        vec_numT = vec_num.T
    
        # j = theoritical modes
        for j in range(n):
            mode_anal = modes_anal[:, j]
            mode_anal /= mode_anal[0]
    
            shape_anal = y(x, mode_anal, 10)
    
            # mode into matrix
            vec_anal = np.zeros([len(shape_anal), 1])
            for k in range(len(shape_anal)):
                vec_anal[k][0] = shape_anal[k]
    
            vec_analT = vec_anal.T
    
            num = np.abs(np.dot(vec_numT, vec_anal))
            num = num**2
    
            denom1 = np.dot(vec_numT, vec_num)
            denom2 = np.dot(vec_analT, vec_anal)
    
            macij = num[0][0] / (denom1[0][0] * denom2[0][0])
    
            M[3 - i][j] = macij
    
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.imshow(M, cmap='Blues', interpolation='nearest', vmin=0, vmax=1)
    plt.colorbar(label='Valeurs matrice MAC')
    
    # Ajouter des annotations manuellement
    for i in range(n):
        for j in range(n):
            plt.text(j, i, f'{M[i, j]:.2f}', ha='center', va='center', color='black')
    
    plt.xlabel("Fréquences théoriques (Rayleigh-Ritz) [Hz]")
    plt.ylabel("Fréquences expérimentales [Hz]")

    freq_th = [3.70510008e+00, 6.74786637e+01, 2.49959966e+02, 5.92337108e+02]
    freq_exp = [451.46, 243.85, 66.98, 3.71]
    # Utiliser les 4 premières fréquences théoriques et expérimentales pour les étiquettes des axes
    plt.xticks(ticks=np.arange(4), labels=[f'{freq:.2f}' for freq in freq_th], rotation='horizontal')
    plt.yticks(ticks=np.arange(4), labels=[f'{freq:.2f}' for freq in freq_exp], rotation='horizontal')
    
    plt.savefig('8-MAC.pdf')
    plt.show()
    
    return M

nb_approx_test = 10

modes_num = np.loadtxt('P2024_Modes_Part4.txt')*0.001


#computing n natural fréquencies with Rayleigh Ritz
w_test, modes_test = n_w(10,nb_approx_test)
print("Fréquences propres (Hz) :", w_test)
#print("Modes propres :", modes_test)
errors = relative_errors(nb_approx_test)
print("Erreurs relatives (%) :", errors)

for i in range(4):
    mode_shapes(i, nb_approx_test)

convergence()

MAC(4, nb_approx_test) 


"""
# Appeler les fonctions pour voir les résultats
convergence()
errors = relative_errors()
print("Erreurs relatives (%) :", errors)
mode_shapes(0)  # Exemple pour le premier mode (m=0)
MAC(6)  # Exemple pour les 6 premiers modes
"""




