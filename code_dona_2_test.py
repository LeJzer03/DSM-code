# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 17:37:16 2023

@author: sdona
"""

import numpy as np 
import scipy
from matplotlib import pyplot as plt 
from scipy.linalg import eigh
from scipy.linalg import eig
import sympy 
import seaborn as sns

L = 1.2  # Longueur du cadre [m]
E = 2.1e11  # Module d'Young [Pa]
I = 1.83866 * 10**-5  # Moment quadratique [m^4]
rho = 7850  # Densité du matériau [kg/m^3]
A = 0.0225  # Section transversale [m^2]


mB = 75.02  # Masse à la position B [kg]
xB = 0.4  # Position de la masse B sur x [m]
jB = 1  # Moment d'inertie de la masse B [kg*m^2]
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


#functions for Rayleigh-Ritz (wi)
def phi(x, L, n):
    if n==0:
        return 1
    else:
        return (x/L)**n

#derivatives of wi
def phi_deriv(x, L, n):
    if n == 0:
        return 0
    if n == 1:
        return 1/L
    else:
        return (n/L) * (x/L)**(n-1)

#matrix of stiffness
def stiff(N) : 
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
            
    return K

#matrix of mass
def mass(N) :
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

            m_J_B = jB * dfi_B * dfj_B
            m_J_Dv = jDv * dfi_Dv * dfj_Dv
            
            M[i][j] = m_frame + m_M_B + m_M_Dv + m_J_B + m_J_Dv
    
    return M

#computing n natural fréquencies with Rayleigh Ritz
def n_w(n, nb_approx):
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
    w6 = np.zeros(13)
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
        
        if len(w) > 5 :
            w6[n-1] = w[5]
    
    x = np.arange(1, 14, 1)
    
    
    fig,ax = plt.subplots()
    plt.grid(True,linestyle='--',color='0.80')
    plt.plot(x, w1, color = 'red', label = 'f1')
    plt.plot(x[1:], w2[1:], color = 'orange', label = 'f2')
    plt.plot(x[2:], w3[2:], color = 'yellow', label = 'f3')
    plt.plot(x[3:], w4[3:], color = 'limegreen', label = 'f4')
    plt.plot(x[4:], w5[4:], color = 'blue', label = 'f5')
    plt.plot(x[5:], w6[5:], color = 'darkmagenta', label = 'f6')
    plt.legend(loc = 'upper right', ncol = 2)
    plt.ylim(-100, 10000)
    plt.xlabel('Ordre de convergence')
    plt.ylabel('Fréquences propres [Hz]')
    plt.savefig('1-convergence.pdf')
    plt.show()
        
#computing relative errors on frequencies with n = 10 (good convergence)     
def relative_errors(nb_approx):
    w_anal, _ = n_w(4,nb_approx)
    print(w_anal)
    w_num = np.loadtxt('P2024_f_Part4.txt')
    errors = np.zeros(4)
    for i in range(4):
        errors[i] = (np.abs(w_anal[i] - w_num[i])/w_num[i])*100
    
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
    f = 0
    for i in range(n):
        f += (x / L)**i * mode[i]
    return f
        
#computing theoritical and numerical modes with m being the number of the mode (beginning with 0)
def mode_shapes(m, nb_approx):
    _, modes_anal = n_w(4,nb_approx)
    
    modes_num = np.loadtxt('P2024_Modes_Part4.txt')*0.001
    
    x = np.linspace(0, 1.2, 1000)

    x_point = [0, 100, 200, 300, 400, 400, 500, 600, 700, 800, 901, 1000, 1101, 1200]
    x_point = [x / 1000 for x in x_point] # Convertir en mètres
    
    mode_anal = modes_anal[:, m]
    mode_num = modes_num[:, m]
    
    #normalization because modes are defined up to a constant so to have the same ones : putting q0 at 1
    mode_anal /= mode_anal[0]
    mode_num /= mode_num[0]
    
    
    f = y(x, mode_anal, 4)
    
    fig,ax = plt.subplots()
    plt.grid(True,linestyle='--',color='0.80')
    plt.scatter(x_point, mode_num, color = 'orange', label = 'Mode expérimental')
    plt.plot(x, f, color = 'blue', label = 'Mode théorique')
    plt.legend()
    plt.xlabel('x [m]')
    plt.ylabel('Délfection z(x) [m]')
    plt.savefig('7-mode6.pdf')
    plt.show()
    
 
#computing the MAC matrice with n being the number of modes
def MAC(n, nb_approx):
    M = np.zeros([n,n])
    
    _, modes_anal = n_w(4,nb_approx)
    
    modes_num = np.loadtxt('P2024_Modes_Part4.txt')*0.001
    
    x_point = [0, 100, 200, 300, 400, 400, 500, 600, 700, 800, 901, 1000, 1101, 1200]
    x = [x_calc / 1000 for x_calc in x_point] # Convertir en mètres

    #i = experimental modes
    for i in range(n):
        shape_num = modes_num[:,i]
        shape_num /= shape_num[0]
        if i ==0:
            print("data:", shape_num)
        
        #mode into matrix
        vec_num = np.zeros([len(shape_num), 1])
        for k in range(len(shape_num)):
            vec_num[k][0] = shape_num[k]
        
        vec_numT = vec_num.T
        
        #j = theoritical modes
        for j in range(n) :
            mode_anal = modes_anal[:, j]
            mode_anal /= mode_anal[0]

            shape_anal = y(x, mode_anal, 4)
            if j==0:
                print("anal :", shape_anal)
        
        
            #mode into matrix
            vec_anal = np.zeros([len(shape_anal), 1])
            for k in range(len(shape_anal)):
                vec_anal[k][0] = shape_anal[k]
        
            vec_analT = vec_anal.T
    
            num = np.abs(np.dot(vec_numT, vec_anal))
            num = num**2
    
            denom1 = np.dot(vec_numT, vec_num)
            denom2 = np.dot(vec_analT, vec_anal)
    
            macij = num[0][0] / (denom1[0][0] * denom2[0][0])
    
            M[3-i][j] = macij
    
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.imshow(M, cmap='Blues', interpolation='nearest', vmin=0, vmax=1)
    plt.colorbar(label='Valeurs matrice MAC')
    
    # Ajouter des annotations manuellement
    for i in range(n):
        for j in range(n):
            plt.text(j, i, f'{M[i, j]:.2f}', ha='center', va='center', color='black')
    
    plt.xlabel("Fréquences théoriques [Hz]")
    plt.ylabel("Fréquences expérimentales [Hz]")
    
    xlabels = ['38.02', '271.0', '729.2', '1493.61']
    plt.xticks(ticks=np.arange(n), labels=xlabels, rotation='horizontal')
    
    ylabels = ['451.46', '243.85', '66.98', '3.71']
    plt.yticks(ticks=np.arange(n), labels=ylabels, rotation='horizontal')
    
    plt.savefig('8-MAC.pdf')
    plt.show()

    return M

nb_approx_test = 14


#computing n natural fréquencies with Rayleigh Ritz
w_test, modes_test = n_w(4,nb_approx_test)
print("Fréquences propres (Hz) :", w_test)
print("Modes propres :", modes_test)
errors = relative_errors(nb_approx_test)
print("Erreurs relatives (%) :", errors)

for i in range(4):
    mode_shapes(i, nb_approx_test)

MAC(4, nb_approx_test) 


"""
# Appeler les fonctions pour voir les résultats
convergence()
errors = relative_errors()
print("Erreurs relatives (%) :", errors)
mode_shapes(0)  # Exemple pour le premier mode (m=0)
MAC(6)  # Exemple pour les 6 premiers modes
"""




