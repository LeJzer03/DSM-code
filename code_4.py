import os
import numpy as np
import scipy.integrate
from scipy.linalg import eigh
import matplotlib.pyplot as plt


# Set the working directory to the script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)


# Chargement des données des fréquences naturelles et des modes observés
Freq = np.loadtxt("P2024_f_Part4.txt", dtype="float")  # Fréquences pour les modes de tangage
Modes = np.loadtxt("P2024_Modes_Part4.txt", dtype="float")*1e-3  # Modes en mètres


# Paramètres du cadre de la moto
n = 8  # Nombre de fonctions d'approximation à utiliser
l = 1.2  # Longueur du cadre [m]
rho = 7850  # Densité du matériau [kg/m^3]
A = 0.0225  # Section transversale [m^2] = 0.15*0.15  #pq marche pas avec 0.15*0.15 - (0.15-0.01)**2
E = 2.1e11  # Module d'Young [Pa]
I = 1.83866 * 10**-5  # Moment quadratique [m^4] , calcule avant au devoir 1

"""
n = 6  # Nombre de fonctions d'approximation à utiliser
l = 1.2  # Longueur du cadre [m]
rho = 7850  # Densité du matériau [kg/m^3]
A = 2.9*10**-3 # Section transversale [m^2] = 0.15*0.15 - (0.15-0.01)**2
E = 2.1e11  # Module d'Young [Pa]
I = 1.83866 * 10**-5  # Moment quadratique [m^4]
"""

# Informations sur les masses concentrées
mB = 75.02  # Masse à la position B [kg]
xB = 0.4  # Position de la masse B sur x [m]
jB = 1  # Moment d'inertie de la masse B [kg*m^2]
lB = 0.25  # Position de la masse B sur y [m]

mDv = 75  # Masse à la position D (driver) [kg]
xDv = 1  # Position de la masse D sur x [m]
jDv = 10  # Moment d'inertie de la masse D [kg*m^2]

# Paramètres des ressorts
k1l = 10**4  # [N/m]
k1r = 10**9  # [Nm/rad]
x1 = 0 # Position des ressort 1 sur x [m]
k2l = 10**5  # [N/m]
k2r = 10**5  # [Nm/rad]
x2 = 0.8  # Position des ressort 2 sur x [m]

# Points de référence
x_point = [0, 100, 200, 300, 400, 400, 500, 600, 700, 800, 901, 1000, 1101, 1200]
x_point = [x / 1000 for x in x_point] # Convertir en mètres


# Calcule Matrice M et K #
##########################

def f(n):
    """
    Parameters
    ----------
    n : int
        Nombre de fonction d'aproximation utilisée.

    Returns
    -------
    tuple, float
        Return la fréquence de chaque mode en Hz et les modes non normalisé.

    """
    
    
    M = np.zeros((n,n))
    K = np.zeros((n,n))
    
    def w(x,i):
        if i == 0:
            return 1/(l**i)
        else:    
            return (x/l)**i
        
        
    def dw(x,i):
        if (i-1) == 0:
            return i/(l**i)
        elif x == 0 and (i-1) < 0:
            return 0
        else:
            return (i*(x**(i-1)))/(l**i)
        
        
    def d2w(x,i):
        if (i-2) == 0:
            return ((i-1)*i)/(l**i)
        elif x == 0 and (i-2)<0:
            return 0
        else:
            return ((i-1)*i*(x**(i-2)))/(l**i)
        
        
    
    for i in range(len(M)):
        for j in range(len(M)):
            
            """
            mDv = 75  # Masse à la position D (driver) [kg]
            xDv = 1  # Position de la masse D sur x [m]
            jDv = 10  # Moment d'inertie de la masse D [kg*m^2]

            mB = 75.02  # Masse à la position B [kg]
            xB = 0.4  # Position de la masse B sur x [m]
            jB = 1  # Moment d'inertie de la masse B [kg*m^2]
            lB = 0.25  # Position de la masse B sur y [m]
            """


            continu_M = rho*A*(scipy.integrate.quad(lambda x: w(x, i)*w(x,j),0, l)[0])
            ponctuel_Dv = jDv * (dw(xDv,i)*dw(xDv,j)) + mDv*w(xDv,i)*w(xDv,j)
            ponctuel_B = (mB*lB**2+jB)*(dw(xB,i)*dw(xB,j)) + mB * w(xB,i)*w(xB,j)

           
            M[i,j] = continu_M + ponctuel_Dv + ponctuel_B

            """
            # Paramètres des ressorts
            k1l = 1e5  # [N/m]
            k1r = 1e9  # [Nm/rad]
            x1 = 0 # Position des ressort 1 sur x [m]
            k2l = 1e6  # [N/m]
            k2r = 1e5  # [Nm/rad]
            x2 = 0.8  # Position des ressort 2 sur x [m]
            
            """
            
            continu_K = E*I*(scipy.integrate.quad(lambda x: ((d2w(x,i)*d2w(x,j))),0, l)[0])
            ponctuel_k1l = k1l*w(x1,i)*w(x1,j)
            ponctuel_k1r = k1r*dw(x1,i)*dw(x1,j)
            ponctuel_k2l = k2l*w(x2,i)*w(x2,j)
            ponctuel_k2r = k2r*dw(x2,i)*dw(x2,j)

            
            K[i,j] = continu_K + ponctuel_k1l + ponctuel_k1r + ponctuel_k2l + ponctuel_k2r
                
    #M = M/((rho*A*l)+mF+mG+mH+mI)
    print("M : ", M)
    print("K : ", K)
    

    omega_squared, vect_pro = eigh(K,M)
    
    omega = np.sqrt(omega_squared)
    freq = omega /(2 * np.pi)
    
    return freq, vect_pro


freq, vec = f(n)
print(freq)

# Normalisation des modes #
for i in range(len(vec)):
    vec[:, i] = vec[:, i]/vec[0,i]
    
for i in range(len(Modes[0,:])):
    Modes[:,i] = Modes[:,i]/Modes[0,i]



############################################ CONVERGENCE ############################################
# Range of approximation functions
n_values = range(1, 14)  # Calculer jusqu'à n = 13
all_frequencies = []

# Run the frequency calculation for different values of n
for n_calcul in n_values:
    frequencies, _ = f(n_calcul)
    all_frequencies.append(frequencies)

# Plot the convergence of frequencies for each mode
plt.figure(figsize=(10, 6))
for mode in range(6):  # Afficher seulement les 6 premiers modes
    freqs = [freq[mode] if mode < len(freq) else None for freq in all_frequencies]
    plt.plot(n_values, freqs, label=f'Mode {mode + 1}')

plt.xlabel('Ordre de convergence')
plt.ylabel('Fréquence propre (Hz)')
plt.title('Convergence des fréquences propres en fonction de l\'ordre de convergence')
plt.legend()
plt.grid(True)
plt.show()




############################################ RELATIVE ERRORS ############################################
# Calcul des erreurs relatives
def generate_latex_tables(n, nb_modes):
    # Calculer les fréquences analytiques avec n = 
    w_anal, _ = f(n)
    
    # Charger les fréquences numériques à partir du fichier
    w_num = np.loadtxt('P2024_f_Part4.txt', dtype="float")
    
    # Initialiser un tableau pour les erreurs
    relative_errors = []
    
    # Calculer les erreurs relatives pour les nb_modes premiers modes
    for i in range(nb_modes):
        error = (np.abs(w_anal[i] - w_num[i]) / w_num[i]) * 100
        relative_errors.append(error)
    
    # Générer le tableau LaTeX pour les fréquences
    latex_table = "\\begin{table}[h!] \n"
    latex_table += "   \\centering \n"
    latex_table += "   \\begin{tabular}{|" + "c|" * (nb_modes + 1) + "} \n"
    latex_table += "       \\hline \n"
    latex_table += "       Mode & " + " & ".join([f"$f_{i+1}$ (Hz)" for i in range(nb_modes)]) + " \\\\ \\hline \n"
    latex_table += "       Théorique & " + " & ".join([f"{w_anal[i]:.3f}" for i in range(nb_modes)]) + " \\\\ \\hline \n"
    latex_table += "       Numérique & " + " & ".join([f"{w_num[i]:.3f}" for i in range(nb_modes)]) + " \\\\ \\hline \n"
    latex_table += "   \\end{tabular} \n"
    latex_table += "   \\caption{Comparaison des fréquences propres théoriques et numériques.} \n"
    latex_table += "   \\label{tab:frequences} \n"
    latex_table += "\\end{table}"
    
    # Générer le tableau LaTeX pour les erreurs relatives
    latex_table_err = "\\begin{table}[h!] \n"
    latex_table_err += "   \\centering \n"
    latex_table_err += "   \\begin{tabular}{|" + "c|" * (nb_modes + 1) + "} \n"
    latex_table_err += "       \\hline \n"
    latex_table_err += "       Mode & " + " & ".join([f"$\\varepsilon_{i+1}$ (\%)" for i in range(nb_modes)]) + " \\\\ \\hline \n"
    latex_table_err += "       Erreur & " + " & ".join([f"{relative_errors[i]:.2f}" for i in range(nb_modes)]) + " \\\\ \\hline \n"
    latex_table_err += "   \\end{tabular} \n"
    latex_table_err += "   \\caption{Erreurs relatives des fréquences propres.} \n"
    latex_table_err += "   \\label{tab:erreurs} \n"
    latex_table_err += "\\end{table}"
    
    print(latex_table)
    print("")
    print(latex_table_err)


# Appeler la fonction pour générer et afficher les tableaux LaTeX
generate_latex_tables(n,nb_modes=4)




############################################

# Fonctions des Modes #
#######################

def y(x, n):
    """
    Parameters
    ----------
    x : flot
        Position sur le bras.
    n : int
        Nombre de fonction d'approximation utilisée.

    Returns
    -------
    sortie : array
        Return un vecteur posédant les valeurs de y(x) pour chacun des modes.

    """
    
    sortie = np.zeros(n)
    for i in range(n):
        for j in range(n):
            sortie[i] += vec[j][i] * (x/l)**j 

    return sortie



##########################################

# Plot des modes #
##################

def mode(l):
    """
    Parameters
    ----------
    l : int
        numéro du mode (1,2,3...).

    Returns
    -------
    Affiche et sauvegarde sous forme de pdf les plots des modes

    """
    
    x= np.linspace(0, 1.2, 1000)
    
    y_x = np.zeros(len(x))
    for i in range(len(y_x)):
        y_x[i] = y(x[i], n)[l-1]
    

    
    plt.figure(figsize=(10,6))
    
    plt.plot(x, y_x, label="Mode théorique")
    plt.plot(x_point, Modes[:,l-1], 'ro', label="Mode expérimental")
    
    
    plt.xlabel('Position [m]', fontsize=15)
    plt.ylabel('Déflexion [-]', fontsize=15)
    
    plt.grid(True,linestyle='--',color='0.80')
    plt.legend()
    
    plt.savefig(f"mode{l}.pdf", dpi=300, bbox_inches="tight")
    plt.show()

for i in range(4):
    mode(i+1)

#######################################

# Matrice MAC #
###############

def mode_anal():
    """
    Returns
    -------
    modes_anal : array
        Return la valeur des modes théorique en chacuns des accéléromètres.

    """
    modes_anal = np.zeros(((len(x_point)), len(Modes[0][:])))
    for i in range(len(x_point)):
        for j in range(len(Modes[0][:])):
            modes_anal[i][j] = y(x_point[i], n)[j]
    return modes_anal
    


def MACij(phi_i, phi_j):
    """
    Parameters
    ----------
    phi_i : array
        valeur de Psi_x en i.
    phi_j : array
        valeur de Psi_a en j.

    Returns
    -------
    mac : float
        valeur i j de la matrice MAC.

    """
    
    numerator = np.abs(np.dot(phi_i.T, phi_j)) ** 2
    denominator = np.dot(phi_i.T, phi_i) * np.dot(phi_j.T, phi_j)
    mac = numerator / denominator
    return mac


def MAC_matrice():
    """

    Returns
    -------
    matrice_MAC : array
        Return la matrice MAC complète.

    """
    matrice_modes_th = mode_anal()
    matrice_MAC = np.zeros((4,4))
    for i in range (0,4):
        data_mode_i = Modes[:, i]
        if i == 0:
            print("data : ", data_mode_i)
        for j in range (0,4):
            calculs_mode_j = matrice_modes_th[:,j]
            if j==0 :
                print("anal : ", calculs_mode_j)
            matrice_MAC[i,j] = MACij(data_mode_i, calculs_mode_j)
    return matrice_MAC


def plot_matrix_with_colorbar(matrix):
    """
    Parameters
    ----------
    matrix : array
        La matrice MAC.

    Returns
    -------
    Affiche et sauvegarde sous forme de pdf le plot de la matrice MAC

    """
    # Convertir la matrice en un tableau NumPy pour utiliser imshow
    matrix_array = np.array(matrix)
    
    plt.figure(figsize=(10,8))
    # Créer un graPsique avec imshow
    plt.imshow(matrix_array, cmap='Blues', interpolation='nearest')

    # Ajouter une colorbar pour représenter les valeurs
    cbar = plt.colorbar()
    cbar.set_label('Valeurs de la matrice MAC')

    # Afficher les valeurs de la matrice dans chaque case
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            plt.text(j, i, f"{matrix[i][j]:.2f}", ha='center', va='center', color='black')

    #Ajuster les ticks des axes pour commencer à 1
    plt.xticks(np.arange(len(matrix[0])))
    plt.yticks(np.arange(len(matrix)))

    # Remplacer les étiquettes des axes par des valeurs commençant à 1
    plt.xticks(np.arange(len(matrix[0])) + 0.5, np.arange(1, len(matrix[0]) + 1, 1))
    plt.yticks(np.arange(len(matrix)) + 0.5, np.arange(1, len(matrix) + 1, 1))

    # Inverser l'axe y pour commencer par le bas
    plt.gca().invert_yaxis()
    
    # Afficher le graPsique
    plt.xlabel('Modes analytiques [-]', fontsize=15)
    plt.ylabel('Modes expérimantaux [-]', fontsize=15)
    
    plt.savefig("MAC_matrice.pdf", dpi=300, bbox_inches="tight")
    plt.show()

plot_matrix_with_colorbar(MAC_matrice())


