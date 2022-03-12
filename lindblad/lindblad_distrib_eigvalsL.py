''' This program plots the distributions of Lindbladian eigenvalues '''
import numpy as np
from matplotlib import colors
import matplotlib.pyplot as plt
from lindblad import *

iterations = 2000
fixed_value = 'g'
path =  "C:\\Users\\cerra\\Desktop\Thesis_data_plot\Dati"

# Plot the distribution of Lindbladian eigenvalues in the case of only dissipator for N = 2
if False:
    N = 2
    k, alpha, gamma = 0, 0, 1

    # Write the data in two lists: real part and imag part of Lindbladian eigenvalues
    real_eigval_L_txt = f"{path}\\{fixed_value}_fissato\\"\
    f"Real_eigvals_L_per_N={N}_k={k}_a={alpha}_g={np.round(gamma,3)}_{iterations}_iterazioni.txt"
    imag_eigval_L_txt = f"{path}\\{fixed_value}_fissato\\"\
    f"Imag_eigvals_L_per_N={N}_k={k}_a={alpha}_g={np.round(gamma,3)}_{iterations}_iterazioni.txt"

    f_real_eigval_L = open(real_eigval_L_txt, 'r')
    real_eigval_L = f_real_eigval_L.read()
    real_eigval_L_list = list(np.fromstring(real_eigval_L, sep = '\n'))
    f_real_eigval_L.close()

    f_imag_eigval_L = open(imag_eigval_L_txt, 'r')
    imag_eigval_L = f_imag_eigval_L.read()
    imag_eigval_L_list = list(np.fromstring(imag_eigval_L, sep = '\n'))
    f_imag_eigval_L.close()

    # Plot the distribution of Lindbladian eigenvalues
    fig, ax = plt.subplots(figsize=(15,10))
    ax.scatter(real_eigval_L_list, imag_eigval_L_list, \
        label = fr'k = {k}, $\alpha$ = {alpha}, $\gamma$ = {gamma}')
    ax.set_ylim(-1,1)
    ax.set_title(fr'Distribution of Lindbladian $\mathcal{{L}}_{{D}}$ eigenvalues for N = {N}',\
        fontsize=20)
    ax.set_xlabel(fr'Real($\lambda_{{D}}$)', fontsize=18)
    ax.set_ylabel(fr'Imag($\lambda_{{D}}$)', fontsize=18)
    ax.legend(fontsize=16)
    
    plt.show()

# Plot the distribution of Lindbladian eigenvalues in the case of only dissipator for N = 3
if False:
    N=3
    k, alpha, gamma = 0, 0, 1

    # Write the data in two lists: real part and imag part of Lindbladian eigenvalues
    real_eigval_L_txt = f"{path}\\{fixed_value}_fissato\\"\
    f"Real_eigvals_L_per_N={N}_k={k}_a={alpha}_g={np.round(gamma,3)}_{iterations}_iterazioni.txt"
    imag_eigval_L_txt = f"{path}\\{fixed_value}_fissato\\"\
    f"Imag_eigvals_L_per_N={N}_k={k}_a={alpha}_g={np.round(gamma,3)}_{iterations}_iterazioni.txt"

    f_real_eigval_L = open(real_eigval_L_txt, 'r')
    real_eigval_L = f_real_eigval_L.read()
    real_eigval_L_list = list(np.fromstring(real_eigval_L, sep = '\n'))
    f_real_eigval_L.close()

    f_imag_eigval_L = open(imag_eigval_L_txt, 'r')
    imag_eigval_L = f_imag_eigval_L.read()
    imag_eigval_L_list = list(np.fromstring(imag_eigval_L, sep = '\n'))
    f_imag_eigval_L.close()

    # Plot the distribution of Lindbladian eigenvalues
    plt.hist2d(real_eigval_L_list, imag_eigval_L_list, bins=(55, 55), cmap = plt.cm.jet,\
        norm = colors.LogNorm(), label = fr'k = {k}, $\alpha$ = {alpha}, $\gamma$ = {gamma}')
    plt.colorbar()
    plt.title(fr'Distribution of Lindbladian $\mathcal{{L}}_{{D}}$ eigenvalues for N = {N}', fontsize=20)
    plt.xlabel(fr'Real($\lambda_{{D}}$)', fontsize=18)
    plt.ylabel(fr'Imag($\lambda_{{D}}$)', fontsize=18)
    plt.legend()
    
    plt.show()

# Plot the distribution of Lindbladian eigenvalues in the general case for N = 2; 
# Returns subplots for different values of k.
if False:
    N = 2
    k_list = [0.1,0.5,0.7,1]
    fixed_value = 'g'
    alpha, gamma = 1, 1
    fig1, ax1 = plt.subplots(nrows = int(len(k_list)/2), ncols = int(len(k_list)/2),\
        sharex = True, sharey = True, figsize=(15,10))
    fig1.suptitle(fr'Distribution of Lindbladian $\mathcal{{L}}$ eigenvalues for N = {N}', fontsize = 18)
    for k in k_list:
        if fixed_value == 'a':
            alpha = 1
            gamma = alpha/k
        if fixed_value == 'g':
            gamma = 1
            alpha = k*gamma

        # Write the data in two lists: real part and imag part of Lindbladian eigenvalues
        real_eigval_L_txt = f"{path}\\{fixed_value}_fissato\\"\
        f"Real_eigvals_L_per_N={N}_k={k}_a={alpha}_g={np.round(gamma,3)}_{iterations}_iterazioni.txt"
        imag_eigval_L_txt = f"{path}\\{fixed_value}_fissato\\"\
        f"Imag_eigvals_L_per_N={N}_k={k}_a={alpha}_g={np.round(gamma,3)}_{iterations}_iterazioni.txt"

        f_real_eigval_L = open(real_eigval_L_txt, 'r')
        real_eigval_L = f_real_eigval_L.read()
        real_eigval_L_list = list(np.fromstring(real_eigval_L, sep = '\n'))
        f_real_eigval_L.close()

        f_imag_eigval_L = open(imag_eigval_L_txt, 'r')
        imag_eigval_L = f_imag_eigval_L.read()
        imag_eigval_L_list = list(np.fromstring(imag_eigval_L, sep = '\n'))
        f_imag_eigval_L.close()
        
        # Plot the distribution of Lindbladian eigenvalues
        ax1[k_list.index(k)//2,k_list.index(k)%2].scatter(real_eigval_L_list, imag_eigval_L_list,\
            label = fr'k = {k}, $\alpha$ = {alpha}, $\gamma$ = {gamma}')
        ax1[k_list.index(k)//2,k_list.index(k)%2].set_xlabel(fr'$Real(\lambda)$',fontsize=16)
        ax1[k_list.index(k)//2,k_list.index(k)%2].set_ylabel(fr'$Imag(\lambda)$',fontsize=16)
        ax1[k_list.index(k)//2,k_list.index(k)%2].legend(fontsize = 12)

        for ax in ax1.flat:
            ax.label_outer()

    plt.show()

