from scipy.special import eval_sh_legendre
import numpy as np
import torch


def check_positivity(kernel_func, degree=3, domain=(0, 1), n_quad=40):
    
    # 1. Récupérer les points et poids de Gauss-Legendre sur [-1, 1]
    x_g, w_g = np.polynomial.legendre.leggauss(n_quad)
    
    # 2. Ajuster les points et poids au domaine [0, 1]
    a, b = domain
    x_shifted = 0.5 * (b - a) * x_g + 0.5 * (b + a)
    w_shifted = 0.5 * (b - a) * w_g
    
    # 3. Évaluation VECTORISÉE du noyau sur la grille (Appel unique à construire_F !)
    X1, X2 = np.meshgrid(x_shifted, x_shifted, indexing="ij")
    P_mat = kernel_func(X1, X2) 
    
    # 4. Évaluer les polynômes de Legendre décalés sur la base de test
    # Shape: (degree + 1, n_quad)
    n = degree + 1
    Phi = np.array([eval_sh_legendre(i, x_shifted) for i in range(n)])
    
    # 5. Calcul matriciel de Gram : G = Phi * W * P * W * Phi^T
    W = np.diag(w_shifted)
    G = Phi @ W @ P_mat @ W @ Phi.T
    
    # 6. Calcul des valeurs propres
    eigenvalues = np.linalg.eigvalsh(G)
    
    return eigenvalues
        
    # tol = 1e-10
    # mini = eigenvalues.min()
    # if np.all(eigenvalues > tol):
    #     print("mini = %.2e DÉFINI POSITIF. ✅" % mini)
    # else:
    #     print("mini = %.2e  PAS défini positif. ❌"  % mini)

