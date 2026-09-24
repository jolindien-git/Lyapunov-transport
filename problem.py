import torch

from positivity.polynomial_test import check_positivity


def grad(outputs, inputs):
    return torch.autograd.grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True,
        retain_graph=True,
    )[0]


def construire_F_torch(lam, mu, sigma, k, device=None, dtype=torch.float32):
    """
    Construit la fonction F(u) pour la solution exacte.
    
    Args:
        lam, mu, sigma: Paramètres physiques
        k: Paramètre d'entrée
        device: Périphérique de calcul
        dtype: Type de données
    
    Returns:
        F: Fonction F(u)
        alpha: Paramètre alpha
        Q: Fonction Q(z)
    """
    alpha = torch.tensor(mu / (2.0 * lam), device=device, dtype=dtype)
    exp_alpha = torch.exp(alpha)
    exp_2alpha = torch.exp(2.0 * alpha)
    
    denom = 1.0 - (k**2) * exp_2alpha
    if torch.isclose(denom, torch.tensor(0.0, device=device, dtype=dtype)):
        raise ValueError("Le dénominateur de F est nul ou presque nul.")
    
    def Q(z):
        z = torch.as_tensor(z, device=device, dtype=dtype)
        return torch.exp(-(z**2) / (2.0 * sigma**2))
    
    def S(x):
        x = torch.as_tensor(x, device=device, dtype=dtype)
        terme1 = Q(1.0 - x) * (torch.exp(alpha * (1.0 + x)) - 1.0)
        terme2 = k * Q(x) * exp_alpha * (torch.exp(alpha * x) - 1.0)
        return (terme1 - terme2) / mu
    
    def F(u):
        u = torch.as_tensor(u, device=device, dtype=dtype)
        return (k * exp_alpha * S(u) + S(1.0 - u)) / denom
    
    return F, alpha, Q


def P_exact_torch(x1, x2, k, lam, mu, sigma, device):
    """
    Calcule la solution exacte P(x1, x2, k).
    
    Args:
        x1, x2: Coordonnées spatiales
        k: Paramètre d'entrée
        lam, mu, sigma: Paramètres physiques
        device: Périphérique de calcul
    
    Returns:
        Valeur exacte de P
    """
    x1 = torch.as_tensor(x1, device=device, dtype=torch.float32)
    x2 = torch.as_tensor(x2, device=device, dtype=torch.float32)
    u = x1 - x2
    v = x1 + x2
    
    F_t, alpha_t, Q_t = construire_F_torch(lam, mu, sigma, k, device=device)
    return F_t(u) * torch.exp(-alpha_t * v) - (Q_t(u) / mu) * (1.0 - torch.exp(-alpha_t * v))



class Problem:
    
    def __init__(self, sigma: float, lambd: float, mu: float, k: float):
        self.sigma = sigma
        self.lambd = lambd
        self.mu = mu
        self.k = k
        
    def get_Q(self, x1, x2):
        return torch.exp(-((x1 - x2) ** 2) / (2.0 * self.sigma**2))
    
    def residual_pde(self, model, x1, x2):
        """
        Calcule le résidu de l'EDP pour les points donnés.
        
        Résidu: L(P) = LAM * (P_x1 + P_x2) + MU * P + Q(x1-x2)
        """
        x1 = x1.clone().detach().requires_grad_(True)
        x2 = x2.clone().detach().requires_grad_(True)
        ks = self.k * torch.ones_like(x1)
        
        P = model(x1, x2, ks)
        dP_dx1 = grad(P, x1)
        dP_dx2 = grad(P, x2)
        
        Q = self.get_Q(x1, x2)
        res = self.lambd * (dP_dx1 + dP_dx2) + self.mu * P + Q
        return res
    
    @staticmethod
    def loss_pde(res):
        return torch.mean(res**2)

    def sample_square(self, n: int, device: str):
        x1 = torch.rand(n, 1, device=device)
        x2 = torch.rand(n, 1, device=device)
        return x1, x2
    
    def P_exact(self, x1, x2, device):
        x1 = torch.as_tensor(x1, device=device, dtype=torch.float32)
        x2 = torch.as_tensor(x2, device=device, dtype=torch.float32)
        return torch.where(
            x1 >= x2, 
            P_exact_torch(x1, x2, self.k, self.lambd, self.mu, self.sigma, device),
            P_exact_torch(x2, x1, self.k, self.lambd, self.mu, self.sigma, device)
        )
    
    def check_positivity_Q(self, model, device, degree=9):
        
        def Q_func(x1, x2):
            x1 = torch.as_tensor(x1, device=device, dtype=torch.float32)
            x2 = torch.as_tensor(x2, device=device, dtype=torch.float32)
            Q = self.get_Q(x1, x2)
            residual = self.residual_pde(model, x1, x2)
            Q_estim =  Q - residual
            return Q_estim.detach().cpu().numpy()
    
        eigenvalues = check_positivity(Q_func, degree=degree, domain=(0, 1))
        return eigenvalues
    
    def check_positivity_P(self, model, device, degree=9):
        
        @torch.no_grad
        def P_func(x1, x2):
            x1 = torch.as_tensor(x1, device=device, dtype=torch.float32)
            x2 = torch.as_tensor(x2, device=device, dtype=torch.float32)
            k = self.k * torch.ones_like(x1)
            P_mat = model(x1, x2, k)
            return P_mat.cpu().numpy()

        eigenvalues = check_positivity(P_func, degree=degree, domain=(0, 1))
        return eigenvalues