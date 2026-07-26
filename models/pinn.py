import torch.nn as nn
import torch

from models.base import MLP


class PINN_P_BC(nn.Module):
    """
    Réseau de neurones pour la solution P(x1, x2, k).
    
    Le réseau prend en entrée (x1, x2, k) et impose les conditions aux limites
    via une construction spécifique.
    """
    
    def __init__(self, hidden_dim=128, n_layers=4):
        super().__init__()
        self.net = MLP(in_dim=3, out_dim=1, hidden_dim=hidden_dim, n_layers=n_layers)
    
    def forward(self, x1, x2, k):
        """
        Calcul de P(x1, x2, k) avec conditions aux limites.
        
        Args:
            x1, x2: Coordonnées spatiales
            k: Paramètre d'entrée (0 ≤ k ≤ 0.7)
        
        Returns:
            P: Solution approchée
        """
        
        # Symétrisation : on travaille sur le triangle x1 >= x2
        a = torch.maximum(x1, x2).unsqueeze(-1)
        b = torch.minimum(x1, x2).unsqueeze(-1)
        k = k.unsqueeze(-1)
        
        # Calcul sur le triangle avec conditions aux limites
        N_ab = self.net(torch.cat([a, b, k], dim=-1))
        N_b0 = self.net(torch.cat([b, torch.zeros_like(b), k], dim=-1))
        N_00 = self.net(torch.cat([torch.zeros_like(a), torch.zeros_like(a), k], dim=-1))
        
        # Construction de P respectant les BC : P(1, x2) = k * P(x2, 0)
        P = (N_ab * (1.0 - a) + 
             k * a * (1.0 - b) * N_b0 + 
             k**2 * a * b * N_00)
        return P.squeeze(-1)