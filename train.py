import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

c = 20.0
lam = 2.0
K_MIN, K_MAX = .94, .97 # 1.2, 2.0 

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class PINN(nn.Module):
    def __init__(self, hidden_dim=128):
        super(PINN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x, k):
        inputs = torch.cat([x, k], dim=-1)
        return self.net(inputs)

def get_p(model, x, k):
    """
    Applique la reformulation (ansatz) pour imposer p(1) = k^2 p(0) par construction.
    """
    N_x = model(x, k)
    zeros = torch.zeros_like(x)
    N_0 = model(zeros, k)
    p = (1 - x) * N_x + x * (k**2) * N_0
    return p


def loss_phys(model, x, k):
    """
    Calcule le résidu de l'EDO : c*p'(x) + lam*p(x) + 1 = 0
    """
    p = get_p(model, x, k)
    
    dp_dx = torch.autograd.grad(
        outputs=p, 
        inputs=x, 
        grad_outputs=torch.ones_like(p), 
        create_graph=True
    )[0]
    
    res = c * dp_dx + lam * p + 1.0
    return torch.mean(res**2)


def exact_solution(x, k):
    '''
    !!! SOLUTION QUI CORRESPOND (VISUELLEMENT) A LA FIGURE 6 DU DRAFT
    !!! NE CORRESPOND PAS A p_exact DU DRAFT
    '''
    num = 1- k**2
    den = lam * (np.exp(-lam / c) - k**2)
    return (num / den) * torch.exp(-lam / c * x) - 1 / lam

    # num = k**2 - 1
    # den = lam * (np.exp(lam/c) - k**2)
    # return (num / den) * np.exp((lam/c) * x) + (1/lam)


# %% Boucle d'entraînement
epochs = 5000*2
N_f = 2000

model = PINN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2/2)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=epochs//5, gamma=0.5)

print("Début de l'entraînement...")
loss_history = []
for epoch in range(epochs):
    model.train()
    
    # Échantillonnage aléatoire uniforme dans [0, 1]x[0, 2] à chaque époque
    x_train = torch.rand(N_f, 1, requires_grad=True, device=device)
    k_train = K_MIN + torch.rand(N_f, 1, device=device) * (K_MAX - K_MIN)
    
    optimizer.zero_grad()
    loss = loss_phys(model, x_train, k_train)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
    optimizer.step()
    scheduler.step()
    
    loss_history.append(loss.item())
    
    if (epoch + 1) % 500 == 0:
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {loss.item():.4e} - LR: {scheduler.get_last_lr()[0]:.4e}")

print("Entraînement terminé !")

# # %% Phase de Raffinement avec L-BFGS
# print("\nDébut du raffinement L-BFGS...")

# # Configuration de L-BFGS avec des paramètres standards pour les PINNs
# optimizer_lbfgs = torch.optim.LBFGS(
#     model.parameters(),
#     lr=1.0,
#     max_iter=1000,        # Nombre max d'itérations par appel à step()
#     max_eval=1250,        # Nombre max d'évaluations de la fonction
#     tolerance_grad=1e-7,
#     tolerance_change=1e-9,
#     history_size=50,
#     line_search_fn="strong_wolfe" # Essentiel pour la stabilité avec les PINNs
# )

# lbfgs_epochs = 100

# model.train()
# x_train_lbfgs = torch.rand(N_f, 1, requires_grad=True, device=device)
# k_train_lbfgs = K_MIN + torch.rand(N_f, 1, device=device) * (K_MAX - K_MIN)
# loss_history_lbfgs = []
# for epoch in range(lbfgs_epochs):
#     def closure():
#         optimizer_lbfgs.zero_grad()
#         loss = loss_phys(model, x_train_lbfgs, k_train_lbfgs)
#         loss.backward()
#         return loss
    
#     # Exécution de l'optimiseur
#     optimizer_lbfgs.step(closure)
    
#     # Évaluation de la loss finale de cette étape
#     current_loss = closure().item()
#     loss_history_lbfgs.append(current_loss)
#     print(f"L-BFGS Étape [{epoch+1}/{lbfgs_epochs}] - Loss: {current_loss:.4e}")

# print("Raffinement L-BFGS terminé !")

# # Ajout à l'historique global pour le tracé
# loss_history.extend(loss_history_lbfgs)


# --- Plot 1: Historique de la Loss ---
plt.figure(figsize=(8, 5))
plt.plot(loss_history, label='Loss Adam')
plt.yscale('log')
plt.grid(True, alpha=0.5)
plt.xlabel('Epoch')
plt.ylabel('Loss (Log Scale)')
plt.title('Historique de la Loss Physique')
plt.legend()
plt.show()


# %% Évaluation et Visualisation
model.eval()

N_GRID = 1000

# --- p(x, k) courbes
xs = torch.linspace(0, 1, N_GRID)
ks = torch.tensor([0.0, 0.4, 0.8, 0.9, 1.3, 2.0])
X, K = torch.meshgrid(xs, ks, indexing='ij')
P_true = exact_solution(X, K)
with torch.no_grad():
    P_pred = get_p(model,
                   X.unsqueeze(-1).to(device),
                   K.unsqueeze(-1).to(device)
                   ).squeeze(-1).cpu()

plt.figure(figsize=(10, 6))
colors = plt.cm.viridis(np.linspace(0, 1, len(ks)))
for i, k_val in enumerate(ks):
    plt.plot(xs, P_true[:, i], color=colors[i], linestyle='-', linewidth=2, label=f'Exact, k={k_val:.2f}')
    plt.plot(xs, P_pred[:, i], color=colors[i], linestyle='--', linewidth=2, label=f'PINN, k={k_val:.2f}')

plt.grid(True, alpha=0.3)
plt.xlabel('x')
plt.ylabel('p(x, k)')
plt.title('Solutions exactes et PINN pour différentes valeurs de k')
plt.legend()
plt.show()

# --- p(x, k) heatmaps
ks = torch.linspace(K_MIN, K_MAX, N_GRID)
X, K = torch.meshgrid(xs, ks, indexing='ij')
P_true = exact_solution(X, K)
with torch.no_grad():
    P_pred = get_p(model,
                   X.unsqueeze(-1).to(device),
                   K.unsqueeze(-1).to(device)
                   ).squeeze(-1).cpu()

Ps = [P_true, P_pred]
titles = ["exact", "PINN"]
mini = min([P.min().item() for P in Ps])
maxi = max([P.max().item() for P in Ps])
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
for i, ax in enumerate(axes):
    import matplotlib.colors as colors
    cp = ax.pcolormesh(X, K, Ps[i],
                        norm=colors.SymLogNorm(linthresh=0.05, linscale=1.0, 
                                               vmin=mini, vmax=maxi)
                       )
    plt.colorbar(cp)
    ax.set_xlabel('x')
    ax.set_ylabel('k')
    ax.set_title(titles[i])
plt.suptitle('p(x, k) (Échelle SymLog)')
plt.show()