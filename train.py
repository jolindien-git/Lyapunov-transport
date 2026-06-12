import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from models.base import MLP


c = 20.0
lam = 2.0
K_MIN, K_MAX = 0., 2. # .9, 1.# 1.2, 2.0 

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class GaussianActivation(nn.Module):
    def forward(self, x):
        return torch.exp(-(100*x**2)) #############
    
class PINN(nn.Module):
    def __init__(self, hidden_dim=128*2, n_layers=3):
        
        super(PINN, self).__init__()
        
        self.net = MLP(in_dim=2,
                         out_dim=1,
                         hidden_dim=hidden_dim,
                         n_layers=n_layers,
                          # activation=GaussianActivation
                         )
        
        self.den = MLP(in_dim=2,
                        out_dim=1,
                        hidden_dim=hidden_dim,
                        n_layers=n_layers)
        
        self.num = MLP(in_dim=2,
                        out_dim=1,
                        hidden_dim=hidden_dim,
                        n_layers=n_layers)
        
    def forward(self, x, k):
        
        def N(inputs):
            # return self.net(inputs)
            return self.net(inputs) / (self.den(inputs)**2 + 1e-8)
            # return self.num(inputs) + self.net(inputs) / (self.den(inputs)**2 + 1e-8)
        
        inputs = torch.cat([x, k], dim=-1)
        N_x = N(inputs)
        
        # -- reformulation pour imposer p(1) = k^2 p(0) par construction        
        zeros = torch.zeros_like(x)
        inputs = torch.cat([zeros, k], dim=-1)
        N_0 = N(inputs)
        p = (1 - x) * N_x + x * (k**2) * N_0
        return p


def get_grad(fx, x):
    return torch.autograd.grad(fx, x, grad_outputs=torch.ones_like(fx),
                              create_graph=True)[0]


def loss_phys(model, x, k):
    """
    Calcule le résidu de l'EDO : c*p'(x) + lam*p(x) + 1 = 0
    """
    p = model(x, k)
    dp_dx = get_grad(p, x)
    res = c * dp_dx + lam * p + 1.0
    
    # res_relative = res / (p.abs() + 1.)
    # return torch.mean(res_relative**2)
    return torch.mean(res**2)


def model_exact(x, k):
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
EPOCHS = 5000*4*2
N_f = 2000
LR = 1e-2/4
SCHEDULER_STEP = EPOCHS // 5

model = PINN().to(device)


optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=SCHEDULER_STEP, gamma=0.5)

print("Début de l'entraînement...")
loss_history = []
for epoch in range(EPOCHS):
    model.train()
    
    # Échantillonnage aléatoire uniforme dans [0, 1]x[0, 2] à chaque époque
    x_train = torch.rand(N_f, 1, requires_grad=True, device=device)
    k_train = K_MIN + torch.rand(N_f, 1, device=device) * (K_MAX - K_MIN)
    
    optimizer.zero_grad()
    
    loss = loss_phys(model, x_train, k_train)
    ######## TEST SUPERVISE
    # p_true = model_exact(x_train, k_train)
    # p_pred = model(x_train, k_train)
    # loss = nn.HuberLoss()(p_true, p_pred)
    
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
    optimizer.step()
    scheduler.step()
    
    loss_history.append(loss.item())
    
    if (epoch + 1) % 500 == 0:
        # idx = p_true.argmax().item()
        # print("k %.6f  x %.2f  p %.2e" % (k_train[idx], x_train[idx], p_true[idx].item()))
        print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {loss.item():.4e} - LR: {scheduler.get_last_lr()[0]:.4e}")

print("Entraînement terminé !")


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

# --- p(x, k) courbes pour quelques k
xs = torch.linspace(0, 1, N_GRID)
ks = torch.tensor([0.0, 0.4, 0.8, 0.9, 1.3, 2.0])
X, K = torch.meshgrid(xs, ks, indexing='ij')
P_true = model_exact(X, K)
with torch.no_grad():
    P_pred = model(X.unsqueeze(-1).to(device), K.unsqueeze(-1).to(device)).squeeze(-1).cpu()

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


# --- p(x, k) courbes pour quelques x
xs = torch.tensor([0.0, 0.5, 1.0])
ks = torch.linspace(K_MIN, K_MAX, N_GRID)
X, K = torch.meshgrid(xs, ks, indexing='ij')
P_true = model_exact(X, K)
with torch.no_grad():
    P_pred = model(X.unsqueeze(-1).to(device), K.unsqueeze(-1).to(device)).squeeze(-1).cpu()

plt.figure(figsize=(10, 6))
for i, k_val in enumerate(xs):
    plt.plot(ks, P_true[i, :], '-', linewidth=2, label=f'Exact, k={k_val:.2f}')
    plt.plot(ks, P_pred[i, :], '--', linewidth=2, label=f'PINN, k={k_val:.2f}')

plt.grid(True, alpha=0.3)
plt.xlabel('x')
plt.ylabel('p(x, k)')
plt.ylim([P_pred.min().item(), P_pred.max().item()])
plt.title('Solutions exactes et PINN pour différentes valeurs de x')
plt.legend()
plt.show()

# --- p(x, k) heatmaps
ks = torch.linspace(K_MIN, K_MAX, N_GRID)
X, K = torch.meshgrid(xs, ks, indexing='ij')
P_true = model_exact(X, K)
with torch.no_grad():
    P_pred = model(X.unsqueeze(-1).to(device), K.unsqueeze(-1).to(device)).squeeze(-1).cpu()

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