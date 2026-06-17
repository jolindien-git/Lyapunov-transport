import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from models.base import MLP

c = 20.0
lam = 2.0
K_MIN, K_MAX = 0., 2. #0.8, 1.1

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class MoEPINN(nn.Module):
    def __init__(self, hidden_dim=128, n_layers=4):
        super(MoEPINN, self).__init__()
        
        self.net_U = MLP(in_dim=2, out_dim=1,
                         hidden_dim=hidden_dim,
                         n_layers=n_layers)
        
        self.net_L = MLP(in_dim=2, out_dim=1, hidden_dim=hidden_dim, n_layers=n_layers)
        # self.pos_act = nn.Softplus()
        
        self.router = MLP(in_dim=1, out_dim=2,
                          hidden_dim=hidden_dim,
                          n_layers=n_layers)
        
    def _compute_ansatz(self, net, x, k, is_constrained=False):
        inputs_x = torch.cat([x, k], dim=-1)
        N_x = net(inputs_x)
        
        zeros = torch.zeros_like(x)
        inputs_0 = torch.cat([zeros, k], dim=-1)
        N_0 = net(inputs_0)
        
        if is_constrained:
            # -- constraint = positive
            N_x = N_x**2# self.pos_act(N_x)
            N_0 = N_0**2 #self.pos_act(N_0)
            
        p = (1 - x) * N_x + x * (k**2) * N_0
        return p

    def forward(self, x, k, tau=.001):
        # -- predict p(x,k) : Unconstraint vs Constraint
        p_U = self._compute_ansatz(self.net_U, x, k, is_constrained=False) # (B, 1)
        p_L = self._compute_ansatz(self.net_L, x, k, is_constrained=True) # (B, 1)
        
        # -- classifier
        logits = self.router(k) # (B, 2)
        w = torch.nn.functional.gumbel_softmax(logits, tau=tau, hard=True, dim=-1) # (B, 2)
        w_L = w[..., 0:1] # (B, 1) \in {0,1}
        w_U = w[..., 1:2] # (B, 1) \in {0,1}
    
        # -- final prediction
        p = w_L * p_L + w_U * p_U # (B, 1)
        return p, w_U


def get_grad(fx, x):
    return torch.autograd.grad(fx, x, grad_outputs=torch.ones_like(fx),
                              create_graph=True)[0]


def get_residual(p, x):
    dp_dx = get_grad(p, x)
    res = c * dp_dx + lam * p + 1.0
    return res


def get_loss(residual):
    return torch.mean(residual**2)


def model_exact(x, k):
    num = 1 - k**2
    den = lam * (np.exp(-lam / c) - k**2)
    return (num / den) * torch.exp(-lam / c * x) - 1 / lam


# %% Training Loop
EPOCHS = 2000//2#*2
N_f = 1000
LR = 1e-2 / 4
SCHEDULER_STEP = EPOCHS // 5

TAU_START = 5.0
TAU_END = 0.001

model = MoEPINN().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-6)
# optimizer = torch.optim.Adam([
#     {'params': model.net_U.parameters()},
#     {'params': model.net_L.parameters()},
#     {'params': model.router.parameters(), 'lr': LR / 200, 'weight_decay': 1e-4} 
# ], lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=SCHEDULER_STEP, gamma=0.5)

print("Starting Adam training...")
loss_history = []


def get_misplaced(k, w_U):
    thresh = 0.9512294
    mask = k[:,0] < thresh
    misplaced_low = w_U[mask].sum().item()
    misplaced_high = (~mask).sum().item() - w_U[~mask].sum().item()
    return misplaced_low, misplaced_high


for epoch in range(EPOCHS):
    model.train()
    
    decay_rate = (TAU_END / TAU_START) ** (epoch / max(1, EPOCHS - 1))
    current_tau = max(TAU_END, TAU_START * decay_rate)
    
    N_BATCHS = 10
    loss_mean = 0
    misplaced_low = 0
    misplaced_high = 0
    
    for batch in range(N_BATCHS):
        x_train = torch.rand(N_f, 1, requires_grad=True, device=device)
        k_train = K_MIN + torch.rand(N_f, 1, device=device) * (K_MAX - K_MIN)
        
        optimizer.zero_grad()
        
        p, w_U = model(x_train, k_train, tau=current_tau)
        residual = get_residual(p, x_train)
        BETA = 0.01 # hard to tune (0 around singularity else .01)
        residual_penalized = residual * (1.0 + BETA * w_U)
        loss = get_loss(residual_penalized)
         
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        loss_mean += loss.item() / N_BATCHS
        
        # monitoring
        low, high = get_misplaced(k_train, w_U)
        misplaced_low += low / N_BATCHS
        misplaced_high += high / N_BATCHS
        
        
    loss_history.append(loss_mean)
    
    if (epoch + 1) % 50 == 0 or epoch == 0 or epoch == EPOCHS - 1:
        print(f"Epoch [{epoch+1}/{EPOCHS}] -Loss: {loss_mean:.3e}",
              f"-LR: {scheduler.get_last_lr()[0]:.2e} -Tau: {current_tau:.1f}",
              f"-mis_low {misplaced_low:.2f} - mis_high {misplaced_high:.2f}")
    
    scheduler.step()

print("Adam training completed.")


# %% Phase d'optimisation L-BFGS
# %% Phase d'optimisation L-BFGS Séparée
print("\nPréparation pour le raffinement L-BFGS (Diviser pour mieux régner)...")

# 1. Geler le routeur
for param in model.router.parameters():
    param.requires_grad = False

# 2. Générer les points denses
N_f_lbfgs = 1000 * 10
x_lbfgs = torch.rand(N_f_lbfgs, 1, requires_grad=True, device=device)
k_lbfgs = K_MIN + torch.rand(N_f_lbfgs, 1, device=device) * (K_MAX - K_MIN)

# 3. Séparer le dataset en deux grâce au routeur figé
model.eval()
with torch.no_grad():
    _, w_U_lbfgs = model(x_lbfgs, k_lbfgs, tau=0.01)
    mask_U = (w_U_lbfgs > 0.5).squeeze()
    mask_L = ~mask_U

# Détacher et recréer les tenseurs pour avoir un requires_grad propre
x_U = x_lbfgs[mask_U].clone().detach().requires_grad_(True)
k_U = k_lbfgs[mask_U].clone().detach()

x_L = x_lbfgs[mask_L].clone().detach().requires_grad_(True)
k_L = k_lbfgs[mask_L].clone().detach()

print(f"Points alloués - Expert L (Contraint): {mask_L.sum().item()} | Expert U (Libre): {mask_U.sum().item()}")

# ==========================================
# 4A. L-BFGS pour l'Expert L (Zone Stable)
# ==========================================
if mask_L.sum() > 0:
    print("\n--- L-BFGS sur Expert L (Contraint) ---")
    opt_L = torch.optim.LBFGS(model.net_L.parameters(), lr=0.1, max_iter=100, max_eval=200, tolerance_grad=1e-11, tolerance_change=1e-11, line_search_fn="strong_wolfe")
    
    def closure_L():
        opt_L.zero_grad()
        p_L = model._compute_ansatz(model.net_L, x_L, k_L, is_constrained=True)
        res_L = get_residual(p_L, x_L)
        loss = get_loss(res_L)
        loss.backward()
        return loss

    model.train()
    for epoch in range(20): # Moins d'époques nécessaires car très efficace
        opt_L.step(closure_L)
        loss_L = closure_L().item()
        print(f"L-BFGS Expert L [{epoch+1}/20] - Loss: {loss_L:.4e}")
        if loss_L < 1e-8: break

# ==========================================
# 4B. L-BFGS pour l'Expert U (Zone Instable)
# ==========================================
if mask_U.sum() > 0:
    print("\n--- L-BFGS sur Expert U (Libre) ---")
    opt_U = torch.optim.LBFGS(model.net_U.parameters(), lr=0.1, max_iter=100, max_eval=200, tolerance_grad=1e-11, tolerance_change=1e-11, line_search_fn="strong_wolfe")
    
    def closure_U():
        opt_U.zero_grad()
        p_U = model._compute_ansatz(model.net_U, x_U, k_U, is_constrained=False)
        res_U = get_residual(p_U, x_U)
        loss = get_loss(res_U)
        loss.backward()
        return loss

    model.train()
    for epoch in range(20):
        opt_U.step(closure_U)
        loss_U = closure_U().item()
        print(f"L-BFGS Expert U [{epoch+1}/20] - Loss: {loss_U:.4e}")
        if loss_U < 1e-8: break

print("\nRaffinement L-BFGS global terminé !")


# Dégeler le routeur
for param in model.router.parameters():
    param.requires_grad = True


# %% test
p, w_U = model(x_train, k_train)
thresh = 0.9512294

plt.figure()
plt.plot([thresh, thresh], [0, 1], 'k--', linewidth=.5)
plt.scatter(k_train[:,0].detach().cpu(), w_U[:, 0].detach().cpu(), s=1, alpha=.3)
plt.ylabel("W_U")
plt.xlabel("k")

logits = model.router(k_train)

plt.figure()
plt.scatter(k_train[:,0].detach().cpu(), logits[:, 1].detach().cpu(), s=1, label='Unconstraint')
plt.scatter(k_train[:,0].detach().cpu(), logits[:, 0].detach().cpu(), s=1, label='Constraint (positive)')
plt.legend()
plt.ylabel("logits")
plt.xlabel("k")


# %% Évaluation et Visualisation
model.eval()

N_GRID = 1000

# --- p(x, k) courbes pour quelques k
xs = torch.linspace(0, 1, N_GRID)
ks = torch.tensor([0.0, 0.4, 0.8, 0.9, 1.3, 2.0])
X, K = torch.meshgrid(xs, ks, indexing='ij')
P_true = model_exact(X, K)
with torch.no_grad():
    P_pred, _ = model(X.unsqueeze(-1).to(device), K.unsqueeze(-1).to(device))
    P_pred = P_pred.squeeze(-1).cpu()

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
    P_pred, _ = model(X.unsqueeze(-1).to(device), K.unsqueeze(-1).to(device))
    P_pred = P_pred.squeeze(-1).cpu()

plt.figure(figsize=(10, 6))
for i, x_val in enumerate(xs):
    plt.plot(ks, P_true[i, :], '-', linewidth=2, label=f'Exact, x={x_val:.2f}')
    plt.plot(ks, P_pred[i, :], '--', linewidth=2, label=f'PINN, x={x_val:.2f}')

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
    P_pred, _ = model(X.unsqueeze(-1).to(device), K.unsqueeze(-1).to(device))
    P_pred = P_pred.squeeze(-1).cpu()

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