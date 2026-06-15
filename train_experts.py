import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from models.base import MLP

c = 20.0
lam = 2.0
K_MIN, K_MAX = 0.8, 1.1

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class MoEPINN(nn.Module):
    def __init__(self, hidden_dim=128, n_layers=3):
        super(MoEPINN, self).__init__()
        
        self.net_U = MLP(in_dim=2, out_dim=1, hidden_dim=hidden_dim, n_layers=n_layers)
        
        self.net_L = MLP(in_dim=2, out_dim=1, hidden_dim=hidden_dim, n_layers=n_layers)
        self.pos_act = nn.Softplus()
        
        self.router = MLP(in_dim=1, out_dim=2, hidden_dim=hidden_dim, n_layers=n_layers)
        
    def _compute_ansatz(self, net, x, k, is_constrained=False):
        inputs_x = torch.cat([x, k], dim=-1)
        N_x = net(inputs_x)
        
        zeros = torch.zeros_like(x)
        inputs_0 = torch.cat([zeros, k], dim=-1)
        N_0 = net(inputs_0)
        
        if is_constrained:
            N_x = N_x**2# self.pos_act(N_x)
            N_0 = N_0**2 #self.pos_act(N_0)
            
        p = (1 - x) * N_x + x * (k**2) * N_0
        return p

    def forward(self, x, k, tau=1.0):
        p_U = self._compute_ansatz(self.net_U, x, k, is_constrained=False)
        p_L = self._compute_ansatz(self.net_L, x, k, is_constrained=True)
        
        logits = self.router(k)
        
        w = torch.nn.functional.gumbel_softmax(logits, tau=tau, hard=True, dim=-1)
        
        w_L = w[:, 0:1]
        w_U = w[:, 1:2]
        
        p = w_L * p_L + w_U * p_U
        return p


def get_grad(fx, x):
    return torch.autograd.grad(fx, x, grad_outputs=torch.ones_like(fx),
                              create_graph=True)[0]


def get_residual(model, x, k, tau):
    p = model(x, k, tau)
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
EPOCHS = 2000
N_f = 1000
LR = 1e-2 / 4
SCHEDULER_STEP = EPOCHS // 5

TAU_START = 5.0
TAU_END = 0.1

model = MoEPINN().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=SCHEDULER_STEP, gamma=0.5)

print("Starting Adam training...")
loss_history = []

for epoch in range(EPOCHS):
    model.train()
    
    decay_rate = (TAU_END / TAU_START) ** (epoch / max(1, EPOCHS - 1))
    current_tau = max(TAU_END, TAU_START * decay_rate)
    
    N_BATCHS = 10
    loss_mean = 0
    
    for batch in range(N_BATCHS):
        x_train = torch.rand(N_f, 1, requires_grad=True, device=device)
        k_train = K_MIN + torch.rand(N_f, 1, device=device) * (K_MAX - K_MIN)
        
        optimizer.zero_grad()
        
        residual = get_residual(model, x_train, k_train, current_tau)
        loss = get_loss(residual)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        
        loss_mean += loss.item() / N_BATCHS
        
    loss_history.append(loss_mean)
    
    if (epoch + 1) % 50 == 0 or epoch == 0 or epoch == EPOCHS - 1:
        print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {loss_mean:.3e} - LR: {scheduler.get_last_lr()[0]:.2e} - Tau: {current_tau:.3f}")
    
    scheduler.step()

print("Adam training completed.")