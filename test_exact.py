import torch
import numpy as np
import matplotlib.pyplot as plt

c = 20.0
lam = 2.0


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


def get_grad(fx, x):
    return torch.autograd.grad(fx, x, grad_outputs=torch.ones_like(fx),
                              create_graph=True)[0]


def residual(p, x):
    """
    Calcule le résidu de l'EDO : c*p'(x) + lam*p(x) + 1 = 0
    """
    dp_dx = get_grad(p, x)
    res = c * dp_dx + lam * p + 1.0
    return res


# --- p(x, k): courbes pour quelques k
xs = torch.linspace(0, 1, 100)
ks = torch.tensor([0.0, 0.4, 0.8, 0.9, 1.3, 2.0])
X, K = torch.meshgrid(xs, ks)
P = model_exact(X, K)

plt.figure(figsize=(10, 6))
colors = plt.cm.viridis(np.linspace(0, 1, len(ks)))
for i, k_val in enumerate(ks):
    plt.plot(xs, P[:, i], color=colors[i], linestyle='-', linewidth=2, label=f'Exact, k={k_val:.2f}')
plt.grid(True, alpha=0.3)
plt.xlabel('x')
plt.ylabel('p(x, k)')
plt.title('Solutions $p(x,k)$ exactes vs $x$')
plt.legend()
plt.show()


# --- p(x, k): courbes pour quelques x
K_MIN, K_MAX = 0, 2.#.93 # .951, .9513# 1.2, 2.0 
xs = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
ks = torch.linspace(K_MIN, K_MAX, 1000)
X, K = torch.meshgrid(xs, ks)
P = model_exact(X, K)

plt.figure(figsize=(10, 6))
colors = plt.cm.viridis(np.linspace(0, 1, len(ks)))
for i, x_val in enumerate(xs):
    plt.plot(ks, P[i], color=colors[i], linestyle='-', linewidth=2, label=f'Exact, x={x_val:.2f}')
plt.grid(True, alpha=0.3)
plt.xlabel('x')
plt.ylabel('p(x, k)')
plt.title('Solutions $p(x,k)$ exactes vs $k$')
plt.legend()
plt.show()



# -- p(x, k) heatmaps
xs = torch.linspace(0, 1, 20)
ks = torch.linspace(K_MIN, K_MAX, 10000)
X, K = torch.meshgrid(xs, ks, indexing='ij')
P = model_exact(X, K)

idx = P.flatten().argmax().item()
print("MAX de p = %.2e pour k %.10f  x %.2f" % (P.flatten()[idx].item(), K.flatten()[idx], X.flatten()[idx]))
idx = P.flatten().argmin().item()
print("MIN de p = %.2e pour k %.10f  x %.2f" % (P.flatten()[idx].item(), K.flatten()[idx], X.flatten()[idx]))

import matplotlib.colors as colors
plt.figure()
cp = plt.pcolormesh(X, K, P,
                    norm=colors.SymLogNorm(linthresh=0.05, linscale=1.0)
                   )
plt.colorbar(cp)
plt.xlabel('x')
plt.ylabel('k')
plt.suptitle('p(x, k) (Échelle SymLog)')


# -- Residual
X.requires_grad_(True)
P = model_exact(X, K)
res = residual(P, X).detach()
dp_dx = get_grad(P, X)
idx = dp_dx.flatten().argmax().item()
print("MAX de dp_dx = %.2e pour k %.10f  x %.2f" % (dp_dx.flatten()[idx].item(), K.flatten()[idx], X.flatten()[idx]))
idx = dp_dx.flatten().argmin().item()
print("MIN de dp_dx = %.2e pour k %.10f  x %.2f" % (dp_dx.flatten()[idx].item(), K.flatten()[idx], X.flatten()[idx]))
X = X.detach()
P = P.detach()
print('Mean |residual| =', torch.mean(res.abs()).item())

plt.figure()
cp = plt.pcolormesh(X, K, res)
plt.colorbar(cp)
plt.xlabel('x')
plt.ylabel('k')
plt.suptitle('Residual exact')

