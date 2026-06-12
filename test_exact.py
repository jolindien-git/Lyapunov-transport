import torch
import numpy as np
import matplotlib.pyplot as plt

c = 20.0
lam = 2.0


def model_exact(*args):
    '''
    !!! SOLUTION QUI CORRESPOND (VISUELLEMENT) A LA FIGURE 6 DU DRAFT
    !!! NE CORRESPOND PAS A p_exact DU DRAFT
    Args:
        x, k
        OR
        inputs: concatenation of (x, k)
    '''
    if len(args) == 1:
        xk = args[0]
        x, k = xk[..., 0:1], xk[..., 1:2]
    else:
        x, k = args
    
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


# --- p(x, k) courbes pour quelques k
xs = torch.linspace(0, 1, 100)
ks = torch.tensor([0.0, 0.4, 0.8, 0.9, 1.3, 2.0])
X, K = torch.meshgrid(xs, ks)
P = model_exact(X, K)

plt.figure()
plt.plot(xs, P)
plt.title('p(x,k) exact')
plt.xlabel('x')

# --- p(x, k) courbes pour quelques x
K_MIN, K_MAX = 0, .93 # .951, .9513# 1.2, 2.0 
xs = torch.linspace(0, 1, 5)
ks = torch.linspace(K_MIN, K_MAX, 1000)
X, K = torch.meshgrid(xs, ks)
P = model_exact(X, K)

plt.figure()
plt.plot(ks, P.T)
plt.title('p(x,k) exact')
plt.xlabel('k')


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


# -- courbes