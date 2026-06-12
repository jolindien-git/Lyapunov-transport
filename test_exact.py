import torch
import numpy as np
import matplotlib.pyplot as plt

c = 20.0
lam = 2.0


def exact_solution(x, k):
    num = 1- k**2
    den = lam * (np.exp(-lam / c) - k**2)
    return (num / den) * torch.exp(-lam / c * x) - 1 / lam


xs = torch.linspace(0, 1, 10)
ks = torch.tensor([0.0, 0.4, 0.8, 0.9, 1.3, 2.0])
X, K = torch.meshgrid(xs, ks)
P = exact_solution(X, K)

plt.plot(xs, P)
# plt.figure(figsize=(10, 6))
# colors = plt.cm.viridis(np.linspace(0, 1, len(ks)))
# for i, k in enumerate(ks):
#     ps = exact_solution(xs, k)
#     plt.plot(xs, ps, color=colors[i], linestyle='-', linewidth=2, label=f'Exact, k={k}')

# plt.legend()
# plt.grid(True, alpha=0.3)
# plt.xlabel('x')
# plt.ylabel('p(x, k)')
# plt.title('Solutions exactes et PINN pour différentes valeurs de k')