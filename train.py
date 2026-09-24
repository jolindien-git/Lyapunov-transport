import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import argparse

from models.pinn import PINN_P_BC
from problem import Problem


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--lambd', type=float, default=2.0, help="system")
    parser.add_argument('--mu', type=float, default=1.0, help="system")
    parser.add_argument('--sigma', type=float, default=0.3, help="system")
    parser.add_argument('--k', type=float, default=.6, help="gain")
    
    parser.add_argument('--epochs', type=int, default=100, help="number of epochs (stochastic gradient descent)")
    parser.add_argument('--epochs_lbfgs', type=int, default=50, help="number of epochs (LBFGS)")
    parser.add_argument('--batch_size', type=int, default=2_000, help="number of collocation point per batch")
    parser.add_argument('--batch_number', type=int, default=20, help="number of batch per epochs (stochastic gradient descent)")
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()
    return args


def plot_result(model, loss_history, loss_history_lbfgs, device, problem: Problem):
    
    N_GRID = 150
    x = torch.linspace(0.0, 1.0, N_GRID, device=device)
    X1_t, X2_t = torch.meshgrid(x, x, indexing="ij")
    
    # -- training loss
    plt.figure(figsize=(8, 5))
    len_adam = len(loss_history)
    len_lbfgs = len(loss_history_lbfgs)
    
    plt.semilogy(range(len_adam), loss_history, label="Loss Adam")
    if len_lbfgs > 0:
        plt.semilogy(range(len_adam, len_adam + len_lbfgs), loss_history_lbfgs,
                     label="Loss L-BFGS")
    plt.grid(True, alpha=0.5)
    plt.xlabel("Iterations")
    plt.title("Loss")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # -- 2D maps : compare P_exact and P_theta
    model.eval()    
    k_tensor = torch.full((N_GRID*N_GRID, 1), fill_value=problem.k, device=device)
    with torch.no_grad():
        P_pred = model(X1_t.reshape(-1, 1), X2_t.reshape(-1, 1), k_tensor)
        P_pred = P_pred.reshape(N_GRID, N_GRID).cpu().numpy()
        P_true = problem.P_exact(X1_t, X2_t, device).cpu().numpy()
        
    plt.figure(figsize=(6, 5))
    im = plt.imshow(np.abs(P_pred - P_true), origin='lower', extent=[0, 1, 0, 1])
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title(f'k={problem.k:.2f} - Error |P_PINN - P_exact|')
    plt.colorbar(im)
    plt.tight_layout()
    plt.show()
    
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    vmin = min(P_true.min(), P_pred.min())
    vmax = max(P_true.max(), P_pred.max())
    levels = 10
    axes[0].imshow(P_true, origin="lower", extent=[0, 1, 0, 1], vmin=vmin, vmax=vmax)
    axes[0].contour(P_true, origin="lower", extent=[0, 1, 0, 1], colors="white", linewidths=0.4, alpha=.4, levels=levels)
    axes[0].set_title(r"$P$")
    axes[0].set_xlabel("$x_1$")
    axes[0].set_ylabel("$x_2$")
    im = axes[1].imshow(P_pred, origin="lower", extent=[0, 1, 0, 1], vmin=vmin, vmax=vmax)
    axes[1].contour(P_pred, origin="lower", extent=[0, 1, 0, 1], colors="white", linewidths=0.4, alpha=.4, levels=levels)
    axes[1].set_title(r"$P_{\theta}$")
    axes[1].set_xlabel("$x_1$")
    fig.colorbar(im, ax=axes, shrink=0.85)
    plt.show()
    fig.savefig('results/train_P_Ptheta.pdf', dpi=200, bbox_inches="tight", pad_inches=0.03)
    
    # -- 1D curve (fixed x2): compare P_exact and P_theta
    x2_fixed = 0.5
    
    x1_line = torch.linspace(0.0, 1.0, N_GRID, device=device).unsqueeze(-1)
    x2_line = torch.full_like(x1_line, fill_value=x2_fixed)
    k_line = torch.full_like(x1_line, fill_value=problem.k)
    with torch.no_grad():
        P_true_line = problem.P_exact(x1_line, x2_line, device).squeeze().cpu().numpy()
        P_pred_line = model(x1_line, x2_line, k_line).squeeze().cpu().numpy()
    
    plt.figure(figsize=(5, 5))
    plt.plot(x1_line.cpu().numpy(), P_true_line, 'b-', label='Exact')
    plt.plot(x1_line.cpu().numpy(), P_pred_line, 'r--', label='PINN')
    plt.xlabel('x1')
    plt.ylabel('P(x1, %.2f)' % x2_fixed)
    plt.title(f'k={problem.k:.2f}')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
    


# %% main
if __name__ == "__main__":
    args = parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    problem = Problem(args.sigma, args.lambd, args.mu, args.k)
    model = PINN_P_BC(hidden_dim=128, n_layers=4).to(device)
    
    # %% -- Phase Adam
    print("Train: Adam loop")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, args.epochs // 5), gamma=0.5)
    
    start_time = time.time()
    loss_history = []
    for epoch in range(args.epochs):
        model.train()
        
        losses = []
        for _ in range(args.batch_number):
            x1, x2 = problem.sample_square(args.batch_size, device)
            
            optimizer.zero_grad()
            res = problem.residual_pde(model, x1, x2)
            loss = problem.loss_pde(res)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            losses.append(loss.item())
        
        epoch_loss = np.mean(losses)
        loss_history.append(epoch_loss)
        print(f"Ep [{epoch+1}/{args.epochs}] Loss={epoch_loss:.2e}  "
                  f"LR {scheduler.get_last_lr()[0]:.1e}")
        scheduler.step()
        
        from search import check_stability
        stable = check_stability(model, problem, device, degree=9)
        print("stable ?", stable)
    
    print("\t Done.")
    
    # %% L-BFGS
    print("L-BFGS...")
    
    x1, x2 = problem.sample_square(args.batch_size, device)
    
    lbfgs_optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=1.0,
        max_iter=args.batch_number,
        max_eval=args.batch_number*2,
        tolerance_grad=1e-12,
        tolerance_change=1e-12,
        history_size=100,
        line_search_fn="strong_wolfe"
    )
    
    loss_history_lbfgs = []
    def closure():
        lbfgs_optimizer.zero_grad(set_to_none=True)
        
        res = problem.residual_pde(model, x1, x2)
        loss = problem.loss_pde(res)
        loss.backward()
        
        return loss
    
    model.train()
    for epoch in range(args.epochs_lbfgs):
        loss = lbfgs_optimizer.step(closure)
        loss_history_lbfgs.append(loss.item())
        
        print(f"L-BFGS [{epoch+1}/{args.epochs_lbfgs}] -"
              f"Loss: {loss.item():.2e}")
        
        from search import check_stability
        stable = check_stability(model, problem, device, degree=9)
        print("stable ?", stable)
    
    
    # elapsed time
    total_duration = time.time() - start_time
    print("=" * 60)
    print(f"Elapsed time: {total_duration/60:.2f} min ({total_duration:.2f} s)")
    
    # %% plots
    plot_result(model, loss_history, loss_history_lbfgs, device, problem)
    
    # %% Test P_theta > 0 & Q_theta > 0

    eigenvalues = problem.check_positivity_P(model, device)
    if np.all(eigenvalues > 0):
        print("P: min(eigen values) = %.2e Positive definite. ✅" % eigenvalues.min())
    else:
        print("P min(eigen values) = %.2e  NOT Positive definite. ❌"  % eigenvalues.min())
    
    
    eigenvalues = problem.check_positivity_Q(model, device)
    if np.all(eigenvalues > 0):
        print("Q: min(eigen values) = %.2e Positive definite. ✅" % eigenvalues.min())
    else:
        print("Q min(eigen values) = %.2e  NOT Positive definite. ❌"  % eigenvalues.min())
    