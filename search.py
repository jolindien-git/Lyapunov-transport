import torch
import numpy as np
import time
import argparse
import copy

from models.pinn import PINN_P_BC
from problem import Problem


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--time_budget', type=float, default=360.0*2, help="search (total) time")
    
    parser.add_argument('--lambd', type=float, default=2.0, help="system")
    parser.add_argument('--mu', type=float, default=1.0, help="system")
    parser.add_argument('--sigma', type=float, default=0.3, help="system")
    parser.add_argument('--k', type=float, default=.0, help="gain")
    
    parser.add_argument('--epochs', type=int, default=100//2, help="number of epochs (stochastic gradient descent)")
    parser.add_argument('--epochs_lbfgs', type=int, default=50//2, help="number of epochs (LBFGS)")
    parser.add_argument('--batch_size', type=int, default=2_000, help="number of collocation point per batch")
    parser.add_argument('--batch_number', type=int, default=20, help="number of batch per epochs (stochastic gradient descent)")
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()
    return args


def check_stability(model: PINN_P_BC, problem: Problem, device: str):
    eigenvalues = problem.check_positivity_P(model, device)
    if np.all(eigenvalues > 0):
        eigenvalues = problem.check_positivity_Q(model, device)
        if np.all(eigenvalues > 0):
            return True
    return False


if __name__ == "__main__":
    args = parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PINN_P_BC(hidden_dim=128, n_layers=4).to(device)
    
    k = args.k
    dk = .1
    model_back = copy.deepcopy(model)
    best_k = None
    
    start_time = time.time()
    time_end = lambda : time.time() - start_time >= args.time_budget
    while not time_end():
        # -- initiaize problem (new k value)
        print("=" * 10 , "k %.4f" % k, "dk %.4f" % dk,
              "time %i / %i" % (time.time() - start_time, args.time_budget))
        problem = Problem(args.sigma, args.lambd, args.mu, k)
        
        # -- train (stochastic gradient descent)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
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
            print(f"k {k:.4f} Ep [{epoch+1}/{args.epochs}] Loss={epoch_loss:.2e}")
            
            # -- stability check (early stopping)
            stable = check_stability(model, problem, device)
            if stable or time_end():
                break

        if not stable:
            # -- train (L-BFGS)
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
                lbfgs_optimizer.zero_grad()
                res = problem.residual_pde(model, x1, x2)
                loss = problem.loss_pde(res)
                loss.backward()
                return loss
            
            for epoch in range(args.epochs_lbfgs):
                loss = lbfgs_optimizer.step(closure)
                print(f"k {k:.4f} L-BFGS [{epoch+1}/{args.epochs_lbfgs}] -"
                      f"Loss: {loss.item():.2e}")
                # -- stability check (early stopping)
                stable = check_stability(model, problem, device)
                if stable or time_end():
                    break
            
        if stable:
            print(f"k {k:.4f} ✅")
            # -- save
            model_back = copy.deepcopy(model)
            best_k = k
            # -- update k & dk
            k += dk
            dk *= 1.1
        else:
            print(f"k {k:.4f} ❌")
            # -- load
            model = copy.deepcopy(model_back)
            k = best_k
            # -- update k & dk
            dk /= 3
            k += dk
            
    print("out of time")