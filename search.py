"""
python search.py --dk0 -.01 --k0 1 --search_nonGES --name nonGES_decrease
python search.py --dk0 .01 --k0 -1 --search_nonGES --name nonGES_increase
python search.py --dk0 -.01 --k0 0 --name GES_decrease
python search.py --dk0 .01 --k0 0 --name GES_increase
"""

import torch
import numpy as np
import time
import argparse
import copy
import os, csv

from models.pinn import PINN_P_BC
from problem import Problem


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--time_budget', type=float, default=60*15, help="search (total) time")
    parser.add_argument('--name', type=str, default='temp', help="result: saving name")
    parser.add_argument('--search_nonGES', action='store_true', help="search non GES pocket (else search GES pocket)")
    
    parser.add_argument('--lambd', type=float, default=2.0, help="system parameter")
    parser.add_argument('--mu', type=float, default=1.0, help="system parameter")
    parser.add_argument('--sigma', type=float, default=0.3, help="Lyapunov system")
    
    parser.add_argument('--k0', type=float, default=.0, help="initial guess")
    parser.add_argument('--dk0', type=float, default=.01, help="initial step")
    parser.add_argument('--alpha', type=float, default=1.2, help="adaptation factor +")
    parser.add_argument('--beta', type=float, default=2.0, help="adaptation factor -")
    
    parser.add_argument('--epochs', type=int, default=100//2, help="number of epochs (stochastic gradient descent)")
    parser.add_argument('--epochs_lbfgs', type=int, default=50//2, help="number of epochs (LBFGS)")
    parser.add_argument('--batch_size', type=int, default=2_000, help="number of collocation point per batch")
    parser.add_argument('--batch_number', type=int, default=20, help="number of batch per epochs (stochastic gradient descent)")
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()
    return args


def check_stability(model: PINN_P_BC, problem: Problem, device: str):
    eigenvalues = problem.check_positivity_Q(model, device)
    if np.all(eigenvalues > 0):
        eigenvalues = problem.check_positivity_P(model, device)
        if np.all(eigenvalues > 0):
            return True # GES
        else:
            return False # non-GES
    else:
        return None # inconclusive


def is_success(stable, search_notGES=False):
    """
    args:
        stable (True, False, or None): returned by check_stability
        search_notGES (True or False): search GES pocket or non GES pocket
    """
    if search_notGES:
        return stable == False
    else:
        return stable == True


if __name__ == "__main__":
    args = parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PINN_P_BC(hidden_dim=128, n_layers=4).to(device)
    
    k = args.k0
    dk = args.dk0
    model_back = copy.deepcopy(model)
    best_k = None
    history = {"k": [], "dk": [], "time": [], "best_k": [], "success": []}
    
    start_time = time.time()
    time_end = lambda : time.time() - start_time >= args.time_budget
    while not time_end():
        # -- initiaize problem (new k value)
        print("=" * 10 , "k %.4f" % k, "dk %.2e" % dk,
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
            success = is_success(check_stability(model, problem, device),
                                 search_notGES=args.search_nonGES)
            if success or time_end():
                break

        if not success:
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
                success = is_success(check_stability(model, problem, device),
                                     search_notGES=args.search_nonGES)
                if success or time_end():
                    break
        
        # save history
        history['k'].append(k)
        history['dk'].append(dk)
        history['time'].append(time.time() - start_time)
        history['best_k'].append(k if success else best_k)
        history['success'].append(success)
        
        # update
        if success:
            print(f"k {k:.4f} ✅")
            # -- save
            model_back = copy.deepcopy(model)
            best_k = k
            # -- update k & dk
            k += dk
            dk *= args.alpha
        else:
            print(f"k {k:.4f} ❌")
            # -- load
            model = copy.deepcopy(model_back)
            k = best_k
            # -- update k & dk
            dk /= args.beta
            k += dk
        
    print("out of time")
    
    #-- CSV file
    os.makedirs("results", exist_ok=True)
    csv_path = os.path.join("results", args.name + ".csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([k for k in history.keys()])
        for i in range(len(history['k'])):
            row = [history[k][i] for k in history.keys()]
            writer.writerow(row)