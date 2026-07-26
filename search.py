import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import argparse
import copy
import warnings

from models.pinn import PINN_P_BC
from problem import Problem


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--time_budget', type=float, default=360.0*2, help="search (total) time")
    
    parser.add_argument('--lambd', type=float, default=2.0, help="system")
    parser.add_argument('--mu', type=float, default=1.0, help="system")
    parser.add_argument('--sigma', type=float, default=0.3, help="system")
    parser.add_argument('--k', type=float, default=.0, help="gain")
    
    parser.add_argument('--epochs', type=int, default=100, help="number of epochs (stochastic gradient descent)")
    parser.add_argument('--epochs_lbfgs', type=int, default=50, help="number of epochs (LBFGS)")
    parser.add_argument('--batch_size', type=int, default=2_000, help="number of collocation point per batch")
    parser.add_argument('--batch_number', type=int, default=20, help="number of batch per epochs (stochastic gradient descent)")
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()
    return args


from torch.optim.lr_scheduler import SequentialLR, LinearLR, StepLR

def configure_schedulers(optimizer: torch.optim.Optimizer, epochs: int) -> SequentialLR:
    """
    Configures a sequential learning rate scheduler with a warmup phase 
    followed by a step decay phase.
    
    Args:
        optimizer: The optimizer to schedule.
        epochs: Total number of training epochs.
        
    Returns:
        A sequential learning rate scheduler.
    """
    warmup_epochs = epochs // 5
    decay_step = max(1, epochs // 6)
    
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=1.0 / 100.0,
        end_factor=1.0,
        total_iters=warmup_epochs
    )
    
    decay_scheduler = StepLR(
        optimizer,
        step_size=decay_step,
        gamma=0.5
    )
    
    sequential_scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, decay_scheduler],
        milestones=[warmup_epochs]
    )
    
    return sequential_scheduler


# %% main
if __name__ == "__main__":
    args = parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PINN_P_BC(hidden_dim=128, n_layers=4).to(device)
    
    k = args.k
    dk = .1
    model_back = copy.deepcopy(model)
    # optim_back = copy.deepcopy(optimizer)
    
    start_time = time.time()
    time_end = lambda : time.time() - start_time >= args.time_budget
    while not time_end():
        print("=" * 20 , "k %.4f" % k, "dk %.4f" % dk)
        problem = Problem(args.sigma, args.lambd, args.mu, k)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = configure_schedulers(optimizer, args.epochs)
        
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
            print(f"k {k:.4f} Ep [{epoch+1}/{args.epochs}] Loss={epoch_loss:.2e} "
                  f"LR {scheduler.get_last_lr()[0]:.1e}")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                scheduler.step()
            
            stable = False
            eigenvalues = problem.check_positivity_P(model, device)
            if np.all(eigenvalues > 0):
                eigenvalues = problem.check_positivity_Q(model, device)
                if np.all(eigenvalues > 0):
                    stable = True
                    break
            if time_end():
                print("end of time")
                break
        if stable:
            print(f"k {k:.4f} ✅" % eigenvalues.min())
        else:
            print(f"k {k:.4f} ❌"  % eigenvalues.min())
                    
            # Nouvel échantillonnage pour L-BFGS
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
                print(f"k {k:.4f} L-BFGS [{epoch+1}/{args.epochs_lbfgs}] -"
                      f"Loss: {loss.item():.2e}")
                
                stable = False
                eigenvalues = problem.check_positivity_P(model, device)
                if np.all(eigenvalues > 0):
                    eigenvalues = problem.check_positivity_Q(model, device)
                    if np.all(eigenvalues > 0):
                        stable = True
                        break
                if time_end():
                    print("end of time")
                    break
            if stable:
                print(f"k {k:.4f} ✅" % eigenvalues.min())
            else:
                print(f"k {k:.4f} ❌"  % eigenvalues.min())
            
        if stable:
            k += dk
            dk *= 1.1
            model_back = copy.deepcopy(model)
            # optim_back = copy.deepcopy(optimizer)
        else:
            k -= dk
            dk /= 3
            k += dk
            model = copy.deepcopy(model_back)
            # optimizer = copy.deepcopy(optim_back)