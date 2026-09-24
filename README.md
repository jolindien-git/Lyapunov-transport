# Identifying Lyapunov functionals with a PINN: application to a hyperbolic transport equation

This repository contains the PyTorch implementation of the numerical results presented in the joint SYSID 2027 / IFAC Journal of Systems and Control submission. 

It provides a mesh-free, parametric Physics-Informed Neural Network (PINN) approach to identify Lyapunov operators for infinite-dimensional systems (hyperbolic transport equations). Unavoidable numerical residuals are folded into a redefined right-hand side, allowing strict certification of $\mu$-exponential stability ($\mu$-GES) via finite-dimensional Gram-matrix positivity tests.

## 📂 Project Structure

* **`train.py`**: Script to perform a nominal PINN training for a fixed parameter $\kappa$.
* **`search.py`**: Iterative adaptive search algorithm to find the critical stability boundaries $\kappa_{\mathrm{crit}}$.
* **`problem.py`**: Defines the `Problem` class, including the PDE residual evaluation, exact constant-speed transport kernel computation, and the interface for positivity checks.
* **`positivity/polynomial_test.py`**: Implements the finite-section certification via a Gram-matrix projection onto shifted Legendre polynomials.
* **`models/pinn.py`**: Contains `PINN_P_BC`, the neural network architecture that inherently strictly satisfies the symmetry and non-local boundary conditions of the Lyapunov equation.
* **`models/base.py`**: Core definitions of the Multi-Layer Perceptron (MLP) and SIREN networks.

## ⚙️ Dependencies
* Python 3.x
* PyTorch (`torch`)
* NumPy (`numpy`)
* SciPy (`scipy`)
* Matplotlib (`matplotlib`)

## 🚀 Usage & Reproducibility

### 1. Nominal Training (Validation at a fixed $\kappa$)
To reproduce the nominal training (e.g., at an interior point $\kappa=0.6$) and validate the nominal identification, run the `train.py` script with default options:

```bash
python train.py
```
This script will:
1. Optimize the network using an Adam optimizer followed by an L-BFGS refinement.
2. Evaluate the continuous residual and dynamically check the stability criteria.
3. Generate and display the training loss curve, the 2D heatmaps comparing the PINN kernel $P_\theta$ against the exact kernel $P$, and a 1D cross-section plot. 
4. Output the final Gram-matrix positivity verification for both $P$ and $Q$.

### 2. Adaptive Search for Stability Boundaries
To reproduce the experiments bounding the stability pocket (identifying $\pm \kappa_{\mathrm{crit}}$), the `search.py` script utilizes transfer learning (warm-starting) across varying values of $\kappa$. 

Execute the following four commands to replicate the four searches described in the paper's numerical results:

**Search I: Certifying $\mu$-GES from the center (increasing $\kappa$)**
```bash
python search.py --dk0 .01 --k0 0 --name GES_increase
```

**Search II: Certifying $\mu$-GES from the center (decreasing $\kappa$)**
```bash
python search.py --dk0 -.01 --k0 0 --name GES_decrease
```

**Search III: Certifying non-$\mu$-GES from the outside (decreasing $\kappa$)**
```bash
python search.py --dk0 -.01 --k0 1 --search_nonGES --name nonGES_decrease
```

**Search IV: Certifying non-$\mu$-GES from the outside (increasing $\kappa$)**
```bash
python search.py --dk0 .01 --k0 -1 --search_nonGES --name nonGES_increase
```

Each run has a default time budget of 15 minutes (`--time_budget 900`). The results, including the sequence of tested $\kappa$ and processing times, are logged and exported as CSV files in the `results/` directory.