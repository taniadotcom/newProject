from pathlib import Path
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

import time
from core.utils import generate_points, generate_boundary_points, create_output_directory
from core.logger import log_experiment
import config as cfg

# Ρυθμίσεις πειράματος
cfg.EXP_ID = "exp_05_pielm_gauss"
cfg.PROBLEM = "gauss"
cfg.METHOD = "sobol"
cfg.DIM = 2
cfg.SIGMA = 0.5
cfg.RESULTS_DIR = str(create_output_directory(
    Path(__file__).resolve().parents[2] / "Results" / cfg.EXP_ID))

N_f = 1024
N_b = 256  # 256 σημεία ανά πλευρά, 1024 συνολικά
N_star = N_f + 4 * N_b

print(f"PIELM Gauss | σ={cfg.SIGMA} | N_f={N_f} | N_b={4*N_b} | N*={N_star}")

# Σταθερά τυχαία βάρη του κρυφού layer
torch.manual_seed(42)

scale = 1.0 / cfg.SIGMA  # Κλίμακα 1/σ για τα τυχαία βάρη
W = torch.randn(N_star, cfg.DIM) * scale
b = torch.randn(N_star, 1) * scale

# Δημιουργία collocation points
X_f = generate_points(method=cfg.METHOD, dim=cfg.DIM, n_points=N_f)

# Δημιουργία boundary points
X_b = generate_boundary_points(method=cfg.METHOD, dim=2, n_points=N_b)

# Κατασκευή του πίνακα H
start = time.time()

m = W[:, 0:1]  # βάρη ως προς x: N* × 1
n = W[:, 1:2]  # βάρη ως προς y: N* × 1

# Block H_f για το αρνητικό Laplacian
Z_f = W @ X_f.T + b  # N* × N_f
phi_f = torch.tanh(Z_f)  # φ(z)
dphi_f = 1 - phi_f**2  # φ'(z)
d2phi_f = -2 * phi_f * dphi_f  # φ''(z)
H_f = -((m**2 + n**2) * d2phi_f).T  # N_f × N*

# Block H_b με τις τιμές στα σύνορα
Z_b = W @ X_b.T + b
H_b = torch.tanh(Z_b).T  # N_bc × N*

# Συνολικός H
H = torch.cat([H_f, H_b], dim=0)  # (N_f + N_bc) × N*

# Κατασκευή του δεξιού μέλους K
# K_f: δεξί μέλος της Gaussian Poisson
x_c = X_f[:, 0:1] - 0.5
y_c = X_f[:, 1:2] - 0.5
r_sq = x_c**2 + y_c**2
u_exact_f = torch.exp(-r_sq / (2 * cfg.SIGMA**2))
f_vals = u_exact_f * (2.0 / cfg.SIGMA**2 - r_sq / cfg.SIGMA**4)

# K_b: οριακές τιμές από αναλυτική λύση
r_b_sq = (X_b[:, 0:1] - 0.5)**2 + (X_b[:, 1:2] - 0.5)**2
u_b_vals = torch.exp(-r_b_sq / (2 * cfg.SIGMA**2))

K = torch.cat([f_vals, u_b_vals], dim=0)  # (N_f + N_bc) × 1


# Επίλυση Hc = K με least squares

c = torch.linalg.lstsq(H, K, driver='gelsd').solution

elapsed = time.time() - start
print(f"Training time: {elapsed:.4f} seconds")

# Αξιολόγηση
n_eval = 100
x = np.linspace(0, 1, n_eval)
y = np.linspace(0, 1, n_eval)
X_grid, Y_grid = np.meshgrid(x, y)

XY_eval = torch.tensor(
    np.stack([X_grid.flatten(), Y_grid.flatten()], axis=1),
    dtype=torch.float32
)

Z_eval = W @ XY_eval.T + b
u_pred = (torch.tanh(Z_eval).T @ c).detach().numpy().reshape(n_eval, n_eval)

r_sq_np = (X_grid - 0.5)**2 + (Y_grid - 0.5)**2
u_exact = np.exp(-r_sq_np / (2 * cfg.SIGMA**2))
error = np.linalg.norm(u_pred - u_exact) / np.linalg.norm(u_exact)

print(f"Relative L2 Error: {error:.3e}")

log_experiment(
    exp_id = cfg.EXP_ID,
    problem = cfg.PROBLEM,
    method = f"PIELM-{cfg.METHOD}",
    sigma = cfg.SIGMA,
    n_points = N_star,
    epochs = 0,
    l2_error = error,
    time_sec = elapsed,
    notes = f"scale=1/sigma={scale:.1f}, seed=42, N_f={N_f}, N_b={4*N_b}, rhs=-laplacian"
)
