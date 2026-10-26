from pathlib import Path
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import math
import time
from core.utils import generate_points, generate_boundary_points, create_output_directory
import config as cfg
from core.logger import log_experiment

# Ρυθμίσεις πειράματος
cfg.EXP_ID = "exp_04_pielm_sine"
cfg.PROBLEM = "sine"
cfg.METHOD = "sobol"
cfg.DIM = 2
cfg.SIGMA = None
cfg.RESULTS_DIR = str(create_output_directory(
    Path(__file__).resolve().parents[2] / "Results" / cfg.EXP_ID))

N_f = 1024  # Πλήθος collocation points
N_b = 256  # Boundary points ανά πλευρά, 4 × 256 = 1024 συνολικά
N_star = N_f + 4 * N_b  # Ένας κρυφός νευρώνας ανά εξίσωση του συστήματος

print(f"PIELM | N_f={N_f} | N_b={4*N_b} | N*={N_star}")


# Σταθερά τυχαία βάρη του κρυφού layer

torch.manual_seed(42)


scale = math.pi  # για sine πρόβλημα
W = torch.randn(N_star, cfg.DIM) * scale
b = torch.randn(N_star, 1) * scale

# Δημιουργία collocation points
X_f = generate_points(method=cfg.METHOD, dim=cfg.DIM, n_points=N_f)
# X_f: N_f × 2

# Δημιουργία boundary points
X_b = generate_boundary_points(method=cfg.METHOD, dim=2, n_points=N_b)


# Κατασκευή του πίνακα H

start = time.time()

# Block H_f στα collocation points
# z = W · x^T + b  →  σχήμα: N* × N_f
Z_f = W @ X_f.T + b  # N* × N_f
phi_f = torch.tanh(Z_f)  # φ(z)
dphi_f = 1 - phi_f**2  # φ'(z) = 1 - tanh²(z)

# Για Poisson: -(u_xx + u_yy) = f
# u_xx για νευρώνα j = m_j² · φ''(z_j)  όπου φ''= -2·tanh·(1-tanh²)
# Αλλά χρησιμοποιούμε: -(u_xx + u_yy) = -(m_j² + n_j²) · φ''(z_j)
m = W[:, 0:1]  # βάρη ως προς x: N* × 1
n = W[:, 1:2]  # βάρη ως προς y: N* × 1

d2phi_f = -2 * phi_f * dphi_f  # φ''(z)
H_f = -((m**2 + n**2) * d2phi_f).T  # Αρνητικό Laplacian των basis functions

# Block H_b με τις τιμές στα boundary points
Z_b = W @ X_b.T + b  # N* × N_bc
H_b = torch.tanh(Z_b).T  # N_bc × N*

# Συνολικός H
H = torch.cat([H_f, H_b], dim=0)  # (N_f + N_bc) × N*


# Κατασκευή του δεξιού μέλους K

# K_f τιμές f στα collocation points
f_vals = 2 * math.pi**2 * torch.sin(math.pi * X_f[:, 0:1]) * torch.sin(math.pi * X_f[:, 1:2])

# K_b τιμές u στα boundary points (u=0 για sine)
u_b_vals = torch.zeros(X_b.shape[0], 1)

K = torch.cat([f_vals, u_b_vals], dim=0)  # (N_f + N_bc) × 1


# Επίλυση Hc = K με least squares

c = torch.linalg.lstsq(H, K, driver='gelsd').solution  # N* × 1

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

u_exact = np.sin(math.pi * X_grid) * np.sin(math.pi * Y_grid)
error = np.linalg.norm(u_pred - u_exact) / np.linalg.norm(u_exact)

print(f"Relative L2 Error: {error:.3e}")

log_experiment(
    exp_id = cfg.EXP_ID,
    problem = cfg.PROBLEM,
    method = f"PIELM-{cfg.METHOD}",
    sigma = None,
    n_points = N_star,
    epochs = 0,
    l2_error = error,
    time_sec = elapsed,
    notes = f"scale=pi, seed=42, N_f={N_f}, N_b={4*N_b}"
)
