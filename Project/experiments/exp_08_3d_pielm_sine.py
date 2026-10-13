from pathlib import Path
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import numpy as np
import math
import time
from core.utils import generate_points, generate_boundary_points, create_output_directory
from core.logger import log_experiment
import config as cfg

# Ρυθμίσεις πειράματος

cfg.EXP_ID = "exp_08_3d_pielm_sine"
cfg.PROBLEM = "sine_3d"
cfg.METHOD = "sobol"
cfg.DIM = 3
cfg.RESULTS_DIR = str(create_output_directory(
    Path(__file__).resolve().parents[2] / "Results" / cfg.EXP_ID))

N_f = 2048
N_b = 256
N_star = N_f + N_b * 6

print(f"PIELM 3D | N_f={N_f} | N_b_per_face={N_b} | N_b_total={6*N_b} | N*={N_star}")


# Σταθερά τυχαία βάρη του κρυφού layer

torch.manual_seed(42)
scale = math.pi
W = torch.randn(N_star, 3) * scale
b = torch.randn(N_star, 1) * scale

# Βάρη ως προς x
m = W[:, 0:1]

# Βάρη ως προς y
l = W[:, 1:2]

# Βάρη ως προς z
k = W[:, 2:3]


# Δημιουργία collocation points
X_f = generate_points(method=cfg.METHOD, dim=3, n_points=N_f)


# Δημιουργία boundary points
X_b = generate_boundary_points(method=cfg.METHOD, dim=3, n_points=N_b)

# Κατασκευή του πίνακα H

start = time.time()

# Block H_f στα collocation points
Z_f = W @ X_f.T + b  # N* × N_f
phi_f = torch.tanh(Z_f)
dphi_f = 1 - phi_f**2
d2phi_f = -2 * phi_f * dphi_f

# -(u_xx + u_yy + u_zz) = -(m² + l² + k²) · φ''(z)
H_f = -((m**2 + l**2 + k**2) * d2phi_f).T  # N_f × N*

# Block H_b: N_bc × N*
Z_b = W @ X_b.T + b
H_b = torch.tanh(Z_b).T

# Συνολικός H: (N_f + N_bc) × N*
H = torch.cat([H_f, H_b], dim=0)

# f = 3π²·sin(πx)·sin(πy)·sin(πz)
f_vals = 3 * math.pi**2 * (
    torch.sin(math.pi * X_f[:, 0:1]) *
    torch.sin(math.pi * X_f[:, 1:2]) *
    torch.sin(math.pi * X_f[:, 2:3])
)

# Μηδενικές τιμές σε όλα τα boundary points
u_b_vals = torch.zeros(X_b.shape[0], 1)

K = torch.cat([f_vals, u_b_vals], dim=0)


# Επίλυση Hc = K με least squares
c = torch.linalg.lstsq(H, K, driver='gelsd').solution

elapsed = time.time() - start
print(f"Training time: {elapsed:.4f} seconds")

# Αξιολόγηση

n_eval = 20
x = np.linspace(0, 1, n_eval)
y = np.linspace(0, 1, n_eval)
z = np.linspace(0, 1, n_eval)
X_g, Y_g, Z_g = np.meshgrid(x, y, z)

XYZ_eval = torch.tensor(
    np.stack([X_g.flatten(), Y_g.flatten(), Z_g.flatten()], axis=1),
    dtype=torch.float32
)

Z_eval = W @ XYZ_eval.T + b
u_pred = (torch.tanh(Z_eval).T @ c).detach().numpy().flatten()

u_exact = (np.sin(math.pi * X_g) *
           np.sin(math.pi * Y_g) *
           np.sin(math.pi * Z_g)).flatten()

error = np.linalg.norm(u_pred - u_exact) / np.linalg.norm(u_exact)

print(f"Relative L2 Error: {error:.3e}")

# Καταγραφή μετρικών
log_experiment(
    exp_id = cfg.EXP_ID,
    problem = cfg.PROBLEM,
    method = f"PIELM-{cfg.METHOD}",
    sigma = None,
    n_points = N_star,
    epochs = 0,
    l2_error = error,
    time_sec = elapsed,
    notes = f"3D PIELM, scale=pi, seed=42, N_f={N_f}, N_b={6*N_b}"
)
