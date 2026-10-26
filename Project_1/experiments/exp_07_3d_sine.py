from pathlib import Path
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import math
import time
from core.model import PINN3D
from core.pinn import poisson_residual_3d
from core.utils import generate_points, generate_boundary_points, create_output_directory
from core.logger import log_experiment
import config as cfg


# Ρυθμίσεις πειράματος

cfg.EXP_ID = "exp_07_3d_sine"
cfg.PROBLEM = "sine_3d"
cfg.METHOD = "sobol"
cfg.DIM = 3
cfg.EPOCHS = 20000
cfg.LR = 1e-3
cfg.RESULTS_DIR = str(create_output_directory(
    Path(__file__).resolve().parents[2] / "Results" / cfg.EXP_ID))

print(f"Ξεκινάει: {cfg.EXP_ID} | {cfg.METHOD.upper()} | 3D SINE")


# Δημιουργία collocation points

x_f = generate_points(method=cfg.METHOD, dim=cfg.DIM, n_points=8192)
x_f.requires_grad_(True)


# Boundary points στις έξι επιφάνειες του [0,1]³

N_b = 512

X_b = generate_boundary_points(method=cfg.METHOD, dim=3, n_points=N_b)

u_b = torch.zeros((X_b.shape[0], 1))

print(f"Collocation: {x_f.shape[0]} | Boundary: {X_b.shape[0]} (6×{N_b})")


# Εκπαίδευση

model = PINN3D()
optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=200, factor=0.5
)

start = time.time()

for epoch in range(cfg.EPOCHS):
    optimizer.zero_grad()

    loss_f = torch.mean(poisson_residual_3d(model, x_f)**2)
    loss_b = torch.mean((model(X_b) - u_b)**2)
    loss = loss_f + loss_b

    loss.backward()
    optimizer.step()
    scheduler.step(loss.item())

    if epoch % 500 == 0:
        print(f"Epoch {epoch:5d} | Loss: {loss.item():.3e}")

elapsed = time.time() - start


# Αξιολόγηση

n_eval = 20
x = np.linspace(0, 1, n_eval)
y = np.linspace(0, 1, n_eval)
z = np.linspace(0, 1, n_eval)
X_g, Y_g, Z_g = np.meshgrid(x, y, z)

XYZ = torch.tensor(
    np.stack([X_g.flatten(), Y_g.flatten(), Z_g.flatten()], axis=1),
    dtype=torch.float32
)

u_pred = model(XYZ).detach().numpy().flatten()
u_exact = (np.sin(math.pi * X_g) *
           np.sin(math.pi * Y_g) *
           np.sin(math.pi * Z_g)).flatten()

error = np.linalg.norm(u_pred - u_exact) / np.linalg.norm(u_exact)

print(f"\nRelative L2 Error : {error:.3e}")
print(f"Training time     : {elapsed:.1f}s")


# Αποθήκευση αποτελεσμάτων

torch.save(model.state_dict(), f"{cfg.RESULTS_DIR}/model.pt")

log_experiment(
    exp_id = cfg.EXP_ID,
    problem = cfg.PROBLEM,
    method = cfg.METHOD,
    sigma = None,
    n_points = x_f.shape[0],
    epochs = cfg.EPOCHS,
    l2_error = error,
    time_sec = elapsed,
    notes = f"3D sine, f=3pi^2*sin*sin*sin, 6 faces boundary"
)
