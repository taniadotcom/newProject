from pathlib import Path
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import math
import time


import config as cfg
from core.model import PINN
from core.pinn import poisson_residual
from core.utils import generate_points, generate_boundary_points, create_output_directory
from core.logger import log_experiment

# Ρυθμίσεις πειράματος

cfg.EXP_ID = "exp_01_2d_sine"
cfg.PROBLEM = "sine"
cfg.METHOD = "halton"
cfg.DIM = 2
cfg.EPOCHS = 20000
cfg.RESULTS_DIR = str(create_output_directory(
    Path(__file__).resolve().parents[2] / "Results" / cfg.EXP_ID))

print(f"Ξεκινάει: {cfg.EXP_ID} | {cfg.METHOD.upper()} | {cfg.PROBLEM.upper()}")


# Δημιουργία collocation points

x_f = generate_points(method=cfg.METHOD, dim=cfg.DIM)
x_f.requires_grad_(True)


# Δημιουργία boundary points

X_b = generate_boundary_points(method=cfg.METHOD, dim=2, n_points=cfg.N_BOUNDARY)


# Αναλυτική λύση: u(x,y) = sin(πx)·sin(πy)
# Στα σύνορα (x=0,1 ή y=0,1): u=0

u_b = torch.zeros((X_b.shape[0], 1))


# Εκπαίδευση

model = PINN()
optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=200, factor=0.5)

start = time.time()

for epoch in range(cfg.EPOCHS):
    optimizer.zero_grad()

    loss_f = torch.mean(poisson_residual(model, x_f)**2)
    loss_b = torch.mean((model(X_b) - u_b)**2)
    loss = loss_f + loss_b

    loss.backward()
    optimizer.step()

    scheduler.step(loss)

    if epoch % 500 == 0:
        print(f"Epoch {epoch:5d} | Loss: {loss.item():.3e}")

elapsed = time.time() - start


# Αξιολόγηση

n = 100
x = np.linspace(0, 1, n)
y = np.linspace(0, 1, n)
X, Y = np.meshgrid(x, y)

XY = torch.tensor(
    np.stack([X.flatten(), Y.flatten()], axis=1),
    dtype=torch.float32
)

u_pred = model(XY).detach().numpy().reshape(n, n)
u_exact = np.sin(math.pi * X) * np.sin(math.pi * Y)
error = np.linalg.norm(u_pred - u_exact) / np.linalg.norm(u_exact)

print(f"\nRelative L2 Error : {error:.3e}")
print(f"Training time     : {elapsed:.1f}s")


# Αποθήκευση αποτελεσμάτων

torch.save(model.state_dict(), f"{cfg.RESULTS_DIR}/model.pt")
relative_output = Path(cfg.RESULTS_DIR).relative_to(Path(__file__).resolve().parents[2])
print(f"Model saved → {relative_output.as_posix()}/model.pt")

log_experiment(
    exp_id = cfg.EXP_ID,
    problem = cfg.PROBLEM,
    method = cfg.METHOD,
    sigma = None,
    n_points = x_f.shape[0],
    epochs = cfg.EPOCHS,
    l2_error = error,
    time_sec = elapsed,
    notes = cfg.NOTES
)
