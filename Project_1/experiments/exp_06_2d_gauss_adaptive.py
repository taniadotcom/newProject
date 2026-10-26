from pathlib import Path
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

import time
from core.model import PINN_Large
from core.pinn import poisson_residual_gauss
from core.utils import generate_points, generate_boundary_points, create_output_directory
from core.logger import log_experiment
import config as cfg

# Ρυθμίσεις πειράματος
cfg.EXP_ID = "exp_06_gauss_adaptive"
cfg.PROBLEM = "gauss"
cfg.METHOD = "sobol"
cfg.DIM = 2
cfg.SIGMA = 0.5
cfg.EPOCHS = 20000
cfg.RESULTS_DIR = str(create_output_directory(
    Path(__file__).resolve().parents[2] / "Results" / cfg.EXP_ID))

ADAPT_EVERY = 2000  # Απόσταση μεταξύ προσαρμογών σε epochs
ADAPT_N_NEW = 256  # Νέα σημεία ανά προσαρμογή
ADAPT_CANDS = 8192  # Πλήθος υποψήφιων σημείων

print(f"Ξεκινάει: {cfg.EXP_ID} | σ={cfg.SIGMA} | adaptive every {ADAPT_EVERY} epochs")

# Δημιουργία collocation points
x_f = generate_points(method=cfg.METHOD, dim=cfg.DIM)
x_f.requires_grad_(True)

# Δημιουργία boundary points
X_b = generate_boundary_points(method=cfg.METHOD, dim=2, n_points=cfg.N_BOUNDARY)

r_b_sq = (X_b[:, 0:1] - 0.5)**2 + (X_b[:, 1:2] - 0.5)**2
u_b = torch.exp(-r_b_sq / (2 * cfg.SIGMA**2))

# Εκπαίδευση

model = PINN_Large()
optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=200, factor=0.5
)

start = time.time()

for epoch in range(cfg.EPOCHS):
    optimizer.zero_grad()

    loss_f = torch.mean(poisson_residual_gauss(model, x_f, cfg.SIGMA)**2)
    loss_b = torch.mean((model(X_b) - u_b)**2)
    loss = loss_f + loss_b

    loss.backward()
    optimizer.step()
    scheduler.step(loss)

    # Προσαρμοστική προσθήκη collocation points

    if epoch > 0 and epoch % ADAPT_EVERY == 0:
        # Δειγματοληψία υποψήφιων σημείων
        x_cands = generate_points(method="random", dim=cfg.DIM, n_points=ADAPT_CANDS)
        x_cands.requires_grad_(True)

        # Υπολογισμός residual στα υποψήφια σημεία
        with torch.enable_grad():
            res = poisson_residual_gauss(model, x_cands, cfg.SIGMA)**2

        # Επιλογή των σημείων με το μεγαλύτερο residual
        _, idx = torch.topk(res.squeeze().detach(), ADAPT_N_NEW)
        x_new = x_cands[idx].detach()
        x_new.requires_grad_(True)


        x_f = torch.cat([x_f.detach(), x_new], dim=0)
        x_f.requires_grad_(True)

        print(f"  [Adapt] Epoch {epoch} | +{ADAPT_N_NEW} σημεία | Σύνολο: {x_f.shape[0]}")

    if epoch % 500 == 0:
        print(f"Epoch {epoch:5d} | Loss: {loss.item():.3e} | N_f: {x_f.shape[0]}")

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
r_sq_np = (X - 0.5)**2 + (Y - 0.5)**2
u_exact = np.exp(-r_sq_np / (2 * cfg.SIGMA**2))
error = np.linalg.norm(u_pred - u_exact) / np.linalg.norm(u_exact)

print(f"\nRelative L2 Error : {error:.3e}")
print(f"Training time     : {elapsed:.1f}s")
print(f"Final N_f         : {x_f.shape[0]}")

# Αποθήκευση αποτελεσμάτων
torch.save(model.state_dict(), f"{cfg.RESULTS_DIR}/model.pt")

log_experiment(
    exp_id = cfg.EXP_ID,
    problem = cfg.PROBLEM,
    method = f"adaptive-{cfg.METHOD}",
    sigma = cfg.SIGMA,
    n_points = x_f.shape[0],
    epochs = cfg.EPOCHS,
    l2_error = error,
    time_sec = elapsed,
    notes = f"adapt_every={ADAPT_EVERY}, adapt_n_new={ADAPT_N_NEW}, rhs=-laplacian"
)
