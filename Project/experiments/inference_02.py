import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
import math
from core.model import PINN
from core.utils import create_output_directory


# Φόρτωση μοντέλου και εκτίμησης alpha

checkpoint = torch.load(str(Path(__file__).resolve().parents[2] / "Results" / "exp_03_inverse" / "run_001" / "poisson_inverse_pinn.pt"), map_location="cpu", weights_only=True)

output = create_output_directory(Path(__file__).resolve().parents[2] / "Results" / "inference_02")

model = PINN()

model.load_state_dict(checkpoint["model"])
model.eval()

alpha_identified = checkpoint["alpha"]
print(f"Identified alpha = {alpha_identified:.6f}")

# Πλέγμα αξιολόγησης

n = 100
x = np.linspace(0, 1, n)
y = np.linspace(0, 1, n)
X, Y = np.meshgrid(x, y)

XY = torch.tensor(
    np.stack([X.flatten(), Y.flatten()], axis=1),
    dtype=torch.float32
)

# Πρόβλεψη λύσης

with torch.no_grad():
    u_pred = model(XY).numpy().reshape(n, n)

# Αναλυτική λύση
u_exact = np.sin(math.pi * X) * np.sin(math.pi * Y)


# Σχετικό L2 σφάλμα

error = np.linalg.norm(u_pred - u_exact) / np.linalg.norm(u_exact)
print(f"Relative L2 Error = {error:.3e}")


# Διαγράμματα λύσης και σφάλματος

plt.figure(figsize=(15, 4))


# Πρόβλεψη PINN
plt.subplot(1, 3, 1)
plt.contourf(X, Y, u_pred, 50, cmap="viridis")
plt.colorbar()
plt.title("PINN Solution")
plt.xlabel("x")
plt.ylabel("y")

# Αναλυτική λύση
plt.subplot(1, 3, 2)
plt.contourf(X, Y, u_exact, 50, cmap="viridis")
plt.colorbar()
plt.title("Exact Solution")
plt.xlabel("x")
plt.ylabel("y")

# Απόλυτο σφάλμα
plt.subplot(1, 3, 3)
plt.contourf(X, Y, np.abs(u_pred - u_exact), 50, cmap="inferno")
plt.colorbar()
plt.title("Absolute Error")
plt.xlabel("x")
plt.ylabel("y")

plt.tight_layout()
plt.savefig(output / "inference_inverse_results.png")
plt.close()
