import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path
import torch

import math
from core.model import PINN
from core.pinn import poisson_residual_inverse
from core.utils import create_output_directory


output = create_output_directory(Path(__file__).resolve().parents[2] / "Results" / "exp_03_inverse")

N_f = 8000
x_f = torch.rand(N_f, 2, requires_grad=True)


N_u = 100  # Σημεία παρατήρησης της λύσης
x_u = torch.rand(N_u, 2)
u_u = torch.sin(math.pi * x_u[:, 0:1]) * torch.sin(math.pi * x_u[:, 1:2])

# Δίκτυο και άγνωστη παράμετρος alpha
model = PINN()
alpha = torch.nn.Parameter(torch.tensor([0.5], requires_grad=True))

optimizer = torch.optim.Adam(
    list(model.parameters()) + [alpha],
    lr=1e-3
)


for epoch in range(10000):
    optimizer.zero_grad()

    loss_f = torch.mean(poisson_residual_inverse(model, x_f, alpha)**2)
    loss_u = torch.mean((model(x_u) - u_u)**2)  # Loss ως προς τις παρατηρήσεις u_u

    loss = loss_f + loss_u
    loss.backward()
    optimizer.step()

    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.3e}, alpha: {alpha.item():.5f}")

print(f"Identified alpha = {alpha.item():.6f}")
# Αποθήκευση μοντέλου και εκτίμησης alpha

torch.save({
    "model": model.state_dict(),
    "alpha": alpha.item()
}, output / "poisson_inverse_pinn.pt")
