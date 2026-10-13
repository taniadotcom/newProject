from pathlib import Path
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import csv
import hashlib
import json
import math
import platform
import time
import torch
import numpy as np
import scipy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as cfg
from core.model import PINN3D
from core.pinn import poisson_residual_3d
from core.utils import generate_points, generate_boundary_points, create_output_directory


def train_pinn(model, X_f, X_b, epochs):
    x_f = X_f.detach().clone().requires_grad_(True)
    assert torch.equal(x_f.detach(), X_f)
    u_b = torch.zeros((X_b.shape[0], 1), dtype=X_b.dtype)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=200, factor=0.5)
    history = []

    start = time.perf_counter()
    for epoch in range(epochs):
        optimizer.zero_grad()
        x_f.grad = None
        loss_f = torch.mean(poisson_residual_3d(model, x_f)**2)
        loss_b = torch.mean((model(X_b) - u_b)**2)
        loss = loss_f + loss_b
        loss.backward()
        optimizer.step()
        scheduler.step(loss.item())
        history.append((epoch + 1, loss.item(), loss_f.item(), loss_b.item()))
        if (epoch + 1) % 500 == 0:
            print(f"Epoch {epoch + 1:5d} | Loss: {loss.item():.6e}", flush=True)
    elapsed = time.perf_counter() - start
    assert torch.equal(x_f.detach(), X_f)
    return elapsed, np.asarray(history)


def build_pielm_system(X_f, X_b, W, b):
    Z_f = W @ X_f.T + b
    phi_f = torch.tanh(Z_f)
    dphi_f = 1 - phi_f**2
    d2phi_f = -2 * phi_f * dphi_f
    m, l, k = W[:, 0:1], W[:, 1:2], W[:, 2:3]
    H_f = -((m**2 + l**2 + k**2) * d2phi_f).T
    H_b = torch.tanh(W @ X_b.T + b).T
    H = torch.cat([H_f, H_b], dim=0)
    f_vals = 3 * math.pi**2 * (
        torch.sin(math.pi * X_f[:, 0:1]) *
        torch.sin(math.pi * X_f[:, 1:2]) *
        torch.sin(math.pi * X_f[:, 2:3]))
    u_b = torch.zeros((X_b.shape[0], 1), dtype=X_b.dtype)
    K = torch.cat([f_vals, u_b], dim=0)
    return H, K


def solve_pielm(X_f, X_b, W, b):
    start = time.perf_counter()
    H, K = build_pielm_system(X_f, X_b, W, b)
    solution = torch.linalg.lstsq(H, K, driver="gelsd")
    elapsed = time.perf_counter() - start
    return solution.solution, elapsed, int(solution.rank)


def check_boundary(X_b, n_face):
    assert X_b.shape == (6 * n_face, 3)
    assert torch.all((X_b >= 0) & (X_b <= 1))
    assert torch.all(((X_b == 0) | (X_b == 1)).any(dim=1))
    for dim in range(3):
        for value in (0, 1):
            face = X_b[(2 * dim + value) * n_face:(2 * dim + value + 1) * n_face]
            assert torch.all(face[:, dim] == value)


def main(smoke=False):
    cfg.EXP_ID = "exp_10_3d_pinn_vs_pielm"
    cfg.PROBLEM = "sine_3d"
    cfg.DIM = 3
    cfg.METHOD = "sobol"
    cfg.LR = 1e-3
    cfg.EPOCHS = 3 if smoke else 20000
    # Τα παλιά 3D scripts χρησιμοποιούν float32, παρά το default του config
    cfg.DTYPE = torch.float32
    N_f = 32 if smoke else 2048
    N_b = 8 if smoke else 256
    seed = 42
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    np.random.seed(seed)

    base = Path(__file__).resolve().parents[2] / "Results" / cfg.EXP_ID
    kind = "smoke" if smoke else "run"
    output = create_output_directory(base, prefix=kind)
    cfg.RESULTS_DIR = str(output)
    relative_output = output.relative_to(Path(__file__).resolve().parents[2])
    print(f"Output: {relative_output.as_posix()}", flush=True)

    start = time.perf_counter()
    X_f = generate_points("sobol", 3, N_f, seed=seed, dtype=cfg.DTYPE)
    X_b = generate_boundary_points("sobol", 3, N_b, seed=seed + 1, dtype=cfg.DTYPE)
    sampling_sobol = time.perf_counter() - start
    check_boundary(X_b, N_b)
    X_f_saved, X_b_saved = X_f.clone(), X_b.clone()
    start = time.perf_counter()
    X_d = generate_points("delaunay", 3, N_f, seed=seed, dtype=cfg.DTYPE)
    sampling_delaunay = time.perf_counter() - start
    np.savez(output / "points.npz", X_f=X_f.numpy(), X_b=X_b.numpy(), X_d=X_d.numpy())

    torch.manual_seed(seed)
    model = PINN3D().to(dtype=cfg.DTYPE)
    initial_state = {name: value.clone() for name, value in model.state_dict().items()}
    N_star = N_f + X_b.shape[0]
    torch.manual_seed(seed)
    W = torch.randn(N_star, 3, dtype=cfg.DTYPE) * math.pi
    b = torch.randn(N_star, 1, dtype=cfg.DTYPE) * math.pi

    metadata = {
        "status": "running", "kind": kind, "seed": seed, "boundary_seed": seed + 1,
        "N_f": N_f, "N_b_per_face": N_b, "N_b_total": X_b.shape[0], "N_total": N_star,
        "metrics_columns": {"N_f": "interior points", "N_b": "total boundary points (all 6 faces)"},
        "sampling": "sobol", "boundary_sampling": "sobol", "dtype": str(cfg.DTYPE),
        "domain": "[0,1]^3", "PDE": "-Delta u = 3*pi^2*sin(pi*x)*sin(pi*y)*sin(pi*z)",
        "solution": "sin(pi*x)*sin(pi*y)*sin(pi*z)", "boundary_values": 0,
        "PINN": {"hidden_layers": [100, 100, 100], "activation": "tanh",
                 "output": "linear", "initialization": "unchanged torch.nn.Linear default",
                 "parameters": sum(p.numel() for p in model.parameters()),
                 "epochs": cfg.EPOCHS, "optimizer": "Adam", "lr": cfg.LR,
                 "scheduler": "ReduceLROnPlateau(patience=200, factor=0.5)",
                 "loss": "mean(residual^2) + mean(boundary_error^2)"},
        "PIELM": {"N_star": N_star, "scale": math.pi, "driver": "gelsd",
                  "rcond": "torch default: eps(dtype) * max(H.shape)",
                  "loss": "unweighted least squares on stacked H_f, H_b"},
        "device": "cpu", "processor": platform.processor(), "threads": torch.get_num_threads(),
        "python": sys.version, "torch": torch.__version__, "numpy": np.__version__,
        "scipy": scipy.__version__, "matplotlib": matplotlib.__version__,
        "evaluation": "20 x 20 x 20 meshgrid, endpoints included",
        "timing": "PINN loss/backward/Adam/scheduler/history; PIELM H/K construction+lstsq. "
                  "Both exclude sampling, initialization, evaluation, plotting, file I/O; PINN includes progress prints.",
        "sampling_sobol_sec": sampling_sobol, "sampling_delaunay_sec": sampling_delaunay,
        "X_f_sha256": hashlib.sha256(X_f.numpy().tobytes()).hexdigest(),
        "X_b_sha256": hashlib.sha256(X_b.numpy().tobytes()).hexdigest(),
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    # Τα ίδια X_f και X_b περνούν και στους δύο solvers, χωρίς δεύτερο sampling
    print("PINN / Sobol", flush=True)
    time_pinn, loss_sobol = train_pinn(model, X_f, X_b, cfg.EPOCHS)
    assert torch.equal(X_f, X_f_saved) and torch.equal(X_b, X_b_saved)
    torch.save(model.state_dict(), output / "pinn_sobol.pt")
    np.savetxt(output / "loss_sobol.csv", loss_sobol, delimiter=",",
               header="epoch,loss,loss_f,loss_b", comments="")

    print("PIELM / shared Sobol points", flush=True)
    c, time_pielm, rank = solve_pielm(X_f, X_b, W, b)
    assert torch.equal(X_f, X_f_saved) and torch.equal(X_b, X_b_saved)
    torch.save({"W": W, "b": b, "c": c}, output / "pielm_sobol.pt")
    metadata["shared_points_verified"] = True
    metadata["PIELM"]["rank"] = rank

    # Ίδια αρχικά βάρη και boundaries: αλλάζουν μόνο τα interior points
    model_d = PINN3D().to(dtype=cfg.DTYPE)
    model_d.load_state_dict(initial_state)
    assert all(torch.equal(model_d.state_dict()[key], value) for key, value in initial_state.items())
    print("PINN / Delaunay", flush=True)
    time_delaunay, loss_delaunay = train_pinn(model_d, X_d, X_b, cfg.EPOCHS)
    torch.save(model_d.state_dict(), output / "pinn_delaunay.pt")
    np.savetxt(output / "loss_delaunay.csv", loss_delaunay, delimiter=",",
               header="epoch,loss,loss_f,loss_b", comments="")

    n_eval = 20
    x = np.linspace(0, 1, n_eval)
    X_g, Y_g, Z_g = np.meshgrid(x, x, x)
    XYZ = torch.tensor(np.stack([X_g.flatten(), Y_g.flatten(), Z_g.flatten()], axis=1),
                       dtype=cfg.DTYPE)
    u_exact = (np.sin(math.pi * X_g) * np.sin(math.pi * Y_g) * np.sin(math.pi * Z_g)).flatten()
    with torch.no_grad():
        u_pinn = model(XYZ).numpy().flatten()
        u_delaunay = model_d(XYZ).numpy().flatten()
        u_pielm = (torch.tanh(W @ XYZ.T + b).T @ c).numpy().flatten()
    np.savez(output / "predictions.npz", XYZ=XYZ.numpy(), exact=u_exact,
             pinn=u_pinn, pielm=u_pielm, delaunay=u_delaunay)

    rows = []
    for method, sampling, prediction, elapsed in [
            ("PINN", "sobol", u_pinn, time_pinn),
            ("PIELM", "sobol", u_pielm, time_pielm),
            ("PINN", "delaunay", u_delaunay, time_delaunay)]:
        error = np.linalg.norm(prediction - u_exact) / np.linalg.norm(u_exact)
        assert np.isfinite(error)
        rows.append({"method": method, "sampling": sampling, "N_f": N_f,
                     "N_b": X_b.shape[0], "relative_L2": float(error), "time_sec": elapsed})
        print(f"{method} / {sampling}: L2={error:.12e}, time={elapsed:.6f}s", flush=True)
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    # Κοινή τομή και κοινή κλίμακα χρωμάτων για όλες τις προβλέψεις
    index = n_eval // 2
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    predictions = [u_pinn, u_pielm, u_delaunay]
    labels = ["PINN / Sobol", "PIELM / Sobol", "PINN / Delaunay"]
    exact_slice = u_exact.reshape(n_eval, n_eval, n_eval)[:, :, index]
    slices = [u.reshape(n_eval, n_eval, n_eval)[:, :, index] for u in predictions]
    lo = min(0, min(s.min() for s in slices))
    hi = max(1, max(s.max() for s in slices))
    max_error = max(np.abs(s - exact_slice).max() for s in slices)
    for j, (label, values) in enumerate(zip(labels, slices)):
        im = axes[0, j].contourf(X_g[:, :, index], Y_g[:, :, index], values,
                                levels=np.linspace(lo, hi, 40), cmap="viridis")
        fig.colorbar(im, ax=axes[0, j])
        axes[0, j].set_title(label)
        im = axes[1, j].contourf(X_g[:, :, index], Y_g[:, :, index], np.abs(values - exact_slice),
                                levels=np.linspace(0, max_error, 40), cmap="inferno")
        fig.colorbar(im, ax=axes[1, j])
        axes[1, j].set_title("Absolute error")
    for ax in axes.flat:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    fig.suptitle(f"3D sine Poisson, z={x[index]:.4f} ({kind})")
    fig.tight_layout()
    fig.savefig(output / "comparison.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.semilogy(loss_sobol[:, 0], loss_sobol[:, 1], label="Sobol")
    ax.semilogy(loss_delaunay[:, 0], loss_delaunay[:, 1], label="Delaunay")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("PINN loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "convergence.png", dpi=160)
    plt.close(fig)

    metadata["status"] = "completed"
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Μικρό test, όχι τελικό πείραμα")
    args = parser.parse_args()
    main(smoke=args.smoke)
