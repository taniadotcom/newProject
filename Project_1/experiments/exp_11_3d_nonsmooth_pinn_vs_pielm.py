from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import math
import platform
import time
import numpy as np
import scipy
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core.model import PINN3D
from core.utils import generate_points, generate_boundary_points, create_output_directory, check_boundary
from core.poisson_problems import exact_nonsmooth_3d, poisson_rhs_nonsmooth_3d
from core.poisson_solvers import (train_pinn, solve_pielm, predict_pielm, relative_l2,
    point_hash, state_hash, write_metrics, memory_plan, require_memory)


def save_plots(output, XYZ, exact, predictions, histories, kind):
    grid = XYZ.numpy().reshape(20, 20, 20, 3)
    reference = exact.reshape(20, 20, 20)[:, :, 10]
    slices = [v.reshape(20, 20, 20)[:, :, 10] for v in predictions.values()]
    lo = min(reference.min(), *(s.min() for s in slices))
    hi = max(reference.max(), *(s.max() for s in slices))
    maximum = max(float(np.max(np.abs(s-reference))) for s in slices)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for j, (label, values) in enumerate(zip(predictions, slices)):
        for i, data, levels, cmap in [(0, values, np.linspace(lo, hi, 40), "viridis"),
                (1, abs(values-reference), np.linspace(0, max(maximum, 1e-12), 40), "inferno")]:
            im = axes[i, j].contourf(grid[:, :, 10, 0], grid[:, :, 10, 1], data, levels=levels, cmap=cmap)
            fig.colorbar(im, ax=axes[i, j])
            axes[i, j].set(xlabel="x", ylabel="y", title=label if i == 0 else "Absolute error")
    fig.suptitle(f"Less-smooth Poisson, z={grid[0, 0, 10, 2]:.4f} ({kind})")
    fig.tight_layout()
    fig.savefig(output / "comparison.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4))
    for name, history in histories.items():
        ax.semilogy(history[:, 0], history[:, 1], label=name)
    ax.set(xlabel="Epoch", ylabel="PINN loss", title=f"Less-smooth benchmark ({kind})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "convergence.png", dpi=160)
    plt.close(fig)


def main(smoke=False):
    n_f, n_face, epochs = (32, 8, 3) if smoke else (2048, 256, 20000)
    plan = memory_plan(n_f, 6 * n_face)
    print(json.dumps(plan), flush=True)
    require_memory(plan)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(42)
    np.random.seed(42)
    root = Path(__file__).resolve().parents[2]
    kind = "smoke" if smoke else "run"
    output = create_output_directory(root / "Results/exp_11_3d_nonsmooth_pinn_vs_pielm", prefix=kind)
    print(f"Output: {output.relative_to(root).as_posix()}", flush=True)
    start = time.perf_counter()
    X_f = generate_points("sobol", 3, n_f, seed=42)
    X_b = generate_boundary_points("sobol", 3, n_face, seed=43)
    sampling_sobol = time.perf_counter() - start
    check_boundary(X_b, n_face)
    start = time.perf_counter()
    X_d = generate_points("delaunay", 3, n_f, seed=42)
    sampling_delaunay = time.perf_counter() - start
    X_f_saved, X_b_saved, X_d_saved = X_f.clone(), X_b.clone(), X_d.clone()
    np.savez(output / "points.npz", X_f=X_f.numpy(), X_b=X_b.numpy(), X_d=X_d.numpy())
    torch.manual_seed(42)
    model = PINN3D()
    initial_state = {k: v.clone() for k, v in model.state_dict().items()}
    torch.save(initial_state, output / "initial_state.pt")
    torch.manual_seed(42)
    n_star = n_f + len(X_b)
    W = torch.randn(n_star, 3) * math.pi
    b = torch.randn(n_star, 1) * math.pi
    metadata = {
        "status": "running", "kind": kind, "problem": "nonsmooth", "dimension": 3,
        "seed": 42, "boundary_seed": 43, "dtype": "torch.float32", "device": "cpu", "threads": 2,
        "N_f": n_f, "N_b_per_face": n_face, "N_b_total": len(X_b), "boundary_sampling": "sobol",
        "PINN": {"hidden_layers": [100,100,100], "epochs": epochs, "optimizer": "Adam", "lr": 0.001,
                 "activation": "tanh", "scheduler": "ReduceLROnPlateau(patience=200, factor=0.5)",
                 "loss": "mean(residual^2) + mean(boundary_error^2)"},
        "PIELM": {"N_star": n_star, "scale": math.pi, "driver": "gelsd", "rcond": "torch default",
                  "loss": "unweighted stacked least squares"},
        "solution": "prod(sin(pi*X)) * (1 + 8*abs(x-0.5)^3)", "boundary_values": 0,
        "regularity": "u is C2,1 but not C3; f is C0,1 but not C1 across x=0.5",
        "X_f_sha256": point_hash(X_f), "X_b_sha256": point_hash(X_b), "X_d_sha256": point_hash(X_d),
        "initial_state_sha256": state_hash(initial_state), "memory_plan": plan,
        "sampling_sobol_sec": sampling_sobol, "sampling_delaunay_sec": sampling_delaunay,
        "evaluation": "20 x 20 x 20 meshgrid, endpoints included; same order as exp_10",
        "timing": "PINN training incl. scheduler/history; PIELM H/K construction and lstsq. Excludes sampling, evaluation and IO.",
        "python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
        "scipy": scipy.__version__, "matplotlib": matplotlib.__version__}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("PINN / Sobol", flush=True)
    t_pinn, history_s = train_pinn(model, X_f, X_b, epochs, poisson_rhs_nonsmooth_3d)
    assert torch.equal(X_f, X_f_saved) and torch.equal(X_b, X_b_saved)
    torch.save(model.state_dict(), output / "pinn_sobol.pt")
    print("PIELM / shared Sobol points", flush=True)
    c, t_pielm, rank = solve_pielm(X_f, X_b, W, b, poisson_rhs_nonsmooth_3d)
    assert torch.equal(X_f, X_f_saved) and torch.equal(X_b, X_b_saved)
    torch.save({"W": W, "b": b, "c": c}, output / "pielm_sobol.pt")
    model_d = PINN3D()
    model_d.load_state_dict(initial_state)
    assert all(torch.equal(model_d.state_dict()[k], v) for k, v in initial_state.items())
    metadata["delaunay_initial_state_sha256"] = state_hash(model_d.state_dict())
    print("PINN / Delaunay", flush=True)
    t_d, history_d = train_pinn(model_d, X_d, X_b, epochs, poisson_rhs_nonsmooth_3d)
    assert torch.equal(X_d, X_d_saved) and torch.equal(X_b, X_b_saved)
    torch.save(model_d.state_dict(), output / "pinn_delaunay.pt")
    for name, values in [("sobol", history_s), ("delaunay", history_d)]:
        np.savetxt(output / f"loss_{name}.csv", values, delimiter=",", header="epoch,loss,loss_f,loss_b", comments="")
    grid = np.meshgrid(*([np.linspace(0, 1, 20)] * 3))
    XYZ = torch.tensor(np.stack([v.ravel() for v in grid], axis=1), dtype=torch.float32)
    exact = exact_nonsmooth_3d(XYZ.double()).numpy().ravel()
    with torch.no_grad():
        predictions = {"PINN / Sobol": model(XYZ).numpy().ravel(),
                       "PIELM / Sobol": predict_pielm(XYZ, W, b, c).numpy().ravel(),
                       "PINN / Delaunay": model_d(XYZ).numpy().ravel()}
    np.savez(output / "predictions.npz", XYZ=XYZ.numpy(), exact=exact,
             pinn=predictions["PINN / Sobol"], pielm=predictions["PIELM / Sobol"], delaunay=predictions["PINN / Delaunay"])
    rows = []
    for (method, sampling), prediction, elapsed in zip(
            [("PINN","sobol"),("PIELM","sobol"),("PINN","delaunay")], predictions.values(), [t_pinn,t_pielm,t_d]):
        rows.append(dict(method=method, sampling=sampling, N_f=n_f, N_b=len(X_b),
                         relative_L2=relative_l2(prediction, exact), time_sec=elapsed))
    write_metrics(output / "metrics.csv", rows)
    save_plots(output, XYZ, exact, predictions, {"Sobol":history_s,"Delaunay":history_d}, kind)
    line = torch.tensor(np.column_stack([np.linspace(0,1,401), np.full(401,0.5), np.full(401,0.5)]), dtype=torch.float64)
    fig, ax = plt.subplots(figsize=(7,4))
    ax.plot(line[:,0], poisson_rhs_nonsmooth_3d(line).ravel())
    ax.axvline(0.5, color="gray", linestyle=":")
    ax.set(xlabel="x", ylabel="f(x, 0.5, 0.5)", title="Continuous RHS with a kink at x=0.5")
    fig.tight_layout(); fig.savefig(output / "rhs_profile.png", dpi=160); plt.close(fig)
    metadata.update(status="completed", shared_points_verified=True, same_initial_weights_verified=True)
    metadata["PIELM"]["rank"] = rank
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(rows), flush=True)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    main(args.smoke)
