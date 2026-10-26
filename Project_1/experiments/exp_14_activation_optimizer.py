from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt

from core.model import PINN
from core.utils import generate_points, generate_boundary_points
from core.poisson_problems import exact_sine_nd, poisson_rhs_sine_nd
from core.poisson_solvers import point_hash, state_hash, relative_l2, write_metrics
from core.poisson_extensions import fit_pinn
from core.experiment_utils import start_run, complete_run, evaluation_grid, plot_fields


def main(smoke=False):
    n_f, n_face, epochs = (16, 4, 2) if smoke else (512, 64, 3000)
    variants = [("Adam", "tanh", 0.0), ("AdamW", "tanh", 1e-4),
                ("AdamW", "gelu", 1e-4), ("AdamW", "silu", 1e-4)]
    out, meta = start_run("exp_14_activation_optimizer", smoke, dict(
        problem="smooth sine Poisson", dimension=3, dtype="torch.float32", N_f=n_f,
        N_b_per_face=n_face, N_b_total=6*n_face, epochs=epochs, lr=0.001,
        hidden_layers=[100,100,100], boundary_seed=43, evaluation_grid=[20,20,20],
        scheduler="ReduceLROnPlateau(patience=200,factor=0.5)",
        weight_decay_note="1e-4 is a fixed small study setting, not a tuned optimum; Adam uses zero decay.",
        timing="Training includes backward, optimizer, scheduler and history; excludes setup/evaluation/IO.",
        scope="Separate 3000-epoch study; not a replacement for exp_10's 20000-epoch baseline."))
    X_f = generate_points("sobol", 3, n_f, seed=42)
    X_b = generate_boundary_points("sobol", 3, n_face, seed=43)
    saved_f, saved_b = X_f.clone(), X_b.clone()
    X = evaluation_grid(3, 20)
    exact = exact_sine_nd(X.double()).numpy().ravel()
    torch.manual_seed(42)
    initial = PINN(3, (100,100,100)).state_dict()
    torch.save(initial, out / "initial_state.pt")
    np.savez(out / "points.npz", X_f=X_f.numpy(), X_b=X_b.numpy())
    predictions, metrics, losses = {}, [], []
    fig, ax = plt.subplots(figsize=(8,4))
    hashes = {}
    for opt, act, decay in variants:
        model = PINN(3, (100,100,100), activation=act)
        model.load_state_dict(initial)
        name = f"{opt}_{act}"
        hashes[name] = state_hash(model.state_dict())
        elapsed, history = fit_pinn(model, X_f, X_b, epochs, poisson_rhs_sine_nd, opt, decay)
        assert torch.equal(X_f, saved_f) and torch.equal(X_b, saved_b)
        with torch.no_grad():
            pred = model(X).numpy().ravel()
        predictions[name] = pred
        torch.save(model.state_dict(), out / f"{name}.pt")
        metrics.append(dict(optimizer=opt, activation=act, weight_decay=decay,
            relative_L2=relative_l2(pred, exact), time_sec=elapsed, epochs=epochs, N_f=n_f, N_b=len(X_b)))
        losses.extend(dict(optimizer=opt, activation=act, epoch=int(row[0]), loss=row[1],
                           loss_f=row[2], loss_b=row[3], lr=row[4]) for row in history)
        ax.semilogy(history[:,0], history[:,1], label=f"{opt}/{act}")
    assert len(set(hashes.values())) == 1
    write_metrics(out / "metrics.csv", metrics)
    write_metrics(out / "losses.csv", losses)
    np.savez(out / "predictions.npz", X_eval=X.numpy(), exact=exact, **predictions)
    ax.set(xlabel="Epoch", ylabel="PINN loss"); ax.legend(); fig.tight_layout()
    fig.savefig(out / "convergence.png", dpi=140); plt.close(fig)
    plot_fields(out, X, exact, predictions, 20)
    meta.update(X_f_sha256=point_hash(X_f), X_b_sha256=point_hash(X_b),
                initial_state_sha256=hashes, same_initial_weights=True, shared_points_verified=True,
                variants=[dict(optimizer=o,activation=a,weight_decay=w) for o,a,w in variants])
    complete_run(out, meta)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    main(parser.parse_args().smoke)
