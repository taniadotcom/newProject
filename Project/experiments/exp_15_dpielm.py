from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import math
import time
import torch
import numpy as np
import matplotlib.pyplot as plt

from core.utils import generate_points, generate_boundary_points
from core.poisson_problems import exact_sine_nd, poisson_rhs_sine_nd
from core.poisson_solvers import solve_pielm, predict_pielm, point_hash, relative_l2, write_metrics
from core.dpielm import build_dpielm_system, predict_dpielm, interface_jumps, subdomain_ids
from core.experiment_utils import start_run, complete_run, evaluation_grid, plot_fields


def main(smoke=False):
    n_f, n_face, n_i, n_local = (32, 4, 8, 12) if smoke else (512, 64, 64, 256)
    out, meta = start_run("exp_15_dpielm", smoke, dict(
        problem="2D sine Poisson", dimension=2, dtype="torch.float64", N_f=n_f,
        N_b_per_face=n_face, N_b_total=4*n_face, N_interface=n_i, N_star_total=2*n_local,
        boundary_seed=43, interface_seed=44, subdomains=[[[0,0.5],[0,1]], [[0.5,1],[0,1]]],
        coordinate_map="local coordinates (X-lower)/width; derivatives include inverse widths",
        interfaces="u_left=u_right and dx_u_left=dx_u_right; all rows weight 1",
        driver="gelsd", rcond="torch default", scale=math.pi,
        comparison="Same interior/boundary/evaluation points and total basis count; DPIELM adds interface rows.",
        reference="Dwivedi and Srinivasan (2020), section 6, equations 68-75; two-domain 2D Poisson adaptation.",
        timing="H/K assembly and least squares; excludes sampling/evaluation/IO"))
    X_f = generate_points("sobol", 2, n_f, seed=42, dtype=torch.float64)
    X_b = generate_boundary_points("sobol", 2, n_face, seed=43, dtype=torch.float64)
    y = generate_points("sobol", 1, n_i, seed=44, dtype=torch.float64)
    X_i = torch.cat([torch.full_like(y, 0.5), y], dim=1)
    X = evaluation_grid(2, 64, torch.float64)
    exact = exact_sine_nd(X).numpy().ravel()
    torch.manual_seed(42)
    W = torch.randn(2*n_local, 2, dtype=torch.float64) * math.pi
    b = torch.randn(2*n_local, 1, dtype=torch.float64) * math.pi
    saved_f, saved_b = X_f.clone(), X_b.clone()
    c, elapsed, rank = solve_pielm(X_f, X_b, W, b, poisson_rhs_sine_nd)
    pred_p = predict_pielm(X, W, b, c).numpy().ravel()
    torch.save(dict(W=W,b=b,c=c), out / "pielm.pt")
    W_d, b_d = W.reshape(2,n_local,2).clone(), b.reshape(2,n_local,1).clone()
    start = time.perf_counter()
    H, K = build_dpielm_system(X_f, X_b, X_i, W_d, b_d, poisson_rhs_sine_nd)
    solution = torch.linalg.lstsq(H, K, driver="gelsd")
    elapsed_d = time.perf_counter() - start
    c_d = solution.solution
    assert torch.isfinite(c_d).all()
    pred_d = predict_dpielm(X, W_d, b_d, c_d).numpy().ravel()
    assert torch.equal(X_f, saved_f) and torch.equal(X_b, saved_b)
    t = torch.linspace(0,1,257,dtype=torch.float64)
    line = torch.stack([torch.full_like(t,0.5),t],dim=1)
    jump, dx_jump = interface_jumps(line, W_d, b_d, c_d)
    metrics = [dict(method="PIELM", relative_L2=relative_l2(pred_p,exact), time_sec=elapsed,
                    N_f=n_f,N_b=len(X_b),N_star=2*n_local,rank=rank,interface_value_rms="",interface_dx_rms=""),
               dict(method="DPIELM",relative_L2=relative_l2(pred_d,exact),time_sec=elapsed_d,
                    N_f=n_f,N_b=len(X_b),N_star=2*n_local,rank=int(solution.rank),
                    interface_value_rms=float(jump.square().mean().sqrt()),
                    interface_dx_rms=float(dx_jump.square().mean().sqrt()))]
    write_metrics(out / "metrics.csv", metrics)
    np.savez(out / "points.npz",X_f=X_f.numpy(),X_b=X_b.numpy(),X_i=X_i.numpy())
    np.savez(out / "predictions.npz",X_eval=X.numpy(),exact=exact,PIELM=pred_p,DPIELM=pred_d)
    torch.save(dict(W=W_d,b=b_d,c=c_d),out / "dpielm.pt")
    plot_fields(out,X,exact,{"PIELM":pred_p,"DPIELM":pred_d},64)
    fig, axes = plt.subplots(1,2,figsize=(9,3))
    for ax, values, label in zip(axes,[jump,dx_jump],["u_left-u_right","dx_u_left-dx_u_right"]):
        ax.plot(t.numpy(),values.numpy().ravel());ax.set(xlabel="Interface y",ylabel=label)
    fig.tight_layout();fig.savefig(out/"interface.png",dpi=140);plt.close(fig)
    meta.update(X_f_sha256=point_hash(X_f),X_b_sha256=point_hash(X_b),shared_points_verified=True,
                N_f_subdomains=[int((subdomain_ids(X_f)==i).sum()) for i in (0,1)],
                H_shape=list(H.shape),matrix_residual_rms=float((H@c_d-K).square().mean().sqrt()))
    complete_run(out,meta)
    return out


if __name__ == "__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--smoke",action="store_true")
    main(parser.parse_args().smoke)
