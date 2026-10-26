from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt

from core.model import PINN
from core.utils import generate_points, generate_boundary_points
from core.poisson_problems import exact_sine_nd
from core.poisson_extensions import nonlinear_rhs, fit_pinn
from core.poisson_solvers import point_hash, relative_l2, write_metrics
from core.experiment_utils import start_run, complete_run, evaluation_grid, plot_fields


def main(smoke=False):
    n_f,n_face,epochs=(16,4,2) if smoke else (512,64,3000)
    out,meta=start_run("exp_17_nonlinear_pde",smoke,dict(
        dimension=2,dtype="torch.float32",N_f=n_f,N_b_per_face=n_face,N_b_total=4*n_face,
        epochs=epochs,boundary_seed=43,hidden_layers=[50,50],activation="tanh",optimizer="Adam",
        lr=0.001,weight_decay=0.0,scheduler="ReduceLROnPlateau(patience=200,factor=0.5)",
        PDE="-Delta u + u^3 = f",exact="prod(sin(pi*X))",RHS="2*pi^2*u_exact + u_exact^3",
        boundary_values=0,evaluation_grid=[64,64],method="PINN",
        scope="Semilinear manufactured benchmark; no linear PIELM claim.",
        timing="Training includes backward/optimizer/scheduler/history; excludes setup/evaluation/IO"))
    X_f=generate_points("sobol",2,n_f,seed=42)
    X_b=generate_boundary_points("sobol",2,n_face,seed=43)
    model=PINN()
    elapsed,history=fit_pinn(model,X_f,X_b,epochs,nonlinear_rhs,nonlinear=True)
    X=evaluation_grid(2,64);exact=exact_sine_nd(X.double()).numpy().ravel()
    with torch.no_grad():pred=model(X).numpy().ravel()
    write_metrics(out/"metrics.csv",[dict(method="PINN",relative_L2=relative_l2(pred,exact),
        time_sec=elapsed,epochs=epochs,N_f=n_f,N_b=len(X_b))])
    np.savetxt(out/"losses.csv",history,delimiter=",",header="epoch,loss,loss_f,loss_b,lr",comments="")
    np.savez(out/"points.npz",X_f=X_f.numpy(),X_b=X_b.numpy())
    np.savez(out/"predictions.npz",X_eval=X.numpy(),exact=exact,PINN=pred)
    torch.save(model.state_dict(),out/"pinn.pt")
    plot_fields(out,X,exact,{"PINN":pred},64)
    fig,ax=plt.subplots(figsize=(7,3));ax.semilogy(history[:,0],history[:,1])
    ax.set(xlabel="Epoch",ylabel="PINN loss");fig.tight_layout()
    fig.savefig(out/"convergence.png",dpi=140);plt.close(fig)
    meta.update(X_f_sha256=point_hash(X_f),X_b_sha256=point_hash(X_b))
    complete_run(out,meta)
    return out


if __name__ == "__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--smoke",action="store_true")
    main(parser.parse_args().smoke)
