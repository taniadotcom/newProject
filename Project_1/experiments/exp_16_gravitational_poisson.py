from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import argparse
import math
import torch
import numpy as np
import matplotlib.pyplot as plt

from core.model import PINN3D
from core.utils import generate_points, generate_boundary_points
from core.poisson_extensions import gravitational_potential, gravitational_rhs, fit_pinn
from core.poisson_solvers import point_hash, solve_pielm, predict_pielm, relative_l2, write_metrics
from core.experiment_utils import start_run, complete_run, evaluation_grid, plot_fields


def main(smoke=False):
    n_f,n_face,epochs=(16,4,2) if smoke else (512,64,3000)
    out,meta=start_run("exp_16_gravitational_poisson",smoke,dict(
        dimension=3,dtype="torch.float32",N_f=n_f,N_b_per_face=n_face,N_b_total=6*n_face,
        epochs=epochs,boundary_seed=43,G=1.0,units="nondimensional",hidden_layers=[100,100,100],
        activation="tanh",optimizer="Adam",lr=0.001,weight_decay=0.0,
        scheduler="ReduceLROnPlateau(patience=200,factor=0.5)",
        PDE="Delta Phi = 4*pi*G*rho; code solves -Delta Phi = -4*pi*G*rho",
        exact="Phi=-prod(sin(pi*X))",density="rho=3*pi/(4*G)*prod(sin(pi*X))",
        boundary_values=0,scope="Manufactured finite-box Dirichlet benchmark, not an isolated-space mass model or N-body simulation.",
        evaluation_grid=[20,20,20],driver="gelsd",rcond="torch default",scale=math.pi,
        timing="PINN training and PIELM H/K+lstsq; excludes sampling, initialization, evaluation, IO"))
    X_f=generate_points("sobol",3,n_f,seed=42)
    X_b=generate_boundary_points("sobol",3,n_face,seed=43)
    saved_f,saved_b=X_f.clone(),X_b.clone()
    X=evaluation_grid(3,20)
    exact=gravitational_potential(X.double()).numpy().ravel()
    torch.manual_seed(42);model=PINN3D()
    elapsed,history=fit_pinn(model,X_f,X_b,epochs,gravitational_rhs)
    with torch.no_grad():pred=model(X).numpy().ravel()
    torch.save(model.state_dict(),out/"pinn.pt")
    torch.manual_seed(42);n_star=n_f+len(X_b)
    W,b=torch.randn(n_star,3)*math.pi,torch.randn(n_star,1)*math.pi
    c,time_p,rank=solve_pielm(X_f,X_b,W,b,gravitational_rhs)
    pred_p=predict_pielm(X,W,b,c).numpy().ravel()
    assert torch.equal(X_f,saved_f) and torch.equal(X_b,saved_b)
    rows=[dict(method=m,relative_L2=relative_l2(v,exact),time_sec=t,N_f=n_f,N_b=len(X_b),
               epochs=epochs if m=="PINN" else 0) for m,v,t in [("PINN",pred,elapsed),("PIELM",pred_p,time_p)]]
    write_metrics(out/"metrics.csv",rows)
    np.savetxt(out/"losses.csv",history,delimiter=",",header="epoch,loss,loss_f,loss_b,lr",comments="")
    np.savez(out/"points.npz",X_f=X_f.numpy(),X_b=X_b.numpy())
    np.savez(out/"predictions.npz",X_eval=X.numpy(),exact=exact,PINN=pred,PIELM=pred_p)
    torch.save(dict(W=W,b=b,c=c),out/"pielm.pt")
    plot_fields(out,X,exact,{"PINN":pred,"PIELM":pred_p},20)
    fig,ax=plt.subplots(figsize=(7,3));ax.semilogy(history[:,0],history[:,1])
    ax.set(xlabel="Epoch",ylabel="PINN loss");fig.tight_layout()
    fig.savefig(out/"convergence.png",dpi=140);plt.close(fig)
    meta.update(X_f_sha256=point_hash(X_f),X_b_sha256=point_hash(X_b),shared_points_verified=True,
                PIELM=dict(N_star=n_star,rank=rank))
    complete_run(out,meta)
    return out


if __name__ == "__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--smoke",action="store_true")
    main(parser.parse_args().smoke)
