from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import math
import platform
import time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core.model import PINN
from core.utils import generate_points, generate_boundary_points, create_output_directory, latest_completed
from core.poisson_problems import exact_sine_nd, poisson_rhs_sine_nd
from core.poisson_solvers import (train_pinn, solve_pielm, predict_pielm, relative_l2,
    point_hash, write_metrics, memory_plan, require_memory, available_memory)


def load_selection(comparison):
    decision = json.loads((comparison / "selection.json").read_text(encoding="utf-8"))
    if decision["kind"] != "run" or not decision["usable_for_full_extension"]:
        raise ValueError("A full extension requires a completed full 3D comparison")
    if (decision["method"],decision["sampling"]) not in [("PINN","sobol"),("PINN","delaunay"),("PIELM","sobol")]:
        raise ValueError("Unknown selected method")
    return decision


def save_scaling(output, row, kind):
    entries = {row["dimension"]: row}
    sources = {str(row["dimension"]): output.name}
    for path in sorted(output.parent.glob(f"{kind}_*")):
        if path == output or not (path / "metadata.json").exists():
            continue
        meta = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        if meta["status"] != "completed":
            continue
        previous = meta["metrics"]
        if all(previous[key] == row[key] for key in ("method","sampling","seed","N_f","N_b_per_face","epochs")):
            entries[previous["dimension"]] = previous
            sources[str(previous["dimension"])] = path.name
    entries[row["dimension"]] = row
    sources[str(row["dimension"])] = output.name
    values = [entries[d] for d in sorted(entries)]
    fig, axes = plt.subplots(1,2,figsize=(10,4))
    for ax, metric, label in zip(axes,["relative_L2","training_time"],["Relative L2","Training/solve time (s)"]):
        ax.plot([r["dimension"] for r in values],[r[metric] for r in values],"o-")
        ax.set(xlabel="Dimension",ylabel=label,xticks=[r["dimension"] for r in values])
    fig.suptitle(f"{row['method']} / {row['sampling']} ({kind})")
    fig.tight_layout(); fig.savefig(output / "scaling.png",dpi=160); plt.close(fig)
    return sources


def main(dim, smoke=False, comparison=None, smoke_method="pielm"):
    if dim < 2:
        raise ValueError("dim must be at least 2")
    root = Path(__file__).resolve().parents[2]
    if smoke:
        method, sampling = {"pielm":("PIELM","sobol"),"pinn":("PINN","sobol"),
                            "delaunay":("PINN","delaunay")}[smoke_method]
        decision = {"kind":"smoke", "interpretation":"Functional test only, no method selection."}
    else:
        comparison = Path(comparison).resolve() if comparison else latest_completed(root / "Results/exp_12_compare_smooth_nonsmooth")
        decision = load_selection(comparison)
        method,sampling = decision["method"],decision["sampling"]
    n_f,n_face,epochs = (32,8,3) if smoke else (2048,256,20000)
    n_eval = 256 if smoke else 16384
    plan = memory_plan(n_f,2*dim*n_face)
    if method == "PIELM":
        print(json.dumps(plan),flush=True)
        require_memory(plan)
    if sampling == "delaunay" and (dim > 5 or (not smoke and available_memory() < 2*1024**3)):
        raise MemoryError("Delaunay extension limited to dim<=5 and at least 2 GiB free RAM for full runs")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(42)
    np.random.seed(42)
    kind = "smoke" if smoke else "run"
    output = create_output_directory(root / "Results/exp_13_nd_best_method",prefix=kind)
    print(f"Output: {output.relative_to(root).as_posix()}",flush=True)
    metadata = {"status":"running","kind":kind,"dimension":dim,"selection":decision,
                "selection_source":comparison.relative_to(root).as_posix() if comparison else None,
                "dtype":"torch.float32","seed":42,"boundary_seed":43,"evaluation_seed":2026,
                "N_eval":n_eval,"device":"cpu","threads":2,"memory_plan":plan,
                "PDE":"-Delta u = dim*pi^2*prod(sin(pi*X))","boundary_values":0,
                "method":method,"sampling":sampling,"activation":"tanh",
                "python":platform.python_version(),"torch":torch.__version__,"numpy":np.__version__,
                "timing":"training_time excludes sampling/evaluation/IO; evaluation_time measures prediction only"}
    if method == "PIELM":
        metadata["PIELM"] = {"N_star":n_f+2*dim*n_face,"scale":math.pi,"driver":"gelsd","rcond":"torch default"}
    else:
        metadata["PINN"] = {"hidden_layers":[100,100,100],"optimizer":"Adam","lr":0.001,"epochs":epochs,
                            "scheduler":"ReduceLROnPlateau(patience=200, factor=0.5)"}
    (output / "metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    start = time.perf_counter()
    X_f = generate_points(sampling,dim,n_f,seed=42)
    X_b = generate_boundary_points("sobol",dim,n_face,seed=43)
    metadata["sampling_time"] = time.perf_counter()-start
    assert X_b.shape == (2*dim*n_face,dim)
    assert torch.all(((X_b == 0) | (X_b == 1)).any(dim=1))
    saved_f,saved_b = X_f.clone(),X_b.clone()
    np.savez(output / "points.npz",X_f=X_f.numpy(),X_b=X_b.numpy())
    metadata.update(X_f_sha256=point_hash(X_f),X_b_sha256=point_hash(X_b))
    torch.manual_seed(42)
    n_star = n_f+len(X_b)
    rank = None
    if method == "PIELM":
        W,b = torch.randn(n_star,dim)*math.pi,torch.randn(n_star,1)*math.pi
        c,elapsed,rank = solve_pielm(X_f,X_b,W,b,poisson_rhs_sine_nd)
        torch.save({"W":W,"b":b,"c":c,"dimension":dim},output / "model.pt")
    else:
        model = PINN(dim=dim,hidden_layers=(100,100,100))
        elapsed,history = train_pinn(model,X_f,X_b,epochs,poisson_rhs_sine_nd)
        torch.save({"state_dict":model.state_dict(),"dimension":dim,"hidden_layers":(100,100,100)},output / "model.pt")
        np.savetxt(output / "convergence.csv",history,delimiter=",",header="epoch,loss,loss_f,loss_b",comments="")
    assert torch.equal(X_f,saved_f) and torch.equal(X_b,saved_b)
    X_eval = generate_points("sobol",dim,n_eval,seed=2026)
    exact = exact_sine_nd(X_eval.double()).numpy().ravel()
    start = time.perf_counter()
    with torch.no_grad():
        prediction = (predict_pielm(X_eval,W,b,c) if method == "PIELM" else model(X_eval)).numpy().ravel()
    evaluation_time = time.perf_counter()-start
    np.savez(output / "predictions.npz",X_eval=X_eval.numpy(),exact=exact,prediction=prediction)
    row = dict(dimension=dim,method=method,sampling=sampling,N_f=n_f,N_b_per_face=n_face,N_b_total=len(X_b),
               epochs=0 if method == "PIELM" else epochs,N_star=n_star if method == "PIELM" else 0,
               relative_L2=relative_l2(prediction,exact),training_time=elapsed,evaluation_time=evaluation_time,seed=42)
    write_metrics(output / "metrics.csv",[row])
    metadata.update(status="completed",metrics=row,rank=rank,X_eval_sha256=point_hash(X_eval),points_unchanged=True)
    metadata["scaling_sources"] = save_scaling(output,row,kind)
    (output / "metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    print(json.dumps(row),flush=True)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim",type=int,required=True)
    parser.add_argument("--smoke",action="store_true")
    parser.add_argument("--comparison",type=Path)
    parser.add_argument("--smoke-method",choices=["pielm","pinn","delaunay"],default="pielm")
    args = parser.parse_args()
    main(args.dim,args.smoke,args.comparison,args.smoke_method)
