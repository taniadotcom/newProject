from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import argparse
import csv
import json
import time
import torch
import numpy as np
import matplotlib.pyplot as plt

from core.cnn_poisson import MODES,make_dataset,PoissonDilatedCNN,sample_relative_l2
from core.poisson_solvers import relative_l2,write_metrics,point_hash
from core.experiment_utils import start_run,complete_run


def validation_falling(history):
    if len(history)<20:return False
    val=np.asarray(history)[:,2]
    return bool(val[-10:].mean()<0.98*val[-20:-10].mean() and val.argmin()+1>len(val)-10)


def main(smoke=False):
    root=Path(__file__).resolve().parents[2]
    baseline=root/"Results/exp_18_cnn_poisson/run_001"
    with (baseline/"losses.csv").open(encoding="utf-8",newline="") as f:
        baseline_loss=list(csv.DictReader(f))
    baseline_val=min(float(r["val_MSE_normalized"]) for r in baseline_loss)
    out,meta=start_run("exp_18_cnn_dilated",smoke,dict(
        dimension=2,dtype="torch.float32",dataset_seed=42,split_seed=43,model_seed=44,shuffle_seed=45,
        n_train=256,n_val=64,n_test=64,dataset_size=384,grid_size=32,modes=MODES,
        architecture="Conv(1,32,3)-Tanh-Conv(32,32,3,d=2)-Tanh-Conv(32,32,3,d=4)-Tanh-Conv(32,32,3,d=8)-Tanh-Conv(32,1,3)",
        activation="tanh",receptive_field=33,baseline_receptive_field=7,
        batch_size=32,optimizer="Adam",lr=0.001,weight_decay=0.0,
        initial_epochs=2 if smoke else 100,max_epochs=2 if smoke else 200,
        continuation_rule="At epoch 100: last 10 validation losses average at least 2% below previous 10, and best epoch in last 10",
        selection="Lowest validation MSE checkpoint; architecture needs at least 10% lower validation MSE than baseline",
        normalization="Global RMS of training f and u only",loss="Normalized solution MSE",
        metric="Mean per-test-sample Relative L2; global Relative L2 also recorded",
        baseline_source=baseline.relative_to(root).as_posix(),baseline_best_validation_MSE=baseline_val,
        timing="Offline training includes validation and checkpoint selection; inference is one held-out batch, prediction only",
        scope="One architecture comparison, one seed. Width and depth also change; not an isolated receptive-field ablation."))
    data=make_dataset()
    f,u=data["f"],data["u"]
    train,val,test=data["train"],data["val"],data["test"]
    scale_f=f[train].square().mean().sqrt();scale_u=u[train].square().mean().sqrt()
    with np.load(baseline/"dataset.npz",allow_pickle=False) as saved:
        for key in ("coeff","train","val","test"):
            np.testing.assert_array_equal(saved[key],data[key].numpy())
    old=torch.load(baseline/"cnn.pt",map_location="cpu",weights_only=True)
    assert torch.equal(scale_f,old["scale_f"]) and torch.equal(scale_u,old["scale_u"])
    torch.manual_seed(44);model=PoissonDilatedCNN()
    opt=torch.optim.Adam(model.parameters(),lr=0.001)
    gen=torch.Generator().manual_seed(45)
    best=float("inf");history=[];best_state=None;best_epoch=0;continued=False
    start=time.perf_counter()
    for epoch in range(2 if smoke else 200):
        model.train();total=0.0
        ids=train[torch.randperm(len(train),generator=gen)]
        for batch in ids.split(32):
            opt.zero_grad()
            loss=(model(f[batch]/scale_f)-u[batch]/scale_u).square().mean()
            if not torch.isfinite(loss):raise FloatingPointError("Non-finite CNN loss")
            loss.backward();opt.step();total+=loss.item()*len(batch)
        model.eval()
        with torch.no_grad():val_loss=float((model(f[val]/scale_f)-u[val]/scale_u).square().mean())
        if not np.isfinite(val_loss):raise FloatingPointError("Non-finite validation loss")
        history.append([epoch+1,total/len(train),val_loss])
        if val_loss<best:
            best=val_loss;best_epoch=epoch+1
            best_state={k:v.clone() for k,v in model.state_dict().items()}
        if epoch+1==100:
            continued=validation_falling(history)
            meta.update(best_validation_at100=best,best_epoch_at100=best_epoch,
                        validation_mean81_90=float(np.mean(history[-20:-10],axis=0)[2]),
                        validation_mean91_100=float(np.mean(history[-10:],axis=0)[2]),
                        validation_falling_at100=continued)
            if not continued:break
    elapsed=time.perf_counter()-start
    selected="dilated" if not smoke and best<=0.9*baseline_val else "baseline"
    meta.update(epochs=len(history),best_epoch=best_epoch,best_validation_MSE=best,
                continued_to200=continued,validation_falling_at_end=validation_falling(history),
                selected_architecture=selected,selection_usable=not smoke,test_used_for_selection=False,
                parameters=sum(p.numel() for p in model.parameters()))
    (out/"metadata.json").write_text(json.dumps(meta,indent=2),encoding="utf-8")
    model.load_state_dict(best_state);model.eval()
    start=time.perf_counter()
    with torch.no_grad():pred=model(f[test]/scale_f)*scale_u
    inference=time.perf_counter()-start
    errors=sample_relative_l2(pred,u[test])
    row=dict(method="CNN",architecture="dilated",relative_L2=float(errors.mean()),
        global_relative_L2=relative_l2(pred.numpy(),u[test].numpy()),test_MSE=float((pred-u[test]).square().mean()),
        time_sec=elapsed,inference_time_sec=inference,inference_per_sample_sec=inference/len(test),
        epochs=len(history),n_test=len(test),best_epoch=best_epoch,best_validation_MSE=best)
    write_metrics(out/"metrics.csv",[row])
    write_metrics(out/"test_metrics.csv",[dict(sample_id=int(i),relative_L2=float(e)) for i,e in zip(test,errors)])
    np.savetxt(out/"losses.csv",history,delimiter=",",header="epoch,train_MSE_normalized,val_MSE_normalized",comments="")
    np.savez(out/"dataset.npz",coeff=data["coeff"].numpy(),train=train.numpy(),val=val.numpy(),test=test.numpy())
    np.savez(out/"predictions.npz",sample_ids=test.numpy(),exact=u[test].numpy(),CNN=pred.numpy())
    torch.save(dict(state_dict=best_state,scale_f=scale_f,scale_u=scale_u,grid_size=32,channels=32,
                    architecture="dilated"),out/"cnn_dilated.pt")
    with (baseline/"metrics.csv").open(encoding="utf-8",newline="") as f:base=next(csv.DictReader(f))
    fields=["relative_L2","global_relative_L2","test_MSE","time_sec","inference_time_sec",
            "inference_per_sample_sec","epochs","best_epoch"]
    comparison=[dict(architecture="baseline",**{k:base[k] for k in fields},best_validation_MSE=baseline_val,
                     receptive_field=7,parameters=sum(v.numel() for k,v in old["state_dict"].items() if k!="mask"),
                     selected_by_validation=selected=="baseline"),
                dict(architecture="dilated",**{k:row[k] for k in fields},best_validation_MSE=best,
                     receptive_field=33,parameters=meta["parameters"],selected_by_validation=selected=="dilated")]
    write_metrics(out/"cnn_architecture_comparison.csv",comparison)
    fig,axes=plt.subplots(2,2,figsize=(9,7))
    for ax,field,label in zip(axes.flat,["relative_L2","best_validation_MSE","time_sec","inference_time_sec"],
            ["Mean test Relative L2","Best validation MSE (normalized)","Offline training time (s)","Test batch inference time (s)"]):
        ax.bar(["Baseline (RF 7)","Dilated (RF 33)"],[float(r[field]) for r in comparison])
        ax.set(ylabel=label)
    fig.tight_layout();fig.savefig(out/"cnn_architecture_comparison.png",dpi=140);plt.close(fig)
    h=np.asarray(history);fig,ax=plt.subplots(figsize=(8,4))
    ax.semilogy(h[:,0],h[:,1],label="Dilated train")
    ax.semilogy(h[:,0],h[:,2],label="Dilated validation")
    ax.semilogy([int(float(r["epoch"])) for r in baseline_loss],
                [float(r["val_MSE_normalized"]) for r in baseline_loss],label="Baseline validation")
    ax.set(xlabel="Epoch",ylabel="Normalized MSE");ax.legend();fig.tight_layout()
    fig.savefig(out/"convergence.png",dpi=140);plt.close(fig)
    with np.load(baseline/"predictions.npz",allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["sample_ids"],test.numpy())
        np.testing.assert_array_equal(saved["exact"],u[test].numpy())
        shown=pred.numpy() if selected=="dilated" else saved["CNN"].copy()
    fig,axes=plt.subplots(3,3,figsize=(9,8))
    for i in range(3):
        exact=u[test[i],0].numpy();p=shown[i,0];lo=min(exact.min(),p.min());hi=max(exact.max(),p.max())
        for j,(v,title) in enumerate([(exact,"Exact"),(p,"Prediction"),(abs(p-exact),"Absolute error")]):
            im=axes[i,j].imshow(v,origin="lower",extent=[0,1,0,1],cmap="inferno" if j==2 else "viridis",
                                vmin=0 if j==2 else lo,vmax=None if j==2 else hi)
            fig.colorbar(im,ax=axes[i,j]);axes[i,j].set(title=f"Test {int(test[i])}: {title}",xlabel="x",ylabel="y")
    fig.suptitle(f"{selected.capitalize()} CNN, selected by validation")
    fig.tight_layout();fig.savefig(out/"best_cnn_examples.png",dpi=140);plt.close(fig)
    meta.update(normalization_f=float(scale_f),normalization_u=float(scale_u),
                coefficients_sha256=point_hash(data["coeff"]),train_ids=train.tolist(),val_ids=val.tolist(),test_ids=test.tolist(),
                test_improvement_percent=100*(float(base["relative_L2"])-row["relative_L2"])/float(base["relative_L2"]),
                baseline_timing="Previously recorded baseline run; no baseline retraining")
    complete_run(out,meta)
    return out


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--smoke",action="store_true")
    main(parser.parse_args().smoke)
