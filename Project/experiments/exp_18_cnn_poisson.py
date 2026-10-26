from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import argparse
import time
import torch
import numpy as np
import matplotlib.pyplot as plt

from core.cnn_poisson import MODES, make_dataset, PoissonCNN, sample_relative_l2
from core.poisson_solvers import relative_l2, write_metrics, point_hash
from core.experiment_utils import start_run, complete_run


def main(smoke=False):
    n_train,n_val,n_test,epochs=(8,4,4,2) if smoke else (256,64,64,100)
    out,meta=start_run("exp_18_cnn_poisson",smoke,dict(
        dimension=2,dtype="torch.float32",dataset_seed=42,split_seed=43,model_seed=44,shuffle_seed=45,
        n_train=n_train,n_val=n_val,n_test=n_test,dataset_size=n_train+n_val+n_test,grid_size=32,
        modes=MODES,coefficient_distribution="independent standard normal; float64 synthesis, float32 grids",
        architecture="Conv(1,16,3)-Tanh-Conv(16,16,3)-Tanh-Conv(16,1,3), padding=1, zero-boundary mask",
        epochs=epochs,batch_size=32,optimizer="Adam",lr=0.001,weight_decay=0.0,
        selection="Lowest validation MSE; test set used only after checkpoint selection",
        normalization="Global RMS of training f and u only",
        metric="Mean per-test-sample Relative L2; global Relative L2 also recorded",
        timing="Offline training incl validation and checkpoint selection; inference is the whole held-out batch, prediction only",
        scope="Supervised surrogate for four sine modes; not compared to single-instance PINN training by inference time."))
    data=make_dataset(n_train,n_val,n_test)
    f,u=data["f"],data["u"]
    train,val,test=data["train"],data["val"],data["test"]
    scale_f=f[train].square().mean().sqrt();scale_u=u[train].square().mean().sqrt()
    torch.manual_seed(44);model=PoissonCNN()
    opt=torch.optim.Adam(model.parameters(),lr=0.001)
    gen=torch.Generator().manual_seed(45)
    best=float("inf");history=[];best_state=None;best_epoch=0
    start=time.perf_counter()
    for epoch in range(epochs):
        model.train();total=0.0
        ids=train[torch.randperm(len(train),generator=gen)]
        for batch in ids.split(32):
            opt.zero_grad()
            loss=(model(f[batch]/scale_f)-u[batch]/scale_u).square().mean()
            if not torch.isfinite(loss):raise FloatingPointError("Non-finite CNN loss")
            loss.backward();opt.step();total+=loss.item()*len(batch)
        model.eval()
        with torch.no_grad():val_loss=float((model(f[val]/scale_f)-u[val]/scale_u).square().mean())
        history.append([epoch+1,total/len(train),val_loss])
        if val_loss<best:
            best=val_loss;best_epoch=epoch+1
            best_state={k:v.clone() for k,v in model.state_dict().items()}
    elapsed=time.perf_counter()-start
    model.load_state_dict(best_state);model.eval()
    start=time.perf_counter()
    with torch.no_grad():pred=model(f[test]/scale_f)*scale_u
    inference=time.perf_counter()-start
    errors=sample_relative_l2(pred,u[test])
    row=dict(method="CNN",relative_L2=float(errors.mean()),global_relative_L2=relative_l2(pred.numpy(),u[test].numpy()),
        test_MSE=float((pred-u[test]).square().mean()),time_sec=elapsed,inference_time_sec=inference,
        inference_per_sample_sec=inference/len(test),epochs=epochs,n_test=len(test),best_epoch=best_epoch)
    write_metrics(out/"metrics.csv",[row])
    write_metrics(out/"test_metrics.csv",[dict(sample_id=int(i),relative_L2=float(e)) for i,e in zip(test,errors)])
    np.savetxt(out/"losses.csv",history,delimiter=",",header="epoch,train_MSE_normalized,val_MSE_normalized",comments="")
    np.savez(out/"dataset.npz",coeff=data["coeff"].numpy(),train=train.numpy(),val=val.numpy(),test=test.numpy())
    np.savez(out/"predictions.npz",sample_ids=test.numpy(),exact=u[test].numpy(),CNN=pred.numpy())
    torch.save(dict(state_dict=best_state,scale_f=scale_f,scale_u=scale_u,grid_size=32,channels=16),out/"cnn.pt")
    fig,axes=plt.subplots(3,3,figsize=(9,8))
    for i in range(3):
        exact=u[test[i],0].numpy();p=pred[i,0].numpy();lo=min(exact.min(),p.min());hi=max(exact.max(),p.max())
        for j,(v,title) in enumerate([(exact,"Exact"),(p,"Prediction"),(abs(p-exact),"Absolute error")]):
            im=axes[i,j].imshow(v,origin="lower",extent=[0,1,0,1],cmap="inferno" if j==2 else "viridis",
                                vmin=0 if j==2 else lo,vmax=None if j==2 else hi)
            fig.colorbar(im,ax=axes[i,j]);axes[i,j].set(title=f"Test {int(test[i])}: {title}",xlabel="x",ylabel="y")
    fig.tight_layout();fig.savefig(out/"comparison.png",dpi=140);plt.close(fig)
    h=np.asarray(history);fig,ax=plt.subplots(figsize=(7,3))
    ax.semilogy(h[:,0],h[:,1],label="Train");ax.semilogy(h[:,0],h[:,2],label="Validation")
    ax.set(xlabel="Epoch",ylabel="Normalized MSE");ax.legend();fig.tight_layout()
    fig.savefig(out/"convergence.png",dpi=140);plt.close(fig)
    meta.update(best_epoch=best_epoch,normalization_f=float(scale_f),normalization_u=float(scale_u),
                coefficients_sha256=point_hash(data["coeff"]),train_ids=train.tolist(),val_ids=val.tolist(),test_ids=test.tolist())
    complete_run(out,meta)
    return out


if __name__ == "__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--smoke",action="store_true")
    main(parser.parse_args().smoke)
