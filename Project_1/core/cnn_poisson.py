import math
import torch
from torch import nn


MODES = ((1,1),(1,2),(2,1),(2,2))


def make_dataset(n_train=256,n_val=64,n_test=64,n_grid=32,seed=42):
    x=torch.linspace(0,1,n_grid,dtype=torch.float64)
    y,x_grid=torch.meshgrid(x,x,indexing="ij")
    basis=torch.stack([torch.sin(m*math.pi*x_grid)*torch.sin(n*math.pi*y) for m,n in MODES])
    basis[:,[0,-1],:]=0;basis[:,:,[0,-1]]=0
    factors=torch.tensor([math.pi**2*(m*m+n*n) for m,n in MODES],dtype=torch.float64)
    gen=torch.Generator().manual_seed(seed)
    coeff=torch.randn(n_train+n_val+n_test,len(MODES),generator=gen,dtype=torch.float64)
    u=torch.einsum("sm,mxy->sxy",coeff,basis).float().unsqueeze(1)
    f=torch.einsum("sm,m,mxy->sxy",coeff,factors,basis).float().unsqueeze(1)
    ids=torch.randperm(len(coeff),generator=torch.Generator().manual_seed(seed+1))
    return dict(f=f,u=u,coeff=coeff,train=ids[:n_train],val=ids[n_train:n_train+n_val],
                test=ids[n_train+n_val:],grid=x.float())


class PoissonCNN(nn.Module):
    def __init__(self,n_grid=32,channels=16):
        super().__init__()
        self.net=nn.Sequential(nn.Conv2d(1,channels,3,padding=1),nn.Tanh(),
            nn.Conv2d(channels,channels,3,padding=1),nn.Tanh(),nn.Conv2d(channels,1,3,padding=1))
        mask=torch.ones(1,1,n_grid,n_grid)
        mask[:,:,[0,-1],:]=0;mask[:,:,:,[0,-1]]=0
        self.register_buffer("mask",mask)

    def forward(self,f):
        return self.net(f)*self.mask


class PoissonDilatedCNN(nn.Module):
    def __init__(self,n_grid=32,channels=32):
        super().__init__()
        self.net=nn.Sequential(
            nn.Conv2d(1,channels,3,padding=1),nn.Tanh(),
            nn.Conv2d(channels,channels,3,padding=2,dilation=2),nn.Tanh(),
            nn.Conv2d(channels,channels,3,padding=4,dilation=4),nn.Tanh(),
            nn.Conv2d(channels,channels,3,padding=8,dilation=8),nn.Tanh(),
            nn.Conv2d(channels,1,3,padding=1))
        mask=torch.ones(1,1,n_grid,n_grid)
        mask[:,:,[0,-1],:]=0;mask[:,:,:,[0,-1]]=0
        self.register_buffer("mask",mask)

    def forward(self,f):
        return self.net(f)*self.mask


def sample_relative_l2(pred,exact):
    pred,exact=pred.double().flatten(1),exact.double().flatten(1)
    errors=torch.linalg.vector_norm(pred-exact,dim=1)/torch.linalg.vector_norm(exact,dim=1)
    if not torch.isfinite(errors).all():
        raise FloatingPointError("Non-finite CNN metric")
    return errors
