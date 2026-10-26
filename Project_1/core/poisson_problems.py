import math
import torch


def exact_sine_nd(X):
    return torch.sin(math.pi * X).prod(dim=1, keepdim=True)


def poisson_rhs_sine_nd(X):
    return X.shape[1] * math.pi**2 * exact_sine_nd(X)


def exact_nonsmooth_3d(X):
    t = X[:, :1] - 0.5
    return exact_sine_nd(X) * (1 + 8 * t.abs()**3)


def poisson_rhs_nonsmooth_3d(X):
    if X.shape[1] != 3:
        raise ValueError("The nonsmooth benchmark requires three coordinates.")
    t = X[:, :1] - 0.5
    g = 1 + 8 * t.abs()**3
    dg = 24 * t * t.abs()
    ddg = 48 * t.abs()
    sine = exact_sine_nd(X)
    dx_sine = math.pi * torch.cos(math.pi * X[:, :1]) * torch.sin(math.pi * X[:, 1:]).prod(1, keepdim=True)
    return 3 * math.pi**2 * sine * g - 2 * dx_sine * dg - sine * ddg


def poisson_residual_nd(model, X, rhs=poisson_rhs_sine_nd):
    X.requires_grad_(True)
    u = model(X)
    grad = torch.autograd.grad(u, X, torch.ones_like(u), create_graph=True)[0]
    laplace = sum(torch.autograd.grad(grad[:, d:d+1], X,
                  torch.ones_like(grad[:, d:d+1]), create_graph=True)[0][:, d:d+1]
                  for d in range(X.shape[1]))
    return -laplace - rhs(X)
