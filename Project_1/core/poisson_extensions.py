import math
import time

import numpy as np
import torch

from core.poisson_problems import exact_sine_nd, poisson_residual_nd


def gravitational_potential(X):
    return -exact_sine_nd(X)


def gravitational_density(X, G=1.0):
    return X.shape[1] * math.pi * exact_sine_nd(X) / (4 * G)


def gravitational_rhs(X):
    # Delta Phi = 4*pi*G*rho, άρα -Delta Phi = -4*pi*G*rho.
    return -4 * math.pi * gravitational_density(X)


def nonlinear_rhs(X):
    u = exact_sine_nd(X)
    return X.shape[1] * math.pi**2 * u + u**3


def nonlinear_residual(model, X):
    return poisson_residual_nd(model, X, nonlinear_rhs) + model(X)**3


def make_optimizer(model, name="Adam", lr=0.001, weight_decay=0.0):
    if name == "Adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "AdamW":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer: {name}")


def fit_pinn(model, X_f, X_b, epochs, rhs, optimizer="Adam", weight_decay=0.0, nonlinear=False):
    x = X_f.detach().clone().requires_grad_(True)
    saved_f, saved_b = X_f.clone(), X_b.clone()
    opt = make_optimizer(model, optimizer, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=200, factor=0.5)
    history = []
    start = time.perf_counter()
    for epoch in range(epochs):
        opt.zero_grad()
        x.grad = None
        res = nonlinear_residual(model, x) if nonlinear else poisson_residual_nd(model, x, rhs)
        loss_f = res.square().mean()
        loss_b = model(X_b).square().mean()
        loss = loss_f + loss_b
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite PINN loss")
        loss.backward()
        opt.step()
        scheduler.step(loss.item())
        history.append([epoch + 1, loss.item(), loss_f.item(), loss_b.item(), opt.param_groups[0]["lr"]])
    elapsed = time.perf_counter() - start
    assert torch.equal(X_f, saved_f) and torch.equal(X_b, saved_b)
    return elapsed, np.asarray(history)
