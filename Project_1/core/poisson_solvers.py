import csv
import ctypes
import hashlib
import os
import time

import numpy as np
import torch

from core.poisson_problems import poisson_residual_nd


def point_hash(X):
    return hashlib.sha256(X.detach().cpu().numpy().tobytes()).hexdigest()


def state_hash(state):
    digest = hashlib.sha256()
    for key, value in state.items():
        digest.update(key.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def train_pinn(model, X_f, X_b, epochs, rhs, learning_rate=0.001):
    x_f = X_f.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=200, factor=0.5)
    history = []
    start = time.perf_counter()
    for epoch in range(epochs):
        optimizer.zero_grad()
        x_f.grad = None
        loss_f = poisson_residual_nd(model, x_f, rhs).square().mean()
        loss_b = model(X_b).square().mean()
        loss = loss_f + loss_b
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at epoch {epoch + 1}")
        loss.backward()
        optimizer.step()
        scheduler.step(loss.item())
        history.append((epoch + 1, loss.item(), loss_f.item(), loss_b.item()))
    elapsed = time.perf_counter() - start
    assert torch.equal(x_f.detach(), X_f)
    return elapsed, np.asarray(history)


def build_pielm_system(X_f, X_b, W, b, rhs):
    # Μικρά blocks για τη μνήμη.
    H = X_f.new_empty((len(X_f) + len(X_b), len(W)))
    norm_squared = W.square().sum(dim=1, keepdim=True)
    with torch.no_grad():
        for start in range(0, len(X_f), 128):
            batch = X_f[start:start+128]
            phi = torch.tanh(W @ batch.T + b)
            ddphi = -2 * phi * (1 - phi**2)
            H[start:start+len(batch)] = -(norm_squared * ddphi).T
        for start in range(0, len(X_b), 128):
            batch = X_b[start:start+128]
            H[len(X_f)+start:len(X_f)+start+len(batch)] = torch.tanh(W @ batch.T + b).T
    K = torch.cat([rhs(X_f), X_b.new_zeros((len(X_b), 1))])
    return H, K


def solve_pielm(X_f, X_b, W, b, rhs):
    start = time.perf_counter()
    H, K = build_pielm_system(X_f, X_b, W, b, rhs)
    solution = torch.linalg.lstsq(H, K, driver="gelsd")
    elapsed = time.perf_counter() - start
    if not torch.isfinite(solution.solution).all():
        raise FloatingPointError("Non-finite PIELM solution")
    return solution.solution, elapsed, int(solution.rank)


def predict_pielm(X, W, b, c):
    # Πρόβλεψη σε μικρά batches.
    with torch.no_grad():
        return torch.cat([torch.tanh(W @ batch.T + b).T @ c for batch in X.split(512)])


def relative_l2(prediction, exact):
    prediction = np.asarray(prediction, dtype=np.float64).ravel()
    exact = np.asarray(exact, dtype=np.float64).ravel()
    if prediction.shape != exact.shape or not np.isfinite(prediction).all() or not np.isfinite(exact).all():
        raise ValueError("Invalid predictions or exact values")
    norm = np.linalg.norm(exact)
    if norm == 0:
        raise ValueError("Relative L2 requires a nonzero reference")
    return float(np.linalg.norm(prediction - exact) / norm)


def write_metrics(path, rows):
    with path.open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def available_memory():
    if os.name == "nt":
        class MemoryStatus(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
                (name, ctypes.c_ulonglong) for name in
                ("total", "available", "page_total", "page_available", "virtual_total", "virtual_available", "extended")]
        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            raise OSError("Cannot read available physical memory")
        return status.available
    return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")


def memory_plan(n_f, n_b, dtype=torch.float32):
    n = n_f + n_b
    matrix = n * n * torch.empty((), dtype=dtype).element_size()
    # Μνήμη για H, αντίγραφα/SVD και runtime.
    estimate = 4 * matrix + 64 * 1024**2
    free = available_memory()
    return {"H_shape": [n, n], "H_MiB": matrix / 1024**2,
            "estimated_peak_extra_MiB": estimate / 1024**2,
            "available_MiB": free / 1024**2,
            "safe": estimate <= 0.75 * free,
            "estimate_note": "Block construction: 4*H + 64 MiB reserve. Heuristic, not a measured peak. Keep 25% of free RAM unused."}


def require_memory(plan):
    if not plan["safe"]:
        raise MemoryError(f"PIELM preflight: estimated {plan['estimated_peak_extra_MiB']:.0f} MiB, "
                          f"available {plan['available_MiB']:.0f} MiB. Close other applications and retry.")
