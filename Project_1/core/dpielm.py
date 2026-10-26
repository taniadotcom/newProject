import torch


def subdomain_ids(X):
    return (X[:, 0] >= 0.5).long()


def local_basis(X, W, b, side):
    lower = X.new_tensor([0.5 * side, 0.0])
    width = X.new_tensor([0.5, 1.0])
    phi = torch.tanh(((X - lower) / width) @ W.T + b.T)
    dphi = 1 - phi**2
    dx = dphi * (W[:, 0] / width[0])
    minus_laplace = 2 * phi * dphi * (W / width).square().sum(dim=1)
    return phi, dx, minus_laplace


def build_dpielm_system(X_f, X_b, X_i, W, b, rhs):
    n = W.shape[1]
    n_f, n_b, n_i = len(X_f), len(X_b), len(X_i)
    H = X_f.new_zeros((n_f + n_b + 2*n_i, 2*n))
    K = torch.cat([rhs(X_f), X_b.new_zeros((n_b + 2*n_i, 1))])
    ids_f, ids_b = subdomain_ids(X_f), subdomain_ids(X_b)
    for side in (0, 1):
        cols = slice(side*n, (side+1)*n)
        rows_f = torch.where(ids_f == side)[0]
        rows_b = torch.where(ids_b == side)[0]
        H[rows_f, cols] = local_basis(X_f[rows_f], W[side], b[side], side)[2]
        H[n_f+rows_b, cols] = local_basis(X_b[rows_b], W[side], b[side], side)[0]
        phi, dx, _ = local_basis(X_i, W[side], b[side], side)
        # Συνέχεια u και du/dx στον ίδιο άξονα x.
        sign = 1 if side == 0 else -1
        H[n_f+n_b:n_f+n_b+n_i, cols] = sign * phi
        H[n_f+n_b+n_i:, cols] = sign * dx
    return H, K


def predict_dpielm(X, W, b, c):
    pred = X.new_empty((len(X), 1))
    ids = subdomain_ids(X)
    n = W.shape[1]
    for side in (0, 1):
        mask = ids == side
        pred[mask] = local_basis(X[mask], W[side], b[side], side)[0] @ c[side*n:(side+1)*n]
    return pred


def interface_jumps(X, W, b, c):
    n = W.shape[1]
    left, dx_left, _ = local_basis(X, W[0], b[0], 0)
    right, dx_right, _ = local_basis(X, W[1], b[1], 1)
    return left @ c[:n] - right @ c[n:], dx_left @ c[:n] - dx_right @ c[n:]
