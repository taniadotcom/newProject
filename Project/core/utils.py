from pathlib import Path

import torch
import math
import numpy as np
from scipy.stats import qmc
from scipy.spatial import Delaunay


def generate_points(method="sobol", dim=2, n_points=None, seed=None, dtype=torch.float32):
    """
    Παραγωγή collocation points στο [0,1]^dim.
    Μέθοδοι: 'sobol', 'halton', 'random', 'delaunay'
    """

    # Προεπιλεγμένο πλήθος collocation points
    if n_points is None:
        if method == "sobol":
            n_points = 8192
        else:
            n_points = 10000
        print(f"Collocation points: {n_points}, method: '{method}'.")


    if method == "sobol":
        if not (n_points > 0 and (n_points & (n_points - 1)) == 0):
            power = round(math.log2(n_points))
            n_points = 2 ** power
            print(f"Sobol points: {n_points}.")
        sampler = qmc.Sobol(d=dim, scramble=True, seed=seed)
        sample = sampler.random(n=n_points)
        return torch.tensor(sample, dtype=dtype)


    elif method == "halton":
        sampler = qmc.Halton(d=dim, scramble=True, seed=seed)
        sample = sampler.random(n=n_points)
        return torch.tensor(sample, dtype=dtype)


    elif method == "random":
        generator = None if seed is None else torch.Generator().manual_seed(seed)
        return torch.rand(n_points, dim, dtype=dtype, generator=generator)


    elif method == "delaunay":

        sampler = qmc.Sobol(d=dim, scramble=True, seed=seed)

        power = round(math.log2(n_points * 2))
        raw_n_points = 2 ** power

        raw = sampler.random_base2(m=int(math.log2(raw_n_points)))

        tri = Delaunay(raw)
        # Centroids των simplices ως collocation points
        centroids = raw[tri.simplices].mean(axis=1)
        # Υποσύνολο χωρίς επανάθεση, ώστε να μη μετρά η σειρά των simplices
        rng = np.random if seed is None else np.random.default_rng(seed)
        idx = rng.choice(len(centroids), n_points, replace=False)
        return torch.tensor(centroids[idx], dtype=dtype)

    else:
        raise ValueError(f"Άγνωστη μέθοδος: '{method}'. Επίλεξε 'sobol', 'halton', 'random' ή 'delaunay'.")


def generate_boundary_points(method="sobol", dim=3, n_points=512, seed=None,
                             dtype=torch.float32):
    """Σημεία στις 2*dim πλευρές του [0,1]^dim, n_points ανά πλευρά."""
    if dim < 2:
        raise ValueError("Τα boundary points υποστηρίζουν dim >= 2.")

    if dim == 2:
        # Ίδια σειρά και κοινό t με τα υπάρχοντα 2D experiments
        t = generate_points(method, 1, n_points, seed=seed, dtype=dtype)
        zeros = torch.zeros_like(t)
        ones = torch.ones_like(t)
        return torch.cat([
            torch.cat([t, zeros], dim=1),
            torch.cat([t, ones], dim=1),
            torch.cat([zeros, t], dim=1),
            torch.cat([ones, t], dim=1),
        ], dim=0)

    faces = []
    for fixed_dim in range(dim):
        dims = [d for d in range(dim) if d != fixed_dim]
        for fixed_val in (0, 1):
            face_seed = None if seed is None else seed + 2 * fixed_dim + fixed_val
            t = generate_points(method, dim - 1, n_points,
                                seed=face_seed, dtype=dtype)
            face = torch.zeros(t.shape[0], dim, dtype=dtype)
            face[:, dims] = t
            face[:, fixed_dim] = fixed_val
            faces.append(face)
    return torch.cat(faces, dim=0)


def create_output_directory(base, prefix="run"):
    """Δημιουργεί νέο φάκελο εκτέλεσης χωρίς αντικατάσταση προηγούμενων outputs."""
    base = Path(base)
    run = 1
    while (base / f"{prefix}_{run:03d}").exists():
        run += 1
    output = base / f"{prefix}_{run:03d}"
    output.mkdir(parents=True, exist_ok=False)
    return output
