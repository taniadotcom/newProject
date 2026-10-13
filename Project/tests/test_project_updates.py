from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ast
import json
import math
import tempfile
import unittest
import torch
import numpy as np
from scipy.spatial import Delaunay
from scipy.stats import qmc

from core.model import PINN, PINN_Large, PINN3D
from core.pinn import poisson_residual_3d, poisson_residual_gauss
from core.utils import generate_points, generate_boundary_points, create_output_directory
from experiments.exp_10_3d_pinn_vs_pielm import (
    build_pielm_system, solve_pielm, train_pinn, check_boundary)


class ProjectUpdateTests(unittest.TestCase):
    def test_original_models_and_checkpoints(self):
        root = Path(__file__).resolve().parents[2]
        baseline = json.loads((Path(__file__).resolve().parent / "fixtures/model_baseline.json").read_text(encoding="utf-8"))
        original = {}
        exec(baseline["Project/core/model.py"]["source"], original)
        for cls, dim in [(PINN, 2), (PINN_Large, 2), (PINN3D, 3)]:
            torch.manual_seed(42)
            before = original[cls.__name__]()
            torch.manual_seed(42)
            after = cls()
            self.assertEqual(list(before.state_dict()), list(after.state_dict()))
            for key in before.state_dict():
                self.assertTrue(torch.equal(before.state_dict()[key], after.state_dict()[key]))
            X = torch.rand(9, dim)
            self.assertTrue(torch.equal(before(X), after(X)))
            after.load_state_dict(before.state_dict(), strict=True)
        tree = ast.parse((root / "Project/core/model.py").read_bytes())
        self.assertEqual(sum(isinstance(n, ast.ClassDef) and n.name == "PINN3D" for n in tree.body), 1)

    def test_boundaries_and_reproducibility(self):
        for method in ["sobol", "halton", "random", "delaunay"]:
            for dtype in [torch.float32, torch.float64]:
                X_b = generate_boundary_points(method, 3, 16, seed=42, dtype=dtype)
                check_boundary(X_b, 16)
                self.assertEqual(X_b.dtype, dtype)
                self.assertTrue(torch.equal(X_b, generate_boundary_points(method, 3, 16, seed=42, dtype=dtype)))
                self.assertFalse(torch.equal(X_b, generate_boundary_points(method, 3, 16, seed=99, dtype=dtype)))

    def test_legacy_2d_boundaries(self):
        torch.manual_seed(42)
        t = generate_points("random", 1, 16)
        zeros, ones = torch.zeros_like(t), torch.ones_like(t)
        expected = torch.cat([torch.cat([t, zeros], 1), torch.cat([t, ones], 1),
                              torch.cat([zeros, t], 1), torch.cat([ones, t], 1)])
        torch.manual_seed(42)
        actual = generate_boundary_points("random", 2, 16)
        self.assertTrue(torch.equal(expected, actual))

    def test_legacy_3d_boundaries(self):
        torch.manual_seed(42)
        faces = []
        for fixed_dim in range(3):
            for fixed_val in (0, 1):
                t = generate_points("random", 2, 16)
                face = torch.zeros(16, 3)
                dims = [d for d in range(3) if d != fixed_dim]
                face[:, dims[0]] = t[:, 0]
                face[:, dims[1]] = t[:, 1]
                face[:, fixed_dim] = fixed_val
                faces.append(face)
        torch.manual_seed(42)
        actual = generate_boundary_points("random", 3, 16)
        self.assertTrue(torch.equal(torch.cat(faces), actual))

    def test_sampling_and_delaunay_centroids(self):
        for method in ["sobol", "halton", "random", "delaunay"]:
            X = generate_points(method, 3, 32, seed=42)
            self.assertEqual(X.shape, (32, 3))
            self.assertTrue(torch.all((X >= 0) & (X <= 1)))
            self.assertTrue(torch.equal(X, generate_points(method, 3, 32, seed=42)))
        raw = qmc.Sobol(d=3, scramble=True, seed=42).random_base2(m=6)
        tri = Delaunay(raw)
        centroids = raw[tri.simplices].mean(axis=1).astype(np.float32)
        for point in X.numpy():
            self.assertTrue(np.any(np.all(centroids == point, axis=1)))
        self.assertEqual(len(torch.unique(X, dim=0)), 32)

    def test_exact_poisson_residual(self):
        class Exact(torch.nn.Module):
            def forward(self, x):
                return torch.sin(math.pi * x).prod(dim=1, keepdim=True)
        X = generate_points("sobol", 3, 16, seed=42, dtype=torch.float64)
        self.assertLess(poisson_residual_3d(Exact(), X).abs().max().item(), 1e-12)

    def test_gaussian_poisson_sign(self):
        class Exact(torch.nn.Module):
            def __init__(self, sigma):
                super().__init__()
                self.sigma = sigma

            def forward(self, x):
                return torch.exp(-((x - 0.5)**2).sum(dim=1, keepdim=True) / (2 * self.sigma**2))

        for dtype, tolerance in [(torch.float32, 2e-5), (torch.float64, 1e-12)]:
            for sigma in [0.2, 0.5, 1.0]:
                with self.subTest(dtype=dtype, sigma=sigma):
                    X = generate_points("sobol", 2, 16, seed=42, dtype=dtype)
                    X = torch.cat([X, torch.tensor([[0.5, 0.5]], dtype=dtype)])
                    residual = poisson_residual_gauss(Exact(sigma), X, sigma)
                    self.assertLess(residual.abs().max().item(), tolerance)
                    model = PINN().to(dtype=dtype)
                    poisson_residual_gauss(model, X.detach().clone(), sigma).square().mean().backward()
                    self.assertTrue(torch.isfinite(model.net[0].weight.grad).all())

    def test_gaussian_pielm_rhs(self):
        path = Path(__file__).resolve().parents[1] / "experiments/exp_05_2d_pielm_gauss.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        # Εκτέλεση μόνο του RHS, χωρίς training ή εγγραφές του experiment
        names = {"x_c", "y_c", "r_sq", "u_exact_f", "f_vals"}
        assignments = [node for node in tree.body if isinstance(node, ast.Assign)
                       and isinstance(node.targets[0], ast.Name) and node.targets[0].id in names]
        self.assertEqual(len(assignments), len(names))
        for sigma in [0.2, 0.5, 1.0]:
            X = generate_points("sobol", 2, 16, seed=42, dtype=torch.float64).requires_grad_(True)
            u = torch.exp(-((X - 0.5)**2).sum(dim=1, keepdim=True) / (2 * sigma**2))
            grad = torch.autograd.grad(u.sum(), X, create_graph=True)[0]
            laplace = sum(torch.autograd.grad(grad[:, d].sum(), X, retain_graph=True)[0][:, d:d+1]
                          for d in range(2))
            class Parameters:
                SIGMA = sigma
            scope = {"torch": torch, "cfg": Parameters, "X_f": X}
            exec(compile(ast.Module(body=assignments, type_ignores=[]), str(path), "exec"), scope)
            torch.testing.assert_close(scope["f_vals"], -laplace, rtol=1e-12, atol=1e-12)

    def test_output_directories_preserve_existing_files(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            legacy = base / "model.pt"
            legacy.write_bytes(b"historical checkpoint")
            first = create_output_directory(base)
            checkpoint = first / "model.pt"
            checkpoint.write_bytes(b"first run")
            second = create_output_directory(base)
            smoke = create_output_directory(base, prefix="smoke")
            self.assertNotEqual(first, second)
            self.assertEqual(checkpoint.read_bytes(), b"first run")
            self.assertEqual(legacy.read_bytes(), b"historical checkpoint")
            self.assertTrue(second.is_dir() and smoke.is_dir())

    def test_pielm_matrix_against_autograd(self):
        torch.manual_seed(42)
        X_f = generate_points("sobol", 3, 8, seed=42, dtype=torch.float64)
        X_b = generate_boundary_points("sobol", 3, 4, seed=43, dtype=torch.float64)
        W, b = torch.randn(5, 3, dtype=torch.float64), torch.randn(5, 1, dtype=torch.float64)
        H, K = build_pielm_system(X_f, X_b, W, b)
        self.assertEqual(H.shape, (32, 5))
        self.assertEqual(K.shape, (32, 1))
        X = X_f.clone().requires_grad_(True)
        for j in range(5):
            phi = torch.tanh(X @ W[j] + b[j])
            grad = torch.autograd.grad(phi.sum(), X, create_graph=True)[0]
            laplace = sum(torch.autograd.grad(grad[:, d].sum(), X, retain_graph=True)[0][:, d]
                          for d in range(3))
            torch.testing.assert_close(H[:8, j], -laplace, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(H[8:], torch.tanh(W @ X_b.T + b).T)
        self.assertTrue(torch.equal(K[8:], torch.zeros_like(K[8:])))

    def test_shared_data_solve_and_backward(self):
        X_f = generate_points("sobol", 3, 16, seed=42)
        X_b = generate_boundary_points("sobol", 3, 4, seed=43)
        before_f, before_b = X_f.clone(), X_b.clone()
        torch.manual_seed(42)
        model = PINN3D()
        old_weights = model.net[0].weight.detach().clone()
        elapsed, history = train_pinn(model, X_f, X_b, 2)
        self.assertTrue(np.isfinite(history).all())
        self.assertFalse(torch.equal(old_weights, model.net[0].weight))
        W, b = torch.randn(40, 3), torch.randn(40, 1)
        c, elapsed, rank = solve_pielm(X_f, X_b, W, b)
        self.assertEqual(c.shape, (40, 1))
        self.assertTrue(torch.isfinite(c).all())
        self.assertGreater(rank, 0)
        self.assertTrue(torch.equal(X_f, before_f))
        self.assertTrue(torch.equal(X_b, before_b))


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
