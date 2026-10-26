from pathlib import Path
import json
import platform

import numpy as np
import scipy
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core.utils import create_output_directory


def start_run(exp_id, smoke, settings):
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(42)
    np.random.seed(42)
    root = Path(__file__).resolve().parents[2]
    kind = "smoke" if smoke else "run"
    out = create_output_directory(root / "Results" / exp_id, prefix=kind)
    meta = dict(status="running", kind=kind, seed=42, device="cpu", threads=2,
                python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
                scipy=scipy.__version__, matplotlib=matplotlib.__version__, **settings)
    (out / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Output: {out.relative_to(root).as_posix()}", flush=True)
    return out, meta


def complete_run(out, meta):
    meta["status"] = "completed"
    (out / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def evaluation_grid(dim, n, dtype=torch.float32):
    grid = np.meshgrid(*([np.linspace(0, 1, n)] * dim))
    return torch.tensor(np.stack([v.ravel() for v in grid], axis=1), dtype=dtype)


def plot_fields(out, X, exact, predictions, n):
    dim = X.shape[1]
    grid = X.numpy().reshape(*([n] * dim), dim)
    shape = [n] * dim
    ref = np.asarray(exact).reshape(shape)
    values = [np.asarray(v).reshape(shape) for v in predictions.values()]
    if dim == 3:
        grid, ref = grid[:, :, n//2], ref[:, :, n//2]
        values = [v[:, :, n//2] for v in values]
    lo = min(float(ref.min()), *(float(v.min()) for v in values))
    hi = max(float(ref.max()), *(float(v.max()) for v in values))
    err = max(float(abs(v-ref).max()) for v in values)
    fig, axes = plt.subplots(2, len(values), figsize=(4*len(values), 6), squeeze=False)
    for j, (name, val) in enumerate(zip(predictions, values)):
        for i, data, levels, cmap in [(0, val, np.linspace(lo, max(hi, lo+1e-12), 35), "viridis"),
                                      (1, abs(val-ref), np.linspace(0, max(err, 1e-12), 35), "inferno")]:
            im = axes[i,j].contourf(grid[...,0], grid[...,1], data, levels=levels, cmap=cmap)
            fig.colorbar(im, ax=axes[i,j])
            axes[i,j].set(xlabel="x", ylabel="y", title=name if i == 0 else "Absolute error")
    if dim == 3:
        fig.suptitle(f"z={grid[0,0,2]:.4f}")
    fig.tight_layout()
    fig.savefig(out / "comparison.png", dpi=140)
    plt.close(fig)
