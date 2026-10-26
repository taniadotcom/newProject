from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import csv
import json
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from core.utils import create_output_directory, latest_completed
from core.poisson_solvers import relative_l2, write_metrics


def read_run(path, problem, kind):
    meta = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    if meta["status"] != "completed" or meta["kind"] != kind:
        raise ValueError("Only completed runs of the requested kind can be compared")
    with (path / "metrics.csv").open(encoding="utf-8-sig", newline="") as f:
        metrics = list(csv.DictReader(f))
    expected = {("PINN","sobol"): "pinn", ("PIELM","sobol"): "pielm", ("PINN","delaunay"): "delaunay"}
    if len(metrics) != 3 or {(r["method"],r["sampling"]) for r in metrics} != set(expected):
        raise ValueError("Expected all three methods exactly once")
    rows = []
    with np.load(path / "predictions.npz") as data:
        for row in metrics:
            error = float(row["relative_L2"])
            measured = relative_l2(data[expected[row["method"],row["sampling"]]], data["exact"])
            if not math.isclose(measured, error, rel_tol=2e-5, abs_tol=1e-8):
                raise ValueError("Metrics disagree with saved predictions")
            elapsed = float(row["time_sec"])
            if not np.isfinite([error,elapsed]).all() or error <= 0 or elapsed <= 0:
                raise ValueError("Invalid error or runtime")
            rows.append(dict(problem=problem, method=row["method"], sampling=row["sampling"],
                N_f=int(row["N_f"]), N_b=int(row["N_b"]), relative_L2=error,
                time_sec=elapsed, seed=meta["seed"]))
    return meta, rows


def select_method(rows):
    methods = sorted({(r["method"], r["sampling"]) for r in rows})
    by_problem = {p: {(r["method"], r["sampling"]): r for r in rows if r["problem"] == p}
                  for p in ("smooth", "nonsmooth")}
    if any(set(values) != set(methods) for values in by_problem.values()):
        raise ValueError("Both problems must contain all methods")
    scores = {}
    for method in methods:
        ratios = [v[method]["relative_L2"] / min(r["relative_L2"] for r in v.values()) for v in by_problem.values()]
        scores[method] = max(ratios)
    selected = min(methods, key=lambda m: (scores[m], sum(v[m]["time_sec"] for v in by_problem.values())))
    dominant = all(by_problem[p][selected][metric] <= min(r[metric] for r in by_problem[p].values())
                   for p in by_problem for metric in ("relative_L2", "time_sec"))
    return {"method": selected[0], "sampling": selected[1], "clear_winner": dominant,
            "criterion": "Custom experimental heuristic: minimize worst relative error ratio to the best method on each problem; runtime breaks exact ties.",
            "scores": {"/".join(m): score for m,score in scores.items()},
            "interpretation": "Best observed accuracy and runtime on both problems" if dominant else
                              "Trade-off or differing problem rankings. Accuracy-based extension choice, not general superiority.",
            "stability": "One seed per problem. Finite outputs and numerical rank do not establish statistical stability."}


def main(smooth=None, nonsmooth=None, smoke=False):
    root = Path(__file__).resolve().parents[2]
    kind = "smoke" if smoke else "run"
    smooth = Path(smooth).resolve() if smooth else (latest_completed(root / "Results/exp_10_3d_pinn_vs_pielm", kind)
                    if smoke else root / "Results/exp_10_3d_pinn_vs_pielm/run_002")
    nonsmooth = Path(nonsmooth).resolve() if nonsmooth else latest_completed(root / "Results/exp_11_3d_nonsmooth_pinn_vs_pielm", kind)
    ms, rs = read_run(smooth, "smooth", kind)
    mn, rn = read_run(nonsmooth, "nonsmooth", kind)
    for key in ("seed", "boundary_seed", "dtype", "N_f", "N_b_per_face", "N_b_total", "X_f_sha256", "X_b_sha256"):
        if ms[key] != mn[key]:
            raise ValueError(f"Unmatched comparison setting: {key}")
    for key in ("epochs", "hidden_layers", "optimizer", "lr", "activation"):
        if ms["PINN"][key] != mn["PINN"][key]:
            raise ValueError(f"Unmatched PINN setting: {key}")
    rows = rs + rn
    decision = select_method(rows)
    decision["kind"] = kind
    decision["usable_for_full_extension"] = not smoke
    sources = {"smooth": smooth.relative_to(root).as_posix(), "nonsmooth": nonsmooth.relative_to(root).as_posix()}
    decision["sources"] = sources
    output = create_output_directory(root / "Results/exp_12_compare_smooth_nonsmooth", prefix=kind)
    write_metrics(output / "combined_metrics.csv", rows)
    (output / "selection.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    labels = ["PINN/Sobol", "PIELM/Sobol", "PINN/Delaunay"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, metric, ylabel in zip(axes, ["relative_L2","time_sec"], ["Relative L2", "Training/solve time (s)"]):
        for offset, problem_rows, label in [(-0.18,rs,"Smooth"),(0.18,rn,"Less smooth")]:
            values = [next(r[metric] for r in problem_rows if f"{r['method']}/{r['sampling'].capitalize()}" == name) for name in labels]
            ax.bar(np.arange(3)+offset, values, width=0.36, label=label)
        ax.set(xticks=np.arange(3), xticklabels=labels, ylabel=ylabel, yscale="log")
        ax.legend()
    fig.suptitle(f"3D comparison ({kind})")
    fig.tight_layout(); fig.savefig(output / "comparison_summary.png", dpi=160); plt.close(fig)
    lines = ["# Smooth και less-smooth 3D", "", f"Τύπος εκτέλεσης: {kind}.", "",
             "| Problem | Method | Sampling | Relative L2 | Seconds |", "|---|---|---|---:|---:|"]
    lines += [f"| {r['problem']} | {r['method']} | {r['sampling']} | {r['relative_L2']:.10g} | {r['time_sec']:.6f} |" for r in rows]
    lines += ["", "## Επιλογή για περισσότερες διαστάσεις", "",
              "SMOKE: έλεγχος λειτουργίας μόνο. Καμία επιστημονική επιλογή." if smoke else f"{decision['method']} / {decision['sampling']}",
              decision["criterion"], decision["interpretation"], "", decision["stability"],
              f"PIELM numerical rank: smooth {ms['PIELM']['rank']}/{ms['PIELM']['N_star']}, nonsmooth {mn['PIELM']['rank']}/{mn['PIELM']['N_star']}.",
              "Τα objectives και η χωρητικότητα διαφέρουν. Οι χρόνοι μετρήθηκαν σε διαφορετικές εκτελέσεις στο ίδιο σύστημα και επηρεάζονται από το φορτίο.",
              "", "Πηγές:", *(f"- {p}: {v}/metrics.csv και predictions.npz" for p,v in sources.items())]
    (output / "SUMMARY.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    (output / "metadata.json").write_text(json.dumps({"status":"completed", "kind":kind,"sources":sources}, indent=2), encoding="utf-8")
    print(f"Output: {output.relative_to(root).as_posix()}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smooth", type=Path)
    parser.add_argument("--nonsmooth", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    main(args.smooth, args.nonsmooth, args.smoke)
