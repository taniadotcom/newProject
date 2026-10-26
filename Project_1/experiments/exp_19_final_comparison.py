from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import argparse
import csv
import json
import math

from core.utils import latest_completed
from core.poisson_solvers import write_metrics
from core.experiment_utils import start_run, complete_run


GROUPS={"A":"Classical Poisson and inverse benchmarks", "B":"Optimizer/activation study",
        "C":"PIELM and DPIELM", "D":"Gravitational Poisson", "E":"Semilinear Poisson", "F":"CNN surrogate"}


def collect_results(root, smoke=False):
    rows=[]
    fields=["category","experiment","kind","problem","dimension","method","sampling","optimizer",
            "activation","relative_L2","training_solve_sec","inference_sec","epochs","N_f","N_b",
            "seed","metric_scope","source"]
    def add(**values):
        row=dict.fromkeys(fields,"N/A");row.update(values)
        for key,value in row.items():
            if value=="":row[key]="N/A"
        if row["method"] in ("PIELM","DPIELM"):
            row["optimizer"]=row["activation"]="N/A"
        assert math.isfinite(float(row["relative_L2"]))
        rows.append(row)
    for i in range(1,10):
        base=next((root/"Results").glob(f"exp_{i:02d}_*"))
        run=base/"run_001"
        if i==3:run=root/"Results/inference_02/run_001"
        path=run/"rerun_metrics.json"
        meta=json.loads(path.read_text(encoding="utf-8"));params=meta["actual_parameters"]
        add(category="A",experiment=f"exp_{i:02d}",kind="run",problem=base.name,dimension=params["dim"],
            method="PIELM" if i in (4,5,8) else "PINN",sampling=params.get("method",params.get("sampling","")),
            optimizer=params.get("optimizer","N/A"),activation="N/A" if i in (4,5,8) else "tanh",
            relative_L2=meta["error"],training_solve_sec=meta.get("elapsed","") if i!=3 else "",
            epochs=meta.get("epochs_completed",""),N_f=params.get("N_f",""),N_b=params.get("N_b_total",""),
            seed=params.get("sampling_seed") if params.get("sampling_seed") is not None else "unrecorded",
            metric_scope="inverse checkpoint evaluation" if i==3 else "single PDE instance",
            source=path.relative_to(root).as_posix())
    for i,name in [(10,"exp_10_3d_pinn_vs_pielm/run_002"),(11,"exp_11_3d_nonsmooth_pinn_vs_pielm/run_001")]:
        run=root/"Results"/name
        meta=json.loads((run/"metadata.json").read_text(encoding="utf-8"))
        with (run/"metrics.csv").open(encoding="utf-8") as f:metrics=list(csv.DictReader(f))
        for row in metrics:
            add(category="A",experiment=f"exp_{i}",kind="run",problem="smooth" if i==10 else "less-smooth",
                dimension=3,method=row["method"],sampling=row["sampling"],relative_L2=row["relative_L2"],
                optimizer=meta["PINN"]["optimizer"] if row["method"]=="PINN" else "N/A",
                activation=meta["PINN"]["activation"] if row["method"]=="PINN" else "N/A",
                training_solve_sec=row["time_sec"],epochs=20000 if row["method"]=="PINN" else 0,
                N_f=row["N_f"],N_b=row["N_b"],seed=42,metric_scope="controlled single PDE instance",
                source=(run/"metrics.csv").relative_to(root).as_posix())
    for run in sorted((root/"Results/exp_13_nd_best_method").glob("run_*")):
        meta=json.loads((run/"metadata.json").read_text(encoding="utf-8"))
        if meta["status"]!="completed":continue
        row=meta["metrics"]
        add(category="A",experiment="exp_13",kind="run",problem="smooth ND",dimension=row["dimension"],
            method=row["method"],sampling=row["sampling"],relative_L2=row["relative_L2"],
            optimizer=meta.get("PINN",{}).get("optimizer","N/A"),activation=meta["activation"],
            training_solve_sec=row["training_time"],inference_sec=row["evaluation_time"],epochs=row["epochs"],
            N_f=row["N_f"],N_b=row["N_b_total"],seed=row["seed"],metric_scope="16384 Sobol evaluation points",
            source=(run/"metrics.csv").relative_to(root).as_posix())
    missing=[]
    for i,group in [(14,"B"),(15,"C"),(16,"D"),(17,"E"),(18,"F")]:
        base=root/"Results/exp_18_cnn_poisson" if i==18 else next((root/"Results").glob(f"exp_{i}_*"),None)
        if base is None or not list(base.glob("smoke_*" if smoke else "run_*")):
            missing.append(i);continue
        run=latest_completed(base,"smoke" if smoke else "run")
        meta=json.loads((run/"metadata.json").read_text(encoding="utf-8"))
        if i==18 and not smoke:
            try:
                candidate=latest_completed(root/"Results/exp_18_cnn_dilated")
            except FileNotFoundError:
                candidate=None
            if candidate is not None:
                chosen=json.loads((candidate/"metadata.json").read_text(encoding="utf-8"))
                if chosen["selection_usable"] and chosen["selected_architecture"]=="dilated":
                    run,meta=candidate,chosen
        with (run/"metrics.csv").open(encoding="utf-8") as f:metrics=list(csv.DictReader(f))
        for row in metrics:
            add(category=group,experiment=f"exp_{i}",kind=meta["kind"],problem=GROUPS[group],
                dimension=meta["dimension"],method=row.get("method","PINN"),sampling="sine-mode dataset" if i==18 else "sobol",
                optimizer=row.get("optimizer",meta.get("optimizer","N/A")),
                activation="tanh" if i==18 else row.get("activation",meta.get("activation","N/A")),
                relative_L2=row["relative_L2"],training_solve_sec=row["time_sec"],
                inference_sec=row.get("inference_time_sec",""),epochs=row.get("epochs",0),
                N_f=row.get("N_f",""),N_b=row.get("N_b",""),seed=meta["dataset_seed"] if i==18 else meta["seed"],
                metric_scope=("mean held-out per-sample L2; offline training; dilated CNN" if meta.get("selected_architecture")=="dilated"
                              else "mean held-out per-sample L2; offline training") if i==18 else "single PDE instance",
                source=(run/"metrics.csv").relative_to(root).as_posix())
    return rows,missing


def main(smoke=False):
    root=Path(__file__).resolve().parents[2]
    rows,missing=collect_results(root,smoke)
    out,meta=start_run("exp_19_final_comparison",smoke,dict(missing_full_experiments=missing,training=False))
    write_metrics(out/"FINAL_RESULTS_SUMMARY.csv",rows)
    lines=["# Τελική σύνοψη πρακτικού", "",
        "Οι ενότητες αφορούν διαφορετικά προβλήματα, διαστάσεις και budgets. Δεν αποτελούν κοινή κατάταξη μεθόδων.",
        "Οι χρόνοι CNN διαχωρίζουν offline training και inference σε held-out δεδομένα.",
        "Smoke rows are functional checks only." if smoke else "Παρατίθενται μόνο ολοκληρωμένα full runs.", ""]
    for group,title in GROUPS.items():
        lines += [f"## {group}. {title}","", "| Experiment | Method | Optimizer/activation | Relative L2 | Training/solve s | Inference s |",
                  "|---|---|---|---:|---:|---:|"]
        for row in rows:
            if row["category"]==group:
                lines.append(f"| {row['experiment']} ({row['dimension']}D) | {row['method']}/{row['sampling']} | "
                    f"{row['optimizer']}/{row['activation']} | {float(row['relative_L2']):.10g} | "
                    f"{row['training_solve_sec']} | {row['inference_sec']} |")
        lines.append("")
    lines += ["Το exp_10/11 δεν δίνει καθολικό νικητή: το PIELM είναι ταχύτερο, αλλά το PINN/Delaunay είναι ακριβέστερο στο less-smooth.",
        "Το exp_14 χρησιμοποιεί 3000 epochs αντί των 20000 του exp_10 και αξιολογείται μόνο εσωτερικά ως controlled study.",
        "Η DPIELM προσθέτει interface εξισώσεις με ίδια συνολικά interior/boundary σημεία και ίδιο total basis count.",
        "Το gravitational benchmark είναι nondimensional finite-box manufactured πρόβλημα, όχι N-body ή isolated-space μοντέλο.",
        "Το nonlinear benchmark λύνεται μόνο με PINN. Το linear PIELM δεν εφαρμόστηκε στο u^3.",
        "Το CNN είναι supervised surrogate για τέσσερα sine modes, με επιλογή checkpoint από validation και αξιολόγηση σε ανεξάρτητα test samples.",
        "Μία seed και ένα προκαθορισμένο budget ανά νέο πείραμα δεν αποδεικνύουν γενική υπεροχή ή βέλτιστο weight decay.",
        f"Χωρίς διαθέσιμο ολοκληρωμένο run: {missing or 'κανένα'}.", "", "## Πηγές", ""]
    lines += [f"- {p}" for p in sorted({r['source'] for r in rows})]
    (out/"FINAL_PRACTICAL_SUMMARY.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    complete_run(out,meta)
    return out


if __name__ == "__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--smoke",action="store_true")
    main(parser.parse_args().smoke)
