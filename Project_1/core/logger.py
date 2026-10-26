from pathlib import Path


import csv
import os
from datetime import datetime

LOG_FILE = str(Path(__file__).resolve().parents[2] / "Logs" / "experiments.csv")

COLUMNS = [
    "date", "exp_id", "problem", "method",
    "sigma", "n_points", "epochs", "L2_error", "time_sec", "notes"
]


def log_experiment(exp_id, problem, method, sigma, n_points, epochs, l2_error, time_sec, notes=""):
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

    write_header = not os.path.exists(LOG_FILE)

    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow({
            "date":      datetime.now().strftime("%Y-%m-%d %H:%M"),
            "exp_id":    exp_id,
            "problem":   problem,
            "method":    method,
            "sigma":     sigma,
            "n_points":  n_points,
            "epochs":    epochs,
            "L2_error":  f"{l2_error:.4e}",
            "time_sec":  f"{time_sec:.1f}",
            "notes":     notes
        })
    print(f"[LOG] Αποθηκεύτηκε: {exp_id} | L2={l2_error:.4e}")
