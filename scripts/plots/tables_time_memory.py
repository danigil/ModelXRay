"""Render the deployment-feasibility tables (Table 4, Table 5).

Aggregates `results/runtime_memory/measure_time.csv` to mean ± std over the
(image_rep, n_models, n_weights) grid and prints the LaTeX source for the two
tables embedded in the paper Section 7 (Practical Deployment Analysis).

Usage:
    python scripts/plots/tables_time_memory.py --table time
    python scripts/plots/tables_time_memory.py --table memory
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from model_xray.plots._style import repo_root


CSV_PATH = os.path.join(repo_root(), "results", "runtime_memory", "measure_time.csv")


def _aggregate(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    g = (df.groupby(["image_rep", "n_models", "n_weights"])[value_col]
         .agg(mean="mean", std="std").reset_index())
    g["std"] = g["std"].fillna(0)
    return g


def _format_latex(df: pd.DataFrame, value_col: str, unit_label: str) -> str:
    lines = [
        r"\begin{tabular}{l|" + "c" * df["image_rep"].nunique() + "}",
        r"\hline",
    ]
    image_reps = sorted(df["image_rep"].unique())
    lines.append("n\\_weights & " + " & ".join(image_reps) + r" \\")
    lines.append(r"\hline")
    for nw in sorted(df["n_weights"].unique()):
        sub = df[df["n_weights"] == nw]
        cells = []
        for rep in image_reps:
            row = sub[sub["image_rep"] == rep]
            if row.empty:
                cells.append("--")
            else:
                m = float(row["mean"].iloc[0]); s = float(row["std"].iloc[0])
                cells.append(f"${m:.3f} \\pm {s:.3f}$")
        lines.append(f"$10^{{{int(np.log10(nw))}}}$ & " + " & ".join(cells) + r" \\")
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    return f"% Auto-generated from {CSV_PATH} ({unit_label})\n" + "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", choices=["time", "memory"], required=True)
    parser.add_argument("--n-models", type=int, default=None,
                        help="Restrict to a single n_models row (default: all).")
    args = parser.parse_args()

    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"Missing {CSV_PATH}; run scripts/experiments/run_runtime_memory.py first.")
    df = pd.read_csv(CSV_PATH)
    if args.n_models is not None:
        df = df[df["n_models"] == args.n_models]
    if args.table == "time":
        agg = _aggregate(df, "time")
        print(_format_latex(agg, "time", "wall-clock seconds"))
    else:
        if df["peak_memory"].isna().all():
            print("% peak_memory column is NaN — install memory-profiler and rerun "
                  "scripts/experiments/run_runtime_memory.py to populate it.")
            return
        agg = _aggregate(df, "peak_memory")
        print(_format_latex(agg, "peak_memory", "MiB"))


if __name__ == "__main__":
    main()
