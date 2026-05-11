"""Render the deployment-feasibility tables (Table 4, Table 5).

Paper layout:
                 # Parameters       10^2  10^3  10^4  10^5  10^6  10^7  10^8
    Features
    Ours - GF                       ...
    Baseline Yin et al. (2022)      ...

Total parameters = n_models * n_weights. We aggregate over every CSV row
whose product matches the requested 10^k bucket so the table is robust to
which (n_models, n_weights) cells the runner happened to sweep.

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

# Paper row labels (Tables 4 and 5).
PAPER_LABELS = {
    "gf":   "Ours - GF",
    "phis": "Baseline Yin et al. (2022) (Reproduction)",
}
# When the cached CSV does not include `gf` (older measurement runs used the
# byte-decomp `rgb` representation as the GF stand-in), fall back to `rgb`
# and warn.
GF_FALLBACK = "rgb"


def _bucket_10k(df: pd.DataFrame) -> pd.DataFrame:
    """Add a `params_log10` column = round(log10(n_models * n_weights))."""
    df = df.copy()
    df["params_log10"] = np.round(np.log10(df["n_models"].astype(float) * df["n_weights"].astype(float))).astype(int)
    return df


def _aggregate_by_bucket(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    g = (df.groupby(["image_rep", "params_log10"])[value_col]
         .agg(mean="mean", std="std", n="count").reset_index())
    g["std"] = g["std"].fillna(0.0)
    return g


def _format_latex(agg: pd.DataFrame, rows_in_order: list[tuple[str, str]],
                  buckets: list[int], unit_label: str) -> str:
    n_cols = len(buckets)
    lines = [r"\begin{tabular}{l|" + "c" * n_cols + "}", r"\hline"]
    header_cells = [f"$10^{{{k}}}$" for k in buckets]
    lines.append(r"\# Parameters & " + " & ".join(header_cells) + r" \\")
    lines.append(r"\hline")
    for rep, label in rows_in_order:
        cells = []
        for k in buckets:
            row = agg[(agg["image_rep"] == rep) & (agg["params_log10"] == k)]
            if row.empty:
                cells.append("--")
            else:
                m = float(row["mean"].iloc[0]); s = float(row["std"].iloc[0])
                cells.append(f"${m:.3f} \\pm {s:.3f}$")
        lines.append(label + " & " + " & ".join(cells) + r" \\")
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    return f"% Auto-generated from {CSV_PATH} ({unit_label})\n" + "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", choices=["time", "memory"], required=True)
    args = parser.parse_args()

    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"Missing {CSV_PATH}; run scripts/experiments/run_runtime_memory.py first.")
    df = _bucket_10k(pd.read_csv(CSV_PATH))

    have_reps = set(df["image_rep"].unique())
    if "gf" in have_reps:
        gf_rep = "gf"
    elif GF_FALLBACK in have_reps:
        gf_rep = GF_FALLBACK
        print(f"% NOTE: csv does not include 'gf'; using '{GF_FALLBACK}' as GF stand-in (see docstring).")
    else:
        raise SystemExit(f"CSV has no 'gf' or '{GF_FALLBACK}' image_rep; available: {sorted(have_reps)}")
    if "phis" not in have_reps:
        raise SystemExit(f"CSV has no 'phis' image_rep; available: {sorted(have_reps)}")

    rows = [(gf_rep, PAPER_LABELS["gf"]), ("phis", PAPER_LABELS["phis"])]
    buckets = sorted(df["params_log10"].unique().tolist())

    if args.table == "time":
        agg = _aggregate_by_bucket(df[df["image_rep"].isin([gf_rep, "phis"])], "time")
        print(_format_latex(agg, rows, buckets, "wall-clock seconds"))
    else:
        if df["peak_memory"].isna().all():
            print("% peak_memory column is NaN — install memory-profiler and rerun "
                  "scripts/experiments/run_runtime_memory.py to populate it.")
            return
        agg = _aggregate_by_bucket(df[df["image_rep"].isin([gf_rep, "phis"])], "peak_memory")
        print(_format_latex(agg, rows, buckets, "peak memory MiB"))


if __name__ == "__main__":
    main()
