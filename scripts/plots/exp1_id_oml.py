"""Regenerate exp1_id_oml.png — Experiment 1, SCZ STL10 OML (paper Figure 4)."""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from model_xray.plots._style import (
    FSL_STYLES, GILKAROV_STYLE, MALCONV_STYLE, NAIVE_STYLES,
    agg_with_ci, default_out_dir, finalize, init_axes, plot_band, repo_root, save_fig,
)


RESULTS_DIR = os.path.join(repo_root(), "results", "exp1")
B1_CSV = os.path.join(RESULTS_DIR, "b1_gilkarov_per_x.csv")

# Fallback B1 (Gilkarov) values, transcribed from the dotted-green line in the
# originally published exp1_id_oml.png. Used only when results/exp1/b1_gilkarov_per_x.csv
# is missing — i.e. when scripts/experiments/baselines/run_b1_gilkarov.py hasn't
# been executed yet. Once that runs, the real CSV takes precedence.
GILKAROV_PIECEWISE = {
    1: 0.50, 2: 0.50, 3: 0.50, 4: 0.50, 5: 0.50, 6: 0.50, 7: 0.50, 8: 0.50,
    9: 0.50, 10: 0.50, 11: 0.50, 12: 0.50, 13: 0.50, 14: 0.50, 15: 0.50,
    16: 0.55, 17: 0.65, 18: 0.80, 19: 0.92, 20: 0.97, 21: 0.99, 22: 1.00, 23: 1.00,
}


def _b1_curve_from_csv(baseline: str = "xgboost") -> pd.DataFrame:
    df = pd.read_csv(B1_CSV)
    df = df[df["baseline"] == baseline]
    return agg_with_ci(df, "X", "accuracy")


def _fsl_curve(eval_col: str) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(RESULTS_DIR, "fsl_osl.csv"))
    diag = df[df["X_hat"] == df["X"]]
    per_run = diag.groupby(["repeat", "X"])[eval_col].mean().reset_index()
    return agg_with_ci(per_run, "X", eval_col)


def _malconv_curve() -> pd.DataFrame:
    p = os.path.join(RESULTS_DIR, "b3_malconv.csv")
    if not os.path.exists(p):
        return pd.DataFrame()
    return agg_with_ci(pd.read_csv(p), "X", "accuracy")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=os.path.join(default_out_dir(), "exp1_id_oml.png"))
    args = parser.parse_args()

    fig, ax = init_axes()

    for label, col in (("OSL CNN (Centroid)", "centroid"),
                       ("OSL CNN (1NN)", "nn")):
        c = _fsl_curve(col)
        plot_band(ax, c, FSL_STYLES[label], linewidth=2.0, alpha=0.15)

    if os.path.exists(B1_CSV):
        b1 = _b1_curve_from_csv()
        plot_band(ax, b1.sort_values("X"), GILKAROV_STYLE, linewidth=1.6, alpha=0.12)
    else:
        xs = sorted(GILKAROV_PIECEWISE)
        ax.plot(xs, [GILKAROV_PIECEWISE[x] for x in xs], **GILKAROV_STYLE)

    mc = _malconv_curve()
    if not mc.empty:
        mc = mc.sort_values("X")
        plot_band(ax, mc, MALCONV_STYLE, linewidth=1.3, alpha=0.12, marker_size=4)

    naive_groups = pd.read_csv(os.path.join(RESULTS_DIR, "b4_b7_threshold.csv"))
    for baseline_name, style in NAIVE_STYLES.items():
        sub = naive_groups[naive_groups["baseline"] == baseline_name]
        if sub.empty:
            continue
        c = agg_with_ci(sub, "X", "accuracy").sort_values("X")
        plot_band(ax, c, style, linewidth=1.3, alpha=0.12, marker_size=4)

    finalize(ax, title="Model Collection = SCZ (STL-10)",
             ylim=(0.45, 1.02),
             ylabel="Test Accuracy (Benign + Malicious) (X=Model LSB)")
    save_fig(fig, args.out)


if __name__ == "__main__":
    main()
