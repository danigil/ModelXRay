"""Regenerate exp1_id_oml.png — Experiment 1, SCZ STL10 OML (paper fig:exp1)."""

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

# B1 (Gilkarov) was not re-run in our reproduction; values transcribed from the
# dotted-green line in the originally published exp1_id_oml.png. See main.tex
# Section sec:baselines_academic for the source.
GILKAROV_PIECEWISE = {
    1: 0.50, 2: 0.50, 3: 0.50, 4: 0.50, 5: 0.50, 6: 0.50, 7: 0.50, 8: 0.50,
    9: 0.50, 10: 0.50, 11: 0.50, 12: 0.50, 13: 0.50, 14: 0.50, 15: 0.50,
    16: 0.55, 17: 0.65, 18: 0.80, 19: 0.92, 20: 0.97, 21: 0.99, 22: 1.00, 23: 1.00,
}


def _fsl_curve(eval_col: str) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(RESULTS_DIR, "fsl_osl.csv"))
    df = df[df["mc"] == "ghrp_stl10"]
    diag = df[df["lsb"] == df["model_lsb"]]
    per_run = diag.groupby(["run num", "model_lsb"])[eval_col].mean().reset_index()
    return agg_with_ci(per_run, "model_lsb", eval_col).rename(columns={"model_lsb": "X"})


def _naive_curve() -> pd.DataFrame:
    df = pd.read_csv(os.path.join(RESULTS_DIR, "b4_b7_threshold.csv"))
    return agg_with_ci(df.assign(_x=df["X"]).rename(columns={}), "X", "acc_mean_test")


def _malconv_curve() -> pd.DataFrame:
    p = os.path.join(RESULTS_DIR, "b3_malconv.csv")
    if not os.path.exists(p):
        return pd.DataFrame()
    return agg_with_ci(pd.read_csv(p), "X", "acc_mean_test")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=os.path.join(default_out_dir(), "exp1_id_oml.png"))
    args = parser.parse_args()

    fig, ax = init_axes()

    for label, col in (("OSL CNN (Centroid)", "test_acc_centroid"),
                       ("OSL CNN (1NN)", "test_acc_nn")):
        c = _fsl_curve(col)
        plot_band(ax, c, FSL_STYLES[label], linewidth=2.0, alpha=0.15)

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
        c = agg_with_ci(sub, "X", "acc_mean_test").sort_values("X")
        plot_band(ax, c, style, linewidth=1.3, alpha=0.12, marker_size=4)

    finalize(ax, title="Model Collection = SCZ (STL-10)",
             ylim=(0.45, 1.02),
             ylabel="Test Accuracy (Benign + Malicious) (X=Model LSB)")
    save_fig(fig, args.out)


if __name__ == "__main__":
    main()
