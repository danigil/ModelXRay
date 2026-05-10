"""Regenerate exp2_id_oml.png + exp2_ood_oml.png — Experiment 2 OML view.

ID  = Famous Small CNNs  (paper Figure 5)
OOD = Famous Large CNNs  (paper Figure 7)
"""

from __future__ import annotations

import argparse
import os
from typing import Tuple

import numpy as np
import pandas as pd

from model_xray.plots._style import (
    FSL_STYLES_EXP2, MALCONV_STYLE, NAIVE_STYLES,
    agg_with_ci, default_out_dir, finalize, init_axes, plot_band, repo_root, save_fig,
)


RESULTS_DIR = os.path.join(repo_root(), "results", "exp2")

DATASETS = {
    "famous_le_10m":  ("Famous Small CNNs", "small"),
    "famous_le_100m": ("Famous Large CNNs", "large"),
}


def _fsl_curve(csv_path: str, eval_set: str, eval_col: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "eval_set" in df.columns:
        df = df[df["eval_set"] == eval_set]
    diag = df[df["X_hat"] == df["X"]]
    per_run = diag.groupby(["repeat", "X"])[eval_col].mean().reset_index()
    return agg_with_ci(per_run, "X", eval_col)


def _malconv_curve(suffix: str) -> pd.DataFrame:
    p = os.path.join(RESULTS_DIR, f"b3_malconv_{suffix}.csv")
    if not os.path.exists(p):
        return pd.DataFrame()
    return agg_with_ci(pd.read_csv(p), "X", "accuracy")


def _naive_groups(suffix: str) -> pd.DataFrame:
    p = os.path.join(RESULTS_DIR, f"b4_b7_threshold_{suffix}.csv")
    return pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()


def plot_one(dataset_key: str, dataset_label: str, suffix: str, out_path: str, ylim: Tuple[float, float]):
    fig, ax = init_axes()

    for arch_label, suffix_arch in (("OSL CNN", "osl"), ("SRNet", "srnet")):
        path = os.path.join(RESULTS_DIR, f"fsl_{suffix_arch}_{suffix}.csv")
        if not os.path.exists(path):
            print(f"  skip missing FSL {path}")
            continue
        for eval_label, col in (("Centroid", "centroid"), ("1NN", "nn")):
            c = _fsl_curve(path, eval_set=dataset_key, eval_col=col)
            plot_band(ax, c, FSL_STYLES_EXP2[f"{arch_label} ({eval_label})"], linewidth=2.0, alpha=0.10)

    mc = _malconv_curve(suffix)
    if not mc.empty:
        mc = mc.sort_values("X")
        plot_band(ax, mc, MALCONV_STYLE, linewidth=1.3, alpha=0.12, marker_size=4)

    nv = _naive_groups(suffix)
    for baseline_name, style in NAIVE_STYLES.items():
        sub = nv[nv["baseline"] == baseline_name]
        if sub.empty:
            continue
        c = agg_with_ci(sub, "X", "accuracy").sort_values("X")
        plot_band(ax, c, style, linewidth=1.3, alpha=0.12, marker_size=4)

    finalize(ax, title=f"Model Collection = {dataset_label}",
             ylim=ylim,
             ylabel="Test Accuracy (Benign + Malicious) (X=Model LSB)")
    save_fig(fig, out_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--id-out",  default=None,
                        help=f"Output path for exp2_id_oml.png (default: {os.path.join(default_out_dir(), 'exp2_id_oml.png')}).")
    parser.add_argument("--ood-out", default=None,
                        help=f"Output path for exp2_ood_oml.png (default: {os.path.join(default_out_dir(), 'exp2_ood_oml.png')}).")
    parser.add_argument("--only", choices=["id", "ood", "both"], default=None,
                        help="Render only one of the two panels (default: render whichever --*-out was passed; both if neither).")
    args = parser.parse_args()

    if args.only is None:
        # If exactly one --*-out is set, treat that as a single-panel request.
        if args.id_out is not None and args.ood_out is None:
            args.only = "id"
        elif args.ood_out is not None and args.id_out is None:
            args.only = "ood"
        else:
            args.only = "both"

    if args.only in ("id", "both"):
        out = args.id_out or os.path.join(default_out_dir(), "exp2_id_oml.png")
        plot_one("famous_le_10m",  "Famous Small CNNs", "small", out, ylim=(0.0, 1.02))
    if args.only in ("ood", "both"):
        out = args.ood_out or os.path.join(default_out_dir(), "exp2_ood_oml.png")
        plot_one("famous_le_100m", "Famous Large CNNs", "large", out, ylim=(0.4, 1.05))


if __name__ == "__main__":
    main()
