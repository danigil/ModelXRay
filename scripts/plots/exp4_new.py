"""Regenerate exp4_new.png — Experiment 4, ResNet18-TinyImageNet (paper Figure 8)."""

from __future__ import annotations

import argparse
import os

import pandas as pd

from model_xray.plots._style import (
    B2_STYLE, MALCONV_STYLE, NAIVE_STYLES, OURS_STYLES_EXP4,
    agg_with_ci, default_out_dir, finalize, init_axes, plot_band, repo_root, save_fig,
)


RESULTS_DIR = os.path.join(repo_root(), "results", "exp4")


def _clf_curve(name: str) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(RESULTS_DIR, name))
    return agg_with_ci(df, "X", "accuracy")


def _yin_curve() -> pd.DataFrame:
    p = os.path.join(RESULTS_DIR, "b2_yin.csv")
    if not os.path.exists(p):
        return pd.DataFrame()
    return agg_with_ci(pd.read_csv(p), "X", "accuracy")


def _naive_curve(baseline_name: str) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(RESULTS_DIR, "b5_b7_threshold.csv"))
    df = df[df["baseline"] == baseline_name]
    return agg_with_ci(df, "X", "accuracy")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=os.path.join(default_out_dir(), "exp4_new.png"))
    args = parser.parse_args()

    fig, ax = init_axes()

    for name, fname in (("Ours - GF + XGBoost", "gf_xgboost.csv"),
                        ("Ours - GF + 1NN",     "gf_1nn.csv")):
        path = os.path.join(RESULTS_DIR, fname)
        if not os.path.exists(path):
            print(f"  skip missing {path}")
            continue
        plot_band(ax, _clf_curve(fname), OURS_STYLES_EXP4[name], linewidth=2.0, alpha=0.10)

    yin = _yin_curve()
    if not yin.empty:
        plot_band(ax, yin, B2_STYLE, linewidth=2.0, alpha=0.10)

    mc_path = os.path.join(RESULTS_DIR, "b3_malconv.csv")
    if os.path.exists(mc_path):
        mc = _clf_curve("b3_malconv.csv")
        if not mc.empty:
            plot_band(ax, mc, MALCONV_STYLE, linewidth=1.5, alpha=0.10, marker_size=4)

    for baseline_name in ("byte_entropy", "weight_value_dist"):
        c = _naive_curve(baseline_name)
        if c.empty:
            continue
        plot_band(ax, c, NAIVE_STYLES[baseline_name], linewidth=1.3, alpha=0.12, marker_size=4)

    finalize(ax, title="Model Zoo = tiny-imagenet ResNet18",
             ylim=(0.0, 1.02),
             ylabel="Test Accuracy (Benign + Malicious) (X=Model LSB)")
    # Override legend title to "Method" for this figure
    ax.legend(title="Method", loc="center left",
              bbox_to_anchor=(1.02, 0.5), fontsize=8, frameon=True)
    save_fig(fig, args.out)


if __name__ == "__main__":
    main()
