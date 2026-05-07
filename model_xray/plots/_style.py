"""Shared plot styles + small CSV-aggregation helpers for `scripts/plots/`.

Keeps every per-figure script under ~150 lines while ensuring the regenerated
PNGs match the colors/markers of the published paper figures exactly.
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import seaborn as sns


# -------------------- shared style palettes --------------------

FSL_STYLES = {
    "OSL CNN (Centroid)": dict(color="#3C6E99", linestyle="-",  label="OSL CNN (Centroid)"),
    "OSL CNN (1NN)":      dict(color="#E09A3A", linestyle="--", label="OSL CNN (1NN)"),
    "SRNet (Centroid)":   dict(color="#E09A3A", linestyle="--", label="SRNet (Centroid)"),
    "SRNet (1NN)":        dict(color="#BC3D40", linestyle="--", label="SRNet (1NN)"),
}

# exp2_oml uses an alternate FSL palette to keep the four curves distinguishable
FSL_STYLES_EXP2 = {
    "OSL CNN (Centroid)": dict(color="#3C6E99", linestyle="-",  label="OSL CNN (Centroid)"),
    "SRNet (Centroid)":   dict(color="#E09A3A", linestyle="--", label="SRNet (Centroid)"),
    "OSL CNN (1NN)":      dict(color="#4A9A66", linestyle=":",  label="OSL CNN (1NN)"),
    "SRNet (1NN)":        dict(color="#BC3D40", linestyle="--", label="SRNet (1NN)"),
}

GILKAROV_STYLE = dict(color="#3C9F3C", linestyle=":", marker="None",
                      label="B1: Gilkarov et al.")

MALCONV_STYLE = dict(color="#1F78B4", linestyle="--", marker="x",
                     label="B3: MalConv-lite (raw bytes)")

NAIVE_STYLES = {
    "byte_autocorr":     dict(color="#A7A9AC", linestyle="-.", marker="o", label="B4: Byte Autocorrelation"),
    "byte_entropy":      dict(color="#6A6B6E", linestyle=":",  marker="s", label="B5: Byte Entropy"),
    "histogram_kl":      dict(color="#8B5EAA", linestyle="-.", marker="^", label="B6: Histogram KL-Divergence"),
    "weight_value_dist": dict(color="#C48BAE", linestyle=":",  marker="v", label="B7: Weight-Value Distribution"),
}

OURS_STYLES_EXP4 = {
    "Ours - GF + XGBoost": dict(color="#2E3D8F", linestyle="-",  label="Ours - GF + XGBoost"),
    "Ours - GF + 1NN":     dict(color="#6F93C9", linestyle="--", label="Ours - GF + 1NN"),
}
B2_STYLE = dict(color="#C2533D", linestyle="-", label="B2: Yin et al. + XGBoost")


# -------------------- aggregation helpers --------------------

def agg_with_ci(rows: pd.DataFrame, x_col: str, val_col: str) -> pd.DataFrame:
    """Mean ± 95% CI of the mean over `val_col`, grouped by `x_col`."""
    g = (rows.groupby(x_col)[val_col]
         .agg(mean="mean", std="std", n="count").reset_index())
    g["std"] = g["std"].fillna(0)
    g["half_ci"] = 1.96 * g["std"] / np.sqrt(g["n"].clip(lower=1))
    return g


def add_ci(df: pd.DataFrame, n_col: str = "n_seeds") -> pd.DataFrame:
    df = df.copy()
    df["std"] = df["std"].fillna(0)
    df["half_ci"] = 1.96 * df["std"] / np.sqrt(df[n_col].clip(lower=1))
    return df


# -------------------- output path helper --------------------

def repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))


def default_out_dir() -> str:
    return os.path.join(repo_root(), "out", "plots")


def init_axes(figsize=(8.5, 5.2)):
    """Apply seaborn theme + return (fig, ax) with consistent paper styling."""
    import matplotlib.pyplot as plt
    sns.set_theme(style="whitegrid", rc={"grid.linewidth": 0.5})
    return plt.subplots(figsize=figsize)


def plot_band(ax, curve: pd.DataFrame, style: dict, x_col: str = "X",
              *, linewidth: float, alpha: float, marker_size: int = 0):
    plot_kwargs = {k: v for k, v in style.items() if k != "marker"}
    ax.plot(curve[x_col], curve["mean"], linewidth=linewidth, markersize=marker_size,
            marker=style.get("marker", None), **plot_kwargs)
    if curve["half_ci"].sum() > 0:
        lower = (curve["mean"] - curve["half_ci"]).clip(lower=0, upper=1)
        upper = (curve["mean"] + curve["half_ci"]).clip(lower=0, upper=1)
        ax.fill_between(curve[x_col], lower, upper,
                        color=style["color"], alpha=alpha, linewidth=0)


def finalize(ax, *, title: str, ylim=(0.45, 1.02), ylabel: str | None = None):
    ax.axhline(0.5, color="k", linestyle=":", linewidth=0.7, alpha=0.5)
    ax.set_xlabel("Model LSB")
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(list(range(1, 24)))
    ax.tick_params(axis="x", rotation=45)
    ax.set_ylim(*ylim)
    ax.set_xlim(0.5, 23.5)
    ax.grid(True, which="major", alpha=0.5)
    ax.legend(title="Model Type", loc="center left",
              bbox_to_anchor=(1.02, 0.5), fontsize=8, frameon=True)


def save_fig(fig, out_path: str, dpi: int = 160):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    print(f"  wrote {out_path}")
