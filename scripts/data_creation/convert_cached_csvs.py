"""One-shot converter: cached paper CSVs → canonical schema (see _schemas.py).

Reads the historical schemas under `results/{exp1,exp2,exp4}/`, rewrites them
in-place to the canonical schema declared in `model_xray/data/_schemas.py`,
and removes the now-obsolete subdirectory layout (`exp4/b2_yin_per_x/`).

Idempotent: re-running on already-converted files is a no-op (it detects the
canonical column set on disk and skips).
"""
from __future__ import annotations

import os
import shutil
from glob import glob

import pandas as pd

from model_xray.data.paths import results_dir


CANONICAL_FSL_COLS = {"repeat", "X_hat", "X", "centroid", "nn"}
CANONICAL_BASE_COLS = {"repeat", "baseline", "X", "accuracy"}


def _is_already_canonical(path: str) -> bool:
    try:
        cols = set(pd.read_csv(path, nrows=0).columns)
    except Exception:
        return False
    return CANONICAL_FSL_COLS.issubset(cols) or CANONICAL_BASE_COLS.issubset(cols)


def _convert_fsl(path: str):
    if _is_already_canonical(path):
        print(f"  skip (already canonical): {path}")
        return
    df = pd.read_csv(path)
    out = pd.DataFrame({
        "repeat": df["run num"].astype(int),
        "X_hat": df["model_lsb"].astype(int),
        "X": df["lsb"].astype(int),
        "centroid": df["test_acc_centroid"].astype(float),
        "nn": df["test_acc_nn"].astype(float),
    })
    # Exp 2 ships per-eval-set rows interleaved under the `mc` column; preserve
    # it as `eval_set` so plot scripts can filter (id vs ood vs maleficnet).
    if "mc" in df.columns and df["mc"].nunique() > 1:
        out["eval_set"] = df["mc"].astype(str)
    out.to_csv(path, index=False)
    print(f"  wrote {path}  ({len(out)} rows)")


def _convert_b1(path: str):
    if _is_already_canonical(path):
        print(f"  skip (already canonical): {path}")
        return
    df = pd.read_csv(path)
    out = pd.DataFrame({
        "repeat": df["repeat"].astype(int),
        "baseline": df["classifier"].astype(str),
        "X": df["X"].astype(int),
        "accuracy": df["accuracy"].astype(float),
    })
    out.to_csv(path, index=False)
    print(f"  wrote {path}  ({len(out)} rows)")


def _convert_threshold_or_b3(path: str):
    """For files with cols `seed, dataset, baseline, X, ..., acc_mean_test, ...`."""
    if _is_already_canonical(path):
        print(f"  skip (already canonical): {path}")
        return
    df = pd.read_csv(path)
    out = pd.DataFrame({
        "repeat": df["seed"].astype(int),
        "baseline": df["baseline"].astype(str),
        "X": df["X"].astype(int),
        "accuracy": df["acc_mean_test"].astype(float),
    })
    if "train_archs" in df.columns:
        out["train_archs"] = df["train_archs"].astype(str)
    out.to_csv(path, index=False)
    print(f"  wrote {path}  ({len(out)} rows)")


def _convert_b3_crossx(path: str):
    """`b3_malconv_crossx_*.csv` cols: seed, dataset, baseline, X_hat, lsb, train_archs, n_test, acc."""
    df = pd.read_csv(path)
    needed = {"repeat", "baseline", "X_hat", "X", "accuracy"}
    if needed.issubset(set(df.columns)):
        print(f"  skip (already canonical): {path}")
        return
    out = pd.DataFrame({
        "repeat": df["seed"].astype(int),
        "baseline": df["baseline"].astype(str),
        "X_hat": df["X_hat"].astype(int),
        "X": df["lsb"].astype(int),
        "accuracy": df["acc"].astype(float),
    })
    if "train_archs" in df.columns:
        out["train_archs"] = df["train_archs"].astype(str)
    out.to_csv(path, index=False)
    print(f"  wrote {path}  ({len(out)} rows)")


def _convert_exp4_clf(path: str):
    """`gf_xgboost.csv`, `gf_1nn.csv`, `b3_malconv.csv` (exp4) cols: clf, acc, ds_name, split_idx, n_splits, n_repeats, x, ...

    Pre-conversion CSVs interleaved iid_train and iid_test rows under one
    `ds_name` column; the plot script only ever consumed `ds_name=="iid_test"`.
    Filter to test rows here so the canonical CSV needs no `ds_name` filter.
    """
    if _is_already_canonical(path):
        print(f"  skip (already canonical): {path}")
        return
    df = pd.read_csv(path)
    if "ds_name" in df.columns:
        df = df[df["ds_name"] == "iid_test"].copy()
    fold_col = "split_idx" if "split_idx" in df.columns else None
    out = pd.DataFrame({
        "repeat": df.get("split_idx", 0).astype(int),
        "baseline": df["clf"].astype(str),
        "X": df["x"].astype(int),
        "accuracy": df["acc"].astype(float),
    })
    if fold_col:
        out["fold"] = df[fold_col].astype(int)
    out.to_csv(path, index=False)
    print(f"  wrote {path}  ({len(out)} rows)")


def _convert_exp4_b5b7(path: str):
    """`b5_b7_threshold.csv` cols: baseline, X, split_idx, n_splits, n_benign_test, n_malicious_test, threshold, acc_benign_test, acc_malicious_test, acc_mean_test, mz_name, payload."""
    if _is_already_canonical(path):
        print(f"  skip (already canonical): {path}")
        return
    df = pd.read_csv(path)
    out = pd.DataFrame({
        "repeat": df["split_idx"].astype(int),
        "baseline": df["baseline"].astype(str),
        "X": df["X"].astype(int),
        "accuracy": df["acc_mean_test"].astype(float),
    })
    if "split_idx" in df.columns:
        out["fold"] = df["split_idx"].astype(int)
    out.to_csv(path, index=False)
    print(f"  wrote {path}  ({len(out)} rows)")


def _convert_exp4_b2_subdir(subdir: str, target_path: str):
    """Collapse `b2_yin_per_x/{X}.csv` files into a single `b2_yin.csv` with col X.

    Each per-X file follows the same `ds_name`-mixed schema as the other Exp 4
    classifier CSVs; filter to `ds_name == "iid_test"` here.
    """
    if os.path.exists(target_path) and _is_already_canonical(target_path):
        print(f"  skip (already canonical): {target_path}")
        return
    files = sorted(glob(os.path.join(subdir, "*.csv")))
    if not files:
        print(f"  skip (no inputs): {subdir}")
        return
    frames = []
    for fp in files:
        df = pd.read_csv(fp)
        x = int(os.path.basename(fp).split(".")[0])
        df["x"] = x
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    if "ds_name" in df.columns:
        df = df[df["ds_name"] == "iid_test"].copy()
    fold_col = "split_idx" if "split_idx" in df.columns else None
    out = pd.DataFrame({
        "repeat": df["split_idx"].astype(int),
        "baseline": df["clf"].astype(str),
        "X": df["x"].astype(int),
        "accuracy": df["acc"].astype(float),
    })
    if fold_col:
        out["fold"] = df[fold_col].astype(int)
    out.to_csv(target_path, index=False)
    print(f"  wrote {target_path}  ({len(out)} rows from {len(files)} per-X files)")
    # Move the now-obsolete subdir aside (don't delete; user can git-rm later).
    obsolete = subdir + ".obsolete"
    if os.path.exists(obsolete):
        shutil.rmtree(obsolete)
    os.rename(subdir, obsolete)
    print(f"  moved {subdir} -> {obsolete}  (delete with `git rm -r`)")


def main():
    rd = results_dir()

    # Exp 1
    print("== Exp 1 ==")
    _convert_fsl(os.path.join(rd, "exp1", "fsl_osl.csv"))
    _convert_b1(os.path.join(rd, "exp1", "b1_gilkarov_per_x.csv"))
    _convert_threshold_or_b3(os.path.join(rd, "exp1", "b3_malconv.csv"))
    _convert_threshold_or_b3(os.path.join(rd, "exp1", "b4_b7_threshold.csv"))

    # Exp 2
    print("== Exp 2 ==")
    for tag in ("small", "large"):
        for arch in ("osl", "srnet"):
            p = os.path.join(rd, "exp2", f"fsl_{arch}_{tag}.csv")
            if os.path.exists(p):
                _convert_fsl(p)
        for fname in (f"b3_malconv_{tag}.csv", f"b4_b7_threshold_{tag}.csv"):
            p = os.path.join(rd, "exp2", fname)
            if os.path.exists(p):
                _convert_threshold_or_b3(p)
        p = os.path.join(rd, "exp2", f"b3_malconv_crossx_{tag}.csv")
        if os.path.exists(p):
            _convert_b3_crossx(p)

    # Exp 4
    print("== Exp 4 ==")
    for fname in ("gf_xgboost.csv", "gf_1nn.csv", "b3_malconv.csv"):
        p = os.path.join(rd, "exp4", fname)
        if os.path.exists(p):
            _convert_exp4_clf(p)
    p = os.path.join(rd, "exp4", "b5_b7_threshold.csv")
    if os.path.exists(p):
        _convert_exp4_b5b7(p)
    subdir = os.path.join(rd, "exp4", "b2_yin_per_x")
    target = os.path.join(rd, "exp4", "b2_yin.csv")
    if os.path.isdir(subdir):
        _convert_exp4_b2_subdir(subdir, target)


if __name__ == "__main__":
    main()
