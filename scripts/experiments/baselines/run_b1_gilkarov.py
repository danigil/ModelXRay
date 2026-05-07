"""B1 standalone runner (Gilkarov et al. weights-feature reproduction).

Reproduces the protocol from
github.com/ArielCyber/AI_Model_Steganalysis/blob/main/classification.py:
for each X in [1, 23], concatenate (benign weights, X-LSB-attacked weights),
80/20 stratified split, fit classifier(s), record per-X (acc, recall, precision, f1).

By default runs the THREE classifier variants the paper / original repo both
mention so the comparison is honest:
  - XGBoost  (the paper's reproduction; B1 column in the paper)
  - HistGradientBoosting (original ArielCyber default for weights features)
  - RandomForest (original ArielCyber alternate for weights features)

Output: results/exp1/b1_gilkarov_per_x.csv with schema
    classifier, X, accuracy, recall, precision, f1, repeat
matching the schema described in results/SCHEMA.md.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from model_xray.baselines.b1_gilkarov import split_and_score
from model_xray.baselines.byte_attack import attacked_weights
from model_xray.data import paths as _paths
from model_xray.data.ghrp_zoos import load_mz_weights


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--mz-name", default="stl10",
                        help="GHRP zoo name (default: stl10 = SCZ STL10).")
    parser.add_argument("--classifiers", nargs="+",
                        default=["xgboost", "hgb", "rf"],
                        help="Classifier variants to run (default: all three).")
    parser.add_argument("--x-range", type=int, nargs="+", default=list(range(1, 24)))
    parser.add_argument("--n-repeats", type=int, default=1,
                        help="Number of (random_state-varied) repeats per X.")
    parser.add_argument("--n-models", type=int, default=None,
                        help="Subset of benign models to use (default: all). "
                             "5000 STL10 models -> ~10000 (benign+attack) instances.")
    parser.add_argument("--payload-file", default=None)
    parser.add_argument("--out", default=os.path.join(_paths.results_dir(), "exp1", "b1_gilkarov_per_x.csv"))
    args = parser.parse_args()

    payload = args.payload_file or _paths.get_payload_file()
    print(f"[B1] Loading SCZ {args.mz_name} weights ...")
    weights = load_mz_weights(args.mz_name).astype(np.float32)
    if args.n_models is not None:
        weights = weights[: args.n_models]
    print(f"[B1] Loaded {weights.shape}")

    rows = []
    for r in range(args.n_repeats):
        for x in args.x_range:
            print(f"[B1] X={x} repeat={r}")
            attacked = np.stack([attacked_weights(w, x=x, malware_bits_or_path=payload) for w in weights])
            X_full = np.concatenate([weights, attacked])
            y_full = np.concatenate([np.zeros(len(weights), dtype=int),
                                     np.ones(len(weights), dtype=int)])
            for clf_name in args.classifiers:
                m = split_and_score(X_full, y_full, classifier=clf_name, random_state=r)
                rows.append({"classifier": clf_name, "X": x, "repeat": r, **{k: v for k, v in m.items() if k != "classifier"}})

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"[B1] Wrote {args.out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
