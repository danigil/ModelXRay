"""Runtime / memory study (Table 4, Table 5).

Measures wall-clock and peak-memory cost of feature extraction across the
parameter ladder (n_models, n_weights) in (10, 100, 1000, 10000) ** 2, for the
four image-representation backends used in the paper:

    s       (straight bytes; cheap reference)
    rgb     (3-byte RGB rep)
    bb      (bitbytes rep)
    phis    (Yin et al. 92-d feature vector — B2)

Output: results/runtime_memory/measure_time.csv with columns
    image_rep, time, peak_memory, n_models, n_weights, run_i
matching the original IngestModelZoo/results/time/measure_time.csv schema.

Optional dependency: memory_profiler (pip install memory-profiler). If
unavailable, peak memory column is filled with NaN.
"""

from __future__ import annotations

import argparse
import itertools
import os
import time
from typing import Iterable

import numpy as np
import pandas as pd

from model_xray.data.attack_pipeline import img_pp_xlsb_attack
from model_xray.data.paths import results_dir


try:
    from memory_profiler import memory_usage  # type: ignore
    _HAS_MEMPROF = True
except Exception:
    _HAS_MEMPROF = False


def _time_call(fn, *args, **kwargs) -> float:
    t0 = time.time()
    fn(*args, **kwargs)
    return time.time() - t0


def _peak_mem_mb(fn, *args, **kwargs) -> float:
    if not _HAS_MEMPROF:
        return float("nan")
    return float(memory_usage((fn, args, kwargs), max_usage=True))


def _row(image_rep: str, ws: np.ndarray, run_i: int) -> dict:
    n_models, n_weights = ws.shape
    t = _time_call(img_pp_xlsb_attack, ws, image_rep=image_rep, imsize=None, x=0)
    peak = _peak_mem_mb(img_pp_xlsb_attack, ws, image_rep=image_rep, imsize=None, x=0)
    return {
        "image_rep": image_rep,
        "time": t,
        "peak_memory": peak,
        "n_models": n_models,
        "n_weights": n_weights,
        "run_i": run_i,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--repeats", type=int, default=5,
                        help="Number of repeated runs (paper: 5).")
    parser.add_argument("--ns", type=int, nargs="+", default=[10, 100, 1000, 10000],
                        help="n_models ladder (paper: 10..1e4).")
    parser.add_argument("--ms", type=int, nargs="+", default=[10, 100, 1000, 10000],
                        help="n_weights ladder per model (paper: 10..1e4).")
    parser.add_argument("--image-reps", nargs="+", default=["s", "rgb", "bb", "phis"])
    parser.add_argument("--out", default=os.path.join(results_dir(), "runtime_memory", "measure_time.csv"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    rows = []
    for run_i in range(args.repeats):
        for n, m in itertools.product(args.ns, args.ms):
            ws = rng.standard_normal(size=(n, m)).astype(np.float32)
            for rep in args.image_reps:
                print(f"  run {run_i} | image_rep={rep} | n={n}, m={m}")
                rows.append(_row(rep, ws, run_i))

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {args.out} ({len(df)} rows)")
    if not _HAS_MEMPROF:
        print("(memory_profiler not installed; peak_memory column is NaN. "
              "Install with: pip install memory-profiler)")


if __name__ == "__main__":
    main()
