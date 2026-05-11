# `results/` schema

Each subdirectory holds the cached CSVs that the matching `scripts/plots/` script
consumes to regenerate one paper figure or table. End-to-end re-runs in
`scripts/experiments/` overwrite these files in place; the committed copies are
what produced the figures in the published PDF.

The column conventions for every CSV below are the canonical schema documented
in [`model_xray/data/_schemas.py`](../model_xray/data/_schemas.py); runner
outputs and plot inputs share that schema, so reruns plot without translation.

## `results/exp1/` — Experiment 1, SCZ STL10 OML (Figure 4)

| File | Producer |
|---|---|
| `fsl_osl.csv` | `scripts/experiments/run_exp1_scz_oml.py --methods fsl` |
| `b1_gilkarov_per_x.csv` | `scripts/experiments/run_exp1_scz_oml.py --methods b1` |
| `b3_malconv.csv` | `scripts/experiments/run_exp1_scz_oml.py --methods b3` |
| `b4_b7_threshold.csv` | `scripts/experiments/run_exp1_scz_oml.py --methods thresholds` |

## `results/exp2/` — Experiment 2, Famous Small + Large CNNs (figs:exp2_*)

| File | Notes |
|---|---|
| `fsl_osl_small.csv`, `fsl_osl_large.csv` | OSL CNN (100x100), Section 4.7.1 / 4.7.4 |
| `fsl_srnet_small.csv`, `fsl_srnet_large.csv` | SRNet (256x256) |
| `b3_malconv_small.csv`, `b3_malconv_large.csv` | MalConv-lite per-X repeats |
| `b3_malconv_crossx_small.csv`, `b3_malconv_crossx_large.csv` | MalConv-lite cross-X (anchor `X_hat`, evaluate at all `X`); feeds the AL plot |
| `b4_b7_threshold_small.csv`, `b4_b7_threshold_large.csv` | naive baseline per-X repeats |
| `threshold_features_small.npz`, `threshold_features_large.npz` | cached per-X feature scores feeding the AL plot reuse |

## `results/exp4/` — Experiment 4, ResNet18-TinyImageNet vs Yin (Figure 8)

| File | Method |
|---|---|
| `gf_xgboost.csv` | "Ours - GF + XGBoost" (XGBoost on GF pixels, 5-fold x 5-rep CV) |
| `gf_1nn.csv` | "Ours - GF + 1NN" (1-NN on GF pixels, same protocol) |
| `b2_yin.csv` | B2 Yin XGBoost on the 92-d feature vector, all `X` in one CSV |
| `b3_malconv.csv` | B3 MalConv-lite (raw bytes, 512KB window) |
| `b5_b7_threshold.csv` | B5 (Byte Entropy) and B7 (Weight-Value Distribution) |

CSVs contain one row per CV fold; the plot script aggregates by `X` with
mean ± 95% CI bands.

## `results/exp2_5/` — Experiment 2.5, MaleficNet OOD (Table 2)

Populated only by `scripts/experiments/run_exp2_5_maleficnet_ood.py`. Not
shipped pre-cached in this initial release because the table values are
embedded directly in the paper and the experiment requires the
trained Exp 2 FSL detector + the MaleficNet image dataset (D4) to reproduce.

## `results/runtime_memory/` — Deployment study (Table 4, Table 5)

| File | Producer |
|---|---|
| `measure_time.csv` | `scripts/experiments/run_runtime_memory.py` |
