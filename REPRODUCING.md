# Reproducing the paper

This document maps every quantitative artifact in the paper *"Model X-Ray: Detection of Hidden Malware in AI Model Weights using Few Shot Learning"* (JISAS-D-25-03332) to the exact command that regenerates it.

There are two reproduction modes:

- **Plot-only** (seconds): regenerate the paper's figures and tables from the cached CSVs that ship with this repo.
- **End-to-end** (hours-to-days): rebuild the datasets from scratch and rerun every experiment.

## Quick start (plot-only)

```bash
pip install -r requirements.txt
python scripts/plots/plot_all.py
```

This regenerates the five paper figures and prints the LaTeX source for both timing tables, all from the CSVs under `results/`. No GPU or zoo data required.

## Per-artifact reproduction

| Paper artifact | Plot/table command | End-to-end command |
|---|---|---|
| `fig:exp1` (`exp1_id_oml.png`) | `python scripts/plots/exp1_id_oml.py` | `python scripts/experiments/run_exp1_scz_oml.py` |
| `fig:exp2_oml_id` (`exp2_id_oml.png`) | `python scripts/plots/exp2_oml.py` | `python scripts/experiments/run_exp2_famous_cnns.py` |
| `fig:exp2` (`exp2_id_al.png`) | `python scripts/plots/exp2_al.py` | `python scripts/experiments/run_exp2_famous_cnns.py` |
| `fig:exp2_oml` (`exp2_ood_oml.png`) | `python scripts/plots/exp2_oml.py` | `python scripts/experiments/run_exp2_famous_cnns.py` |
| `fig:exp4` (`exp4_new.png`) | `python scripts/plots/exp4_new.py` | `python scripts/experiments/run_exp4_resnet18_yin.py` |
| `tab:exp_time` | `python scripts/plots/tables_time_memory.py --table time` | `python scripts/experiments/run_runtime_memory.py` |
| `tab:exp_memory` | `python scripts/plots/tables_time_memory.py --table memory` | `python scripts/experiments/run_runtime_memory.py` |
| `tab:malefic` (Exp 2.5) | (printed by `run_exp2_5_maleficnet_ood.py`) | `python scripts/experiments/run_exp2_5_maleficnet_ood.py` |

Each end-to-end script supports `--quick` for a small smoke-test subset.

## Data setup (for end-to-end runs only)

End-to-end reruns require the GHRP model zoos and (for MaleficNet OOD) attacked model files. Set the env vars listed in [SETUP.md](SETUP.md), at minimum:

```bash
export MODELXRAY_GHRP_DIR=/path/to/zoos               # GHRP zoo parent dir (D1, D2, D3)
export MODELXRAY_RESNET_MZ_ROOT=/path/to/resnet18     # ResNet18 checkpoint root (D5)
export MODELXRAY_MALEFICNET_DIR=/path/to/maleficnet_imgs   # D4 image cache
```

Then fetch the GHRP zoos:

```bash
bash scripts/data_creation/download_zoos.sh
```

Build datasets:

```bash
python scripts/data_creation/01_create_scz_stl10.py            # D1 (Exp 1)
python scripts/data_creation/02_create_famous_small_cnns.py    # D2 (Exp 2 ID, Exp 2.5 train)
python scripts/data_creation/03_create_famous_large_cnns.py    # D3 (Exp 2 OOD)
python scripts/data_creation/05_create_resnet18_tinyimagenet.py# D5 (Exp 4)
# D4 requires the maleficnet env (see requirements-maleficnet.txt):
python scripts/data_creation/04_create_maleficnet_attacks.py   # D4 (Exp 2.5 OOD test)
```

## Datasets and baselines reference

Datasets:

- **D1** SCZ STL10 GHRP zoo (small CNNs, ≈5000 models)
- **D2** Famous Small CNNs (Keras pretrained, ≤10M params)
- **D3** Famous Large CNNs (Keras pretrained, ≤100M params)
- **D4** MaleficNet attacked CNNs (DenseNet121, ResNet50, ResNet101)
- **D5** ResNet18-TinyImageNet GHRP zoo
- **D6** Synthetic float32 tensors at sizes 10²-10⁸ (timing only)

Baselines (`sec:baselines_academic`, `sec:baselines_naive`):

- **B1** Gilkarov et al. — XGBoost on flattened weights
- **B2** Yin et al. — 92-dim NIST stats (φ₁-φ₄ × 23 bit positions) + XGBoost
- **B3** MalConv-lite — raw-byte 1D-CNN
- **B4** Byte autocorrelation threshold
- **B5** Byte entropy threshold
- **B6** Histogram KL-divergence threshold
- **B7** Weight-value distribution threshold

## Result CSV schema

See `results/SCHEMA.md` for the column conventions of each cached CSV.
