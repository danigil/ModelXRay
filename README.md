# ModelXRay

Reproducibility artifact for the paper

> **Model X-Ray: Detection of Hidden Malware in AI Model Weights using Few Shot Learning**
> JISAS 2026, paper-ID `JISAS-D-25-03332`.
> [arXiv:2409.19310](https://doi.org/10.48550/arXiv.2409.19310)

The paper proposes a parameter-position-stable image representation of float32
neural-network weights (Grayscale-Fourpart, GF) and pairs it with a metric-
learning few-shot CNN detector. The detector trains from as few as **6 model
files** and consistently flags X-LSB-Attack-Fill at attack severity X≥25%
(severity X≥6% in some cases), and transfers to spread-spectrum MaleficNet
attacks despite training only on LSB perturbations.

This repository contains the full software framework:

- **Data creation**: GHRP small-CNN zoo and Famous Pretrained CNN ingestion;
  X-LSB-Attack-Fill embedding; MaleficNet attack regeneration.
- **Image representation**: GF and the alternate `rgb`, `bb`, `s` baselines.
- **FSL training & evaluation**: OSL CNN (100x100) and SRNet (256x256) with
  centroid + 1-NN classifiers (Section sec:model_train).
- **Baseline reproductions B1-B7** (Section sec:baseline):
  B1 Gilkarov et al., B2 Yin et al., B3 MalConv-lite, B4 Byte Autocorrelation,
  B5 Byte Entropy, B6 Histogram KL-Divergence, B7 Weight-Value Distribution.
- **Cached result CSVs** under `results/` so every paper figure / table can be
  regenerated in seconds without re-running any experiment.

## Quick start: regenerate every paper figure

```bash
pip install -r requirements.txt
python scripts/plots/plot_all.py
```

This produces five PNGs (matching `latex_code/plots_ieee/exp{1,2,4}_*.png`) under
`./out/plots/` plus the LaTeX source for `tab:exp_time` and `tab:exp_memory`.
No GPU or zoo data required — the cached CSVs in `results/` are committed.

## Reproduce an experiment end-to-end

See [REPRODUCING.md](REPRODUCING.md) for the per-figure command table. In short:

```bash
# 1. Install the primary env
pip install -r requirements.txt

# 2. Set data paths and download the model zoos
export MODELXRAY_GHRP_DIR=/path/to/zoos
export MODELXRAY_RESNET_MZ_ROOT=/path/to/resnet18_zoo
bash scripts/data_creation/download_zoos.sh

# 3. Build datasets
python scripts/data_creation/01_create_scz_stl10.py
python scripts/data_creation/02_create_famous_small_cnns.py
python scripts/data_creation/03_create_famous_large_cnns.py
python scripts/data_creation/05_create_resnet18_tinyimagenet.py

# 4. Run an experiment (~hours; --quick for a smoke test)
python scripts/experiments/run_exp1_scz_oml.py
python scripts/experiments/run_exp4_resnet18_yin.py

# 5. Regenerate plots
python scripts/plots/plot_all.py
```

The MaleficNet attack regeneration (D4, Experiment 2.5 OOD) requires an isolated
env — see [SETUP.md](SETUP.md) and `requirements-maleficnet.txt`.

## Layout

```
model_xray/                        # core library
  procedures/                      # GF, X-LSB-Fill, MaleficNet, image preproc
  baselines/                       # B1-B7 implementations + shared XGBoost config
  data/                            # paths, attack pipeline, zoo loaders
  fsl/                             # FSL train + evaluate (centroid + 1-NN + WM)
  models/                          # OSL CNN (siamese) + SRNet
  configs/                         # enums + Pydantic dataclasses
  plots/                           # shared plot helpers
  utils/                           # general helpers (byte decomposition, etc.)
external_code/                     # vendored: maleficnet, ghrp
scripts/
  data_creation/                   # 01-05 entry points + download_zoos.sh
  experiments/                     # one runner per paper experiment
  plots/                           # one regenerator per figure + plot_all.py
results/                           # cached CSVs that produce the paper figures
```

## Citation

```bibtex
@article{gilkarov2026modelxray,
  title   = {Model X-Ray: Detection of Hidden Malware in AI Model Weights using Few Shot Learning},
  author  = {Gilkarov, Daniel and ...},
  journal = {Journal of Information Security and Applications},
  year    = {2026},
}
```

## License & patent

This work is licensed under [CC BY-NC-ND 4.0](http://creativecommons.org/licenses/by-nc-nd/4.0/)
and is covered by US Provisional Patent Application No. 63/524,681. Code in
`external_code/` is vendored from upstream MaleficNet and GHRP repositories
under their original licenses.
