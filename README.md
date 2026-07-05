# [ModelXRay](https://doi.org/10.1016/j.jisa.2026.104517)

Reproducibility artifact for the paper

> **[Model X-Ray: Detection of hidden malware in AI model weights using few shot learning](https://doi.org/10.1016/j.jisa.2026.104517)**

Abstract: AI model repositories such as Hugging Face and TensorFlow Hub have become an attractive surface for steganographic malware: attackers exploit the redundancy in float32 weights to embed payloads while preserving model accuracy. Existing AI-model steganalysis methods require tens of thousands of labeled training samples and only detect attacks at high embedding rates ( ≥ 50%), limiting their practical utility. We address both gaps with a few-shot learning approach. We propose a novel parameter-position-stable image representation, Grayscale-Fourpart (GF), that maps float32 weights to a square grayscale image, and pair it with a metric-learning few-shot CNN detector. The detector trains from as few as 6 model files and consistently flags attacks down to 25% embedding rate, with 6% in some cases. We benchmark against a seven-baseline matrix spanning two prior academic method, the canonical raw-byte 1D-CNN paradigm, and four threshold-based statistics, and identify the conjoint conditions under which the simpler baselines collapse and ours retains accuracy. The trained detectors transfer to novel out-of-distribution spread-spectrum attacks despite training only on LSB perturbations. A deployment-feasibility study shows that GF feature extraction scales linearly to 108 parameters at  ≈ 0.49 s and  ∼ 1.77 GiB peak memory,  ∼ 352 ×  faster than the strongest prior baseline at the same scale, making this, to our knowledge, the first AI-model steganalysis pipeline practical for repository-scale deployment. The full code framework, including baseline reproductions, is released as open-source.

This repository contains the full software framework:

- **Data creation**: GHRP small-CNN zoo and Famous Pretrained CNN ingestion;
  X-LSB-Attack-Fill embedding; MaleficNet attack regeneration.
- **Image representation**: GF and additional baselines.
- **FSL training & evaluation**: OSL CNN (100x100) and SRNet (256x256) with
  centroid + 1-NN classifiers.
- **Baseline reproductions B1-B7**:
  B1 Gilkarov et al., B2 Yin et al., B3 MalConv-lite, B4 Byte Autocorrelation,
  B5 Byte Entropy, B6 Histogram KL-Divergence, B7 Weight-Value Distribution.
- **Cached result CSVs** under `results/` so every paper figure / table can be
  regenerated in seconds without re-running any experiment.

## Quick start: regenerate every paper figure

```bash
pip install -r requirements.txt
python scripts/plots/plot_all.py
```

This produces five PNGs under
`./out/plots/` plus LaTeX tables. 
No GPU or zoo data required - the cached CSVs in `results/` are committed.

## Reproduce an experiment end-to-end

See [REPRODUCING.md](REPRODUCING.md) for the per-figure command table. In short:

```bash
# 1. Install the primary env
pip install -r requirements.txt

# 2. Download the model zoos and then set the data paths
bash scripts/data_creation/download_zoos.sh
export MODELXRAY_GHRP_DIR=/path/to/zoos
export MODELXRAY_RESNET_MZ_ROOT=/path/to/resnet18_zoo

# 3. Build datasets
python scripts/data_creation/01_create_scz_stl10.py
python scripts/data_creation/02_create_famous_small_cnns.py
python scripts/data_creation/03_create_famous_large_cnns.py
python scripts/data_creation/05_create_resnet18_tinyimagenet.py

# 4. Run an experiment (~hours; --quick for a quick test)
python scripts/experiments/run_exp1_scz_oml.py
python scripts/experiments/run_exp4_resnet18_yin.py

# 5. Regenerate plots
python scripts/plots/plot_all.py
```

The MaleficNet attack regeneration (D4, Experiment 2.5 OOD) requires an isolated
env - see [SETUP.md](SETUP.md) and `requirements-maleficnet.txt`.

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
```

## License & patent

This work is licensed under [CC BY-NC-ND 4.0](http://creativecommons.org/licenses/by-nc-nd/4.0/)
and is covered by US Provisional Patent Application No. 63/524,681. Code in
`external_code/` is vendored from upstream MaleficNet and GHRP repositories
under their original licenses.
