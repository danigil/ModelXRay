# Setup

## Primary environment

Tested on Python 3.11.9, Ubuntu 20.04, CUDA 11.8.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

This pulls TensorFlow 2.13 + Torch 2.4 (cu118) + Keras + scikit-learn +
xgboost + matplotlib/seaborn + Pydantic + Pillow + scikit-image, plus
`bitstring`, `tqdm`, `transformers`, `huggingface-hub`, and `h5py`.

**Optional**: `pip install memory-profiler` to populate the `peak_memory`
column produced by `scripts/experiments/run_runtime_memory.py`.

## MaleficNet environment (D4 attack regeneration only)

The vendored MaleficNet implementation has upstream pins that conflict with the
primary stack. If you need to regenerate the MaleficNet attacked-model dataset
(Experiment 2.5), create a separate venv and install:

```bash
python -m venv .venv-maleficnet && source .venv-maleficnet/bin/activate
pip install -r requirements-maleficnet.txt
pip install -e .
```

Loading the pre-generated MaleficNet image cache (`maleficnet_imgs<gf50>.npy`)
works fine in the primary env via `model_xray.data.maleficnet.ret_maleficnet_data`;
only the attack-generation step needs the isolated env.

## Environment variables

Required only for end-to-end re-runs (the plot-only path needs none):

| Variable | Used by | Purpose |
|---|---|---|
| `MODELXRAY_GHRP_DIR` | scripts 01-03, exp1/2/2.5 | parent dir of GHRP zoo subdirs |
| `MODELXRAY_RESNET_MZ_ROOT` | script 05, exp4 | parent dir of `tiny-imagenet_resnet18_*` checkpoint trees |
| `MODELXRAY_MALEFICNET_DIR` | exp2.5 | dir holding `maleficnet_imgs*.npy` + metadata |
| `MODELXRAY_MALEFICNET_DOWNLOADS` | script 04 | injector cache dir for attacked `.pt` checkpoints |
| `MODELXRAY_MALEFICNET_PAYLOADS` | script 04 | dir of malware payload binaries |
| `MODELXRAY_PAYLOAD_FILE` | exp1, exp2 | optional path to a malware payload binary; defaults to a uniform random payload |
| `HF_HOME` | optional | HuggingFace cache dir; default `~/.cache/huggingface` |

Example:

```bash
export MODELXRAY_GHRP_DIR=/data/modelxray/ghrp_zoos
export MODELXRAY_RESNET_MZ_ROOT=/data/modelxray/resnet18
export MODELXRAY_MALEFICNET_DIR=/data/modelxray/maleficnet_imgs
```

## Smoke test

Plot-only path (no zoos required, ~30 seconds):

```bash
python scripts/plots/plot_all.py
ls out/plots/
```

End-to-end smoke test (~5 minutes on one GPU):

```bash
python scripts/experiments/run_exp1_scz_oml.py --quick
```

## ZenML

This repository previously used ZenML for pipeline orchestration. The public
artifact dropped that dependency in favor of plain numbered Python scripts to
keep the reproduction pipeline lean. The historical ZenML pipelines remain in
the `zenml` branch of the upstream repository if needed for reference.
