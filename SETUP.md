# Setup

## Primary environment

Tested on Python 3.11.9, Ubuntu 20.04, CUDA 11.8.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
# Required only for GHRP zoo loading (D1 SCZ STL10, D5 ResNet18-TinyImageNet).
# `dataset.pt` files are pickled with `ghrp.*` module references, so the
# vendored upstream package must be importable at unpickle time.
pip install -e external_code/ghrp/
```

This pulls TensorFlow 2.15 (CPU + auto-GPU) + Torch 2.4 (cu121) + Keras 2.15 +
scikit-learn + xgboost + matplotlib/seaborn + Pydantic + Pillow +
scikit-image, plus `bitstring`, `tqdm`, `transformers`, `huggingface-hub`,
and `h5py`. Verified end-to-end on Python 3.11.9.

If you need TF on GPU, replace `tensorflow==2.15.1` with
`tensorflow[and-cuda]==2.15.1` and reinstall — note that the `[and-cuda]`
extra pins `nvidia-cublas-cu12==12.2.5.6`, which conflicts with torch 2.4's
`12.1.3.1`. To use both on GPU, install torch in a separate venv from TF.

### Verified GPU recipe (TF on GPU + torch on CPU; no system-CUDA changes)

This works on a host with system CUDA 11.8 because TF brings its own CUDA 12
libs via `tensorflow[and-cuda]`:

```bash
python -m venv .venv-gpu && source .venv-gpu/bin/activate
pip install --upgrade pip wheel
pip install 'tensorflow[and-cuda]==2.15.1' keras==2.15.0
pip install 'torch==2.4.0+cpu' 'torchvision==0.19.0+cpu' \
            --index-url https://download.pytorch.org/whl/cpu
pip install -r <(grep -v '^tensorflow\|^torch\|^keras' requirements.txt)
pip install -e . -e external_code/ghrp/
```

Verify:

```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
# -> [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

**Optional**: `pip install memory-profiler` to populate the `peak_memory`
column produced by `scripts/experiments/run_runtime_memory.py`.

## MaleficNet environment (D4 attack regeneration only)

The vendored MaleficNet implementation has upstream pins that conflict with the
primary stack. If you need to regenerate the MaleficNet attacked-model dataset
(Experiment 2.5), create a separate venv and install. `pyldpc==0.7.9` is a
source-only release whose `setup.py` imports `numpy` at build time, so install
numpy + wheel first and then disable build isolation for pyldpc:

```bash
python -m venv .venv-maleficnet && source .venv-maleficnet/bin/activate
pip install --upgrade pip wheel setuptools
pip install numpy==1.26.4 cython
pip install --no-build-isolation pyldpc==0.7.9
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

## Quick test

Plot-only path (no zoos required, ~30 seconds):

```bash
python scripts/plots/plot_all.py
ls out/plots/
```

End-to-end quick test (~5 minutes on one GPU):

```bash
python scripts/experiments/run_exp1_scz_oml.py --quick
```