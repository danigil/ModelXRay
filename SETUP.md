# Steps

## Configure Conda Environment

```bash
    conda create -n modelxray python=3.11.9 -y
    conda activate modelxray
    pip install zenml[server]==0.64.0
    pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu118
    pip install tensorflow[and-cuda]==2.13.0
    pip install matplotlib==3.9.1.post1 scipy==1.14.0 transformers==4.31.0 pandas==2.2.2 scikit-learn==1.5.1 seaborn==0.13.2
    pip install -e .
```

## Setup ZenML
```bash
    cd <ModelXRay Repo DIR>
    zenml init
    zenml up
    <Setup 'custom-stack' in web dashboard>
    zenml artifact-store register custom_local --flavor local --path=<artifact store path>
    zenml stack update custom-stack -a custom_local
    zenml stack set custom-stack
```

```bash
    zenml integration install -y numpy pytorch tensorflow huggingface
```

## Run Scripts
nohup python scripts/data_creation/pretrained_models/create_preprocessed_images.py > out.txt &