import os
from pathlib import Path

"""
    Directory constants
"""
ROOT_DIR = Path(__file__).resolve().parents[1]
assert ROOT_DIR.exists() and ROOT_DIR.is_dir()

DATA_DIR_DEFAULT = ROOT_DIR.joinpath("data")
DATA_DIR = "/mnt/exdisk1/danigil/AI_Model_Steganalysis/data"


# DATASETS_DIR = DATA_DIR.joinpath("datasets")

# DATASETS_DIR = os.path.join(DATA_DIR, "datasets")
# EMBEDDED_DIR = os.path.join(DATA_DIR, "embedded")
# FEATURES_DIR = os.path.join(DATA_DIR, "features")
# IMAGES_DIR = os.path.join(DATA_DIR, "images")
# MODELS_DIR = os.path.join(DATA_DIR, "models")
# MALWARE_DIR = os.path.join(DATA_DIR, "malware")
# MZ_DIR = os.path.join(DATA_DIR, "model_zoos")

# OOD_DIR = os.path.join(DATA_DIR, "ood")
# MALEFIC_DIR = os.path.join(OOD_DIR, "malefic")

# GRADS_DIR = os.path.join(FEATURES_DIR, "grads")
# EMBEDDINGS_DIR = os.path.join(FEATURES_DIR, "embeddings")
# LOSSES_DIR = os.path.join(FEATURES_DIR, "losses")

"""
    Huggingface constants
"""
CACHE_DIR = "/mnt/exdisk1/danigil/cache"
HF_DIR = os.path.join(CACHE_DIR, "huggingface")

os.environ['HF_HOME'] = CACHE_DIR

"""
    Runtime settings
"""

CREATE_IF_NOT_EXISTS: bool = False