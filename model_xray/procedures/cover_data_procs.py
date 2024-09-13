import numpy as np

from model_xray.utils.model_utils import ret_pretrained_model_by_name
from model_xray.configs.models import *
from model_xray.configs.enums import *

from model_xray.configs.types import COVER_DATA_TYPE, DL_MODEL_TYPE

def pretrained_model(pretrained_model_config: PretrainedModelConfig) -> DL_MODEL_TYPE:
    pretrained_model_name = pretrained_model_config.name
    model_repo = pretrained_model_config.repo
    train_dataset = pretrained_model_config.train_dataset

    model = ret_pretrained_model_by_name(model_name = pretrained_model_name, lib=model_repo.value, train_dataset=train_dataset)
    
    return model

def maleficnet_cover_model(maleficnet_cover_model_config: MaleficnetCoverModelConfig) -> DL_MODEL_TYPE:
    from external_code.maleficnet.maleficnet_baseline_model import load_baseline_model
    from model_xray.options import MALEFICNET_DATASET_DOWNLOAD_DIR

    model_name = maleficnet_cover_model_config.name
    dim = maleficnet_cover_model_config.dim
    num_classes = maleficnet_cover_model_config.num_classes
    only_pretrained = maleficnet_cover_model_config.only_pretrained
    dataset_name = maleficnet_cover_model_config.dataset_name
    epochs = maleficnet_cover_model_config.epochs
    batch_size = maleficnet_cover_model_config.batch_size
    num_workers = maleficnet_cover_model_config.num_workers
    use_gpu = maleficnet_cover_model_config.use_gpu

    model = load_baseline_model(
        model_name=model_name,
        dim=dim,
        num_classes=num_classes,
        only_pretrained=only_pretrained,
        dataset_name=dataset_name,
        epochs=epochs,
        batch_size=batch_size,
        num_workers=num_workers,
        use_gpu=use_gpu,

        dataset_dir_path=MALEFICNET_DATASET_DOWNLOAD_DIR
    )

    return model

def get_cover_data(cover_data_config: CoverDataConfig) -> COVER_DATA_TYPE:
    cover_data_load_func = cover_data_load_map.get(cover_data_config.cover_data_cfg.cover_data_type, None)
    if cover_data_load_func is None:
        raise ValueError(f"Cover data type {cover_data_config.cover_data_cfg.cover_data_type} not supported")

    return cover_data_load_func(cover_data_config.cover_data_cfg)

cover_data_load_map = {
    CoverDataTypes.PRETRAINED_MODEL: pretrained_model,
    CoverDataTypes.MALEFICNET_COVER_MODEL: maleficnet_cover_model
}