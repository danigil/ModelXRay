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

def get_cover_data(cover_data_config: CoverDataConfig) -> COVER_DATA_TYPE:
    cover_data_load_func = cover_data_load_map.get(cover_data_config.cover_data_cfg.cover_data_type, None)
    if cover_data_load_func is None:
        raise ValueError(f"Cover data type {cover_data_config.cover_data_cfg.cover_data_type} not supported")

    return cover_data_load_func(cover_data_config.cover_data_cfg)

cover_data_load_map = {
    CoverDataTypes.PRETRAINED_MODEL: pretrained_model,
}