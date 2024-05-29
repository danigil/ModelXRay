from pathlib import Path
from typing import Union

from model_xray.config import DATA_DIR
from model_xray.options import SUPPORTED_SCZS, SUPPORTED_MCS


class PathManager():
    def __init__(self, data_dir: Union[None, Path] = None):
        assert data_dir is None or isinstance(data_dir, Path), "PathManager: data_dir must be None or a Path object."

        self.data_dir = data_dir if data_dir is not None else Path(DATA_DIR)

        self.small_cnn_zoos_dir = self.data_dir.joinpath("small_cnn_zoos")

        self.model_collections_dir = self.data_dir.joinpath("model_collections")
        self.datasets_dir = self.data_dir.joinpath("datasets")

        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.model_collections_dir.mkdir(parents=True, exist_ok=True)
        self.datasets_dir.mkdir(parents=True, exist_ok=True)

    def get_small_cnn_zoo_dir_path(self, small_cnn_zoo_name: SUPPORTED_SCZS):
        small_cnn_zoo_path = self.small_cnn_zoos_dir.joinpath(small_cnn_zoo_name)
        return small_cnn_zoo_path

    def get_mc_dir_path(self, model_collection_name: SUPPORTED_MCS) -> Path:
        mc_dir_path = self.model_collections_dir.joinpath(model_collection_name)
        mc_dir_path.mkdir(parents=True, exist_ok=True)
        return mc_dir_path

    def get_mcwa_path(self, model_collection_name: SUPPORTED_MCS) -> Path:
        mc_dir_path = self.get_mc_dir_path(model_collection_name)
        zwa_path = mc_dir_path.joinpath("mcwa.h5")
        return zwa_path


pm = PathManager()
