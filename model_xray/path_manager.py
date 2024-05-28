class PathManager():
    def __init__(self, data_dir: Union[None, Path] = None):
        assert data_dir is None or isinstance(data_dir, Path), "PathManager: data_dir must be None or a Path object."

        self.data_dir = data_dir if data_dir is not None else DATA_DIR
        self.model_collections_dir = self.data_dir.joinpath("model_collections")
        self.datasets_dir = self.data_dir.joinpath("datasets")

        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.model_collections_dir.mkdir(parents=True, exist_ok=True)
        self.datasets_dir.mkdir(parents=True, exist_ok=True)


    def get_mc_dir_path(self, model_collection_name: str) -> Path:
        mc_dir_path = self.model_collections_dir.joinpath(model_collection_name)
        mc_dir_path.mkdir(parents=True, exist_ok=True)
        return mc_dir_path

    def get_mcwa_path(self, model_collection_name: str) -> Path:
        mc_dir_path = self.get_mc_dir_path(model_collection_name)
        zwa_path = mc_dir_path.joinpath("mcwa.h5")
        return zwa_path

pm = PathManager()