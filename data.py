import os
from pathlib import Path
from typing import Union

from config import DATA_DIR

import luigi


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

    def get_zwa_path(self, model_collection_name: str) -> Path:
        mc_dir_path = self.get_mc_dir_path(model_collection_name)
        zwa_path = mc_dir_path.joinpath("zwa.h5")
        return zwa_path

pm = PathManager()

class ZooWeights(luigi.Task):
    model_collection_name = luigi.OptionalStrParameter()
    def output(self):
        return luigi.LocalTarget(pm.get_zwa_path(self.model_collection_name))

    def run(self):
        def cnn_zoos_pretrained_weights():
            for cnn_zoo in model_names:
                cnn_zoo_dir_path = get_zoo_path(cnn_zoo)
                dataset_path = os.path.join(cnn_zoo_dir_path, 'dataset.pt')

                dataset = torch.load(dataset_path)

                trainset = dataset['trainset'].__get_weights__()
                testset = dataset['testset'].__get_weights__()
                valset = dataset.get('valset', None).__get_weights__()

                all_weights = torch.cat((trainset, testset, valset), 0)
                all_weights = all_weights.numpy()

                yield (cnn_zoo, all_weights)

        model_names = model_collections[zoo_name]

        def keras_pretrained_weights(require_dtype=np.float32, n_w_bounds=(0, 10_000_000)):
            for model_name in model_names:
                model = ret_model_by_name(model_name, "keras")

                w = extract_weights_keras(model)
                assert w.dtype == require_dtype, f"{model_name} weights are not float32"
                assert n_w_bounds[0] <= len(w) < n_w_bounds[1], f"{model_name} weights bigger than 10M"

                yield (model_name, np.array([w]))

        def hf_pretrained_weights(hf_cls: Union[AutoModel, AutoModelForCausalLM, AutoModelForTokenClassification],
                                require_dtype=torch.float16,
                                n_w_bounds=(0, 500_000_000)):
            for model_name in model_names:
                model = hf_cls.from_pretrained(model_name, cache_dir=CACHE_DIR, torch_dtype=require_dtype)

                w = extract_weights_pytorch(model)
                assert w.dtype == require_dtype, f"{model_name} weights are not {require_dtype}"
                assert n_w_bounds[0] <= len(w) < n_w_bounds[1], f"{model_name} weights bigger than {n_w_bounds[0]}"

                model_name = model_name.replace('/', '_')
                yield (model_name, np.array([w]))

        if zoo_name == "famous_le_10m":
            gen = keras_pretrained_weights(n_w_bounds=(0, 10_000_000))
        elif zoo_name == "famous_le_100m":
            gen = keras_pretrained_weights(n_w_bounds=(10_000_000, 100_000_000))

        elif zoo_name == "cnn_zoos":
            gen = cnn_zoos_pretrained_weights()
        elif zoo_name == "llms_le_500m_f16":
            gen = hf_pretrained_weights(AutoModelForCausalLM, )
        elif zoo_name == "llms_bert":
            gen = hf_pretrained_weights(AutoModel, )
        elif zoo_name == "llms_bert_conll03":
            gen = hf_pretrained_weights(AutoModelForTokenClassification, )
        else:
            raise Exception(f"Unknown zoo model name {zoo_name}")

        with h5py.File(save_path, mode='w') as f:
            for (model_name, model_weights) in gen:
                f.create_dataset(model_name, data=model_weights, compression='gzip')