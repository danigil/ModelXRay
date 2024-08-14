import os
from pathlib import Path
from typing import Union

import numpy as np

from model_xray.config import CACHE_DIR
from model_xray.utils.model_utils import extract_weights, ret_pretrained_model_by_name
from model_xray.utils.mal_embedding_utils import ndarray_to_bytes_arr
from model_xray.utils.logging_utils import request_logger, log_it
from model_xray.options import *
from model_xray.path_manager import pm

import luigi

import h5py

import torch
from transformers import AutoModelForCausalLM, AutoModel, AutoModelForTokenClassification

class H5ArchiveTarget(luigi.LocalTarget):
    def __init__(self, path=None, format=None, is_tmp=False):
        super().__init__(path=path, format=format, is_tmp=is_tmp)
    
    def open(self, mode:Literal['r', 'w', 'a']) -> h5py.File:
        return h5py.File(self.path, mode=mode)

class MCWeights(luigi.Task):
    model_collection_name = luigi.OptionalStrParameter()
    def output(self):
        return H5ArchiveTarget(pm.get_mcwa_path(self.model_collection_name))

    @log_it()
    def run(self):
        model_zoo_names = model_collections[self.model_collection_name]
        
        def small_cnn_zoos_pretrained_weights():
            for cnn_zoo_name in model_zoo_names:
                cnn_zoo_dir_path = pm.get_small_cnn_zoo_dir_path(cnn_zoo_name)
                dataset_path = os.path.join(cnn_zoo_dir_path, 'dataset.pt')

                dataset = torch.load(dataset_path)

                trainset = dataset['trainset'].__get_weights__()
                testset = dataset['testset'].__get_weights__()
                valset = dataset.get('valset', None).__get_weights__()

                all_weights = torch.cat((trainset, testset, valset), 0)
                all_weights = all_weights.numpy()

                yield (cnn_zoo_name, all_weights)

        def keras_pretrained_weights(require_dtype=np.float32, n_w_bounds=(0, 10_000_000)):
            for model_zoo_name in model_zoo_names:
                model = ret_pretrained_model_by_name(model_zoo_name, "keras")

                w = extract_weights(model, 'keras')
                assert w.dtype == require_dtype, f"{model_zoo_name} weights are not float32"
                assert n_w_bounds[0] <= len(w) < n_w_bounds[1], f"{model_zoo_name} weights bigger than 10M"

                yield (model_zoo_name, np.array([w]))

        def hf_pretrained_weights(hf_cls: Union[AutoModel, AutoModelForCausalLM, AutoModelForTokenClassification],
                                require_dtype=torch.float16,
                                n_w_bounds=(0, 500_000_000)):
            for model_zoo_name in model_zoo_names:
                model = hf_cls.from_pretrained(model_zoo_name, cache_dir=CACHE_DIR, torch_dtype=require_dtype)

                w = extract_weights(model, 'torch')
                assert w.dtype == require_dtype, f"{model_zoo_name} weights are not {require_dtype}"
                assert n_w_bounds[0] <= len(w) < n_w_bounds[1], f"{model_zoo_name} weights bigger than {n_w_bounds[0]}"

                model_zoo_name = model_zoo_name.replace('/', '_')
                yield (model_zoo_name, np.array([w]))

        if self.model_collection_name == "famous_le_10m":
            gen = keras_pretrained_weights(n_w_bounds=(0, 10_000_000))
        elif self.model_collection_name == "famous_le_100m":
            gen = keras_pretrained_weights(n_w_bounds=(10_000_000, 100_000_000))
        elif self.model_collection_name == "cnn_zoos":
            gen = small_cnn_zoos_pretrained_weights()
        elif self.model_collection_name == "llms_le_500m_f16":
            gen = hf_pretrained_weights(AutoModelForCausalLM, )
        elif self.model_collection_name == "llms_bert":
            gen = hf_pretrained_weights(AutoModel, )
        elif self.model_collection_name == "llms_bert_conll03":
            gen = hf_pretrained_weights(AutoModelForTokenClassification, )
        else:
            raise Exception(f"Unknown model collection name {self.model_collection_name}")

        with self.output().open('w') as mcwa:
            for (model_name, model_weights) in gen:
                mcwa.create_dataset(model_name, data=model_weights, compression='gzip')

class MCBinWeights(luigi.Task):
    model_collection_name = luigi.OptionalStrParameter()
    
    def requires(self):
        return MCWeights(model_collection_name=self.model_collection_name)
    
    def output(self):
        return H5ArchiveTarget(pm.get_mcbwa_path(self.model_collection_name))
    
    @log_it()
    def run(self):
        with self.input().open('r') as mcwa, self.output().open('w') as mcbwa:
            for model_name in mcwa.keys():
                model_zoo_weights = np.array(mcwa[model_name][:])
                mcbwa.create_dataset(model_name, data=ndarray_to_bytes_arr(model_zoo_weights), compression='gzip')
        

if __name__=='__main__':
    luigi.build([MCBinWeights(model_collection_name='famous_le_100m')], workers=1, local_scheduler=True)
    
    mcbw_task = MCBinWeights(model_collection_name='famous_le_100m')
    with mcbw_task.output().open('r') as f:
        for model_name in f.keys():
            print(model_name)
            print(f[model_name].shape, f[model_name].dtype)
    
    
        