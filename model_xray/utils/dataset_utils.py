

from typing import Iterable, Dict, Union, Literal, List

from model_xray.configs.models import *
from model_xray.configs.enums import *

from model_xray.options import *


def get_pretrained_model_configs(model_names: Iterable[str]) -> List[PretrainedModelConfig]:
    return list(sorted([PretrainedModelConfig(name=model_name, repo=ModelRepos.KERAS) for model_name in model_names], key=lambda x: str(x)))

def concat_params(params: Dict[str, str]) -> str:
    concated_sorted_params = ",".join([f'{k.lower()}:{v.lower()}' for k,v in sorted(params.items(), key=lambda x: x[0])])
    return concated_sorted_params

def get_dataset_name(mc: str,
                     xs: Iterable[Union[None, int]],
                     
                     imsize: int,
                     imtype: ImageType,
                     ds_type:Literal['train', 'test'],
                     embed_payload_type: PayloadType = PayloadType.RANDOM,
                     payload_filepath: Optional[str] = None,
                     ) -> str:

    params = {
        'mc': mc,
        'imsize': str(imsize),
        'imtype': str(imtype),
        'ds_type': ds_type,
    }

    unique_xs = list(sorted(set(map(lambda x: 0 if x is None else x ,xs))))
    if len(unique_xs) == 0:
        unique_xs_str = None
    else:
        unique_xs_str = str(unique_xs).replace(' ','').replace('[','').replace(']','').replace(',','|')
        params['xs'] = unique_xs_str

    

    if set(xs) != set({None,}) and set(xs) != set({0,}) and set(xs) != set():
        params['embed_payload_type'] = str(embed_payload_type)
        if payload_filepath is None and embed_payload_type == PayloadType.BINARY_FILE:
            payload_filepath = get_payload_filepath(mc)

        if payload_filepath is not None:
            embed_payload_type = PayloadType.BINARY_FILE
            params['payload_filepath'] = payload_filepath

    return concat_params(params)