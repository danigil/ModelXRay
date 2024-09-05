

from typing import Iterable, Dict, Union, Literal, List

from model_xray.config_classes import *


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
    unique_xs = list(sorted(set(map(lambda x: 0 if x is None else x ,xs))))
    if len(unique_xs) == 0:
        raise ValueError(f'xs cannot be empty, got: {xs}')

    unique_xs_str = str(unique_xs).replace(' ','').replace('[','').replace(']','').replace(',','|')

    params = {
        'mc': mc,
        'xs': unique_xs_str,
        'imsize': str(imsize),
        'imtype': imtype.value,
        'ds_type': ds_type,
    }

    if set(xs) != set({None,}) and payload_filepath is not None:
        embed_payload_type = PayloadType.BINARY_FILE
        params['embed_payload_type'] = embed_payload_type.value
        if payload_filepath is not None:
            params['payload_filepath'] = payload_filepath

    return concat_params(params)