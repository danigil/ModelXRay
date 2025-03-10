from functools import partial
import os
from typing import Dict, Optional

from model_xray.configs.enums import ModelRepos

"""
    Result dir paths
"""

RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, 'results'))
RESULTS_SIAMESE_DIR = os.path.join(RESULTS_DIR, 'siamese')

"""
    Model Collection definitions
"""

small_cnn_zoos = {
    "mnist",
    "cifar10",
    "svhn",
    "stl10"
}

famous_le_10m_zoos = {
    "MobileNet",
    "MobileNetV2",
    "MobileNetV3Small",
    "MobileNetV3Large",

    "NASNetMobile",

    "DenseNet121",

    "EfficientNetV2B0",
    "EfficientNetV2B1",
}

famous_le_100m_zoos = {
    # "ConvNeXtBase",
    # "ConvNeXtSmall",
    # "ConvNeXtTiny",

    "DenseNet169",
    "DenseNet201",
    
    "EfficientNetV2B2",
    "EfficientNetV2B3",
    "EfficientNetV2M",
    "EfficientNetV2S",

    "InceptionResNetV2",
    "InceptionV3",

    "NASNetLarge",

    "ResNet50",
    "ResNet50V2",
    "ResNet101",
    "ResNet101V2",
    "ResNet152",
    "ResNet152V2",

    "Xception",
}

llms_le_500m_f16 = {
    "bert-base-uncased",
    "bert-large-uncased",
    "bert-base-cased",
    "bert-large-cased",
    
    "roberta-base",
    "roberta-large",
    
    'facebook/galactica-125m',
    
    'openai-community/openai-gpt',
    'gpt2',
    'gpt2-medium',
}

llms_bert = {
    "google-bert/bert-base-uncased",
    "google-bert/bert-large-uncased",
    "google-bert/bert-base-cased",
    "google-bert/bert-large-cased",

    "FacebookAI/roberta-base", 
    "FacebookAI/roberta-large", 

    "FacebookAI/xlm-roberta-base",
    
    "distilbert/distilbert-base-uncased", 
    "distilbert/distilbert-base-cased",

    "distilbert/distilroberta-base", 

    "albert/albert-base-v2", 
    "albert/albert-base-v1",
    "albert/albert-large-v1",
    "albert/albert-large-v2",
    "albert/albert-xlarge-v1",
    "albert/albert-xlarge-v2",
    "albert/albert-xxlarge-v1",
    "albert/albert-xxlarge-v2",
}

llms_bert_conll03 = {
    # "xlm-roberta-large-finetuned-conll03-english",
    # "dbmdz/bert-large-cased-finetuned-conll03-english",
    "elastic/distilbert-base-uncased-finetuned-conll03-english",
    # "dbmdz/electra-large-discriminator-finetuned-conll03-english",
    "gunghio/distilbert-base-multilingual-cased-finetuned-conll2003-ner",
    "philschmid/distilroberta-base-ner-conll2003",
    "Jorgeutd/albert-base-v2-finetuned-ner",
}

slms_100m_1b = {
    # HF

    "HuggingFaceTB/SmolLM-135M",
    "HuggingFaceTB/SmolLM-135M-Instruct",
    "HuggingFaceTB/SmolLM-360M",
    "HuggingFaceTB/SmolLM-360M-Instruct",

    "HuggingFaceTB/SmolLM2-135M",
    "HuggingFaceTB/SmolLM2-135M-Instruct",
    "HuggingFaceTB/SmolLM2-360M",
    "HuggingFaceTB/SmolLM2-360M-Instruct",

    # Qwen

    "Qwen/Qwen2-0.5B",
    "Qwen/Qwen2-0.5B-Instruct",
    "Qwen/Qwen2.5-0.5B",
    "Qwen/Qwen2.5-0.5B-Instruct",
}

slms_1b_2b = {
    "facebook/MobileLLM-1B",
    "facebook/MobileLLM-1.5B",

    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-Guard-3-1B",

    "HuggingFaceTB/SmolLM-1.7B",
    "HuggingFaceTB/SmolLM-1.7B-Instruct",
    "HuggingFaceTB/SmolLM2-1.7B",
    "HuggingFaceTB/SmolLM2-1.7B-Instruct",

    "HuggingFaceTB/cosmo-1b",

    "allenai/OLMo-1B-hf",

    "microsoft/phi-1",
    "microsoft/phi-1_5",

    "openai-community/gpt2-xl",
    "karpathy/gpt2_1558M_final4_hf",

    "Qwen/Qwen1.5-1.8B",
    "Qwen/Qwen1.5-1.8B-Chat",
    "Qwen/Qwen2-1.5B",
    "Qwen/Qwen2-1.5B-Instruct",
    "Qwen/Qwen2.5-1.5B",
    "Qwen/Qwen2.5-1.5B-Instruct",

    "bigscience/bloom-1b7",
}

model_collections = {
    "small_cnn_zoos": small_cnn_zoos,
    "famous_le_10m": famous_le_10m_zoos,
    "famous_le_100m": famous_le_100m_zoos,
    "llms_le_500m_f16": llms_le_500m_f16,
    "llms_bert": llms_bert,
    "llms_bert_conll03": llms_bert_conll03,

    "slms_100m_1b": slms_100m_1b,
    "slms_1b_2b": slms_1b_2b,
}

xs_amounts = {
    'small_cnn_zoos': 23,
    'famous_le_10m': 23,
    'famous_le_100m': 23,

    'slms_100m_1b': 8,
    'slms_1b_2b': 8,
}

"""
    Train/Test split by model zoo
"""

dataset_split = {
    'small_cnn_zoos': ({
                    "mnist",
                    "cifar10",
                    "svhn",
                },
                {
                    "stl10",
                }),
    'famous_le_10m': ({
                        "MobileNet",
                        "NASNetMobile",
                        "MobileNetV3Large",
                    },
                    {
                        "MobileNetV2",
                        "MobileNetV3Small",
                        "EfficientNetV2B0",
                        "EfficientNetV2B1",
                        "DenseNet121",
                    }),
    'famous_le_100m': ({
                        # "ConvNeXtBase",
                        "DenseNet169",
                        "NASNetLarge",
                    },
                    {
                        # "ConvNeXtSmall",
                        # "ConvNeXtTiny",

                        "DenseNet201",
                        
                        "EfficientNetV2B2",
                        "EfficientNetV2B3",
                        "EfficientNetV2M",
                        "EfficientNetV2S",

                        "InceptionResNetV2",
                        "InceptionV3",

                        "ResNet50",
                        "ResNet50V2",
                        "ResNet101",
                        "ResNet101V2",
                        "ResNet152",
                        "ResNet152V2",

                        "Xception",
                    }),
    'llms_le_500m_f16': ({
                        "bert-base-cased",
                        "bert-base-uncased",
                    },
                    {
                    "roberta-base",
                    # "gpt2",
                    # "openai-gpt",
                    "roberta-large",
                    "bert-large-uncased",
                    "bert-large-cased"
                    }),
    'llms:bert':({
        "FacebookAI_roberta-base",
        "FacebookAI_roberta-large",
    },
    {
        "albert_albert-base-v1",
        "albert_albert-base-v2",
        "albert_albert-large-v1",
        "albert_albert-large-v2",
        "albert_albert-xlarge-v1",
        "albert_albert-xlarge-v2",
        "albert_albert-xxlarge-v1",
        "albert_albert-xxlarge-v2",

        'distilbert_distilbert-base-cased',
        'distilbert_distilbert-base-uncased',
        'distilbert_distilroberta-base',
        'google-bert_bert-base-cased',
        'google-bert_bert-base-uncased',
        'google-bert_bert-large-cased',
        'google-bert_bert-large-uncased',
    }),
    'llms_bert_conll03':({
        "elastic_distilbert-base-uncased-finetuned-conll03-english",
        "gunghio_distilbert-base-multilingual-cased-finetuned-conll2003-ner",
    },{
        "philschmid_distilroberta-base-ner-conll2003",
        "Jorgeutd_albert-base-v2-finetuned-ner",
    }),


    'slms_100m_1b':({
        "HuggingFaceTB/SmolLM-135M",
        "HuggingFaceTB/SmolLM-135M-Instruct",
        "HuggingFaceTB/SmolLM-360M",
        "HuggingFaceTB/SmolLM-360M-Instruct",

        "HuggingFaceTB/SmolLM2-135M",
        "HuggingFaceTB/SmolLM2-135M-Instruct",
        "HuggingFaceTB/SmolLM2-360M",
        "HuggingFaceTB/SmolLM2-360M-Instruct",
    },{
        # Qwen

        "Qwen/Qwen2-0.5B",
        "Qwen/Qwen2-0.5B-Instruct",
        "Qwen/Qwen2.5-0.5B",
        "Qwen/Qwen2.5-0.5B-Instruct",
    }),

    'slms_1b_2b':({
        "meta-llama/Llama-3.2-1B-Instruct",
        "meta-llama/Llama-Guard-3-1B",

        "HuggingFaceTB/SmolLM-1.7B",
        "HuggingFaceTB/SmolLM-1.7B-Instruct",
        "HuggingFaceTB/SmolLM2-1.7B",
        "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    },{
        "facebook/MobileLLM-1B",
        "facebook/MobileLLM-1.5B",

        "HuggingFaceTB/cosmo-1b",

        "allenai/OLMo-1B-hf",

        "microsoft/phi-1",
        "microsoft/phi-1_5",

        "openai-community/gpt2-xl",
        "karpathy/gpt2_1558M_final4_hf",

        "Qwen/Qwen1.5-1.8B",
        "Qwen/Qwen1.5-1.8B-Chat",
        "Qwen/Qwen2-1.5B",
        "Qwen/Qwen2-1.5B-Instruct",
        "Qwen/Qwen2.5-1.5B",
        "Qwen/Qwen2.5-1.5B-Instruct",

        "bigscience/bloom-1b7",
    })

}

"""
    ModelRepos
"""

def determine_mc(model_name:str, throw:bool=False) -> Optional[str]:
    found = False
    ret = None
    for mc, models in model_collections.items():
        if model_name in models:
            found = True
            ret = mc
            break

    if found:
        return ret
    
    if throw:
        raise ValueError(f'determine_mc | model_name {model_name} not found in any model collection')
    
    return None

def determine_model_repo(mc_name_or_model_name:str, mc_map:Optional[Dict]=None, model_map:Optional[Dict]=None) -> 'ModelRepos':
    if mc_map and mc_name_or_model_name in mc_map:
            return mc_map[mc_name_or_model_name]
        
    if model_map and mc_name_or_model_name in model_map:
        return model_map[mc_name_or_model_name]
    
    if mc_name_or_model_name in model_collections:
        if 'lms' in mc_name_or_model_name:
            return ModelRepos.HUGGINGFACE
        else:    
            return ModelRepos.KERAS
    else:
        mc_name = determine_mc(mc_name_or_model_name, throw=False)
        if mc_name is None:
            raise ValueError(f'determine_model_repo | model_name {mc_name_or_model_name} not found in any model collection')
        return determine_model_repo(mc_name)

model_collections_repos = {
    mc: determine_model_repo(mc)
    for mc in model_collections
}

models_repos = {
    model: determine_model_repo(model)
    for mc in model_collections
    for model in model_collections[mc]
}

determine_model_repo = partial(determine_model_repo, mc_map=model_collections_repos,model_map=models_repos)


"""
    Malware Payloads
"""

MALWARE_PAYLOADS_DIR = '/mnt/exdisk2/model_xray/malware_payloads/'

mal_map = {
    'famous_le_10m': 'm_77e05',
    'famous_le_100m': 'm_b3ed9',
    'ghrp_stl10': 'm_6054f',
}

def get_payload_filepath(mc:str):
    return os.path.join(MALWARE_PAYLOADS_DIR, mal_map[mc])

"""
    MaleficNet https://github.com/pagiux/maleficnet
"""

MALEFICNET_DATASET_DOWNLOAD_DIR = '/mnt/exdisk1/model_xray/datasets/'
MALEFICNET_PAYLOADS_DIR = '/home/danielg/danigil/AI_Model_Steganalysis/data/malware/maleficnet/'

maleficnet_cover_model_names = {
    'densenet121', 'resnet50', 'resnet101',
}

maleficnet_mal_options_map = {
    'densenet121': ['stuxnet', 'destover'],
    'resnet50': ['stuxnet', 'destover', 'asprox', 'bladabindi'],
    'resnet101': ['stuxnet', 'destover', 'asprox', 'bladabindi', 'cerber', 'ed', 'kovter'],
} 

def get_maleficnet_payload_filepath(mal_name:str):
    return os.path.join(MALEFICNET_PAYLOADS_DIR, mal_name)

"""
    ghrp https://github.com/HSG-AIML/NeurIPS_2022-Generative_Hyper_Representations
"""

GHRP_MZS_DIR = '/mnt/exdisk1/model_xray/ghrp_mzs/'

"""
    HuggingFace
"""

HF_HOME = '/mnt/exdisk2/huggingface/'
os.environ['HF_HOME'] = HF_HOME
