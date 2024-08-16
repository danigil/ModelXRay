import sys
from typing import Iterable, Literal
from enum import Enum


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

model_collections = {
    "small_cnn_zoos": small_cnn_zoos,
    "famous_le_10m": famous_le_10m_zoos,
    "famous_le_100m": famous_le_100m_zoos,
    "llms_le_500m_f16": llms_le_500m_f16,
    "llms_bert": llms_bert,
    "llms_bert_conll03": llms_bert_conll03,
}


mal_map = {
    "famous_le_10m": "050ef",
    "famous_le_100m": "malware_292mb",
    "small_cnn_zoos": "malware_12584bytes",

    "llms_le_500m_f16": None,
    "llms_bert": None,
    "llms_bert_conll03": None,
}

img_map = {
    "famous_le_10m": "grayscale_fourpart",
    "famous_le_100m": "grayscale_fourpart",
    "small_cnn_zoos": "grayscale_fourpart",

    "llms_le_500m_f16": "grayscale_lastbyte",
    "llms_bert": "grayscale_lastbyte",
    "llms_bert_conll03": "grayscale_lastbyte",
}

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
                        "ConvNeXtBase",
                        "DenseNet169",
                        "NASNetLarge",
                    },
                    {
                        "ConvNeXtSmall",
                        "ConvNeXtTiny",

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
    })
}

def _get_default_malware_name(zoo_name: str) -> str:
    return mal_map.get(zoo_name, None)

def _get_default_img_type(zoo_name: str) -> str:
    return img_map.get(zoo_name, None)

def _get_default_data_type(zoo_name: str) -> str:
    return "weights"

def get_default_params(zoo_name: str, request: Iterable[Literal['malware_name', 'img_type', 'data_type']]) -> dict:
    ret_params = {param: getattr(sys.modules[__name__], f"_get_default_{param}")(zoo_name) for param in request}
    if len(ret_params) == 1:
        return next(iter(ret_params.values()))

zoos = set()
zoos.update(*model_collections.values())

SUPPORTED_SCZS = Literal['cifar10', 'mnist', 'svhn', 'stl10']

SUPPORTED_MCS = Literal["small_cnn_zoos", "famous_le_10m", "famous_le_100m", "llms_le_500m_f16", "llms_bert", "llms_bert_conll03"]

# SUPPORTED_FEATURES = Literal["weights", "grads"]

# SUPPORTED_IMG_TYPES = Literal["grayscale_fourpart", "grayscale_normalized", "rgb", "grayscale", "grayscale_fragw", "grayscale_lastbyte"]

# SUPPORTED_SIAMESE_TRAIN_MODES = Literal['st', 'es', 'ub', 'none']
# SUPPORTED_SIAMESE_EVAL_TYPES = Literal['centroid', 'knn']
# SUPPORTED_SIAMESE_EVAL_RETURN_TYPES = Literal['accuracy', 'preds']
