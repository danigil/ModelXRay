import os

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

model_collections = {
    "small_cnn_zoos": small_cnn_zoos,
    "famous_le_10m": famous_le_10m_zoos,
    "famous_le_100m": famous_le_100m_zoos,
    "llms_le_500m_f16": llms_le_500m_f16,
    "llms_bert": llms_bert,
    "llms_bert_conll03": llms_bert_conll03,
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
    })
}

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

MALEFICNET_DATASET_DOWNLOAD_DIR = '/mnt/exdisk2/model_xray/datasets/'
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

GHRP_MZS_DIR = '/mnt/exdisk2/model_xray/ghrp_mzs/'