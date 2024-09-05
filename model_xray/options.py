
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

