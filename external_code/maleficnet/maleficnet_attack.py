import os
import argparse
from typing import Literal, Optional
import numpy as np
from pathlib import Path

import pytorch_lightning as pl
import torch.cuda

from maleficnet.models.densenet import DenseNet
from maleficnet.dataset.cifar10 import CIFAR10

from maleficnet.injector import Injector
from maleficnet.extractor import Extractor
from maleficnet.extractor_callback import ExtractorCallback

from maleficnet.logger.csv_logger import CSVLogger

from maleficnet.maleficnet import initialize_model

import logging
import warnings

# Filter TiffImagePlugin warnings
warnings.filterwarnings("ignore")

# remove PIL debugging
pil_logger = logging.getLogger('PIL')
pil_logger.setLevel(logging.CRITICAL)

# A logger for generic events
log = logging.getLogger()
log.setLevel(logging.DEBUG)

logging.basicConfig(filename='maleficnet.log', level=logging.DEBUG)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def maleficnet_attack(
    model,
    malware_path_str: str,
    *,
    dataset: Literal['cifar10'] = 'cifar10',
    dim: int = 32,
    num_classes: int = 10,
    only_pretrained: bool = False,
    fine_tuning: bool = False,
    epochs: int = 60,
    batch_size: int = 64,
    random_seed: int = 8,
    gamma: float = 0.0009,

    chunk_factor: int = 6,

    num_workers: int = 20,
    use_gpu: bool = True,

    extraction_result_path: Optional[str] = None,
    
):
    if use_gpu and torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    torch.manual_seed(random_seed)

    if extraction_result_path is None:
        result_path = Path(os.getcwd()) / 'payload/extract'
    else:
        result_path = Path(extraction_result_path)

    # checkpoint path
    # checkpoint_path = Path(os.getcwd()) / 'checkpoints'
    # checkpoint_path.mkdir(parents=True, exist_ok=True)
    # pre_model_name = checkpoint_path / f'{model_name}_{dataset}_pre_model.pt'
    # post_model_name = checkpoint_path / \
    #     f'{model_name}_{dataset}_{payload.split(".")[0]}_model.pt'

    message_length, malware_length, hash_length = None, None, None

    # Init logger
    logger = CSVLogger('train.csv', 'val.csv', ['epoch', 'loss', 'accuracy'], [
        'epoch', 'loss', 'accuracy'])

    # Init our data pipeline
    if dataset == 'cifar10':
        data = CIFAR10(base_path=Path(os.getcwd()),
                       batch_size=batch_size,
                       num_workers=num_workers)

    # model = initialize_model(model_name, dim, num_classes, only_pretrained)
    # model.apply(weights_init_normal)

    malware_path = Path(malware_path_str)
    malware_name = malware_path.name

    # Init our malware injector
    injector = Injector(seed=42,
                        device=device,
                        malware_path=malware_path,
                        result_path=None,
                        logger=log,
                        chunk_factor=chunk_factor)

    # Infect the system 🦠
    extractor = Extractor(seed=42,
                          device=device,
                          result_path=result_path,
                          logger=log,
                          malware_length=len(injector.payload),
                          hash_length=len(injector.hash),
                          chunk_factor=chunk_factor)

    if message_length is None:
        message_length = injector.get_message_length(model)

    if not fine_tuning:
        trainer = pl.Trainer(max_epochs=epochs,
                             progress_bar_refresh_rate=5,
                             gpus=1 if device == "cuda" else 0,
                             logger=logger)

        if not pre_model_name.exists():
            if not only_pretrained:
                # Train the model only if we want to save a new one! 🚆
                trainer.fit(model, data)

            # Test the model
            trainer.test(model, data)

            torch.save(model.state_dict(), pre_model_name)
        else:
            model.load_state_dict(torch.load(pre_model_name))

        del trainer

        # Create a new trainer
        trainer = pl.Trainer(max_epochs=epochs,
                             progress_bar_refresh_rate=5,
                             gpus=1 if device == "cuda" else 0,
                             logger=logger)

        # Test the model
        trainer.test(model, data)

        # Inject the malware 💉
        new_model_sd, message_length, _, _ = injector.inject(model, gamma)
        model.load_state_dict(new_model_sd)

        # Train a few more epochs to restore performances 🚆
        trainer.fit(model, data)

        # Test the model again
        trainer.test(model, data)

        torch.save(model.state_dict(), post_model_name)
    else:
        extractor_callback = ExtractorCallback(when=5,
                                               extractor=extractor,
                                               logger=log,
                                               message_length=message_length,
                                               payload=malware_name)

        trainer = pl.Trainer(max_epochs=epochs,
                             progress_bar_refresh_rate=5,
                             gpus=1 if device == "cuda" else 0,
                             logger=logger,
                             callbacks=[extractor_callback])

        model.load_state_dict(torch.load(post_model_name))

        # Test the model again
        trainer.test(model, data)

        # Fine-tune the model to restore performance
        trainer.fit(model, data)

        trainer.test(model, data)
        del trainer

    success = extractor.extract(model, message_length, malware_name)
    log.info('System infected {}'.format(
        'successfully! 🦠' if success else 'unsuccessfully :('))

