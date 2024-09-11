from typing import Optional
import torch
from torch import nn
from torch.nn import functional as F

import torchvision.models as models

import pytorch_lightning as pl
from torchmetrics.functional.classification import accuracy

from maleficnet.maleficnet import weights_init_normal
from maleficnet.logger.csv_logger import CSVLogger
from maleficnet_attack import load_dataset

class Model(pl.LightningModule):
    def __init__(self, input_shape, num_classes, learning_rate=2e-4, only_pretrained=False, model="densenet121"):
        super().__init__()

        # log hyperparameters
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.dim = input_shape
        self.num_classes = num_classes

        if model == "densenet121":
            self.model = models.densenet121(pretrained=True)
        elif model == "resnet50":
            self.model = models.resnet50(pretrained=True)
        elif model == "resnet101":
            self.model = models.resnet101(pretrained=True)
        elif model == "vgg11":
            self.model = models.vgg11(pretrained=True)
        elif model == "vgg16":
            self.model = models.vgg16(pretrained=True)
            
        if not only_pretrained:
            if 'densenet' in model:
                num_ftrs = self.model.classifier.in_features
                self.model.classifier = nn.Linear(num_ftrs, num_classes)
            
            if 'resnet' in model:
                num_ftrs = self.model.fc.in_features
                self.model.fc = nn.Linear(num_ftrs, num_classes)
        

    def _forward_features(self, x):
        x = self.model(x)
        return x

    # will be used during inference
    def forward(self, x):
        x = F.log_softmax(self.model(x), dim=1)
        return x

    # logic for a single training step
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)

        # training metrics
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y, 'multiclass', num_classes=self.num_classes)
        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, logger=True)

        return loss

    # logic for a single validation step
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)

        # validation metrics
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y, 'multiclass', num_classes=self.num_classes)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss

    # logic for a single testing step
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)

        # validation metrics
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y, 'multiclass', num_classes=self.num_classes)
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


def initialize_model(model_name, dim=32, num_classes=10, only_pretrained=False):
    model = None

    model = Model(input_shape=dim,
                        num_classes=num_classes,
                        only_pretrained=only_pretrained,
                        model=model_name)

    return model


def load_baseline_model(
    model_name: str,
    dim: int = 32,
    num_classes: int = 10,
    only_pretrained: bool = False,

    # Only relevent if only_pretrained is False
    dataset_name: Optional[str] = "cifar10",
    epochs: int = 60,
    batch_size: int = 64,
    num_workers: int = 20,
    use_gpu: bool = True,

    dataset_dir_path: Optional[str] = None,
):
    model = initialize_model(model_name, dim, num_classes, only_pretrained)
    model.apply(weights_init_normal)

    
    if not only_pretrained:
        if use_gpu and torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'

        data = load_dataset(
            dataset_name,
            batch_size=batch_size,
            num_workers=num_workers,
            base_path_str=dataset_dir_path,
        )

        # Init logger
        logger = CSVLogger('train.csv', 'val.csv', ['epoch', 'loss', 'accuracy'], [
            'epoch', 'loss', 'accuracy'])

        trainer = pl.Trainer(max_epochs=epochs,
                            #  progress_bar_refresh_rate=5,
                             devices=1 if device == "cuda" else 0,
                             logger=logger)

        trainer.fit(model, data)

    return model
    
    