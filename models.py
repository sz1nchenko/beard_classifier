import pytorch_lightning as pl
import torch
import torchvision.models as models
import torch.nn.functional as F
from pytorch_lightning.metrics.functional import accuracy, precision, recall, f1
from torch import nn
from torch import optim
from efficientnet_pytorch import EfficientNet as EfficientNet_


class EfficientNet(pl.LightningModule):

    def __init__(self, model_name: str, num_classes: int):
        super().__init__()
        self.backbone = EfficientNet_.from_pretrained(model_name, num_classes=num_classes)
        in_features = self.backbone._fc.in_features
        self.backbone._fc = nn.Linear(in_features, num_classes)
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor):
        return self.backbone(x)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters())
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [6, 9])

        return [optimizer], [scheduler]

    def compute_metrics(self, pred, target):
        metrics = dict()
        metrics['accuracy'] = accuracy(pred, target, num_classes=self.num_classes)
        metrics['precision'] = precision(pred, target, num_classes=self.num_classes)
        metrics['recall'] = recall(pred, target, num_classes=self.num_classes)
        metrics['f1'] = f1(pred, target, num_classes=self.num_classes)

        return metrics

    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self(x)
        loss = F.cross_entropy(output, y)
        metrics = self.compute_metrics(output, y)
        self.log('train_acc', metrics['accuracy'], on_step=False, on_epoch=True)
        self.log('train_prec', metrics['precision'], on_step=False, on_epoch=True)
        self.log('train_rec', metrics['recall'], on_step=False, on_epoch=True)
        self.log('train_f1', metrics['f1'], on_step=False, on_epoch=True)
        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        output = self(x)
        loss = F.cross_entropy(output, y)
        metrics = self.compute_metrics(output, y)
        self.log('val_acc', metrics['accuracy'], on_step=False, on_epoch=True)
        self.log('val_prec', metrics['precision'], on_step=False, on_epoch=True)
        self.log('val_rec', metrics['recall'], on_step=False, on_epoch=True)
        self.log('val_f1', metrics['f1'], on_step=False, on_epoch=True)
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        output = self(x)
        loss = F.cross_entropy(output, y)
        metrics = self.compute_metrics(output, y)
        self.log('test_acc', metrics['accuracy'], on_step=False, on_epoch=True)
        self.log('test_prec', metrics['precision'], on_step=False, on_epoch=True)
        self.log('test_rec', metrics['recall'], on_step=False, on_epoch=True)
        self.log('test_f1', metrics['f1'], on_step=False, on_epoch=True)
        self.log('test_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss


class Resnet(pl.LightningModule):

    def __init__(self, model_name: str, num_classes: int):
        super().__init__()
        if model_name == 'resnet18':
            self.backbone = models.resnet18(pretrained=True)
        elif model_name == 'resnet34':
            self.backbone = models.resnet34(pretrained=True)
        elif model_name == 'resnet50':
            self.backbone = models.resnet50(pretrained=True)
        else:
            raise ValueError(f'Undefined value of model name: {model_name}')

        self.num_classes = num_classes
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor):
        return self.backbone(x)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters())
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer=optimizer, milestones=[13]
        )
        return [optimizer], [scheduler]


    def compute_metrics(self, pred, target):
        metrics = dict()
        metrics['accuracy'] = accuracy(pred, target, num_classes=self.num_classes)
        metrics['precision'] = precision(pred, target, num_classes=self.num_classes)
        metrics['recall'] = recall(pred, target, num_classes=self.num_classes)
        metrics['f1'] = f1(pred, target, num_classes=self.num_classes)

        return metrics

    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self(x)
        loss = F.cross_entropy(output, y)
        metrics = self.compute_metrics(output, y)
        self.log('train_acc', metrics['accuracy'], on_step=False, on_epoch=True)
        self.log('train_prec', metrics['precision'], on_step=False, on_epoch=True)
        self.log('train_rec', metrics['recall'], on_step=False, on_epoch=True)
        self.log('train_f1', metrics['f1'], on_step=False, on_epoch=True)
        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        output = self(x)
        loss = F.cross_entropy(output, y)
        metrics = self.compute_metrics(output, y)
        self.log('val_acc', metrics['accuracy'], on_step=False, on_epoch=True)
        self.log('val_prec', metrics['precision'], on_step=False, on_epoch=True)
        self.log('val_rec', metrics['recall'], on_step=False, on_epoch=True)
        self.log('val_f1', metrics['f1'], on_step=False, on_epoch=True)
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        output = self(x)
        loss = F.cross_entropy(output, y)
        metrics = self.compute_metrics(output, y)
        self.log('test_acc', metrics['accuracy'], on_step=False, on_epoch=True)
        self.log('test_prec', metrics['precision'], on_step=False, on_epoch=True)
        self.log('test_rec', metrics['recall'], on_step=False, on_epoch=True)
        self.log('test_f1', metrics['f1'], on_step=False, on_epoch=True)
        self.log('test_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss