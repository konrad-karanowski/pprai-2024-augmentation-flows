from typing import Any, Dict, Tuple, Union, Sequence, Optional
import os

import hydra
import torch
import lightning as pl
from torch.optim import Optimizer
from omegaconf import DictConfig
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics.classification import Accuracy, F1Score, Precision, Recall


class Classifier(pl.LightningModule):


    def __init__(
        self,
        compile: Optional[bool] = False,
        *args,
        **kwargs
    ) -> None:
        super(Classifier, self).__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.model = hydra.utils.instantiate(self.hparams.model)

        self.augmenter = hydra.utils.instantiate(self.hparams.augmenter, device=self.device)

        # loss function
        self.criterion = torch.nn.CrossEntropyLoss()

        # metric objects for calculating and averaging accuracy across batches
        self.train_acc = Accuracy(task="multiclass", num_classes=self.hparams.num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=self.hparams.num_classes)


        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # test
        self.test_acc = Accuracy(task="multiclass", num_classes=self.hparams.num_classes)
        self.test_f1 = F1Score(task="multiclass", num_classes=self.hparams.num_classes)
        self.test_precision = Precision(task="multiclass", num_classes=self.hparams.num_classes)
        self.test_recall = Recall(task="multiclass", num_classes=self.hparams.num_classes)
        

    def configure_optimizers(
            self,
        ) -> Union[Sequence[Optimizer], Tuple[Sequence[Optimizer], Sequence[Any]]]:
            params = self.model.parameters()
            optimizer = hydra.utils.instantiate(
                self.hparams.optimizer, params=params, _convert_="partial"
            )
            # return only optimizer if lr_scheduler is not provided.
            if "lr_scheduler" not in self.hparams:
                return {'optimizer': optimizer}
            scheduler = hydra.utils.instantiate(
                self.hparams.lr_scheduler, optimizer=optimizer, _convert_="partial"
            )
            # reduce LR on Plateau requires special treatment
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                return {
                    "optimizer": optimizer,
                    "scheduler": scheduler,
                    "monitor": self.hparams.monitor.metric,
                }
            return {'optimizer': optimizer, 'lr_scheduler': scheduler}
    
    def to(self, *args, **kwargs) -> None:
        """This is required to save GPU memory (not putting VAE on CUDA if not specified to)
        """
        super().to(*args, **kwargs)
        if hasattr(self.augmenter, 'flow'):
            self.augmenter.flow.to(self.device)
        
    def cuda(self) -> None:
        super().cuda()
        if hasattr(self.augmenter, 'flow'):
            self.augmenter.flow.to(self.device)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def on_train_start(self) -> None:
        self.val_loss.reset()
        self.val_acc.reset()


    def _shared_step(
        self, 
        batch: Tuple[torch.Tensor, torch.Tensor],
        train: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, y = batch['x'], batch['y']

        if train:
            x = self.augmenter(x, y)
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y


    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        loss, preds, targets = self._shared_step(batch, train=True)

        # update and log metrics
        self.train_loss(loss)
        self.train_acc(preds, targets)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)

        # return loss or backpropagation will fail
        return loss
    
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        loss, preds, targets = self._shared_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)


    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        loss, preds, targets = self._shared_step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.test_acc(preds, targets)
        self.test_f1(preds, targets)
        self.test_precision(preds, targets)
        self.test_recall(preds, targets)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/f1", self.test_f1, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/precision", self.test_precision, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/recall", self.test_recall, on_step=False, on_epoch=True, prog_bar=True)


@hydra.main(config_path=os.environ["CONFIG_DIR"], config_name="default")
def _run(config: DictConfig) -> None:
    """
    Run to test if the module works.
    """
    model = hydra.utils.instantiate(config.lightning_model, _recursive_=False)
    print(f'{model}')


if __name__ == "__main__":
    _run()