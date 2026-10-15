import pytorch_lightning as pl
import torch
from torchmetrics.classification import BinaryAccuracy
from torchmetrics.functional.classification import binary_auroc

from jet_tagging.utils.metrics import BackgroundRejectionAtEfficiency


class JetClassificationModule(pl.LightningModule):
    def __init__(self, model: torch.nn.Module, lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.criterion = torch.nn.CrossEntropyLoss()

        self.train_acc = BinaryAccuracy()
        self.val_acc = BinaryAccuracy()
        self.test_acc = BinaryAccuracy()

        self.val_rej50 = BackgroundRejectionAtEfficiency(target_eff=0.50)
        self.test_rej50 = BackgroundRejectionAtEfficiency(target_eff=0.50)

        self.val_step_outputs = []
        self.test_step_outputs = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        preds = torch.argmax(logits, dim=-1)
        self.train_acc(preds, y)

        self.log("train/loss", loss, on_step=False, on_epoch=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        probs = torch.softmax(logits, dim=-1)[:, 1].detach().cpu()
        preds = torch.argmax(logits, dim=-1).detach().cpu()
        y_cpu = y.detach().cpu()

        self.val_acc(preds, y_cpu)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

        self.val_step_outputs.append({"probs": probs, "targets": y_cpu})

    def on_validation_epoch_end(self):
        if not self.val_step_outputs:
            return
        probs = torch.cat([x["probs"] for x in self.val_step_outputs])
        targets = torch.cat([x["targets"] for x in self.val_step_outputs])

        val_auroc = binary_auroc(probs, targets)
        val_rej = self.val_rej50(probs, targets)

        self.log("val/auroc", val_auroc, prog_bar=True)
        self.log("val/rej50", val_rej, prog_bar=True)

        self.val_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        probs = torch.softmax(logits, dim=-1)[:, 1].detach().cpu()
        preds = torch.argmax(logits, dim=-1).detach().cpu()
        y_cpu = y.detach().cpu()

        self.test_acc(preds, y_cpu)
        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)

        self.test_step_outputs.append({"probs": probs, "targets": y_cpu})

    def on_test_epoch_end(self):
        if not self.test_step_outputs:
            return
        probs = torch.cat([x["probs"] for x in self.test_step_outputs])
        targets = torch.cat([x["targets"] for x in self.test_step_outputs])

        test_auroc = binary_auroc(probs, targets)
        test_rej = self.test_rej50(probs, targets)

        self.log("test/auroc", test_auroc, prog_bar=True)
        self.log("test/rej50", test_rej, prog_bar=True)

        self.test_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val/loss"},
        }