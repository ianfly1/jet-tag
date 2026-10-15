import torch
import pytorch_lightning as pl
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC
from src.jet_tagging.utils.metrics import BackgroundRejectionAtEfficiency

class JointAutoencoderModule(pl.LightningModule):
    def __init__(self, model: torch.nn.Module, alpha: float = 0.1, lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model

        # Metrics
        self.train_acc = BinaryAccuracy()
        self.val_acc = BinaryAccuracy()
        self.val_auroc = BinaryAUROC()
        self.val_rej50 = BackgroundRejectionAtEfficiency(target_eff=0.50)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        reconstructed, logits = self(x)
        
        loss_recon = torch.nn.functional.mse_loss(reconstructed, x)
        loss_class = torch.nn.functional.cross_entropy(logits, y)
        total_loss = (self.hparams.alpha * loss_recon) + loss_class

        preds = torch.argmax(logits, dim=-1)
        self.train_acc(preds, y)

        self.log("train/total_loss", total_loss, on_step=False, on_epoch=True)
        self.log("train/recon_loss", loss_recon, on_step=False, on_epoch=True)
        self.log("train/class_loss", loss_class, on_step=False, on_epoch=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        reconstructed, logits = self(x)

        loss_recon = torch.nn.functional.mse_loss(reconstructed, x)
        loss_class = torch.nn.functional.cross_entropy(logits, y)
        total_loss = (self.hparams.alpha * loss_recon) + loss_class

        probs = torch.softmax(logits, dim=-1)[:, 1]
        preds = torch.argmax(logits, dim=-1)

        self.val_acc(preds, y)
        self.val_auroc(probs, y)
        self.val_rej50(probs, y)

        self.log("val/loss", total_loss, prog_bar=True)
        self.log("val/acc", self.val_acc, prog_bar=True)
        self.log("val/auroc", self.val_auroc, prog_bar=True)
        self.log("val/rej50", self.val_rej50, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val/loss"}}