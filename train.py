import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from src.jet_tagging.datamodules.jetnet_datamodule import JetDataModule
from src.jet_tagging.models.encdec import encdec
from src.jet_tagging.modules.joint_ae_module import JointAutoencoderModule

@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig):
    pl.seed_everything(cfg.seed)

    datamodule = JetDataModule(
        data_mode=cfg.data.mode,
        data_dir=cfg.data.data_dir,
        batch_size=cfg.data.batch_size
    )

    model = encdec(input_size=cfg.model.input_size, output_size=cfg.model.output_size)
    system = JointAutoencoderModule(model=model, alpha=cfg.alpha, lr=cfg.lr)

    logger = TensorBoardLogger("runs", name=cfg.model.name)
    callbacks = [
        ModelCheckpoint(dirpath="checkpoints", monitor="val/loss", mode="min", save_top_k=1),
        EarlyStopping(monitor="val/loss", patience=5, mode="min")
    ]

    trainer = pl.Trainer(
        max_epochs=cfg.trainer.epochs,
        accelerator="auto",
        devices=1,
        logger=logger,
        callbacks=callbacks
    )

    trainer.fit(system, datamodule=datamodule)
    trainer.test(system, datamodule=datamodule)

if __name__ == "__main__":
    main()