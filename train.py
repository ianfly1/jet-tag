import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from jet_tagging.datamodules.jetnet_datamodule import JetDataModule
from jet_tagging.models import EncDec, JetCNN, JetMLP
from jet_tagging.modules.classification_module import JetClassificationModule
from jet_tagging.modules.joint_ae_module import JointAutoencoderModule


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig):
    pl.seed_everything(cfg.seed)

    # 1. Instantiate DataModule
    datamodule = JetDataModule(
        data_mode=cfg.data.mode,
        data_dir=cfg.data.data_dir,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        im_size=cfg.data.im_size,
        val_split=cfg.data.val_split,
        test_split=cfg.data.test_split,
        seed=cfg.seed,
    )

    # 2. Instantiate Model and Task Module
    if cfg.model.name == "encdec":
        model = EncDec(
            input_size=cfg.model.input_size,
            latent_dim=cfg.model.latent_dim,
            output_size=cfg.model.output_size,
        )
        system = JointAutoencoderModule(model=model, alpha=cfg.alpha, lr=cfg.lr)
    elif cfg.model.name == "cnn":
        model = JetCNN(
            in_channels=cfg.model.in_channels,
            im_size=cfg.model.im_size,
            num_classes=cfg.model.num_classes,
        )
        system = JetClassificationModule(model=model, lr=cfg.lr)
    else:
        raise ValueError(f"Unsupported model type: {cfg.model.name}")

    # 3. Setup Callbacks and Loggers
    logger = TensorBoardLogger("runs", name=cfg.model.name)
    callbacks = [
    ModelCheckpoint(
        dirpath=f"checkpoints/{cfg.model.name}",
        filename="{epoch:02d}-{val_loss:.4f}",
        monitor="val/loss",
        mode="min",
        save_top_k=1,
    ),
    EarlyStopping(monitor="val/loss", patience=5, mode="min"),
]

    # 4. Trainer Initialization
    trainer = pl.Trainer(
        max_epochs=cfg.trainer.epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        logger=logger,
        callbacks=callbacks,
    )

    # 5. Training & Testing
    trainer.fit(system, datamodule=datamodule)
    trainer.test(system, datamodule=datamodule, ckpt_path="best")


if __name__ == "__main__":
    main()