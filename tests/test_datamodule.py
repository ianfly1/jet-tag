import pytest
import torch
from jet_tagging.datamodules.jetnet_datamodule import JetDataModule

def test_jet_datamodule_particle_shapes(tmp_path):
    dm = JetDataModule(data_mode="particle", data_dir=str(tmp_path), batch_size=32)
    dm.prepare_data()
    dm.setup(stage="fit")

    train_loader = dm.train_dataloader()
    x, y = next(iter(train_loader))

    assert x.shape == (32, 90)
    assert y.shape == (32,)
    assert x.dtype == torch.float32
    assert y.dtype == torch.long