from unittest.mock import patch

import numpy as np
import torch

from jet_tagging.datamodules.jetnet_datamodule import JetDataModule


@patch("jet_tagging.datamodules.jetnet_datamodule.JetNet")
def test_jet_datamodule_particle_shapes(mock_jetnet, tmp_path):
    # 1. Generate 100 fake particle point clouds (30 particles x 3 coordinates)
    fake_particles = np.random.randn(100, 30, 3).astype(np.float32)

    # 2. Generate 100 fake jet attributes (jet type class indices 0, 1, or 2 in col 0)
    fake_jets = np.random.choice([0, 1, 2], size=(100, 4)).astype(np.float32)

    # 3. Mock the class method return value
    mock_jetnet.getData.return_value = (fake_particles, fake_jets)

    dm = JetDataModule(
        data_mode="particle",
        data_dir=str(tmp_path),
        batch_size=16,
        num_workers=0,
    )
    dm.setup(stage="fit")

    train_loader = dm.train_dataloader()
    x, y = next(iter(train_loader))

    # Assert shape and type expectations
    assert x.shape == (16, 90)  # 30 particles * 3 features = 90
    assert y.shape == (16,)
    assert isinstance(x, torch.Tensor)
