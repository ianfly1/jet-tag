from unittest.mock import MagicMock, patch

import numpy as np
import torch

from jet_tagging.datamodules.jetnet_datamodule import JetDataModule


@patch("jet_tagging.datamodules.jetnet_datamodule.JetNet")
def test_jet_datamodule_particle_shapes(mock_jetnet, tmp_path):
    # Create 100 fake jet particles (30 particles x 3 coordinates)
    fake_particles = np.random.randn(100, 30, 3).astype(np.float32)
    mock_instance = MagicMock()
    mock_instance.data = fake_particles
    mock_jetnet.return_value = mock_instance

    dm = JetDataModule(
        data_mode="particle",
        data_dir=str(tmp_path),
        batch_size=16,
        num_workers=0,
    )
    dm.setup(stage="fit")

    train_loader = dm.train_dataloader()
    x, y = next(iter(train_loader))

    # Assert correct tensor shapes without downloading real HDF5 files
    assert x.shape == (16, 90)  # 30 * 3 = 90
    assert y.shape == (16,)
    assert isinstance(x, torch.Tensor)
