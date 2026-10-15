import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler
from jetnet.datasets import JetNet
from jetnet.utils import to_image
import numpy as np

class JetDataModule(LightningDataModule):
    def __init__(
        self, 
        data_mode: str = "image",  # "image" or "particle"
        data_dir: str = "./data", 
        batch_size: int = 256, 
        num_workers: int = 4,
        im_size: int = 25,
        val_split: float = 0.15,
        test_split: float = 0.15,
        seed: int = 42
    ):
        super().__init__()
        self.save_hyperparameters()
        self.scaler = StandardScaler()

    def prepare_data(self):
        # Trigger download once on main process
        JetNet.getData(
            jet_type=["g", "q", "t"], 
            data_dir=self.hparams.data_dir, 
            particle_features=["etarel", "phirel", "ptrel"], 
            download=True
        )

    def setup(self, stage: str | None = None):
        if self.hparams.data_mode == "image":
            particle_data, jet_data = JetNet.getData(
                jet_type=["g", "q", "t"],
                data_dir=self.hparams.data_dir,
                particle_features=["etarel", "phirel", "ptrel", "mask"],
                num_particles=30,
                jet_features=["type", "pt", "eta", "mass"],
                download=False
            )
            raw_images = to_image(
                jets=particle_data[:, :, :3], 
                im_size=self.hparams.im_size, 
                mask=particle_data[:, :, 3], 
                maxR=0.4
            )
            images = torch.tensor(raw_images, dtype=torch.float32).unsqueeze(1)
            sums = images.sum(dim=(1, 2, 3), keepdim=True) + 1e-12
            images = images / sums

            raw_labels = jet_data[:, 0].astype(np.int64)
            labels = torch.tensor(np.where(raw_labels == 2, 1, 0), dtype=torch.long)
            
            full_ds = TensorDataset(images, labels)
            total = len(full_ds)
            val_len = int(total * self.hparams.val_split)
            test_len = int(total * self.hparams.test_split)
            train_len = total - val_len - test_len

            gen = torch.Generator().manual_seed(self.hparams.seed)
            self.train_ds, self.val_ds, self.test_ds = random_split(
                full_ds, [train_len, val_len, test_len], generator=gen
            )

        elif self.hparams.data_mode == "particle":
            particle_data, jet_data = JetNet.getData(
                jet_type=["g", "q", "t"],
                data_dir=self.hparams.data_dir,
                particle_features=["etarel", "phirel", "ptrel"],
                download=False
            )
            raw_labels = jet_data[:, 0].astype(np.int64)
            labels = np.where(raw_labels == 2, 1, 0)
            features = particle_data.reshape(particle_data.shape[0], -1)

            num_samples = len(features)
            indices = np.arange(num_samples)
            np.random.seed(self.hparams.seed)
            np.random.shuffle(indices)

            val_len = int(num_samples * self.hparams.val_split)
            test_len = int(num_samples * self.hparams.test_split)
            train_idx = indices[val_len + test_len:]
            val_idx = indices[:val_len]
            test_idx = indices[val_len:val_len + test_len]

            X_train = self.scaler.fit_transform(features[train_idx])
            X_val = self.scaler.transform(features[val_idx])
            X_test = self.scaler.transform(features[test_idx])

            self.train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(labels[train_idx], dtype=torch.long))
            self.val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(labels[val_idx], dtype=torch.long))
            self.test_ds = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(labels[test_idx], dtype=torch.long))

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.hparams.batch_size, shuffle=True, num_workers=self.hparams.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.hparams.batch_size, shuffle=False, num_workers=self.hparams.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.hparams.batch_size, shuffle=False, num_workers=self.hparams.num_workers)