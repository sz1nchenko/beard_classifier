import os
from typing import Optional

import numpy as np
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from pathlib import Path
from PIL import Image

from config import IMAGE_SIZE, BATCH_SIZE, VAL_SIZE, NUM_WORKERS
from transforms import transform
from config import SEED


class BeardData(Dataset):

    def __init__(self, data_path, train=True, transform=None):
        super().__init__()

        self.data_path = data_path

        if train:
            self.data_part_path = os.path.join(self.data_path, 'train')
        else:
            self.data_part_path = os.path.join(self.data_path, 'test')
        unique_labels = sorted(os.listdir(self.data_part_path))
        self.labels_to_idx = dict(zip(unique_labels, np.arange(len(unique_labels))))
        self.idx_to_labels = dict(zip(np.arange(len(unique_labels)), unique_labels))

        self.data = list()
        self.targets = list()
        for path in Path(self.data_part_path).rglob('*.png'):
            self.data.append(path.as_posix())
            self.targets.append(self.labels_to_idx[path.parts[-2]])

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = Image.open(self.data[idx]).convert('RGB')
        target = self.targets[idx]

        if self.transform:
            image = self.transform(image)

        return image, target


class BeardDataModule(pl.LightningDataModule):

    def __init__(
            self, data_path, image_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE, val_size=VAL_SIZE, num_workers=NUM_WORKERS
    ):
        super().__init__()
        self.data_path = data_path
        self.image_size = image_size
        self.batch_size = batch_size
        self.val_size = val_size
        self.num_workers = num_workers
        self.transforms = transform()

    def setup(self, stage: Optional[str] = None):
        self.train_data = BeardData(self.data_path, transform=self.transforms)
        self.test_data = BeardData(self.data_path, train=False, transform=self.transforms)

        indices = np.arange(len(self.train_data))
        np.random.seed(SEED)
        np.random.shuffle(indices)
        val_len = int(len(indices) * self.val_size)
        train_indices, val_indices = indices[val_len:], indices[:val_len]

        self.train_sampler = SubsetRandomSampler(train_indices)
        self.val_sampler = SubsetRandomSampler(val_indices)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_data,
            batch_size=self.batch_size,
            sampler=self.train_sampler,
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.train_data,
            batch_size=self.batch_size,
            sampler=self.val_sampler,
            num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )