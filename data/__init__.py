import numpy as np
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets.folder import ImageFolder

from logger import logging

from .CustomDataset import CustomDataset


def get_dataloaders(
        train_dir,
        test_dir,
        train_transform=None,
        test_transform=None,
        split=(0.5, 0.5),
        batch_size=32,
        *args, **kwargs):

    train_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_ds = CustomDataset(root_dir=train_dir, mode='train', transform=train_transform)
    test_ds = CustomDataset(root_dir=test_dir, mode='test', transform=test_transform)

    logging.info(f'Train samples={len(train_ds)}, Test samples={len(test_ds)}')

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, *args, **kwargs)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, *args, **kwargs)

    return train_dl, test_dl
