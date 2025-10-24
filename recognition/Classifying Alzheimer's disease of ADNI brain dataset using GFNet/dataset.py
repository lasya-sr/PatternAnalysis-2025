"""
This script does the preprocessing and loading the data

"""

from pathlib import Path
import os
from typing import Tuple, Dict, List
import random

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.utils import make_grid

DEFAULT_IMAGE_SIZE: Tuple[int, int] = (256, 256)
# Hardcoded mean and standard deviation computed from training set 
MEAN, STD = 0.1156, 0.2202


def resolve_data_root() -> Path:
    env_root = os.getenv("DATA_ROOT")
    if env_root:
        p = Path(env_root).expanduser()
        if p.exists():
            return p

    mypath = Path("/content/drive/MyDrive/ADNI/AD_NC")
    if mypath.exists():
        return mypath

    rangpur = Path("/home/groups/comp3710/ADNI/AD_NC")
    if rangpur.exists():
        return rangpur

def build_transforms(image_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
                     mean: float = MEAN, std: float = STD,
                     train: bool = True) -> transforms.Compose:
    """
    The train includes augmentation, but no augmentation for test

    """
    common = [
        transforms.Resize(image_size),
        transforms.Grayscale(num_output_channels=1),  #1-channel grayscale       
        transforms.ToTensor(),
        transforms.Normalize((mean,), (std,)),
    ]

    if train:
        aug = [
            transforms.RandAugment(num_ops=2),
            transforms.RandomHorizontalFlip(p=0.5),
        ]
        return transforms.Compose(aug + common)
    else:
        return transforms.Compose(common)


def _stratified_indices(targets: List[int], train_ratio: float, seed: int = 42):
    rnd = random.Random(seed)
    by_class: Dict[int, List[int]] = {}
    for idx, y in enumerate(targets):
        by_class.setdefault(y, []).append(idx)

    train_ids, val_ids = [], []
    for y, idxs in by_class.items():
        rnd.shuffle(idxs)
        k = int(len(idxs) * train_ratio)
        train_ids.extend(idxs[:k])
        val_ids.extend(idxs[k:])

    rnd.shuffle(train_ids)
    rnd.shuffle(val_ids)
    return train_ids, val_ids


def _make_imagefolder(split_dir: Path, train: bool) -> datasets.ImageFolder:
    tfm = build_transforms(train=train)
    return datasets.ImageFolder(root=str(split_dir), transform=tfm)


def train_val_loaders(batch_size: int,
                      train_ratio: float = 0.8,
                      num_workers: int = 2,
                      seed: int = 42):
    data_root = resolve_data_root()
    train_dir = data_root / "train"

    # Loading with train transforms
    base = _make_imagefolder(train_dir, train=True)

    # Splits on targets
    targets = [y for _, y in base.samples]  
    train_idx, val_idx = _stratified_indices(targets, train_ratio, seed=seed)

    train_ds = Subset(base, train_idx)
    # For validation, we want deterministic transforms so no augmentation done
    val_base = _make_imagefolder(train_dir, train=False)
    val_ds = Subset(val_base, val_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers, pin_memory=True)

    # Print basic info
    print(f"[Data] Root: {data_root}")
    print(f"[Split] Train: {len(train_idx)} | Val: {len(val_idx)}")
    print(f"[Classes] {base.class_to_idx}")

    return train_loader, val_loader, base.class_to_idx


def test_loader(batch_size: int, num_workers: int = 2):
    """
    Create a test DataLoader from the 'test' split on disk.
    """
    data_root = resolve_data_root()
    test_dir = data_root / "test"
    test_ds = _make_imagefolder(test_dir, train=False)

    loader = DataLoader(test_ds, batch_size=batch_size,
                        shuffle=False, num_workers=num_workers, pin_memory=True)

    print(f"[Data] Root: {data_root}")
    print(f"[Test]  N={len(test_ds)} | Classes: {test_ds.class_to_idx}")
    return loader, test_ds

def visualize_batch(loader: DataLoader, max_images: int = 32):
    batch = next(iter(loader))[0]  
    if batch.size(0) > max_images:
        batch = batch[:max_images]
    grid = make_grid(batch, nrow=8, padding=2)
    return grid

