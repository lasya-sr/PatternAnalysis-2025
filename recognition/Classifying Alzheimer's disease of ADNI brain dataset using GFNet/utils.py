"""
This is a util script used to compute dataset mean and standard deviation for normalization.
This script need to be run on the training dataset as the values from these computations are hardcoded into dataset.py

"""
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from dataset import resolve_data_root

# Configurations
IMG_SIZE = (256, 256)   
BATCH_SIZE = 64
NUM_WORKERS = 2
GRAYSCALE_CHANNELS = 1  

# Transform without normalization
transform_no_norm = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.Grayscale(num_output_channels=GRAYSCALE_CHANNELS),
    transforms.ToTensor(),  # values in [0, 1]
])

# Creating training dataset
def make_train_dataset():
    data_root: Path = resolve_data_root()
    train_dir = data_root / "train"
    if not train_dir.exists():
        raise FileNotFoundError(f"Train directory not found: {train_dir}")
    return datasets.ImageFolder(root=str(train_dir), transform=transform_no_norm)

# Function to calculate mean and std
def calculate_mean_std(loader: DataLoader):
    """ Calculate the mean and standard deviation of a dataset."""
    n_pixels_total = 0
    sum_c = None
    sumsq_c = None

    for images, _ in loader:
        # images: [B, C, H, W]
        b, c, h, w = images.shape
        pixels = b * h * w

        # Initializes accumulators on first batch
        if sum_c is None:
            sum_c = torch.zeros(c, dtype=torch.double)
            sumsq_c = torch.zeros(c, dtype=torch.double)

        # Accumulates sums and squared sums per channel
        images = images.to(dtype=torch.double)
        sum_c += images.sum(dim=[0, 2, 3])
        sumsq_c += (images ** 2).sum(dim=[0, 2, 3])
        n_pixels_total += pixels

    mean = (sum_c / n_pixels_total).to(dtype=torch.float32)
    var = (sumsq_c / n_pixels_total) - (mean.double() ** 2)
    std = torch.sqrt(var.clamp_min(1e-12)).to(dtype=torch.float32)
    return mean, std

# Main function
def main():
    dataset = make_train_dataset()
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    mean, std = calculate_mean_std(loader)
    if mean.numel() == 1:
        print(f"Mean: {mean.item():.4f}")
        print(f"Std:  {std.item():.4f}")
        print(f"Use Normalize(mean=({mean.item():.4f},), std=({std.item():.4f},))")
    else:
        m = ", ".join(f"{x:.4f}" for x in mean.tolist())
        s = ", ".join(f"{x:.4f}" for x in std.tolist())
        print(f"Mean: [{m}]")
        print(f"Std:  [{s}]")
        print(f"Use Normalize(mean=({m}), std=({s}))")

if __name__ == "__main__":
    main()
