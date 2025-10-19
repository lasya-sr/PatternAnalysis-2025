"""
Example usage of a trained model for inference and  visuals.
"""

import os
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from functools import partial

from modules import GFNet
from dataset import make_test_loader
from train import test_loop  # formerly `test`

# GPU if available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model loading
def load_model(ckpt_path: str) -> nn.Module:
    """Instantiate GFNet and load weights from a checkpoint."""
    model = GFNet(
        img_size=256,
        patch_size=16,
        embed_dim=512,
        num_classes=2,
        in_channels=1,
        drop_rate=0.5,
        depth=19,
        mlp_ratio=4.0,
        drop_path_rate=0.15,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    ).to(DEVICE)

    state = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(state, strict=False)
    model.eval()
    print(f"✓ Loaded checkpoint: {ckpt_path}\n")
    return model

# visualisations
def show_samples(images: torch.Tensor, labels: torch.Tensor, preds: torch.Tensor, save_path: str):
    """Plot a small grid of (image, true, pred) and save to file."""
    label_names = {0: "AD", 1: "NC"}

    imgs_np = images.cpu().numpy()
    fig = plt.figure(figsize=(10, 10))

    for i in range(1, 5):
        img = np.transpose(imgs_np[i - 1], (1, 2, 0))  # (C,H,W) -> (H,W,C)
        ax = fig.add_subplot(2, 2, i)
        ax.imshow(img)
        t = label_names[labels[i - 1].item()]
        p = label_names[preds[i - 1].item()]
        ax.set_title(f"True: {t}, Pred: {p}")
        ax.axis("off")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.show()

# model inference helper function
def infer(model: nn.Module, test_dataset, out_dir: str):
    """Run the trained model on 4 random test samples and visualise results."""
    print("→ Generating predictions for random test samples...")
    with torch.no_grad():
        idxs = random.sample(range(len(test_dataset)), 4)
        batch = [test_dataset[i] for i in idxs]

        imgs = torch.stack([x[0] for x in batch]).to(DEVICE)
        labs = torch.tensor([x[1] for x in batch])

        logits = model(imgs)
        _, preds = torch.max(logits, 1)

        save_path = os.path.join(out_dir, "random_test_predictions.png")
        show_samples(imgs, labs.cpu(), preds.cpu(), save_path)

        print(f"Selected indices: {idxs}")
        print(f"True labels:      {labs.cpu().numpy()}")
        print(f"Predicted labels: {preds.cpu().numpy()}")

# main
def main():
    parser = argparse.ArgumentParser(description="Inference for AD classification (GFNet)")
    parser.add_argument(
        "--model-path",
        type=str,
        default="trained_model.pth",
        help="Path to the trained model file (default: trained_model.pth)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="prediction_outputs",
        help="Directory to save prediction images (default: prediction_outputs)",
    )
    args, _ = parser.parse_known_args()


    # data
    loader_test, ds_test = make_test_loader(batch_size=32)

    # model
    model = load_model(args.model_path)

    # quick quantitative check on full test split
    criterion = nn.CrossEntropyLoss()
    test_loop(DEVICE, args.output_dir, model, criterion, loader_test)

    # qualitative preview: 4 random images
    infer(model, ds_test, args.output_dir)

if __name__ == "__main__":
    main()
