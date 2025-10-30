"""
This script is used to make predictions and plot visualizations of the model.
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
from dataset import test_loader
from train import test_loop

# GPU if available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Helper to resolve a file from a directory 
def resolve_ckpt_path(path: str) -> str:
    if os.path.isdir(path):
        cand = os.path.join(path, "gfnet_last.pth")
        if os.path.exists(cand):
            return cand
        pths = sorted(glob.glob(os.path.join(path, "*.pth")), key=os.path.getmtime, reverse=True)
        if pths:
            return pths[0]
        raise FileNotFoundError(f"No .pth files found in directory: {path}")
    return path

# Model loading
def load_model(ckpt_path: str) -> nn.Module:
    """Instantiate GFNet and load weights from a checkpoint."""
    ckpt_path = resolve_ckpt_path(ckpt_path)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
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
    print(f"Loaded model: {ckpt_path}\n")
    return model

# Visualisations
def show_samples(images: torch.Tensor, labels: torch.Tensor, preds: torch.Tensor, save_path: str):
    """Plot a small grid of (image, true, pred) and save to file."""
    # Map label indices to names to be plotted
    label_names = {0: "AD", 1: "NC"}

    # Convert tensors to numpy for plotting
    imgs_np = images.cpu().numpy()
    fig = plt.figure(figsize=(10, 10))

    for i in range(1, 5):
        img = np.transpose(imgs_np[i - 1], (1, 2, 0))
        ax = fig.add_subplot(2, 2, i)
        ax.imshow(img)
        t_label = label_names[labels[i - 1].item()]
        p_label = label_names[preds[i - 1].item()]
        ax.set_title(f"True: {t_label}, Pred: {p_label}")
        ax.axis("off")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # Save the plot
    plt.savefig(save_path)
    plt.show()

# Prediction function to make predictions on test samples
def predict(model: nn.Module, test_dataset, out_dir: str):
    """Run the trained model on 4 random test samples and visualise results."""
    print("Generating predictions for test samples...")
    with torch.no_grad():

        # Select 4 random samples from the test dataset
        idxs = random.sample(range(len(test_dataset)), 4)
        batch = [test_dataset[i] for i in idxs]

        imgs = torch.stack([x[0] for x in batch]).to(DEVICE)
        labels = torch.tensor([x[1] for x in batch])

        results = model(imgs)
        _, preds = torch.max(results, 1)

        save_path = os.path.join(out_dir, "test_predictions.png")
        show_samples(imgs, labels.cpu(), preds.cpu(), save_path)

        print(f"Selected indices: {idxs}")
        print(f"True labels:      {labels.cpu().numpy()}")
        print(f"Predicted labels: {preds.cpu().numpy()}")

# Main function to run predictions
def main():
    parser = argparse.ArgumentParser(description="Predictions for AD classification using GFNet")
    parser.add_argument(
        "--model-path",
        type=str,
        default="/content/drive/MyDrive/checkpoints_gfnet",
        required=False,
        help="Path to the trained model file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/content/drive/MyDrive/gfnet_outputs",
        #required=False,
        help="Directory to save prediction images (default: prediction_outputs)",
    )

    # parse arguments
    args, _ = parser.parse_known_args()

    print(f"[args] model_path={args.model_path}  output_dir={args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)

    loader_test, ds_test = test_loader(batch_size=32)

    # Loading model
    model = load_model(args.model_path)

    # Call test loop
    criterion = nn.CrossEntropyLoss()
    test_loop(DEVICE, args.output_dir, model, criterion, loader_test)

    # Predict on random samples
    predict(model, ds_test, args.output_dir)

if __name__ == "__main__":
    main()
