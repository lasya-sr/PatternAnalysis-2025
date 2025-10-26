"""
Training, validation and Testing script for the model.
Uses GFNet from modules.py and data loaders from dataset.py
"""

import os
import time
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.nn as nn
from functools import partial
from modules import *            
from dataset import train_val_loaders, test_loader
from sklearn import metrics
from timm.scheduler import create_scheduler
from types import SimpleNamespace

# hyperparameters
batch_size    = 32
num_epochs   = 10
learning_rate = 0.0005

# setup
def setup_run():
    """Set device and assets directory (plots, confusion matrix)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = "outputs"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return device, output_dir

# training
def train_loop(device, output_dir, model, criterion, optimizer, scheduler, loader_train, loader_val):
    print("Training started ...")
    start_time = time.time()

    model.train()
    tr_losses, val_losses, val_accs = [], [], []

    for epoch in range(num_epochs):
        running_loss = 0.0

        for i, (inputs, labels) in enumerate(loader_train):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # epoch stats
        train_loss = running_loss / len(loader_train)
        tr_losses.append(train_loss)

        # validation
        val_loss, val_acc = validate(device, model, criterion, loader_val)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # schedule step (timm schedulers expect epoch step)
        scheduler.step(epoch)

        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Training Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    dur = time.time() - start_time
    print(f"Training completed in: {dur:.2f} seconds")

    # plots
    plt.figure(figsize=(10, 5))
    plt.plot(tr_losses, label="Training Loss", color='purple')
    plt.plot(val_losses, label="Validation Loss", color='orange')
    plt.title("Training and Validation Loss Plot")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "training_and_validation_losses.png"))
    loss_plot_path = os.path.join(output_dir, "training_and_validation_losses.png")
    plt.savefig(loss_plot_path)
    plt.show()
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(val_accs, label="Validation Accuracy", color='green')
    plt.title("Validation Accuracy Plot")
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.legend()
    acc_plot_path = os.path.join(output_dir, "validation_accuracies.png")
    plt.savefig(acc_plot_path)
    plt.show()
    plt.close()

# validation
def validate(device, model, criterion, loader_val):
    val_loss = 0.0
    correct, total = 0, 0

    model.eval()
    with torch.no_grad():
        for inputs, labels in loader_val:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_loss /= len(loader_val)
    val_acc = correct / total
    model.train()  # return to train mode for outer loop
    return val_loss, val_acc

# testing
def test_loop(device, output_dir, model, criterion, loader_test):
    print("Testing Starting ...")
    start_time = time.time()

    model.eval()
    test_loss = 0.0
    correct, total = 0, 0
    all_labels, all_preds = [], []

    with torch.no_grad():
        for inputs, labels in loader_test:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    test_loss /= len(loader_test)
    test_acc = correct / total

    dur = time.time() - start_time
    print(f"Testing completed in: {dur:.2f} seconds")
    print(f"Testing loss: {test_loss:.4f} | Test accuracy: {test_acc:.4f}")

    # confusion matrix
    cm = metrics.confusion_matrix(all_labels, all_preds)
    disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm)

    plt.figure(figsize=(7,7))
    disp.plot(cmap="viridis", values_format='d')
    plt.title("Confusion Matrix", fontsize=14, fontweight='bold', pad=10)
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.show()
    plt.close()

    return test_loss, test_acc

# main
def main():
    # setup
    device, output_dir = setup_run()

    # data
    loader_train, loader_val = train_val_loaders(batch_size=batch_size)
    loader_test, _ = test_loader(batch_size=batch_size)

    # model (GFNet from modules.py)
    model = GFNet(
        img_size=256,
        patch_size=16,
        embed_dim=512,
        num_classes=2,
        in_channels=1,
        drop_rate=0.5,
        depth=19,
        mlp_ratio=4.0,
        drop_path_rate=0.25,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    # scheduler (timm)
    args = SimpleNamespace()
    args.sched = "cosine"
    args.num_epochs = num_epochs
    args.decay_epochs = 30
    args.min_lr = 1e-6
    args.warmup_lr = 1e-5
    args.cooldown_epochs = 10

    scheduler, _ = create_scheduler(args, optimizer)

    # train
    train_loop(device, output_dir, model, criterion, optimizer, scheduler, loader_train, loader_val)

    # Making sure a Drive folder exists
    ckpt_dir = "/content/drive/MyDrive/checkpoints_gfnet"
    os.makedirs(ckpt_dir, exist_ok=True)

    # Save the final weights to Drive
    model_save_path = os.path.join(ckpt_dir, "gfnet_last.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model is saved to: {model_save_path}")

    # test after saving
    test_loop(device, output_dir, model, criterion, loader_test)

if __name__ == "__main__":
    main()
