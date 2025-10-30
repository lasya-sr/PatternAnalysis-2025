"""
Training, validation and Testing script for the model.
Uses GFNet modules from modules.py and data loaders from dataset.py.

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

# Hyperparameters
batch_size    = 32
num_epochs   = 60
learning_rate = 0.0005

# Setup
def setup_run():
    """Set device and assets directory (plots, confusion matrix)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = "/content/drive/MyDrive/gfnet_outputs"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return device, output_dir

# Training
def train_loop(device, output_dir, model, criterion, optimizer, scheduler, loader_train, loader_val):
    print("Training started ...")

    # Starting time  
    start_time = time.time()

    model.train()
    tr_losses, val_losses, val_accs = [], [], []

    for epoch in range(num_epochs):
        running_loss = 0.0

        for i, (inputs, labels) in enumerate(loader_train):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backpropogation
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Epoch stats
        train_loss = running_loss / len(loader_train)
        tr_losses.append(train_loss)

        # Validation
        val_loss, val_acc = validate(device, model, criterion, loader_val)   #eval on validation set
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # Schedule step
        scheduler.step(epoch)

        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Training Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    # Total duration
    dur = time.time() - start_time
    print(f"Training completed in: {dur:.2f} seconds")

    # Plotting loss curves
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

    # Plotting accuracy curves
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

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            # Predictions
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_loss /= len(loader_val)
    val_acc = correct / total
    model.train()  # return to train mode for outer loop
    return val_loss, val_acc

# Testing
def test_loop(device, output_dir, model, criterion, loader_test):
    print("Testing Starting ...")

    # Starting time
    start_time = time.time()

    model.eval()
    test_loss = 0.0
    correct, total = 0, 0
    all_labels, all_preds = [], []

    with torch.no_grad():
        for inputs, labels in loader_test:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
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

    # Total duration
    dur = time.time() - start_time
    print(f"Testing completed in: {dur:.2f} seconds")
    print(f"Testing loss: {test_loss:.4f} | Test accuracy: {test_acc:.4f}")

    # Confusion matrix
    cm = metrics.confusion_matrix(all_labels, all_preds)
    disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm)

    # Plot and save confusion matrix
    plt.figure(figsize=(7,7))
    disp.plot(cmap="viridis", values_format='d')
    plt.title("Confusion Matrix", fontsize=14, fontweight='bold', pad=10)
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.show()
    plt.close()

    return test_loss, test_acc

# Main
def main():
    # Setup
    device, output_dir = setup_run()

    # Data loaders
    loader_train, loader_val, class_to_idx = train_val_loaders(batch_size=batch_size)
    loader_test, _ = test_loader(batch_size=batch_size)

    # Model initialization
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

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    # Scheduler
    args = SimpleNamespace()
    args.sched = "cosine"
    args.num_epochs = num_epochs
    args.decay_epochs = 30
    args.min_lr = 1e-6
    args.warmup_lr = 1e-5
    args.cooldown_epochs = 10

    scheduler, _ = create_scheduler(args, optimizer)

    # Train
    train_loop(device, output_dir, model, criterion, optimizer, scheduler, loader_train, loader_val)

    # Making sure a Drive folder exists
    ckpt_dir = "/content/drive/MyDrive/checkpoints_gfnet"
    os.makedirs(ckpt_dir, exist_ok=True)

    # Saving the final weights to Drive
    model_save_path = os.path.join(ckpt_dir, "gfnet_last.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model is saved to: {model_save_path}")

if __name__ == "__main__":
    main()
