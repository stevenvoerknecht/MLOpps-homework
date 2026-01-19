import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from ml_core.data import get_dataloaders
from ml_core.models import MLP

# Je kan alle time dingen uncommenten als je wil testen hoe lang het runnen duurt
# import time
# from time import perf_counter

# 1. Setup Configuration
data_dir = sys.argv[1]
out_dir = Path(sys.argv[2])

out_dir.mkdir(parents=True, exist_ok=True)

print(f"Data dir: {data_dir}", flush=True)
print(f"Output dir: {out_dir}", flush=True)

# Test voor time
# t0 = perf_counter()

config = {
    "data": {"data_path": data_dir, "batch_size": 32, "num_workers": 8},
    "model": {"input_shape": [3, 96, 96], "hidden_units": [64, 32], "num_classes": 2},
}

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Training on: {device}")

# 2. Initialize Data, Model, Optimizer
print("Creating dataloaders...", flush=True)
train_loader, val_loader = get_dataloaders(config)
print("Dataloaders ready", flush=True)

# Test voor time
# t1 = perf_counter()
# print(f"[TIMING] Time for dataloader to get ready: {t1-t0}", flush=True)

model = MLP(**config["model"]).to(device)
optimizer = optim.SGD(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# 3. Training Loop
train_losses = []
val_losses = []

for epoch in range(3):
    model.train()
    epoch_train_loss = 0

    # Test voor time
    # epoch_start = perf_counter()
    # print(f"[TIMING] Epoch {epoch+1} started at {epoch_start-t0}")

    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        epoch_train_loss += loss.item()

        if i % 100 == 0:  # Log-after-n-steps granularity
            print(f"Epoch {epoch+1}, Step {i}, Loss: {loss.item():.4f}")

    train_losses.append(epoch_train_loss / len(train_loader))

    # Validation
    model.eval()
    epoch_val_loss = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            epoch_val_loss += loss.item()

    val_losses.append(epoch_val_loss / len(val_loader))
    print(
        f"--- Epoch {epoch+1} Summary: Train Loss {train_losses[-1]:.4f}, Val Loss {val_losses[-1]:.4f} ---"
    )

    # Test voor time
    # epoch_end = perf_counter()
    # print(f"[TIMING] Epoch {epoch+1} ended at {epoch_end-t0}")

plt.figure(figsize=(10, 5))
plt.plot(range(1, 4), train_losses, label="Train Loss", marker="o")
plt.plot(range(1, 4), val_losses, label="Val Loss", marker="o")
plt.title("PCAM Training: First 3 Epochs")
plt.xlabel("Epoch")
plt.ylabel("CrossEntropy Loss")
plt.legend()
plt.grid(True)
plt.savefig(out_dir / "pcam_learning_curves.png")
print("Training complete. Plot saved as pcam_learning_curves.png")
