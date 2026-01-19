from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        config: Dict[str, Any],
        device: str,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.config = config
        self.device = device

        self.criterion = nn.CrossEntropyLoss()

        # TODO: Initialize ExperimentTracker
        self.tracker = None

        # TODO: Initialize metric calculation (like accuracy/f1-score) if needed

    def train_epoch(
        self, dataloader: DataLoader, epoch_idx: int
    ) -> Tuple[float, float, float]:
        self.model.train()
        epoch_train_loss = 0
        predictions = []
        targets = []

        # 1. Iterate over dataloader
        for images, labels in dataloader:
            # 2. Move data to device
            images, labels = images.to(self.device), labels.to(self.device)

            # 3. Forward pass, Calculate Loss
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            # 4. Backward pass, Optimizer step
            loss.backward()
            self.optimizer.step()

            # 5. Track metrics (Loss, Accuracy, F1)
            epoch_train_loss += loss.item()
            prediction = torch.argmax(outputs, dim=1)

            # moving tensors to cpu for sklearn
            predictions.append(prediction.cpu())
            targets.append(labels.cpu())

        # concatenating tensors from different batches to one tensor
        predictions = torch.cat(predictions)
        targets = torch.cat(targets)

        # calculating and returning loss, accuracy and f1
        train_loss = epoch_train_loss / len(dataloader)
        train_acc = (predictions == targets).float().mean().item()
        train_f1 = f1_score(targets, predictions, average="binary")
        return (train_loss, train_acc, train_f1)

    def validate(
        self, dataloader: DataLoader, epoch_idx: int
    ) -> Tuple[float, float, float]:
        self.model.eval()
        epoch_val_loss = 0
        predictions = []
        targets = []

        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                epoch_val_loss += loss.item()
                prediction = torch.argmax(outputs, dim=1)

                # moving tensors to cpu for sklearn
                predictions.append(prediction.cpu())
                targets.append(labels.cpu())

        # concatenating tensors from different batches to one tensor
        predictions = torch.cat(predictions)
        targets = torch.cat(targets)

        # calculating and returning loss, accuracy and f1
        val_loss = epoch_val_loss / len(dataloader)
        val_acc = (predictions == targets).float().mean().item()
        val_f1 = f1_score(targets, predictions, average="binary")
        return (val_loss, val_acc, val_f1)

    def save_checkpoint(self, epoch: int, val_loss: float) -> None:
        # Save model state, optimizer state, and config
        checkpoint = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "val_loss": val_loss,
            "config": self.config,
        }

        # Saving the checkpoint
        save_dir = self.config["training"]["save_dir"]
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = save_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> None:
        epochs = self.config["training"]["epochs"]

        print(f"Starting training for {epochs} epochs...")

        for epoch in range(epochs):
            # Call train_epoch and validate
            print(f"Epoch {epoch+1} has started")
            train_metrics = self.train_epoch(train_loader, epoch)
            val_metrics = self.validate(val_loader, epoch)

            # TODO: Log metrics to tracker

            # Save checkpoints
            self.save_checkpoint(epoch, val_metrics[0])
            print(f"Epoch {epoch+1} has ended and saved")
            print(
                f"Training loss is {train_metrics[0]} and Validation loss is {val_metrics[0]}"
            )


# Remember to handle the trackers properly
