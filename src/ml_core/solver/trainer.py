import csv
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
from sklearn.metrics import (
    auc,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from ml_core.tracking.mlflow_tracker import MLflowTracker


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        config: Dict[str, Any],
        device: str = None,  # als je None geeft, kiest hij automatisch
    ):
        # Device automatisch detecteren
        self.device = (
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.config = config
        self.criterion = nn.CrossEntropyLoss()
        self.tracker = MLflowTracker(config)
        self.best_val_loss = float("inf")

        # CSV logging
        self.csv_path = Path("experiments/results/metrics.csv")
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "epoch",
                    "train_loss",
                    "train_acc",
                    "train_f1",
                    "val_loss",
                    "val_acc",
                    "val_f1",
                    "val_f2",
                    "val_roc_auc",
                    "val_pr_auc",
                ]
            )

    def log_csv(
        self,
        epoch,
        train_loss,
        train_acc,
        train_f1,
        val_loss,
        val_acc,
        val_f1,
        val_f2,
        val_roc_auc,
        val_pr_auc,
    ):
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    epoch + 1,
                    train_loss,
                    train_acc,
                    train_f1,
                    val_loss,
                    val_acc,
                    val_f1,
                    val_f2,
                    val_roc_auc,
                    val_pr_auc,
                ]
            )

    def train_epoch(
        self, dataloader: DataLoader, epoch_idx: int
    ) -> Tuple[float, float, float]:
        self.model.train()
        epoch_loss = 0
        predictions, targets = [], []

        for images, labels in dataloader:
            images, labels = images.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()
            predictions.append(torch.argmax(outputs, dim=1).cpu())
            targets.append(labels.cpu())

        predictions = torch.cat(predictions)
        targets = torch.cat(targets)

        train_loss = epoch_loss / len(dataloader)
        train_acc = (predictions == targets).float().mean().item()
        train_f1 = f1_score(targets, predictions, average="binary")

        # Log metrics naar MLflow
        self.tracker.log_metrics(
            {"train_loss": train_loss, "train_acc": train_acc, "train_f1": train_f1},
            step=epoch_idx,
        )

        return train_loss, train_acc, train_f1

    def validate(
        self, dataloader: DataLoader, epoch_idx: int
    ) -> Tuple[float, float, float]:
        self.model.eval()
        val_loss_total = 0
        predictions, targets = [], []

        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                val_loss_total += loss.item()
                predictions.append(torch.argmax(outputs, dim=1).cpu())
                targets.append(labels.cpu())

        predictions = torch.cat(predictions)
        targets = torch.cat(targets)

        val_loss = val_loss_total / len(dataloader)
        val_acc = (predictions == targets).float().mean().item()
        val_f1 = f1_score(targets, predictions, average="binary")
        val_f2 = fbeta_score(targets, predictions, beta=2)

        # ROC-AUC & PR-AUC
        all_probs = []
        with torch.no_grad():
            for images, _ in dataloader:
                images = images.to(self.device)
                outputs = self.model(images)
                probs = torch.softmax(outputs, dim=1)[:, 1]
                all_probs.append(probs.cpu())
        all_probs = torch.cat(all_probs)

        val_roc_auc = roc_auc_score(targets, all_probs)
        precision, recall, _ = precision_recall_curve(targets, all_probs)
        val_pr_auc = auc(recall, precision)

        # Log metrics naar MLflow
        self.tracker.log_metrics(
            {
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_f1": val_f1,
                "val_f2": val_f2,
                "val_roc_auc": val_roc_auc,
                "val_pr_auc": val_pr_auc,
            },
            step=epoch_idx,
        )

        # CSV logging
        self.log_csv(
            epoch_idx,
            *self.train_epoch(dataloader, epoch_idx),
            val_loss,
            val_acc,
            val_f1,
            val_f2,
            val_roc_auc,
            val_pr_auc,
        )

        # Save best checkpoint
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.save_checkpoint(epoch_idx, val_loss)

        return val_loss, val_acc, val_f1

    def save_checkpoint(self, epoch: int, val_loss: float) -> None:
        checkpoint = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "val_loss": val_loss,
            "config": self.config,
        }
        save_dir = Path(self.config["training"]["save_dir"])
        save_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = save_dir / "best_checkpoint.pt"
        torch.save(checkpoint, checkpoint_path)
        self.tracker.log_artifact(checkpoint_path)

    def fit(
        self, train_loader: DataLoader, val_loader: DataLoader, start_epoch=0
    ) -> None:
        epochs = self.config["training"]["epochs"]

        # Log hyperparameters naar MLflow
        self.tracker.log_metrics(
            metrics={
                "learning_rate": self.config["training"]["learning_rate"],
                "batch_size": self.config["data"]["batch_size"],
                "epochs": epochs,
            },
            step=0,
        )

        for epoch in range(start_epoch, epochs + start_epoch):
            # Call train_epoch and validate
            print(f"Epoch {epoch+1} has started")

            train_loss, train_acc, train_f1 = self.train_epoch(train_loader, epoch)
            val_loss, val_acc, val_f1 = self.validate(val_loader, epoch)

            print(f"Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}")
