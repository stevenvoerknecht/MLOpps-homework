from itertools import product

import mlflow
import torch
import torch.optim as optim
from ml_core.data import get_dataloaders
from ml_core.models import MLP
from ml_core.solver.trainer import Trainer
from ml_core.utils import load_config

# Config & device
config = load_config("experiments/configs/train_config.yaml")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Data
train_loader, val_loader = get_dataloaders(config)

# Hyperparameter search space
learning_rates = [0.001, 0.0005]
batch_sizes = [16, 32]
hidden_units_list = [[64, 32], [128, 64]]

combinations = list(product(learning_rates, batch_sizes, hidden_units_list))
print(f"Totale combinaties: {len(combinations)}")

# Grid search
for lr, batch_size, hidden_units in combinations:
    print(
        f"Running combination: lr={lr}, batch_size={batch_size}, hidden_units={hidden_units}"
    )

    # Update config
    config["training"]["learning_rate"] = lr
    config["data"]["batch_size"] = batch_size
    config["model"]["hidden_units"] = hidden_units

    if mlflow.active_run() is not None:
        mlflow.end_run()

    # Model + optimizer
    model = MLP(**config["model"]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Trainer
    trainer = Trainer(model, optimizer, config, device)

    # Train & validate
    trainer.fit(train_loader, val_loader)

    print("Combination finished.\n")
