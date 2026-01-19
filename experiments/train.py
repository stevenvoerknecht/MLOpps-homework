import argparse

import torch
import torch.optim as optim
from ml_core.data import get_dataloaders
from ml_core.models import MLP
from ml_core.solver import Trainer
from ml_core.utils import load_config, seed_everything, setup_logger

logger = setup_logger("Experiment_Runner")


def main(args):
    # 1. Load Config & Set Seed
    config = load_config(args.config)
    name = config["experiment_name"]
    seed = config["seed"]
    model_config = config["model"]
    training_config = config["training"]

    seed_everything(seed)
    print(f"Experiment {name} has started")
    print("Configuration has loaded and seeds have been set")

    # 2. Setup Device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 3. Data
    train_loader, val_loader = get_dataloaders(config)
    print("Dataloaders are ready")

    # 4. Model
    model = MLP(**model_config).to(device)

    # 5. Optimizer
    optimizer = optim.SGD(model.parameters(), lr=training_config["learning_rate"])

    # 6. Trainer & Fit
    trainer = Trainer(model, optimizer, config, device)
    trainer.fit(train_loader, val_loader)


if __name__ == "__main__":
    print("Run has started")
    parser = argparse.ArgumentParser(description="Train a Simple MLP on PCAM")
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    args = parser.parse_args()

    main(args)
    print("Full run completed")
