# experiments/inference.py

import argparse
from pathlib import Path

import h5py
import torch
import torch.optim as optim
from ml_core.models import MLP
from ml_core.utils import load_checkpoint, load_config, seed_everything
from torchvision import transforms


def load_sample(h5_x_path: Path, index: int = 0):
    """Load a single PCAM sample from an H5 file."""
    with h5py.File(h5_x_path, "r") as f:
        x = f["x"][index]

    # Convert uint8 → float tensor
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    return transform(x).unsqueeze(0)


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load config & checkpoint
    config = load_config(args.config)
    seed_everything(config["seed"])

    model = MLP(**config["model"]).to(device)
    optimizer = optim.SGD(model.parameters(), lr=config["training"]["learning_rate"])

    start_epoch, checkpoint = load_checkpoint(args.checkpoint, model, optimizer, device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    # Load one sample
    x = load_sample(Path(args.h5_x), args.index).to(device)

    # Inference
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()

    print("Prediction:")
    print(f"  Class: {pred}")
    print(f"  Probabilities: {probs.squeeze().cpu().numpy()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PCAM inference from checkpoint")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--h5_x", type=str, required=True)
    parser.add_argument("--index", type=int, default=0)

    args = parser.parse_args()
    main(args)
