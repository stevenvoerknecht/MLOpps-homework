from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Config
RESULTS_DIR = Path("experiments/results")
CSV_FILE = RESULTS_DIR / "metrics.csv"


def main():
    # 1. Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 2. Load previously saved results
    y_true_path = RESULTS_DIR / "y_true.npy"
    y_pred_path = RESULTS_DIR / "y_pred.npy"
    y_prob_path = RESULTS_DIR / "y_prob.npy"

    if not (y_true_path.exists() and y_pred_path.exists() and y_prob_path.exists()):
        print("Error: one of the .npy result files does not exist.")
        return

    y_true = np.load(y_true_path)
    y_pred = np.load(y_pred_path)
    # y_prob = np.load(y_prob_path)

    # 3. Compute metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print("\n=== TEST SET RESULTS (CHAMPION) ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")

    # 4. Save to CSV
    df = pd.DataFrame(
        {"accuracy": [acc], "precision": [prec], "recall": [rec], "f1_score": [f1]}
    )

    df.to_csv(CSV_FILE, index=False)
    print(f"\n Metrics saved to {CSV_FILE}")


if __name__ == "__main__":
    main()
