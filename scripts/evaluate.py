import torch
from pathlib import Path
from sklearn.metrics import f1_score, fbeta_score, roc_auc_score, precision_recall_curve, auc

from ml_core.models import MLP
from ml_core.data import get_dataloaders
from ml_core.utils import load_config
from ml_core.tracking.mlflow_tracker import MLflowTracker

# === Config & device (EXPLICIET CPU) ===
config = load_config("experiments/configs/train_config.yaml")
device = "cpu"  # thin_course node

# === Dataloaders ===
_, test_loader = get_dataloaders(config)  # geen return_test argument

# === Model + checkpoint ===
model = MLP(**config["model"]).to(device)
checkpoint_path = Path("experiments/results/best_checkpoint.pt")  # zoals opgegeven in config
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint["model_state"])
model.eval()

# === Evaluatie ===
all_preds, all_probs, all_labels = [], [], []

with torch.no_grad():
    for imgs, labels in test_loader:
        imgs = imgs.to(device)
        outputs = model(imgs)
        probs = torch.softmax(outputs, dim=1)[:, 1]
        preds = torch.argmax(outputs, dim=1)

        all_preds.append(preds.cpu())
        all_probs.append(probs.cpu())
        all_labels.append(labels)

all_preds = torch.cat(all_preds)
all_probs = torch.cat(all_probs)
all_labels = torch.cat(all_labels)

# === Metrics ===
f1 = f1_score(all_labels, all_preds)
f2 = fbeta_score(all_labels, all_preds, beta=2)
roc_auc = roc_auc_score(all_labels, all_probs)
precision, recall, _ = precision_recall_curve(all_labels, all_probs)
pr_auc = auc(recall, precision)


# Init tracker met dezelfde config
tracker = MLflowTracker(config)

# Log de test metrics
tracker.log_metrics(
    {
        "test_f1": f1,
        "test_f2": f2,
        "test_roc_auc": roc_auc,
        "test_pr_auc": pr_auc
    }
)

print("Test set evaluation")
print(f"F1-score : {f1:.4f}")
print(f"F2-score : {f2:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")
print(f"PR-AUC   : {pr_auc:.4f}")
