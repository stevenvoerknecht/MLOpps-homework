import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve, roc_curve, auc
from pandas.plotting import parallel_coordinates

# Pad naar je resultaten
RESULTS_DIR = Path("experiments/results")
METRICS_CSV = RESULTS_DIR / "metrics.csv"
Y_TRUE_NPY = RESULTS_DIR / "y_true.npy"
Y_PRED_NPY = RESULTS_DIR / "y_pred.npy"
Y_PROB_NPY = RESULTS_DIR / "y_prob.npy"

# --- 1. Load CSV en .npy bestanden ---
metrics = pd.read_csv(METRICS_CSV)
y_true = np.load(Y_TRUE_NPY)
y_pred = np.load(Y_PRED_NPY)
y_prob = np.load(Y_PROB_NPY)

# --- 2. Confusion Matrix ---
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=[0,1])
fig, ax = plt.subplots(figsize=(5,5))
disp.plot(ax=ax)
plt.title("Confusion Matrix (Champion)")
plt.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=150)
plt.close()
print("Saved confusion matrix")

# --- 3. PR Curve ---
precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
pr_auc = auc(recall, precision)
plt.figure(figsize=(6,5))
plt.plot(recall, precision, label=f"PR AUC={pr_auc:.3f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.grid(True)
plt.savefig(RESULTS_DIR / "pr_curve.png", dpi=150)
plt.close()
print("Saved PR curve")

# --- 4. ROC Curve ---
fpr, tpr, roc_thresholds = roc_curve(y_true, y_prob)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f"ROC AUC={roc_auc:.3f}")
plt.plot([0,1], [0,1], "k--", alpha=0.5)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid(True)
plt.savefig(RESULTS_DIR / "roc_curve.png", dpi=150)
plt.close()
print("Saved ROC curve")

# --- 5. Parallel Coordinates Plot (Hyperparameters vs Metrics) ---
# Zorg dat je CSV kolommen bevat: epochs, learning_rate, batch_size, val_f1, val_roc_auc, etc.
# Kies de metrics die je wilt vergelijken
metrics_plot = metrics.copy()
# Voeg eventueel een 'run' kolom voor labeling
metrics_plot["run"] = metrics_plot.index.astype(str)
columns_to_plot = ["learning_rate", "batch_size", "val_f1", "val_roc_auc", "val_pr_auc", "run"]

plt.figure(figsize=(10,6))
parallel_coordinates(metrics_plot[columns_to_plot], class_column="run", colormap=plt.get_cmap("tab10"))
plt.title("Parallel Coordinates Plot (Hyperparameters vs Metrics)")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "parallel_coordinates.png", dpi=150)
plt.close()
print("Saved parallel coordinates plot")

print("All visualizations saved in experiments/results/")
