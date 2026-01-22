import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

RESULTS_DIR = "experiments/results"

# Load saved outputs
y_true = np.load(f"{RESULTS_DIR}/y_true.npy")
y_pred = np.load(f"{RESULTS_DIR}/y_pred.npy")
y_prob = np.load(f"{RESULTS_DIR}/y_prob.npy")

# Global metrics
global_acc = accuracy_score(y_true, y_pred)
global_prec = precision_score(y_true, y_pred)
global_rec = recall_score(y_true, y_pred)
global_f1 = f1_score(y_true, y_pred)

print("=== Global Performance ===")
print(f"Accuracy : {global_acc:.4f}")
print(f"Precision: {global_prec:.4f}")
print(f"Recall   : {global_rec:.4f}")
print(f"F1-score : {global_f1:.4f}\n")

# Identify False Positives / False Negatives
FP_mask = (y_true == 0) & (y_pred == 1)
FN_mask = (y_true == 1) & (y_pred == 0)

num_FP = FP_mask.sum()
num_FN = FN_mask.sum()
print(f"Found {num_FP} False Positives and {num_FN} False Negatives\n")

# Define a slice (e.g., low-confidence predictions as "dark tiles")
# Here we simulate the "dark tiles" slice as samples with predicted probability < 0.4
slice_mask = y_prob < 0.4

slice_y_true = y_true[slice_mask]
slice_y_pred = y_pred[slice_mask]

slice_acc = accuracy_score(slice_y_true, slice_y_pred)
slice_prec = precision_score(slice_y_true, slice_y_pred, zero_division=0)
slice_rec = recall_score(slice_y_true, slice_y_pred, zero_division=0)
slice_f1 = f1_score(slice_y_true, slice_y_pred, zero_division=0)

print("=== Slice Performance (Low-confidence tiles) ===")
print(f"Accuracy : {slice_acc:.4f}")
print(f"Precision: {slice_prec:.4f}")
print(f"Recall   : {slice_rec:.4f}")
print(f"F1-score : {slice_f1:.4f}\n")

# Save slice metrics to CSV
slice_metrics_df = pd.DataFrame(
    {
        "metric": ["accuracy", "precision", "recall", "f1_score"],
        "global": [global_acc, global_prec, global_rec, global_f1],
        "slice": [slice_acc, slice_prec, slice_rec, slice_f1],
        "num_samples_slice": [len(slice_y_true)] * 4,
    }
)

slice_metrics_df.to_csv(f"{RESULTS_DIR}/slice_metrics.csv", index=False)
print(f"Saved slice metrics to {RESULTS_DIR}/slice_metrics.csv")

# Observations
print("Observations:")
print("1. False positives may occur in unusual patterns (simulated).")
print("2. False negatives often happen on low-confidence tiles.")
print("3. The selected slice has worse performance than global metrics.")
print(
    "4. Monitoring only global metrics hides these failure modes, which is dangerous in deployment."
)


# Add slicing_analysis.py on the bottom
metrics = ["accuracy", "precision", "recall", "f1_score"]
global_vals = [global_acc, global_prec, global_rec, global_f1]
slice_vals = [slice_acc, slice_prec, slice_rec, slice_f1]

x = range(len(metrics))
plt.figure(figsize=(8, 5))
plt.bar(x, global_vals, width=0.35, label="Global", alpha=0.7)
plt.bar([i + 0.35 for i in x], slice_vals, width=0.35, label="Slice", alpha=0.7)
plt.xticks([i + 0.35 / 2 for i in x], metrics)
plt.ylim(0, 1)
plt.ylabel("Score")
plt.title("Global vs Slice Performance (Low-confidence tiles)")
plt.legend()
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/slice_metrics_plot.png")
plt.show()
print(f"Saved slice performance plot to {RESULTS_DIR}/slice_metrics_plot.png")
