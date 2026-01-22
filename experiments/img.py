import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_curve,
)

# Path to files
results_dir = "experiments/results/"

# Laad de resultaten van het model
y_true = np.load(f"{results_dir}y_true.npy")
y_pred = np.load(f"{results_dir}y_pred.npy")
y_prob = np.load(f"{results_dir}y_prob.npy")

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(cm)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix - Champion Model")
plt.savefig(f"{results_dir}confusion_matrix.png")  # opslaan
plt.close()

# ROC curve
fpr, tpr, _ = roc_curve(y_true, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})")
plt.plot([0, 1], [0, 1], "--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Champion Model")
plt.legend()
plt.savefig(f"{results_dir}roc_curve.png")
plt.close()

# Precision-recall curve
precision, recall, _ = precision_recall_curve(y_true, y_prob)
pr_auc = auc(recall, precision)

plt.figure()
plt.plot(recall, precision, label=f"PR curve (AUC = {pr_auc:.3f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve - Champion Model")
plt.legend()
plt.savefig(f"{results_dir}pr_curve.png")
plt.close()

# Threshold analysis
thresholds = np.arange(0.0, 1.01, 0.05)
threshold_results = []

for thresh in thresholds:
    y_thresh_pred = (y_prob >= thresh).astype(int)
    acc = accuracy_score(y_true, y_thresh_pred)
    f1 = f1_score(y_true, y_thresh_pred)
    threshold_results.append((thresh, acc, f1))

threshold_results = np.array(threshold_results)
plt.figure()
plt.plot(threshold_results[:, 0], threshold_results[:, 1], label="Accuracy")
plt.plot(threshold_results[:, 0], threshold_results[:, 2], label="F1-score")
plt.xlabel("Classification Threshold")
plt.ylabel("Score")
plt.title("Threshold Analysis - Champion Model")
plt.legend()
plt.grid(True)
plt.savefig(f"{results_dir}threshold_analysis.png")
plt.close()

# Find threshold for maximizing recall
recall_scores = []
for thresh in thresholds:
    y_thresh_pred = (y_prob >= thresh).astype(int)
    recall = np.sum((y_thresh_pred == 1) & (y_true == 1)) / np.sum(y_true == 1)
    recall_scores.append(recall)

best_recall_idx = np.argmax(recall_scores)
best_thresh = thresholds[best_recall_idx]
print(f"Suggested threshold for max recall: {best_thresh:.2f}")

# Compare baseline
majority_class = np.bincount(y_true).argmax()
baseline_pred = np.full_like(y_true, majority_class)

baseline_acc = accuracy_score(y_true, baseline_pred)
baseline_f1 = f1_score(y_true, baseline_pred)

champion_acc = accuracy_score(y_true, y_pred)
champion_f1 = f1_score(y_true, y_pred)

print("=== Baseline vs Champion ===")
print(f"Baseline Accuracy : {baseline_acc:.4f}, F1-score: {baseline_f1:.4f}")
print(f"Champion Accuracy : {champion_acc:.4f}, F1-score: {champion_f1:.4f}")
