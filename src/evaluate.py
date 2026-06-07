"""
evaluate.py
===========
Step 12 — Evaluation.

Computes all metrics for the test set:
  • Accuracy
  • Balanced Accuracy   ← primary metric (handles imbalance, matches paper)
  • Precision           ← macro-averaged
  • Recall              ← macro-averaged
  • F1-score            ← macro-averaged
  • ROC-AUC             ← macro OvR

Also generates:
  • Confusion Matrix
  • Full Classification Report

Why Balanced Accuracy is the PRIMARY metric:
  Dataset is imbalanced (43% Healthy, 16% Both).
  A dumb model that always says "Healthy" gets 43% accuracy
  but 25% balanced accuracy — correctly exposed as useless.
  The paper uses Balanced Accuracy as its primary metric throughout.

Author : YOU (Person 2) — Step 12
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

CLASS_NAMES = ["Healthy", "Distressed Only", "Addicted Only", "Both"]


def evaluate_all(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    split_name: str = "Test",
) -> dict:
    """
    Compute and print all evaluation metrics.

    Parameters
    ----------
    y_true   : (N,)    true class labels 0..3
    y_pred   : (N,)    predicted class labels 0..3
    y_proba  : (N, 4)  predicted probabilities (from SVM)
    split_name: label for printing

    Returns
    -------
    metrics : dict of all computed values
    """
    acc  = accuracy_score(y_true, y_pred)
    ba   = balanced_accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec  = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1   = f1_score(y_true, y_pred, average="macro", zero_division=0)

    # ROC-AUC: one-vs-rest, macro-averaged across 4 classes
    try:
        auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
    except ValueError:
        auc = float("nan")

    cm     = confusion_matrix(y_true, y_pred)
    report = classification_report(
        y_true, y_pred, target_names=CLASS_NAMES, zero_division=0
    )

    print("=" * 56)
    print(f"  STEP 12 — EVALUATION  [{split_name} Set]")
    print("=" * 56)
    print(f"  Accuracy          : {acc:.4f}")
    print(f"  Balanced Accuracy : {ba:.4f}  ← primary metric")
    print(f"  Precision (macro) : {prec:.4f}")
    print(f"  Recall    (macro) : {rec:.4f}")
    print(f"  F1-score  (macro) : {f1:.4f}")
    print(f"  ROC-AUC   (macro) : {auc:.4f}")
    print()
    print("  Classification Report:")
    print(report)

    return {
        "accuracy": acc,
        "balanced_accuracy": ba,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "roc_auc": auc,
        "confusion_matrix": cm,
    }


def compare_splits(metrics_train, metrics_val, metrics_test):
    """Print a side-by-side comparison table of all three splits."""
    print("=" * 56)
    print("  METRICS COMPARISON — Train / Val / Test")
    print("=" * 56)
    print(f"  {'Metric':<22} {'Train':>8} {'Val':>8} {'Test':>8}")
    print("  " + "-" * 48)
    keys = ["accuracy", "balanced_accuracy", "precision", "recall", "f1_score", "roc_auc"]
    labels = ["Accuracy", "Balanced Acc.", "Precision", "Recall", "F1-score", "ROC-AUC"]
    for key, label in zip(keys, labels):
        print(f"  {label:<22} "
              f"{metrics_train[key]:>8.4f} "
              f"{metrics_val[key]:>8.4f} "
              f"{metrics_test[key]:>8.4f}")
    print()
