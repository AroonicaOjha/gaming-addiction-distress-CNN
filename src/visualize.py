"""
visualize.py
============
Step 13 — Visualisations.

Generates all result plots:
  1. training_curves.png        — loss + balanced accuracy per epoch
  2. confusion_matrix.png       — normalised + raw counts
  3. feature_importance.png     — which features drive predictions
  4. roc_curves.png             — ROC curve per class (OvR)
  5. severity_scatter.png       — predicted vs true IGD/PHQ scores

All plots saved to results/

Author : YOU (Person 2) — Step 13
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import roc_curve, auc
import os

CLASS_NAMES  = ["Healthy", "Distressed\nOnly", "Addicted\nOnly", "Both"]
CLASS_COLORS = ["#2196F3", "#4CAF50", "#FF9800", "#F44336"]
SAVE_DIR     = "results"


def _save(fig, filename):
    os.makedirs(SAVE_DIR, exist_ok=True)
    path = os.path.join(SAVE_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✓ {filename}")


# ── Plot 1: Training curves ────────────────────────────────────────────────
def plot_training_curves(history):
    """
    Plot training + validation loss and balanced accuracy over epochs.
    Shows whether the model is overfitting (train loss ↓ but val loss ↑).
    """
    hist   = history.history
    epochs = range(1, len(hist["loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("CNN-BiLSTM Training Curves", fontsize=13, fontweight="bold")

    # Loss
    ax1.plot(epochs, hist["loss"],     color="#185FA5", lw=2, label="Train Loss")
    ax1.plot(epochs, hist["val_loss"], color="#D85A30", lw=2, ls="--", label="Val Loss")
    ax1.fill_between(epochs, hist["loss"],
                     [min(hist["loss"])]*len(epochs), alpha=0.08, color="#185FA5")
    ax1.set_xlabel("Epoch", fontsize=10)
    ax1.set_ylabel("Categorical Cross-Entropy Loss", fontsize=10)
    ax1.set_title("Loss — Train vs Validation", fontsize=11)
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.25)

    # Balanced Accuracy
    if "val_balanced_acc" in hist:
        ax2.plot(epochs, hist["val_balanced_acc"],
                 color="#1D9E75", lw=2, label="Val Balanced Accuracy")
        ax2.axhline(max(hist["val_balanced_acc"]), color="#1D9E75",
                    lw=1, ls=":", alpha=0.6,
                    label=f"Best: {max(hist['val_balanced_acc']):.3f}")
    if "accuracy" in hist:
        ax2.plot(epochs, hist["accuracy"],
                 color="#185FA5", lw=2, ls="--", alpha=0.7, label="Train Accuracy")

    ax2.set_xlabel("Epoch", fontsize=10)
    ax2.set_ylabel("Score", fontsize=10)
    ax2.set_title("Balanced Accuracy — Validation", fontsize=11)
    ax2.set_ylim(0, 1.05)
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.25)

    _save(fig, "training_curves.png")


# ── Plot 2: Confusion matrix ───────────────────────────────────────────────
def plot_confusion_matrix(cm: np.ndarray, title: str = "CNN-BiLSTM-SVM"):
    """
    Two-panel: left = normalised (% per true class), right = raw counts.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"Confusion Matrix — {title}", fontsize=13, fontweight="bold")

    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                ax=ax1, cbar=True, linewidths=0.5, annot_kws={"size": 10})
    ax1.set_title("Normalised (row %)", fontsize=11)
    ax1.set_xlabel("Predicted", fontsize=10)
    ax1.set_ylabel("True", fontsize=10)
    ax1.tick_params(labelsize=8)

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                ax=ax2, cbar=False, linewidths=0.5, annot_kws={"size": 10})
    ax2.set_title("Raw Counts", fontsize=11)
    ax2.set_xlabel("Predicted", fontsize=10)
    ax2.set_ylabel("True", fontsize=10)
    ax2.tick_params(labelsize=8)

    _save(fig, "confusion_matrix.png")


# ── Plot 3: ROC curves ─────────────────────────────────────────────────────
def plot_roc_curves(y_true: np.ndarray, y_proba: np.ndarray):
    """
    One ROC curve per class (One-vs-Rest).
    AUC close to 1.0 = near-perfect discrimination.
    """
    from sklearn.preprocessing import label_binarize
    y_bin = label_binarize(y_true, classes=[0, 1, 2, 3])

    fig, ax = plt.subplots(figsize=(8, 6))
    names_flat = ["Healthy", "Distressed Only", "Addicted Only", "Both"]

    for cls, (name, color) in enumerate(zip(names_flat, CLASS_COLORS)):
        fpr, tpr, _ = roc_curve(y_bin[:, cls], y_proba[:, cls])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"{name}  (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random classifier")
    ax.fill_between([0, 1], [0, 1], alpha=0.04, color="gray")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("ROC Curves — One vs Rest (per class)", fontsize=12,
                 fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(alpha=0.25)

    _save(fig, "roc_curves.png")


# ── Plot 4: Feature importance ─────────────────────────────────────────────
def plot_feature_importance(svm_pipeline, feature_names: list):
    """
    Approximate feature importance from SVM dual coefficients.
    Uses the SVM's support vector weights to rank features.
    """
    from sklearn.inspection import permutation_importance
    # For SVM with RBF kernel, we use the absolute mean of
    # dual_coef_ × support_vectors_ as a proxy for importance
    svm = svm_pipeline.named_steps["svm"]
    scaler = svm_pipeline.named_steps["scaler"]

    if not hasattr(svm, "dual_coef_"):
        print("  [skip] SVM not fitted — skipping feature importance plot")
        return

    # proxy: L2 norm of contribution per feature across all support vectors
    importance_proxy = np.abs(svm.dual_coef_).sum(axis=0)

    # dual_coef shape: (n_classes-1, n_support_vectors)
    # support_vectors shape: (n_support_vectors, n_features=384)
    sv = svm.support_vectors_                         # (n_sv, 384)
    dc = np.abs(svm.dual_coef_).sum(axis=0)          # (n_sv,)
    feat_importance = np.abs(sv * dc[:, None]).mean(axis=0)  # (384,)

    # We have 3 modality blocks of 128 each
    # Average within each block to get per-modality importance
    block_names = ["Session (CNN)", "Gameplay (CNN)", "Social (CNN)"]
    block_size  = 128
    block_imps  = [
        feat_importance[i*block_size:(i+1)*block_size].mean()
        for i in range(3)
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Feature Importance from CNN-BiLSTM-SVM",
                 fontsize=13, fontweight="bold")

    # Modality-level importance
    colors = ["#185FA5", "#FF9800", "#1D9E75"]
    bars = ax1.bar(block_names, block_imps, color=colors, width=0.5, edgecolor="white")
    for bar, val in zip(bars, block_imps):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                 f"{val:.4f}", ha="center", va="bottom", fontsize=10)
    ax1.set_title("Modality-Level Importance\n(avg SVM feature weight)", fontsize=10)
    ax1.set_ylabel("Mean |weight|", fontsize=10)
    ax1.grid(axis="y", alpha=0.3)

    # Top-15 individual features from full importance vector
    # Map back to original feature names (each modality repeated 128 times)
    # Show the top 15 compressed by averaging per-input-feature across 128
    n_orig = len(feature_names)
    per_feat = []
    for fi in range(n_orig):
        # Each original feature contributes to all 3 modality blocks
        # where it appears; simple approach: sum across full 384 vector
        # at positions corresponding to the feature index within blocks
        feat_positions = [fi % block_size + b*block_size for b in range(3)
                          if fi < block_size]
        val = sum(feat_importance[p] for p in feat_positions
                  if p < len(feat_importance)) / max(len(feat_positions), 1)
        per_feat.append(val)

    per_feat = np.array(per_feat)
    top_idx  = np.argsort(per_feat)[::-1][:min(15, n_orig)]
    top_vals = per_feat[top_idx]
    top_labs = [feature_names[i] for i in top_idx]

    y_pos = np.arange(len(top_labs))
    ax2.barh(y_pos, top_vals, color="#185FA5", alpha=0.8, edgecolor="white")
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(top_labs, fontsize=8.5)
    ax2.invert_yaxis()
    ax2.set_xlabel("Importance Score", fontsize=10)
    ax2.set_title("Top Feature Importances\n(SVM weight proxy)", fontsize=10)
    ax2.grid(axis="x", alpha=0.3)

    _save(fig, "feature_importance.png")


# ── Plot 5: Severity scatter ───────────────────────────────────────────────
def plot_severity_scatter(y_true_igd, y_pred_igd,
                          y_true_phq=None, y_pred_phq=None):
    """
    Scatter plot of predicted vs true severity scores.
    Points close to the diagonal = accurate prediction.
    """
    from sklearn.metrics import mean_absolute_error
    ncols = 2 if y_true_phq is not None else 1
    fig, axes = plt.subplots(1, ncols, figsize=(6*ncols + 1, 5))
    if ncols == 1:
        axes = [axes]
    fig.suptitle("Severity Score: Predicted vs True",
                 fontsize=13, fontweight="bold")

    for ax, yt, yp, label, thresh, color in zip(
        axes,
        [y_true_igd, y_true_phq] if ncols==2 else [y_true_igd],
        [y_pred_igd, y_pred_phq] if ncols==2 else [y_pred_igd],
        ["IGD Score (Addiction, 0–60)", "PHQ-9 Score (Distress, 0–27)"],
        [40, 14],
        ["#185FA5", "#1D9E75"],
    ):
        if yt is None:
            continue
        mae = mean_absolute_error(yt, yp)
        sc = ax.scatter(yt, yp, alpha=0.4, s=25, c=color, edgecolors="none")
        mn, mx = min(yt.min(), yp.min()), max(yt.max(), yp.max())
        ax.plot([mn,mx],[mn,mx], "r--", lw=1.5, label="Perfect prediction")
        ax.axvline(thresh, color="gray", lw=1, ls=":", alpha=0.7,
                   label=f"Threshold ({thresh})")
        ax.set_xlabel(f"True {label}", fontsize=9)
        ax.set_ylabel(f"Predicted {label}", fontsize=9)
        ax.set_title(f"{label}\nMAE = {mae:.2f}", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)

    _save(fig, "severity_scatter.png")


def run_all_visualisations(history, cm, y_test, y_proba,
                           svm_pipeline, feature_names,
                           y_igd_true=None, y_igd_pred=None):
    print("=" * 56)
    print("  STEP 13 — VISUALISATIONS")
    print("=" * 56)
    plot_training_curves(history)
    plot_confusion_matrix(cm)
    plot_roc_curves(y_test, y_proba)
    plot_feature_importance(svm_pipeline, feature_names)
    if y_igd_true is not None:
        plot_severity_scatter(y_igd_true, y_igd_pred)
    print()
