import numpy as np
import os, sys
sys.path.insert(0, os.path.dirname(__file__))

# ── Load pre-split data (created by Steps 1–5) ────────────────────────────
SPLITS_DIR = "data/splits"
print("Loading split data from", SPLITS_DIR)

X_train = np.load(os.path.join(SPLITS_DIR, "X_train.npy"))
X_val   = np.load(os.path.join(SPLITS_DIR, "X_val.npy"))
X_test  = np.load(os.path.join(SPLITS_DIR, "X_test.npy"))
y_train = np.load(os.path.join(SPLITS_DIR, "y_train.npy"))
y_val   = np.load(os.path.join(SPLITS_DIR, "y_val.npy"))
y_test  = np.load(os.path.join(SPLITS_DIR, "y_test.npy"))

print(f"  X_train: {X_train.shape}  y_train: {y_train.shape}")
print(f"  X_val  : {X_val.shape}    y_val  : {y_val.shape}")
print(f"  X_test : {X_test.shape}   y_test : {y_test.shape}")
print()

# ── Step 8 + 9 + 10 + 11: Training Pipeline ───────────────────────────────
from src.train import train_pipeline

fusion_model, svm, history, backbones, feat_arrays = train_pipeline(
    X_train, y_train,
    X_val,   y_val,
    X_test,  y_test,
    epochs=50,
    batch_size=32,
    learning_rate=1e-3,
    save_dir="results",
)

X_feat_train, X_feat_val, X_feat_test = feat_arrays

# ── Step 12: Evaluation ────────────────────────────────────────────────────
from src.evaluate import evaluate_all, compare_splits
from src.svm_classifier import predict_svm

# Evaluate on all three splits
y_pred_train, y_proba_train = predict_svm(svm, X_feat_train)
y_pred_val,   y_proba_val   = predict_svm(svm, X_feat_val)
y_pred_test,  y_proba_test  = predict_svm(svm, X_feat_test)

metrics_train = evaluate_all(y_train, y_pred_train, y_proba_train, "Train")
metrics_val   = evaluate_all(y_val,   y_pred_val,   y_proba_val,   "Validation")
metrics_test  = evaluate_all(y_test,  y_pred_test,  y_proba_test,  "Test")

compare_splits(metrics_train, metrics_val, metrics_test)

# ── Step 13: Visualisations ────────────────────────────────────────────────
from src.visualize import run_all_visualisations
from src.preprocessing import ALL_FEATURES

run_all_visualisations(
    history        = history,
    cm             = metrics_test["confusion_matrix"],
    y_test         = y_test,
    y_proba        = y_proba_test,
    svm_pipeline   = svm,
    feature_names  = ALL_FEATURES,
)

# ── Step 14: Benchmark ─────────────────────────────────────────────────────
from src.benchmark import run_benchmark

benchmark_results = run_benchmark(
    X_train, y_train,
    X_val,   y_val,
    X_test,  y_test,
    cnn_bilstm_svm_metrics=metrics_test,  # pass our real results
)

# ── Final Summary ──────────────────────────────────────────────────────────
print("=" * 56)
print("  COMPLETE PIPELINE — FINAL RESULTS SUMMARY")
print("=" * 56)
print()
print(f"  {'Metric':<24} {'Test Score':>12}  {'Paper (ref)':>12}")
print("  " + "-" * 52)
m = metrics_test
ref = {
    "balanced_accuracy": "0.948 (dep.)",
    "f1_score"         : "—",
    "accuracy"         : "0.946",
    "precision"        : "—",
    "recall"           : "—",
    "roc_auc"          : "—",
}
for key, label in [
    ("balanced_accuracy", "Balanced Accuracy"),
    ("accuracy",          "Accuracy"),
    ("precision",         "Precision (macro)"),
    ("recall",            "Recall (macro)"),
    ("f1_score",          "F1-score (macro)"),
    ("roc_auc",           "ROC-AUC (macro)"),
]:
    print(f"  {label:<24} {m[key]:>12.4f}  {ref.get(key,'—'):>12}")

print()
print("  All result plots saved to results/")
print("  Model saved to results/best_fusion_model.keras")
print("  SVM  saved to results/svm_classifier.pkl")
print("=" * 56)
print()
print("  Steps 8–14 complete. ✓")

