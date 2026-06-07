"""
train.py
========
Step 11 — Training Pipeline.

Trains the CNN-BiLSTM fusion model, then extracts deep features
and trains the SVM classifier on top.

Training strategy:
  • Adam optimiser         — adaptive learning rate
  • Categorical Cross-Entropy — standard multi-class loss
  • Class weights          — handles 42% Healthy vs 16% Both imbalance
  • Cosine LR scheduler    — smoothly decays learning rate
  • Early stopping         — stops when val loss stops improving

Monitored metric: Balanced Accuracy (handles class imbalance,
same as the paper).

Author : YOU (Person 2) — Step 11
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import optimizers, losses, callbacks
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import balanced_accuracy_score
import os

from src.fusion import build_fusion_training_model
from src.svm_classifier import build_svm, train_svm


# ── Custom Balanced Accuracy Keras callback ────────────────────────────────
class BalancedAccuracyCallback(callbacks.Callback):
    """
    Computes balanced accuracy on the validation set at each epoch.
    Logged as 'val_balanced_acc' in history.
    """
    def __init__(self, X_val, y_val):
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val

    def on_epoch_end(self, epoch, logs=None):
        y_pred = np.argmax(self.model.predict(self.X_val, verbose=0), axis=1)
        ba = balanced_accuracy_score(self.y_val, y_pred)
        logs["val_balanced_acc"] = ba
        if (epoch + 1) % 5 == 0:
            print(f"    → Epoch {epoch+1:3d} | "
                  f"Loss: {logs['loss']:.4f} | "
                  f"Val Loss: {logs['val_loss']:.4f} | "
                  f"Val BA: {ba:.4f}")


# ── Compute class weights ──────────────────────────────────────────────────
def get_class_weights(y_train: np.ndarray) -> dict:
    """
    Compute class weights inversely proportional to class frequency.
    Ensures "Both" (only 16%) gets as much attention as "Healthy" (43%).
    """
    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    weight_dict = {c: w for c, w in zip(classes, weights)}
    print("  Class weights (for imbalance):")
    names = ["Healthy", "Distressed", "Addicted", "Both"]
    for c, w in weight_dict.items():
        print(f"    Class {c} ({names[c]:<12}): {w:.3f}")
    return weight_dict


# ── Cosine annealing LR schedule ──────────────────────────────────────────
def cosine_lr_schedule(initial_lr=1e-3, epochs=50):
    """
    Cosine annealing: LR starts at initial_lr, smoothly decays to near 0.
    Prevents the model from getting stuck in poor local minima.
    """
    def scheduler(epoch, lr):
        return initial_lr * 0.5 * (1 + np.cos(np.pi * epoch / epochs))
    return callbacks.LearningRateScheduler(scheduler, verbose=0)


# ── Main training function ─────────────────────────────────────────────────
def train_pipeline(
    X_train, y_train,
    X_val,   y_val,
    X_test,  y_test,
    epochs=50,
    batch_size=32,
    learning_rate=1e-3,
    save_dir="results",
):
    """
    Full training pipeline: CNN-BiLSTM → feature extraction → SVM.

    Parameters
    ----------
    X_train/val/test : (N, 3, 6)  chunk tensors
    y_train/val/test : (N,)        integer class labels 0..3

    Returns
    -------
    fusion_model    : trained keras Model
    svm_classifier  : trained sklearn SVM
    history         : keras training history
    backbones       : tuple of 3 trained backbone models
    """
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 56)
    print("  STEP 11 — TRAINING PIPELINE")
    print("=" * 56)
    print(f"  Train : {len(y_train)} players")
    print(f"  Val   : {len(y_val)} players")
    print(f"  Batch : {batch_size} | Epochs: {epochs} | LR: {learning_rate}")
    print()

    # ── 1. Class weights ───────────────────────────────────────────────────
    class_weights = get_class_weights(y_train)
    print()

    # ── 2. Build model ─────────────────────────────────────────────────────
    fusion_model, backbones = build_fusion_training_model(num_classes=4)

    # ── 3. Compile ─────────────────────────────────────────────────────────
    fusion_model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss=losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    # ── 4. Callbacks ───────────────────────────────────────────────────────
    cb_list = [
        BalancedAccuracyCallback(X_val, y_val),
        cosine_lr_schedule(learning_rate, epochs),
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ModelCheckpoint(
            filepath=os.path.join(save_dir, "best_fusion_model.keras"),
            monitor="val_loss",
            save_best_only=True,
            verbose=0,
        ),
    ]

    # ── 5. Train CNN-BiLSTM ────────────────────────────────────────────────
    print("  Training CNN-BiLSTM fusion model...")
    print()
    history = fusion_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weights,
        callbacks=cb_list,
        verbose=0,           # our callback handles printing
    )
    print()
    print(f"  Training complete. Best val loss: "
          f"{min(history.history['val_loss']):.4f}")
    print()

    # ── 6. Extract deep features using trained backbones ───────────────────
    print("  Extracting 384-d fused features for SVM...")
    bb_s, bb_g, bb_c = backbones

    def extract(X):
        fs = bb_s (X[:, 0:1, :], training=False).numpy()
        fg = bb_g (X[:, 1:2, :], training=False).numpy()
        fc = bb_c (X[:, 2:3, :], training=False).numpy()
        return np.hstack([fs, fg, fc]).astype(np.float32)

    X_feat_train = extract(X_train)
    X_feat_val   = extract(X_val)
    X_feat_test  = extract(X_test)
    print(f"  Feature shape : {X_feat_train.shape}  (N × 384)")
    print()

    # ── 7. Train SVM on extracted features ────────────────────────────────
    svm = build_svm(kernel="rbf", C=10.0)
    svm = train_svm(svm, X_feat_train, y_train)

    return fusion_model, svm, history, backbones, \
           (X_feat_train, X_feat_val, X_feat_test)
