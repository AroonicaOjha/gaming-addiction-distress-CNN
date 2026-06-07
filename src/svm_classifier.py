"""
svm_classifier.py
=================
Step 10 — SVM Classifier.

Replaces the MLP softmax head with an SVM, exactly as described in
the paper (Section 4.3):

  "We incorporated an SVM into our workflow by using deep learning
   models such as CNNs or BiLSTMs for feature extraction. For
   inference, the model was modified to output the features produced
   either by the convolutional layer of the CNN or the last hidden
   layer of the BiLSTM."

Here:
  CNN-BiLSTM  → 384-d fused feature vector  (feature extractor)
  SVM         → 4-class prediction           (final classifier)

Why SVM beats MLP here:
  • The 384-d feature space from CNN-BiLSTM is HIGH-DIMENSIONAL
  • SVMs find the maximum-margin hyperplane — optimal for this
  • SVMs are robust to overfitting on small datasets (500 players)
  • Paper showed +4 BA improvement for multi-class with SVM

Author : YOU (Person 2) — Step 10
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score
import joblib
import os


def build_svm(kernel="rbf", C=10.0, gamma="scale", random_state=42):
    """
    Build SVM with RBF kernel.

    Parameters
    ----------
    kernel       : 'rbf' is best for high-dimensional CNN features
    C            : regularisation — higher = tighter margin = less error
    gamma        : 'scale' = 1/(n_features × var(X))  (auto-tuned)
    class_weight : 'balanced' = automatically handles class imbalance

    Returns
    -------
    svm : sklearn SVC with StandardScaler pipeline
    """
    svm_pipeline = Pipeline([
        # Scale the 384-d features again before SVM
        # (CNN features are not necessarily zero-mean)
        ("scaler", StandardScaler()),
        ("svm", SVC(
            kernel=kernel,
            C=C,
            gamma=gamma,
            class_weight="balanced",   # handles imbalanced classes
            probability=True,          # enables predict_proba for AUC
            random_state=random_state,
            decision_function_shape="ovr",  # one-vs-rest for 4 classes
        )),
    ])
    return svm_pipeline


def train_svm(svm_pipeline, X_features_train, y_train):
    """
    Fit SVM on extracted CNN-BiLSTM features.

    Parameters
    ----------
    X_features_train : (N_train, 384)  — fused deep features
    y_train          : (N_train,)       — class labels 0..3
    """
    print("=" * 56)
    print("  STEP 10 — SVM CLASSIFIER TRAINING")
    print("=" * 56)
    print(f"  Input features shape : {X_features_train.shape}")
    print(f"  Classes              : {np.unique(y_train)}")
    print(f"  Kernel               : RBF")
    print(f"  Class weighting      : balanced (handles imbalance)")
    print()

    svm_pipeline.fit(X_features_train, y_train)

    # Quick training-set performance check
    y_pred_train = svm_pipeline.predict(X_features_train)
    train_ba = balanced_accuracy_score(y_train, y_pred_train)
    print(f"  Train Balanced Accuracy : {train_ba:.4f}")
    print()
    return svm_pipeline


def predict_svm(svm_pipeline, X_features):
    """Return class predictions and probabilities."""
    y_pred  = svm_pipeline.predict(X_features)
    y_proba = svm_pipeline.predict_proba(X_features)
    return y_pred, y_proba


def save_svm(svm_pipeline, path="results/svm_classifier.pkl"):
    """Save fitted SVM to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(svm_pipeline, path)
    print(f"  SVM saved : {path}")


def load_svm(path="results/svm_classifier.pkl"):
    """Load fitted SVM from disk."""
    return joblib.load(path)
