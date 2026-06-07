# ============================================================
# cnn_bilstm.py
# Step 6 & 7: CNN Feature Extraction + BiLSTM Feature Learning
#
# Architecture:
#   Input (batch, 3, 6)
#   --> Conv1D (pattern detector)
#   --> BatchNormalization
#   --> ReLU Activation
#   --> MaxPooling1D
#   --> Bidirectional LSTM (temporal arc learner)
#   --> Dense (feature vector for SVM)
# ============================================================

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.regularizers import l2


# ============================================================
# WHY EACH LAYER EXISTS
# ============================================================
#
# Conv1D  – slides a kernel of width=2 across the 3 windows
#           (Session → Gameplay → Social). Each filter learns
#           one co-occurring pattern, e.g.:
#               filter_1: high hours + high escalation slope
#               filter_2: high toxicity + solo play
#               filter_3: high spending + rage-quit spikes
#
# BatchNormalization – normalises each mini-batch so training
#           is stable and converges faster. Reduces sensitivity
#           to random weight initialisation.
#
# ReLU    – non-linearity: zero-out negative activations so the
#           network only "fires" when a pattern is actually found.
#
# MaxPooling1D – keeps only the strongest activation from the
#           sliding window; makes the representation invariant
#           to minor positional shifts (e.g. a pattern at window
#           1 vs window 2 is treated the same).
#
# BiLSTM  – reads the pooled feature sequence both forward and
#           backward. Captures the behavioural ARC:
#               "sessions escalating → rage-quits increasing
#                → toxicity spiking"
#           and also the reverse direction to detect
#           withdrawal / disengagement patterns.
#
# Dense   – compresses the BiLSTM hidden state into a fixed-size
#           feature vector (default 128-d) that SVM will classify.
# ============================================================


def build_cnn_bilstm(
    input_shape=(3, 6),       # (timesteps=3 windows, features=6 per window)
    conv_filters=64,           # number of CNN filter kernels
    kernel_size=2,             # kernel slides across 2 consecutive windows
    lstm_units=64,             # BiLSTM units per direction (x2 for bidirectional)
    dense_units=128,           # output feature vector size for SVM
    num_classes=4,             # 0:Healthy  1:Distressed  2:Addicted  3:Both
    dropout_rate=0.3,          # dropout after BiLSTM to prevent overfitting
    l2_reg=1e-4,               # L2 weight regularisation
):
    """
    Step 6 & 7 – Builds the full CNN-BiLSTM feature extractor.

    Parameters
    ----------
    input_shape  : tuple  (timesteps, features)  default (3, 6)
    conv_filters : int    number of Conv1D filters
    kernel_size  : int    Conv1D kernel width (windows scanned at once)
    lstm_units   : int    BiLSTM units per direction
    dense_units  : int    dimension of the deep feature vector
    num_classes  : int    output classes for the softmax head
    dropout_rate : float  dropout probability
    l2_reg       : float  L2 regularisation coefficient

    Returns
    -------
    model        : keras.Model  full CNN-BiLSTM with classification head
    feature_extractor : keras.Model  sub-model that outputs the dense vector
                        (used to feed extracted features into SVM)
    """

    # ------------------------------------------------------------------
    # Input
    # ------------------------------------------------------------------
    inputs = keras.Input(shape=input_shape, name="behaviour_chunks")
    # Shape entering the network: (batch, 3, 6)

    # ==================================================================
    # STEP 6 – CNN FEATURE EXTRACTION
    # ==================================================================

    # --- Conv1D Layer ---
    # kernel_size=2: the filter scans pairs of consecutive windows.
    # This means it can detect cross-modality patterns, e.g.:
    #   window[0]+window[1] → "Session + Gameplay co-pattern"
    #   window[1]+window[2] → "Gameplay + Social co-pattern"
    x = layers.Conv1D(
        filters=conv_filters,
        kernel_size=kernel_size,
        padding="same",           # keep output length = 3
        kernel_regularizer=l2(l2_reg),
        name="conv1d_pattern_detector"
    )(inputs)
    # Shape: (batch, 3, 64)

    # --- Batch Normalisation ---
    # Normalises each feature map across the batch.
    # Stabilises gradient flow, allows higher learning rates.
    x = layers.BatchNormalization(name="batch_norm")(x)
    # Shape: (batch, 3, 64)

    # --- ReLU Activation ---
    # Introduces non-linearity. Negative activations (absent patterns)
    # become 0 — only positive detections "pass through".
    x = layers.Activation("relu", name="relu")(x)
    # Shape: (batch, 3, 64)

    # --- MaxPooling1D ---
    # pool_size=2: compresses the 3-step sequence.
    # Keeps only the strongest detected pattern per region.
    # Makes the model robust to slight timing variations.
    x = layers.MaxPooling1D(
        pool_size=2,
        padding="same",           # keep output from vanishing (3→2)
        name="max_pooling"
    )(x)
    # Shape: (batch, 2, 64)

    # ==================================================================
    # STEP 7 – BiLSTM FEATURE LEARNING
    # ==================================================================

    # --- Bidirectional LSTM ---
    # return_sequences=False: we only want the final summary state.
    # Bidirectional:
    #   forward  pass → reads Session→Gameplay→Social progression
    #   backward pass → reads Social→Gameplay→Session regression
    # Combined output = 2 * lstm_units = 128-d vector.
    x = layers.Bidirectional(
        layers.LSTM(
            units=lstm_units,
            kernel_regularizer=l2(l2_reg),
            recurrent_regularizer=l2(l2_reg),
            name="lstm_core"
        ),
        name="bilstm"
    )(x)
    # Shape: (batch, 128)

    # --- Dropout ---
    # Randomly zeros 30% of units during training to prevent co-adaptation
    # and force the network to learn redundant, robust representations.
    x = layers.Dropout(dropout_rate, name="dropout")(x)
    # Shape: (batch, 128)

    # --- Dense Feature Vector (the deep representation) ---
    # This 128-d vector is what gets handed to the SVM classifier.
    # We name it "feature_vector" so we can extract it easily.
    feature_vector = layers.Dense(
        dense_units,
        activation="relu",
        kernel_regularizer=l2(l2_reg),
        name="feature_vector"
    )(x)
    # Shape: (batch, 128)

    # --- Softmax Output Head (used only during training) ---
    # The SVM replaces this in the final pipeline (Step 10).
    # We keep it here to enable standard cross-entropy training.
    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        name="classification_head"
    )(feature_vector)
    # Shape: (batch, 4)

    # ------------------------------------------------------------------
    # Build the two models
    # ------------------------------------------------------------------

    # Full model: Input → CNN → BiLSTM → Dense → Softmax
    model = Model(inputs=inputs, outputs=outputs, name="CNN_BiLSTM")

    # Feature extractor sub-model: stops at the dense layer.
    # Used in Step 10 to generate features for the SVM.
    feature_extractor = Model(
        inputs=inputs,
        outputs=feature_vector,
        name="CNN_BiLSTM_FeatureExtractor"
    )

    return model, feature_extractor


def compile_model(model, learning_rate=1e-3):
    """
    Compiles the CNN-BiLSTM model with:
      - Adam optimiser
      - Categorical cross-entropy loss
      - Accuracy + AUC metrics
    """
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.AUC(name="auc", multi_label=False),
        ],
    )
    return model


def print_architecture_summary(model):
    """
    Prints a clean layer-by-layer summary of the model.
    """
    print("\n" + "=" * 65)
    print("  CNN-BiLSTM Architecture Summary")
    print("=" * 65)
    model.summary(line_length=65)
    print("=" * 65)
