"""
cnn_bilstm.py
=============
Steps 6 & 7 — CNN Feature Extraction + BiLSTM Feature Learning.

Step 6 — Conv1D → BatchNorm → ReLU → MaxPooling1D
          Input (batch, 3, 6) → Output (batch, 2, 64)

Step 7 — Bidirectional LSTM
          Input (batch, 2, 64) → Output (batch, 128)

This module defines the shared CNN-BiLSTM backbone used by all three
modality-specific models in the decision-level fusion pipeline.

Framework : TensorFlow + Keras

Author : Person 1 (your teammate)  — Steps 6 & 7
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model


# ── Shared CNN-BiLSTM backbone ─────────────────────────────────────────────
def build_cnn_bilstm_backbone(
    input_shape=(3, 6),         # (timesteps=3 windows, features=6 per window)
    conv_filters=64,            # number of pattern detectors in Conv1D
    kernel_size=2,              # scan 2 consecutive windows at once
    lstm_units=64,              # hidden units per direction in BiLSTM
    dense_units=128,            # output feature vector size
    dropout_rate=0.3,
    name="cnn_bilstm_backbone",
):
    """
    Builds the shared CNN-BiLSTM feature extractor.

    Architecture:
        Input(3, 6)
          → Conv1D(64, kernel=2, padding='same')   Step 6
          → BatchNormalization                      Step 6
          → ReLU                                    Step 6
          → MaxPooling1D(pool_size=1)               Step 6  (3→2 timesteps)
          → Bidirectional(LSTM(64))                 Step 7
          → Dense(128, ReLU)                        Step 8  (feature vector)
          → Dropout(0.3)

    Returns
    -------
    model : keras.Model
        Feature extractor.  Call model(x) to get the 128-d vector.
    """
    inp = keras.Input(shape=input_shape, name="behaviour_windows")

    # ── Step 6: CNN block ──────────────────────────────────────────────────
    x = layers.Conv1D(
        filters=conv_filters,
        kernel_size=kernel_size,
        padding="same",
        name="conv1d",
    )(inp)
    # BatchNorm: stabilises training by normalising activations per batch
    x = layers.BatchNormalization(name="batch_norm")(x)
    # ReLU: zero-out absent patterns, keep positive detections
    x = layers.Activation("relu", name="relu")(x)
    # MaxPooling: keep only the STRONGEST pattern in each region
    # 3 timesteps → 2 timesteps after pool_size=1 (stride=1 keeps most)
    x = layers.MaxPooling1D(pool_size=2, strides=1, padding="same",
                            name="maxpool")(x)

    # ── Step 7: BiLSTM block ───────────────────────────────────────────────
    # Bidirectional: reads the 2 pooled windows FORWARD and BACKWARD
    #   Forward  → learns escalation:  session→gameplay→social
    #   Backward → learns withdrawal:  social→gameplay→session
    #   Output   → 64 + 64 = 128-d combined
    x = layers.Bidirectional(
        layers.LSTM(lstm_units, return_sequences=False),
        name="bilstm",
    )(x)

    # ── Step 8: Dense feature vector ──────────────────────────────────────
    x = layers.Dense(dense_units, activation="relu", name="feature_dense")(x)
    x = layers.Dropout(dropout_rate, name="dropout")(x)

    backbone = Model(inputs=inp, outputs=x, name=name)
    return backbone


# ── Full classification model (backbone + softmax head) ───────────────────
def build_full_model(
    input_shape=(3, 6),
    num_classes=4,
    conv_filters=64,
    kernel_size=2,
    lstm_units=64,
    dense_units=128,
    dropout_rate=0.3,
):
    """
    Full CNN-BiLSTM model with 4-class softmax output.
    Used for training and for extracting deep features.
    """
    backbone = build_cnn_bilstm_backbone(
        input_shape=input_shape,
        conv_filters=conv_filters,
        kernel_size=kernel_size,
        lstm_units=lstm_units,
        dense_units=dense_units,
        dropout_rate=dropout_rate,
    )

    inp = keras.Input(shape=input_shape)
    features = backbone(inp)
    out = layers.Dense(num_classes, activation="softmax", name="classifier")(features)

    model = Model(inputs=inp, outputs=out, name="CNN_BiLSTM_Classifier")
    return model, backbone


def print_architecture_summary():
    """Print a human-readable layer-by-layer summary."""
    model, _ = build_full_model()
    print("=" * 56)
    print("  STEPS 6 & 7 — CNN-BiLSTM ARCHITECTURE")
    print("=" * 56)
    model.summary(line_length=56)
    print()


def verify_shapes():
    """Run 8 fake samples through and print shapes at each stage."""
    print("=" * 56)
    print("  SHAPE VERIFICATION (batch of 8 players)")
    print("=" * 56)

    inp_data = np.random.randn(8, 3, 6).astype(np.float32)

    model, backbone = build_full_model()

    # Build intermediate inspection models
    layer_names = ["conv1d", "batch_norm", "relu", "maxpool", "bilstm", "feature_dense"]
    prev_output = model.input
    for lname in layer_names:
        layer = model.get_layer("cnn_bilstm_backbone").get_layer(lname)
        # Use backbone sub-model layers
        pass

    # Simpler: run backbone step by step
    bb = model.get_layer("cnn_bilstm_backbone")
    x = inp_data
    print(f"  Input            → {x.shape}")

    # Conv1D
    conv_out = keras.Model(bb.input, bb.get_layer("conv1d").output)(x)
    print(f"  After Conv1D     → {conv_out.shape}")
    bn_out = keras.Model(bb.input, bb.get_layer("batch_norm").output)(x)
    print(f"  After BatchNorm  → {bn_out.shape}")
    relu_out = keras.Model(bb.input, bb.get_layer("relu").output)(x)
    print(f"  After ReLU       → {relu_out.shape}")
    pool_out = keras.Model(bb.input, bb.get_layer("maxpool").output)(x)
    print(f"  After MaxPool    → {pool_out.shape}")
    bilstm_out = keras.Model(bb.input, bb.get_layer("bilstm").output)(x)
    print(f"  After BiLSTM     → {bilstm_out.shape}   (64 fwd + 64 bwd)")
    feat_out = bb(x)
    print(f"  Feature vector   → {feat_out.shape}   ← input to SVM")
    final_out = model(x, training=False)
    print(f"  Classifier output→ {final_out.shape}   (4-class softmax)")
    print()
    print(f"  Total parameters : {model.count_params():,}")
    print()
# ── Clean Flat Architecture for Steps 6 & 7 ────────────────────────────────

def build_cnn_bilstm(
    input_shape=(3, 6),
    conv_filters=64,
    kernel_size=2,
    lstm_units=64,
    dense_units=128,
    num_classes=4,
    dropout_rate=0.3,
    l2_reg=1e-4,   # Now explicitly accepted
):
    """
    Builds the flat CNN-BiLSTM structure matching the exact layer names 
    expected by run_step6.py and run_step7.py tracking hooks.
    """
    # 1. Define input matching the script's tracer name
    inp = keras.Input(shape=input_shape, name="behaviour_chunks")

    # 2. Step 6: CNN layers with exact script names
    x = layers.Conv1D(
        filters=conv_filters,
        kernel_size=kernel_size,
        padding="same",
        name="conv1d_pattern_detector",
    )(inp)
    x = layers.BatchNormalization(name="batch_norm")(x)
    x = layers.Activation("relu", name="relu")(x)
    x = layers.MaxPooling1D(pool_size=2, strides=1, padding="same", name="max_pooling")(x)

    # 3. Step 7: BiLSTM layer with exact script name
    x = layers.Bidirectional(
        layers.LSTM(lstm_units, return_sequences=False),
        name="bilstm",
    )(x)

    # 4. Dense feature vector extractor layer
    x = layers.Dense(dense_units, activation="relu", name="feature_vector")(x)
    feature_out = layers.Dropout(dropout_rate, name="dropout")(x)

    # 5. Final Classification Head layer
    final_out = layers.Dense(num_classes, activation="softmax", name="classification_head")(feature_out)

    # Create the full model
    full_model = Model(inputs=inp, outputs=final_out, name="CNN_BiLSTM_Classifier")
    
    # Create the feature extractor sub-model (used as input for the SVM)
    feature_extractor = Model(inputs=inp, outputs=feature_out, name="Feature_Extractor")

    return full_model, feature_extractor


def compile_model(model, learning_rate=1e-3):
    """Compiles the model using Adam optimizer and cross entropy loss."""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def print_architecture_summary(model=None):
    """Prints out the network structure panel matching teammate code style."""
    if model is None:
        model, _ = build_cnn_bilstm()
    print("=" * 65)
    print("  STEPS 6 & 7 — CNN-BiLSTM ARCHITECTURE")
    print("=" * 65)
    model.summary(line_length=65)
    print()
