"""
run_step6.py
============
Builds and verifies the CNN-BiLSTM architecture (Steps 6 & 7).

Loads the pre-saved splits from Step 5, constructs the model,
prints the architecture summary, and runs one forward pass to
confirm shapes are correct end-to-end.
"""

import os
import numpy as np
import tensorflow as tf

from src.cnn_bilstm import build_cnn_bilstm, compile_model, print_architecture_summary


def main():
    print("=" * 65)
    print("  Step 6 – CNN Feature Extraction")
    print("  Step 7 – BiLSTM Feature Learning")
    print("=" * 65)

    # ------------------------------------------------------------------
    # 1. Load the splits saved by Step 5
    # ------------------------------------------------------------------
    splits_dir = os.path.join("data", "splits")
    print(f"\nLoading splits from: {splits_dir}")

    X_train = np.load(os.path.join(splits_dir, "X_train.npy"))
    X_val   = np.load(os.path.join(splits_dir, "X_val.npy"))
    X_test  = np.load(os.path.join(splits_dir, "X_test.npy"))
    y_train = np.load(os.path.join(splits_dir, "y_train.npy"))
    y_val   = np.load(os.path.join(splits_dir, "y_val.npy"))
    y_test  = np.load(os.path.join(splits_dir, "y_test.npy"))

    print(f"  X_train : {X_train.shape}  y_train : {y_train.shape}")
    print(f"  X_val   : {X_val.shape}    y_val   : {y_val.shape}")
    print(f"  X_test  : {X_test.shape}   y_test  : {y_test.shape}")

    # ------------------------------------------------------------------
    # 2. Build the CNN-BiLSTM model
    # ------------------------------------------------------------------
    input_shape = X_train.shape[1:]   # (3, 6)
    print(f"\nBuilding CNN-BiLSTM for input shape: {input_shape}")

    model, feature_extractor = build_cnn_bilstm(
        input_shape=input_shape,
        conv_filters=64,
        kernel_size=2,
        lstm_units=64,
        dense_units=128,
        num_classes=4,
        dropout_rate=0.3,
        l2_reg=1e-4,
    )

    model = compile_model(model, learning_rate=1e-3)

    # ------------------------------------------------------------------
    # 3. Print architecture summaries
    # ------------------------------------------------------------------
    print_architecture_summary(model)

    print("\n  Feature Extractor sub-model (used for SVM input):")
    print("=" * 65)
    feature_extractor.summary(line_length=65)
    print("=" * 65)

    # ------------------------------------------------------------------
    # 4. Forward pass – verify shapes are correct
    # ------------------------------------------------------------------
    print("\nRunning forward pass on 1 batch (all training data)...")

    # Full model: outputs class probabilities (4 classes)
    sample_batch = X_train[:8]                     # 8 players
    preds = model(sample_batch, training=False)
    print(f"\n  Input  shape : {sample_batch.shape}")
    print(f"  Output shape : {preds.shape}    (batch x 4 class probabilities)")
    print(f"\n  Sample softmax output (first 3 players):")
    for i in range(3):
        row = preds[i].numpy()
        predicted = int(np.argmax(row))
        label_map = {0: "Healthy", 1: "Distressed", 2: "Addicted", 3: "Both"}
        print(f"    Player {i+1}: {[f'{v:.3f}' for v in row]}  => {label_map[predicted]}")

    # Feature extractor: outputs 128-d vectors for SVM
    features = feature_extractor(sample_batch, training=False)
    print(f"\n  Feature vector shape : {features.shape}")
    print(f"  (This 128-d vector is what the SVM will receive as input)")

    # ------------------------------------------------------------------
    # 5. Layer-by-layer shape trace
    # ------------------------------------------------------------------
    print("\n  Layer-by-layer shape trace:")
    print("  " + "-" * 50)
    layer_names = [
        "behaviour_chunks",
        "conv1d_pattern_detector",
        "batch_norm",
        "relu",
        "max_pooling",
        "bilstm",
        "dropout",
        "feature_vector",
        "classification_head",
    ]
    trace_model = tf.keras.Model(
        inputs=model.input,
        outputs=[model.get_layer(n).output for n in layer_names]
    )
    trace_outputs = trace_model(sample_batch, training=False)
    for name, out in zip(layer_names, trace_outputs):
        print(f"  {name:<30} -> {out.shape}")

    # ------------------------------------------------------------------
    # 6. Save model and feature extractor
    # ------------------------------------------------------------------
    os.makedirs("results", exist_ok=True)
    model_save_path   = os.path.join("results", "cnn_bilstm_init.keras")
    feature_save_path = os.path.join("results", "feature_extractor_init.keras")

    model.save(model_save_path)
    feature_extractor.save(feature_save_path)
    print(f"\n  Model saved           -> {model_save_path}")
    print(f"  Feature extractor saved -> {feature_save_path}")

    print("\n" + "=" * 65)
    print("  Step 6 & 7 verification complete!")
    print("  CNN-BiLSTM architecture is ready for training (Step 11).")
    print("=" * 65)


if __name__ == "__main__":
    main()
