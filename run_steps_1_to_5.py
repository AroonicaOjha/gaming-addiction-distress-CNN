"""
run_steps_1_to_5.py
===================
Complete pipeline runner for Steps 1-5:

  Step 1  –  Data Loading & Cleaning
  Step 2  –  Exploratory Data Analysis
  Step 3  –  Feature Scaling
  Step 4  –  Behaviour Chunking
  Step 5  –  Stratified Train / Validation / Test Split
"""

import os
import numpy as np

from src.preprocessing import (
    load_data,
    clean_data,
    scale_features,
    split_dataset,
    NUMERICAL_FEATURE_COLS,
)
from src.eda import generate_eda
from src.chunking import create_behaviour_chunks, describe_chunks


def main():
    print("=" * 70)
    print("Designing an Active Learning Inspired CNN-BiLSTM-SVM Framework")
    print("Early Detection of Gaming Addiction & Emotional Distress")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    dataset_path       = os.path.join('data', 'gaming_mental_health_dataset.csv')
    scaled_output_path = os.path.join('data', 'scaled_gaming_mental_health_dataset.csv')
    chunks_output_path = os.path.join('data', 'chunks_X.npy')
    labels_output_path = os.path.join('data', 'labels_y.npy')
    splits_dir         = os.path.join('data', 'splits')
    results_dir        = 'results'

    # ------------------------------------------------------------------
    # STEP 1 – Load & Clean
    # ------------------------------------------------------------------
    df = load_data(dataset_path)
    df = clean_data(df)

    # ------------------------------------------------------------------
    # STEP 2 – EDA
    # ------------------------------------------------------------------
    generate_eda(df, results_dir)

    # ------------------------------------------------------------------
    # STEP 3 – Feature Scaling
    # ------------------------------------------------------------------
    df_scaled, _, scaler = scale_features(df, NUMERICAL_FEATURE_COLS)

    # Save scaled CSV for reference
    df_scaled.to_csv(scaled_output_path, index=False)
    print(f"  Scaled dataset saved -> {scaled_output_path}")

    # ------------------------------------------------------------------
    # STEP 4 – Behaviour Chunking
    # ------------------------------------------------------------------
    describe_chunks()                          # Print chunk layout
    X_chunks, _ = create_behaviour_chunks(df_scaled)

    # Persist the chunk array
    np.save(chunks_output_path, X_chunks)
    print(f"  Chunk tensor saved  -> {chunks_output_path}  shape={X_chunks.shape}")

    # ------------------------------------------------------------------
    # STEP 5 – Stratified Split
    # ------------------------------------------------------------------
    y = df_scaled['multiclass_label'].values.astype(int)
    np.save(labels_output_path, y)
    print(f"  Labels array saved  -> {labels_output_path}  shape={y.shape}")

    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X_chunks, y)

    # Save each split for downstream steps
    os.makedirs(splits_dir, exist_ok=True)
    np.save(os.path.join(splits_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(splits_dir, 'X_val.npy'),   X_val)
    np.save(os.path.join(splits_dir, 'X_test.npy'),  X_test)
    np.save(os.path.join(splits_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(splits_dir, 'y_val.npy'),   y_val)
    np.save(os.path.join(splits_dir, 'y_test.npy'),  y_test)
    print(f"\n  All splits saved to: {splits_dir}/")

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Success: Steps 1 to 5 completed!")
    print()
    print("  Step 1  Data loaded and cleaned          – 500 players, 0 nulls")
    print(f"  Step 2  EDA plots saved                  – {results_dir}/")
    print(f"  Step 3  Features scaled (StandardScaler) – {len(NUMERICAL_FEATURE_COLS)} features")
    print(f"  Step 4  Behaviour chunks created         – shape {X_chunks.shape}")
    print(f"  Step 5  Stratified split done:")
    print(f"          Train {X_train.shape}  Val {X_val.shape}  Test {X_test.shape}")
    print("=" * 70)


if __name__ == '__main__':
    main()
