import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_data(file_path):
    """
    Step 1: Load the dataset from a CSV file.
    """
    print(f"Loading dataset from: {file_path}")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found at {file_path}")
    df = pd.read_csv(file_path)
    return df

def clean_data(df):
    """
    Step 1 (continued): Clean and inspect the dataset.
    Checks for null values, duplicates, and verifies class distributions.
    """
    print("\n--- Step 1: Data Cleaning & Inspection ---")
    print(f"Shape of dataset: {df.shape} (Players: {df.shape[0]}, Columns: {df.shape[1]})")
    
    # Check for null values
    null_counts = df.isnull().sum()
    total_nulls = null_counts.sum()
    print(f"Total null values: {total_nulls}")
    if total_nulls > 0:
        print("Null values per column:")
        print(null_counts[null_counts > 0])
    else:
        print("No null values found.")
        
    # Check for duplicates (excluding player_id if present)
    dup_cols = [col for col in df.columns if col != 'player_id']
    duplicate_count = df.duplicated(subset=dup_cols).sum()
    print(f"Duplicate records found (excluding player_id): {duplicate_count}")
    if duplicate_count > 0:
        print("Dropping duplicates...")
        df = df.drop_duplicates(subset=dup_cols).reset_index(drop=True)
        print(f"New shape after dropping duplicates: {df.shape}")
        
    # Verify class distributions
    print("\nClass distributions:")
    label_cols = ['addicted_label', 'distressed_label', 'multiclass_label']
    for col in label_cols:
        if col in df.columns:
            counts = df[col].value_counts().sort_index()
            percentages = df[col].value_counts(normalize=True).sort_index() * 100
            print(f"\nDistribution for '{col}':")
            for val, count in counts.items():
                print(f"  Class {val}: {count} players ({percentages[val]:.2f}%)")
                
    return df

def scale_features(df, feature_cols):
    """
    Step 3: Feature Scaling.
    Standardizes all numerical features to have 0 mean and unit variance.
    Returns the scaled DataFrame, a separate DataFrame of just the scaled features,
    and the trained StandardScaler object.
    """
    print("\n--- Step 3: Feature Scaling ---")
    print(f"Scaling {len(feature_cols)} numerical features using StandardScaler...")
    
    # Verify feature columns exist
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing feature columns in dataset: {missing_cols}")
        
    # Initialize and fit scaler
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[feature_cols])
    
    # Create scaled DataFrame
    df_scaled = df.copy()
    df_scaled[feature_cols] = scaled_data
    
    # Create DataFrame of only scaled features
    df_features_only = pd.DataFrame(scaled_data, columns=feature_cols)
    
    print("Scaling complete. Summary statistics (Mean / Std Dev) of first few scaled features:")
    for col in feature_cols[:3]:
        mean_val = df_scaled[col].mean()
        std_val = df_scaled[col].std()
        print(f"  {col} -> Mean: {mean_val:.4f}, Std Dev: {std_val:.4f}")
        
    return df_scaled, df_features_only, scaler


def split_dataset(X, y, train_pct=0.70, val_pct=0.15, test_pct=0.15, random_state=42):
    """
    Step 5: Stratified Train / Validation / Test Split.

    Splits data while maintaining the same class distribution in each
    partition — critical because the multiclass_label is imbalanced.

    Strategy
    --------
    1. First split:  (train+val) vs test  — keeps test_pct aside.
    2. Second split: train vs val          — divides the remainder.

    Parameters
    ----------
    X : np.ndarray  shape (N, 3, 6)
        Chunk tensor from Step 4.
    y : np.ndarray  shape (N,)
        Multiclass labels (0=Healthy, 1=Distressed, 2=Addicted, 3=Both).
    train_pct : float   default 0.70
    val_pct   : float   default 0.15
    test_pct  : float   default 0.15
    random_state : int  for reproducibility

    Returns
    -------
    X_train, X_val, X_test : np.ndarray
    y_train, y_val, y_test : np.ndarray
    """
    assert abs(train_pct + val_pct + test_pct - 1.0) < 1e-9, \
        "Train, val, and test percentages must sum to 1."

    print("\n--- Step 5: Train / Validation / Test Split ---")

    # Step A: split off the test set
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y,
        test_size=test_pct,
        stratify=y,
        random_state=random_state
    )

    # Step B: split the remaining data into train and validation
    val_relative = val_pct / (train_pct + val_pct)   # val fraction of trainval pool
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=val_relative,
        stratify=y_trainval,
        random_state=random_state
    )

    # Print split statistics
    total = len(y)
    class_names = {0: 'Healthy', 1: 'Distressed Only', 2: 'Addicted Only', 3: 'Both'}

    for split_name, X_s, y_s in [
        ('Train',      X_train, y_train),
        ('Validation', X_val,   y_val),
        ('Test',       X_test,  y_test),
    ]:
        print(f"\n  {split_name:12s}: {len(y_s):4d} players  "
              f"({100.0 * len(y_s) / total:.1f}%)   shape: {X_s.shape}")
        for cls, name in class_names.items():
            n = (y_s == cls).sum()
            print(f"    Class {cls} ({name:18s}): {n:3d}  "
                  f"({100.0 * n / len(y_s):.1f}%)")

    return X_train, X_val, X_test, y_train, y_val, y_test


# Feature groups defined in the project description
SESSION_BEHAVIOUR_COLS = [
    'avg_daily_hours',
    'sessions_per_week',
    'late_night_fraction',
    'session_escalation_slope',
    'rage_quits_per_week'
]

GAMEPLAY_METRICS_COLS = [
    'kd_ratio',
    'team_damage_incidents',
    'performance_volatility',
    'monthly_spending_usd'
]

SOCIAL_SIGNALS_COLS = [
    'solo_play_fraction',
    'chat_msgs_per_session',
    'toxic_chat_score',
    'guild_changes_per_month'
]

CLINICAL_INDICATORS_COLS = [
    'igd_score',
    'phq9_score'
]

NUMERICAL_FEATURE_COLS = SESSION_BEHAVIOUR_COLS + GAMEPLAY_METRICS_COLS + SOCIAL_SIGNALS_COLS + CLINICAL_INDICATORS_COLS
