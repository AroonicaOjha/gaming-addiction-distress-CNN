import os
import pandas as pd
from src.preprocessing import (
    load_data, 
    clean_data, 
    scale_features, 
    NUMERICAL_FEATURE_COLS
)
from src.eda import generate_eda

def main():
    print("======================================================================")
    # Project Title
    print("Designing an Active Learning Inspired CNN-BiLSTM-SVM Framework")
    print("Early Detection of Gaming Addiction & Emotional Distress")
    print("======================================================================")
    
    # Define paths
    dataset_path = os.path.join('data', 'gaming_mental_health_dataset.csv')
    scaled_output_path = os.path.join('data', 'scaled_gaming_mental_health_dataset.csv')
    results_dir = 'results'
    
    # STEP 1: Load and clean data
    try:
        df = load_data(dataset_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure you run this script from the project root (c:\\CNN).")
        return
        
    df_clean = clean_data(df)
    
    # STEP 2: Exploratory Data Analysis
    generate_eda(df_clean, results_dir)
    
    # STEP 3: Feature Scaling
    df_scaled, df_features_only, scaler = scale_features(df_clean, NUMERICAL_FEATURE_COLS)
    
    # Save the scaled dataset for subsequent steps
    print(f"\nSaving scaled dataset to: {scaled_output_path}")
    df_scaled.to_csv(scaled_output_path, index=False)
    
    print("\n======================================================================")
    print("Success: Steps 1 to 3 completed successfully!")
    print("1. Data Loading & Cleaning verified.")
    print("2. Visualizations saved to results/ folder.")
    print("3. Scaling completed and dataset saved.")
    print("======================================================================")

if __name__ == '__main__':
    main()
