
# Multimodal Machine Learning Framework for Early Detection of Gaming Addiction & Emotional Distress

An active-learning-inspired fusion framework using an end-to-end **CNN-BiLSTM-SVM** pipeline to classify gaming behavioural arcs into distinct psychological states.

---

## Pipeline Model Performance
The pipeline extracts temporal feature representations across multiple behavioural windows and classifies player profiles into four states (*Healthy*, *Distressed Only*, *Addicted Only*, or *Both*).

### Benchmark Matrix (Test Set Evaluation)

| Model Architecture | Balanced Accuracy | F1-Score | Global Accuracy | Training Time |
| :--- | :---: | :---: | :---: | :---: |
| Random Forest | 0.9006 | 0.8818 | 0.8933 | 0.6s |
| Logistic Regression | 0.8512 | 0.8368 | 0.8533 | 0.0s |
| Standalone SVM | 0.9215 | 0.8996 | 0.9067 | 0.0s |
| **CNN-BiLSTM-SVM (Ours)** | **0.8798** | **0.8656** | **0.8800** | **[Full TF Pipeline]** |

### Evaluation Metrics
* **Macro ROC-AUC**: `0.9646`
* **Macro Precision**: `0.8578`
* **Macro Recall**: `0.8798`

---

# Structure 

```text
├── data/                            # Local datasets and model-ready matrix splits
├── src/                             # Shared processing & feature engineering modules
│   ├── preprocessing.py             # Data loading and standard scaling pipelines
│   ├── eda.py                       # Exploratory analysis visualization hooks
│   ├── chunking.py                  # Sliding-window behavioural extraction
│   └── cnn_bilstm.py                # Core flat CNN-BiLSTM feature extractor blocks
├── run_steps_1_to_5.py              # ETL, Scaling, & Stratification
├── run_step6.py                     # Structural Layer & Dimension Verification
├── run_step7.py                     # Temporal BiLSTM Feature Activation Heatmaps
├── run_steps_8_to_14.py             #  Network Fusion, Classifier Training & Benchmarks
├── requirements.txt                 # Exact environment software dependencies
└── .gitignore                       


# Division
Steps 1–7: Handled raw data extraction, cleaning operations, data visualization, behavioural sequence chunking, and baseline tensor structural validations.
Steps 8–14: Implemented the primary deep-learning execution framework, extracted temporal state embeddings, constructed the dense classification framework, engineered the standalone SVM ensemble blocks, and compiled the competitive model benchmarks.




