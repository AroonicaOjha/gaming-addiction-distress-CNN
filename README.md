
# Advanced Behavioural Machine Learning Framework for Early Detection of Gaming Distress

An active-learning-inspired fusion framework using an end-to-end **CNN-BiLSTM-SVM** pipeline to classify gaming behavioural data into distinct psychological states.
The system categorizes user states into four classes: *Healthy*, *Distressed Only*, *Addicted Only*, or *Both*.

### Performance Metrics

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





