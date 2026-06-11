import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from sklearn.preprocessing import label_binarize


df = pd.read_csv('gaming_mental_health_dataset.csv')

columns_to_drop = ['player_id', 'addicted_label', 'distressed_label', 'multiclass_label']
existing_drops = [col for col in columns_to_drop if col in df.columns]

X = df.drop(columns=existing_drops)
y = df['multiclass_label']
class_names = ['Healthy', 'Distressed Only', 'Addicted Only', 'Both']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Training
model = RandomForestClassifier(random_state=42, n_estimators=100)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_score = model.predict_proba(X_test)
y_test_bin = label_binarize(y_test, classes=[0, 1, 2, 3])


print("\n" + "═"*60)
print("  CRITICAL ACADEMIC BENCHMARK RESULTS ")
print("═"*60)

# 1. CONFUSION MATRIX TEXT GRID
print("\n[1] CONFUSION MATRIX GRID:")
print(f"Columns (Predicted) -> | {class_names[0]} | {class_names[1]} | {class_names[2]} | {class_names[3]} |")
cm = confusion_matrix(y_test, y_pred)
for idx, row in enumerate(cm):
    print(f"Actual: {class_names[idx]:<15} -> {row}")

# 2. ROC-AUC SCORES PER CLASS
print("\n" + "-"*40)
print("[2] MULTI-CLASS ROC-AUC SCORES:")
print("-"*40)
for i in range(4):
    score = roc_auc_score(y_test_bin[:, i], y_score[:, i])
    print(f"• {class_names[i]:<15} ROC-AUC: {score:.4f}")

# 3. PRECISION, RECALL, AND F1-SCORE BENCHMARKS
print("\n" + "-"*40)
print("[3] CLASSIFICATION REPORT (PRECISION & RECALL):")
print("-"*40)
print(classification_report(y_test, y_pred, target_names=class_names))
print("═"*60)


