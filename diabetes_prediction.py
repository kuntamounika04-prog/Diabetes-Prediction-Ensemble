# Diabetes Prediction using Soft Voting Ensemble
# Source: diabetes document.docx
# Cleaned from the document so that it can be executed as Python code.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    auc,
    confusion_matrix,
)

# ---------------------------------------------------------
# 1. Load dataset
# ---------------------------------------------------------
# Keep diabetes.csv in the same folder as this script.
df = pd.read_csv("diabetes.csv")

print("Dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

print("\nMissing values:")
print(df.isna().sum())

# ---------------------------------------------------------
# 2. Split features and target
# ---------------------------------------------------------
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# 80% training, 20% testing, as used in the document
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ---------------------------------------------------------
# 3. Create individual models
# ---------------------------------------------------------
nb = GaussianNB()

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

lr = LogisticRegression(
    random_state=42,
    max_iter=1000
)

svm = SVC(
    kernel="linear",
    probability=True
)

# ---------------------------------------------------------
# 4. Create Soft Voting Ensemble
# ---------------------------------------------------------
# The document combines Naive Bayes, Random Forest,
# and Logistic Regression in the soft voting classifier.
ensemble = VotingClassifier(
    estimators=[
        ("nb", nb),
        ("rf", rf),
        ("lr", lr)
    ],
    voting="soft"
)

# ---------------------------------------------------------
# 5. Train models
# ---------------------------------------------------------
nb.fit(X_train, y_train)
rf.fit(X_train, y_train)
lr.fit(X_train, y_train)
svm.fit(X_train, y_train)
ensemble.fit(X_train, y_train)

# ---------------------------------------------------------
# 6. Predictions
# ---------------------------------------------------------
y_pred_nb = nb.predict(X_test)
y_pred_rf = rf.predict(X_test)
y_pred_lr = lr.predict(X_test)
y_pred_svm = svm.predict(X_test)
y_pred_ensemble = ensemble.predict(X_test)

# Prediction probabilities for ROC curves
y_prob_nb = nb.predict_proba(X_test)[:, 1]
y_prob_rf = rf.predict_proba(X_test)[:, 1]
y_prob_lr = lr.predict_proba(X_test)[:, 1]
y_prob_svm = svm.predict_proba(X_test)[:, 1]
y_prob_ensemble = ensemble.predict_proba(X_test)[:, 1]

# ---------------------------------------------------------
# 7. Evaluation function
# ---------------------------------------------------------
def evaluate_model(name, y_true, y_pred, y_prob):
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_prob)

    print(f"\n{name}")
    print("-" * len(name))
    print(f"Accuracy : {accuracy * 100:.2f}%")
    print(f"F1 Score : {f1 * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall   : {recall * 100:.2f}%")
    print(f"ROC AUC  : {roc_auc * 100:.2f}%")

    return {
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "roc_auc": roc_auc,
    }


nb_metrics = evaluate_model(
    "Naive Bayes", y_test, y_pred_nb, y_prob_nb
)

rf_metrics = evaluate_model(
    "Random Forest", y_test, y_pred_rf, y_prob_rf
)

lr_metrics = evaluate_model(
    "Logistic Regression", y_test, y_pred_lr, y_prob_lr
)

svm_metrics = evaluate_model(
    "Support Vector Machine", y_test, y_pred_svm, y_prob_svm
)

ensemble_metrics = evaluate_model(
    "Ensemble - Soft Voting", y_test, y_pred_ensemble, y_prob_ensemble
)

# ---------------------------------------------------------
# 8. Confusion matrices
# ---------------------------------------------------------
models_for_cm = {
    "Logistic Regression": y_pred_lr,
    "Naive Bayes": y_pred_nb,
    "SVM": y_pred_svm,
    "Random Forest": y_pred_rf,
    "Ensemble - Soft Voting": y_pred_ensemble,
}

for name, predictions in models_for_cm.items():
    cm = confusion_matrix(y_test, predictions)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.tight_layout()
    plt.show()

# ---------------------------------------------------------
# 9. ROC curves
# ---------------------------------------------------------
fpr_nb, tpr_nb, _ = roc_curve(y_test, y_prob_nb)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
fpr_svm, tpr_svm, _ = roc_curve(y_test, y_prob_svm)
fpr_ensemble, tpr_ensemble, _ = roc_curve(y_test, y_prob_ensemble)

roc_auc_nb = auc(fpr_nb, tpr_nb)
roc_auc_rf = auc(fpr_rf, tpr_rf)
roc_auc_lr = auc(fpr_lr, tpr_lr)
roc_auc_svm = auc(fpr_svm, tpr_svm)
roc_auc_ensemble = auc(fpr_ensemble, tpr_ensemble)

plt.figure(figsize=(9, 7))

plt.plot(
    fpr_nb, tpr_nb,
    label=f"Naive Bayes (AUC = {roc_auc_nb:.2f})"
)

plt.plot(
    fpr_rf, tpr_rf,
    label=f"Random Forest (AUC = {roc_auc_rf:.2f})"
)

plt.plot(
    fpr_lr, tpr_lr,
    label=f"Logistic Regression (AUC = {roc_auc_lr:.2f})"
)

plt.plot(
    fpr_svm, tpr_svm,
    label=f"SVM (AUC = {roc_auc_svm:.2f})"
)

plt.plot(
    fpr_ensemble, tpr_ensemble,
    label=f"Ensemble (AUC = {roc_auc_ensemble:.2f})"
)

plt.title("Receiver Operating Characteristic (ROC) Curves")
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.legend()
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 10. Compare evaluation metrics
# ---------------------------------------------------------
labels = [
    "Naive Bayes",
    "Random Forest",
    "Logistic Regression",
    "SVM",
    "Ensemble"
]

metrics = [
    nb_metrics,
    rf_metrics,
    lr_metrics,
    svm_metrics,
    ensemble_metrics
]

accuracy_scores = [m["accuracy"] for m in metrics]
precision_scores = [m["precision"] for m in metrics]
recall_scores = [m["recall"] for m in metrics]
f1_scores = [m["f1"] for m in metrics]

x = np.arange(len(labels))
width = 0.2

fig, ax = plt.subplots(figsize=(12, 8))

ax.bar(
    x - 1.5 * width,
    accuracy_scores,
    width,
    label="Accuracy"
)

ax.bar(
    x - 0.5 * width,
    precision_scores,
    width,
    label="Precision"
)

ax.bar(
    x + 0.5 * width,
    recall_scores,
    width,
    label="Recall"
)

ax.bar(
    x + 1.5 * width,
    f1_scores,
    width,
    label="F1 Score"
)

ax.set_ylabel("Scores")
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=15)
ax.legend()

fig.tight_layout()
plt.show()

print("\nDiabetes prediction model execution completed.")
