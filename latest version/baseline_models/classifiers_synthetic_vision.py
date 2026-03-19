# baseline_models/classifiers_synthetic_vision.py

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                              recall_score, roc_curve, auc, precision_recall_curve)
import pandas as pd
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# --- Generate the same synthetic data as the framework ---
np.random.seed(42)
torch.manual_seed(42)

num_train, num_test = 2000, 500

# Flatten 3x64x64 = 12288 features
X_train_raw = torch.randn(num_train, 3, 64, 64).numpy().reshape(num_train, -1)
y_train = torch.randint(0, 2, (num_train,)).numpy()

X_test_raw = torch.randn(num_test, 3, 64, 64).numpy().reshape(num_test, -1)
y_test = torch.randint(0, 2, (num_test,)).numpy()

sens_test = torch.randint(0, 2, (num_test,)).numpy()
sex_test = np.array(["Male" if v == 1 else "Female" for v in sens_test])


def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)

    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    prec_c, rec_c, _ = precision_recall_curve(y_test, y_score)
    pr_auc = auc(rec_c, prec_c)

    print(f"\n{'='*48}\n{model_name}")
    print(pd.DataFrame({'Accuracy': [acc], 'Precision': [prec],
                        'Recall': [rec], 'F1': [f1]}).to_string(index=False))
    return fpr, tpr, roc_auc, rec_c, prec_c, pr_auc


models = {}
models["Logistic Regression"] = evaluate_model(
    LogisticRegression(max_iter=200, random_state=42).fit(X_train_raw, y_train),
    X_test_raw, y_test, "Logistic Regression")

models["Random Forest"] = evaluate_model(
    RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train_raw, y_train),
    X_test_raw, y_test, "Random Forest")

models["Decision Tree"] = evaluate_model(
    DecisionTreeClassifier(random_state=42).fit(X_train_raw, y_train),
    X_test_raw, y_test, "Decision Tree")

models["AdaBoost"] = evaluate_model(
    AdaBoostClassifier(random_state=42).fit(X_train_raw, y_train),
    X_test_raw, y_test, "AdaBoost")

models["Naive Bayes"] = evaluate_model(
    GaussianNB().fit(X_train_raw, y_train),
    X_test_raw, y_test, "Naive Bayes")

# --- Plot ROC and PR curves ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for name, (fpr, tpr, roc_auc, rec, prec, pr_auc) in models.items():
    axes[0].plot(fpr, tpr, lw=2, label=f'{name} (AUC={roc_auc:.2f})')
    axes[1].plot(rec, prec, lw=2, label=f'{name} (AUC={pr_auc:.2f})')

axes[0].plot([0,1],[0,1],'k--'); axes[0].set_title("ROC Curve"); axes[0].legend()
axes[1].set_title("Precision-Recall Curve"); axes[1].legend()
plt.tight_layout(); plt.show()
