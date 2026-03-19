import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score, accuracy_score, classification_report, f1_score, roc_curve, auc, precision_recall_curve
import xgboost as xgb
from imblearn.combine import SMOTETomek
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# Load the Adult Income dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
data = pd.read_csv(url, header=None, names=column_names, na_values=' ?', sep=',\s', engine='python')

# Drop rows with missing values
data.dropna(inplace=True)

# Select features and target
features = ['age', 'workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'hours-per-week', 'native-country']
target = 'income'

# One-hot encode categorical columns
data = pd.get_dummies(data, columns=features, drop_first=True)

# Encode the target variable
le = LabelEncoder()
data[target] = le.fit_transform(data[target])

# Define features and target
X = data.drop(target, axis=1)
y = data[target]

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Balance the dataset using SMOTETomek
smote_tomek = SMOTETomek(random_state=42)
X_train_balanced, y_train_balanced = smote_tomek.fit_resample(X_train, y_train)

# Function to evaluate and print metrics for a given model
def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    metrics_dict = {'Precision': precision, 'Recall': recall, 'F1-Score': f1, 'Accuracy': accuracy}
    metrics_df = pd.DataFrame.from_dict(metrics_dict, orient='index', columns=['Score'])
    print(f"\n===============================================\n{model_name} Evaluation Metrics")
    print(metrics_df)

    # Compute ROC curve and ROC area
    y_score = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)

    # Compute Precision-Recall curve and area
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_score)
    pr_auc = auc(recall_curve, precision_curve)

    return fpr, tpr, roc_auc, recall_curve, precision_curve, pr_auc

# Plotting function for ROC and Precision-Recall curves
def plot_curves(models):
    plt.figure(figsize=(14, 6))

    # ROC Curve
    plt.subplot(1, 2, 1)
    for model_name, (fpr, tpr, roc_auc, recall, precision, pr_auc) in models.items():
        plt.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')

    # Precision-Recall Curve
    plt.subplot(1, 2, 2)
    for model_name, (fpr, tpr, roc_auc, recall, precision, pr_auc) in models.items():
        plt.plot(recall, precision, lw=2, label=f'{model_name} (AUC = {pr_auc:.2f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')

    plt.show()

models = {}

# XGBoost
params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'tree_method': 'hist',
    'max_depth': 8,
    'learning_rate': 0.1,
    'n_estimators': 100,
    'seed': 42
}
xg_model = xgb.XGBClassifier(**params)
xg_model.fit(X_train_balanced, y_train_balanced)
models["XGBoost"] = evaluate_model(xg_model, X_test, y_test, "XGBoost Classifier")

# Random Forest
rf_model = RandomForestClassifier(n_estimators=1000, random_state=42)
rf_model.fit(X_train_balanced, y_train_balanced)
models["Random Forest"] = evaluate_model(rf_model, X_test, y_test, "Random Forest Classifier")

# Naive Bayes
nb_model = GaussianNB()
nb_model.fit(X_train_balanced, y_train_balanced)
models["Naive Bayes"] = evaluate_model(nb_model, X_test, y_test, "Naive Bayes Classifier")

# AdaBoost
ada_model = AdaBoostClassifier(random_state=42)
ada_model.fit(X_train_balanced, y_train_balanced)
models["AdaBoost"] = evaluate_model(ada_model, X_test, y_test, "AdaBoost Classifier")

# Decision Tree
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train_balanced, y_train_balanced)
models["Decision Tree"] = evaluate_model(dt_model, X_test, y_test, "Decision Tree Classifier")

# Plot ROC and Precision-Recall curves
plot_curves(models)