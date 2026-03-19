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


url = "https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv"
data = pd.read_csv(url)


data = data[(data.days_b_screening_arrest <= 30) &
            (data.days_b_screening_arrest >= -30) &
            (data.is_recid != -1) &
            (data.c_charge_degree != 'O') &
            (data.score_text != 'N/A')]

data['length_of_stay'] = (pd.to_datetime(data['c_jail_out']) - pd.to_datetime(data['c_jail_in'])).dt.days


features = ['age', 'c_charge_degree', 'race', 'age_cat', 'score_text', 'sex', 'priors_count', 'length_of_stay']
target = 'two_year_recid'

data = data[features + [target]]


categorical_cols = ['c_charge_degree', 'race', 'age_cat', 'score_text', 'sex']
data = pd.get_dummies(data, columns=categorical_cols)


X = data.drop(target, axis=1)
y = data[target]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


smote_tomek = SMOTETomek(random_state=42)
X_train_balanced, y_train_balanced = smote_tomek.fit_resample(X_train, y_train)


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


    y_score = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)

    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_score)
    pr_auc = auc(recall_curve, precision_curve)

    return fpr, tpr, roc_auc, recall_curve, precision_curve, pr_auc


def plot_curves(models):
    plt.figure(figsize=(14, 6))


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


plot_curves(models)