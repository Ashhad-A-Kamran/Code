import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score, accuracy_score, classification_report, f1_score
import xgboost as xgb
from imblearn.combine import SMOTETomek
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, loguniform, randint
from sklearn.metrics import recall_score, precision_score, accuracy_score, classification_report, f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.svm import SVC
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.exceptions import FitFailedWarning
import xgboost as xgb
from imblearn.combine import SMOTETomek
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings("ignore")


data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data',
                   header=None, sep=' ')

feature_names = ['status', 'duration', 'credit_history', 'purpose', 'amount',
                 'savings', 'employment_duration', 'installment_rate', 'statussex',
                 'other_debtors', 'residence_since', 'property', 'age', 'other_installment_plans',
                 'housing', 'number_credits', 'job', 'people_liable', 'telephone', 'foreign_worker',
                 'credit_risk']

data.columns = feature_names

categorical_cols = ['status', 'credit_history', 'purpose', 'savings', 'employment_duration',
                    'statussex', 'other_debtors', 'property', 'other_installment_plans',
                    'housing', 'job', 'telephone', 'foreign_worker']

data = pd.get_dummies(data, columns=categorical_cols)
data.credit_risk.replace([1,2], [1,0], inplace=True)

target_col = 'credit_risk'
features = data.drop(target_col, axis=1)
target = data[target_col]
sensitive_features = ['statussex', 'age']

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

smote_tomek = SMOTETomek(random_state=42)
X_train_balanced, y_train_balanced = smote_tomek.fit_resample(X_train, y_train)


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
xgboost_y_pred = xg_model.predict(X_test)

xg_precision = precision_score(y_test, xgboost_y_pred)
xg_recall = recall_score(y_test, xgboost_y_pred)
xg_f1 = f1_score(y_test, xgboost_y_pred)
xg_accuracy = accuracy_score(y_test, xgboost_y_pred)

xgb_metrics_dict = {'Precision': xg_precision, 'Recall': xg_recall, 'F1-Score': xg_f1, 'Accuracy': xg_accuracy}

xgb_metrics_df = pd.DataFrame.from_dict(xgb_metrics_dict, orient='index', columns=['Score'])
print("===============================================\n")
print("XGBoost Classifier Evaluation Metrics")
print(xgb_metrics_df)


# Initialize the Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=1000, random_state=42)
rf_model.fit(X_train_balanced, y_train_balanced)
y_pred = rf_model.predict(X_test)

rf_precision = precision_score(y_test, y_pred)
rf_recall = recall_score(y_test, y_pred)
rf_f1 = f1_score(y_test, y_pred)
rf_accuracy = accuracy_score(y_test, y_pred)

rf_metrics_dict = {'Precision': rf_precision, 'Recall': rf_recall, 'F1-Score': rf_f1, 'Accuracy': rf_accuracy}

rf_metrics_df = pd.DataFrame.from_dict(rf_metrics_dict, orient='index', columns=['Score'])
print("\n===============================================")
print("Random Forest Classifier Evaluation Metrics")
print(rf_metrics_df)


# Initialize the KNN Classifier
knn_model = KNeighborsClassifier()
knn_model.fit(X_train_balanced, y_train_balanced)
knn_y_pred = knn_model.predict(X_test)

knn_precision = precision_score(y_test, knn_y_pred)
knn_recall = recall_score(y_test, knn_y_pred)
knn_f1 = f1_score(y_test, knn_y_pred)
knn_accuracy = accuracy_score(y_test, knn_y_pred)

knn_metrics_dict = {'Precision': knn_precision, 'Recall': knn_recall, 'F1-Score': knn_f1, 'Accuracy': knn_accuracy}

knn_metrics_df = pd.DataFrame.from_dict(knn_metrics_dict, orient='index', columns=['Score'])
print("\n===============================================")
print("K-Nearest Neighbors Classifier Evaluation Metrics")
print(knn_metrics_df)



# Initialize the Naive Bayes Classifier
nb_model = GaussianNB()
nb_model.fit(X_train_balanced, y_train_balanced)
nb_y_pred = nb_model.predict(X_test)

nb_precision = precision_score(y_test, nb_y_pred)
nb_recall = recall_score(y_test, nb_y_pred)
nb_f1 = f1_score(y_test, nb_y_pred)
nb_accuracy = accuracy_score(y_test, nb_y_pred)

nb_metrics_dict = {'Precision': nb_precision, 'Recall': nb_recall, 'F1-Score': nb_f1, 'Accuracy': nb_accuracy}

nb_metrics_df = pd.DataFrame.from_dict(nb_metrics_dict, orient='index', columns=['Score'])
print("\n===============================================")
print("Naive Bayes Classifier Evaluation Metrics")
print(nb_metrics_df)

# Initialize the AdaBoost Classifier
ada_model = AdaBoostClassifier(random_state=42)
ada_model.fit(X_train_balanced, y_train_balanced)
ada_y_pred = ada_model.predict(X_test)

ada_precision = precision_score(y_test, ada_y_pred)
ada_recall = recall_score(y_test, ada_y_pred)
ada_f1 = f1_score(y_test, ada_y_pred)
ada_accuracy = accuracy_score(y_test, ada_y_pred)

ada_metrics_dict = {'Precision': ada_precision, 'Recall': ada_recall, 'F1-Score': ada_f1, 'Accuracy': ada_accuracy}

ada_metrics_df = pd.DataFrame.from_dict(ada_metrics_dict, orient='index', columns=['Score'])
print("\n===============================================")
print("Adaboost Classifier Evaluation Metrics")
print(ada_metrics_df)


# Initialize the MLP Classifier
mlp_model = MLPClassifier(random_state=42)
mlp_model.fit(X_train_balanced, y_train_balanced)
mlp_y_pred = mlp_model.predict(X_test)

mlp_precision = precision_score(y_test, mlp_y_pred)
mlp_recall = recall_score(y_test, mlp_y_pred)
mlp_f1 = f1_score(y_test, mlp_y_pred)
mlp_accuracy = accuracy_score(y_test, mlp_y_pred)

mlp_metrics_dict = {'Precision': mlp_precision, 'Recall': mlp_recall, 'F1-Score': mlp_f1, 'Accuracy': mlp_accuracy}

mlp_metrics_df = pd.DataFrame.from_dict(mlp_metrics_dict, orient='index', columns=['Score'])
print("\n===============================================")
print("MultiLayer Perceptron Classifier Evaluation Metrics")
print(mlp_metrics_df)


# Initialize the SVM Classifier
svm_model = SVC(random_state=42)
svm_model.fit(X_train_balanced, y_train_balanced)
svm_y_pred = svm_model.predict(X_test)

svm_precision = precision_score(y_test, svm_y_pred)
svm_recall = recall_score(y_test, svm_y_pred)
svm_f1 = f1_score(y_test, svm_y_pred)
svm_accuracy = accuracy_score(y_test, svm_y_pred)

svm_metrics_dict = {'Precision': svm_precision, 'Recall': svm_recall, 'F1-Score': svm_f1, 'Accuracy': svm_accuracy}

svm_metrics_df = pd.DataFrame.from_dict(svm_metrics_dict, orient='index', columns=['Score'])
print("\n===============================================")
print("SVM Classifier Evaluation Metrics")
print(svm_metrics_df)


from sklearn.gaussian_process import GaussianProcessClassifier

# Initialize the Gaussian Process Classifier
gaussian_model = GaussianProcessClassifier(random_state=42)
gaussian_model.fit(X_train_balanced, y_train_balanced)
gaussian_y_pred = gaussian_model.predict(X_test)

gm_precision = precision_score(y_test, gaussian_y_pred)
gm_recall = recall_score(y_test, gaussian_y_pred)
gm_f1 = f1_score(y_test, gaussian_y_pred)
gm_accuracy = accuracy_score(y_test, gaussian_y_pred)

gaussian_metrics_dict = {'Precision': gm_precision, 'Recall': gm_recall, 'F1-Score': gm_f1, 'Accuracy': gm_accuracy}

gaussian_metrics_df = pd.DataFrame.from_dict(gaussian_metrics_dict, orient='index', columns=['Score'])
print("\n===============================================")
print("Gaussian Process Classifier Evaluation Metrics")
print(gaussian_metrics_df)


# Initialize the Logistic Regression Classifier
lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_train_balanced, y_train_balanced)
lr_y_pred = lr_model.predict(X_test)


lr_precision = precision_score(y_test, lr_y_pred)
lr_recall = recall_score(y_test, lr_y_pred)
lr_f1 = f1_score(y_test, lr_y_pred)
lr_accuracy = accuracy_score(y_test, lr_y_pred)

lr_metrics_dict = {'Precision': lr_precision, 'Recall': lr_recall, 'F1-Score': lr_f1, 'Accuracy': lr_accuracy}

lr_metrics_df = pd.DataFrame.from_dict(lr_metrics_dict, orient='index', columns=['Score'])
print("\n===============================================")
print("Logistic Regression Classifier Evaluation Metrics")
print(lr_metrics_df)


# Initialize the Decision Tree Classifier
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train_balanced, y_train_balanced)
dt_y_pred = dt_model.predict(X_test)

dt_precision = precision_score(y_test, dt_y_pred)
dt_recall = recall_score(y_test, dt_y_pred)
dt_f1 = f1_score(y_test, dt_y_pred)
dt_accuracy = accuracy_score(y_test, dt_y_pred)

dt_metrics_dict = {'Precision': dt_precision, 'Recall': dt_recall, 'F1-Score': dt_f1, 'Accuracy': dt_accuracy}

dt_metrics_df = pd.DataFrame.from_dict(dt_metrics_dict, orient='index', columns=['Score'])
print("\n===============================================")
print("Decision Tree Classifier Evaluation Metrics")
print(dt_metrics_df)