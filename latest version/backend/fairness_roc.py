"""
Reject Option Classification (ROC) based fairness calculation.

This module trains an XGBoost classifier with SMOTETomek balancing,
then evaluates both Group Fairness (Statistical Parity via rejected/accepted
instances) and Individual Fairness (Counterfactual Fairness using sex/race
attribute flipping).
"""
import logging
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from imblearn.combine import SMOTETomek

logger = logging.getLogger(__name__)

def _find_sensitive_col(X_test: pd.DataFrame, candidates: list) -> str | None:
    """Find the first matching sensitive column in a DataFrame."""
    for col in candidates:
        if col in X_test.columns:
            return col
    return None


def compute_roc_fairness(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    dataset_name: str = "adult",
    alpha: float = 0.65,
) -> dict:
    """
    Compute fairness metrics using Reject Option Classification (ROC).

    Steps:
    1. Balance training data with SMOTETomek.
    2. Train an XGBoost binary classifier.
    3. Use a probability threshold (alpha) to split test set into
       'accepted' (confident) and 'rejected' (uncertain) instances.
    4. Compute Statistical Parity (group fairness) on both splits.
    5. Compute Counterfactual Fairness by flipping the sensitive attribute
       and comparing prediction probabilities.

    Returns a dict with fairness metrics.
    """
    logger.info(f"[ROC] Starting ROC fairness computation for dataset: {dataset_name}")

    # 1. Balance training data
    try:
        smote_tomek = SMOTETomek(random_state=42)
        X_train_bal, y_train_bal = smote_tomek.fit_resample(X_train, y_train)
        logger.info(f"[ROC] SMOTETomek balanced: {len(X_train)} -> {len(X_train_bal)} samples")
    except Exception as e:
        logger.warning(f"[ROC] SMOTETomek failed ({e}), using original data.")
        X_train_bal, y_train_bal = X_train, y_train

    # 2. Train XGBoost
    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",
        "max_depth": 6,
        "learning_rate": 0.1,
        "n_estimators": 100,
        "seed": 42,
        "verbosity": 0,
    }
    xgb = XGBClassifier(**params)
    xgb.fit(X_train_bal, y_train_bal)

    # 3. Get probabilities and predictions
    y_prob = xgb.predict_proba(X_test)[:, 1]
    y_pred = xgb.predict(X_test)
    overall_accuracy = float(accuracy_score(y_test, y_pred))

    # 4. Reject Option Classification at threshold alpha
    y_prob_sorted = np.sort(y_prob)
    idx = min(int(alpha * len(y_prob)), len(y_prob) - 1)
    reject_threshold = float(y_prob_sorted[idx])

    accepted_mask = y_prob >= reject_threshold
    rejected_mask = y_prob < reject_threshold

    acc_accepted = float(accuracy_score(y_test[accepted_mask], y_pred[accepted_mask])) if accepted_mask.any() else 0.0
    acc_rejected = float(accuracy_score(y_test[rejected_mask], y_pred[rejected_mask])) if rejected_mask.any() else 0.0

    # Group fairness (Statistical Parity) = actual positive rate - predicted positive rate
    gf_accepted = 0.0
    gf_rejected = 0.0
    if accepted_mask.any():
        gf_accepted = float((y_test[accepted_mask] == 1).mean()) - float((y_pred[accepted_mask] == 1).mean())
    if rejected_mask.any():
        gf_rejected = float((y_test[rejected_mask] == 1).mean()) - float((y_pred[rejected_mask] == 1).mean())

    # 5. Counterfactual Fairness — flip the sex sensitive attribute
    # Detect which column to flip based on dataset
    counterfactual_fairness_score = 0.0
    cf_accepted = 0.0
    cf_rejected = 0.0

    sex_col_candidates = {
        "adult":         ["sex_ Male", "sex_Male", "sex_ Female"],
        "compass":       ["Female"],
        "german_credit": ["statussex_A92"],  # A92 = female
    }
    flip_col = _find_sensitive_col(X_test, sex_col_candidates.get(dataset_name, []))

    if flip_col:
        counterfactual_X_test = X_test.copy()
        counterfactual_X_test[flip_col] = 1 - counterfactual_X_test[flip_col]
        y_prob_cf = xgb.predict_proba(counterfactual_X_test)[:, 1]
        cf_diff = np.abs(y_prob - y_prob_cf)

        cf_sorted = np.sort(cf_diff)
        cf_idx = min(int(alpha * len(cf_diff)), len(cf_diff) - 1)
        cf_threshold = float(cf_sorted[cf_idx])

        cf_acc_mask = cf_diff >= cf_threshold
        cf_rej_mask = cf_diff < cf_threshold

        cf_accepted = float(cf_diff[cf_acc_mask].mean()) if cf_acc_mask.any() else 0.0
        cf_rejected = float(cf_diff[cf_rej_mask].mean()) if cf_rej_mask.any() else 0.0
        counterfactual_fairness_score = float(cf_diff.mean())
    else:
        logger.warning(f"[ROC] No flip column found for counterfactual fairness in dataset '{dataset_name}'")

    result = {
        "method": "roc",
        "overall_accuracy": overall_accuracy,
        "accuracy_accepted": acc_accepted,
        "accuracy_rejected": acc_rejected,
        "group_fairness_accepted": gf_accepted,
        "group_fairness_rejected": gf_rejected,
        "counterfactual_fairness_accepted": cf_accepted,
        "counterfactual_fairness_rejected": cf_rejected,
        "counterfactual_fairness_mean": counterfactual_fairness_score,
        # Primary scalar used for the live dashboard charts:
        "fairness_score": abs(gf_accepted),
        "bias_score": counterfactual_fairness_score,
    }
    logger.info(f"[ROC] Done. Group Fairness (accepted): {gf_accepted:.4f}, CF mean: {counterfactual_fairness_score:.4f}")
    return result
