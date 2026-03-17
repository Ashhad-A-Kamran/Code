"""
NSGA-II based multi-objective fairness optimization.

Objectives:
  F[0] = 1 - accuracy          (minimize → maximize accuracy)
  F[1] = |statistical_parity|  (minimize → maximize group fairness)

Uses pymoo NSGA2 to find a Pareto-optimal trade-off between these two
objectives, then selects the best compromise solution and returns its metrics.
"""
import logging
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
from imblearn.combine import SMOTETomek

from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize as pymoo_minimize
from pymoo.termination import get_termination

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Sensitive column config per dataset
# ---------------------------------------------------------------------------
PRIVILEGED_COLS = {
    "german_credit": ["statussex_A91", "statussex_A93", "statussex_A94"],
    "adult":         [],   # Uses 'sex' column differently
    "compass":       [],   # Non-female = privileged
}
UNPRIVILEGED_COLS = {
    "german_credit": ["statussex_A92"],
    "adult":         ["sex_ Female", "sex_Female"],
    "compass":       ["Female"],
}
AGE_COL = "age"


def _statistical_parity(y_pred: np.ndarray, X_test: pd.DataFrame, dataset_name: str) -> float:
    """Compute Statistical Parity Difference between privileged and unprivileged groups."""
    priv_cols = [c for c in PRIVILEGED_COLS.get(dataset_name, []) if c in X_test.columns]
    unpriv_cols = [c for c in UNPRIVILEGED_COLS.get(dataset_name, []) if c in X_test.columns]

    if not priv_cols and not unpriv_cols:
        # Fallback: no sensitive col found — return 0
        return 0.0

    if priv_cols:
        priv_mask = X_test[priv_cols[0]] == 1
        for col in priv_cols[1:]:
            priv_mask = priv_mask | (X_test[col] == 1)
    else:
        # For adult/compass: non-unprivileged = privileged
        if unpriv_cols and unpriv_cols[0] in X_test.columns:
            priv_mask = X_test[unpriv_cols[0]] == 0
        else:
            return 0.0

    if unpriv_cols and unpriv_cols[0] in X_test.columns:
        unpriv_mask = X_test[unpriv_cols[0]] == 1
    else:
        return 0.0

    # Apply age filter if available
    if AGE_COL in X_test.columns:
        age_filter = X_test[AGE_COL] > 18
        priv_mask = priv_mask & age_filter
        unpriv_mask = unpriv_mask & age_filter

    priv_pred = y_pred[priv_mask]
    unpriv_pred = y_pred[unpriv_mask]

    if len(priv_pred) == 0 or len(unpriv_pred) == 0:
        return 0.0

    return float(np.mean(priv_pred) - np.mean(unpriv_pred))


# ---------------------------------------------------------------------------
# pymoo Problem Definition
# ---------------------------------------------------------------------------
class FairnessProblem(Problem):
    """
    2-objective problem for NSGA-II:
      - Decision variables: XGBoost hyperparameters (max_depth, n_estimators, learning_rate)
        encoded as continuous values in [0, 1] and decoded.
      - Objectives:
          F[0] = 1 - accuracy
          F[1] = |statistical_parity_difference|
    """

    def __init__(self, X_train, y_train, X_test, y_test, dataset_name):
        # 3 real-valued decision variables (hyperparams)
        super().__init__(n_var=3, n_obj=2, n_ieq_constr=0, xl=0.0, xu=1.0)
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.dataset_name = dataset_name

    def _decode_params(self, x):
        """Map [0,1]^3 decision variables to XGBoost hyperparameters."""
        max_depth = int(2 + x[0] * 6)           # [2, 8]
        n_estimators = int(50 + x[1] * 150)     # [50, 200]
        learning_rate = 0.001 + x[2] * 0.299    # [0.001, 0.3]
        return max_depth, n_estimators, learning_rate

    def _evaluate(self, X, out, *args, **kwargs):
        F = np.full((len(X), 2), fill_value=1.0)

        for i, xi in enumerate(X):
            try:
                max_depth, n_estimators, learning_rate = self._decode_params(xi)
                model = XGBClassifier(
                    max_depth=max_depth,
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    objective="binary:logistic",
                    eval_metric="auc",
                    tree_method="hist",
                    seed=42,
                    verbosity=0,
                )
                model.fit(self.X_train, self.y_train)
                y_pred = model.predict(self.X_test)

                acc = float(accuracy_score(self.y_test, y_pred))
                spd = _statistical_parity(y_pred, self.X_test, self.dataset_name)

                F[i, 0] = 1.0 - acc        # minimize → maximize accuracy
                F[i, 1] = abs(spd)          # minimize → minimize fairness gap
            except Exception as e:
                logger.debug(f"[NSGA2] Individual {i} failed: {e}")
                F[i, 0] = 1.0
                F[i, 1] = 1.0

        out["F"] = F


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def run_nsga2_fairness(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    dataset_name: str = "german_credit",
    pop_size: int = 50,
    n_evals: int = 200,
) -> dict:
    """
    Run NSGA-II to find Pareto-optimal (accuracy, fairness) trade-offs.

    Returns the best compromise solution (closest to utopia point) as a dict.
    """
    logger.info(f"[NSGA2] Starting for dataset='{dataset_name}', pop_size={pop_size}, n_evals={n_evals}")

    # Balance training data
    try:
        smote_tomek = SMOTETomek(random_state=42)
        X_train_bal, y_train_bal = smote_tomek.fit_resample(X_train, y_train)
        logger.info(f"[NSGA2] Balanced: {len(X_train)} -> {len(X_train_bal)}")
    except Exception as e:
        logger.warning(f"[NSGA2] SMOTETomek failed ({e}), using original data.")
        X_train_bal, y_train_bal = X_train, y_train

    problem = FairnessProblem(X_train_bal, y_train_bal, X_test, y_test, dataset_name)

    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True,
    )

    termination = get_termination("n_evals", n_evals)

    try:
        res = pymoo_minimize(
            problem,
            algorithm,
            termination,
            seed=42,
            verbose=False,
        )

        pareto_F = res.F  # shape (n_solutions, 2)
        pareto_size = len(pareto_F)

        # Select best compromise: closest to utopia (0, 0)
        norms = np.linalg.norm(pareto_F, axis=1)
        best_idx = int(np.argmin(norms))
        best_acc = float(1.0 - pareto_F[best_idx, 0])
        best_fairness_gap = float(pareto_F[best_idx, 1])

        logger.info(f"[NSGA2] Done. Pareto size={pareto_size}, best acc={best_acc:.4f}, fairness_gap={best_fairness_gap:.4f}")

        return {
            "method": "nsga2",
            "overall_accuracy": best_acc,
            "pareto_size": pareto_size,
            "fairness_score": best_fairness_gap,   # used in dashboard
            "bias_score": best_fairness_gap,
            "all_pareto_accuracies": (1.0 - pareto_F[:, 0]).tolist(),
            "all_pareto_fairness": pareto_F[:, 1].tolist(),
        }

    except Exception as e:
        logger.error(f"[NSGA2] Optimization failed: {e}")
        return {
            "method": "nsga2",
            "overall_accuracy": 0.0,
            "pareto_size": 0,
            "fairness_score": 0.0,
            "bias_score": 0.0,
        }
