"""
AGROSMART AI - Ensemble Learning Module
Implements Stacking Classifiers and Stacking Regressors to combine diverse learning
paradigms (Tree ensembles, Gradient boosting, Support Vector Machines)
with cross-validated meta-learners.
"""

from sklearn.ensemble import (
    StackingClassifier, StackingRegressor,
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import RANDOM_STATE


def build_stacking_classifier(base_estimators=None, final_estimator=None) -> StackingClassifier:
    """
    Constructs a meta-stacking ensemble classifier combining Random Forest,
    XGBoost, and RBF-kernel Support Vector Machine.
    """
    if base_estimators is None:
        base_estimators = [
            ("rf", RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)),
            ("xgb", XGBClassifier(n_estimators=100, eval_metric="mlogloss", random_state=RANDOM_STATE, n_jobs=-1)),
            ("svc", SVC(probability=True, kernel="rbf", C=1.5, random_state=RANDOM_STATE))
        ]

    if final_estimator is None:
        final_estimator = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)

    ensemble = StackingClassifier(
        estimators=base_estimators,
        final_estimator=final_estimator,
        cv=5,
        n_jobs=-1
    )
    return ensemble


def build_stacking_regressor(base_estimators=None, final_estimator=None) -> StackingRegressor:
    """
    Constructs a meta-stacking ensemble regressor combining Random Forest,
    XGBoost, and Gradient Boosting with a regularized Ridge meta-estimator.
    """
    if base_estimators is None:
        base_estimators = [
            ("rf", RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)),
            ("xgb", XGBRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)),
            ("gb", GradientBoostingRegressor(n_estimators=100, random_state=RANDOM_STATE))
        ]

    if final_estimator is None:
        final_estimator = Ridge(alpha=1.0, random_state=RANDOM_STATE)

    ensemble = StackingRegressor(
        estimators=base_estimators,
        final_estimator=final_estimator,
        cv=5,
        n_jobs=-1
    )
    return ensemble
