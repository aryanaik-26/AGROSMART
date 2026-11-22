"""
AGROSMART AI - Hyperparameter Optimization Module
Implements systematic model tuning using:
1. GridSearchCV (exhaustive grid search over designated param grids)
2. RandomizedSearchCV (stochastic sampling over parameter distributions)
3. Performance comparison before and after optimization across RF, XGBoost, SVM, and GB.
"""

from typing import Dict, Any, Tuple
import time
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor
)
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import RANDOM_STATE
from src.evaluation import evaluate_classification_model, evaluate_regression_model


# ==============================================================================
# PARAMETER SPACES
# ==============================================================================
OPTIMIZATION_GRIDS = {
    "random_forest_classification": {
        "n_estimators": [50, 100, 150],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5]
    },
    "xgboost_classification": {
        "n_estimators": [50, 100, 150],
        "max_depth": [3, 6, 9],
        "learning_rate": [0.05, 0.1, 0.2]
    },
    "svm_classification": {
        "C": [0.5, 1.0, 5.0],
        "kernel": ["rbf", "linear"],
        "gamma": ["scale", "auto"]
    },
    "gradient_boosting_classification": {
        "n_estimators": [50, 100],
        "learning_rate": [0.05, 0.1],
        "max_depth": [3, 5]
    },
    "random_forest_regression": {
        "n_estimators": [50, 100, 150],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5]
    },
    "xgboost_regression": {
        "n_estimators": [50, 100, 150],
        "max_depth": [3, 6, 8],
        "learning_rate": [0.03, 0.1, 0.2]
    }
}


def tune_model_grid_search(
    estimator,
    param_grid: Dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv: int = 5,
    scoring: str = "accuracy"
) -> Tuple[Any, Dict[str, Any]]:
    """
    Exhaustively searches over parameter grid with cross-validation.
    """
    grid = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        verbose=0
    )
    start_time = time.time()
    grid.fit(X_train, y_train)
    duration = time.time() - start_time

    results = {
        "best_params": grid.best_params_,
        "best_score": float(grid.best_score_),
        "tuning_duration_seconds": round(duration, 2)
    }
    return grid.best_estimator_, results


def tune_model_random_search(
    estimator,
    param_dist: Dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_iter: int = 10,
    cv: int = 5,
    scoring: str = "accuracy"
) -> Tuple[Any, Dict[str, Any]]:
    """
    Randomly samples from parameter space for fast hyperparameter exploration.
    """
    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0
    )
    start_time = time.time()
    search.fit(X_train, y_train)
    duration = time.time() - start_time

    results = {
        "best_params": search.best_params_,
        "best_score": float(search.best_score_),
        "tuning_duration_seconds": round(duration, 2)
    }
    return search.best_estimator_, results


def compare_optimization_impact(
    baseline_model,
    optimized_model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    mode: str = "classification"
) -> Dict[str, Any]:
    """
    Quantifies metric shifts before and after hyperparameter tuning.
    """
    if mode == "classification":
        base_metrics = evaluate_classification_model(baseline_model, X_test, y_test, "Baseline")
        opt_metrics = evaluate_classification_model(optimized_model, X_test, y_test, "Optimized")
        improvement = {
            "accuracy_delta": round(opt_metrics["accuracy"] - base_metrics["accuracy"], 4),
            "f1_delta": round(opt_metrics["f1_macro"] - base_metrics["f1_macro"], 4)
        }
    else:
        base_metrics = evaluate_regression_model(baseline_model, X_test, y_test, "Baseline")
        opt_metrics = evaluate_regression_model(optimized_model, X_test, y_test, "Optimized")
        improvement = {
            "r2_delta": round(opt_metrics["r2_score"] - base_metrics["r2_score"], 4),
            "rmse_reduction": round(base_metrics["rmse"] - opt_metrics["rmse"], 4)
        }

    return {
        "baseline": base_metrics,
        "optimized": opt_metrics,
        "improvement": improvement
    }
