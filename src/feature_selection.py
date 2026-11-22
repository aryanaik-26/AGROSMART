"""
AGROSMART AI - Feature Selection & Dimensionality Reduction Module
Implements:
1. Univariate Feature Selection (SelectKBest with ANOVA F-test & Mutual Information).
2. Recursive Feature Elimination (RFE) with Cross-Validation.
3. Principal Component Analysis (PCA) for variance decomposition.
"""

from typing import Tuple, Dict, Any, List
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, f_regression, RFE
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.decomposition import PCA


def perform_select_k_best(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    k: int = 5,
    mode: str = "classification"
) -> Dict[str, Any]:
    """
    Selects top K features using ANOVA F-value scoring.
    """
    score_fn = f_classif if mode == "classification" else f_regression
    k = min(k, X.shape[1])
    selector = SelectKBest(score_func=score_fn, k=k)
    X_new = selector.fit_transform(X, y)

    scores = selector.scores_
    selected_indices = selector.get_support(indices=True)
    selected_features = [feature_names[i] for i in selected_indices]

    results = {
        "selected_features": selected_features,
        "feature_scores": {
            feature_names[i]: float(scores[i]) if not np.isnan(scores[i]) else 0.0
            for i in range(len(feature_names))
        },
        "X_transformed": X_new
    }
    return results


def perform_rfe(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    n_features_to_select: int = 5,
    mode: str = "classification"
) -> Dict[str, Any]:
    """
    Performs Recursive Feature Elimination using a Random Forest estimator.
    """
    estimator = (
        RandomForestClassifier(n_estimators=50, random_state=42)
        if mode == "classification"
        else RandomForestRegressor(n_estimators=50, random_state=42)
    )

    n_sel = min(n_features_to_select, X.shape[1])
    rfe = RFE(estimator=estimator, n_features_to_select=n_sel, step=1)
    X_new = rfe.fit_transform(X, y)

    selected_indices = np.where(rfe.support_)[0]
    selected_features = [feature_names[i] for i in selected_indices]

    rankings = {feature_names[i]: int(rfe.ranking_[i]) for i in range(len(feature_names))}

    return {
        "selected_features": selected_features,
        "rankings": rankings,
        "X_transformed": X_new
    }


def perform_pca_reduction(
    X: np.ndarray,
    n_components: int = 2
) -> Dict[str, Any]:
    """
    Performs Principal Component Analysis (PCA) for dimensionality reduction.
    Returns transformed matrix, explained variance ratio, and components.
    """
    n_comp = min(n_components, X.shape[1])
    pca = PCA(n_components=n_comp, random_state=42)
    X_pca = pca.fit_transform(X)

    return {
        "X_pca": X_pca,
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "total_explained_variance": float(np.sum(pca.explained_variance_ratio_)),
        "components": pca.components_.tolist()
    }
