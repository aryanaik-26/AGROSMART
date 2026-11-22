"""
AGROSMART AI - Model Evaluation & Performance Metrics Module
Calculates standard and advanced metrics for classification and regression models,
computes inference latency, and saves metrics to reports/metrics/.
"""

import time
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score
)

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import METRICS_DIR


def evaluate_classification_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
    class_labels: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Evaluates multi-class classification model across:
    - Accuracy
    - Precision (macro, weighted)
    - Recall (macro, weighted)
    - F1-score (macro, weighted)
    - ROC-AUC (One-vs-Rest, when probability available)
    - Inference Latency (ms per sample)
    - Confusion Matrix
    """
    start_time = time.perf_counter()
    y_pred = model.predict(X_test)
    inference_time = (time.perf_counter() - start_time) / len(X_test) * 1000.0  # ms/sample

    acc = float(accuracy_score(y_test, y_pred))
    prec_macro = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
    prec_weighted = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
    rec_macro = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
    rec_weighted = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
    f1_macro = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
    f1_weighted = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))

    cm = confusion_matrix(y_test, y_pred).tolist()

    roc_auc = None
    if hasattr(model, "predict_proba"):
        try:
            y_prob = model.predict_proba(X_test)
            roc_auc = float(roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro"))
        except Exception:
            roc_auc = None

    metrics = {
        "model_name": model_name,
        "accuracy": round(acc, 4),
        "precision_macro": round(prec_macro, 4),
        "precision_weighted": round(prec_weighted, 4),
        "recall_macro": round(rec_macro, 4),
        "recall_weighted": round(rec_weighted, 4),
        "f1_macro": round(f1_macro, 4),
        "f1_weighted": round(f1_weighted, 4),
        "roc_auc": round(roc_auc, 4) if roc_auc is not None else "N/A",
        "inference_latency_ms": round(inference_time, 3),
        "confusion_matrix": cm
    }

    return metrics


def evaluate_regression_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str
) -> Dict[str, Any]:
    """
    Evaluates regression model across:
    - Mean Absolute Error (MAE)
    - Mean Squared Error (MSE)
    - Root Mean Squared Error (RMSE)
    - Coefficient of Determination (R²)
    - Adjusted R²
    - Inference Latency (ms per sample)
    """
    start_time = time.perf_counter()
    y_pred = model.predict(X_test)
    inference_time = (time.perf_counter() - start_time) / len(X_test) * 1000.0  # ms/sample

    mae = float(mean_absolute_error(y_test, y_pred))
    mse = float(mean_squared_error(y_test, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_test, y_pred))

    n = len(y_test)
    p = X_test.shape[1]
    adj_r2 = 1.0 - (1.0 - r2) * ((n - 1) / max(1, n - p - 1))

    metrics = {
        "model_name": model_name,
        "mae": round(mae, 4),
        "mse": round(mse, 4),
        "rmse": round(rmse, 4),
        "r2_score": round(r2, 4),
        "adjusted_r2": round(adj_r2, 4),
        "inference_latency_ms": round(inference_time, 3)
    }

    return metrics


def save_metrics_to_json(metrics: Dict[str, Any], filepath: Path):
    """Saves metrics dictionary to a JSON file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
