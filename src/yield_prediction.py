"""
AGROSMART AI - Crop Yield Prediction Engine
Implements multi-model regression algorithms:
1. Linear / Ridge Regression
2. Decision Tree Regressor
3. Random Forest Regressor
4. Support Vector Regressor (SVR)
5. K-Nearest Neighbors Regressor
6. XGBoost Regressor
7. Gradient Boosting Regressor
8. Stacking Regressor (Meta-Ensemble)

Evaluates MAE, MSE, RMSE, R², and adjusted R².
Predicts expected yield in tonnes per hectare (t/ha) and calculates
production estimations and confidence intervals.
"""

from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
import joblib

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import (
    YIELD_PREDICTION_MODELS, METRICS_DIR, RANDOM_STATE, TEST_SIZE
)
from src.data_loader import load_yield_data
from src.preprocessing import YieldPredictionPreprocessor
from src.ensemble import build_stacking_regressor
from src.evaluation import evaluate_regression_model, save_metrics_to_json


def get_yield_regression_models() -> Dict[str, Any]:
    """Returns dictionary of baseline regression models."""
    return {
        "Ridge Regression": Ridge(alpha=1.0, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeRegressor(random_state=RANDOM_STATE),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
        "Support Vector Regressor": SVR(kernel="rbf", C=10.0),
        "K-Nearest Neighbors": KNeighborsRegressor(n_neighbors=5, n_jobs=-1),
        "XGBoost": XGBRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=RANDOM_STATE),
        "Stacking Ensemble": build_stacking_regressor()
    }


def train_and_evaluate_all_yield_models(
    df: Optional[pd.DataFrame] = None
) -> Tuple[Dict[str, Any], Any, YieldPredictionPreprocessor]:
    """
    Trains all candidate regression models on train/test split.
    Identifies best model based on R² score, saves serialized artifacts,
    and returns comparative metrics.
    """
    if df is None:
        df = load_yield_data()

    y = df["Yield"].values
    df_features = df.drop(columns=["Yield", "Production"], errors="ignore")

    X_train_df, X_test_df, y_train, y_test = train_test_split(
        df_features, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    preprocessor = YieldPredictionPreprocessor()
    X_train = preprocessor.fit_transform(X_train_df)
    X_test = preprocessor.transform(X_test_df)

    models = get_yield_regression_models()
    all_metrics = {}
    best_model = None
    best_r2 = -float("inf")
    best_model_name = ""

    print(f"Training {len(models)} regression models for Crop Yield Prediction...")

    for name, model in models.items():
        t0 = time.time()
        model.fit(X_train, y_train)
        train_time = round(time.time() - t0, 3)

        metrics = evaluate_regression_model(
            model=model,
            X_test=X_test,
            y_test=y_test,
            model_name=name
        )
        metrics["training_time_seconds"] = train_time
        all_metrics[name] = metrics

        print(f"  -> {name:25s} | R²: {metrics['r2_score']:.4f} | RMSE: {metrics['rmse']:.4f} t/ha | Train Time: {train_time}s")

        if metrics["r2_score"] > best_r2:
            best_r2 = metrics["r2_score"]
            best_model = model
            best_model_name = name

    print(f"\nOptimal Model: '{best_model_name}' with R²: {best_r2:.4f}")

    # Persist artifacts
    YIELD_PREDICTION_MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, YIELD_PREDICTION_MODELS / "best_yield_model.joblib")
    preprocessor.save(YIELD_PREDICTION_MODELS / "preprocessor.joblib")

    # Save individual models for comparison page
    for name, m in models.items():
        clean_name = name.lower().replace(" ", "_")
        joblib.dump(m, YIELD_PREDICTION_MODELS / f"{clean_name}.joblib")

    # Save best model's test RMSE for confidence interval computation
    best_rmse = all_metrics[best_model_name]["rmse"]
    joblib.dump(best_rmse, YIELD_PREDICTION_MODELS / "best_rmse.joblib")

    save_metrics_to_json(all_metrics, METRICS_DIR / "crop_yield_metrics.json")

    return all_metrics, best_model, preprocessor


def predict_yield(
    input_features: Dict[str, Any],
    model=None,
    preprocessor=None,
    confidence_level: float = 0.95
) -> Dict[str, Any]:
    """
    Predicts yield in tonnes/hectare and estimates total production.
    Computes 95% prediction interval: [predicted - 1.96 * RMSE, predicted + 1.96 * RMSE].
    """
    if preprocessor is None:
        prep_path = YIELD_PREDICTION_MODELS / "preprocessor.joblib"
        if not prep_path.exists():
            raise FileNotFoundError("Trained preprocessor not found. Run model training first.")
        preprocessor = YieldPredictionPreprocessor.load(prep_path)

    if model is None:
        model_path = YIELD_PREDICTION_MODELS / "best_yield_model.joblib"
        if not model_path.exists():
            raise FileNotFoundError("Trained yield model not found. Run model training first.")
        model = joblib.load(model_path)

    rmse_path = YIELD_PREDICTION_MODELS / "best_rmse.joblib"
    rmse = joblib.load(rmse_path) if rmse_path.exists() else 0.85

    df_in = pd.DataFrame([input_features])
    X_processed = preprocessor.transform(df_in)

    pred_yield = float(model.predict(X_processed)[0])
    pred_yield = max(0.1, round(pred_yield, 2))

    area = float(input_features.get("Area", 1.0))
    estimated_production = round(pred_yield * area, 2)

    # 95% Confidence Interval
    margin = 1.96 * rmse
    lower_bound = max(0.0, round(pred_yield - margin, 2))
    upper_bound = round(pred_yield + margin, 2)

    return {
        "predicted_yield_tha": pred_yield,
        "unit": "tonnes per hectare (t/ha)",
        "area_cultivated_ha": area,
        "estimated_total_production_tonnes": estimated_production,
        "confidence_level": f"{int(confidence_level * 100)}%",
        "prediction_interval": {
            "lower": lower_bound,
            "upper": upper_bound
        },
        "model_rmse": round(rmse, 3)
    }


if __name__ == "__main__":
    train_and_evaluate_all_yield_models()
