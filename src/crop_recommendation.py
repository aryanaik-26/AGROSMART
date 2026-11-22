"""
AGROSMART AI - Crop Recommendation Engine
Implements multi-model classification algorithms:
1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier
4. Support Vector Machine (SVC)
5. K-Nearest Neighbors (KNN)
6. XGBoost Classifier
7. Gradient Boosting Classifier
8. Stacking Classifier (Meta-Ensemble)

Evaluates accuracy, macro-F1, precision, recall, and ROC-AUC.
Persists trained models and preprocessors.
"""

from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
import joblib

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import (
    CROP_RECOMMENDATION_MODELS, METRICS_DIR, RANDOM_STATE, TEST_SIZE
)
from src.data_loader import load_crop_recommendation_data
from src.preprocessing import CropRecommendationPreprocessor
from src.ensemble import build_stacking_classifier
from src.evaluation import evaluate_classification_model, save_metrics_to_json


def get_crop_classification_models() -> Dict[str, Any]:
    """Returns dictionary of baseline classification models."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
        "Support Vector Machine": SVC(probability=True, kernel="rbf", random_state=RANDOM_STATE),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        "XGBoost": XGBClassifier(n_estimators=100, eval_metric="mlogloss", random_state=RANDOM_STATE, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE),
        "Stacking Ensemble": build_stacking_classifier()
    }


def train_and_evaluate_all_crop_models(
    df: Optional[pd.DataFrame] = None
) -> Tuple[Dict[str, Any], Any, CropRecommendationPreprocessor]:
    """
    Trains all candidate classification models on stratified train/test split.
    Identifies best model, saves serialized artifacts, and returns comparative metrics.
    """
    if df is None:
        df = load_crop_recommendation_data()

    X = df[["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]]
    y = df["label"]

    # Stratified split to preserve class proportions
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Preprocessing strictly on train data
    preprocessor = CropRecommendationPreprocessor()
    X_train, y_train = preprocessor.fit_transform(X_train_raw, y_train_raw)
    X_test = preprocessor.transform(X_test_raw)
    y_test = preprocessor.label_encoder.transform(y_test_raw)

    models = get_crop_classification_models()
    all_metrics = {}
    best_model = None
    best_f1 = -1.0
    best_model_name = ""

    print(f"Training {len(models)} classification models for Crop Recommendation...")

    for name, model in models.items():
        t0 = time.time()
        model.fit(X_train, y_train)
        train_time = round(time.time() - t0, 3)

        metrics = evaluate_classification_model(
            model=model,
            X_test=X_test,
            y_test=y_test,
            model_name=name,
            class_labels=list(preprocessor.label_encoder.classes_)
        )
        metrics["training_time_seconds"] = train_time
        all_metrics[name] = metrics

        print(f"  -> {name:25s} | Accuracy: {metrics['accuracy']:.4f} | Macro-F1: {metrics['f1_macro']:.4f} | Train Time: {train_time}s")

        if metrics["f1_macro"] > best_f1:
            best_f1 = metrics["f1_macro"]
            best_model = model
            best_model_name = name

    print(f"\nOptimal Model: '{best_model_name}' with Macro-F1: {best_f1:.4f}")

    # Persist artifacts
    CROP_RECOMMENDATION_MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, CROP_RECOMMENDATION_MODELS / "best_crop_model.joblib")
    preprocessor.save(CROP_RECOMMENDATION_MODELS / "preprocessor.joblib")

    # Save individual models for comparison page
    for name, m in models.items():
        clean_name = name.lower().replace(" ", "_")
        joblib.dump(m, CROP_RECOMMENDATION_MODELS / f"{clean_name}.joblib")

    save_metrics_to_json(all_metrics, METRICS_DIR / "crop_recommendation_metrics.json")

    return all_metrics, best_model, preprocessor


def predict_top_crops(
    input_features: Dict[str, float],
    model=None,
    preprocessor=None,
    top_k: int = 3
) -> List[Dict[str, Any]]:
    """
    Generates top K recommended crops with calibrated confidence probabilities.
    Input dict format:
        {"N": 90, "P": 42, "K": 43, "temperature": 20.8, "humidity": 82.0, "ph": 6.5, "rainfall": 202.9}
    """
    if preprocessor is None:
        prep_path = CROP_RECOMMENDATION_MODELS / "preprocessor.joblib"
        if not prep_path.exists():
            raise FileNotFoundError("Trained preprocessor not found. Run model training first.")
        preprocessor = CropRecommendationPreprocessor.load(prep_path)

    if model is None:
        model_path = CROP_RECOMMENDATION_MODELS / "best_crop_model.joblib"
        if not model_path.exists():
            raise FileNotFoundError("Trained crop model not found. Run model training first.")
        model = joblib.load(model_path)

    df_in = pd.DataFrame([input_features])
    X_scaled = preprocessor.transform(df_in)

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_scaled)[0]
    else:
        # Fallback for decision tree/svm without proba
        pred_idx = model.predict(X_scaled)[0]
        probs = np.zeros(len(preprocessor.label_encoder.classes_))
        probs[pred_idx] = 1.0

    top_indices = np.argsort(probs)[::-1][:top_k]
    classes = preprocessor.label_encoder.classes_

    recommendations = []
    for rank, idx in enumerate(top_indices, 1):
        crop_name = str(classes[idx])
        probability = float(probs[idx])
        recommendations.append({
            "rank": rank,
            "crop": crop_name.title(),
            "confidence_score": round(probability, 4),
            "percentage": round(probability * 100.0, 2)
        })

    return recommendations


if __name__ == "__main__":
    train_and_evaluate_all_crop_models()
