"""
AGROSMART AI - Unified High-Level Prediction Orchestrator
Provides single-call interfaces for:
1. Leaf Disease Classification with Grad-CAM and Agronomic Advisory.
2. Crop Yield Regression with Confidence Intervals and SHAP attributions.
3. Crop Recommendation with Top-3 probabilities and feature explainability.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import json
import numpy as np
import pandas as pd
from PIL import Image
import joblib

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import (
    DISEASE_DETECTION_MODELS, YIELD_PREDICTION_MODELS, CROP_RECOMMENDATION_MODELS,
    DISEASE_INFO, IMAGE_SIZE
)
from src.preprocessing import CropRecommendationPreprocessor, YieldPredictionPreprocessor
from src.explainability import compute_gradcam, explain_tabular_prediction_shap


# ==============================================================================
# 1. CROP DISEASE PREDICTION PIPELINE
# ==============================================================================
def predict_crop_disease(
    image_input: Union[str, Path, Image.Image, np.ndarray],
    model_name: str = "Best Model"
) -> Dict[str, Any]:
    """
    Performs end-to-end leaf disease prediction and Grad-CAM generation.
    Returns:
        dict with class, confidence, healthy/diseased status, pathogen info,
        preventive measures, and Grad-CAM overlay image.
    """
    import tensorflow as tf

    # 1. Normalize image input to PIL RGB
    if isinstance(image_input, (str, Path)):
        img = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, np.ndarray):
        img = Image.fromarray(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):
        img = image_input.convert("RGB")
    else:
        raise ValueError("Unsupported image input type.")

    # 2. Resize and prepare tensor
    img_resized = img.resize(IMAGE_SIZE)
    img_array = np.array(img_resized, dtype=np.float32) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)

    # 3. Load model and class indices
    model_file_map = {
        "Best Model": "best_disease_model.keras",
        "Custom CNN": "custom_cnn.keras",
        "MobileNetV2": "mobilenetv2.keras",
        "EfficientNetB0": "efficientnetb0.keras"
    }
    target_file = model_file_map.get(model_name, "best_disease_model.keras")
    model_path = DISEASE_DETECTION_MODELS / target_file

    if not model_path.exists():
        # Fallback to any available keras model
        avail = list(DISEASE_DETECTION_MODELS.glob("*.keras"))
        if not avail:
            raise FileNotFoundError("No trained disease detection model found. Please run training first.")
        model_path = avail[0]

    model = tf.keras.models.load_model(model_path)

    # Class mappings
    mapping_path = DISEASE_DETECTION_MODELS / "class_indices.json"
    if not mapping_path.exists():
        raise FileNotFoundError("class_indices.json not found in models directory.")
    with open(mapping_path, "r") as f:
        class_indices = json.load(f)

    # 4. Predict
    predictions = model.predict(img_batch, verbose=0)[0]
    predicted_idx = int(np.argmax(predictions))
    predicted_class = str(class_indices.get(str(predicted_idx), f"Class_{predicted_idx}"))
    confidence = float(predictions[predicted_idx])

    # 5. Retrieve agronomic profile
    info = DISEASE_INFO.get(predicted_class, {
        "crop": "Unknown",
        "disease": predicted_class,
        "pathogen": "Botanical pathogen",
        "status": "Healthy" if "healthy" in predicted_class.lower() else "Diseased",
        "symptoms": "Detailed agronomic evaluation recommended.",
        "organic_control": "Apply standard botanical and bio-control protocols.",
        "chemical_control": "Consult regional agricultural extension office.",
        "prevention": "Maintain optimal irrigation and field hygiene."
    })

    # 6. Generate Grad-CAM visualization
    try:
        heatmap, gradcam_superimposed = compute_gradcam(
            model=model,
            img_array=img_array,
            target_class_idx=predicted_idx
        )
    except Exception as e:
        heatmap, gradcam_superimposed = None, np.array(img_resized)

    return {
        "predicted_class": predicted_class,
        "confidence_score": round(confidence, 4),
        "confidence_percentage": round(confidence * 100.0, 2),
        "crop_name": info["crop"],
        "disease_name": info["disease"],
        "status": info["status"],
        "pathogen": info["pathogen"],
        "symptoms": info["symptoms"],
        "organic_control": info["organic_control"],
        "chemical_control": info["chemical_control"],
        "prevention": info["prevention"],
        "original_image": img_resized,
        "gradcam_image": Image.fromarray(gradcam_superimposed) if gradcam_superimposed is not None else None,
        "heatmap": heatmap
    }


# ==============================================================================
# 2. CROP YIELD PREDICTION PIPELINE
# ==============================================================================
def predict_crop_yield_unified(
    features: Dict[str, Any],
    model_name: str = "Best Model"
) -> Dict[str, Any]:
    """
    Unified endpoint for yield prediction, confidence intervals, and SHAP attribution.
    """
    from src.yield_prediction import predict_yield

    prep_path = YIELD_PREDICTION_MODELS / "preprocessor.joblib"
    if not prep_path.exists():
        raise FileNotFoundError("Yield preprocessor not found. Run model training first.")
    preprocessor = YieldPredictionPreprocessor.load(prep_path)

    clean_name = "best_yield_model.joblib" if model_name == "Best Model" else f"{model_name.lower().replace(' ', '_')}.joblib"
    model_path = YIELD_PREDICTION_MODELS / clean_name
    if not model_path.exists():
        model_path = YIELD_PREDICTION_MODELS / "best_yield_model.joblib"

    model = joblib.load(model_path)

    # Core prediction
    yield_results = predict_yield(features, model=model, preprocessor=preprocessor)

    # Compute SHAP explanation
    df_sample = pd.DataFrame([features])
    X_processed = preprocessor.transform(df_sample)
    feature_names = preprocessor.transformed_feature_names

    try:
        shap_res = explain_tabular_prediction_shap(
            model=model,
            X_sample=X_processed,
            feature_names=feature_names,
            task="regression"
        )
    except Exception:
        shap_res = {"narrative": "Explanation unavailable for ensemble meta-model.", "feature_contributions": {}}

    yield_results["shap_explanation"] = shap_res
    return yield_results


# ==============================================================================
# 3. CROP RECOMMENDATION PIPELINE
# ==============================================================================
def recommend_crops_unified(
    features: Dict[str, float],
    model_name: str = "Best Model",
    top_k: int = 3
) -> Dict[str, Any]:
    """
    Unified endpoint for crop recommendation with top-3 crops and SHAP importance.
    """
    from src.crop_recommendation import predict_top_crops

    prep_path = CROP_RECOMMENDATION_MODELS / "preprocessor.joblib"
    if not prep_path.exists():
        raise FileNotFoundError("Crop preprocessor not found. Run model training first.")
    preprocessor = CropRecommendationPreprocessor.load(prep_path)

    clean_name = "best_crop_model.joblib" if model_name == "Best Model" else f"{model_name.lower().replace(' ', '_')}.joblib"
    model_path = CROP_RECOMMENDATION_MODELS / clean_name
    if not model_path.exists():
        model_path = CROP_RECOMMENDATION_MODELS / "best_crop_model.joblib"

    model = joblib.load(model_path)

    top_crops = predict_top_crops(features, model=model, preprocessor=preprocessor, top_k=top_k)

    # Compute SHAP explanation for top-1 crop
    df_sample = pd.DataFrame([features])
    X_processed = preprocessor.transform(df_sample)
    feature_names = preprocessor.feature_names

    try:
        shap_res = explain_tabular_prediction_shap(
            model=model,
            X_sample=X_processed,
            feature_names=feature_names,
            task="classification"
        )
    except Exception:
        shap_res = {"narrative": "Explanation unavailable.", "feature_contributions": {}}

    disclaimer = (
        "Agricultural Disclaimer: Model recommendations are probabilistic agronomic suitability scores "
        "based on historical benchmark data. Local farm decisions should incorporate expert extension testing, "
        "market economics, and specific seed variety requirements."
    )

    return {
        "top_recommendations": top_crops,
        "primary_recommendation": top_crops[0]["crop"] if top_crops else "None",
        "shap_explanation": shap_res,
        "disclaimer": disclaimer
    }
