"""
AGROSMART AI - Explainable AI (XAI) Engine
Implements:
1. Grad-CAM (Gradient-weighted Class Activation Mapping) for deep learning leaf disease models.
2. SHAP (SHapley Additive exPlanations) for crop recommendation and yield prediction models.
3. Automated natural language reasoning from mathematical explanations.
"""

from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from PIL import Image

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import THEME_COLORS


# ==============================================================================
# 1. GRAD-CAM (LEAF DISEASE VISUAL EXPLANATION)
# ==============================================================================
def find_last_conv_layer(model) -> str:
    """
    Introspects model graph backwards to identify the final convolutional layer.
    Handles nested sub-models (like MobileNetV2 or EfficientNetB0 backbones).
    """
    import tensorflow as tf

    # Check top-level layers backwards
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
        # If nested functional backbone
        if hasattr(layer, "layers"):
            for sub_layer in reversed(layer.layers):
                if isinstance(sub_layer, tf.keras.layers.Conv2D):
                    return f"{layer.name}:{sub_layer.name}"

    # Fallback to known common names
    for name in ["conv2d_3", "Conv_1", "top_conv"]:
        try:
            model.get_layer(name)
            return name
        except Exception:
            continue

    raise ValueError("No convolutional layer found in model architecture.")


def compute_gradcam(
    model,
    img_array: np.ndarray,
    target_class_idx: Optional[int] = None,
    layer_name: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates Grad-CAM localization heatmap for a specified class index.
    Arguments:
        model: Trained Keras/TensorFlow model
        img_array: Preprocessed image tensor with shape (1, 224, 224, 3) and values in [0, 1]
        target_class_idx: Target index to explain (defaults to model's predicted class)
        layer_name: Name of target convolutional layer (auto-detected if None)
    Returns:
        (heatmap_normalized, superimposed_image_rgb)
    """
    import tensorflow as tf

    if img_array.ndim == 3:
        img_array = np.expand_dims(img_array, axis=0)

    # Auto-detect target conv layer if not provided
    if layer_name is None:
        layer_name = find_last_conv_layer(model)

    # Handle nested model vs flat sequential
    if ":" in layer_name:
        base_name, sub_name = layer_name.split(":")
        base_model = model.get_layer(base_name)
        sub_conv = base_model.get_layer(sub_name)

        # Grad-CAM on nested backbone
        grad_model = tf.keras.models.Model(
            inputs=base_model.input,
            outputs=[sub_conv.output, base_model.output]
        )

        with tf.GradientTape() as tape:
            # Pass through initial rescaling if present
            x = img_array
            for l in model.layers:
                if l.name == base_name:
                    break
                x = l(x)
            conv_outputs, base_features = grad_model(x)

            # Pass through top classifier
            top_out = base_features
            passed_base = False
            for l in model.layers:
                if l.name == base_name:
                    passed_base = True
                    continue
                if passed_base:
                    top_out = l(top_out)

            preds = top_out
            if target_class_idx is None:
                target_class_idx = tf.argmax(preds[0])
            loss = preds[:, target_class_idx]

        grads = tape.gradient(loss, conv_outputs)
    else:
        conv_layer = model.get_layer(layer_name)
        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[conv_layer.output, model.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, preds = grad_model(img_array)
            if target_class_idx is None:
                target_class_idx = tf.argmax(preds[0])
            loss = preds[:, target_class_idx]

        grads = tape.gradient(loss, conv_outputs)

    # Global average pooling of gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weighted sum of feature maps
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Apply ReLU to retain only features having positive impact
    heatmap = tf.maximum(heatmap, 0.0)
    max_val = tf.math.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val
    heatmap = heatmap.numpy()

    # Resize heatmap to match image dimensions (224, 224)
    heatmap_resized = cv2.resize(heatmap, (224, 224))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)

    # Colorize using JET colormap
    colormap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    colormap_rgb = cv2.cvtColor(colormap, cv2.COLOR_BGR2RGB)

    # Superimpose on original image
    orig_img = np.uint8(np.clip(img_array[0] * 255.0, 0, 255))
    superimposed = cv2.addWeighted(orig_img, 0.6, colormap_rgb, 0.4, 0)

    return heatmap_resized, superimposed


# ==============================================================================
# 2. SHAP EXPLANATION ENGINE (TABULAR MODELS)
# ==============================================================================
def explain_tabular_prediction_shap(
    model,
    X_sample: np.ndarray,
    feature_names: List[str],
    background_data: Optional[np.ndarray] = None,
    task: str = "regression"
) -> Dict[str, Any]:
    """
    Computes genuine SHAP values for an individual tabular prediction.
    Utilizes TreeExplainer for tree ensembles or KernelExplainer as fallback.
    """
    import shap

    if X_sample.ndim == 1:
        X_sample = X_sample.reshape(1, -1)

    # Select appropriate explainer
    is_tree = hasattr(model, "estimators_") or "XGB" in type(model).__name__

    if is_tree:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
    else:
        bg = background_data if background_data is not None else X_sample
        explainer = shap.KernelExplainer(model.predict, bg[:50])
        shap_values = explainer.shap_values(X_sample)

    # For multi-class classification, shap_values is a list of arrays (one per class)
    if isinstance(shap_values, list):
        pred_class = int(np.argmax(model.predict_proba(X_sample)[0])) if hasattr(model, "predict_proba") else 0
        vals = shap_values[pred_class][0]
        base_val = float(explainer.expected_value[pred_class]) if isinstance(explainer.expected_value, (list, np.ndarray)) else float(explainer.expected_value)
    elif shap_values.ndim == 3:  # newer SHAP format (samples, features, classes)
        pred_class = int(np.argmax(model.predict_proba(X_sample)[0])) if hasattr(model, "predict_proba") else 0
        vals = shap_values[0, :, pred_class]
        base_val = float(explainer.expected_value[pred_class]) if isinstance(explainer.expected_value, (list, np.ndarray)) else float(explainer.expected_value)
    else:
        vals = shap_values[0]
        base_val = float(explainer.expected_value) if not isinstance(explainer.expected_value, (list, np.ndarray)) else float(explainer.expected_value[0])

    contributions = {}
    for i, name in enumerate(feature_names[:len(vals)]):
        contributions[name] = float(vals[i])

    # Rank features by absolute magnitude of contribution
    sorted_features = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)

    # Generate domain narrative
    top_pos = [f"{k} (+{v:.2f})" for k, v in sorted_features if v > 0][:3]
    top_neg = [f"{k} ({v:.2f})" for k, v in sorted_features if v < 0][:2]

    narrative_parts = []
    if top_pos:
        narrative_parts.append(f"Strongest positive drivers: {', '.join(top_pos)}.")
    if top_neg:
        narrative_parts.append(f"Primary limiting factors: {', '.join(top_neg)}.")

    narrative = " ".join(narrative_parts) if narrative_parts else "All factors contributed evenly to baseline prediction."

    return {
        "base_value": round(base_val, 4),
        "feature_contributions": {k: round(v, 4) for k, v in contributions.items()},
        "ranked_features": sorted_features,
        "narrative": narrative
    }


def plot_shap_waterfall_bar(
    feature_contributions: Dict[str, float],
    title: str = "SHAP Feature Attribution Breakdown",
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Renders an aesthetic horizontal divergence bar plot showing positive and
    negative feature attributions.
    """
    items = sorted(feature_contributions.items(), key=lambda x: abs(x[1]))[-10:]
    names = [x[0] for x in items]
    vals = [x[1] for x in items]
    colors = [THEME_COLORS["primary"] if v >= 0 else THEME_COLORS["warning"] for v in vals]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(names, vals, color=colors, edgecolor="black", height=0.6)

    ax.axvline(0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12, color=THEME_COLORS["primary_dark"])
    ax.set_xlabel("SHAP Impact on Model Prediction", fontsize=11, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

    for bar, val in zip(bars, vals):
        offset = 0.02 if val >= 0 else -0.02
        ha = "left" if val >= 0 else "right"
        ax.text(val + offset, bar.get_y() + bar.get_height() / 2, f"{val:+.3f}",
                va="center", ha=ha, fontsize=9, fontweight="bold")

    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300)
    return fig
