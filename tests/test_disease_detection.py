"""
Unit tests for Crop Disease detection deep learning and Grad-CAM modules.
"""

import pytest
import numpy as np
from PIL import Image
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from config import PLANTVILLAGE_DIR, DISEASE_DETECTION_MODELS
from src.prediction import predict_crop_disease


def test_disease_model_files_exist():
    # Verify trained weights exist
    assert (DISEASE_DETECTION_MODELS / "best_disease_model.keras").exists()
    assert (DISEASE_DETECTION_MODELS / "class_indices.json").exists()


def test_predict_crop_disease_on_image():
    # Find a sample image
    sample_classes = [d for d in PLANTVILLAGE_DIR.iterdir() if d.is_dir() and len(list(d.glob("*.jpg"))) > 0]
    assert len(sample_classes) > 0

    sample_img_path = list(sample_classes[0].glob("*.jpg"))[0]
    res = predict_crop_disease(sample_img_path)

    assert "predicted_class" in res
    assert "confidence_score" in res
    assert res["confidence_score"] >= 0.0
    assert "status" in res
    assert res["status"] in ["Healthy", "Diseased"]
    assert "pathogen" in res
    assert "organic_control" in res
    assert "gradcam_image" in res
