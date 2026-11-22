"""
Unit tests for Crop Recommendation classification suite and prediction pipeline.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.data_loader import load_crop_recommendation_data
from src.prediction import recommend_crops_unified
from src.crop_recommendation import predict_top_crops


def test_load_crop_recommendation_data():
    df = load_crop_recommendation_data()
    assert len(df) >= 2000
    assert "label" in df.columns
    assert set(["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]).issubset(set(df.columns))


def test_recommend_top_crops():
    sample_input = {
        "N": 80.0, "P": 45.0, "K": 40.0,
        "temperature": 24.0, "humidity": 80.0,
        "ph": 6.5, "rainfall": 210.0
    }
    recs = predict_top_crops(sample_input, top_k=3)
    assert len(recs) == 3
    assert "crop" in recs[0]
    assert "confidence_score" in recs[0]
    assert recs[0]["confidence_score"] >= recs[1]["confidence_score"]


def test_recommend_crops_unified():
    sample_input = {
        "N": 80.0, "P": 45.0, "K": 40.0,
        "temperature": 24.0, "humidity": 80.0,
        "ph": 6.5, "rainfall": 210.0
    }
    res = recommend_crops_unified(sample_input, top_k=3)
    assert "top_recommendations" in res
    assert len(res["top_recommendations"]) == 3
    assert "disclaimer" in res
    assert "shap_explanation" in res
