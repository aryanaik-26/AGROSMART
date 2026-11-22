"""
Unit tests for Crop Yield regression suite and prediction pipeline.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.data_loader import load_yield_data
from src.prediction import predict_crop_yield_unified
from src.yield_prediction import predict_yield


def test_load_yield_data():
    df = load_yield_data()
    assert len(df) >= 3000
    assert "Yield" in df.columns
    assert "Production" in df.columns
    assert "Crop" in df.columns


def test_predict_yield():
    sample_input = {
        "Crop": "Rice",
        "State": "Punjab",
        "Season": "Kharif",
        "Soil_Type": "Alluvial",
        "Area": 100.0,
        "Rainfall": 800.0,
        "Fertilizer_N": 120.0,
        "Fertilizer_P": 60.0,
        "Fertilizer_K": 40.0,
        "Temperature": 28.0,
        "Humidity": 75.0,
        "Soil_pH": 6.8
    }
    res = predict_yield(sample_input)
    assert res["predicted_yield_tha"] > 0
    assert res["estimated_total_production_tonnes"] > 0
    assert "prediction_interval" in res
    assert res["prediction_interval"]["lower"] <= res["predicted_yield_tha"]
    assert res["prediction_interval"]["upper"] >= res["predicted_yield_tha"]


def test_predict_crop_yield_unified():
    sample_input = {
        "Crop": "Wheat",
        "State": "Haryana",
        "Season": "Rabi",
        "Soil_Type": "Alluvial",
        "Area": 50.0,
        "Rainfall": 400.0,
        "Fertilizer_N": 100.0,
        "Fertilizer_P": 50.0,
        "Fertilizer_K": 30.0,
        "Temperature": 20.0,
        "Humidity": 65.0,
        "Soil_pH": 7.0
    }
    res = predict_crop_yield_unified(sample_input)
    assert "predicted_yield_tha" in res
    assert "shap_explanation" in res
