"""
Unit tests for data preprocessing, outlier detection, scaling, and SMOTE.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.preprocessing import (
    detect_outliers_iqr, detect_outliers_zscore, clip_outliers_iqr,
    CropRecommendationPreprocessor, YieldPredictionPreprocessor, apply_smote
)


def test_detect_outliers_iqr():
    data = [10.0, 12.0, 11.0, 10.5, 11.5, 100.0]  # 100 is outlier
    df = pd.DataFrame({"feat": data})
    res = detect_outliers_iqr(df, ["feat"], factor=1.5)
    assert "feat" in res
    assert res["feat"]["count"] == 1
    assert res["feat"]["upper_bound"] < 100.0


def test_detect_outliers_zscore():
    data = [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 100.0]
    df = pd.DataFrame({"feat": data})
    res = detect_outliers_zscore(df, ["feat"], threshold=2.0)
    assert res["feat"]["count"] >= 1


def test_clip_outliers_iqr():
    data = [1.0, 2.0, 3.0, 4.0, 50.0]
    df = pd.DataFrame({"feat": data})
    clipped_df = clip_outliers_iqr(df, ["feat"], factor=1.5)
    assert clipped_df["feat"].max() < 50.0


def test_crop_recommendation_preprocessor():
    df = pd.DataFrame({
        "N": [80, 40, 20],
        "P": [40, 50, 60],
        "K": [30, 40, 50],
        "temperature": [22.0, 24.0, 26.0],
        "humidity": [80.0, 82.0, 84.0],
        "ph": [6.5, 6.7, 6.9],
        "rainfall": [200.0, 150.0, 100.0],
        "label": ["rice", "maize", "cotton"]
    })
    prep = CropRecommendationPreprocessor()
    X_scaled, y_encoded = prep.fit_transform(df, df["label"])

    assert X_scaled.shape == (3, 7)
    assert len(y_encoded) == 3
    assert prep.is_fitted

    # Test transform on unseen single sample
    sample_df = pd.DataFrame([{
        "N": 85, "P": 42, "K": 35,
        "temperature": 23.0, "humidity": 81.0,
        "ph": 6.6, "rainfall": 190.0
    }])
    X_test_scaled = prep.transform(sample_df)
    assert X_test_scaled.shape == (1, 7)


def test_yield_preprocessor():
    df = pd.DataFrame({
        "Crop": ["Rice", "Wheat"],
        "State": ["Punjab", "Haryana"],
        "Season": ["Kharif", "Rabi"],
        "Soil_Type": ["Alluvial", "Clayey"],
        "Area": [100.0, 150.0],
        "Rainfall": [800.0, 400.0],
        "Fertilizer_N": [120.0, 100.0],
        "Fertilizer_P": [60.0, 50.0],
        "Fertilizer_K": [40.0, 30.0],
        "Temperature": [28.0, 20.0],
        "Humidity": [75.0, 60.0],
        "Soil_pH": [6.8, 7.2]
    })
    prep = YieldPredictionPreprocessor()
    X_processed = prep.fit_transform(df)
    assert X_processed.shape[0] == 2
    assert X_processed.shape[1] > 8  # 8 numeric + one-hot categories


def test_smote():
    X = np.array([
        [1.0, 2.0], [1.1, 2.1], [0.9, 1.9], [1.2, 2.2], [1.0, 2.0], [1.1, 1.9],
        [10.0, 20.0], [10.2, 20.1]  # Minority
    ])
    y = np.array([0, 0, 0, 0, 0, 0, 1, 1])

    from imblearn.over_sampling import SMOTE
    smote = SMOTE(k_neighbors=1, random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    assert sum(y_res == 1) == sum(y_res == 0)
