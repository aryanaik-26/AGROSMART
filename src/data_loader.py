"""
AGROSMART AI - Data Loader Module
Handles loading of tabular datasets (Crop Recommendation, Yield Prediction)
and image datasets (PlantVillage) with robust validation, integrity checks,
and metadata extraction.
"""

from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import pandas as pd
import numpy as np

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import (
    RAW_DATA_DIR, PLANTVILLAGE_DIR, IMAGE_SIZE, BATCH_SIZE, RANDOM_STATE
)


def load_crop_recommendation_data(filepath: Optional[Path] = None) -> pd.DataFrame:
    """
    Loads the benchmark Crop Recommendation dataset.
    Validates required columns: N, P, K, temperature, humidity, ph, rainfall, label.
    """
    path = filepath or (RAW_DATA_DIR / "crop_recommendation.csv")
    if not path.exists():
        raise FileNotFoundError(
            f"Crop recommendation dataset not found at: {path}. "
            "Please run 'python data/generate_benchmark_data.py' or provide a valid CSV."
        )

    df = pd.read_csv(path)
    expected_cols = {"N", "P", "K", "temperature", "humidity", "ph", "rainfall", "label"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"Dataset at {path} is missing required columns: {missing}")

    return df


def load_yield_data(filepath: Optional[Path] = None) -> pd.DataFrame:
    """
    Loads the Agricultural Crop Yield dataset.
    Validates numeric and categorical features and target column 'Yield'.
    """
    path = filepath or (RAW_DATA_DIR / "crop_yield.csv")
    if not path.exists():
        raise FileNotFoundError(
            f"Crop yield dataset not found at: {path}. "
            "Please run 'python data/generate_benchmark_data.py' or provide a valid CSV."
        )

    df = pd.read_csv(path)
    expected_cols = {
        "Crop", "State", "Season", "Soil_Type", "Area", "Rainfall",
        "Fertilizer_N", "Fertilizer_P", "Fertilizer_K", "Temperature",
        "Humidity", "Soil_pH", "Production", "Yield"
    }
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"Yield dataset at {path} is missing required columns: {missing}")

    return df


def inspect_dataset_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes quick statistical overview, missing counts, duplicate counts,
    and memory footprint.
    """
    summary = {
        "num_rows": len(df),
        "num_cols": len(df.columns),
        "duplicates": int(df.duplicated().sum()),
        "missing_values": df.isnull().sum().to_dict(),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "numerical_stats": df.describe().to_dict()
    }
    return summary


def get_available_disease_classes(data_dir: Optional[Path] = None) -> list:
    """
    Discovers available class subdirectories with leaf images.
    """
    target_dir = data_dir or PLANTVILLAGE_DIR
    if not target_dir.exists():
        return []

    classes = [
        d.name for d in target_dir.iterdir()
        if d.is_dir() and len(list(d.glob("*.jpg")) + list(d.glob("*.png")) + list(d.glob("*.JPG"))) > 0
    ]
    return sorted(classes)
