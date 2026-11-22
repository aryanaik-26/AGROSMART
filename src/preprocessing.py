"""
AGROSMART AI - Data Preprocessing & Leakage Prevention Pipeline
Implements:
1. Missing value handling (median for numeric, mode for categorical).
2. Outlier detection using IQR & Z-score with intelligent capping.
3. Feature scaling (StandardScaler fitted strictly on training data).
4. Categorical encoding (One-Hot and Label Encoders).
5. Class imbalance treatment with SMOTE.
6. Computer vision data augmentation (Rotation, Flip, Zoom, Rescaling).
7. Serialized pipeline persistence for strict inference consistency.
"""

from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import joblib

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import RANDOM_STATE, TEST_SIZE


# ==============================================================================
# 1. OUTLIER DETECTION (IQR & Z-SCORE)
# ==============================================================================
def detect_outliers_iqr(df: pd.DataFrame, columns: List[str], factor: float = 1.5) -> Dict[str, Dict[str, Any]]:
    """
    Detects outliers in specified columns using the Interquartile Range (IQR) technique.
    Formula:
        IQR = Q3 - Q1
        Lower Bound = Q1 - (factor * IQR)
        Upper Bound = Q3 + (factor * IQR)
    """
    outlier_info = {}
    for col in columns:
        if col not in df.columns or not np.issubdtype(df[col].dtype, np.number):
            continue
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - factor * iqr
        upper_bound = q3 + factor * iqr

        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outlier_info[col] = {
            "q1": float(q1),
            "q3": float(q3),
            "iqr": float(iqr),
            "lower_bound": float(lower_bound),
            "upper_bound": float(upper_bound),
            "count": int(len(outliers)),
            "percentage": float(len(outliers) / len(df) * 100.0)
        }
    return outlier_info


def detect_outliers_zscore(df: pd.DataFrame, columns: List[str], threshold: float = 3.0) -> Dict[str, Dict[str, Any]]:
    """
    Detects outliers using standardized Z-score.
    Formula:
        z = (x - mu) / sigma
    Points with |z| > threshold are flagged.
    """
    outlier_info = {}
    for col in columns:
        if col not in df.columns or not np.issubdtype(df[col].dtype, np.number):
            continue
        mean = df[col].mean()
        std = df[col].std()
        if std == 0:
            continue
        z_scores = (df[col] - mean) / std
        outliers = df[np.abs(z_scores) > threshold]
        outlier_info[col] = {
            "mean": float(mean),
            "std": float(std),
            "threshold": float(threshold),
            "count": int(len(outliers)),
            "percentage": float(len(outliers) / len(df) * 100.0)
        }
    return outlier_info


def clip_outliers_iqr(df: pd.DataFrame, columns: List[str], factor: float = 1.5) -> pd.DataFrame:
    """
    Winsorizes/clips outliers to the calculated upper and lower IQR bounds.
    """
    df_clean = df.copy()
    for col in columns:
        if col in df_clean.columns and np.issubdtype(df_clean[col].dtype, np.number):
            q1 = df_clean[col].quantile(0.25)
            q3 = df_clean[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - factor * iqr
            upper_bound = q3 + factor * iqr
            df_clean[col] = np.clip(df_clean[col], lower_bound, upper_bound)
    return df_clean


# ==============================================================================
# 2. TABULAR PREPROCESSING PIPELINE (CROP RECOMMENDATION)
# ==============================================================================
class CropRecommendationPreprocessor:
    """
    Fitted exclusively on training data to prevent data leakage.
    Handles scaling and label encoding for 22 crops.
    """
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
        self.is_fitted = False

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        X_scaled = self.scaler.fit_transform(X[self.feature_names])
        y_encoded = self.label_encoder.fit_transform(y)
        self.is_fitted = True
        return X_scaled, y_encoded

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Preprocessor has not been fitted yet!")
        return self.scaler.transform(X[self.feature_names])

    def inverse_transform_labels(self, y_encoded: np.ndarray) -> np.ndarray:
        return self.label_encoder.inverse_transform(y_encoded)

    def save(self, file_path: Path):
        file_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, file_path)

    @classmethod
    def load(cls, file_path: Path) -> "CropRecommendationPreprocessor":
        return joblib.load(file_path)


# ==============================================================================
# 3. TABULAR PREPROCESSING PIPELINE (CROP YIELD REGRESSION)
# ==============================================================================
class YieldPredictionPreprocessor:
    """
    Fitted exclusively on training data to prevent data leakage.
    Encodes categorical features (Crop, State, Season, Soil_Type)
    and standardizes numerical agronomic features.
    """
    def __init__(self):
        self.num_features = [
            "Area", "Rainfall", "Fertilizer_N", "Fertilizer_P", "Fertilizer_K",
            "Temperature", "Humidity", "Soil_pH"
        ]
        self.cat_features = ["Crop", "State", "Season", "Soil_Type"]
        self.scaler = StandardScaler()
        self.onehot = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        self.imputer_num = SimpleImputer(strategy="median")
        self.imputer_cat = SimpleImputer(strategy="most_frequent")
        self.transformed_feature_names: List[str] = []
        self.is_fitted = False

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        # Imputation
        X_num = self.imputer_num.fit_transform(df[self.num_features])
        X_cat = self.imputer_cat.fit_transform(df[self.cat_features])

        # Transformations
        X_num_scaled = self.scaler.fit_transform(X_num)
        X_cat_encoded = self.onehot.fit_transform(X_cat)

        cat_names = self.onehot.get_feature_names_out(self.cat_features).tolist()
        self.transformed_feature_names = self.num_features + cat_names

        X_processed = np.hstack([X_num_scaled, X_cat_encoded])
        self.is_fitted = True
        return X_processed

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Preprocessor has not been fitted yet!")

        X_num = self.imputer_num.transform(df[self.num_features])
        X_cat = self.imputer_cat.transform(df[self.cat_features])

        X_num_scaled = self.scaler.transform(X_num)
        X_cat_encoded = self.onehot.transform(X_cat)

        return np.hstack([X_num_scaled, X_cat_encoded])

    def save(self, file_path: Path):
        file_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, file_path)

    @classmethod
    def load(cls, file_path: Path) -> "YieldPredictionPreprocessor":
        return joblib.load(file_path)


# ==============================================================================
# 4. SMOTE RESAMPLING FOR CLASS IMBALANCE
# ==============================================================================
def apply_smote(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """
    Applies Synthetic Minority Over-sampling Technique (SMOTE) to ensure equal representation.
    Returns:
        X_resampled, y_resampled, class_distribution_before_after
    """
    unique_before, counts_before = np.unique(y, return_counts=True)
    stats_before = dict(zip(unique_before.tolist(), counts_before.tolist()))

    smote = SMOTE(random_state=RANDOM_STATE)
    X_res, y_res = smote.fit_resample(X, y)

    unique_after, counts_after = np.unique(y_res, return_counts=True)
    stats_after = dict(zip(unique_after.tolist(), counts_after.tolist()))

    summary = {
        "before": stats_before,
        "after": stats_after
    }
    return X_res, y_res, summary
