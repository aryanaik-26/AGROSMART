"""
AGROSMART AI - Agronomic Feature Engineering Module
Calculates bio-physical and chemical indices:
1. NPK stoichoimetric ratios (N/P, N/K, P/K, Total NPK).
2. De Martonne Aridity Index (Precipitation effectiveness).
3. Temperature-Humidity Index (THI) for agro-climatic stress.
4. Soil Reaction Deviation (|pH - 7.0|).
"""

import pandas as pd
import numpy as np


def engineer_crop_recommendation_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Augments soil nutrient and meteorological data with domain-specific agro-indices.
    """
    df_eng = df.copy()

    eps = 1e-5
    # Stoichiometric nutrient balance
    df_eng["Total_NPK"] = df_eng["N"] + df_eng["P"] + df_eng["K"]
    df_eng["N_P_Ratio"] = df_eng["N"] / (df_eng["P"] + eps)
    df_eng["N_K_Ratio"] = df_eng["N"] / (df_eng["K"] + eps)
    df_eng["P_K_Ratio"] = df_eng["P"] / (df_eng["K"] + eps)

    # De Martonne Aridity Index: I = P / (T + 10)
    df_eng["Aridity_Index"] = df_eng["rainfall"] / (df_eng["temperature"] + 10.0)

    # Temperature Humidity Stress Index
    df_eng["THI"] = (
        (1.8 * df_eng["temperature"] + 32.0)
        - (0.55 - 0.0055 * df_eng["humidity"])
        * (1.8 * df_eng["temperature"] - 26.0)
    )

    # Soil pH Deviation from neutrality (7.0)
    df_eng["pH_Deviation"] = np.abs(df_eng["ph"] - 7.0)

    return df_eng


def engineer_yield_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Augments crop production and weather features with agronomic intensity measures.
    """
    df_eng = df.copy()
    eps = 1e-5

    # Total Fertilizer application
    df_eng["Total_Fertilizer"] = df_eng["Fertilizer_N"] + df_eng["Fertilizer_P"] + df_eng["Fertilizer_K"]
    df_eng["N_P_Fert_Ratio"] = df_eng["Fertilizer_N"] / (df_eng["Fertilizer_P"] + eps)

    # Rain to Area moisture index
    df_eng["Rain_Per_Hectare"] = df_eng["Rainfall"] / (df_eng["Area"] + eps)

    # Aridity index
    df_eng["Aridity_Index"] = df_eng["Rainfall"] / (df_eng["Temperature"] + 10.0)

    return df_eng
