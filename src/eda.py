"""
AGROSMART AI - Exploratory Data Analysis (EDA) Module
Generates comprehensive agronomic visualizations and saves them to reports/graphs/:
1. Correlation Heatmaps
2. Rainfall, Temperature, and Soil Nutrient Distributions
3. Yield Boxplots and Crop Comparisons
4. Outlier Analysis Plots
5. Class Imbalance Charts (Pre vs Post SMOTE)
6. Disease Leaf Image Sample Galleries
"""

from pathlib import Path
from typing import List, Optional, Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from PIL import Image

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import GRAPHS_DIR, PLANTVILLAGE_DIR, THEME_COLORS

# Set universal aesthetic styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["figure.dpi"] = 300


def save_or_show(fig, save_path: Optional[Path] = None):
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.close(fig)
    else:
        plt.show()


def plot_correlation_heatmap(df: pd.DataFrame, title: str, save_path: Optional[Path] = None):
    """Generates an annotated correlation heatmap for numeric columns."""
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = sns.diverging_palette(135, 15, as_cmap=True)
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap=cmap, vmin=-1, vmax=1,
        linewidths=0.8, cbar_kws={"shrink": 0.8}, ax=ax
    )
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15, color=THEME_COLORS["primary_dark"])
    save_or_show(fig, save_path)


def plot_nutrient_distributions(df: pd.DataFrame, save_path: Optional[Path] = None):
    """Plots kernel density and histograms for N, P, K soil nutrients."""
    nutrients = ["N", "P", "K"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    colors = [THEME_COLORS["primary"], THEME_COLORS["earth_brown"], THEME_COLORS["primary_light"]]
    for i, col in enumerate(nutrients):
        if col in df.columns:
            sns.histplot(df[col], kde=True, ax=axes[i], color=colors[i], bins=25, edgecolor="black")
            axes[i].set_title(f"Soil {col} Distribution", fontsize=12, fontweight="bold")
            axes[i].set_xlabel(f"{col} Content (mg/kg or kg/ha)")
            axes[i].set_ylabel("Frequency")

    plt.tight_layout()
    save_or_show(fig, save_path)


def plot_climate_distributions(df: pd.DataFrame, save_path: Optional[Path] = None):
    """Plots distributions of temperature, humidity, and rainfall."""
    climates = [("temperature", "Temperature (°C)"), ("humidity", "Relative Humidity (%)"), ("rainfall", "Rainfall (mm)")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    palette = [THEME_COLORS["warning"], THEME_COLORS["accent"], THEME_COLORS["primary"]]
    for i, (col, label) in enumerate(climates):
        if col in df.columns:
            sns.histplot(df[col], kde=True, ax=axes[i], color=palette[i], bins=25, edgecolor="black")
            axes[i].set_title(f"{label} Distribution", fontsize=12, fontweight="bold")
            axes[i].set_xlabel(label)
            axes[i].set_ylabel("Frequency")

    plt.tight_layout()
    save_or_show(fig, save_path)


def plot_crop_yield_comparison(df: pd.DataFrame, save_path: Optional[Path] = None):
    """Plots comparative boxplots of crop yields across different crops."""
    if "Crop" not in df.columns or "Yield" not in df.columns:
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    order = df.groupby("Crop")["Yield"].median().sort_values(ascending=False).index

    sns.boxplot(
        x="Crop", y="Yield", data=df, order=order,
        palette="Greens_r", ax=ax, flierprops={"marker": "o", "markersize": 4, "alpha": 0.5}
    )
    ax.set_title("Agricultural Yield Distribution by Crop (t/ha)", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Crop Type", fontweight="bold")
    ax.set_ylabel("Yield (Tonnes per Hectare)", fontweight="bold")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    save_or_show(fig, save_path)


def plot_outliers_boxplots(df: pd.DataFrame, features: List[str], title: str, save_path: Optional[Path] = None):
    """Visualizes outliers using side-by-side boxplots."""
    valid_features = [f for f in features if f in df.columns and np.issubdtype(df[f].dtype, np.number)]
    if not valid_features:
        return

    fig, axes = plt.subplots(1, len(valid_features), figsize=(len(valid_features) * 3, 4.5))
    if len(valid_features) == 1:
        axes = [axes]

    for i, feat in enumerate(valid_features):
        sns.boxplot(y=df[feat], ax=axes[i], color=THEME_COLORS["primary_light"])
        axes[i].set_title(feat, fontsize=11, fontweight="bold")
        axes[i].set_ylabel("Value")

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    save_or_show(fig, save_path)


def plot_missing_values(df: pd.DataFrame, title: str, save_path: Optional[Path] = None):
    """Visualizes missing value frequencies across columns."""
    missing = df.isnull().sum()
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x=missing.index, y=missing.values, color=THEME_COLORS["warning"], ax=ax)
    ax.set_title(f"Missing Values: {title}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Missing Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    save_or_show(fig, save_path)


def plot_class_balance_comparison(before_dict: Dict[Any, int], after_dict: Dict[Any, int], save_path: Optional[Path] = None):
    """Visualizes class distributions before and after SMOTE."""
    df_compare = pd.DataFrame({
        "Class": list(before_dict.keys()) + list(after_dict.keys()),
        "Count": list(before_dict.values()) + list(after_dict.values()),
        "Stage": ["Before SMOTE"] * len(before_dict) + ["After SMOTE"] * len(after_dict)
    })

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=df_compare, x="Class", y="Count", hue="Stage", palette=["#E76F51", "#2D6A4F"], ax=ax)
    ax.set_title("Class Distribution Before & After SMOTE Balancing", fontsize=13, fontweight="bold")
    ax.set_xlabel("Target Class")
    ax.set_ylabel("Sample Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    save_or_show(fig, save_path)


def plot_disease_image_gallery(data_dir: Optional[Path] = None, save_path: Optional[Path] = None):
    """Generates a composite visual gallery of leaf samples across classes."""
    target_dir = data_dir or PLANTVILLAGE_DIR
    classes = [d for d in target_dir.iterdir() if d.is_dir() and len(list(d.glob("*.jpg"))) > 0][:8]
    if not classes:
        return

    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    axes = axes.flatten()

    for idx, cls_dir in enumerate(classes):
        imgs = list(cls_dir.glob("*.jpg"))
        if imgs:
            img = Image.open(imgs[0])
            axes[idx].imshow(img)
            name = cls_dir.name.replace("___", " - ").replace("_", " ")
            axes[idx].set_title(name[:22], fontsize=9, fontweight="bold")
            axes[idx].axis("off")

    plt.suptitle("Plant Disease Leaf Samples (PlantVillage)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_or_show(fig, save_path)


def generate_all_eda_visualizations():
    """Master workflow generating and saving all required EDA plots."""
    from src.data_loader import load_crop_recommendation_data, load_yield_data

    print("--- Generating Comprehensive EDA Figures ---")
    df_crop = load_crop_recommendation_data()
    df_yield = load_yield_data()

    plot_correlation_heatmap(df_crop, "Crop Recommendation Feature Correlations", GRAPHS_DIR / "crop_recommendation_corr.png")
    plot_correlation_heatmap(df_yield, "Crop Yield Feature Correlations", GRAPHS_DIR / "crop_yield_corr.png")
    plot_nutrient_distributions(df_crop, GRAPHS_DIR / "soil_nutrient_distributions.png")
    plot_climate_distributions(df_crop, GRAPHS_DIR / "climate_distributions.png")
    plot_crop_yield_comparison(df_yield, GRAPHS_DIR / "crop_yield_comparison.png")
    plot_outliers_boxplots(df_crop, ["N", "P", "K", "rainfall", "temperature"], "Crop Recommendation Outliers", GRAPHS_DIR / "crop_outliers.png")
    plot_missing_values(df_crop, "Crop Recommendation", GRAPHS_DIR / "crop_missing.png")
    plot_disease_image_gallery(PLANTVILLAGE_DIR, GRAPHS_DIR / "disease_samples_gallery.png")

    print(f"All EDA visualizations successfully saved to: {GRAPHS_DIR}")


if __name__ == "__main__":
    generate_all_eda_visualizations()
