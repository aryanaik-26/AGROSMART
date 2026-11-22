# AGROSMART AI - Dataset Sources & Setup Guide

This directory houses the datasets utilized by the **AGROSMART AI** system across its three primary prediction modules.

---

## 1. Plant Disease Dataset (PlantVillage)
- **Source**: [Kaggle Plant Disease Dataset (Emma Rex)](https://www.kaggle.com/datasets/emmarex/plantdisease)
- **Description**: Contains over 54,000 high-resolution leaf images spanning 38 classes of healthy and diseased crops (Apple, Corn, Grape, Potato, Tomato, Pepper, Peach, etc.).
- **Local Path**: `data/plantvillage/`
- **Manual Setup**:
  1. Download the archive from Kaggle.
  2. Extract the folders directly into `data/plantvillage/` such that subdirectories are named after each class (e.g., `data/plantvillage/Tomato___Early_blight/`).
  3. The automated loader handles dataset splits into Train (70%), Validation (15%), and Test (15%) splits with stratified grouping.
- **Offline Benchmark**:
  - The repository includes a representative image generator in `data/generate_benchmark_data.py` to allow immediate offline verification, end-to-end model training, test assertions, and Grad-CAM generation without requiring the 1.5 GB Kaggle download.

---

## 2. Crop Recommendation Dataset
- **Source**: [Kaggle Crop Recommendation Dataset by Atharva Ingle](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset)
- **Benchmark Size**: 2,200 records (100 samples per crop across 22 crops).
- **Features**:
  - `N`: Ratio of Nitrogen content in soil (kg/ha or mg/kg)
  - `P`: Ratio of Phosphorus content in soil (kg/ha or mg/kg)
  - `K`: Ratio of Potassium content in soil (kg/ha or mg/kg)
  - `temperature`: Temperature in degrees Celsius (°C)
  - `humidity`: Relative humidity in percentage (%)
  - `ph`: Soil pH level (0 - 14 scale)
  - `rainfall`: Rainfall in millimeters (mm)
  - `label`: Crop category (target)
- **Local Path**: `data/raw/crop_recommendation.csv`

---

## 3. Agricultural Yield & Climate Dataset
- **Source**: Integrated from [FAOSTAT (Food and Agriculture Organization)](https://www.fao.org/faostat/en/) and Indian Ministry of Agriculture & Farmers Welfare, enhanced with [NASA POWER Agroclimatology](https://power.larc.nasa.gov/).
- **Features**:
  - `Crop`: Crop type (Rice, Wheat, Maize, Cotton, Sugarcane, Potato, etc.)
  - `State`: State/Province (Punjab, Maharashtra, Uttar Pradesh, etc.)
  - `Season`: Kharif, Rabi, Summer, Whole Year
  - `Soil_Type`: Alluvial, Black, Red, Clayey, Sandy Loam
  - `Area`: Cultivated area in hectares (ha)
  - `Rainfall`: Annual/seasonal precipitation in mm
  - `Fertilizer_N`, `Fertilizer_P`, `Fertilizer_K`: Nutrient application rates (kg/ha)
  - `Temperature`: Mean seasonal temperature (°C)
  - `Humidity`: Mean relative humidity (%)
  - `Soil_pH`: Soil acidity/alkalinity
  - `Production`: Total harvest in tonnes (t)
  - `Yield`: Crop yield in tonnes per hectare (t/ha) [Target]
- **Local Path**: `data/raw/crop_yield.csv`

---

## 4. Quick Initialization Command
To generate the calibrated benchmark datasets and sample leaf imagery for offline verification and model training:
```bash
python data/generate_benchmark_data.py
```
