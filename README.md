# 🌱 AGROSMART AI – Intelligent Agriculture Prediction System

> **A comprehensive Machine Learning, Deep Learning, and Explainable AI (XAI) precision agriculture system built from scratch.**

![Python 3.11](https://img.shields.io/badge/Python-3.11-green.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-orange.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.9-blue.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-3.2-red.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.66-FF4B4B.svg)
![SHAP & Grad-CAM](https://img.shields.io/badge/XAI-SHAP%20%2B%20Grad--CAM-purple.svg)

---

## 📌 1. Project Overview & Objective

**AGROSMART AI** is an end-to-end intelligent agriculture platform designed to tackle three core challenges in precision farming:
1. **Module 1: Crop Disease Detection**: Upload an image of an affected leaf blade. Three deep learning vision backbones (**Custom CNN**, **MobileNetV2**, and **EfficientNetB0**) classify the pathogen across 38 PlantVillage classes, generate **Grad-CAM heatmaps**, and retrieve verified biological & chemical management protocols.
2. **Module 2: Crop Yield Prediction**: Input soil nutrients (N, P, K), localized climate (temperature, humidity, precipitation, pH), and cultivated acreage. An ensemble of 8 regression models (**Random Forest**, **XGBoost**, **Gradient Boosting**, **SVR**, **Ridge**, **Stacking Regressor**) forecasts expected harvest yield in **tonnes per hectare (t/ha)** with **95% statistical confidence intervals** and **SHAP waterfall attributions**.
3. **Module 3: Smart Crop Recommendation**: Input chemical soil fertility and localized micro-climate to receive probabilistic recommendations for the **top three optimal crops** out of 22 benchmark agricultural species, complete with **SHAP feature explanations** and agronomic guidance.

---

## 🏗️ 2. Project Architecture & Directory Structure

```
AGROSMART-AI/
├── app.py                             # Main Streamlit Dashboard & KPI Hub
├── config.py                          # Centralized paths, taxonomy, disease encyclopedia, theme colors
├── requirements.txt                   # Pinned compatible Python 3.11 dependencies
├── README.md                          # Comprehensive setup & viva guide
├── data/
│   ├── raw/                           # Raw datasets (crop_recommendation.csv, crop_yield.csv)
│   ├── processed/                     # Preprocessed partitions
│   ├── plantvillage/                  # PlantVillage leaf images organized by disease class
│   ├── generate_benchmark_data.py     # Script to generate calibrated benchmark datasets & leaf images
│   └── README.md                      # Dataset provenance and download instructions
├── notebooks/
│   ├── 01_eda.ipynb                   # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb         # Outliers, Scaling, SMOTE & Leakage Prevention
│   ├── 03_crop_recommendation.ipynb   # 8-Algorithm Classification Suite
│   ├── 04_yield_prediction.ipynb      # 8-Algorithm Regression Suite & Prediction Intervals
│   ├── 05_disease_detection.ipynb     # CNN, MobileNetV2, EfficientNetB0 & Grad-CAM
│   └── 06_model_evaluation.ipynb      # Multi-model benchmarking & SHAP interpretability
├── src/
│   ├── data_loader.py                 # Robust CSV & image loaders
│   ├── preprocessing.py               # IQR, Z-score, StandardScaler, SMOTE, OneHotEncoder
│   ├── eda.py                         # High-resolution figure generators saved to reports/
│   ├── feature_engineering.py         # Agronomic indices (NPK ratios, aridity, THI)
│   ├── feature_selection.py           # SelectKBest (ANOVA), RFE, and PCA
│   ├── crop_recommendation.py         # Classification trainer & top-3 predictor
│   ├── yield_prediction.py            # Regression trainer & t/ha predictor
│   ├── disease_detection.py           # Custom CNN, MobileNetV2, EfficientNetB0 architectures
│   ├── ensemble.py                    # StackingClassifier & StackingRegressor meta-models
│   ├── optimization.py                # GridSearchCV & RandomizedSearchCV pipelines
│   ├── evaluation.py                  # Accuracy, F1, RMSE, R², confusion matrix, latency
│   ├── explainability.py              # Grad-CAM conv hooks & SHAP game-theoretic explainers
│   ├── prediction.py                  # Unified inference facade for UI
│   └── model_training.py              # Master CLI training runner
├── pages/
│   ├── 1_Crop_Disease_Detection.py    # Leaf image upload, Grad-CAM, & disease treatment
│   ├── 2_Crop_Yield_Prediction.py     # Yield (t/ha), total harvest, 95% interval, & SHAP
│   ├── 3_Crop_Recommendation.py       # Top-3 crop ranking, confidence gauges, & SHAP
│   ├── 4_Dataset_Explorer.py          # Interactive dataset preview, distributions & gallery
│   ├── 5_Preprocessing.py             # Interactive Outliers, SMOTE, SelectKBest, & PCA
│   ├── 6_Model_Training.py            # Live re-training studio for ML & Deep Learning
│   ├── 7_Model_Comparison.py          # Side-by-side benchmark tables, bar charts, & curves
│   ├── 8_Explainable_AI.py            # Mathematical XAI theory, SHAP simulator, & Grad-CAM
│   └── 9_Reports.py                   # One-click metric JSONs, graph zip, & Markdown summary
├── models/
│   ├── crop_recommendation/           # Serialized classifiers & preprocessors (.joblib)
│   ├── yield_prediction/              # Serialized regressors & preprocessors (.joblib)
│   └── disease_detection/             # Saved Keras models (.keras) & class_indices.json
├── reports/
│   ├── graphs/                        # Correlation heatmaps, learning curves, histograms
│   ├── metrics/                       # JSON metrics across all 19 trained models
│   └── generated_reports/             # Technical summary exports
├── research_paper/
│   └── Research_Paper_AGROSMART_AI.md # Full peer-review quality academic research paper
└── tests/
    ├── test_preprocessing.py          # Preprocessing & outlier unit tests
    ├── test_crop_recommendation.py    # Crop recommendation unit tests
    ├── test_yield_prediction.py       # Yield prediction unit tests
    └── test_disease_detection.py      # Vision model & Grad-CAM unit tests
```

---

## ⚡ 3. Quick Start & Execution Commands

### Step 1: Install Dependencies
```bash
python -m pip install -r requirements.txt
```

### Step 2: Initialize Calibrated Benchmark Datasets
```bash
python data/generate_benchmark_data.py
```

### Step 3: Train All Machine Learning & Deep Learning Models
```bash
python src/model_training.py --epochs 3
```

### Step 4: Run Automated Test Suite
```bash
python -m pytest tests/
```

### Step 5: Launch the AGROSMART AI Web Application
```bash
streamlit run app.py
```

---

## 🧪 4. Machine Learning & Deep Learning Algorithms Implemented

### Multi-Class Classification (Crop Recommendation)
1. **Multinomial Logistic Regression** (Baseline linear classifier)
2. **Decision Tree Classifier** (CART with Gini impurity splitting)
3. **Random Forest Classifier** (100 bootstrap estimators with feature subsampling)
4. **Support Vector Machine (SVC)** (RBF non-linear kernel mapping)
5. **K-Nearest Neighbors (KNN)** (Distance-weighted neighborhood voting)
6. **XGBoost Classifier** (Gradient boosted regularized decision trees)
7. **Gradient Boosting Classifier** (Sequential gradient descent boosting)
8. **Stacking Ensemble Classifier** (RF + XGBoost + SVC with Logistic Regression meta-learner)

### Multi-Variable Regression (Crop Yield Prediction)
1. **Ridge Regression** ($L_2$ regularized linear model)
2. **Decision Tree Regressor** (Mean squared error variance reduction)
3. **Random Forest Regressor** (Ensemble variance reduction)
4. **Support Vector Regressor (SVR)** ($\epsilon$-insensitive tube loss)
5. **K-Nearest Neighbors Regressor** (Local metric averaging)
6. **XGBoost Regressor** (Second-order Taylor expansion tree boosting)
7. **Gradient Boosting Regressor** (Stochastic gradient boosting)
8. **Stacking Ensemble Regressor** (RF + XGBoost + GB with Ridge meta-regressor)

### Deep Learning Foliar Vision (Plant Disease Detection)
1. **Custom 4-Block CNN**: Built from scratch with Conv2D, Batch Normalization, Max Pooling, Dropout (0.3/0.4), Dense layers, and Softmax output.
2. **MobileNetV2**: Transfer learning with ImageNet weights, inverted residuals, and linear bottlenecks.
3. **EfficientNetB0**: Transfer learning with compound depth/width/resolution scaling.

---

## 💡 5. Explainable AI (XAI)

- **Grad-CAM (Gradient-Weighted Class Activation Mapping)**: Backpropagates gradients from the predicted disease score directly into the final convolutional feature maps, visually highlighting leaf necrosis and pustules with a jet colormap overlay.
- **SHAP (Shapley Additive Explanations)**: Grounded in cooperative game theory, calculating the exact additive impact (+/- tonnes/ha or class probability) contributed by rainfall, temperature, soil nitrogen, phosphorus, potassium, and soil pH.

---

## 🎓 6. College Viva Q&A Guide

**Q1: How do you prevent data leakage during preprocessing?**  
*Answer:* All transformers (`StandardScaler`, `SimpleImputer`, `OneHotEncoder`) are fitted strictly on the training partition ($D_{\text{train}}$) and frozen. The test set and real-time inference inputs only call `.transform()`, ensuring no statistical information from the test set influences model parameters.

**Q2: Why use both Custom CNN and Pretrained Models (MobileNetV2, EfficientNetB0)?**  
*Answer:* To evaluate the trade-off between domain-specific feature learning from scratch versus inductive transfer learning from generic visual features (edges, textures) learned on ImageNet.

**Q3: How does Grad-CAM differ from ordinary saliency maps?**  
*Answer:* Plain saliency maps compute input-pixel gradients, which are notoriously noisy and class-agnostic. Grad-CAM computes gradients at the deepest convolutional layer (which retains high-level semantic abstractions) and uses global average pooling and ReLU to capture only features positively supporting the selected diagnosis.

**Q4: Why distinguish crop recommendations from guaranteed agricultural outcomes?**  
*Answer:* Model recommendations are probabilistic decision-support scores based on historical benchmark distributions. Actual farm productivity is subject to micro-climatic shocks, pest infestations, and seed variety differences, necessitating clear disclaimers for responsible AI deployment.
