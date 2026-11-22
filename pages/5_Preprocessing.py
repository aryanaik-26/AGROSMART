"""
AGROSMART AI - Page 5: Data Preprocessing & Leakage Prevention Pipeline
Enhanced UI & Visual Graphics Edition
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from config import THEME_COLORS
from src.data_loader import load_crop_recommendation_data, load_yield_data
from src.preprocessing import detect_outliers_iqr, detect_outliers_zscore, clip_outliers_iqr, apply_smote
from src.feature_selection import perform_select_k_best, perform_pca_reduction
from src.ui_theme import inject_custom_css, render_page_header, render_sidebar_branding

st.set_page_config(page_title="Data Preprocessing | AGROSMART AI", page_icon="⚙️", layout="wide")

# Inject Design System
inject_custom_css()
render_sidebar_branding()

# Render High-Tech Page Header
render_page_header(
    title="Data Preprocessing & Feature Engineering",
    subtitle="Demonstrate data transformations with zero leakage: Outlier winsorization (IQR vs Z-Score), SMOTE synthetic resampling, ANOVA F-Test SelectKBest, and PCA orthogonal projections.",
    icon="⚙️",
    badge="DATA ENGINEERING"
)

target_dataset = st.selectbox(
    "Select Target Pipeline to Demonstrate:",
    ["Crop Recommendation Tabular Pipeline", "Agricultural Yield Regression Pipeline"]
)

if "Crop" in target_dataset:
    df_raw = load_crop_recommendation_data()
    feature_cols = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
    target_col = "label"
else:
    df_raw = load_yield_data()
    feature_cols = ["Area", "Rainfall", "Fertilizer_N", "Fertilizer_P", "Fertilizer_K", "Temperature", "Humidity", "Soil_pH"]
    target_col = "Yield"

tab_outliers, tab_smote, tab_selection, tab_pca = st.tabs([
    "🎯 1. Outlier Detection (IQR vs Z-Score)",
    "⚖️ 2. SMOTE Class Balancing",
    "✨ 3. Feature Selection (SelectKBest)",
    "📉 4. PCA Dimensionality Reduction"
])

# 1. OUTLIER DETECTION
with tab_outliers:
    st.markdown("""
    <div style="background: white; border: 1px solid #E2E8F0; border-radius: 16px; padding: 1.4rem; margin-top: 0.8rem; margin-bottom: 1.2rem;">
        <h4 style="color: #064E3B; margin-top: 0;">🎯 Outlier Identification & Winsorization</h4>
        <p style="font-size: 0.88rem; color: #475569; margin: 0; line-height: 1.55;">
            Outliers distort regression hyperplanes and distance-based neighborhood metrics (KNN, SVM). We support both the <strong>Interquartile Range (IQR)</strong> rule and <strong>Z-Score</strong> standardization.
        </p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns([1, 1.2], gap="medium")
    with c1:
        selected_feat = st.selectbox("Select Feature for Outlier Analysis:", feature_cols)
        iqr_factor = st.slider("IQR Multiplier ($Q_3 + k \\times IQR$):", 1.0, 3.0, 1.5, 0.1)

        iqr_stats = detect_outliers_iqr(df_raw, [selected_feat], factor=iqr_factor)[selected_feat]
        z_stats = detect_outliers_zscore(df_raw, [selected_feat], threshold=3.0)[selected_feat]

        st.markdown(f"""
        <div style="background: #F8FAF9; border: 1px solid #E2E8F0; border-left: 4px solid #10B981; border-radius: 12px; padding: 1.2rem;">
            <div style="font-size: 0.82rem; font-weight: 700; color: #065F46; text-transform: uppercase; margin-bottom: 0.4rem;">
                STATISTICAL DISPERSION ({selected_feat})
            </div>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; font-size: 0.88rem; color: #334155;">
                <div><b>Q1 (25th):</b> {iqr_stats['q1']:.2f}</div>
                <div><b>Q3 (75th):</b> {iqr_stats['q3']:.2f}</div>
                <div><b>IQR:</b> {iqr_stats['iqr']:.2f}</div>
                <div><b>Lower Bound:</b> {iqr_stats['lower_bound']:.2f}</div>
                <div><b>Upper Bound:</b> {iqr_stats['upper_bound']:.2f}</div>
                <div style="color: #EF4444; font-weight: 700;"><b>Outliers:</b> {iqr_stats['count']} ({iqr_stats['percentage']:.2f}%)</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        fig_box = px.box(
            df_raw, y=selected_feat, points="all",
            title=f"Boxplot Dispersion: {selected_feat}",
            color_discrete_sequence=["#10B981"]
        )
        fig_box.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Plus Jakarta Sans", color="#0F291E")
        )
        st.plotly_chart(fig_box, use_container_width=True)

# 2. SMOTE BALANCING
with tab_smote:
    st.markdown("""
    <div style="background: white; border: 1px solid #E2E8F0; border-radius: 16px; padding: 1.4rem; margin-top: 0.8rem; margin-bottom: 1.2rem;">
        <h4 style="color: #064E3B; margin-top: 0;">⚖️ SMOTE (Synthetic Minority Over-sampling Technique)</h4>
        <p style="font-size: 0.88rem; color: #475569; margin: 0; line-height: 1.55;">
            SMOTE synthetically generates interpolated feature vectors for minority classes along the line segments connecting $k$-nearest neighbors:
            $$\\mathbf{x}_{\\text{new}} = \\mathbf{x}_i + \\lambda (\\mathbf{x}_{zi} - \\mathbf{x}_i), \\quad \\lambda \\in [0, 1]$$
        </p>
    </div>
    """, unsafe_allow_html=True)

    if target_col == "label":
        df_imbalanced = pd.concat([
            df_raw[df_raw["label"] == "rice"].sample(100),
            df_raw[df_raw["label"] == "maize"].sample(40),
            df_raw[df_raw["label"] == "cotton"].sample(25),
            df_raw[df_raw["label"] == "coffee"].sample(15)
        ])

        X_imb = df_imbalanced[feature_cols].values
        y_imb = df_imbalanced["label"].values

        X_res, y_res, smote_stats = apply_smote(X_imb, y_imb)

        c_s1, c_s2 = st.columns(2)
        with c_s1:
            df_b = pd.DataFrame(list(smote_stats["before"].items()), columns=["Crop", "Samples"])
            fig_b = px.bar(
                df_b, x="Crop", y="Samples",
                title="Class Imbalance (Before SMOTE)",
                color_discrete_sequence=["#EF4444"]
            )
            fig_b.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_b, use_container_width=True)

        with c_s2:
            df_a = pd.DataFrame(list(smote_stats["after"].items()), columns=["Crop", "Samples"])
            fig_a = px.bar(
                df_a, x="Crop", y="Samples",
                title="Balanced Representation (After SMOTE)",
                color_discrete_sequence=["#10B981"]
            )
            fig_a.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_a, use_container_width=True)
    else:
        st.info("SMOTE is designed for discrete classification labels (e.g. Crop Recommendation). For Yield Regression, target transformations (log/quantile) are applied.")

# 3. FEATURE SELECTION
with tab_selection:
    st.markdown("""
    <div style="background: white; border: 1px solid #E2E8F0; border-radius: 16px; padding: 1.4rem; margin-top: 0.8rem; margin-bottom: 1.2rem;">
        <h4 style="color: #064E3B; margin-top: 0;">✨ Univariate Feature Selection (SelectKBest)</h4>
        <p style="font-size: 0.88rem; color: #475569; margin: 0; line-height: 1.55;">
            Evaluates statistical dependency using ANOVA F-Value scoring to extract the most predictive agronomic subset.
        </p>
    </div>
    """, unsafe_allow_html=True)

    k_num = st.slider("Select Top K Features:", min_value=2, max_value=len(feature_cols), value=4)

    X_mat = df_raw[feature_cols].values
    if target_col == "label":
        from sklearn.preprocessing import LabelEncoder
        y_enc = LabelEncoder().fit_transform(df_raw[target_col])
        mode = "classification"
    else:
        y_enc = df_raw[target_col].values
        mode = "regression"

    res_kbest = perform_select_k_best(X_mat, y_enc, feature_names=feature_cols, k=k_num, mode=mode)

    st.markdown(f"""
    <div style="background: #ECFDF5; border: 1px solid #A7F3D0; border-radius: 12px; padding: 1rem; margin-bottom: 1rem; color: #065F46; font-weight: 600;">
        ✨ <strong>Top {k_num} Selected Features:</strong> {', '.join(res_kbest['selected_features'])}
    </div>
    """, unsafe_allow_html=True)

    df_scores = pd.DataFrame(list(res_kbest["feature_scores"].items()), columns=["Feature", "F-Score"]).sort_values("F-Score", ascending=True)
    fig_k = px.bar(
        df_scores, x="F-Score", y="Feature", orientation="h",
        title="ANOVA F-Test Ranking",
        color_discrete_sequence=["#059669"]
    )
    fig_k.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_k, use_container_width=True)

# 4. PCA
with tab_pca:
    st.markdown("""
    <div style="background: white; border: 1px solid #E2E8F0; border-radius: 16px; padding: 1.4rem; margin-top: 0.8rem; margin-bottom: 1.2rem;">
        <h4 style="color: #064E3B; margin-top: 0;">📉 Principal Component Analysis (PCA)</h4>
        <p style="font-size: 0.88rem; color: #475569; margin: 0; line-height: 1.55;">
            Decomposes high-dimensional collinear features into orthogonal eigenvectors maximizing variance.
        </p>
    </div>
    """, unsafe_allow_html=True)

    from sklearn.preprocessing import StandardScaler
    X_std = StandardScaler().fit_transform(df_raw[feature_cols].values)

    res_pca = perform_pca_reduction(X_std, n_components=2)
    evr = res_pca["explained_variance_ratio"]

    st.markdown(f"""
    <div class="glass-metric" style="max-width: 320px; margin-bottom: 1.2rem;">
        <div class="metric-title">Variance Captured (PC1 + PC2)</div>
        <div class="metric-number" style="color: #065F46;">{res_pca['total_explained_variance'] * 100.0:.2f}%</div>
        <div class="metric-desc">Orthogonal Subspace Projection</div>
    </div>
    """, unsafe_allow_html=True)

    df_pca_plot = pd.DataFrame({
        "PC1": res_pca["X_pca"][:, 0],
        "PC2": res_pca["X_pca"][:, 1],
        "Target": df_raw[target_col]
    })

    fig_pca = px.scatter(
        df_pca_plot, x="PC1", y="PC2", color="Target",
        title=f"2D PCA Projection (PC1: {evr[0]*100:.1f}%, PC2: {evr[1]*100:.1f}%)"
    )
    fig_pca.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_pca, use_container_width=True)
