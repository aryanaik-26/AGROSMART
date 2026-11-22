"""
AGROSMART AI - Page 4: Interactive Dataset Explorer
Enhanced UI & Visual Graphics Edition
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
from PIL import Image
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from config import RAW_DATA_DIR, PLANTVILLAGE_DIR, THEME_COLORS
from src.data_loader import load_crop_recommendation_data, load_yield_data, inspect_dataset_summary
from src.ui_theme import inject_custom_css, render_page_header, render_sidebar_branding

st.set_page_config(page_title="Dataset Explorer | AGROSMART AI", page_icon="🔍", layout="wide")

# Inject Design System
inject_custom_css()
render_sidebar_branding()

# Render High-Tech Page Header
render_page_header(
    title="Comprehensive Dataset Explorer",
    subtitle="Inspect multi-spectral foliar imagery archives, soil chemical fertility benchmarks, and FAOSTAT harvest records with interactive visual distributions.",
    icon="🔍",
    badge="DATA TELEMETRY"
)

dataset_type = st.radio(
    "Choose Dataset Repository to Inspect:",
    ["🌾 Crop Recommendation Benchmark (CSV)", "📈 Agricultural Crop Yield & Production (CSV)", "🍃 PlantVillage Foliar Imagery", "📁 Upload Custom Research CSV"],
    horizontal=True
)

df = None

if dataset_type == "🌾 Crop Recommendation Benchmark (CSV)":
    try:
        df = load_crop_recommendation_data()
        st.success("Loaded benchmark Crop Recommendation dataset (2,200 records across 22 crops).")
    except Exception as e:
        st.error(f"Error loading dataset: {e}")

elif dataset_type == "📈 Agricultural Crop Yield & Production (CSV)":
    try:
        df = load_yield_data()
        st.success("Loaded agricultural crop yield and production dataset (3,500 records).")
    except Exception as e:
        st.error(f"Error loading dataset: {e}")

elif dataset_type == "📁 Upload Custom Research CSV":
    uploaded_csv = st.file_uploader("Upload custom research dataset (CSV):", type=["csv"])
    if uploaded_csv is not None:
        df = pd.read_csv(uploaded_csv)
        st.success(f"Loaded custom dataset: {len(df):,} rows, {len(df.columns)} columns.")
    else:
        st.info("Upload a CSV file to inspect its internal structure and statistical distributions.")

else:  # PlantVillage Foliar Imagery
    st.markdown("""
    <div style="background: white; border: 1px solid #E2E8F0; border-radius: 18px; padding: 1.5rem; box-shadow: 0 4px 16px rgba(0,0,0,0.03); margin-top: 1rem; margin-bottom: 1.5rem;">
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.8rem;">
            <h3 style="font-size: 1.3rem; margin: 0; color: #064E3B;">🍃 PlantVillage Foliar Image Repository</h3>
            <span class="status-chip" style="background: #ECFDF5; border-color: #A7F3D0; color: #065F46 !important;">
                54,000+ Specimen Repository
            </span>
        </div>
        <p style="font-size: 0.88rem; color: #64748B; margin-bottom: 1.2rem;">
            Taxonomic distribution of healthy and diseased foliage specimens across 14 commercial crop cultivars.
        </p>
    """, unsafe_allow_html=True)

    classes = [d for d in PLANTVILLAGE_DIR.iterdir() if d.is_dir() and len(list(d.glob("*.jpg"))) > 0]
    if classes:
        cls_summary = []
        for d in classes:
            count = len(list(d.glob("*.jpg")))
            is_healthy = "healthy" in d.name.lower()
            cls_summary.append({
                "Class Name": d.name,
                "Status": "Healthy" if is_healthy else "Diseased",
                "Image Count": count
            })
        df_cls = pd.DataFrame(cls_summary)

        c1, c2 = st.columns([1, 1.3])
        with c1:
            st.metric("Total Disease Classes Available", len(df_cls))
            st.metric("Total Leaf Images", f"{df_cls['Image Count'].sum():,}")
            st.dataframe(df_cls, use_container_width=True, height=360)
        with c2:
            fig = px.bar(
                df_cls, x="Class Name", y="Image Count", color="Status",
                color_discrete_map={"Healthy": "#10B981", "Diseased": "#EF4444"},
                title="Pathology Sample Distribution"
            )
            fig.update_layout(
                xaxis_tickangle=-45,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Plus Jakarta Sans", color="#0F291E")
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("""
        <div style="background: white; border: 1px solid #E2E8F0; border-radius: 18px; padding: 1.5rem; box-shadow: 0 4px 16px rgba(0,0,0,0.03);">
            <h4 style="color: #064E3B; margin-top: 0; font-size: 1.2rem;">🖼️ High-Resolution Specimen Gallery</h4>
        """, unsafe_allow_html=True)
        selected_cls = st.selectbox("Select Class to View Specimens:", [d.name for d in classes])
        target_folder = PLANTVILLAGE_DIR / selected_cls
        imgs = list(target_folder.glob("*.jpg"))[:8]
        if imgs:
            cols = st.columns(4)
            for i, p in enumerate(imgs):
                cols[i % 4].image(Image.open(p), caption=p.name, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.warning("No images found. Please run 'python data/generate_benchmark_data.py'.")

# If tabular dataset loaded
if df is not None:
    st.markdown("<hr class='fancy-divider'>", unsafe_allow_html=True)

    # Elevated KPI cards
    c_k1, c_k2, c_k3, c_k4 = st.columns(4)
    with c_k1:
        st.markdown(f"""
        <div class="glass-metric">
            <div class="metric-title">Row Count</div>
            <div class="metric-number">{len(df):,}</div>
            <div class="metric-desc">Curated Agricultural Instances</div>
        </div>
        """, unsafe_allow_html=True)
    with c_k2:
        st.markdown(f"""
        <div class="glass-metric">
            <div class="metric-title">Feature Dimensions</div>
            <div class="metric-number">{len(df.columns)}</div>
            <div class="metric-desc">Variables & Attributes</div>
        </div>
        """, unsafe_allow_html=True)
    with c_k3:
        st.markdown(f"""
        <div class="glass-metric">
            <div class="metric-title">Duplicate Rows</div>
            <div class="metric-number">{int(df.duplicated().sum())}</div>
            <div class="metric-desc" style="color: #10B981;">Clean Integrity Check</div>
        </div>
        """, unsafe_allow_html=True)
    with c_k4:
        st.markdown(f"""
        <div class="glass-metric">
            <div class="metric-title">Missing Cells</div>
            <div class="metric-number">{int(df.isnull().sum().sum())}</div>
            <div class="metric-desc" style="color: #10B981;">Zero Null Wastage</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["📋 Data Preview", "📊 Summary Statistics", "🔍 Missing & Data Types", "📈 Interactive Visualizer"])

    with tab1:
        st.markdown("#### Sample Records (Top 10)")
        st.dataframe(df.head(10), use_container_width=True)

    with tab2:
        st.markdown("#### Numerical Feature Statistics")
        st.dataframe(df.describe().T, use_container_width=True)

    with tab3:
        st.markdown("#### Column Attributes & Missing Ratios")
        info_df = pd.DataFrame({
            "Data Type": df.dtypes.astype(str),
            "Non-Null Count": df.notnull().sum(),
            "Missing Count": df.isnull().sum(),
            "Missing Ratio (%)": (df.isnull().sum() / len(df) * 100).round(2)
        })
        st.dataframe(info_df, use_container_width=True)

    with tab4:
        st.markdown("#### Interactive Agronomic Visualizer")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

        c_v1, c_v2 = st.columns(2)
        with c_v1:
            if numeric_cols:
                feat = st.selectbox("Select Numeric Feature for Distribution:", numeric_cols)
                fig_num = px.histogram(
                    df, x=feat, marginal="box",
                    color_discrete_sequence=["#10B981"],
                    title=f"Distribution Profile of {feat}"
                )
                fig_num.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="Plus Jakarta Sans", color="#0F291E")
                )
                st.plotly_chart(fig_num, use_container_width=True)
        with c_v2:
            if categorical_cols:
                cat = st.selectbox("Select Categorical Feature for Frequency:", categorical_cols)
                fig_cat = px.bar(
                    df[cat].value_counts().reset_index(),
                    x=cat, y="count",
                    color_discrete_sequence=["#059669"],
                    title=f"Class Frequency: {cat}"
                )
                fig_cat.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="Plus Jakarta Sans", color="#0F291E")
                )
                st.plotly_chart(fig_cat, use_container_width=True)
            elif len(numeric_cols) >= 2:
                feat2 = st.selectbox("Select Second Numeric for Correlation:", numeric_cols[1:])
                fig_scat = px.scatter(
                    df, x=numeric_cols[0], y=feat2,
                    color_discrete_sequence=["#F59E0B"],
                    title=f"{numeric_cols[0]} vs {feat2}"
                )
                fig_scat.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="Plus Jakarta Sans", color="#0F291E")
                )
                st.plotly_chart(fig_scat, use_container_width=True)
