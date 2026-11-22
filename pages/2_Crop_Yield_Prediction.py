"""
AGROSMART AI - Page 2: Crop Yield Prediction & SHAP Explainability
Enhanced UI & Visual Graphics Edition
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from config import THEME_COLORS
from src.prediction import predict_crop_yield_unified
from src.explainability import plot_shap_waterfall_bar
from src.ui_theme import inject_custom_css, render_page_header, render_sidebar_branding, get_asset_image

st.set_page_config(page_title="Crop Yield Prediction | AGROSMART AI", page_icon="📈", layout="wide")

# Inject Design System
inject_custom_css()
render_sidebar_branding()

# Render High-Tech Page Header
render_page_header(
    title="Crop Yield Prediction & SHAP Analysis",
    subtitle="Forecast harvest productivity in tonnes per hectare (t/ha) with statistical 95% confidence bounds and game-theoretic SHAP feature impact attribution.",
    icon="📈",
    badge="REGRESSION META-ENSEMBLE"
)

col_in, col_out = st.columns([1.1, 1.3], gap="large")

with col_in:
    banner_img = get_asset_image("yield_wheat.jpg")
    if banner_img:
        st.image(banner_img, caption="Harvest Yield & Production Forecasting", use_container_width=True)

    st.markdown("""
    <div style="background: white; border: 1px solid #E2E8F0; border-radius: 18px; padding: 1.4rem; box-shadow: 0 4px 16px rgba(0,0,0,0.03); margin-bottom: 1.4rem;">
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 0.6rem;">
            <span style="font-size: 1.3rem;">🌾</span>
            <h3 style="font-size: 1.2rem; margin: 0; color: #064E3B;">Agronomic & Spatial Profile</h3>
        </div>
    """, unsafe_allow_html=True)

    with st.expander("📍 Crop Classification & Location", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            crop = st.selectbox("Cultivated Crop:", ["Rice", "Wheat", "Maize", "Cotton", "Sugarcane", "Potato", "Soybean", "Groundnut", "Mustard"])
            season = st.selectbox("Agricultural Season:", ["Kharif", "Rabi", "Summer", "Whole Year"])
        with col2:
            state = st.selectbox("State / Agro-Zone:", ["Punjab", "Haryana", "Uttar Pradesh", "Maharashtra", "Karnataka", "Tamil Nadu", "Andhra Pradesh", "Gujarat", "Madhya Pradesh", "West Bengal"])
            soil_type = st.selectbox("Soil Taxonomy:", ["Alluvial", "Black", "Red", "Clayey", "Sandy Loam"])

        area = st.number_input("Cultivated Acreage (Hectares):", min_value=0.5, max_value=50000.0, value=100.0, step=5.0)

    with st.expander("🌦️ Agroclimatological Parameters", expanded=True):
        col3, col4 = st.columns(2)
        with col3:
            rainfall = st.slider("Seasonal Precipitation (mm):", min_value=150.0, max_value=2500.0, value=750.0, step=25.0)
            temp = st.slider("Mean Ambient Temperature (°C):", min_value=10.0, max_value=45.0, value=26.5, step=0.5)
        with col4:
            humidity = st.slider("Relative Atmospheric Humidity (%):", min_value=20.0, max_value=100.0, value=70.0, step=1.0)
            soil_ph = st.slider("Soil Reaction (pH):", min_value=4.5, max_value=9.0, value=6.8, step=0.1)

    with st.expander("🧪 Macronutrient Application (kg/ha)", expanded=True):
        c_n, c_p, c_k = st.columns(3)
        with c_n:
            fert_n = st.number_input("Nitrogen (N):", min_value=0.0, max_value=300.0, value=120.0, step=5.0)
        with c_p:
            fert_p = st.number_input("Phosphorus (P):", min_value=0.0, max_value=200.0, value=60.0, step=5.0)
        with c_k:
            fert_k = st.number_input("Potassium (K):", min_value=0.0, max_value=200.0, value=40.0, step=5.0)

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""
    <div style="background: white; border: 1px solid #E2E8F0; border-radius: 18px; padding: 1.4rem; box-shadow: 0 4px 16px rgba(0,0,0,0.03);">
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 0.6rem;">
            <span style="font-size: 1.3rem;">⚙️</span>
            <h3 style="font-size: 1.2rem; margin: 0; color: #064E3B;">Regression Algorithm</h3>
        </div>
    """, unsafe_allow_html=True)

    model_choice = st.selectbox(
        "Select Regression Model:",
        ["Best Model", "Stacking Ensemble", "XGBoost", "Random Forest", "Gradient Boosting", "Ridge Regression", "Support Vector Regressor"]
    )

    st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
    btn_predict_yield = st.button("🚀 Calculate Harvest Forecast", type="primary", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col_out:
    st.markdown("""
    <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.8rem;">
        <h3 style="font-size: 1.35rem; margin: 0; color: #064E3B;">Harvest Forecast & Explainability</h3>
        <span class="status-chip" style="background: #FFFBEB; border-color: #FDE68A; color: #92400E !important;">
            SHAP Waterfall
        </span>
    </div>
    """, unsafe_allow_html=True)

    if btn_predict_yield:
        with st.spinner("Computing non-linear ensemble regression and tree Shapley values..."):
            features_input = {
                "Crop": crop,
                "State": state,
                "Season": season,
                "Soil_Type": soil_type,
                "Area": float(area),
                "Rainfall": float(rainfall),
                "Fertilizer_N": float(fert_n),
                "Fertilizer_P": float(fert_p),
                "Fertilizer_K": float(fert_k),
                "Temperature": float(temp),
                "Humidity": float(humidity),
                "Soil_pH": float(soil_ph)
            }

            try:
                res = predict_crop_yield_unified(features_input, model_name=model_choice)

                # Modern Glowing Yield Prediction Card
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #092B1C 0%, #0F3F2A 50%, #1A563A 100%); border: 1px solid rgba(52, 211, 153, 0.4); border-radius: 20px; padding: 2.2rem; color: white; box-shadow: 0 16px 36px -10px rgba(9, 43, 28, 0.45); margin-bottom: 1.4rem;">
                    <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.6rem;">
                        <span style="font-size: 0.82rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: #A7F3D0;">
                            EXPECTED HARVEST PRODUCTIVITY
                        </span>
                        <span style="background: rgba(255,255,255,0.15); font-size: 0.78rem; padding: 4px 10px; border-radius: 9999px; font-weight: 600;">
                            {crop} ({season})
                        </span>
                    </div>
                    <div style="font-family: 'Outfit', sans-serif; font-size: 3.5rem; font-weight: 800; color: #D1FAE5; line-height: 1.1; margin: 0.2rem 0;">
                        {res['predicted_yield_tha']} <span style="font-size: 1.6rem; font-weight: 500; color: #6EE7B7;">t/ha</span>
                    </div>
                    <div style="font-size: 1.05rem; color: #E2ECE6; margin-top: 0.4rem; font-weight: 500;">
                        Total Estimated Production: <strong style="color: #FFFFFF; font-size: 1.25rem;">{res['estimated_total_production_tonnes']:,} Tonnes</strong> across {area:,.1f} ha
                    </div>
                    <div style="margin-top: 1.2rem; background: rgba(0, 0, 0, 0.25); border: 1px solid rgba(255,255,255,0.12); padding: 0.75rem 1rem; border-radius: 12px; font-size: 0.88rem; display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 8px;">
                        <div>
                            🎯 <strong>95% Prediction Interval:</strong> [{res['prediction_interval']['lower']} &mdash; {res['prediction_interval']['upper']}] t/ha
                        </div>
                        <div style="color: #6EE7B7; font-size: 0.82rem;">
                            RMSE: {res['model_rmse']} t/ha
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # SHAP Interpretability Section
                st.markdown("""
                <div style="background: white; border: 1px solid #E2E8F0; border-radius: 18px; padding: 1.5rem; box-shadow: 0 4px 16px rgba(0,0,0,0.03); margin-bottom: 1.2rem;">
                    <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.6rem;">
                        <h4 style="margin: 0; color: #064E3B; font-size: 1.15rem;">💡 SHAP Feature Attribution Impact</h4>
                        <span style="font-size: 0.78rem; color: #D97706; font-weight: 600; background: #FFFBEB; padding: 4px 10px; border-radius: 12px; border: 1px solid #FDE68A;">
                            Game-Theoretic Values
                        </span>
                    </div>
                    <p style="font-size: 0.86rem; color: #64748B; margin-bottom: 1rem;">
                        Quantifies the exact positive (green) or negative (red) yield impact (t/ha) contributed by each climatic and chemical soil factor relative to expected baseline yield.
                    </p>
                """, unsafe_allow_html=True)

                shap_data = res.get("shap_explanation", {})
                contribs = shap_data.get("feature_contributions", {})

                if contribs:
                    fig = plot_shap_waterfall_bar(contribs, title=f"SHAP Agronomic Influence Profile: {crop}")
                    st.pyplot(fig)
                    plt.close(fig)

                    st.markdown(f"""
                    <div style="background: #F8FAF9; border-left: 4px solid #10B981; border-radius: 8px; padding: 0.9rem 1.1rem; margin-top: 1rem;">
                        <div style="font-weight: 700; color: #065F46; font-size: 0.88rem; margin-bottom: 4px;">Agronomic Interpretation:</div>
                        <div style="font-size: 0.88rem; color: #334155; line-height: 1.5;">{shap_data.get('narrative', '')}</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.info(shap_data.get("narrative", "SHAP attribution computed."))

                st.markdown("</div>", unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Prediction Error: {e}")
                st.info("Ensure models are trained by checking 'Model Training'.")
    else:
        st.markdown("""
        <div style="background: #F8FAF9; border: 2px dashed #CBD5E1; border-radius: 18px; padding: 3rem 2rem; text-align: center; color: #64748B;">
            <div style="font-size: 3rem; margin-bottom: 0.8rem;">📊</div>
            <h4 style="color: #0F291E; font-size: 1.2rem; margin: 0 0 0.5rem 0;">Awaiting Input Parameters</h4>
            <p style="font-size: 0.92rem; max-width: 480px; margin: 0 auto; line-height: 1.55;">
                Configure the crop type, rainfall, temperatures, and fertilizer rates on the left panel, then click <strong>Calculate Harvest Forecast</strong> to project yields and inspect SHAP impacts.
            </p>
        </div>
        """, unsafe_allow_html=True)
