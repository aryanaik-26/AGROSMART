"""
AGROSMART AI - Page 3: Smart Crop Recommendation & SHAP Attribution
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
from src.prediction import recommend_crops_unified
from src.explainability import plot_shap_waterfall_bar
from src.ui_theme import inject_custom_css, render_page_header, render_sidebar_branding, get_asset_image

st.set_page_config(page_title="Crop Recommendation | AGROSMART AI", page_icon="🌾", layout="wide")

# Inject Design System
inject_custom_css()
render_sidebar_branding()

# Render High-Tech Page Header
render_page_header(
    title="Smart Crop Recommendation Engine",
    subtitle="Determine top-3 optimal botanical species based on chemical macronutrients (N, P, K) and localized micro-climate using high-accuracy classification ensembles.",
    icon="🌾",
    badge="MULTI-CLASS ENSEMBLE"
)

col_in, col_out = st.columns([1.05, 1.3], gap="large")

with col_in:
    banner_img = get_asset_image("recommend_crops.jpg")
    if banner_img:
        st.image(banner_img, caption="Soil Fertility & Multi-Species Crop Taxonomy", use_container_width=True)

    st.markdown("""
    <div style="background: white; border: 1px solid #E2E8F0; border-radius: 18px; padding: 1.4rem; box-shadow: 0 4px 16px rgba(0,0,0,0.03); margin-bottom: 1.4rem;">
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 0.6rem;">
            <span style="font-size: 1.3rem;">🧪</span>
            <h3 style="font-size: 1.2rem; margin: 0; color: #064E3B;">Soil & Climate Laboratory Data</h3>
        </div>
        <p style="font-size: 0.86rem; color: #64748B; margin-bottom: 1rem;">
            Adjust laboratory soil assay readings and local meteorological measurements.
        </p>
    """, unsafe_allow_html=True)

    with st.expander("🌱 1. Soil Macronutrients (Ratio / kg/ha)", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            val_n = st.slider("Nitrogen (N):", min_value=0, max_value=140, value=85, step=1)
        with c2:
            val_p = st.slider("Phosphorus (P):", min_value=5, max_value=145, value=48, step=1)
        with c3:
            val_k = st.slider("Potassium (K):", min_value=5, max_value=205, value=42, step=1)

    with st.expander("🌦️ 2. Localized Micro-Climate Parameters", expanded=True):
        c4, c5 = st.columns(2)
        with c4:
            val_temp = st.slider("Ambient Temperature (°C):", min_value=8.0, max_value=45.0, value=24.5, step=0.5)
            val_ph = st.slider("Soil Reaction (pH):", min_value=3.5, max_value=9.5, value=6.5, step=0.1)
        with c5:
            val_hum = st.slider("Relative Atmospheric Humidity (%):", min_value=10.0, max_value=100.0, value=80.0, step=1.0)
            val_rain = st.slider("Cumulative Rainfall (mm):", min_value=20.0, max_value=320.0, value=195.0, step=5.0)

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""
    <div style="background: white; border: 1px solid #E2E8F0; border-radius: 18px; padding: 1.4rem; box-shadow: 0 4px 16px rgba(0,0,0,0.03);">
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 0.6rem;">
            <span style="font-size: 1.3rem;">⚙️</span>
            <h3 style="font-size: 1.2rem; margin: 0; color: #064E3B;">Classifier Model</h3>
        </div>
    """, unsafe_allow_html=True)

    model_choice = st.selectbox(
        "Select Classification Algorithm:",
        ["Best Model", "Stacking Ensemble", "Random Forest", "XGBoost", "Support Vector Machine", "Gradient Boosting"]
    )

    st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
    btn_recommend = st.button("🌱 Recommend Optimal Crops", type="primary", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col_out:
    st.markdown("""
    <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.8rem;">
        <h3 style="font-size: 1.35rem; margin: 0; color: #064E3B;">Optimal Crop Recommendations</h3>
        <span class="status-chip" style="background: #EFF6FF; border-color: #BFDBFE; color: #1E40AF !important;">
            Top-3 Ranking
        </span>
    </div>
    """, unsafe_allow_html=True)

    if btn_recommend:
        with st.spinner("Executing multi-class probability scoring and computing feature impact..."):
            features_input = {
                "N": float(val_n),
                "P": float(val_p),
                "K": float(val_k),
                "temperature": float(val_temp),
                "humidity": float(val_hum),
                "ph": float(val_ph),
                "rainfall": float(val_rain)
            }

            try:
                res = recommend_crops_unified(features_input, model_name=model_choice, top_k=3)
                top_crops = res["top_recommendations"]

                rank_styles = [
                    {"border": "#F59E0B", "bg": "linear-gradient(135deg, #FFFBEB 0%, #FFFFFF 100%)", "badge_bg": "#FEF3C7", "badge_color": "#92400E", "medal": "🥇 First Choice"},
                    {"border": "#94A3B8", "bg": "linear-gradient(135deg, #F8FAFC 0%, #FFFFFF 100%)", "badge_bg": "#E2E8F0", "badge_color": "#334155", "medal": "🥈 Alternative A"},
                    {"border": "#D97706", "bg": "linear-gradient(135deg, #FFF7ED 0%, #FFFFFF 100%)", "badge_bg": "#FFEDD5", "badge_color": "#9A3412", "medal": "🥉 Alternative B"}
                ]

                for i, item in enumerate(top_crops):
                    style = rank_styles[i]
                    pct = item['percentage']
                    st.markdown(f"""
                    <div style="background: {style['bg']}; border: 1.5px solid {style['border']}; border-radius: 16px; padding: 1.3rem 1.6rem; margin-bottom: 1rem; box-shadow: 0 4px 12px rgba(0,0,0,0.03); display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 12px;">
                        <div style="display: flex; align-items: center; gap: 1rem;">
                            <div style="font-size: 2.2rem; font-weight: 800; color: #064E3B; width: 45px; text-align: center;">
                                {i+1}
                            </div>
                            <div>
                                <div style="font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.06em; color: {style['badge_color']}; font-weight: 700;">
                                    {style['medal']}
                                </div>
                                <div style="font-family: 'Outfit', sans-serif; font-size: 1.6rem; font-weight: 800; color: #0F291E;">
                                    {item['crop']}
                                </div>
                            </div>
                        </div>
                        <div style="text-align: right;">
                            <span style="font-size: 1.15rem; font-weight: 800; padding: 6px 14px; border-radius: 9999px; background: {style['badge_bg']}; color: {style['badge_color']}; display: inline-block;">
                                {pct}% Match
                            </span>
                            <div style="font-size: 0.78rem; color: #64748B; margin-top: 4px;">
                                Raw Score: {item['confidence_score']}
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("<div style='height: 0.6rem;'></div>", unsafe_allow_html=True)

                # SHAP Feature Importance Breakdown
                st.markdown("""
                <div style="background: white; border: 1px solid #E2E8F0; border-radius: 18px; padding: 1.5rem; box-shadow: 0 4px 16px rgba(0,0,0,0.03); margin-bottom: 1.2rem;">
                    <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.6rem;">
                        <h4 style="margin: 0; color: #064E3B; font-size: 1.15rem;">💡 Feature Impact on Primary Selection</h4>
                        <span style="font-size: 0.78rem; color: #2563EB; font-weight: 600; background: #EFF6FF; padding: 4px 10px; border-radius: 12px; border: 1px solid #BFDBFE;">
                            SHAP Interpretability
                        </span>
                    </div>
                    <p style="font-size: 0.86rem; color: #64748B; margin-bottom: 1rem;">
                        Visualizes why <strong>{crop}</strong> emerged as the top recommendation under your given soil-climate profile.
                    </p>
                """.format(crop=res['primary_recommendation']), unsafe_allow_html=True)

                shap_data = res.get("shap_explanation", {})
                contribs = shap_data.get("feature_contributions", {})

                if contribs:
                    fig = plot_shap_waterfall_bar(contribs, title=f"Factor Contributions for {res['primary_recommendation']}")
                    st.pyplot(fig)
                    plt.close(fig)

                    st.markdown(f"""
                    <div style="background: #F8FAF9; border-left: 4px solid #10B981; border-radius: 8px; padding: 0.9rem 1.1rem; margin-top: 1rem;">
                        <div style="font-weight: 700; color: #065F46; font-size: 0.88rem; margin-bottom: 4px;">Agronomic Recommendation Insight:</div>
                        <div style="font-size: 0.88rem; color: #334155; line-height: 1.5;">{shap_data.get('narrative', '')}</div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("</div>", unsafe_allow_html=True)
                st.info(f"💡 **Agronomic Guidance**: {res['disclaimer']}")

            except Exception as e:
                st.error(f"Recommendation Error: {e}")
                st.info("Ensure models are trained by checking 'Model Training'.")
    else:
        st.markdown("""
        <div style="background: #F8FAF9; border: 2px dashed #CBD5E1; border-radius: 18px; padding: 3rem 2rem; text-align: center; color: #64748B;">
            <div style="font-size: 3rem; margin-bottom: 0.8rem;">🌾</div>
            <h4 style="color: #0F291E; font-size: 1.2rem; margin: 0 0 0.5rem 0;">Awaiting Soil Assay Inputs</h4>
            <p style="font-size: 0.92rem; max-width: 480px; margin: 0 auto; line-height: 1.55;">
                Adjust soil macronutrient sliders (N, P, K) and localized climate readings on the left panel, then click <strong>Recommend Optimal Crops</strong> to evaluate top-3 suitability.
            </p>
        </div>
        """, unsafe_allow_html=True)
