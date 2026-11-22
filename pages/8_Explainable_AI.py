"""
AGROSMART AI - Page 8: Explainable AI (XAI) Deep Dive
Enhanced UI & Visual Graphics Edition
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from config import THEME_COLORS, PLANTVILLAGE_DIR
from src.prediction import predict_crop_disease, predict_crop_yield_unified
from src.explainability import plot_shap_waterfall_bar
from src.ui_theme import inject_custom_css, render_page_header, render_sidebar_branding

st.set_page_config(page_title="Explainable AI | AGROSMART AI", page_icon="💡", layout="wide")

# Inject Design System
inject_custom_css()
render_sidebar_branding()

# Render High-Tech Page Header
render_page_header(
    title="Explainable AI (XAI) Diagnostic Center",
    subtitle="Mathematically unpack model decisions using cooperative game theory (SHAP) and convolutional activation gradients (Grad-CAM) to establish scientific trust.",
    icon="💡",
    badge="MATHEMATICAL INTERPRETABILITY"
)

tab_shap, tab_gradcam = st.tabs(["📊 SHAP (Tabular Interpretability)", "🍃 Grad-CAM (Computer Vision Localization)"])

# 1. SHAP TAB
with tab_shap:
    st.markdown("""
    <div style="background: white; border: 1px solid #E2E8F0; border-radius: 18px; padding: 1.6rem; margin-top: 0.8rem; margin-bottom: 1.4rem; box-shadow: 0 4px 16px rgba(0,0,0,0.03);">
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.8rem;">
            <h4 style="margin: 0; color: #064E3B; font-size: 1.25rem;">Cooperative Game Theory: Shapley Value Formulation</h4>
            <span class="status-chip" style="background: #EFF6FF; border-color: #BFDBFE; color: #1E40AF !important;">
                Lloyd Shapley (1953) & Lundberg (2017)
            </span>
        </div>
        <p style="font-size: 0.9rem; color: #475569; line-height: 1.6; margin-bottom: 0.8rem;">
            SHAP assigns each agricultural feature $i \in F$ an exact payout value $\phi_i$ by calculating its weighted marginal contribution across all power subsets $S \subseteq F \setminus \{i\}$:
        </p>
        <div style="background: #F8FAF9; padding: 1rem 1.4rem; border-radius: 12px; border: 1px solid #E2E8F0; font-size: 1.05rem; text-align: center; margin: 1rem 0;">
            $$\phi_i(f, x) = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|!(|F| - |S| - 1)!}{|F|!} \left[ f(S \cup \{i\}) - f(S) \right]$$
        </div>
        <div style="font-size: 0.85rem; color: #065F46; font-weight: 600;">
            ✔ Efficiency (Local Accuracy) &bull; ✔ Symmetry &bull; ✔ Dummy Player &bull; ✔ Additivity Guaranteed
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### 🧪 **Interactive SHAP Yield Attribution Simulator**")

    c1, c2 = st.columns([1, 1.25], gap="large")
    with c1:
        st.markdown("""
        <div style="background: white; border: 1px solid #E2E8F0; border-radius: 16px; padding: 1.4rem; box-shadow: 0 4px 16px rgba(0,0,0,0.03);">
            <div style="font-size: 0.82rem; font-weight: 700; color: #065F46; text-transform: uppercase; margin-bottom: 0.8rem;">
                SIMULATION CONTROLS
            </div>
        """, unsafe_allow_html=True)
        sim_crop = st.selectbox("Select Crop Type:", ["Rice", "Wheat", "Maize", "Cotton", "Sugarcane", "Potato"])
        sim_rain = st.slider("Simulated Precipitation (mm):", 200.0, 2000.0, 1100.0, 50.0)
        sim_n = st.slider("Simulated Soil Nitrogen (kg/ha):", 20.0, 250.0, 150.0, 10.0)
        sim_temp = st.slider("Simulated Ambient Temp (°C):", 12.0, 42.0, 24.0, 1.0)
        sim_ph = st.slider("Simulated Soil Reaction (pH):", 4.5, 9.0, 6.8, 0.1)

        st.markdown("<div style='height: 0.6rem;'></div>", unsafe_allow_html=True)
        sim_button = st.button("Generate SHAP Attribution", type="primary", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        if sim_button:
            features = {
                "Crop": sim_crop, "State": "Punjab", "Season": "Kharif", "Soil_Type": "Alluvial",
                "Area": 100.0, "Rainfall": sim_rain, "Fertilizer_N": sim_n, "Fertilizer_P": 60.0,
                "Fertilizer_K": 50.0, "Temperature": sim_temp, "Humidity": 75.0, "Soil_pH": sim_ph
            }
            try:
                res = predict_crop_yield_unified(features, model_name="Best Model")
                shap_data = res.get("shap_explanation", {})
                contribs = shap_data.get("feature_contributions", {})
                if contribs:
                    fig = plot_shap_waterfall_bar(contribs, title=f"SHAP Marginal Attribution Breakdown ({sim_crop})")
                    st.pyplot(fig)
                    plt.close(fig)

                    st.markdown(f"""
                    <div style="background: #F8FAF9; border-left: 4px solid #10B981; border-radius: 8px; padding: 0.9rem 1.1rem; margin-top: 1rem;">
                        <div style="font-weight: 700; color: #065F46; font-size: 0.88rem; margin-bottom: 4px;">Agronomic Insight:</div>
                        <div style="font-size: 0.88rem; color: #334155; line-height: 1.5;">{shap_data.get('narrative', '')}</div>
                    </div>
                    """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error computing SHAP: {e}")
        else:
            st.markdown("""
            <div style="background: #F8FAF9; border: 2px dashed #CBD5E1; border-radius: 18px; padding: 2.5rem 1.5rem; text-align: center; color: #64748B;">
                <div style="font-size: 2.4rem; margin-bottom: 0.6rem;">📊</div>
                <h4 style="color: #0F291E; font-size: 1.1rem; margin: 0 0 0.4rem 0;">Awaiting Simulation Trigger</h4>
                <p style="font-size: 0.88rem; max-width: 380px; margin: 0 auto; line-height: 1.5;">
                    Adjust simulated rainfall and soil NPK sliders, then click <strong>Generate SHAP Attribution</strong> to compute real-time game-theoretic feature contributions.
                </p>
            </div>
            """, unsafe_allow_html=True)

# 2. GRAD-CAM TAB
with tab_gradcam:
    st.markdown("""
    <div style="background: white; border: 1px solid #E2E8F0; border-radius: 18px; padding: 1.6rem; margin-top: 0.8rem; margin-bottom: 1.4rem; box-shadow: 0 4px 16px rgba(0,0,0,0.03);">
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.8rem;">
            <h4 style="margin: 0; color: #064E3B; font-size: 1.25rem;">Convolutional Backpropagation: Grad-CAM Formulation</h4>
            <span class="status-chip" style="background: #FAF5FF; border-color: #E9D5FF; color: #7C3AED !important;">
                Selvaraju et al. (ICCV 2017)
            </span>
        </div>
        <p style="font-size: 0.9rem; color: #475569; line-height: 1.6; margin-bottom: 0.8rem;">
            Grad-CAM calculates the gradient of the predicted class score $y^c$ with respect to the activation maps $A^k$ of the final convolutional layer:
        </p>
        <div style="background: #F8FAF9; padding: 1rem 1.4rem; border-radius: 12px; border: 1px solid #E2E8F0; font-size: 1.05rem; text-align: center; margin: 1rem 0;">
            $$\alpha_k^c = \frac{1}{Z} \sum_{i=1}^{U} \sum_{j=1}^{V} \frac{\partial y^c}{\partial A_{ij}^k}, \quad L_{\text{Grad-CAM}}^c = \text{ReLU}\left( \sum_k \alpha_k^c A^k \right)$$
        </div>
        <div style="font-size: 0.85rem; color: #7C3AED; font-weight: 600;">
            ✔ The ReLU operation captures strictly feature maps that positively support class $c$, ignoring negative inhibitory noise.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### 🖼️ **Live Grad-CAM Foliar Inspection**")

    classes = [d for d in PLANTVILLAGE_DIR.iterdir() if d.is_dir() and len(list(d.glob("*.jpg"))) > 0]
    if classes:
        sample_cls = st.selectbox("Select Leaf Disease Class:", [d.name for d in classes], key="gc_cls")
        cls_dir = PLANTVILLAGE_DIR / sample_cls
        imgs = list(cls_dir.glob("*.jpg"))
        if imgs:
            test_img = Image.open(imgs[0])

            cg1, cg2 = st.columns(2, gap="large")
            with cg1:
                st.image(test_img, caption=f"Specimen: {sample_cls}", use_container_width=True)
            with cg2:
                if st.button("Generate Grad-CAM Heatmap", key="btn_gc", type="primary", use_container_width=True):
                    with st.spinner("Extracting backpropagated gradients from final convolutional layer..."):
                        res_diag = predict_crop_disease(test_img, model_name="Best Model")
                        if res_diag["gradcam_image"] is not None:
                            st.image(res_diag["gradcam_image"], caption="Grad-CAM Activation Map Overlay", use_container_width=True)
                            st.success(f"Classified as: **{res_diag['disease_name']}** ({res_diag['confidence_percentage']}%)")
    else:
        st.warning("No sample images available. Run benchmark generation.")
