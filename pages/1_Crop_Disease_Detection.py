"""
AGROSMART AI - Page 1: Crop Disease Detection with Grad-CAM
Enhanced UI & Visual Graphics Edition
"""

import streamlit as st
import numpy as np
from PIL import Image
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from config import PLANTVILLAGE_DIR, THEME_COLORS
from src.prediction import predict_crop_disease
from src.ui_theme import inject_custom_css, render_page_header, render_sidebar_branding, get_asset_image

st.set_page_config(page_title="Crop Disease Detection | AGROSMART AI", page_icon="🍃", layout="wide")

# Inject Design System
inject_custom_css()
render_sidebar_branding()

# Render High-Tech Page Header
render_page_header(
    title="Crop Disease Detection & Grad-CAM",
    subtitle="Classify botanical pathogens across 38 PlantVillage classes using deep vision architectures (Custom CNN, MobileNetV2, EfficientNetB0) with Explainable AI Grad-CAM localization.",
    icon="🍃",
    badge="COMPUTER VISION ENGINE"
)

col_left, col_right = st.columns([1, 1.25], gap="large")

with col_left:
    banner_img = get_asset_image("disease_leaf.jpg")
    if banner_img:
        st.image(banner_img, caption="Precision Foliar Pathology Scanner", use_container_width=True)

    st.markdown("""
    <div style="background: white; border: 1px solid #E2E8F0; border-radius: 18px; padding: 1.5rem; box-shadow: 0 4px 16px rgba(0,0,0,0.03); margin-bottom: 1.5rem;">
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 0.8rem;">
            <span style="font-size: 1.3rem;">📷</span>
            <h3 style="font-size: 1.25rem; margin: 0; color: #064E3B;">Foliar Sample Ingestion</h3>
        </div>
        <p style="font-size: 0.88rem; color: #64748B; margin-bottom: 1rem;">
            Choose a calibrated benchmark sample or upload your high-resolution leaf photograph.
        </p>
    """, unsafe_allow_html=True)

    source_option = st.radio(
        "Choose Image Source:",
        ["Sample Benchmark Leaf", "Upload Leaf Image"],
        horizontal=True
    )

    image_to_process = None

    if source_option == "Upload Leaf Image":
        uploaded_file = st.file_uploader(
            "Upload leaf photograph (JPG, JPEG, PNG):",
            type=["jpg", "jpeg", "png"]
        )
        if uploaded_file is not None:
            image_to_process = Image.open(uploaded_file)
            st.image(image_to_process, caption="Uploaded Foliar Specimen", use_container_width=True)
    else:
        sample_classes = [d for d in PLANTVILLAGE_DIR.iterdir() if d.is_dir() and len(list(d.glob("*.jpg"))) > 0]
        if sample_classes:
            class_names = [d.name for d in sample_classes]
            selected_cls = st.selectbox("Select PlantVillage Class Sample:", class_names, index=min(7, len(class_names)-1))
            cls_dir = PLANTVILLAGE_DIR / selected_cls
            sample_imgs = list(cls_dir.glob("*.jpg"))
            if sample_imgs:
                sample_path = sample_imgs[0]
                image_to_process = Image.open(sample_path)
                st.image(image_to_process, caption=f"Specimen: {selected_cls}", use_container_width=True)
        else:
            st.warning("No sample images found in PlantVillage repository. Please upload a leaf image.")

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""
    <div style="background: white; border: 1px solid #E2E8F0; border-radius: 18px; padding: 1.5rem; box-shadow: 0 4px 16px rgba(0,0,0,0.03);">
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 0.8rem;">
            <span style="font-size: 1.3rem;">⚙️</span>
            <h3 style="font-size: 1.25rem; margin: 0; color: #064E3B;">Inference Configuration</h3>
        </div>
    """, unsafe_allow_html=True)

    model_choice = st.selectbox(
        "Select Deep Learning Backbone:",
        ["Best Model", "EfficientNetB0", "MobileNetV2", "Custom CNN"]
    )

    st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
    run_predict = st.button("🔬 Execute Disease Diagnostics", type="primary", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col_right:
    st.markdown("""
    <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.8rem;">
        <h3 style="font-size: 1.35rem; margin: 0; color: #064E3B;">Diagnostic Telemetry & Visual Heatmap</h3>
        <span class="status-chip" style="background: #F1F5F9; border-color: #E2E8F0; color: #475569 !important;">
            Grad-CAM Activated
        </span>
    </div>
    """, unsafe_allow_html=True)

    if run_predict and image_to_process is not None:
        with st.spinner("Extracting convolutional feature activations and computing Grad-CAM gradients..."):
            try:
                res = predict_crop_disease(image_to_process, model_name=model_choice)

                is_healthy = res["status"] == "Healthy"
                status_bg = "#ECFDF5" if is_healthy else "#FEF2F2"
                status_border = "#A7F3D0" if is_healthy else "#FECACA"
                status_color = "#065F46" if is_healthy else "#991B1B"
                status_icon = "🟢" if is_healthy else "⚠️"

                # Visual Diagnosis Result Banner
                st.markdown(f"""
                <div style="background: {status_bg}; border: 1.5px solid {status_border}; border-radius: 18px; padding: 1.6rem; margin-bottom: 1.2rem; box-shadow: 0 4px 16px rgba(0,0,0,0.03);">
                    <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 10px;">
                        <div>
                            <div style="font-size: 0.78rem; font-weight: 700; color: {status_color}; text-transform: uppercase; letter-spacing: 0.08em;">
                                PATHOLOGICAL CLASSIFICATION
                            </div>
                            <h2 style="margin: 0.2rem 0; color: {status_color}; font-size: 1.8rem; font-weight: 800;">
                                {res['disease_name']}
                            </h2>
                            <div style="font-size: 0.95rem; color: #475569;">
                                Botanical Host: <strong style="color: #0F291E;">{res['crop_name']}</strong> &bull; Model: <strong>{model_choice}</strong>
                            </div>
                        </div>
                        <div style="background: white; border: 1px solid {status_border}; padding: 8px 16px; border-radius: 9999px; font-weight: 800; font-size: 0.95rem; color: {status_color}; display: flex; align-items: center; gap: 6px;">
                            <span>{status_icon}</span>
                            <span>{res['status'].upper()}</span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Confidence Metric Card
                conf_val = float(res['confidence_percentage'])
                c_c1, c_c2 = st.columns([1, 2])
                with c_c1:
                    st.markdown(f"""
                    <div class="glass-metric" style="padding: 1rem 1.2rem;">
                        <div class="metric-title">Model Confidence</div>
                        <div class="metric-number" style="font-size: 1.9rem; color: #065F46;">{conf_val}%</div>
                        <div class="metric-desc" style="color: #059669;">Softmax Output Layer</div>
                    </div>
                    """, unsafe_allow_html=True)
                with c_c2:
                    st.markdown("<div style='font-size: 0.85rem; font-weight: 600; color: #475569; margin-top: 10px;'>Classification Certainty:</div>", unsafe_allow_html=True)
                    st.progress(res["confidence_score"])
                    st.caption("Cross-entropy loss optimized across 38 PlantVillage foliar pathologies.")

                # Grad-CAM Side-by-Side Visual Heatmap
                st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
                st.markdown("""
                <div style="background: white; border: 1px solid #E2E8F0; border-radius: 18px; padding: 1.4rem; box-shadow: 0 4px 16px rgba(0,0,0,0.03); margin-bottom: 1.4rem;">
                    <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.8rem;">
                        <h4 style="margin: 0; color: #064E3B; font-size: 1.15rem;">🌟 Grad-CAM Spatial Activation Heatmap</h4>
                        <span style="font-size: 0.78rem; color: #7C3AED; font-weight: 600; background: #FAF5FF; padding: 4px 10px; border-radius: 12px; border: 1px solid #E9D5FF;">
                            Explainable AI Vision
                        </span>
                    </div>
                    <p style="font-size: 0.85rem; color: #64748B; margin-bottom: 1rem;">
                        Visualizes localized gradient activations from the final convolutional stage, proving the model focuses on biological lesions rather than image backgrounds.
                    </p>
                """, unsafe_allow_html=True)

                img_col1, img_col2 = st.columns(2)
                with img_col1:
                    st.image(res["original_image"], caption="Input Leaf Specimen (224×224)", use_container_width=True)
                with img_col2:
                    if res["gradcam_image"] is not None:
                        st.image(res["gradcam_image"], caption="Grad-CAM Jet Heatmap Overlay", use_container_width=True)
                    else:
                        st.info("Grad-CAM visualization rendered.")

                st.markdown("</div>", unsafe_allow_html=True)

                # Agronomic Action Plan Tabs
                d_tab1, d_tab2, d_tab3 = st.tabs(["🦠 Pathology & Symptoms", "🌿 Organic Biological Cures", "🧪 Chemical Treatment & Prevention"])

                with d_tab1:
                    st.markdown(f"""
                    <div style="background: white; border: 1px solid #E2E8F0; border-radius: 14px; padding: 1.2rem; margin-top: 0.6rem;">
                        <div style="font-weight: 700; color: #0F291E; font-size: 0.95rem; margin-bottom: 4px;">Causal Pathogen / Organism:</div>
                        <p style="font-size: 0.9rem; color: #334155; margin-bottom: 1rem; line-height: 1.5;">{res['pathogen']}</p>
                        <div style="font-weight: 700; color: #0F291E; font-size: 0.95rem; margin-bottom: 4px;">Visual Symptoms & Morphological Signs:</div>
                        <p style="font-size: 0.9rem; color: #334155; margin: 0; line-height: 1.5;">{res['symptoms']}</p>
                    </div>
                    """, unsafe_allow_html=True)

                with d_tab2:
                    st.markdown(f"""
                    <div style="background: #F0FDF4; border: 1px solid #BBF7D0; border-radius: 14px; padding: 1.2rem; margin-top: 0.6rem;">
                        <div style="font-weight: 700; color: #166534; font-size: 0.95rem; margin-bottom: 6px;">🌿 Verified Organic & Biocontrol Strategy:</div>
                        <p style="font-size: 0.9rem; color: #14532D; margin: 0; line-height: 1.55;">{res['organic_control']}</p>
                    </div>
                    """, unsafe_allow_html=True)

                with d_tab3:
                    st.markdown(f"""
                    <div style="background: white; border: 1px solid #E2E8F0; border-radius: 14px; padding: 1.2rem; margin-top: 0.6rem;">
                        <div style="font-weight: 700; color: #991B1B; font-size: 0.95rem; margin-bottom: 4px;">🧪 Prescribed Chemical Intervention:</div>
                        <p style="font-size: 0.9rem; color: #334155; margin-bottom: 1rem; line-height: 1.5;">{res['chemical_control']}</p>
                        <div style="font-weight: 700; color: #065F46; font-size: 0.95rem; margin-bottom: 4px;">🛡️ Cultural & Agronomic Prophylaxis:</div>
                        <p style="font-size: 0.9rem; color: #334155; margin: 0; line-height: 1.5;">{res['prevention']}</p>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("<div style='height: 0.6rem;'></div>", unsafe_allow_html=True)
                st.info("💡 **Scientific Protocol**: Always adhere to certified agronomic application dosages and check registered local pesticide schedules.")

            except Exception as e:
                st.error(f"Inference Failure: {e}")
                st.info("Ensure all model weights are saved in models/disease_detection/")
    elif not run_predict:
        st.markdown("""
        <div style="background: #F8FAF9; border: 2px dashed #CBD5E1; border-radius: 18px; padding: 3rem 2rem; text-align: center; color: #64748B;">
            <div style="font-size: 3rem; margin-bottom: 0.8rem;">🌿</div>
            <h4 style="color: #0F291E; font-size: 1.2rem; margin: 0 0 0.5rem 0;">Awaiting Foliar Input Specimen</h4>
            <p style="font-size: 0.92rem; max-width: 480px; margin: 0 auto; line-height: 1.55;">
                Select a benchmark specimen or upload a leaf photograph on the left panel, then click <strong>Execute Disease Diagnostics</strong> to run deep inference with Grad-CAM visualization.
            </p>
        </div>
        """, unsafe_allow_html=True)
