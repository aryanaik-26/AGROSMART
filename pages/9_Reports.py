"""
AGROSMART AI - Page 9: Reports & Export Center
Enhanced UI & Visual Graphics Edition
"""

import streamlit as st
import pandas as pd
import json
import io
import zipfile
from pathlib import Path
from datetime import datetime
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from config import METRICS_DIR, GRAPHS_DIR, GENERATED_REPORTS_DIR
from src.ui_theme import inject_custom_css, render_page_header, render_sidebar_branding

st.set_page_config(page_title="Reports & Downloads | AGROSMART AI", page_icon="📑", layout="wide")

# Inject Design System
inject_custom_css()
render_sidebar_branding()

# Render High-Tech Page Header
render_page_header(
    title="System Reports & Technical Artifacts",
    subtitle="Export empirical validation matrices, high-resolution agronomic visual plots, and automated academic project reports for viva voce.",
    icon="📑",
    badge="EXPORT SUITE"
)

# 1. GENERATE PROJECT SUMMARY REPORT
def create_project_summary_report() -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    crop_path = METRICS_DIR / "crop_recommendation_metrics.json"
    yield_path = METRICS_DIR / "crop_yield_metrics.json"
    dl_path = METRICS_DIR / "disease_detection_metrics.json"

    crop_metrics = json.loads(crop_path.read_text()) if crop_path.exists() else {}
    yield_metrics = json.loads(yield_path.read_text()) if yield_path.exists() else {}
    dl_metrics = json.loads(dl_path.read_text()) if dl_path.exists() else {}

    report = f"""# AGROSMART AI - SYSTEM EVALUATION & PERFORMANCE REPORT
**Generated:** {timestamp}  
**Project:** Smart Crop Disease & Yield Prediction System Using Machine Learning and Deep Learning  
**Standard:** IEEE & Scopus Quality Experimental Methodology  

---

## 1. EXECUTIVE SUMMARY
AGROSMART AI is an end-to-end intelligent agricultural decision-support platform engineered to solve yield bottlenecks, disease diagnosis, and crop allocation.
- **Module 1**: Leaf Disease Classification via Deep Convolutional Networks (Custom CNN, MobileNetV2, EfficientNetB0) with Grad-CAM visual interpretability.
- **Module 2**: Crop Yield Regression via Ensemble Stacking (XGBoost, Random Forest, SVR) with SHAP attribution.
- **Module 3**: Crop Recommendation via Multi-Class Classifiers (22 benchmark crops) returning Top-3 probabilistic rankings.

---

## 2. EMPIRICAL MODEL PERFORMANCE BENCHMARKS

### A. Crop Recommendation Classification Suite (8 Algorithms)
| Algorithm | Accuracy | Macro-F1 | Precision (Macro) | Recall (Macro) | Latency (ms/sample) |
| :--- | :---: | :---: | :---: | :---: | :---: |
"""
    for name, m in crop_metrics.items():
        report += f"| {name} | {m.get('accuracy', 0)*100:.2f}% | {m.get('f1_macro', 0):.4f} | {m.get('precision_macro', 0):.4f} | {m.get('recall_macro', 0):.4f} | {m.get('inference_latency_ms', 0):.2f} |\n"

    report += """
### B. Crop Yield Regression Suite (8 Algorithms)
| Regressor | R² Score | Adjusted R² | RMSE (t/ha) | MAE (t/ha) | Latency (ms/sample) |
| :--- | :---: | :---: | :---: | :---: | :---: |
"""
    for name, m in yield_metrics.items():
        report += f"| {name} | {m.get('r2_score', 0):.4f} | {m.get('adjusted_r2', 0):.4f} | {m.get('rmse', 0):.3f} | {m.get('mae', 0):.3f} | {m.get('inference_latency_ms', 0):.2f} |\n"

    report += """
### C. Deep Learning Disease Detection Suite (3 Architectures)
| Architecture | Test Accuracy | Test Loss | Macro-F1 | Training Time (s) |
| :--- | :---: | :---: | :---: | :---: |
"""
    for name, m in dl_metrics.items():
        report += f"| {name} | {m.get('test_accuracy', 0)*100:.2f}% | {m.get('test_loss', 0):.4f} | {m.get('f1_macro', 0):.4f} | {m.get('training_time_seconds', 0):.2f} |\n"

    report += """
---

## 3. EXPLAINABLE AI (XAI) SUMMARY
- **Grad-CAM**: Applied to the terminal convolutional layers, localizing discriminatory necrotic spots and blights on leaf blades.
- **SHAP (Shapley Additive Explanations)**: Employs cooperative game-theoretic formulations to decompose individual yield predictions and crop rankings into exact additive contributions from soil chemical and climatic features.

---
**Verified for Academic Submission & Viva Voce.**
"""
    return report

summary_report_text = create_project_summary_report()

col_d1, col_d2 = st.columns([1, 1.2], gap="large")

with col_d1:
    st.markdown("""
    <div style="background: white; border: 1px solid #E2E8F0; border-radius: 18px; padding: 1.5rem; box-shadow: 0 4px 16px rgba(0,0,0,0.03); margin-bottom: 1.2rem;">
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 0.6rem;">
            <span style="font-size: 1.3rem;">📥</span>
            <h3 style="font-size: 1.25rem; margin: 0; color: #064E3B;">Download Packages</h3>
        </div>
        <p style="font-size: 0.88rem; color: #64748B; margin-bottom: 1.2rem;">
            Export serialized performance metrics, high-resolution figure archives, and compiled project summaries.
        </p>
    """, unsafe_allow_html=True)

    # Download JSON Metrics
    crop_path = METRICS_DIR / "crop_recommendation_metrics.json"
    if crop_path.exists():
        st.download_button(
            "⬇️ Crop Recommendation Metrics (JSON)",
            data=crop_path.read_bytes(),
            file_name="crop_recommendation_metrics.json",
            mime="application/json",
            use_container_width=True
        )

    yield_path = METRICS_DIR / "crop_yield_metrics.json"
    if yield_path.exists():
        st.download_button(
            "⬇️ Yield Prediction Metrics (JSON)",
            data=yield_path.read_bytes(),
            file_name="crop_yield_metrics.json",
            mime="application/json",
            use_container_width=True
        )

    dl_path = METRICS_DIR / "disease_detection_metrics.json"
    if dl_path.exists():
        st.download_button(
            "⬇️ Deep Learning Vision Metrics (JSON)",
            data=dl_path.read_bytes(),
            file_name="disease_detection_metrics.json",
            mime="application/json",
            use_container_width=True
        )

    # Zip all graphs
    graph_files = list(GRAPHS_DIR.glob("*.png"))
    if graph_files:
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zf:
            for gf in graph_files:
                zf.write(gf, arcname=gf.name)
        zip_buffer.seek(0)

        st.download_button(
            f"📦 High-Res Plots ({len(graph_files)} PNGs in ZIP)",
            data=zip_buffer.getvalue(),
            file_name="agrosmart_eda_graphs.zip",
            mime="application/zip",
            use_container_width=True
        )

    st.download_button(
        "📄 Technical Project Summary (Markdown)",
        data=summary_report_text,
        file_name="AGROSMART_AI_Project_Summary_Report.md",
        mime="text/markdown",
        use_container_width=True
    )

    st.markdown("</div>", unsafe_allow_html=True)

with col_d2:
    st.markdown("""
    <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.8rem;">
        <h3 style="font-size: 1.35rem; margin: 0; color: #064E3B;">Technical Summary Document</h3>
        <span class="status-chip" style="background: #F1F5F9; border-color: #E2E8F0; color: #475569 !important;">
            Markdown Export
        </span>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("👁️ View Live Markdown Document", expanded=True):
        st.markdown(summary_report_text)
