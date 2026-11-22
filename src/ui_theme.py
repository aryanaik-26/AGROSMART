"""
AGROSMART AI - High Performance UI & Graphics Design System
Provides modern glassmorphic styling, animated badges, sleek components,
and responsive typography for Streamlit.
"""

import streamlit as st
from pathlib import Path
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent
ASSETS_DIR = BASE_DIR / "assets"

def get_asset_image(name: str):
    """Safely retrieves a PIL Image from the assets directory if it exists."""
    p = ASSETS_DIR / name
    if p.exists():
        try:
            return Image.open(p)
        except Exception:
            return None
    return None

def inject_custom_css():
    """Injects high-end, responsive agriculture-tech CSS into Streamlit."""
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

        /* Global Font and Base Styles */
        html, body, [class*="css"], .stMarkdown, .stText {
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
            color: #0F291E;
        }

        h1, h2, h3, h4, h5, h6 {
            font-family: 'Outfit', sans-serif !important;
            letter-spacing: -0.02em;
            color: #0F291E;
            font-weight: 700;
        }

        /* Background */
        .stApp {
            background: linear-gradient(180deg, #F5FAF7 0%, #FFFFFF 35%, #F4F8F5 100%);
        }

        /* Modern Sidebar */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #071C13 0%, #0D2D1F 55%, #15422E 100%) !important;
            border-right: 1px solid rgba(16, 185, 129, 0.18);
        }

        [data-testid="stSidebar"] * {
            color: #E2ECE6 !important;
        }

        [data-testid="stSidebar"] hr {
            border-color: rgba(255, 255, 255, 0.1) !important;
        }

        [data-testid="stSidebarNav"] span {
            font-weight: 500;
            font-size: 0.95rem;
            letter-spacing: 0.01em;
        }

        /* Modern Hero Banner */
        .hero-banner {
            position: relative;
            background: linear-gradient(135deg, #052115 0%, #0A3422 45%, #145338 85%, #1E704C 100%);
            border: 1px solid rgba(52, 211, 153, 0.35);
            border-radius: 20px;
            padding: 2.4rem 2.8rem;
            color: #FFFFFF;
            overflow: hidden;
            box-shadow: 0 20px 45px -15px rgba(5, 33, 21, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.15);
            margin-bottom: 2rem;
        }

        .hero-title {
            font-size: 2.8rem;
            font-weight: 800;
            background: linear-gradient(120deg, #FFFFFF 0%, #D1FAE5 50%, #A7F3D0 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.6rem;
            line-height: 1.15;
        }

        .hero-subtitle {
            font-size: 1.12rem;
            color: #E2ECE6;
            max-width: 880px;
            line-height: 1.65;
            font-weight: 400;
            margin-bottom: 1.2rem;
        }

        /* Pulsing Status Chips */
        .status-chip {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 6px 14px;
            border-radius: 9999px;
            font-size: 0.82rem;
            font-weight: 600;
            letter-spacing: 0.03em;
            background: rgba(16, 185, 129, 0.18);
            border: 1px solid rgba(52, 211, 153, 0.4);
            color: #D1FAE5 !important;
            backdrop-filter: blur(8px);
        }

        .pulse-dot {
            width: 8px;
            height: 8px;
            background-color: #10B981;
            border-radius: 50%;
            box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7);
            animation: pulse-ring 2s infinite cubic-bezier(0.4, 0, 0.6, 1);
        }

        @keyframes pulse-ring {
            0% {
                transform: scale(0.95);
                box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7);
            }
            70% {
                transform: scale(1);
                box-shadow: 0 0 0 8px rgba(16, 185, 129, 0);
            }
            100% {
                transform: scale(0.95);
                box-shadow: 0 0 0 0 rgba(16, 185, 129, 0);
            }
        }

        /* Glassmorphic Metric Cards */
        .glass-metric {
            position: relative;
            background: rgba(255, 255, 255, 0.85);
            backdrop-filter: blur(14px);
            border: 1px solid rgba(226, 232, 240, 0.8);
            border-radius: 18px;
            padding: 1.5rem 1.6rem;
            box-shadow: 0 8px 24px -6px rgba(15, 41, 30, 0.06), 0 2px 6px -1px rgba(15, 41, 30, 0.03);
            transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
            overflow: hidden;
        }

        .glass-metric:hover {
            transform: translateY(-4px);
            box-shadow: 0 16px 32px -8px rgba(16, 185, 129, 0.18), 0 4px 12px -2px rgba(15, 41, 30, 0.06);
            border-color: rgba(16, 185, 129, 0.4);
        }

        .glass-metric::before {
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: linear-gradient(90deg, #10B981, #34D399, #F59E0B);
            opacity: 0.9;
        }

        .metric-icon-wrap {
            width: 44px;
            height: 44px;
            border-radius: 12px;
            background: linear-gradient(135deg, rgba(16, 185, 129, 0.12) 0%, rgba(5, 150, 105, 0.2) 100%);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.4rem;
            margin-bottom: 0.9rem;
            border: 1px solid rgba(16, 185, 129, 0.2);
        }

        .metric-title {
            font-size: 0.82rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: #64748B;
            margin-bottom: 0.2rem;
        }

        .metric-number {
            font-family: 'Outfit', sans-serif;
            font-size: 2.2rem;
            font-weight: 800;
            color: #064E3B;
            letter-spacing: -0.02em;
            line-height: 1.1;
            margin-bottom: 0.3rem;
        }

        .metric-desc {
            font-size: 0.84rem;
            font-weight: 500;
            color: #059669;
            display: flex;
            align-items: center;
            gap: 4px;
        }

        /* Image Feature Cards */
        .feature-box-with-image {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 20px;
            overflow: hidden;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.04);
            transition: all 0.35s cubic-bezier(0.16, 1, 0.3, 1);
            height: 100%;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }

        .feature-box-with-image:hover {
            transform: translateY(-6px);
            border-color: #6EE7B7;
            box-shadow: 0 20px 38px -10px rgba(16, 185, 129, 0.2);
        }

        .feature-image-container {
            width: 100%;
            height: 160px;
            overflow: hidden;
            position: relative;
        }

        .feature-image-container img {
            width: 100%;
            height: 100%;
            object-fit: cover;
            transition: transform 0.5s ease;
        }

        .feature-box-with-image:hover .feature-image-container img {
            transform: scale(1.05);
        }

        .feature-image-overlay {
            position: absolute;
            bottom: 0;
            left: 0;
            right: 0;
            height: 60px;
            background: linear-gradient(180deg, transparent 0%, rgba(0, 0, 0, 0.6) 100%);
            display: flex;
            align-items: flex-end;
            padding: 8px 14px;
        }

        .feature-content {
            padding: 1.4rem;
            flex-grow: 1;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }

        .feature-title {
            font-size: 1.35rem;
            font-weight: 700;
            color: #0F291E;
            margin-bottom: 0.4rem;
        }

        .feature-desc {
            font-size: 0.92rem;
            color: #475569;
            line-height: 1.58;
            margin-bottom: 1.1rem;
        }

        .tag-pill {
            display: inline-block;
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 0.74rem;
            font-weight: 600;
            margin-right: 6px;
            margin-bottom: 6px;
            background: #F1F5F9;
            color: #334155;
            border: 1px solid #E2E8F0;
        }

        .tag-pill-emerald {
            background: #ECFDF5;
            color: #065F46;
            border-color: #A7F3D0;
        }

        .tag-pill-blue {
            background: #EFF6FF;
            color: #1E40AF;
            border-color: #BFDBFE;
        }

        .tag-pill-amber {
            background: #FFFBEB;
            color: #92400E;
            border-color: #FDE68A;
        }

        /* Enhanced Streamlit Buttons */
        .stButton > button {
            background: linear-gradient(135deg, #059669 0%, #10B981 100%) !important;
            color: white !important;
            font-family: 'Outfit', sans-serif !important;
            font-weight: 600 !important;
            font-size: 0.96rem !important;
            border: none !important;
            border-radius: 12px !important;
            padding: 0.65rem 1.4rem !important;
            box-shadow: 0 4px 14px rgba(16, 185, 129, 0.3) !important;
            transition: all 0.25s ease !important;
        }

        .stButton > button:hover {
            transform: translateY(-2px) scale(1.01) !important;
            box-shadow: 0 8px 20px rgba(16, 185, 129, 0.45) !important;
            background: linear-gradient(135deg, #047857 0%, #059669 100%) !important;
        }

        /* Custom Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background-color: #F1F5F9;
            padding: 6px;
            border-radius: 14px;
            border: 1px solid #E2E8F0;
        }

        .stTabs [data-baseweb="tab"] {
            height: 42px;
            border-radius: 10px;
            font-weight: 600;
            font-size: 0.9rem;
            color: #475569;
            background-color: transparent;
            border: none;
            padding: 0 18px;
            transition: all 0.2s ease;
        }

        .stTabs [aria-selected="true"] {
            background-color: #FFFFFF !important;
            color: #065F46 !important;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08) !important;
        }

        .fancy-divider {
            height: 1px;
            background: linear-gradient(90deg, transparent 0%, rgba(16, 185, 129, 0.3) 30%, rgba(16, 185, 129, 0.3) 70%, transparent 100%);
            margin: 2.2rem 0;
            border: none;
        }

        .arch-diagram-card {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 18px;
            padding: 1.8rem;
            box-shadow: 0 4px 16px rgba(0,0,0,0.04);
            margin-bottom: 2rem;
        }

        /* Custom Scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        ::-webkit-scrollbar-track {
            background: #F1F5F9;
        }
        ::-webkit-scrollbar-thumb {
            background: #CBD5E1;
            border-radius: 4px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: #94A3B8;
        }
    </style>
    """, unsafe_allow_html=True)


def render_page_header(title: str, subtitle: str, icon: str = "🌱", badge: str = "AGROSMART AI v2.4"):
    """Renders a visually striking page header with badges and gradient text."""
    st.markdown(f"""
    <div style="margin-bottom: 1.8rem;">
        <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 10px; margin-bottom: 0.5rem;">
            <div style="display: flex; align-items: center; gap: 12px;">
                <span style="font-size: 2.4rem;">{icon}</span>
                <h1 style="font-size: 2.3rem; font-weight: 800; color: #064E3B; margin: 0; line-height: 1.1;">{title}</h1>
            </div>
            <div class="status-chip">
                <span class="pulse-dot"></span>
                <span>{badge}</span>
            </div>
        </div>
        <p style="font-size: 1.05rem; color: #475569; max-width: 900px; margin: 0; line-height: 1.6;">
            {subtitle}
        </p>
    </div>
    <hr class="fancy-divider">
    """, unsafe_allow_html=True)


def render_hero():
    """Renders the main flagship homepage hero banner with image and graphics."""
    hero_img = get_asset_image("precision_ai_hero.jpg")
    
    col_text, col_img = st.columns([1.3, 1.0], gap="large")
    with col_text:
        st.markdown("""
        <div style="padding: 1rem 0;">
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 1rem;">
                <div class="status-chip" style="background: rgba(16, 185, 129, 0.15); border-color: rgba(52, 211, 153, 0.4); color: #065F46 !important;">
                    <span class="pulse-dot"></span>
                    <span style="color: #065F46 !important;">PRODUCTION SYSTEM ACTIVE</span>
                </div>
                <span style="font-size: 0.85rem; color: #059669; font-weight: 700; letter-spacing: 0.05em;">PRECISION AGRI-TECH 2.4</span>
            </div>
            <div style="font-family: 'Outfit', sans-serif; font-size: 3rem; font-weight: 800; color: #064E3B; line-height: 1.1; margin-bottom: 0.8rem;">
                🌱 AGROSMART AI
            </div>
            <div style="font-size: 1.15rem; color: #334155; line-height: 1.65; margin-bottom: 1.4rem;">
                Next-Generation Intelligent Agriculture Platform powered by Deep Convolutional Vision, High-Dimensional Multi-Task Ensembles, and Game-Theoretic Explainable AI (SHAP & Grad-CAM) for Sustainable Farming.
            </div>
            <div style="display: flex; flex-wrap: wrap; gap: 10px;">
                <span class="tag-pill tag-pill-emerald" style="font-size: 0.85rem; padding: 6px 14px;">🍃 38 Disease Classes</span>
                <span class="tag-pill tag-pill-blue" style="font-size: 0.85rem; padding: 6px 14px;">🌾 22 Crop Recommendations</span>
                <span class="tag-pill tag-pill-amber" style="font-size: 0.85rem; padding: 6px 14px;">📈 95% Yield Prediction Bounds</span>
                <span class="tag-pill" style="font-size: 0.85rem; padding: 6px 14px; background: #FAF5FF; color: #7C3AED; border-color: #E9D5FF;">🧠 Mathematical XAI</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col_img:
        if hero_img is not None:
            st.image(hero_img, caption="Autonomous Drone Monitoring & Holographic Crop Health Analytics", use_container_width=True)
        else:
            st.image("https://images.unsplash.com/photo-1625246333195-78d9c38ad449?auto=format&fit=crop&w=1000&q=80", caption="Precision Agriculture Smart Farming", use_container_width=True)


def render_sidebar_branding():
    """Renders a sleek luxury brand block in the Streamlit sidebar with image."""
    sidebar_img = get_asset_image("soil_seedling.jpg")
    
    with st.sidebar:
        if sidebar_img is not None:
            st.image(sidebar_img, caption="AGROSMART Intelligence", use_container_width=True)
        else:
            st.image("https://images.unsplash.com/photo-1523348837708-15d4a09cfac2?auto=format&fit=crop&w=400&q=80", use_container_width=True)

        st.markdown("""
        <div style="text-align: center; padding: 0.6rem 0.5rem 0.8rem 0.5rem;">
            <h2 style="color: #FFFFFF !important; font-size: 1.4rem; font-weight: 800; margin: 0; letter-spacing: -0.01em;">AGROSMART AI</h2>
            <div style="display: inline-flex; align-items: center; gap: 6px; margin-top: 6px;">
                <span class="pulse-dot"></span>
                <span style="font-size: 0.78rem; color: #6EE7B7 !important; font-weight: 600; letter-spacing: 0.05em; text-transform: uppercase;">Engine v2.4 Live</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<hr style='margin: 0.5rem 0 1.2rem 0; border-color: rgba(255,255,255,0.1);'>", unsafe_allow_html=True)
