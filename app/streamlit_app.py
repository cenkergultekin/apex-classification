# -*- coding: utf-8 -*-
"""Streamlit deployment interface for APEX AI Student Impact classifier.

This app features:
1. Burnout Risk Predictor (Form, real-time prediction, gauges, and advisor suggestions)
2. Project Analytics & Data Story (Interactive EDA visualizations, metrics, team cards)

Skorlar ve confusion matrix `models/best_model_metadata.json` dosyasından dinamik
okunur; bu sayede model yeniden eğitildiğinde arayüz otomatik güncel kalır.
"""

import os
import sys
from pathlib import Path
import json
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# Setup dynamic paths relative to this script
APP_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = APP_DIR.parent.resolve()

# Inject project root to sys.path so we can import app.preprocessing
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.preprocessing import ProductionPreprocessor

# Page configuration
st.set_page_config(
    page_title="APEX Öğrenci Tükenmişlik Risk Analizi",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ════════════════════════════════════════════════════════════════════
# Tema & Stil — Glassmorphism + Premium Dark (emojisiz)
# ════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Space+Grotesk:wght@500;600;700&display=swap');

    :root {
        --accent: #a78bfa;
        --accent-2: #22d3ee;
        --bg: #05070f;
        --surface: rgba(255, 255, 255, 0.03);
        --border: rgba(124, 58, 237, 0.22);
        --text: #f8fafc;
        --muted: #94a3b8;
    }

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* Aurora background */
    .stApp {
        background-color: var(--bg);
        background-image:
            radial-gradient(at 0% 0%, rgba(124, 58, 237, 0.12) 0px, transparent 45%),
            radial-gradient(at 100% 0%, rgba(34, 211, 238, 0.08) 0px, transparent 45%),
            radial-gradient(at 50% 100%, rgba(99, 102, 241, 0.08) 0px, transparent 50%);
        background-attachment: fixed;
    }

    /* ── Hero (çerçevesiz) ── */
    .hero-wrap { padding: 2px 2px 4px; margin-bottom: 16px; }
    .main-title {
        background: linear-gradient(120deg, #e9d5ff 0%, #a78bfa 45%, #22d3ee 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        background-clip: text;
        font-family: 'Space Grotesk', 'Inter', sans-serif;
        font-size: 1.9rem; font-weight: 700;
        letter-spacing: -0.5px; line-height: 1.1; margin: 0 0 8px 0;
    }
    .subtitle {
        color: #cbd5e1; font-size: 0.9rem; font-weight: 400;
        max-width: 880px; line-height: 1.5; margin: 0;
    }

    /* ── Section headers ── */
    .section-head {
        font-family: 'Space Grotesk', 'Inter', sans-serif;
        color: var(--text); font-size: 1.32rem; font-weight: 700;
        margin: 26px 0 14px 0; display: flex; align-items: center; gap: 12px;
    }
    .section-head .bar {
        width: 4px; height: 22px; border-radius: 4px;
        background: linear-gradient(180deg, var(--accent), var(--accent-2));
    }

    /* ── Cards ── */
    div[data-testid="stForm"], div[data-testid="stContainerBordered"] {
        background: linear-gradient(135deg, rgba(30,27,75,0.40) 0%, rgba(15,23,42,0.40) 100%) !important;
        border: 1px solid var(--border) !important;
        border-radius: 18px !important;
        box-shadow: 0 8px 32px rgba(0,0,0,0.37) !important;
        backdrop-filter: blur(10px) !important;
        -webkit-backdrop-filter: blur(10px) !important;
        padding: 14px 18px !important;
        margin-bottom: 8px !important;
    }

    .card-title {
        color: var(--text); font-size: 1.05rem; font-weight: 700;
        margin-bottom: 8px; padding-bottom: 7px;
        border-bottom: 1px solid rgba(255,255,255,0.08);
    }

    /* ── Form group label ── */
    .form-group-label {
        font-family: 'Space Grotesk', 'Inter', sans-serif;
        color: #c4b5fd; font-size: 0.92rem; font-weight: 700;
        letter-spacing: 0.6px; text-transform: uppercase;
        margin: 2px 0 16px; padding-bottom: 9px;
        border-bottom: 1px solid rgba(124,58,237,0.18);
    }

    /* ── Input aesthetics ── */
    div[data-testid="stWidgetLabel"] p,
    div[data-testid="stWidgetLabel"] label {
        color: #cbd5e1 !important;
        font-weight: 600 !important;
        font-size: 0.86rem !important;
        letter-spacing: 0.2px;
    }
    /* selectbox + text/number input control */
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    div[data-baseweb="base-input"] {
        background: rgba(10,14,26,0.66) !important;
        border: 1px solid rgba(148,163,184,0.18) !important;
        border-radius: 12px !important;
        transition: border-color .18s ease, box-shadow .18s ease !important;
    }
    div[data-baseweb="select"] > div:hover {
        border-color: rgba(124,58,237,0.5) !important;
    }
    div[data-baseweb="select"] > div:focus-within {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 3px rgba(124,58,237,0.16) !important;
    }
    /* dropdown menu */
    ul[role="listbox"] {
        background: #0f172a !important;
        border: 1px solid rgba(124,58,237,0.25) !important;
        border-radius: 12px !important;
        box-shadow: 0 14px 40px rgba(0,0,0,0.5) !important;
    }
    ul[role="listbox"] li:hover { background: rgba(124,58,237,0.16) !important; }
    /* slider value bubble + endpoints */
    div[data-testid="stSlider"] [data-testid="stThumbValue"] {
        color: var(--accent) !important; font-weight: 700 !important;
    }
    div[data-testid="stSlider"] [data-testid="stTickBarMin"],
    div[data-testid="stSlider"] [data-testid="stTickBarMax"] {
        color: var(--muted) !important;
    }
    /* checkbox label */
    div[data-testid="stCheckbox"] label p { color: #cbd5e1 !important; font-size: 0.86rem !important; }

    /* ── Primary button (animasyonlu, dikkat çekici) ── */
    @keyframes apexFlow {
        0% { background-position: 0% 50%; }
        100% { background-position: 200% 50%; }
    }
    @keyframes apexPulse {
        0%, 100% { box-shadow: 0 8px 24px rgba(124,58,237,0.35), 0 0 0 0 rgba(34,211,238,0.45); }
        50% { box-shadow: 0 10px 30px rgba(34,211,238,0.45), 0 0 0 7px rgba(124,58,237,0.0); }
    }
    div[data-testid="stButton"] > button[kind="primary"] {
        background: linear-gradient(90deg, #7c3aed, #4f46e5, #22d3ee, #7c3aed) !important;
        background-size: 200% 100% !important;
        border: none !important;
        border-radius: 12px !important;
        color: #ffffff !important;
        font-weight: 800 !important; letter-spacing: 1px !important;
        font-size: 1.02rem !important;
        padding: 13px 0 !important;
        animation: apexFlow 4s linear infinite, apexPulse 2.2s ease-in-out infinite !important;
        transition: transform .15s ease !important;
    }
    div[data-testid="stButton"] > button[kind="primary"]:hover {
        transform: translateY(-2px) scale(1.012) !important;
        filter: brightness(1.08) !important;
    }
    div[data-testid="stButton"] > button[kind="primary"]:active { transform: translateY(0) scale(0.99) !important; }

    /* ── Metric cards ── */
    .metric-box {
        position: relative;
        background: linear-gradient(160deg, rgba(20,18,48,0.7) 0%, rgba(10,14,26,0.7) 100%);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 16px; padding: 18px 16px 16px; text-align: left;
        box-shadow: 0 6px 18px rgba(0,0,0,0.28);
        transition: transform .18s ease, border-color .18s ease, box-shadow .18s ease;
        overflow: hidden;
    }
    .metric-box::before {
        content: ""; position: absolute; top: 0; left: 0; right: 0; height: 3px;
        background: linear-gradient(90deg, var(--accent), var(--accent-2));
        opacity: 0.85;
    }
    .metric-box:hover {
        transform: translateY(-3px);
        border-color: rgba(124,58,237,0.45);
        box-shadow: 0 12px 30px rgba(99,102,241,0.18);
    }
    .metric-label {
        color: var(--muted); font-size: 0.72rem; font-weight: 700;
        text-transform: uppercase; letter-spacing: 1.2px; margin: 0 0 6px;
    }
    .metric-value {
        color: var(--text); font-family: 'Space Grotesk', sans-serif;
        font-size: 1.95rem; font-weight: 700; margin: 0; line-height: 1;
    }
    .metric-sub { color: var(--muted); font-size: 0.74rem; margin: 6px 0 0; }

    /* ── Risk badges ── */
    .risk-badge-high {
        background: linear-gradient(135deg, rgba(239,68,68,0.15), rgba(220,38,38,0.28));
        border: 1px solid rgba(239,68,68,0.5); color: #fca5a5;
        padding: 6px 16px; border-radius: 20px; font-weight: 700;
        font-size: 0.9em; letter-spacing: 0.5px; display: inline-block;
        box-shadow: 0 0 18px rgba(239,68,68,0.25);
    }
    .risk-badge-low {
        background: linear-gradient(135deg, rgba(16,185,129,0.15), rgba(5,150,105,0.28));
        border: 1px solid rgba(16,185,129,0.5); color: #a7f3d0;
        padding: 6px 16px; border-radius: 20px; font-weight: 700;
        font-size: 0.9em; letter-spacing: 0.5px; display: inline-block;
        box-shadow: 0 0 18px rgba(16,185,129,0.25);
    }

    /* ── Advisor ── */
    .advisor-point {
        display: flex; gap: 12px; align-items: flex-start; margin-bottom: 12px;
        font-size: 0.92rem; color: #e2e8f0; line-height: 1.6;
    }
    .advisor-icon { color: var(--accent); font-weight: 900; flex-shrink: 0; margin-top: 2px; }

    /* ── Team cards ── */
    .team-card {
        background: linear-gradient(160deg, rgba(24,22,54,0.6) 0%, rgba(10,14,26,0.6) 100%);
        border: 1px solid rgba(255,255,255,0.07); border-radius: 16px;
        padding: 22px 16px; text-align: center; height: 100%;
        transition: transform .18s ease, border-color .18s ease;
    }
    .team-card:hover { transform: translateY(-3px); border-color: rgba(124,58,237,0.45); }
    .team-avatar {
        width: 54px; height: 54px; margin: 0 auto 12px;
        border-radius: 50%; display: flex; align-items: center; justify-content: center;
        font-family: 'Space Grotesk', sans-serif; font-size: 1.5rem; font-weight: 700;
        color: #f8fafc;
        background: linear-gradient(135deg, rgba(124,58,237,0.9), rgba(34,211,238,0.7));
        border: 1px solid rgba(255,255,255,0.15);
        box-shadow: 0 6px 18px rgba(99,102,241,0.32);
    }
    .team-name { color: var(--text); font-size: 1.08rem; font-weight: 800; }
    .team-role {
        display: inline-block; margin: 6px 0 4px;
        font-size: 0.72rem; font-weight: 700; letter-spacing: 0.5px;
        color: #c4b5fd; background: rgba(124,58,237,0.14);
        border: 1px solid rgba(124,58,237,0.28); padding: 3px 10px; border-radius: 999px;
    }
    .team-phase { color: var(--accent-2); font-size: 0.8rem; font-weight: 600; margin: 6px 0 8px; }
    .team-desc { color: var(--muted); font-size: 0.82rem; line-height: 1.55; margin: 0; }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(185deg, #16122e 0%, #0c0a1d 55%, #080b16 100%) !important;
        border-right: 1px solid rgba(124,58,237,0.20) !important;
    }
    section[data-testid="stSidebar"] hr {
        border-color: rgba(124,58,237,0.18) !important;
    }
    .sb-logo {
        font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 2rem;
        background: linear-gradient(120deg, #a78bfa, #22d3ee);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        letter-spacing: 1px; margin: 0;
    }
    .sb-section {
        color: #c4b5fd; font-size: 0.78rem; font-weight: 700;
        letter-spacing: 1px; text-transform: uppercase; margin: 4px 0 10px;
    }
    .sb-member { font-size: 0.88rem; color: #e2e8f0; margin: 0 0 9px; line-height: 1.35; }
    .sb-member small { color: var(--muted); }
    .sb-kv { display: flex; justify-content: space-between; font-size: 0.86rem; margin: 5px 0; }
    .sb-kv .k { color: var(--muted); }
    .sb-kv .v { color: var(--text); font-weight: 600; }

    /* ── Tabs (biri en solda, biri en sağda) ── */
    div[data-baseweb="tab-list"] {
        width: 100%;
        justify-content: space-between;
        gap: 0;
        background: transparent;
        border-bottom: none;
        margin-bottom: 8px;
    }
    button[data-baseweb="tab"] {
        font-weight: 600 !important; font-size: 0.95rem !important;
        border-radius: 10px !important; padding: 9px 22px !important;
        color: #94a3b8 !important; background: transparent !important;
        transition: all .18s ease !important;
    }
    button[data-baseweb="tab"]:hover {
        color: #e2e8f0 !important; background: rgba(124,58,237,0.10) !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #f8fafc !important;
        background: linear-gradient(135deg, rgba(124,58,237,0.32), rgba(34,211,238,0.18)) !important;
        border: 1px solid rgba(124,58,237,0.4) !important;
        box-shadow: 0 4px 14px rgba(99,102,241,0.22) !important;
    }
    div[data-baseweb="tab-highlight"], div[data-baseweb="tab-border"] { display: none !important; }

    /* ── Segmented control (1-10 ölçekler için tıklanabilir pill'ler) ── */
    div[data-testid="stSegmentedControl"] button {
        border-radius: 10px !important;
        border: 1px solid rgba(148,163,184,0.18) !important;
        color: #cbd5e1 !important;
        transition: all .15s ease !important;
    }
    div[data-testid="stSegmentedControl"] button:hover {
        border-color: rgba(124,58,237,0.5) !important;
        background: rgba(124,58,237,0.10) !important;
    }
    div[data-testid="stSegmentedControl"] button[aria-checked="true"],
    div[data-testid="stSegmentedControl"] button[kind="segmented_controlActive"] {
        background: linear-gradient(135deg, #7c3aed, #4f46e5) !important;
        border-color: rgba(167,139,250,0.6) !important;
        color: #ffffff !important;
        box-shadow: 0 4px 12px rgba(99,102,241,0.3) !important;
    }

    /* ── Modal / Dialog ── */
    div[data-baseweb="modal"] > div:first-child {
        backdrop-filter: blur(8px) !important;
        -webkit-backdrop-filter: blur(8px) !important;
        background: rgba(5,7,15,0.55) !important;
    }
    div[role="dialog"] {
        background: linear-gradient(160deg, rgba(20,18,48,0.97), rgba(8,12,26,0.97)) !important;
        border: 1px solid rgba(124,58,237,0.35) !important;
        border-radius: 22px !important;
        box-shadow: 0 28px 80px rgba(0,0,0,0.62) !important;
    }
    .modal-class {
        text-align: center; padding: 6px 0 4px;
    }
    .modal-class .lbl {
        color: #a78bfa; font-weight: 800; font-size: 0.78rem;
        letter-spacing: 2px; margin-bottom: 12px;
    }

    /* Hide Streamlit chrome */
    [data-testid="stHeader"], [data-testid="stHeaderDeployButton"],
    #MainMenu, footer, .stDeployButton, [data-testid="stToolbar"] {
        display: none !important;
    }
    .block-container { padding-top: 0.4rem !important; padding-bottom: 0.4rem !important; }
    /* ── Kompakt: tek ekrana sığdır ── */
    div[data-testid="stVerticalBlock"] { gap: 0.28rem !important; }
    div[data-testid="stElementContainer"] { margin: 0 !important; }
    /* widget etiketleri */
    div[data-testid="stWidgetLabel"] { margin-bottom: 1px !important; }
    div[data-testid="stWidgetLabel"] p { font-size: 0.8rem !important; }
    /* selectbox yüksekliğini kıs */
    div[data-baseweb="select"] > div { min-height: 34px !important; }
    /* slider alt boşluk/tick'leri kıs */
    div[data-testid="stSlider"] { padding-bottom: 0 !important; }
    div[data-testid="stSlider"] > div { padding-bottom: 0 !important; }
    div[data-testid="stTickBar"] { display: none !important; }
    /* segmented control pill'leri kısalt */
    div[data-testid="stSegmentedControl"] button { padding-top: 3px !important; padding-bottom: 3px !important; min-height: 32px !important; }
    /* checkbox kompakt */
    div[data-testid="stCheckbox"] { margin: 2px 0 !important; }
    /* form grup etiketi kompakt */
    .form-group-label { margin: 0 0 8px !important; padding-bottom: 6px !important; }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
# Artifact yükleme
# ════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_model_artifacts():
    prep_path = PROJECT_ROOT / 'models' / 'artifacts' / 'full_preprocessor.pkl'
    package_path = PROJECT_ROOT / 'models' / 'best_model_package.joblib'
    try:
        preprocessor = joblib.load(prep_path)
        package = joblib.load(package_path)
        return preprocessor, package
    except Exception as e:
        st.error(f"Model yüklenirken bir hata oluştu. Lütfen model eğitimini kontrol edin.\nHata: {e}")
        return None, None


@st.cache_data
def load_metadata():
    """Skorları sabit yazmak yerine metadata dosyasından dinamik okur."""
    meta_path = PROJECT_ROOT / 'models' / 'best_model_metadata.json'
    try:
        with open(meta_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


preprocessor, model_package = load_model_artifacts()
metadata = load_metadata()

# Metrikleri tek noktadan türet (test_eval_decision = saklı test seti, tek-sefer ölçüm)
if metadata and 'test_eval_decision' in metadata:
    _te = metadata['test_eval_decision']
    METRICS = {
        'model_name': metadata.get('model_name', 'Gradient Boosting'),
        'feature_count': metadata.get('feature_count', 15),
        'threshold': _te.get('threshold', 0.53),
        'accuracy': _te.get('accuracy', float('nan')),
        'f1_macro': _te.get('f1_macro', float('nan')),
        'recall_macro': _te.get('recall_macro', float('nan')),
        'recall_yuksek': _te.get('recall_yuksek', float('nan')),
        'precision_yuksek': _te.get('precision_yuksek', float('nan')),
        'roc_auc': _te.get('roc_auc', float('nan')),
        'confusion_matrix': np.array(_te.get('confusion_matrix', [[0, 0], [0, 0]])),
    }
else:
    METRICS = None

# ── Ekip & rol verisi (README.md ile birebir senkron) ──
TEAM = [
    {
        'name': 'Feza', 'role': 'Veri Analisti',
        'phase': 'CRISP-DM Faz 1 & 2',
        'desc': 'Business Understanding ve detaylı Plotly EDA süreçlerini yürüttü; '
                'veri setindeki gizli ilişkileri ve sınıf dengesini ortaya çıkardı.',
    },
    {
        'name': 'Cenker', 'role': 'Veri Mühendisi',
        'phase': 'CRISP-DM Faz 3',
        'desc': 'Veri temizleme, öznitelik mühendisliği ve Lean Core (15 feature) '
                'production-ready ProductionPreprocessor / Pipeline mimarisini kurdu.',
    },
    {
        'name': 'Berkay', 'role': 'ML Mühendisi',
        'phase': 'CRISP-DM Faz 4 & 5',
        'desc': '10+ sınıflandırma algoritmasını stratified CV ile karşılaştırdı, '
                'Gradient Boosting üzerinde hiperparametre optimizasyonu ve nihai değerlendirmeyi yaptı.',
    },
    {
        'name': 'Ethem', 'role': 'Yazılım Mühendisi',
        'phase': 'CRISP-DM Faz 6',
        'desc': 'Modeli üretime aldı: Streamlit canlı tahmin arayüzü, leak-aware retraining '
                'akışı ve kullanıcı dostu analitik ekranların tasarımını tamamladı.',
    },
]


# ════════════════════════════════════════════════════════════════════
# Sidebar
# ════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("<p class='sb-logo'>APEX</p>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#94a3b8; font-size:0.85em; margin-top:10px; line-height:1.5;'>"
        "Akademik Tükenmişlik Risk Tahmin Sistemi</p>",
        unsafe_allow_html=True
    )

    st.markdown("---")
    st.markdown("<p class='sb-section'>Ekip & Sorumluluklar</p>", unsafe_allow_html=True)
    for m in TEAM:
        st.markdown(
            f"<p class='sb-member'><strong>{m['name']}</strong> · {m['role']}<br>"
            f"<small>{m['phase']}</small></p>",
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.markdown("<p class='sb-section'>Model Özeti</p>", unsafe_allow_html=True)
    if METRICS:
        st.markdown(
            f"<div class='sb-kv'><span class='k'>Model</span>"
            f"<span class='v'>{METRICS['model_name']}</span></div>"
            f"<div class='sb-kv'><span class='k'>Özellik</span>"
            f"<span class='v'>{METRICS['feature_count']}</span></div>"
            f"<div class='sb-kv'><span class='k'>Karar Eşiği</span>"
            f"<span class='v'>{METRICS['threshold']:.2f}</span></div>"
            f"<div class='sb-kv'><span class='k'>Test F1-Macro</span>"
            f"<span class='v'>{METRICS['f1_macro']:.3f}</span></div>"
            f"<div class='sb-kv'><span class='k'>Test ROC-AUC</span>"
            f"<span class='v'>{METRICS['roc_auc']:.3f}</span></div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown("Model metadata yüklenemedi.")


# ════════════════════════════════════════════════════════════════════
# Hero
# ════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero-wrap">
    <h1 class="main-title">Akademik Tükenmişlik Risk Analizi</h1>
    <p class="subtitle">
        Üretken yapay zeka kullanım örüntüleri, geleneksel çalışma alışkanlıkları ve sınav kaygısından
        hareketle öğrencilerin akademik tükenmişlik riskini <b>Düşük / Yüksek</b> olarak sınıflandıran
        bir karar destek sistemi.
    </p>
</div>
""", unsafe_allow_html=True)

# Main Application Tabs
tab_predict, tab_analytics = st.tabs(["Tükenmişlik Tahmini", "Proje Analitiği & Hikayesi"])

# ────────────────────────────────────────────────────────────────────
# TAB 1 — Tahmin
# ────────────────────────────────────────────────────────────────────
with tab_predict:
    if not model_package or not preprocessor:
        st.warning("Model dosyaları eksik veya uyumsuz olduğundan tahmin sekmesi devre dışı.")
    else:
        model = model_package['model']
        tuned_threshold = model_package['threshold_decision']['tuned']['threshold']

        @st.dialog("Tükenmişlik Analiz Sonucu", width="large")
        def show_result_dialog():
            a = st.session_state.get("analysis")
            if not a:
                return

            is_high = a["is_high_risk"]
            ph = a["prob_high"]
            thr = a["threshold"]

            # 1) Tahmin edilen sınıf — tam genişlik, üstte
            if is_high:
                badge = "<span class='risk-badge-high' style='font-size:1.5em; padding:10px 28px;'>YÜKSEK RİSK</span>"
                desc = "<p style='color:#fca5a5; font-size:0.92em; line-height:1.6; margin-top:14px;'>Öğrenci, akademik ve YZ kullanım parametrelerine göre ciddi düzeyde tükenmişlik riski altındadır.</p>"
            else:
                badge = "<span class='risk-badge-low' style='font-size:1.5em; padding:10px 28px;'>DÜŞÜK RİSK</span>"
                desc = "<p style='color:#a7f3d0; font-size:0.92em; line-height:1.6; margin-top:14px;'>Öğrencinin mevcut akademik ve YZ kullanım dengesi sağlıklıdır.</p>"

            st.markdown(
                f"<div class='modal-class'><div class='lbl'>TAHMİN EDİLEN SINIF</div>"
                f"<div>{badge}</div>{desc}</div>",
                unsafe_allow_html=True
            )

            # 2) Speed gauge — tam genişlik, altta (alt alta, eşit genişlik)
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=ph * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                number={'suffix': "%", 'font': {'size': 40, 'color': '#f8fafc'}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#475569", 'tickfont': {'color': '#94a3b8'}},
                    'bar': {'color': "#ef4444" if is_high else "#10b981"},
                    'bgcolor': "rgba(30, 41, 59, 0.4)",
                    'borderwidth': 1.5,
                    'bordercolor': "#475569",
                    'steps': [
                        {'range': [0, 35], 'color': 'rgba(16, 185, 129, 0.08)'},
                        {'range': [35, 70], 'color': 'rgba(245, 158, 11, 0.08)'},
                        {'range': [70, 100], 'color': 'rgba(239, 68, 68, 0.08)'}
                    ],
                    'threshold': {'line': {'color': "#ffffff", 'width': 3}, 'thickness': 0.75, 'value': thr * 100}
                }
            ))
            fig_gauge.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font={'color': "#f8fafc", 'family': "Inter"},
                height=260, margin=dict(l=20, r=20, t=30, b=10)
            )
            st.plotly_chart(fig_gauge, use_container_width=True)
            st.markdown(
                f"<p style='text-align:center; color:#94a3b8; font-size:0.8em; margin-top:-6px;'>"
                f"Yüksek risk olasılığı · karar eşiği {thr:.2f}</p>",
                unsafe_allow_html=True
            )

            # 3) Danışman önerileri — aşağı doğru açılan expander
            with st.expander("Akademik Danışman & Rehberlik Önerileri", expanded=False):
                w = a["weekly_ai_hours"]; dep = a["ai_dependency"]
                anx = a["exam_anxiety"]; trad = a["traditional_study_hours"]
                if is_high:
                    st.markdown(
                        f"""<div class="advisor-point"><span class="advisor-icon">›</span><div><strong>Yapay Zeka Dengesi Kurulmalı:</strong> Öğrencinin haftalık YZ kullanım saati (<strong>{w} saat</strong>) ve bağımlılık algısı (<strong>{dep}/10</strong>) yüksek bir bilişsel yük yaratmaktadır. AI araçlarını doğrudan kopyalama/ezberleme odaklı kullanmak yerine, geleneksel çalışma alışkanlıklarını destekleyen bir yardımcı asistan olarak konumlandırması önerilir.</div></div><div class="advisor-point"><span class="advisor-icon">›</span><div><strong>Sınav Kaygısı Mentörlüğü:</strong> Öğrencinin sınav dönemi kaygısı (<strong>{anx}/10</strong>) tükenmişlik riskini tetikleyen başlıca sinyallerden biridir. Rehberlik birimiyle nefes egzersizleri, zaman yönetimi ve odaklanma seansları planlanmalıdır.</div></div><div class="advisor-point"><span class="advisor-icon">›</span><div><strong>Çalışma Dengesi &amp; Beceri Kalıcılığı:</strong> Haftalık geleneksel çalışma süresi (<strong>{trad} saat</strong>) yapay zekaya kıyasla düşük kalıyorsa, beceri kalıcılığı zayıflar. Önümüzdeki dönem için geleneksel çalışma payını artıracak bir çalışma planı ve gerekirse ders yükü ayarlaması danışman kontrolünde yapılmalıdır.</div></div>""",
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"""<div class="advisor-point"><span class="advisor-icon">›</span><div><strong>Sağlıklı Kullanım Teşviki:</strong> Öğrenci, yapay zekayı geleneksel ders çalışma süreleriyle (<strong>{trad} saat/hafta</strong>) dengeli biçimde entegre etmiştir. Bu denge tükenmişlik riskini düşük tutmaktadır. Mevcut süreç teşvik edilmeli, prompt becerilerini geliştirecek ileri seviye atölyelere katılması sağlanmalıdır.</div></div><div class="advisor-point"><span class="advisor-icon">›</span><div><strong>Akran Mentörlüğü Desteği:</strong> YZ kullanım dengesi sağlıklı olan bu öğrenci, bölümünde bağımlılık veya sınav stresi sebebiyle tükenmişlik yaşayan (yüksek risk grubundaki) arkadaşlarına rehberlik etmek üzere <strong>Akran Mentörlüğü</strong> programlarına dahil edilebilir.</div></div>""",
                        unsafe_allow_html=True
                    )

            # 4) Kapat → analiz temizlenir, modal kapanır
            st.write("")
            if st.button("Kapat", use_container_width=True, type="primary"):
                st.session_state.pop("analysis", None)
                st.rerun()

        with st.container(border=True):
            st.markdown(
                "<div class='card-title'>Öğrenci Parametre Giriş Formu</div>",
                unsafe_allow_html=True
            )

            col1, col2 = st.columns(2, gap="large")

            with col1:
                st.markdown("<div class='form-group-label'>Profil &amp; Haftalık Süreler</div>", unsafe_allow_html=True)

                major = st.selectbox(
                    "Okunan Bölüm",
                    options=['Fen-Mühendislik', 'Tıp', 'İşletme', 'Beşeri Bilimler', 'Sanat'],
                    index=0
                )

                year_of_study = st.selectbox(
                    "Sınıf Düzeyi",
                    options=['1. Sınıf', '2. Sınıf', '3. Sınıf', '4. Sınıf', 'Yüksek Lisans'],
                    index=1
                )

                policy = st.selectbox(
                    "Kurum Politikası",
                    options=['Aktif Olarak Teşvik', 'Kaynak Belirterek İzinli', 'Kesin Yasak'],
                    index=1
                )

                weekly_ai_hours = st.slider(
                    "Haftalık AI Kullanım Süresi (saat)",
                    min_value=0, max_value=40, value=10, step=1,
                    help="Haftada AI araçlarıyla geçirilen toplam saat"
                )

                traditional_study_hours = st.slider(
                    "Haftalık Geleneksel Çalışma Süresi (saat)",
                    min_value=0, max_value=40, value=12, step=1,
                    help="AI dışı, klasik ders çalışmaya ayrılan haftalık saat"
                )

                # GPA inputları Lean Core refactor ile kaldırıldı (Pre/Post GNO drop edildi —
                # ablation testinde Yüksek risk recall'unu düşürdüğü için).
                retention_unk = st.checkbox("Beceri Kalıcılık Skoru bilinmiyor", value=False)
                retention = st.slider(
                    "Beceri Kalıcılık Skoru (0-10)",
                    min_value=0, max_value=10, value=7, step=1,
                    disabled=retention_unk,
                    help="0 = hiç kalıcı değil · 10 = tamamen kalıcı"
                )

            with col2:
                st.markdown("<div class='form-group-label'>YZ Kullanımı &amp; Düzey Ölçekleri</div>", unsafe_allow_html=True)

                use_case = st.selectbox(
                    "Birincil AI Kullanım Amacı",
                    options=['Kod Hata Ayıklama', 'Fikir Geliştirme', 'Özetleme/Okuma', 'Metin Yazımı', 'Doğrudan Cevap Üretimi'],
                    index=0
                )

                prompt_skill = st.selectbox(
                    "Prompt Yazma Becerisi",
                    options=['Başlangıç', 'Orta', 'İleri', 'Bilinmiyor'],
                    index=1
                )

                paid_sub = st.selectbox(
                    "Ücretli AI Aboneliği",
                    options=['Var', 'Yok'],
                    index=1
                )

                exam_anxiety = st.segmented_control(
                    "Sınav Dönemi Kaygı Düzeyi",
                    options=list(range(1, 11)),
                    default=5,
                    help="1 = çok düşük kaygı · 10 = çok yüksek kaygı"
                )
                if exam_anxiety is None:
                    exam_anxiety = 5

                ai_dependency = st.segmented_control(
                    "Algılanan AI Bağımlılığı",
                    options=list(range(1, 11)),
                    default=4,
                    help="1 = hiç bağımlı değil · 10 = aşırı bağımlı"
                )
                if ai_dependency is None:
                    ai_dependency = 4

                tool_diversity = st.segmented_control(
                    "Kullanılan AI Araç Çeşitliliği (adet)",
                    options=list(range(1, 11)),
                    default=3,
                    help="Düzenli kullanılan farklı AI araç sayısı"
                )
                if tool_diversity is None:
                    tool_diversity = 3

        # Build the input DataFrame dynamically (Lean Core schema — GPA alanları kaldırıldı)
        input_data = {
            'Okunan Bölüm': [major],
            'Sınıf Düzeyi': [year_of_study],
            'Haftalık AI Saati': [float(weekly_ai_hours)],
            'Birincil Kullanım Amacı': [use_case],
            'Prompt Yazma Becerisi': [np.nan if prompt_skill == 'Bilinmiyor' else prompt_skill],
            'Araç Çeşitliliği': [float(tool_diversity)],
            'Ücretli Abonelik': [1.0 if paid_sub == 'Var' else 0.0],
            'Geleneksel Çalışma Saati': [float(traditional_study_hours)],
            'Algılanan AI Bağımlılığı': [float(ai_dependency)],
            'Kurum Politikası': [policy],
            'Sınav Kaygı Düzeyi': [float(exam_anxiety)],
            # Arayüzde 0-10 ölçeğinde alınır; model 0-100 bekler → ×10
            'Beceri Kalıcılık Skoru': [np.nan if retention_unk else float(retention) * 10],
        }

        input_df = pd.DataFrame(input_data)

        st.write("")
        predict_button = st.button("Analiz Et", use_container_width=True, type="primary")

        if predict_button:
            try:
                processed_df = preprocessor.transform(input_df)
                prob = model.predict_proba(processed_df)
                class_to_col = {cls: i for i, cls in enumerate(model.classes_)}
                prob_high = prob[0][class_to_col[1]]  # Probability for Yüksek (High)
                is_high_risk = prob_high >= tuned_threshold

                # Sonuç session_state'e yazılır ve modal açılır
                st.session_state.analysis = {
                    "is_high_risk": bool(is_high_risk),
                    "prob_high": float(prob_high),
                    "threshold": float(tuned_threshold),
                    "weekly_ai_hours": weekly_ai_hours,
                    "ai_dependency": ai_dependency,
                    "exam_anxiety": exam_anxiety,
                    "traditional_study_hours": traditional_study_hours,
                }
                show_result_dialog()
            except Exception as e:
                st.error(f"Veri ön işleme veya tahmin sırasında hata oluştu. Lütfen girdileri kontrol edin.\nHata: {e}")

# ────────────────────────────────────────────────────────────────────
# TAB 2 — Analitik
# ────────────────────────────────────────────────────────────────────
with tab_analytics:
    st.markdown("<div class='section-head'><span class='bar'></span>Test Seti Başarı Metrikleri</div>", unsafe_allow_html=True)

    if not METRICS:
        st.warning("Model metadata yüklenemediği için metrikler gösterilemiyor.")
    else:
        def metric_card(label, value, sub):
            return (
                f"<div class='metric-box'>"
                f"<p class='metric-label'>{label}</p>"
                f"<p class='metric-value'>{value}</p>"
                f"<p class='metric-sub'>{sub}</p>"
                f"</div>"
            )

        mcols = st.columns(5)
        cards = [
            ("Accuracy", f"%{METRICS['accuracy'] * 100:.2f}", "Test seti doğruluğu"),
            ("F1 Macro", f"{METRICS['f1_macro']:.4f}", "Sınıf-dengeli skor"),
            ("ROC-AUC", f"{METRICS['roc_auc']:.4f}", "Ayrım gücü"),
            ("Recall (Yüksek)", f"{METRICS['recall_yuksek']:.4f}", "Yakalanan risk oranı"),
            ("Precision (Yüksek)", f"{METRICS['precision_yuksek']:.4f}", "Risk tahmini isabeti"),
        ]
        for col, (lbl, val, sub) in zip(mcols, cards):
            col.markdown(metric_card(lbl, val, sub), unsafe_allow_html=True)

        st.markdown(
            f"<p style='color:#94a3b8; font-size:0.82em; margin-top:10px;'>"
            f"En iyi model: <b style='color:#c4b5fd;'>{METRICS['model_name']}</b> · "
            f"{METRICS['feature_count']} feature (Lean Core) · karar eşiği {METRICS['threshold']:.2f}. "
            f"Tüm skorlar yalnızca <b>saklı test seti</b> üzerinde tek sefer ölçülmüştür.</p>",
            unsafe_allow_html=True
        )

    st.write("")
    st.markdown("<div class='section-head'><span class='bar'></span>Model Karar Analitiği</div>", unsafe_allow_html=True)
    fig_col1, fig_col2 = st.columns(2)

    with fig_col1:
        with st.container(border=True):
            st.markdown(
                "<div class='card-title' style='font-size:1.06rem;'>Confusion Matrix (Hata Matrisi)</div>",
                unsafe_allow_html=True
            )

            cm = METRICS['confusion_matrix'] if METRICS else np.array([[2968, 306], [878, 1620]])
            classes = ['Düşük', 'Yüksek']
            row_sums = cm.sum(axis=1, keepdims=True)
            cm_pct = np.divide(cm, row_sums, where=row_sums != 0) * 100

            cell_text = [
                [f"{cm[r, c]}<br>{cm_pct[r, c]:.1f}%" for c in range(2)]
                for r in range(2)
            ]

            fig_cm = go.Figure(data=go.Heatmap(
                z=cm,
                x=classes,
                y=classes,
                colorscale=[[0.0, '#0f172a'], [0.5, '#7c3aed'], [1.0, '#34d399']],
                text=cell_text,
                texttemplate='%{text}',
                textfont=dict(color='#f8fafc', size=14, family='Inter'),
                hovertemplate='Gerçek: %{y}<br>Tahmin: %{x}<br>Adet: %{z}<extra></extra>',
                showscale=False
            ))

            fig_cm.update_layout(
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(family='Inter', color='#cbd5e1'),
                xaxis=dict(title='Tahmin Edilen Sınıf', showgrid=False),
                yaxis=dict(title='Gerçek Sınıf', showgrid=False, autorange='reversed'),
                height=260,
                margin=dict(l=40, r=40, t=10, b=40)
            )
            st.plotly_chart(fig_cm, use_container_width=True)

    with fig_col2:
        with st.container(border=True):
            st.markdown(
                "<div class='card-title' style='font-size:1.06rem;'>Bölüm Bazlı Yüksek Tükenmişlik Oranı</div>",
                unsafe_allow_html=True
            )

            # EDA bulgusu — bölüm bazlı yüksek risk yüzdeleri
            majors = ['Fen-Mühendislik', 'Beşeri Bilimler', 'İşletme', 'Tıp', 'Sanat']
            ratios = [30.0, 20.7, 24.3, 23.2, 26.5]

            fig_major = go.Figure(data=[go.Bar(
                x=majors,
                y=ratios,
                marker_color=['#fb7185', '#10b981', '#fbbf24', '#22d3ee', '#a78bfa'],
                text=[f"%{v}" for v in ratios],
                textposition='outside',
                hovertemplate='<b>%{x}</b><br>Yüksek Risk Oranı: %{y}%<extra></extra>'
            )])

            fig_major.update_layout(
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(family='Inter', color='#cbd5e1'),
                yaxis=dict(title='Yüksek Risk Oranı (%)', range=[0, 40], gridcolor='rgba(255,255,255,0.08)'),
                xaxis=dict(showgrid=False),
                height=260,
                margin=dict(l=40, r=40, t=10, b=40)
            )
            st.plotly_chart(fig_major, use_container_width=True)

    # Team Roles Section
    st.markdown("<div class='section-head'><span class='bar'></span>Ekip & Proje Organizasyonu</div>", unsafe_allow_html=True)

    team_cols = st.columns(4)
    for col, m in zip(team_cols, TEAM):
        col.markdown(
            f"<div class='team-card'>"
            f"<div class='team-avatar'>{m['name'][0]}</div>"
            f"<div class='team-name'>{m['name']}</div>"
            f"<div class='team-role'>{m['role']}</div>"
            f"<div class='team-phase'>{m['phase']}</div>"
            f"<p class='team-desc'>{m['desc']}</p>"
            f"</div>",
            unsafe_allow_html=True
        )
