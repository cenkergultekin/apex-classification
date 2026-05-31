# -*- coding: utf-8 -*-
"""Streamlit deployment interface for APEX AI Student Impact classifier.

This app features:
1. Burnout Risk Predictor (Form, real-time prediction, gauges, and advisor suggestions)
2. Project Analytics & Data Story (Interactive EDA visualizations, metrics, team cards)
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
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Glassmorphism & Premium Dark Theme)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
    
    /* Global styles */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Background and sidebar */
    .stApp {
        background-color: #05070f;
        background-image: radial-gradient(at 0% 0%, rgba(124, 58, 237, 0.08) 0px, transparent 50%),
                          radial-gradient(at 50% 100%, rgba(6, 182, 212, 0.06) 0px, transparent 50%);
    }
    
    /* Glow Titles */
    .main-title {
        background: linear-gradient(135deg, #c4b5fd 0%, #a78bfa 50%, #6366f1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem;
        font-weight: 900;
        letter-spacing: -1px;
        margin-bottom: 0.2rem;
        text-shadow: 0px 4px 20px rgba(99, 102, 241, 0.15);
    }
    
    .subtitle {
        color: #71717a;
        font-size: 1.1rem;
        font-weight: 500;
        margin-bottom: 2rem;
    }
    
    /* Style bordered containers and forms to look premium */
    div[data-testid="stForm"], div[data-testid="stContainerBordered"] {
        background: linear-gradient(135deg, rgba(30, 27, 75, 0.4) 0%, rgba(15, 23, 42, 0.4) 100%) !important;
        border: 1px solid rgba(124, 58, 237, 0.3) !important;
        border-radius: 16px !important;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37) !important;
        backdrop-filter: blur(8px) !important;
        -webkit-backdrop-filter: blur(8px) !important;
        padding: 24px !important;
        margin-bottom: 20px !important;
    }
    
    /* Cards styling (standard HTML elements) */
    .premium-card {
        background: linear-gradient(135deg, rgba(30, 27, 75, 0.4) 0%, rgba(15, 23, 42, 0.4) 100%);
        border: 1px solid rgba(124, 58, 237, 0.2);
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        margin-bottom: 20px;
    }
    
    .card-title {
        color: #f8fafc;
        font-size: 1.25rem;
        font-weight: 700;
        margin-bottom: 15px;
        display: flex;
        align-items: center;
        gap: 10px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        padding-bottom: 8px;
    }
    
    /* Neon badges */
    .risk-badge-high {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(220, 38, 38, 0.25) 100%);
        border: 1px solid rgba(239, 68, 68, 0.5);
        color: #fca5a5;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.9em;
        letter-spacing: 0.5px;
        display: inline-block;
        box-shadow: 0 0 15px rgba(239, 68, 68, 0.2);
    }
    
    .risk-badge-low {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(5, 150, 105, 0.25) 100%);
        border: 1px solid rgba(16, 185, 129, 0.5);
        color: #a7f3d0;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.9em;
        letter-spacing: 0.5px;
        display: inline-block;
        box-shadow: 0 0 15px rgba(16, 185, 129, 0.2);
    }
    
    /* Advisor points */
    .advisor-point {
        display: flex;
        gap: 12px;
        align-items: flex-start;
        margin-bottom: 12px;
        font-size: 0.92rem;
        color: #e2e8f0;
        line-height: 1.6;
    }
    
    .advisor-icon {
        color: #a78bfa;
        font-weight: 900;
        flex-shrink: 0;
        margin-top: 2px;
    }
    
    /* Metrics Row styling */
    .metric-box {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    }

    /* Hide Streamlit deploy button, header, and footer */
    [data-testid="stHeader"] {
        display: none !important;
    }
    [data-testid="stHeaderDeployButton"] {
        display: none !important;
    }
    #MainMenu {
        display: none !important;
    }
    footer {
        display: none !important;
    }
    .stDeployButton {
        display: none !important;
    }
    [data-testid="stToolbar"] {
        display: none !important;
    }
    
    /* Adjust main container padding top since header is hidden */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2rem !important;
    }
</style>
""", unsafe_allow_html=True)


# Load artifacts cleanly with caching
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

preprocessor, model_package = load_model_artifacts()

# Sidebar Info
with st.sidebar:
    st.image("https://img.icons8.com/nolan/256/brain.png", width=90)
    st.markdown("<h2 style='color:#a78bfa; font-weight:800; margin-top:0;'>APEX</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#71717a; font-size:0.85em;'>AI Student Impact — Akademik Tükenmişlik Risk Tahmin Sistemi</p>", unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### Ekip Üyeleri")
    st.markdown("""
    * **Berkay** · Deployment & Arayüz
    * **Feza** · İş Bağlamı & EDA
    * **Ethem** · Modelleme & Değerlendirme
    * **Cenker** · Pipeline & Feature Eng.
    """)
    
    st.markdown("---")
    st.markdown("### Model Özeti")
    if model_package:
        st.markdown(f"**Model:** {model_package['model_name']}")
        st.markdown(f"**Karar Eşiği:** {model_package['threshold_decision']['tuned']['threshold']}")
        st.markdown(f"**Özellik Sayısı:** {model_package['feature_count']} (Compact)")
    else:
        st.markdown("Model yüklenemedi.")

# Header
st.markdown("<h1 class='main-title'>🎓 Yapay Zeka Etkisi Sınıflandırma</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Öğrencilerin AI Kullanım Örüntüleri ve Akademik Tükenmişlik Tahmini</p>", unsafe_allow_html=True)

# Main Application Tabs
tab_predict, tab_analytics = st.tabs(["🔮 Tükenmişlik Tahmini", "📊 Proje Analitiği & Hikayesi"])

with tab_predict:
    if not model_package or not preprocessor:
        st.warning("Model dosyaları eksik veya uyumsuz olduğundan tahmin sekmesi devre dışı.")
    else:
        # Load prediction model and metadata
        model = model_package['model']
        tuned_threshold = model_package['threshold_decision']['tuned']['threshold']
        
        with st.container(border=True):
            st.markdown("<h3 style='color: #f8fafc; font-size: 1.25rem; font-weight: 700; margin-bottom: 15px; display: flex; align-items: center; gap: 10px; border-bottom: 1px solid rgba(255, 255, 255, 0.08); padding-bottom: 8px;'>🔮 Öğrenci Parametre Giriş Formu</h3>", unsafe_allow_html=True)
            
            # User input fields in 2 columns
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("##### 📚 Akademik & Kurumsal Bilgiler")
                
                major = st.selectbox(
                    "Okunan Bölüm (Major Category)",
                    options=['Fen-Mühendislik', 'Tıp', 'İşletme', 'Beşeri Bilimler', 'Sanat'],
                    index=0
                )
                
                year_of_study = st.selectbox(
                    "Sınıf Düzeyi (Year of Study)",
                    options=['1. Sınıf', '2. Sınıf', '3. Sınıf', '4. Sınıf', 'Yüksek Lisans'],
                    index=1
                )
                
                policy = st.selectbox(
                    "Kurum Politikası (Institutional Policy)",
                    options=['Aktif Olarak Teşvik', 'Kaynak Belirterek İzinli', 'Kesin Yasak'],
                    index=1
                )
                
                # GPA inputları Lean Core refactor ile kaldırıldı (Pre/Post GNO drop edildi —
                # ablation testinde Yüksek risk recall'unu düşürdüğü ve canlı tahmin senaryolarında
                # kırılgan olduğu için).
                st.write("")
                retention_unk = st.checkbox("Beceri Kalıcılık Skoru Bilinmiyor", value=False)
                retention = st.slider(
                    "Beceri Kalıcılık Skoru (Skill Retention Score)",
                    min_value=0.0, max_value=100.0, value=75.0, step=0.5,
                    disabled=retention_unk
                )

            with col2:
                st.markdown("##### 💻 Yapay Zeka (AI) Kullanım Alışkanlıkları")
                
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
                    "Ücretli AI Aboneliği (Paid Subscription)",
                    options=['Var', 'Yok'],
                    index=1
                )
                
                st.write("")
                st.write("")
                
                weekly_ai_hours = st.slider(
                    "Haftalık Yapay Zeka Kullanım Süresi (Saat)",
                    min_value=0.0, max_value=40.0, value=10.0, step=0.5
                )
                
                traditional_study_hours = st.slider(
                    "Haftalık Geleneksel Çalışma Süresi (Saat)",
                    min_value=0.0, max_value=40.0, value=12.0, step=0.5
                )
                
                ai_dependency = st.slider(
                    "Algılanan Yapay Zeka Bağımlılığı (1-10)",
                    min_value=1.0, max_value=10.0, value=4.0, step=0.5
                )
                
                exam_anxiety = st.slider(
                    "Sınav Dönemi Kaygı Düzeyi (1-10)",
                    min_value=1.0, max_value=10.0, value=5.0, step=0.5
                )
                
                tool_diversity = st.slider(
                    "Kullanılan AI Araç Çeşitliliği (Adet)",
                    min_value=1, max_value=10, value=3, step=1
                )
        
        # Build the input DataFrame dynamically (Lean Core schema — GPA alanları kaldırıldı)
        input_data = {
            'Okunan Bölüm': [major],
            'Sınıf Düzeyi': [year_of_study],
            'Haftalık AI Saati': [weekly_ai_hours],
            'Birincil Kullanım Amacı': [use_case],
            'Prompt Yazma Becerisi': [np.nan if prompt_skill == 'Bilinmiyor' else prompt_skill],
            'Araç Çeşitliliği': [float(tool_diversity)],
            'Ücretli Abonelik': [1.0 if paid_sub == 'Var' else 0.0],
            'Geleneksel Çalışma Saati': [traditional_study_hours],
            'Algılanan AI Bağımlılığı': [ai_dependency],
            'Kurum Politikası': [policy],
            'Sınav Kaygı Düzeyi': [exam_anxiety],
            'Beceri Kalıcılık Skoru': [np.nan if retention_unk else retention],
        }
        
        input_df = pd.DataFrame(input_data)
        
        # Prediction logic triggered automatically or via a button
        st.write("")
        predict_button = st.button("🔮 Tükenmişlik Riskini Analiz Et", use_container_width=True)
        
        if predict_button:
            # Preprocess the input
            try:
                processed_df = preprocessor.transform(input_df)
                
                # Predict probability
                prob = model.predict_proba(processed_df)
                class_to_col = {cls: i for i, cls in enumerate(model.classes_)}
                prob_high = prob[0][class_to_col[1]] # Probability for Yüksek (High)
                
                # Classification based on tuned decision threshold
                is_high_risk = prob_high >= tuned_threshold
                result_label = "Yüksek" if is_high_risk else "Düşük"
                
                # Visual result section
                st.markdown("---")
                st.markdown("### 🔍 Analiz Sonuçları")
                
                res_col1, res_col2 = st.columns([1.1, 1.3])
                
                with res_col1:
                    with st.container(border=True):
                        if is_high_risk:
                            badge_html = "<span class='risk-badge-high' style='font-size: 1.4em; padding: 10px 25px;'>YÜKSEK RİSK</span>"
                            text_html = "<p style='color: #fca5a5; font-size: 0.92em; font-weight: 500; line-height: 1.6; margin-top: 15px;'>Öğrenci, akademik ve YZ kullanım parametrelerine göre ciddi düzeyde tükenmişlik riski altındadır.</p>"
                        else:
                            badge_html = "<span class='risk-badge-low' style='font-size: 1.4em; padding: 10px 25px;'>DÜŞÜK RİSK</span>"
                            text_html = "<p style='color: #a7f3d0; font-size: 0.92em; font-weight: 500; line-height: 1.6; margin-top: 15px;'>Öğrencinin mevcut akademik ve YZ kullanım dengesi sağlıklıdır.</p>"
                        
                        card_html = f"""
                        <div style='text-align: center; height: 250px; display: flex; flex-direction: column; justify-content: center; align-items: center; padding: 10px; margin-bottom: 0px;'>
                            <div style='color: #a78bfa; font-weight: 800; font-size: 0.85em; letter-spacing: 2px;'>TAHMİN EDİLEN SINIF</div>
                            <div style='margin: 25px 0;'>{badge_html}</div>
                            {text_html}
                        </div>
                        """
                        st.markdown(card_html.replace("\n", " "), unsafe_allow_html=True)
                    
                with res_col2:
                    with st.container(border=True):
                        fig_gauge = go.Figure(go.Indicator(
                            mode = "gauge+number",
                            value = prob_high * 100,
                            domain = {'x': [0, 1], 'y': [0, 1]},
                            number = {'suffix': "%", 'font': {'size': 38, 'color': '#f8fafc', 'weight': 'bold'}},
                            gauge = {
                                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#475569", 'tickfont': {'color': '#94a3b8'}},
                                'bar': {'color': "#ef4444" if is_high_risk else "#10b981"},
                                'bgcolor': "rgba(30, 41, 59, 0.4)",
                                'borderwidth': 1.5,
                                'bordercolor': "#475569",
                                'steps': [
                                    {'range': [0, 35], 'color': 'rgba(16, 185, 129, 0.08)'},
                                    {'range': [35, 70], 'color': 'rgba(245, 158, 11, 0.08)'},
                                    {'range': [70, 100], 'color': 'rgba(239, 68, 68, 0.08)'}
                                ],
                                'threshold': {
                                    'line': {'color': "#ffffff", 'width': 3},
                                    'thickness': 0.75,
                                    'value': tuned_threshold * 100
                                }
                            }
                        ))
                        fig_gauge.update_layout(
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            font={'color': "#f8fafc", 'family': "Inter"},
                            height=298,
                            margin=dict(l=25, r=25, t=50, b=25)
                        )
                        st.plotly_chart(fig_gauge, use_container_width=True)
                
                # Advisor Section
                with st.container(border=True):
                    st.markdown("<h3 style='color: #f8fafc; font-size: 1.25rem; font-weight: 700; margin-bottom: 15px; display: flex; align-items: center; gap: 10px; border-bottom: 1px solid rgba(255, 255, 255, 0.08); padding-bottom: 8px;'>🎯 Akademik Danışman & Rehberlik Önerileri</h3>", unsafe_allow_html=True)
                    
                    if is_high_risk:
                        st.markdown(f"""<div class="advisor-point"><span class="advisor-icon">✦</span><div><strong>Yapay Zeka Dengesi Kurulmalı:</strong> Öğrencinin haftalık YZ kullanım saati (<strong>{weekly_ai_hours} saat</strong>) ve bağımlılık algısı (<strong>{ai_dependency}/10</strong>) oldukça yüksek bir bilişsel yük yaratmaktadır. AI araçlarını ödev veya yazılımlarda doğrudan kopyalama/ezberleme odaklı kullanmak yerine, kütüphanelerde geleneksel çalışma alışkanlıklarını destekleyici yardımcı bir asistan olarak kullanması önerilmelidir.</div></div><div class="advisor-point"><span class="advisor-icon">✦</span><div><strong>Sınav Kaygısı Mentörlüğü:</strong> Öğrencinin sınav dönemi kaygısı (<strong>{exam_anxiety}/10</strong>) yüksek riskli tükenmişlik durumunu tetiklemektedir. Okulun rehberlik birimiyle görüşerek sınav anksiyetesini azaltacak nefes egzersizleri, zaman yönetimi ve odaklanma seansları planlanmalıdır.</div></div><div class="advisor-point"><span class="advisor-icon">✦</span><div><strong>Dönem Sonu Not Takibi & Destek:</strong> GNO değişimi yakından takip edilmelidir. Eğer dönem başından sonuna doğru belirgin bir düşüş gözlenmişse, gelecek dönem için ders yükü azaltılmalı veya ders seçim süreçleri danışman öğretmen kontrolünde yapılmalıdır.</div></div>""", unsafe_allow_html=True)
                    else:
                        st.markdown(f"""<div class="advisor-point"><span class="advisor-icon">✦</span><div><strong>Sağlıklı Kullanım Teşviki:</strong> Öğrenci, yapay zekayı geleneksel ders çalışma süreleriyle (<strong>{traditional_study_hours} saat/hafta</strong>) dengeli bir biçimde entegre etmiştir. Bu durum, tükenmişlik riskinin oldukça düşük kalmasını sağlamaktadır. Mevcut süreç teşvik edilmeli, prompt becerilerini artıracak ileri seviye atölyelere katılması sağlanmalıdır.</div></div><div class="advisor-point"><span class="advisor-icon">✦</span><div><strong>Akran Mentörlüğü Desteği:</strong> Akademik başarısı ve YZ kullanım dengesi yüksek olan bu öğrencimiz, bölümünde yapay zeka bağımlılığı veya sınav stresi sebebiyle akademik tükenmişlik yaşayan (yüksek risk grubundaki) arkadaşlarına rehberlik etmesi amacıyla <strong>Akran Mentörlüğü</strong> programlarına dahil edilebilir.</div></div>""", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Veri ön işleme veya tahmin sırasında hata oluştu. Lütfen girdileri kontrol edin.\nHata: {e}")

with tab_analytics:
    st.markdown("### 📊 Proje Başarı Metrikleri")
    
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.markdown("<div class='metric-box'><p style='color:#a78bfa; font-size:0.78em; font-weight:700; text-transform:uppercase; margin:0 0 5px; letter-spacing:1px;'>Accuracy</p><p style='color:#ecfdf5; font-size:1.8em; font-weight:900; margin:0;'>%80.28</p></div>", unsafe_allow_html=True)
    with m_col2:
        st.markdown("<div class='metric-box'><p style='color:#a78bfa; font-size:0.78em; font-weight:700; text-transform:uppercase; margin:0 0 5px; letter-spacing:1px;'>F1 Macro</p><p style='color:#a7f3d0; font-size:1.8em; font-weight:900; margin:0;'>0.7925</p></div>", unsafe_allow_html=True)
    with m_col3:
        st.markdown("<div class='metric-box'><p style='color:#a78bfa; font-size:0.78em; font-weight:700; text-transform:uppercase; margin:0 0 5px; letter-spacing:1px;'>Recall Yüksek</p><p style='color:#fda4af; font-size:1.8em; font-weight:900; margin:0;'>0.6701</p></div>", unsafe_allow_html=True)
    with m_col4:
        st.markdown("<div class='metric-box'><p style='color:#a78bfa; font-size:0.78em; font-weight:700; text-transform:uppercase; margin:0 0 5px; letter-spacing:1px;'>Precision Yüksek</p><p style='color:#a78bfa; font-size:1.8em; font-weight:900; margin:0;'>0.8555</p></div>", unsafe_allow_html=True)
        
    st.write("")
    
    # Visualizations Row
    st.markdown("### 📈 Model Karar Analitiği")
    fig_col1, fig_col2 = st.columns(2)
    
    with fig_col1:
        with st.container(border=True):
            st.markdown("<h3 style='color: #f8fafc; font-size: 1.15rem; font-weight: 700; margin-bottom: 15px; display: flex; align-items: center; gap: 10px; border-bottom: 1px solid rgba(255, 255, 255, 0.08); padding-bottom: 8px;'>📊 Confusion Matrix (Hata Matrisi)</h3>", unsafe_allow_html=True)
            
            # Build Heatmap
            cm = np.array([[2960, 314], [824, 1674]])
            classes = ['Düşük', 'Yüksek']
            
            fig_cm = go.Figure(data=go.Heatmap(
                z=cm,
                x=classes,
                y=classes,
                colorscale=[[0.0, '#0f172a'], [0.5, '#7c3aed'], [1.0, '#34d399']],
                text=[[f"{cm[0,0]}<br>90.4%", f"{cm[0,1]}<br>9.6%"], [f"{cm[1,0]}<br>33.0%", f"{cm[1,1]}<br>67.0%"]],
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
            st.markdown("<h3 style='color: #f8fafc; font-size: 1.15rem; font-weight: 700; margin-bottom: 15px; display: flex; align-items: center; gap: 10px; border-bottom: 1px solid rgba(255, 255, 255, 0.08); padding-bottom: 8px;'>📈 Bölüm Bazlı Yüksek Tükenmişlik Oranı</h3>", unsafe_allow_html=True)
            
            # Major-based high risk percentage
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
    st.markdown("### 🏆 Ekip & Proje Organizasyonu")
    
    r_col1, r_col2, r_col3, r_col4 = st.columns(4)
    with r_col1:
        with st.container(border=True):
            st.markdown("<div style='text-align:center; padding: 5px 0;'><div style='font-size:2.2em; margin-bottom:8px;'>🕵️‍♂️</div><strong style='color:#f8fafc; font-size:1.05em;'>Feza</strong><p style='color:#a78bfa; font-size:0.85em; margin:4px 0 10px;'>CRISP-DM Faz 1 & 2</p><p style='color:#94a3b8; font-size:0.82em; line-height:1.6; margin:0;'>Business Understanding ve detaylı Plotly EDA süreçlerini yürüterek veri setindeki gizli ilişkileri keşfetti.</p></div>", unsafe_allow_html=True)
    with r_col2:
        with st.container(border=True):
            st.markdown("<div style='text-align:center; padding: 5px 0;'><div style='font-size:2.2em; margin-bottom:8px;'>⚙️</div><strong style='color:#f8fafc; font-size:1.05em;'>Cenker</strong><p style='color:#a78bfa; font-size:0.85em; margin:4px 0 10px;'>CRISP-DM Faz 3</p><p style='color:#94a3b8; font-size:0.82em; line-height:1.6; margin:0;'>Veri temizleme, öznitelik mühendisliği ve production-ready Pipeline / ColumnTransformer mimarisini kurdu.</p></div>", unsafe_allow_html=True)
    with r_col3:
        with st.container(border=True):
            st.markdown("<div style='text-align:center; padding: 5px 0;'><div style='font-size:2.2em; margin-bottom:8px;'>🔬</div><strong style='color:#f8fafc; font-size:1.05em;'>Ethem</strong><p style='color:#a78bfa; font-size:0.85em; margin:4px 0 10px;'>CRISP-DM Faz 4 & 5</p><p style='color:#94a3b8; font-size:0.82em; line-height:1.6; margin:0;'>10 farklı classification algoritmasını CV ile eğiterek karşılaştırdı. En iyi hiperparametre optimizasyonunu yaptı.</p></div>", unsafe_allow_html=True)
    with r_col4:
        with st.container(border=True):
            st.markdown("<div style='text-align:center; padding: 5px 0;'><div style='font-size:2.2em; margin-bottom:8px;'>🚀</div><strong style='color:#f8fafc; font-size:1.05em;'>Berkay</strong><p style='color:#a78bfa; font-size:0.85em; margin:4px 0 10px;'>CRISP-DM Faz 6</p><p style='color:#94a3b8; font-size:0.82em; line-height:1.6; margin:0;'>Projenin üretim aşamasına (Streamlit entegrasyonu) aktarılması ve kullanıcı dostu analitik ekranların tasarımını tamamladı.</p></div>", unsafe_allow_html=True)
