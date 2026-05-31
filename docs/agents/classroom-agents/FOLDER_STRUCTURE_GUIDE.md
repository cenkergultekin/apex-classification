# 📁 CHURN ANALYSIS - PROFESYONEL KLASÖR YAPISI

Tüm agentler bu profesyonel klasör yapısını kullanmalıdır.

## 🎯 Klasör Yapısı

```
churn-analysis/
├── data/
│   ├── raw/                      # Ham veri (değiştirilmez)
│   │   └── churn.csv
│   ├── processed/                # İşlenmiş veri (EDA çıktısı)
│   │   └── churn_cleaned.csv
│   └── model_ready/              # Model-ready veri (DataPrep çıktısı)
│       ├── X_train.csv
│       ├── X_test.csv
│       ├── y_train.csv
│       └── y_test.csv
│
├── scripts/                      # Tüm Python scriptleri
│   ├── phase1_data_overview.py
│   ├── phase2_univariate_analysis.py
│   ├── phase3_bivariate_analysis.py
│   ├── phase4_multivariate_analysis.py
│   ├── phase5_data_quality_analysis.py
│   ├── phase6_insight_generation.py
│   ├── phase7_model_readiness.py
│   ├── data_preparation.py       # DataPrep Expert scripti
│   └── model_training.py         # Model Expert scripti
│
├── figures/                      # Tüm grafikler (HTML + PNG)
│   ├── tenure_distribution.html
│   ├── churn_distribution.html
│   ├── confusion_matrix.html
│   ├── roc_curve.html
│   └── feature_importance.html
│
├── reports/
│   ├── csv/                      # Tüm CSV analiz raporları
│   │   ├── bivariate_summary_numeric.csv
│   │   ├── bivariate_summary_categorical.csv
│   │   ├── high_correlations.csv
│   │   ├── vif_analysis.csv
│   │   ├── outlier_analysis_iqr.csv
│   │   ├── outlier_analysis_zscore.csv
│   │   ├── data_quality_scores.csv
│   │   ├── top_10_insights.csv
│   │   ├── critical_features_for_modeling.csv
│   │   ├── data_preparation_strategy.csv
│   │   ├── feature_engineering_opportunities.csv
│   │   ├── business_action_priorities.csv
│   │   ├── model_readiness_assessment.csv
│   │   ├── data_prep_recommendations_phase*.csv
│   │   ├── data_prep_summary.csv
│   │   ├── model_comparison_results.csv
│   │   └── feature_importance_ranking.csv
│   │
│   └── markdown/                 # Markdown raporlar
│       ├── EDA_FINAL_REPORT.md
│       ├── DATA_PREP_HANDOFF.md
│       ├── MODEL_EVALUATION_REPORT.md
│       └── DEPLOYMENT_GUIDE.md
│
├── models/                       # Modeller ve pipeline'lar
│   ├── preprocessing_pipeline.pkl
│   ├── baseline_model.pkl
│   ├── random_forest.pkl
│   ├── xgboost_model.pkl
│   ├── lightgbm_model.pkl
│   ├── catboost_model.pkl
│   ├── final_model.pkl           # En iyi model
│   └── model_metadata.json
│
├── app/                          # Streamlit deployment (Deployment Expert)
│   ├── app.py
│   ├── utils.py
│   ├── config.py
│   └── assets/
│       ├── logo.png
│       └── style.css
│
├── notebooks/                    # Jupyter notebooklar (opsiyonel)
│   └── exploratory_analysis.ipynb
│
├── .github/
│   └── agents/                   # Agent tanımları
│       ├── eda-expert-agent.md
│       ├── dataprep-expert-agent.md
│       ├── model-expert-agent.md
│       └── deployment-expert-agent.md
│
├── requirements.txt              # Tüm dependencies
├── README.md                     # Proje açıklaması
└── FOLDER_STRUCTURE_GUIDE.md     # Bu dosya
```

---

## 📋 Agent'lere Göre Dosya Yolu Kullanımı

### 🔍 EDA Expert (scripts/ içinde çalışır)

```python
# Ham veri oku
df = pd.read_csv('../data/raw/churn.csv')

# İşlenmiş veri kaydet
df_cleaned.to_csv('../data/processed/churn_cleaned.csv', index=False)

# İşlenmiş veri oku (sonraki phase'lerde)
df = pd.read_csv('../data/processed/churn_cleaned.csv')

# CSV raporu kaydet
summary_df.to_csv('../reports/csv/bivariate_summary_numeric.csv', index=False)

# Grafik kaydet
fig.write_html('../figures/tenure_distribution.html')

# Markdown rapor kaydet
with open('../reports/markdown/EDA_FINAL_REPORT.md', 'w', encoding='utf-8') as f:
    f.write(report)
```

### 🛠️ DataPrep Expert (scripts/ içinde çalışır)

```python
from pathlib import Path

# İşlenmiş veri oku (EDA çıktısı)
df = pd.read_csv('../data/processed/churn_cleaned.csv')

# Model-ready klasörü oluştur
Path('../data/model_ready').mkdir(parents=True, exist_ok=True)

# Train/test split kaydet
X_train.to_csv('../data/model_ready/X_train.csv', index=False)
X_test.to_csv('../data/model_ready/X_test.csv', index=False)
y_train.to_csv('../data/model_ready/y_train.csv', index=False)
y_test.to_csv('../data/model_ready/y_test.csv', index=False)

# Pipeline kaydet
import joblib
Path('../models').mkdir(parents=True, exist_ok=True)
joblib.dump(pipeline, '../models/preprocessing_pipeline.pkl')

# Rapor kaydet
prep_summary.to_csv('../reports/csv/data_prep_summary.csv', index=False)
```

### 🤖 Model Expert (scripts/ içinde çalışır)

```python
from pathlib import Path
import joblib

# Model-ready veri oku
X_train = pd.read_csv('../data/model_ready/X_train.csv')
X_test = pd.read_csv('../data/model_ready/X_test.csv')
y_train = pd.read_csv('../data/model_ready/y_train.csv').values.ravel()
y_test = pd.read_csv('../data/model_ready/y_test.csv').values.ravel()

# Modelleri kaydet
Path('../models').mkdir(parents=True, exist_ok=True)
joblib.dump(rf_model, '../models/random_forest.pkl')
joblib.dump(xgb_model, '../models/xgboost_model.pkl')
joblib.dump(final_model, '../models/final_model.pkl')

# Karşılaştırma raporu
results_df.to_csv('../reports/csv/model_comparison_results.csv', index=False)

# Grafikler
Path('../figures').mkdir(parents=True, exist_ok=True)
fig_cm.write_html('../figures/confusion_matrix.html')
fig_roc.write_html('../figures/roc_curve.html')
```

### 🚀 Deployment Expert (app/ içinde çalışır)

```python
from pathlib import Path
import joblib
import streamlit as st

# Model yükle
MODEL_PATH = Path('../models/final_model.pkl')
PIPELINE_PATH = Path('../models/preprocessing_pipeline.pkl')

model = joblib.load(MODEL_PATH)
pipeline = joblib.load(PIPELINE_PATH)

# Assets
LOGO_PATH = Path('assets/logo.png')
if LOGO_PATH.exists():
    st.image(str(LOGO_PATH))

# Config
import sys
sys.path.append('..')
from scripts.config import PROFESSIONAL_PALETTE
```

---

## ✅ Klasör Oluşturma Best Practice

Her script başlangıcında gerekli klasörlerin varlığını garantilemelidir:

```python
from pathlib import Path

# Klasörleri oluştur (varsa zaten var, yoksa oluştur)
Path('../data/processed').mkdir(parents=True, exist_ok=True)
Path('../data/model_ready').mkdir(parents=True, exist_ok=True)
Path('../figures').mkdir(parents=True, exist_ok=True)
Path('../reports/csv').mkdir(parents=True, exist_ok=True)
Path('../reports/markdown').mkdir(parents=True, exist_ok=True)
Path('../models').mkdir(parents=True, exist_ok=True)
```

---

## ❌ Yaygın Hatalar

**YANLIŞ:**
```python
df = pd.read_csv('churn.csv')  # Root'ta arama yapar - hata!
fig.write_html('figures/plot.html')  # Relative path eksik!
summary.to_csv('report.csv')  # Dağınık dosya yönetimi!
```

**DOĞRU:**
```python
df = pd.read_csv('../data/raw/churn.csv')
fig.write_html('../figures/plot.html')
summary.to_csv('../reports/csv/report.csv', index=False)
```

---

## 📊 Hangi Agent Hangi Klasörde Çalışır?

| Agent | Çalışma Dizini | Okuma Yeri | Yazma Yeri |
|-------|---------------|-----------|-----------|
| **EDA Expert** | `scripts/` | `../data/raw/`, `../data/processed/` | `../data/processed/`, `../figures/`, `../reports/` |
| **DataPrep Expert** | `scripts/` | `../data/processed/` | `../data/model_ready/`, `../models/`, `../reports/` |
| **Model Expert** | `scripts/` | `../data/model_ready/` | `../models/`, `../figures/`, `../reports/` |
| **Deployment Expert** | `app/` | `../models/`, `../reports/` | `app/assets/` |

---

## 🎯 Proje Yaşam Döngüsü Akışı

```
1. EDA Expert:
   data/raw/churn.csv 
   → scripts/phase*.py çalıştır
   → data/processed/churn_cleaned.csv + reports/ + figures/ oluştur

2. DataPrep Expert:
   data/processed/churn_cleaned.csv oku
   → scripts/data_preparation.py çalıştır
   → data/model_ready/* + models/preprocessing_pipeline.pkl oluştur

3. Model Expert:
   data/model_ready/* oku
   → scripts/model_training.py çalıştır
   → models/*.pkl + figures/ + reports/ oluştur

4. Deployment Expert:
   models/final_model.pkl oku
   → app/app.py oluştur
   → Streamlit ile deploy et
```

---

## 📌 Önemli Notlar

1. **scripts/** klasöründe çalışırken **mutlaka** `..` ile relative path kullan
2. **Her agent** kendi çıktısını ilgili klasöre kaydetmeli
3. **Dosya isimlendirme** tutarlı olmalı: `snake_case.csv` veya `snake_case.py`
4. **Grafikler** her zaman `figures/` klasöründe
5. **CSV raporlar** her zaman `reports/csv/` klasöründe
6. **Markdown raporlar** her zaman `reports/markdown/` klasöründe
7. **Modeller** her zaman `models/` klasöründe

---

**✨ Bu yapı tüm veri bilimi projelerinde kullanılabilir profesyonel bir standarttır!**
