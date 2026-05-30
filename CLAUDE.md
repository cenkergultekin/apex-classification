# CLAUDE.md — APEX-V2 Final Proje Rehberi

## Proje Özeti

**Proje:** AI Student Impact — Öğrenci başarısını etkileyen faktörlerin sınıflandırma analizi  
**Veri seti:** `data/raw/ai_student_impact_dataset.csv`  
**Metodoloji:** CRISP-DM (6 faz)  
**Tür:** Classification  
**Ekip:** Berkay · Feza · Ethem · Cenker

---

## Ekip & Faz Dağılımı

| Kişi | Faz | Başlık |
|------|-----|--------|
| Feza | 1–2 | Business Understanding + EDA |
| Cenker | 3 | Data Preparation + Pipeline |
| Berkay | 4–5 | Modeling + Evaluation |
| Ethem | 6 | Deployment + Tanıtım Sitesi |

Detaylı görev listeleri için: `plans_and_agents/TASKS.md`

---

## Klasör Yapısı

```
apex-v2/
├── plans_and_agents/             # Planlar, Agent Yönergeleri ve Proje Günlüğü
├── data/
│   ├── raw/                        # Orijinal, dokunulmamış veri
│   └── processed/                  # Pipeline çıktıları (train/test split)
├── notebooks/
│   └── final_analysis.ipynb        # Ana CRISP-DM notebook
├── models/
│   ├── best_model.joblib           # Final model
│   ├── best_model_metadata.json    # Final model metadata
│   └── artifacts/
│       └── full_preprocessor.pkl   # Production preprocessor artifact
├── app/
│   ├── streamlit_app.py            # veya gradio_app.py
│   └── index.html                  # Tanıtım sitesi
├── figures/                        # Grafik çıktıları (PNG)
├── reports/                        # PDF rapor
├── requirements.txt
├── README.md
├── CLAUDE.md                       # Bu dosya
└── TASKS.md                        # Görev dağılımı ve todo listesi
```

---

## Notebook Yapısı

Notebook `notebooks/final_analysis.ipynb` dosyasında 6 ana bölüm:

```
Section 0: Hero & Proje Tanıtımı (HTML)
Section 1: Business Understanding       ← Feza
Section 2: Data Understanding / EDA     ← Feza
Section 3: Data Preparation + Pipeline  ← Cenker
Section 4: Modeling                     ← Berkay
Section 5: Evaluation                   ← Berkay
Section 6: Deployment Simulation        ← Ethem
```

---

## Teknik Standartlar

### Zorunlular
- Tüm grafikler **Plotly** ile yapılmalı (matplotlib yasak)
- Her grafik altında **veri analisti yorumu** (2-3 cümle)
- Pipeline: **sadece train setinde fit**, test setine sadece transform
- En az **10 farklı** classification modeli
- Tüm metrikler: Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix
- Stratified split: `test_size=0.2`, `random_state=42`
- Model kayıt: `joblib` ile `.joblib` uzantısı

### Notebook Görsellik Standartları
- HTML header'lar: her section için renkli başlık bloğu
- Renk paleti tutarlı olmalı (tüm grafikte aynı tema)
- Dark/modern tema tercih edilmeli
- Tablo çıktıları `display(df.style...)` ile stilize edilmeli
- Kod hücreleri arasına açıklayıcı Markdown hücreleri ekle

---

## Değerlendirme Rubriği

| Kriter | Puan |
|--------|------|
| Problem Tanımı ve İş Bağlamı | 15 |
| EDA ve Veri Kalitesi | 15 |
| Veri Hazırlama ve Pipeline | 20 |
| Modelleme ve Karşılaştırma | 20 |
| Değerlendirme ve Karar Yorumu | 15 |
| Deployment, GitHub ve Sunum | 15 |
| **Toplam** | **100** |

---

## Review Prosedürü

`/review` komutu çalıştırıldığında Claude şunları kontrol eder:

1. Notebook `Restart & Run All` ile hatasız çalışıyor mu?
2. Her CRISP-DM fazı eksiksiz mi?
3. Plotly standartlarına uyuluyor mu?
4. Pipeline veri sızıntısı içeriyor mu?
5. 10+ model denenmiş mi?
6. İş bağlamı yorumu yapılmış mı?
7. Deployment çalışıyor mu?
8. GitHub yapısı standarda uyuyor mu?

---

## Önemli Notlar

- `data/raw/` klasörüne **dokunma** — orijinal veri korunmalı
- `data/processed/` klasörüne Cenker yazar, diğerleri oradan okur
- `models/` klasörüne Ethem yazar, Berkay oradan yükler
- Teslim paketi: notebook + rapor + uygulama + model dosyaları + README + sunum
