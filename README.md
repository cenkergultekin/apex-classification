# APEX

**Öğrenci Akademik Tükenmişlik Sınıflandırma** — CRISP-DM tabanlı uçtan uca veri bilimi projesi.

> Üretken yapay zeka (GenAI) kullanım yoğunluğu, geleneksel çalışma alışkanlıkları, sınav kaygısı ve akademik göstergelerden hareketle öğrencilerin **akademik tükenmişlik riskini** ikili olarak (Düşük / Yüksek) sınıflandırır.

```
─────────────────────────────────────────────────────────────
  Model        Gradient Boosting · Lean Core (12 feature)
  Test F1      0.7822           ROC-AUC      0.8689
  Recall (Y)   0.6433           Precision    0.8449
  Threshold    0.54 (validation üzerinde seçildi)
─────────────────────────────────────────────────────────────
```

---

## Hızlı Başlangıç

```bash
# 1. Bağımlılıklar
pip install -r requirements.txt

# 2. (opsiyonel) Modeli kendi ortamında yeniden eğit
python app/retrain.py

# 3. Streamlit arayüzünü başlat
streamlit run app/streamlit_app.py
```

Tarayıcı otomatik açılır: `http://localhost:8501`

---

## Klasör Yapısı

```
apex/
├── app/
│   ├── streamlit_app.py        — Canlı tahmin + analitik arayüz
│   ├── preprocessing.py        — ProductionPreprocessor (Lean Core, 12 feature)
│   ├── retrain.py              — Train/val/test split → leak-aware retraining
│   └── __init__.py
│
├── data/
│   ├── raw/                    — Orijinal veri (dokunulmaz)
│   │   └── ai_student_impact_dataset.csv
│   └── processed/              — Pipeline çıktıları
│       ├── ai_student_impact_cleaned.csv
│       ├── train.csv
│       └── test.csv
│
├── notebooks/
│   └── final_analysis.ipynb    — Ana CRISP-DM notebook (6 faz)
│
├── models/
│   ├── best_model.joblib              — Eğitilmiş Gradient Boosting modeli
│   ├── best_model_package.joblib      — Model + threshold + metadata paketi
│   ├── best_model_metadata.json       — Yapılandırma ve test metrikleri
│   └── artifacts/
│       └── full_preprocessor.pkl      — Train üzerinde fit edilmiş preprocessor
│
├── docs/
│   ├── DEVELOPER_GUIDE.md      — Ekip için fazlı açıklama (basit dilde)
│   ├── CLAUDE.md               — Geliştirme standartları
│   └── agents/                 — Ajan yönergeleri ve faz planları
│
├── figures/                    — Grafik PNG çıktıları
├── reports/                    — PDF rapor çıktıları
├── requirements.txt
└── README.md
```

---

## Ekip ve Sorumluluklar

CRISP-DM 6 fazı dört kişilik bir ekiple yürütüldü.

| Üye        | Rol                | CRISP-DM Fazı                                  |
|------------|--------------------|------------------------------------------------|
| **Feza**   | Veri Analisti      | Faz 1 & 2 — Business Understanding + EDA       |
| **Cenker** | Veri Mühendisi     | Faz 3 — Data Preparation + Pipeline            |
| **Berkay** | ML Mühendisi       | Faz 4 & 5 — Modeling + Evaluation              |
| **Ethem**  | Yazılım Mühendisi  | Faz 6 — Deployment + Streamlit                 |

Her fazda kim ne yaptı ve neden o yöntemi seçti → **[docs/DEVELOPER_GUIDE.md](docs/DEVELOPER_GUIDE.md)**

---

## Veri ve Hedef

| | |
|---|---|
| **Veri seti**         | `data/raw/ai_student_impact_dataset.csv` |
| **Gözlem sayısı**     | ~28.000 öğrenci |
| **Hedef değişken**    | `Tükenmişlik` (Düşük / Yüksek) |
| **Pozitif sınıf**     | `Yüksek` (yüksek tükenmişlik riski) |
| **Sınıf dağılımı**    | ~%65 Düşük / ~%35 Yüksek |

---

## Üretim Pipeline · Lean Core

Üretim katmanı **12 sabit feature** üreten deterministik bir kontrat. `fit()` yalnızca train setinde çağrılır; test setine sadece `transform()` uygulanır.

```
Ham 12 sütun
   │
   └─► ProductionPreprocessor.fit(X_train, y_train)
         │
         ├─ Median imputer        — sayısal sütunlar
         ├─ Most-frequent imputer — ordinal ve nominal sütunlar
         ├─ OrdinalEncoder        — Sınıf Düzeyi, Prompt, Kurum Politikası
         ├─ WoE Encoder           — Okunan Bölüm, Birincil Kullanım Amacı
         ├─ Missing-indicator     — 2 sütun
         └─ StandardScaler        — sayısal + ordinal + WoE
   │
   └─► 12 feature ─► Gradient Boosting ─► P(Yüksek) ─► threshold uygula
```

**Veri sızıntısı önlemleri**

- Preprocessor yalnızca train üzerinde fit edilir; test setine sadece `transform()` uygulanır.
- Karar eşiği (threshold) **iç validation split** üzerinde seçilir; test seti **tek sefer** ölçüm için kullanılır.

---

## Test Sonuçları

En iyi model: **Gradient Boosting** (tuned, 12 feature)

| Metrik              | Değer  |
|---------------------|--------|
| Test Accuracy       | 0.7945 |
| Test F1-Macro       | 0.7822 |
| Test ROC-AUC        | 0.8689 |
| Recall (Yüksek)     | 0.6433 |
| Precision (Yüksek)  | 0.8449 |
| Karar Eşiği         | 0.54   |

**Confusion Matrix** (test seti)

```
                  Tahmin: Düşük   Tahmin: Yüksek
  Gerçek Düşük       2979              295
  Gerçek Yüksek       891             1607
```

Eğitim hiperparametreleri ve son güncelleme bilgisi: `models/best_model_metadata.json`

---

## Yeniden Eğitim

Yerel scikit-learn sürümünüzle artifact uyuşmazlığı yaşarsanız veya pipeline'ı kendi makinenizde sıfırdan üretmek isterseniz:

```bash
python app/retrain.py
```

Script şunları yapar:

1. Cleaned veriyi yükler.
2. Stratified train/test split uygular (`test_size=0.20`, `random_state=42`).
3. Train içinden iç **validation split** ayırır (`test_size=0.15`) — threshold burada seçilir.
4. `ProductionPreprocessor` train üzerinde fit edilir, test'e yalnızca transform uygulanır.
5. Gradient Boosting model en iyi parametrelerle fit edilir.
6. Tüm artifact'lar güncellenir: `best_model.joblib`, `best_model_package.joblib`, `full_preprocessor.pkl`, `train.csv` / `test.csv`, `best_model_metadata.json`.

---

## Notebook Yapısı

Tek dosya, altı CRISP-DM fazı:

```
notebooks/final_analysis.ipynb
├── Section 0 — Hero / Proje Tanıtımı
├── Section 1 — Business Understanding         · Feza
├── Section 2 — Data Understanding / EDA       · Feza
├── Section 3 — Data Preparation + Pipeline    · Cenker
├── Section 4 — Modeling                       · Berkay
├── Section 5 — Evaluation                     · Berkay
└── Section 6 — Deployment Simulation          · Ethem
```

Notebook bir kez **Restart & Run All** ile uçtan uca çalışacak şekilde tasarlanmıştır.

---

## Daha Fazla Bilgi

| Konu                                    | Konum                              |
|-----------------------------------------|------------------------------------|
| Fazlı detaylı açıklama (basit dilde)    | [docs/DEVELOPER_GUIDE.md](docs/DEVELOPER_GUIDE.md) |
| Geliştirme standartları                 | [docs/CLAUDE.md](docs/CLAUDE.md)   |
| Ajan yönergeleri                        | [docs/agents/](docs/agents/)       |
