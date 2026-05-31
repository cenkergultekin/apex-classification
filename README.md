# APEX-V2 | Öğrenci Akademik Tükenmişlik Sınıflandırma Analizi
> **CRISP-DM Metodolojisi ile Yapay Zeka Etkisi ve Akademik Tükenmişlik Risk Tahmini**

Bu proje, öğrencilerin üretken yapay zeka (GenAI) araçlarını kullanım yoğunlukları, geleneksel çalışma alışkanlıkları, sınav kaygı düzeyleri ve akademik performans göstergelerinden hareketle **Akademik Tükenmişlik Seviyelerini** sınıflandırmayı amaçlamaktadır.

---

## 🚀 Hızlı Başlangıç (Streamlit Arayüzü)

Geliştirilen premium arayüzü yerel ortamınızda ayağa kaldırmak için aşağıdaki adımları takip edebilirsiniz:

### 1. Gereksinimlerin Yüklenmesi
Öncelikle gerekli tüm Python paketlerini yükleyin:
```bash
pip install -r requirements.txt
```

### 2. Streamlit Uygulamasının Çalıştırılması
Terminal üzerinden uygulamayı başlatın:
```bash
streamlit run app/streamlit_app.py
```

Uygulama başarıyla başlatıldığında tarayıcınızda otomatik olarak açılacaktır (varsayılan adres: `http://localhost:8501`).

---

## 📁 Klasör Yapısı

```
apex-classification/
├── app/
│   ├── streamlit_app.py            # Streamlit canlı tahmin ve analitik arayüzü
│   ├── preprocessing.py            # Üretime hazır, robust veri ön işleme modülü
│   └── __init__.py
├── data/
│   ├── raw/                        # Dokunulmamış orijinal veri seti
│   │   └── ai_student_impact_dataset.csv
│   └── processed/                  # Ön işlemeden geçmiş eğitim/test veri setleri
│       ├── ai_student_impact_cleaned.csv
│       ├── train.csv
│       └── test.csv
├── notebooks/
│   └── final_analysis.ipynb        # Ana CRISP-DM notebook'u (6 Faz)
├── models/
│   ├── best_model.joblib           # Eğitilmiş en iyi Gradient Boosting modeli
│   ├── best_model_package.joblib   # Model + Karar kuralları + Test metrikleri paketi
│   ├── best_model_metadata.json    # Model yapılandırma ve başarım dosyası
│   └── artifacts/
│       └── full_preprocessor.pkl   # Eğitilmiş ColumnTransformer / Preprocessor nesnesi
├── figures/                        # Grafik ve arayüz ekran görüntüsü çıktıları
├── requirements.txt                # Gerekli bağımlılıklar listesi
├── README.md                       # Bu doküman
├── CLAUDE.md                       # Geliştirme standartları rehberi
└── TASKS.md                        # Görev dağılımı ve ilerleme takip tablosu
```

---

## 👥 Ekip & Sorumluluklar

Proje **CRISP-DM (Cross-Industry Standard Process for Data Mining)** metodolojisi izlenerek 6 aşamada ve 4 kişilik bir ekiple tamamlanmıştır:

| Ekip Üyesi | Rol | Sorumlu Olduğu CRISP-DM Aşamaları |
| :--- | :--- | :--- |
| **Feza** | Veri Analisti | **Faz 1 & 2:** Business Understanding + EDA (Plotly grafik analizleri) |
| **Cenker** | Veri Mühendisi | **Faz 3:** Data Preparation + ColumnTransformer / Pipeline tasarımı |
| **Ethem** | ML Mühendisi | **Faz 4 & 5:** Modeling + Evaluation (10+ model kıyası, Hiperparametre) |
| **Berkay** | Yazılım Mühendisi | **Faz 6:** Deployment + Streamlit Arayüzü & Sunum |

---

## 📊 Model & Başarım Sonuçları

En iyi sonucu veren tuned **Gradient Boosting** modeline ait test verisi performans metrikleri:

* **Accuracy (Doğruluk):** `%80.28`
* **F1 Macro Score:** `0.7925`
* **Recall (Yüksek Sınıfı):** `0.6701` (Tükenmişliği en az ıskalama ile bulma başarısı)
* **Precision (Yüksek Sınıfı):** `0.8555`
* **Karar Eşiği (Tuned Threshold):** `0.53` (Yüksek riskli öğrencileri yakalamak amacıyla optimize edilmiştir)

---

## 🧪 Model Retraining (Geliştirici Notu)

Eğer yerel scikit-learn sürümünüz ile model pickling uyuşmazlığı yaşıyorsanız, modeli yerel ortamınızda aynı tohum değerleri ve hiperparametrelerle otomatik olarak yeniden eğitmek için geçici retraining scriptini çalıştırabilirsiniz:
```bash
python scratch/retrain.py
```
Bu komut `models/` altındaki tüm model ve preprocessor nesnelerini yerel kütüphane sürümleriniz ile güncelleyecektir.
