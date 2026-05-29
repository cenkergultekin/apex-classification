# APEX-V2 | Görev Dağılımı
### AI Student Impact — Classification Final Projesi
> CRISP-DM metodolojisi · 4 kişi · Sıralı devir zinciri

---

## Ekip & Sorumluluklar

| Kişi | Faz | Başlık | Puan |
|------|-----|--------|------|
| **Feza** | 1–2 | Business Understanding + EDA | 30 |
| **Cenker** | 3 | Data Preparation + Pipeline | 20 |
| **Ethem** | 4–5 | Modeling + Evaluation | 35 |
| **Berkay** | 6 | Deployment + Tanıtım Sitesi | 15 + 10 |

---

## FEZA — Faz 1 & 2 · Business Understanding + EDA
> Notebook: Section 1 & 2 · Çıktı: EDA tamamlanmış, temiz veri profili

### Faz 1 — Business Understanding
- [ ] Proje başlığı ve problem cümlesini yaz (1 paragraf, iş bağlamı açık)
- [ ] Hedef değişkeni tanımla ve tahmin etmeye çalıştığımız kararı belirt
- [ ] Başarı metriklerini proje başında tanımla (F1? ROC-AUC? Recall öncelikli mi?)
- [ ] Yanlış pozitif / yanlış negatif maliyetini iş açısından yorumla
- [ ] Veri setinin gerçek dünya bağlantısını kur (nereden geldi, kime hizmet eder)
- [ ] Hero HTML header'ı notebook'a ekle

### Faz 2 — Data Understanding (EDA)
- [ ] Veri setini yükle (`data/raw/ai_student_impact_dataset.csv`)
- [ ] Shape, dtypes, memory kullanımı — genel tablo
- [ ] Target değişken dağılımı (Plotly bar + pie) — sınıf dengesi analizi
- [ ] Eksik değer ısı haritası (Plotly heatmap)
- [ ] Sayısal değişkenler: dağılım histogramları + box plot (Plotly)
- [ ] Kategorik değişkenler: frekans grafikleri
- [ ] Korelasyon matrisi (Plotly heatmap, target ile ilişki vurgulu)
- [ ] Aykırı değer tespiti (IQR yöntemi, görselleştir)
- [ ] Her grafiğin altına 2-3 cümle **veri analisti yorumu** yaz
- [ ] EDA bulgularını özetleyen bir tablo/markdown bloğu ekle
- [ ] `figures/eda_*.png` olarak grafikleri kaydet

--> EKSTRA RAPOR

**Devir:** Cenker'e → temiz EDA, hangi sütunların encoding/scaling gerektirdiğini belirt

---

## CENKER — Faz 3 · Data Preparation + Pipeline
> Notebook: Section 3 · Çıktı: `data/processed/` klasöründe hazır veri + pipeline

### Veri Temizleme
- [ ] Feza'nın EDA bulgularını referans al (eksik değer stratejisini gerekçelendir)
- [ ] Eksik değerleri impute et (sayısal: median/mean, kategorik: mode/sabit)
- [ ] Aykırı değer stratejisini uygula (clip veya IQR-based removal — gerekçeli)
- [ ] Duplicate satır kontrolü ve temizliği

### Feature Engineering
- [ ] Anlamlı yeni özellikler türet (en az 2-3 feature)
- [ ] Kategorik encoding (OneHot / Label / Ordinal — her biri için gerekçe)
- [ ] Sayısal scaling (StandardScaler / MinMaxScaler — model grubuna göre)
- [ ] Target encode gerekiyorsa uygula

### Pipeline & Veri Bölme
- [ ] `sklearn.pipeline.Pipeline` ile preprocessing zinciri kur
- [ ] `ColumnTransformer` ile farklı sütun tiplerini ayır
- [ ] Stratified train-test split (`test_size=0.2`, `random_state=42`)
- [ ] Veri sızıntısı kontrolü: fit sadece train'de, transform her ikisinde
- [ ] `data/processed/X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv` kaydet
- [ ] `pipeline.joblib` olarak `models/` klasörüne kaydet
- [ ] Pipeline adımlarını görsel şema ile göster (HTML tablo veya Plotly)

**Devir:** Ethem'e → hazır train/test split, pipeline objesi, feature listesi

---

## BERKAY — Faz 4 & 5 · Modeling + Evaluation
> Notebook: Section 4 & 5 · Çıktı: `models/best_model.joblib`, karşılaştırma tablosu

### Faz 4 — Modeling (min. 10 model)
- [ ] Logistic Regression
- [ ] Decision Tree
- [ ] Random Forest
- [ ] Gradient Boosting (sklearn)
- [ ] XGBoost
- [ ] LightGBM
- [ ] AdaBoost
- [ ] K-Nearest Neighbors
- [ ] Support Vector Machine (SVM)
- [ ] Naive Bayes
- [ ] *(Bonus)* Neural Network (MLPClassifier veya Keras)
- [ ] Tüm modeller için cross-validation (5-fold, stratified)
- [ ] Karşılaştırma tablosu: Accuracy, Precision, Recall, F1, ROC-AUC
- [ ] Model karşılaştırma bar chart'ı (Plotly, metrik bazlı)
- [ ] En iyi 3 modeli seç ve gerekçelendir

### Hiperparametre Optimizasyonu
- [ ] En iyi model için GridSearchCV veya RandomizedSearchCV uygula
- [ ] Optimum parametreleri ve iyileşme oranını raporla

### Faz 5 — Evaluation
- [ ] Final model için Confusion Matrix (Plotly annotated heatmap)
- [ ] ROC eğrisi (Plotly, tüm sınıflar için — AUC değerleriyle)
- [ ] Precision-Recall eğrisi
- [ ] Classification report (güzel HTML tablo formatında)
- [ ] Overfitting analizi (train vs test skoru farkı)
- [ ] Feature importance / SHAP görselleştirme (en az top-10 feature)
- [ ] **İş bağlamına çevirme:** hangi hata tipinin maliyeti daha yüksek? neden?
- [ ] Model seçimini savun: neden bu model, alternatiflerine göre avantajı?
- [ ] `models/best_model.joblib` kaydet

**Devir:** Berkay'e → kaydedilmiş model + pipeline, örnek input formatı

---

## ETHEM — Faz 6 + Tanıtım Sitesi · Deployment
> Çıktı: `app/` klasöründe çalışan uygulama + tanıtım sitesi

### Faz 6 — Deployment
- [ ] `app/streamlit_app.py` veya `app/gradio_app.py` yaz
- [ ] Model ve pipeline'ı yükle (`joblib.load`)
- [ ] Kullanıcıdan her feature için input al (slider/selectbox/number_input)
- [ ] Tahmin çıktısını göster (sınıf + olasılık + görsel feedback)
- [ ] Uygulama ekran görüntüsünü `figures/app_screenshot.png` olarak kaydet
- [ ] Notebook'ta deployment bölümünü yaz (nasıl çalıştırılır, demo görseli)

### Tanıtım Sitesi (HTML/CSS)
- [ ] `app/index.html` — tek sayfalık proje tanıtım sitesi
- [ ] Hero section: proje adı, ekip, kısa problem cümlesi
- [ ] Veri hikayesi bölümü (neden bu veri, hangi karar)
- [ ] Ekip rolleri kartları (4 kişi, her biri için kısa açıklama)
- [ ] Model sonuçları özet kartları (en iyi metrikler)
- [ ] Karar değeri bölümü (model ne işe yarar, sınırlılıkları)
- [ ] Modern, koyu tema — gradient aksan renkler

### README & Dokümantasyon
- [ ] `README.md` — Mermaid akış diyagramı dahil
- [ ] Kurulum ve çalıştırma talimatları
- [ ] `requirements.txt` oluştur (tüm kullanılan paketler)
- [ ] Dosya yapısını README'ye ekle

---

## Devir Zinciri

```
Feza (EDA biter)
  → Cenker'e handoff notu: hangi sütunlar encoding/scaling gerektiriyor
    → Cenker (Pipeline biter)
      → Ethem'e handoff notu: processed data yolu, pipeline objesi, feature listesi
        → Ethem (Model + Eval biter)
          → Berkay'e handoff notu: best_model.joblib yolu, örnek input dict
```

Her faz bittikten sonra ilgili kişi `TASKS.md`'deki kutucukları `[x]` olarak işaretler.

---

## Review Kontrol Listesi (tüm faz bittikten sonra)

- [ ] Notebook baştan sona `Restart & Run All` ile hatasız çalışıyor
- [ ] Tüm grafikler Plotly ile yapılmış, statik matplotlib yok
- [ ] Her grafik altında veri yorumu var
- [ ] Pipeline sadece train'de fit edilmiş
- [ ] 10+ model denenmiş ve karşılaştırma tablosu var
- [ ] Deployment uygulaması gerçekten çalışıyor (ekran görüntüsü var)
- [ ] `models/` klasöründe `.joblib` dosyaları var
- [ ] README Mermaid diyagramıyla tamamlanmış
- [ ] GitHub repo yapısı standartta belirtilen şemaya uyuyor
