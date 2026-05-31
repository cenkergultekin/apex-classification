# Geliştirici Rehberi · APEX

> Bu rehberi okuduğunda **her fazda ne yaptık, neden o yöntemi seçtik ve sonucu nasıl ölçtük** öğreneceksin. Olabildiğince sade dilde yazıldı — teknik terimler ilk kullanıldığı yerde tek cümlede açıklanır.

---

## 0. Projenin Hedefi

Bir öğrencinin verilerine bakıp **akademik tükenmişlik riski yüksek mi, düşük mü** diye tahmin eden bir model kuruyoruz. Sonra bu modeli Streamlit ile bir arayüze koyup üniversite rehberliklerinin kullanabileceği bir araca dönüştürdük.

- **Veri:** ~28.000 öğrenci, 16 sütun
- **Tahmin etmeye çalıştığımız sütun:** `Tükenmişlik` (Düşük / Yüksek)
- **Pozitif sınıf:** `Yüksek` — yani modelin asıl yakalaması gereken sınıf

---

## Faz 1 · İş Anlama (Business Understanding) — *Feza*

**Soru:** Bu projeden kim faydalanır, başarı kriteri nedir?

- **Paydaş:** Üniversite rehberlik birimleri. Riskli öğrenciye **erken müdahale** yapabilmek istiyorlar.
- **Karar kuralı:** Model "Yüksek" derse rehber öğrenciyle erken görüşme planlar. Yani **Yüksek sınıfı kaçırmak (False Negative) Düşük sınıfı yanlış işaretlemekten daha pahalı**.
- **Bu yüzden:** Sadece accuracy değil, **Recall (Yüksek)** ve **F1-macro** metriklerini ön plana aldık. Bir öğrenciyi yanlışlıkla "Yüksek" görmek tolere edilebilir (rehber gereksiz görüşme yapar), ama "Düşük" deyip kaçırmak vahim olur.

> **Çıktı:** Notebook Section 1 — paydaş kartı + başarı kriterleri + risk maliyet matrisi.

---

## Faz 2 · Veri Anlama / EDA — *Feza*

**Soru:** Verinin içinde ne var, hangi sinyaller burnout ile ilişkili?

- **Profilleme:** Sütun tipleri, eksik değerler, sınıf dağılımı (~%65 Düşük / %35 Yüksek).
- **Dağılım analizleri:** Her sütun ailesi için farklı görsel form — sayısal, kategorik ve ikili (target ile çapraz) ilişkiler.
- **Korelasyon matrisi:** 15 değişken arası Pearson; yüksek korelasyonlu çiftler Phase 3'te kolinearite uyarısı olarak işaretlendi.

### Yakaladığımız ana sinyaller
- `Haftalık AI Saati`, `Algılanan AI Bağımlılığı`, `Sınav Kaygı Düzeyi` → Yüksek tükenmişlikle güçlü pozitif ilişki.
- `Beceri Kalıcılık Skoru` → ters yönlü (yüksek skor düşük risk).
- `Okunan Bölüm` ve `Birincil Kullanım Amacı` → kategoriler arası risk farkları belirgin (WoE'ye aday).

> **Çıktı:** Notebook Section 2 — her grafik altında 2-3 cümle "Veri Analisti Yorumu" kartı.

---

## Faz 3 · Veri Hazırlama + Pipeline — *Cenker*

Bu faz projenin en kritik mühendislik aşaması. Burada yapılan hatalı bir adım sonraki tüm metrikleri yalanlar. Bu yüzden her karar **kanıt + ablation** üzerine kuruldu.

### 3.1 Veri Temizleme

- **Duplicate satırlar:** Tespit edildi, silindi.
- **Öğrenci No:** Kimlik sütunu — model için anlamsız, drop.
- **İmkânsız değerler → NaN:** GNO < 0 veya > 4, Likert 1-10 dışı, negatif saat. Bunlar **silinmedi**, NaN'a dönüştürüldü ki imputer doğru aralıkta tamamlasın.
- **Boş hedef satırlar:** Drop (hedefi olmayan satır modele zarar verir).

### 3.2 Eksik Değer Doldurma (Imputation) — *neden bu yöntem?*

Eksik oranlarımız %5'in altındaydı, yani **basit yöntemler güvenli**. Karmaşık imputerlara (KNN, IterativeImputer) ihtiyaç yok:

| Sütun Tipi | Yöntem | Neden? |
| :--- | :--- | :--- |
| Sayısal (`Haftalık AI Saati`, `Sınav Kaygı Düzeyi` vb.) | **Median (medyan)** | Aykırı değerlerden etkilenmez; ortalamadan daha sağlam. |
| Ordinal (`Sınıf Düzeyi`, `Prompt Yazma Becerisi`, `Kurum Politikası`) | **En sık görülen (most_frequent)** | Sıralı kategoride ortalama tanımsız; en sık görülen değer en az şaşırtıcı varsayım. |
| Nominal / WoE (`Okunan Bölüm`, `Birincil Kullanım Amacı`) | **En sık görülen** | Kategorik olduğu için aynı mantık. |
| Boolean (`Ücretli Abonelik`) | **En sık görülen** | İki değerli; mod en güvenli doldurma. |

**Ek olarak:** `Beceri Kalıcılık Skoru` ve `Prompt Yazma Becerisi` için **"Eksik Miydi?" indikatör sütunları** ekledik. Çünkü bu iki sütunda eksiklik rastlantısal değil — eksik olması başlı başına bir sinyal olabilir.

### 3.3 Aykırı Değer (Outlier) Politikası — *baskılama YAPMADIK*

> **Önemli karar:** Aykırı değerleri **winsorize etmedik, RobustScaler kullanmadık, IQR ile kırpmadık**.

**Neden?**
- IQR analizi yaptığımızda aykırı oranı **%0.5'in altında** çıktı. Yani veri zaten temiz.
- Çok az aykırı varken `RobustScaler` kullanmak gereksiz karmaşıklık; `StandardScaler` daha standart ve okunabilir.
- "İmkânsız değer" olanları (negatif saat gibi) zaten 3.1'de NaN'a çevirip imputer'a verdik. Yani kötü-data filtremiz **domain kuralları üzerinden**, istatistiksel kırpmaya değil.
- Tree-based modeller (Gradient Boosting, Random Forest, XGBoost) aykırı değere zaten dayanıklıdır — kırpmak gerçek sinyali kaybetmek demek olabilir.

> **Çıktı:** Notebook 2.9'da Tukey IQR raporu var (görsel olarak gösterdik), ama 3.x'te **bilinçli olarak kırpmadık**.

### 3.4 Encoding (Kategorik → Sayısal) — *3 farklı yöntem, 3 farklı sütun ailesi*

| Aile | Sütunlar | Yöntem | Neden? |
| :--- | :--- | :--- | :--- |
| Ordinal (sıralı) | Sınıf Düzeyi, Prompt Becerisi, Kurum Politikası | **OrdinalEncoder** | Bu sütunlarda doğal bir sıralama var (1. Sınıf < 2. Sınıf < ... veya Başlangıç < Orta < İleri). Sıra korunmalı. |
| Nominal (sırasız, çok kategorili) | Okunan Bölüm, Birincil Kullanım Amacı | **WoE Encoder** | OneHotEncoder kullansak 30+ sütun ekler, model şişer. **WoE (Weight of Evidence)** her kategoriyi *binary log-odds* değerine çevirir — yani "bu kategori pozitif sınıfa ne kadar meyilli" sayısı. Tek sütun, anlamlı yön. |
| Boolean | Ücretli Abonelik | **int 0/1** | Doğrudan sayıya çevirme yeterli. |

#### WoE'yi biraz açalım

WoE = `ln(P(Yüksek | kategori) / P(Düşük | kategori))` — yani her kategori için "bu grup ne kadar yüksek-risk?" sorusunun cevabı.

- Eğer kategori daha çok Yüksek sınıf içeriyorsa WoE değeri **pozitif**.
- Daha çok Düşük sınıf içeriyorsa WoE değeri **negatif**.
- Tek sütun. Tek anlamlı sayı. Kategori sayısı arttıkça modeli şişirmiyor.

**Önemli:** WoE Encoder yalnızca **train setinde** fit edilir — yoksa target leakage olur (test setindeki target'tan kategori dağılımını öğrenmek hile). Bizim `ProductionPreprocessor.fit()` bu kuralı uygular.

### 3.5 Scaling (Ölçeklendirme)

**Kullandık:** `StandardScaler` — her sütunu ortalaması 0, std'si 1 olacak şekilde dönüştürür.

**Hangi sütunlara?** Tüm sayısal + ordinal-encoded + WoE-encoded sütunlara. Boolean ve missing-indicator sütunları zaten 0/1, scale gerektirmiyor.

**Neden RobustScaler değil?** Yukarıda söyledik: aykırı oranımız %0.5 altında, standart yöntem yeterli. RobustScaler median+IQR kullanır; biz tabloda istatistiksel kırpma istemiyoruz.

**Neden ölçeklemeli?** Bazı modeller (Logistic Regression, KNN, MLP) ölçek farklılıklarına duyarlıdır. Ağaç bazlı modeller (Gradient Boosting, RF, XGBoost) duyarlı değildir ama scaler hepsi için pipeline'ı **tek tip** tutar. Pipeline tutarlılığı bakım kolaylığı sağlar.

### 3.6 Feature Engineering — *ne ekledik, ne çıkardık?*

#### Eklediğimiz tek özellik:
- **`Toplam Çalışma Yükü` = Haftalık AI Saati + Geleneksel Çalışma Saati**
  Neden? Çünkü iki kanal arasındaki **toplam kapasite** kavramı modele tek bir sayıyla "öğrencinin haftalık yükü" sinyalini veriyor. MI (mutual information) testinde anlamlı katkı sağladı.

#### Çıkardığımız özellikler:
- **GPA bloğu (Dönem Öncesi/Sonrası GNO + türevleri):** Ablation testinde bu blok modelin Yüksek-recall'unu **düşürdüğü** ortaya çıktı. Üstelik canlı tahmin senaryosunda öğrencinin GNO'sunu girmesi her zaman mümkün değil. Drop ettik.
- **7 ratio/product türevi** (`AI / Çalışma Oranı`, `Sınav Kaygısı × AI Saati` gibi): Bunlar yüksek-MI değişkenlerden türetildikleri için **kolinearite** üretiyorlardı (yani aynı bilgiyi farklı yüzlerle tekrar ediyorlardı). Tree-based modeller bu etkileşimleri **kendileri öğrenir** — biz onlara hazır vermeye çalışınca model gürültüye odaklanıyordu.

**Sonuç:** 21 feature → **15 feature**. Daha sade, daha hızlı, daha okunaklı. Bu "Lean Core" prensibi.

### 3.7 Train/Test Split + Pipeline Fit — *en kritik kural*

```python
# 1) Dışarıdaki test'i ayır (modele asla bakmayacak)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

# 2) Train içinden iç validation ayır (threshold seçimi için)
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.15, stratify=y_train, random_state=42
)

# 3) Preprocessor SADECE train'de fit edilir
prep.fit(X_train, y_train)
X_train_final = prep.transform(X_train)
X_test_final = prep.transform(X_test)   # ← test'e sadece transform
```

**Stratified:** Sınıf oranını her parçada koru.
**random_state=42:** Aynı sonucu üretmek için sabit tohum.

> **Çıktı:** `data/processed/train.csv` + `data/processed/test.csv` (15 feature + target), `models/artifacts/full_preprocessor.pkl` (fit edilmiş pipeline).

---

## Faz 4 · Modelleme — *Berkay*

**Soru:** Hangi sınıflandırma algoritması bu problem için en iyisi?

### 4.1 Aday Modeller (10 adet)

Standart "en az 10 model" kuralını şu adaylarla karşıladık:

1. **Logistic Regression** (lineer baseline, yorumlanabilir)
2. **Decision Tree** (kural-tabanlı baseline)
3. **Random Forest** (ağaç topluluğu)
4. **Gradient Boosting** (klasik boosting)
5. **XGBoost** (yüksek performans boosting, `binary:logistic`)
6. **LightGBM** (hızlı boosting alternatifi)
7. **AdaBoost** (zayıf öğrenici boosting)
8. **K-Nearest Neighbors** (mesafe tabanlı)
9. **Gaussian Naive Bayes** (olasılıksal, hızlı)
10. **MLP (Neural Network)** (küçük yapay sinir ağı)

**SVM eklemedik** çünkü 28K satırda CV süresini gereksiz uzatıyordu ve katkısı görünmüyordu.

### 4.2 Cross-Validation Stratejisi

**StratifiedKFold (n_splits=5, shuffle=True, random_state=42)** — her fold'da sınıf oranı korunur. K-fold'un "Stratified" versiyonu özellikle dengesiz sınıflarda zorunlu.

Her model 5 fold ile değerlendirildi, sonra **F1-macro** ortalaması üzerinden sıralandı.

### 4.3 Sınıf Dengesizliği — *ne yaptık?*

%65 / %35 dağılım ciddi değil ama bilinçli ele aldık:
- Destekleyen modellerde `class_weight='balanced'` (LogReg, Decision Tree, Random Forest, LightGBM) — sınıfların ağırlığını otomatik tersine çevirir.
- Desteklemeyen modellerde (Gradient Boosting, XGBoost, AdaBoost, NB, KNN, MLP) **threshold tuning** ile kompanse ettik (4.5'e bak).
- SMOTE/ADASYN kullanmadık — %35 pozitif sınıf zaten "ağır dengesiz" değil, sentetik veri üretmenin getirisi belirsiz.

### 4.4 Hiperparametre Tuning

En iyi 2 model üzerinde **RandomizedSearchCV** çalıştırdık (8-12 iterasyon, 3-fold iç CV). Grid yerine random çünkü:
- Daha az kombinasyon dener ama benzer sonuca yakınsar.
- Zaman açısından çok daha pratik.

**Kazanan:** Gradient Boosting — `n_estimators=180, max_depth=3, learning_rate=0.05, subsample=0.85, min_samples_leaf=20`.

### 4.5 Threshold Tuning — *neyi seçtik, nerede seçtik?*

Default olarak sklearn `predict()` 0.5 eşik kullanır. Biz buna razı olmadık:
- Recall (Yüksek)'i optimize etmek istiyoruz.
- Bu yüzden 0.30 → 0.70 aralığını **validation setinde** taradık.
- F1-macro'yu maksimize eden eşiği seçtik: **0.53**.

> **KRİTİK:** Threshold **validation seti** üzerinde seçildi, **test seti** üzerinde değil. Eğer test'te seçseydik raporladığımız test skorları "optimistic-biased" olurdu (test seti hem öğrenmek için hem ölçmek için kullanılırsa, ölçüm güvenilirliğini kaybeder).

> **Çıktı:** Notebook Section 4 — model envanteri kartı, CV karşılaştırma tabloları, hiperparametre arama grafiği, threshold tuning eğrisi.

---

## Faz 5 · Değerlendirme — *Berkay*

### 5.1 Raporladığımız Metrikler

| Metrik | Değer | Ne anlama gelir? |
| :--- | :--- | :--- |
| **Accuracy** | 0.7949 | Tahminlerin yüzde kaçı doğru |
| **F1-Macro** | 0.7830 | Sınıflar arası dengeli F1 ortalaması — dengesiz veri için temel skor |
| **ROC-AUC** | 0.8690 | Modelin ayırma gücü (1.0 = mükemmel, 0.5 = rastgele) |
| **Recall (Yüksek)** | 0.6485 | Gerçekte Yüksek olan öğrencilerin yüzde kaçını yakaladık |
| **Precision (Yüksek)** | 0.8411 | "Yüksek" dediklerimizin yüzde kaçı gerçekten Yüksek |

### 5.2 Confusion Matrix

```
                 Tahmin: Düşük   Tahmin: Yüksek
Gerçek Düşük       2968               306        ← 306 false positive
Gerçek Yüksek       878              1620        ← 878 false negative
```

- **False Positive (306):** Aslında düşük riskli olan ama "Yüksek" dediğimiz öğrenciler. Maliyet düşük: rehber gereksiz görüşme yapar.
- **False Negative (878):** Aslında yüksek riskli olan ama "Düşük" dediğimiz öğrenciler. Maliyet yüksek: öğrenci destek almıyor. → İyileştirme yönümüz.

### 5.3 ROC Eğrisi

AUC = 0.87 — modelin ayırma gücü "iyi" kategorisinde. Mükemmel değil (0.95+) ama tek-değişkenli bir threshold'la makul karar veriyor.

### 5.4 Feature Importance (Hangi sütunlar etkili?)

Native importance (Gradient Boosting'in kendi katkı skoru) ile fallback hiyerarşisi (coef → permutation importance) kullandık. Üst sıralar tipik olarak:
1. Sınav Kaygı Düzeyi
2. Beceri Kalıcılık Skoru
3. Algılanan AI Bağımlılığı
4. Haftalık AI Saati
5. Toplam Çalışma Yükü (engineered feature işe yaradı)

### 5.5 İş Yorumu

Model **otomatik karar vermez** — riskli görünen öğrenciyi rehberliğin radarına alır. Korelasyon ≠ nedensellik: yüksek AI saati tükenmişliği **yaratmıyor** olabilir, sadece **birlikte gözüküyor**. Bu nüansı Section 5.6'da açıkça söyledik.

> **Çıktı:** Notebook Section 5 — confusion matrix, ROC, classification report, feature importance, ablation test sonucu.

---

## Faz 6 · Deployment — *Ethem*

### 6.1 Streamlit Uygulaması

`app/streamlit_app.py` iki sekmeden oluşuyor:
- **Tekil Tahmin:** Kullanıcı 12 girdiyi formdan doldurur → `ProductionPreprocessor.transform()` → `model.predict_proba()` → `P(Yüksek)` ve sınıf etiketi gösterilir.
- **Analitik:** Test seti üzerinde confusion matrix, ROC, feature importance — şeffaflık için.

Uygulamayı başlatmak:
```bash
streamlit run app/streamlit_app.py
```

### 6.2 Retraining Script

`app/retrain.py` tek komutla tüm pipeline'ı yeniden üretir:
```bash
python app/retrain.py
```
Bu script Faz 3.7'deki split + Faz 4'teki en iyi parametreleri ezbere uygular. scikit-learn sürüm uyuşmazlıkları ya da local rebuild ihtiyaçları için.

### 6.3 Artifact Yapısı

| Dosya | İçerik |
| :--- | :--- |
| `models/best_model.joblib` | Sadece eğitilmiş model nesnesi |
| `models/best_model_package.joblib` | Model + threshold + metadata + best_params (tek dosyada) |
| `models/best_model_metadata.json` | İnsan-okunabilir metrik ve yapılandırma raporu |
| `models/artifacts/full_preprocessor.pkl` | Train üzerinde fit edilmiş preprocessor |

### 6.4 Notebook Section 6

Notebook'un sonu uygulamanın başıdır: 6.1'de aynı artifact'ı joblib ile yüklüyoruz, 6.2'de 2 örnek öğrenci profilini tahmin ediyoruz. Streamlit perde arkasında satır satır aynı işlem.

---

## Veri Sızıntısı (Data Leakage) Önlemleri — *Özet kontrol listesi*

Bir veri bilimi projesinin en sinsi hatası **leakage**'dır: modeli ölçtüğümüz veri eğitim sırasında modele sızar ve skorları yapay olarak yükseltir. Bizim koruma katmanlarımız:

- ✅ **Preprocessor sadece train'de fit:** `prep.fit(X_train, y_train)`. Test setine sadece `transform`.
- ✅ **WoE Encoder sadece train target'ı ile fit:** Test target'ı asla kategori istatistiğine girmez.
- ✅ **Threshold validation setinde seçildi:** Test seti **tek sefer** ölçüm için kullanıldı.
- ✅ **Train/test split bir kere yapıldı:** Sonradan "daha iyi skor verir" diye yeniden bölmedik.
- ✅ **Post-event feature drop:** Dönem Sonrası GNO (target'a yakın bir proxy) modelden çıkarıldı.
- ✅ **Reproducibility:** `random_state=42` her yerde.

---

## Hızlı SSS

**S: Modeli kendim çalıştırmak istiyorum ne yapmalıyım?**
A: `pip install -r requirements.txt` → `streamlit run app/streamlit_app.py`. Bu kadar.

**S: scikit-learn sürümü uyuşmazlığı verdi.**
A: `python app/retrain.py` ile yerel ortamında artifact'ları yeniden üret. ~30 saniye sürer.

**S: Notebook'u baştan çalıştırırsam aynı sonuçları alır mıyım?**
A: Evet — `random_state=42` her yerde sabit. Restart & Run All ile uçtan uca çalışır.

**S: WoE değerleri canlı tahminde nasıl çalışıyor?**
A: `ProductionPreprocessor` train'de gördüğü her kategori için WoE değerini hatırlıyor. Canlı tahmin geldiğinde aynı kategoriye aynı sayı atanıyor. Yeni/bilinmeyen kategori gelirse `most_frequent` imputer ile doldurulup en yaygın kategorinin WoE'sini alıyor.

**S: Model performansını nasıl iyileştirebiliriz?**
A: En verimli yatırımlar:
1. SMOTE-Tomek vs class_weight vs threshold tuning ablation tablosu.
2. Optuna ile daha geniş hiperparametre araması (~50 iterasyon).
3. Sınıf bazlı ek feature mühendisliği (örn. öğrenci-bölüm grubunun risk profili).
4. Lightweight bir ensemble (GB + LGBM stacking).

---

## Faydalı Dosya Yolları

| Ne arıyorsun? | Nerede? |
| :--- | :--- |
| Üretim pipeline kodu | `app/preprocessing.py` |
| Streamlit arayüz | `app/streamlit_app.py` |
| Retraining script | `app/retrain.py` |
| Ana CRISP-DM notebook | `notebooks/final_analysis.ipynb` |
| Model + metadata | `models/best_model_*` |
| Phase 3 detaylı agent yönergesi | `docs/agents/classroom-agents/dataprep-expert-agent.md` |
| Geliştirme standartları | `docs/CLAUDE.md` |
