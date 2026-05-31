# APEX — Faz 3 Agent Talimatları (Cenker)

> **2026-05-31 KRİTİK REVİZYON — Binary classifier + WoE geçişi:**
> Proje binary classification'a (`Yüksek` vs diğer) geçti. Bu revizyondan sonra:
>
> 1. **Encoding:** Nominal kategorikler → **WoE Encoder** (1 sütun, log-odds).
>    Ordinal kategorikler → **OrdinalEncoder kalır** (sıralı doğası WoE ile bozulmamalı).
>    Boolean → `astype(int)` kalır.
> 2. **Scaling:** **StandardScaler** kalır (aykırı oranı %0.5 altında, RobustScaler şart değil).
> 3. **GPA features (Pre/Post/Ortalama/Değişim):** **TAMAMEN DROP**. Ablation
>    incelemesinde recall (Yüksek sınıfı) düşürdüğü ve production'da kırılgan
>    olduğu için elendi. Senaryo notu artık geçersiz.
> 4. **Feature engineering prensibi:** Yüksek-MI değişkenlerden ratio/product
>    türetme YASAK (kolinearite üretir). Yalnızca tek başına zayıf,
>    mantıken birleşince güçlenen sinerji adayları kabul edilir. Eşik:
>    `MI(combined) − max(MI(A), MI(B)) > 0.01`.
> 5. **Feature selection:** IV (Information Value) filtresi ön katmanı eklenir.
>    IV<0.02 otomatik drop, IV>0.5 leakage check.
>
> Detaylar Adım 0.4, 3, 4, 4.5, 5'te güncellenmiştir.
## Data Preparation + Pipeline · Section 3 · 20 Puan

Bu doküman, `notebooks/final_analysis.ipynb` içindeki Section 3'ü doldurmakla görevlendirilen bir kodlama agent'ı için **eksiksiz çalıştırma yönergelerini** tanımlar. Agent bu dosyayı okuyarak Cenker adına Phase 3'ü baştan sona yürütür.

---

## AGENT KİMLİĞİ VE ROLÜ

Sen, makine öğrenmesi pipeline geliştirme konusunda uzman kıdemli bir veri mühendisisin. Görevin, önceki fazların (Phase 1-2, Feza) bulgularını **önce okuyup anlamak**, ardından bu bulgulara birebir uyumlu bir veri hazırlama pipeline'ı inşa etmek ve sonunda doğrulama testleri çalıştırmaktır.

Herhangi bir adımı atlamak veya "yaklaşık" uygulamak kabul edilmez. Her karar gerekçelendirilmeli, her çıktı doğrulanmalıdır.

---

## ADIM 0 — BAŞLAMADAN ÖNCE OKU (ZORUNLU)

Kodu yazmaya başlamadan önce aşağıdaki dosyaları sırayla oku ve içselleştir:

### 0.1 Notebook Section 1-2'yi Oku
```
Dosya: notebooks/final_analysis.ipynb
Oku: Section 1 (Business Understanding) ve Section 2 (EDA) hücrelerinin tamamı
```
Şunları not et:
- Hedef değişken: `Burnout_Risk_Level` — sınıf dağılımı ve dengesizlik durumu
- Başarı metrikleri: Macro F1 > 0.80, Recall(High) > 0.85, ROC-AUC > 0.88
- En kritik hata tipi: False Negative (High riskli öğrencinin kaçırılması)
- Feza'nın handoff notunda Cenker'e ilettiği tüm encoding/imputation kararları
- Hangi değişkenlerde eksik değer var (eksik değer heatmap'ini yorumla)
- Outlier analizi sonuçları

### 0.2 Ham Veriyi İncele
```
Dosya: data/raw/ai_student_impact_dataset.csv
```
Sütunları, tipleri ve ilk 10 satırı göz gez. Aşağıdakileri teyit et:
- Toplam sütun sayısı: 16
- `Student_ID` sütununun model için feature olmayacağını
- `Burnout_Risk_Level` sütununun target olduğunu
- Kullanılabilecek feature sayısı: 14 (Student_ID ve target çıktıktan sonra)

### 0.3 GPA Kararı (KESİN — DEĞİŞTİRME)

| Sütun | Durum | Gerekçe |
|-------|-------|---------|
| `Pre_Semester_GPA` | **DROP** | Ablation'da GPA blok çıkarıldığında recall (Yüksek) +0.7pp arttı; precision düşse de iş hedefiyle (FN azalt) çelişiyor. |
| `Post_Semester_GPA` | **DROP** | Yukarıdaki + her senaryoda kırılgan (dönem başı erken uyarı senaryosuna dönülürse leakage). |
| `GNO Ortalaması`, `GNO Değişimi`, `GNO Düşüş Şiddeti` | **OLUŞTURMA** | Yukarıdakilerden türev, aynı eleme. |
| `Skill_Retention_Score` | **KULLANILACAK — izlenmeli** | Dönem içi ölçüm. Feature importance beklenmedik yüksekse Berkay'a bildir. |

### 0.4 Faz 1-2 Kesin Bulgular Özeti (Doğrulanmış — Yeniden Türetme)

Aşağıdaki tüm bilgiler `plans_and_agents/PHASE_1_2_PLAN.md`, `plans_and_agents/PHASE_1_2_AGENT.md` ve `plans_and_agents/phase_logs.md` dosyalarından derlenerek doğrulanmıştır. **Bu kararları tekrar EDA'dan çıkarmaya çalışma; direkt uygula.**

#### Hedef Değişken Dağılımı (Doğrulanmış Sayılar)
| Sınıf | Sayı | Oran |
|-------|------|------|
| Medium | 21,144 | %42.3 |
| Low | 16,369 | %32.7 |
| High | 12,487 | %25.0 |

> **Sonuç:** Anlamlı sınıf dengesizliği mevcut. High sınıfı en az temsil edilen, ancak Recall hedefi en yüksek (%85) olan sınıf. Berkay'e handoff notunda `class_weight='balanced'` önerisini belirt.

#### Eksik Değer Durumu (Heatmap'ten Doğrulanmış)
Aşağıdaki sütunlarda eksik değer tespit edilmiştir:

| Sütun | Tür | Önerilen İmputation |
|-------|-----|---------------------|
| `Pre_Semester_GPA` | Sayısal | Median |
| `Weekly_GenAI_Hours` | Sayısal | Median |
| `Tool_Diversity` | Sayısal | Median |
| `Traditional_Study_Hours` | Sayısal | Median |
| `Perceived_AI_Dependency` | Sayısal | Median |
| `Anxiety_Level_During_Exams` | Sayısal | Median |
| `Skill_Retention_Score` | Sayısal | Median |
| `Prompt_Engineering_Skill` | Kategorik | Most Frequent (Mode) |
| `Primary_Use_Case` | Kategorik | Most Frequent (Mode) |

Eksik değersiz sütunlar: `Student_ID`, `Major_Category`, `Year_of_Study`, `Paid_Subscription`, `Institutional_Policy`, `Burnout_Risk_Level`.

> **Not:** Feza'nın orijinal handoff notu bu listeyi eksik aktardı (yalnızca 4 sütun yazdı). Yukarıdaki liste missing value heatmap görselinden doğrulanmış tam listedir.

#### Korelasyon Matrisi Değerleri (Hesaplanmış)
| Çift | Korelasyon | Yön |
|------|-----------|-----|
| `Weekly_GenAI_Hours` ↔ `Anxiety_Level_During_Exams` | **0.27** | Pozitif (en güçlü) |
| `Weekly_GenAI_Hours` ↔ `Traditional_Study_Hours` | **-0.16** | Negatif |
| `Traditional_Study_Hours` ↔ `Skill_Retention_Score` | **0.15** | Pozitif |
| `Pre_Semester_GPA` ↔ `Skill_Retention_Score` | **0.10** | Pozitif (zayıf) |
| Diğer tüm çiftler | **≈ 0.00** | Nötr |

> **Sonuç:** Genel korelasyonlar çok zayıf → doğrusal olmayan modeller (Random Forest, XGBoost, LightGBM) tercih edilmeli. Study_Ratio feature engineering'i (-0.16 ilişkisini doğrudan yakalar) bu yüzden değerli.

#### Aykırı Değer Durumu (IQR Analizi Sonuçları)
Tüm sayısal sütunlarda aykırı değer oranı **%0.1'in altında** olduğu doğrulanmıştır.

> **Karar:** Agresif kırpma (clipping) veya silme gerekmez. StandardScaler yeterli. Rapor üret, aksiyon alma.

#### Encoding Kararları (Binary + WoE Revizyonu)

| Sütun | Encoding | Sıra / Not |
|-------|----------|------------|
| `Year_of_Study` | **OrdinalEncoder** | Freshman < Sophomore < Junior < Senior — sıra korunur (WoE'ye geçirme) |
| `Prompt_Engineering_Skill` | **OrdinalEncoder** | Beginner < Intermediate < Advanced — sıra korunur (WoE'ye geçirme) |
| `Major_Category` | **WoEEncoder** | Nominal → 1 sütun log-odds. OHE şişmesi yok, IV bonusu var. |
| `Primary_Use_Case` | **WoEEncoder** | Nominal — 5 kategori → 1 sütun |
| `Institutional_Policy` | **WoEEncoder** | Nominal → 1 sütun |
| `Paid_Subscription` | **astype(int)** | True→1, False→0 |

> **Neden WoE binary'de altın standart?** Kredi skorlama literatüründen
> gelir, binary target için tasarlanmıştır. Encoding log-odds skalasında
> olduğu için Logistic Regression'ın link fonksiyonuyla doğal uyum sağlar;
> ağaç modellerinde de tek bir sürekli değişken üzerinde temiz split bulur.
> Information Value (IV) yan ürünü feature selection için doğrudan eşik
> verir. Kütüphane: `category_encoders.WoEEncoder`. **Sadece train fold'unda
> fit, test'e transform** — pipeline içinde otomatik sağlanır.

> **Ordinaller neden WoE'ye gitmiyor?** Sıralı doğa zaten anlamlı bir
> monoton sinyal taşıyor (junior→senior, beginner→advanced). WoE bu sırayı
> yok eder. Eğer monotonluk varsayımı gerçekte yanlışsa modelleme aşaması
> tespit eder ve gerekirse sonra revize ederiz.

#### Scaling Kararı
Tüm sayısal sütunlar için **StandardScaler**. Aykırı değer oranı tüm
sütunlarda <%0.5 (Phase 2), bu yüzden RobustScaler şart değil. SVM/KNN/LR
gibi ölçeğe duyarlı modeller için tek ortak ölçek. WoE çıktıları doğal
olarak log-odds skalasında olsa da, pipeline'ı tek tip tutmak için onları
da StandardScaler'dan geçiririz (zarar vermez).

#### Teknik Not — Windows + Kaleido
`phase_logs.md`'den: Kaleido kütüphanesinin Windows'ta kilitlenme sorunu `kaleido>=1.3.0` sürümüyle aşıldı. Grafik kaydetme kod bloklarında her zaman `try-except` kullan, requirements.txt'te `kaleido>=1.3.0` olduğunu varsay.

---

## ADIM 1 — HÜCRE HAZIRLIĞI & ACTION LOGGER

Notebook'ta Section 3'ün altındaki tüm `# TODO (Cenker)` içeren hücreleri sırayla doldur. Yeni hücre ekleme; mevcut placeholder hücrelerini yerinde düzenle.

Her kod bloğunun hemen altına bir Markdown hücresi açarak **veri mühendisi yorumu** ekle (2-3 cümle, teknik karar gerekçesi).

### 1.1 Action Logger Kurulumu (Section 3'ün ilk hücresine ekle)

Her karar bu logger'a işlenir. Phase 3 boyunca hiçbir dönüşüm gerekçesiz yapılmaz.

```python
# ── Action Logger (classroom-agents/dataprep-expert-agent.md standardı)
dataprep_actions = []
model_handoff_report = []

def log_action(step, issue, decision, rationale, risk="Düşük"):
    dataprep_actions.append({
        "Aşama": step,
        "Sorun / Gözlem": issue,
        "Karar": decision,
        "Gerekçe": rationale,
        "Risk": risk
    })

def add_handoff(item, status, note):
    model_handoff_report.append({
        "Bileşen": item,
        "Durum": status,
        "Berkay'e Not": note
    })

print("Action Logger hazır.")
```

---

## ADIM 2 — SÜTUN TANIMLARI

Kod boyunca tutarlı kullan. Bu tanımları Section 3'ün en üst hücresine ekle:

```python
# ── Sütun Tanımları (Phase 3 — Binary + WoE Revizyonu)
TARGET = 'Burnout_Risk_Level'  # binary: 'Yüksek' vs diğer

# Drop: ID + GPA blok (recall-hostile, kırılgan)
DROP_COLS = [
    'Student_ID',
    'Pre_Semester_GPA',
    'Post_Semester_GPA',
]

# Sayısal: median imputation + StandardScaler
NUMERIC_COLS = [
    'Weekly_GenAI_Hours',
    'Tool_Diversity',
    'Traditional_Study_Hours',
    'Perceived_AI_Dependency',
    'Anxiety_Level_During_Exams',
    'Skill_Retention_Score',
]

# Kategorik nominal: mode imputation + WoEEncoder (1 sütun her biri)
WOE_COLS = [
    'Major_Category',
    'Primary_Use_Case',
    'Institutional_Policy',
]

# Kategorik ordinal: mode imputation + OrdinalEncoder (sıra korunur)
ORDINAL_COLS = ['Prompt_Engineering_Skill', 'Year_of_Study']
ORDINAL_ORDERS = [
    ['Beginner', 'Intermediate', 'Advanced'],      # Prompt_Engineering_Skill
    ['Freshman', 'Sophomore', 'Junior', 'Senior']  # Year_of_Study
]

# Boolean: direkt int'e cast (encoding gerekmez)
BOOL_COLS = ['Paid_Subscription']

print("Sütun tanımları yüklendi (Binary + WoE).")
print(f"  Sayısal  : {len(NUMERIC_COLS)} sütun")
print(f"  WoE      : {len(WOE_COLS)} sütun (nominal)")
print(f"  Ordinal  : {len(ORDINAL_COLS)} sütun (ordinal)")
print(f"  Boolean  : {len(BOOL_COLS)} sütun")
print(f"  Drop     : {DROP_COLS}")
```

---

## ADIM 3 — VERİ TEMİZLEME (Hücre 3.1)

### Eksik Değer Karar Motoru (classroom-agents standardı)

Her sütun için eksik değer oranına göre strateji belirlenir:

| Oran | Strateji |
|------|----------|
| **< %5** | Median (sayısal) / Mode (kategorik) — direkt uygula |
| **%5 – %30** | Median / Mode / KNN / IterativeImputer — veri dağılımına bak |
| **> %30** | Domain değerlendirmesi; Drop candidate; Advanced imputation |

Bu veri setinde tüm eksik değer oranları **<%5** olarak doğrulanmıştır (Phase 2 EDA) → Median/Mode direkt uygulanır.

```python
# 3.1 Veri Temizleme
df_clean = df.copy()

# 3.1.1 Gereksiz sütunları düşür
df_clean = df_clean.drop(columns=DROP_COLS)
print(f"DROP sonrası shape: {df_clean.shape}")

# 3.1.2 Duplicate satır kontrolü
n_dup = df_clean.duplicated().sum()
print(f"Duplicate satır sayısı: {n_dup}")
if n_dup > 0:
    df_clean = df_clean.drop_duplicates()
    print(f"  → {n_dup} duplicate satır silindi. Yeni shape: {df_clean.shape}")
else:
    print("  → Duplicate yok, işlem gerekmedi.")

# 3.1.3 Paid_Subscription boolean → int
df_clean['Paid_Subscription'] = df_clean['Paid_Subscription'].astype(int)
print(f"Paid_Subscription dtype: {df_clean['Paid_Subscription'].dtype}")

# 3.1.4 Eksik değer özeti (temizlik öncesi son kontrol)
missing_before = df_clean.isnull().sum()
missing_before = missing_before[missing_before > 0]
print("\nEksik değer olan sütunlar:")
display(missing_before.to_frame('Eksik Sayısı').style
    .set_properties(**{'background-color': '#111827', 'color': '#f3f4f6'})
    .bar(color='#ef4444', subset=['Eksik Sayısı']))
```

**Markdown yorumu hücresi ekle:**
```
Gereksiz sütunlar (Student_ID, Post_Semester_GPA) bırakıldı. 
Student_ID bir tanımlayıcıdır, model için anlamsız. Post_Semester_GPA ise dönem sonu verisi 
olduğundan iş problemimizin "dönem başı tahmin" hedefiyle çelişir; dahil edilmesi veri sızıntısına 
yol açar. Duplicate kontrolünde temiz çıkan veri seti, pipeline adımlarına hazırdır.
```

```python
# Action logger — temizleme kararları
log_action("3.1 Drop", "Student_ID identifier sütun", "Drop edildi", "Model için anlamsız", "Düşük")
log_action("3.1 Drop", "Post_Semester_GPA dönem sonu verisi", "Drop edildi", "Veri sızıntısı riski — dönem başı tahmin hedefiyle çelişir", "Yüksek")
log_action("3.1 Duplicate", f"{n_dup} duplicate satır", "Silindi" if n_dup > 0 else "İşlem gerekmedi", "Duplicate satırlar modeli biaslar", "Düşük")
log_action("3.1 Bool Cast", "Paid_Subscription bool", "int'e çevrildi", "Pipeline'da sayısal olarak işlenmeli", "Düşük")
```

---

## ADIM 3.5 — DAĞILIM & ÇARPIKLIK ANALİZİ (Hücre 3.1b)

> classroom-agents standardı: `|skew| > 1` → Log / Yeo-Johnson / Box-Cox dönüşümü değerlendir.

```python
# 3.1b Sayısal Değişken Çarpıklık (Skewness) Analizi
from scipy import stats

skew_report = []
for col in NUMERIC_COLS:
    if col in df_clean.columns:
        skew_val = df_clean[col].skew()
        q1, q3 = df_clean[col].quantile(0.25), df_clean[col].quantile(0.75)
        iqr = q3 - q1
        outlier_rate = ((df_clean[col] < q1 - 1.5*iqr) | (df_clean[col] > q3 + 1.5*iqr)).mean() * 100
        karar = "Log/YeoJohnson dönüşümü değerlendir" if abs(skew_val) > 1 else "StandardScaler yeterli"
        outlier_aksiyon = "Winsorization / RobustScaler düşün" if outlier_rate > 5 else "Müdahale gerekmez"
        skew_report.append({
            'Sütun': col,
            'Skewness': round(skew_val, 3),
            'Outlier Oranı (%)': round(outlier_rate, 3),
            'Dağılım Kararı': karar,
            'Outlier Kararı': outlier_aksiyon
        })
        log_action("3.1b Skewness", col,
                   karar, f"skew={skew_val:.3f}, outlier_rate=%{outlier_rate:.2f}",
                   "Orta" if abs(skew_val) > 1 else "Düşük")

skew_df = pd.DataFrame(skew_report)
display(skew_df.style.set_properties(**{'background-color': '#111827', 'color': '#f3f4f6'})
        .applymap(lambda v: 'color: #ef4444' if 'değerlendir' in str(v) else '', subset=['Dağılım Kararı']))
print(f"\nKritik çarpıklık (|skew|>1) sayısı: {(skew_df['Skewness'].abs() > 1).sum()}")
```

**Markdown yorumu:**
```
Bu veri setinde EDA'dan bilinen üzere outlier oranları %0.1'in altındadır ve StandardScaler 
yeterlidir. Bu hücre bir güvenlik kontrolü — eğer Feza'nın PR sonrası veri yapısı değiştiyse 
otomatik olarak uyarı üretir. |skew|>1 olan sütun çıkarsa log/Yeo-Johnson dönüşümü tartışılır.
```

---

## ADIM 4 — FEATURE ENGINEERING (Hücre 3.2)

> **Prensip (KRİTİK):** Yüksek-MI değişkenlerden ratio/product türetmek
> **YASAK** — kolinearite üretir, tree modeller bu nonlineer dönüşümleri
> zaten kendileri keşfeder. Engineering sadece **tek başına zayıf, mantıken
> birleşince güçlenen** sinerji adayları için kabul edilir.
>
> **Sinerji eşiği:** `MI(combined) − max(MI(A), MI(B)) > 0.01`. Bu eşiği
> geçemeyen aday otomatik elenir.

### 4.1 Önceden Elenmiş Engineered Feature'lar (UYGULAMA)

Aşağıdaki 7 feature **prensip ihlali** olduğu için artık üretilmez:

| Eski Feature | Neden Elendi |
|--------------|--------------|
| `AI Bağımlılık Yükü` (Haftalık AI × Bağımlılık) | İki yüksek-MI'ın çarpımı, kolineer |
| `Sınav Kaygısı × AI Saati` | İki yüksek-MI'ın çarpımı |
| `AI × Kaygı × Bağımlılık` | Üç yüksek-MI triple, ağır kolineer |
| `AI Çalışma Payı` (oran) | Yüksek + orta MI oranı, sinyal duplikasyonu |
| `Kalıcılık / AI Saati` | İki yüksek-MI oranı |
| `AI × Düşük Kalıcılık` | İki yüksek-MI'ın çarpımı |
| `Bağımlılık / Kalıcılık` | İki yüksek-MI oranı |

### 4.2 Korunan Feature

```python
# 3.2 Feature Engineering — Sadeleştirilmiş
df_fe = df_clean.copy()

# KORUNAN: Toplam Çalışma Yükü
# Gerekçe: AI + Geleneksel toplamı, çarpım/oran değil. Kapasite/yük
# kavramı ayrı bir konsept — bileşenlerin tekil sinyallerinden ayrışır.
df_fe['Toplam Çalışma Yükü'] = (
    df_fe['Weekly_GenAI_Hours'] + df_fe['Traditional_Study_Hours']
)
```

### 4.3 Sinerji Adayları (TEST → IV/MI Eşiği)

Aşağıdaki 4 aday **test edilir**, eşiği geçen kalır:

| Aday | Mantıksal gerekçe | Birleşim |
|------|-------------------|----------|
| `Prompt × Araç Çeşitliliği` | İkisi de tek başına orta-düşük MI; birlikte "AI power user" profili | `Prompt_Skill_Ordinal * Tool_Diversity` (sayısal) veya bin'li kombinasyon |
| `Bölüm × Kullanım Amacı` | CS+Debug ≠ Sosyal+Summarizing; etkileşim profili | Joint category string → WoE |
| `Yıl × Politika` | Junior+kısıtlayıcı ≠ Senior+serbest | Joint category string → WoE |
| `Abonelik × Prompt` | Ücretli+Advanced = ciddi kullanıcı; Ücretli+Beginner = çaresiz | Joint category string → WoE |

```python
# 3.2b Sinerji Adayları — MI testleri
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder

def mi_1d(col_series, y):
    """Tek sütun için MI hesapla."""
    le = LabelEncoder()
    x_enc = le.fit_transform(col_series.astype(str))
    return float(mutual_info_classif(x_enc.reshape(-1, 1), y, random_state=42)[0])

y_bin = (df_clean[TARGET] == 'Yüksek').astype(int).values

# Aday 1: Prompt × Araç Çeşitliliği (sayısal etkileşim)
prompt_ord = pd.Categorical(
    df_clean['Prompt_Engineering_Skill'],
    categories=['Beginner', 'Intermediate', 'Advanced'], ordered=True
).codes
combo_1 = (prompt_ord + 1) * df_clean['Tool_Diversity'].fillna(df_clean['Tool_Diversity'].median())
mi_a = mi_1d(df_clean['Prompt_Engineering_Skill'].fillna('Intermediate'), y_bin)
mi_b = mi_1d(df_clean['Tool_Diversity'].fillna(df_clean['Tool_Diversity'].median()), y_bin)
mi_combo = mi_1d(combo_1, y_bin)
synergy_1 = mi_combo - max(mi_a, mi_b)
print(f"Prompt × Tool: MI_combo={mi_combo:.4f}, sinerji={synergy_1:+.4f}")

# Aday 2-4: Joint category (string concat)
candidates_str = {
    'Bölüm × Amaç' : ('Major_Category', 'Primary_Use_Case'),
    'Yıl × Politika': ('Year_of_Study', 'Institutional_Policy'),
    'Abonelik × Prompt': ('Paid_Subscription', 'Prompt_Engineering_Skill'),
}
synergy_results = [('Prompt × Tool', mi_a, mi_b, mi_combo, synergy_1)]
for name, (a, b) in candidates_str.items():
    sa = df_clean[a].astype(str).fillna('NA')
    sb = df_clean[b].astype(str).fillna('NA')
    joint = sa + '||' + sb
    mi_a2 = mi_1d(sa, y_bin)
    mi_b2 = mi_1d(sb, y_bin)
    mi_j  = mi_1d(joint, y_bin)
    syn   = mi_j - max(mi_a2, mi_b2)
    synergy_results.append((name, mi_a2, mi_b2, mi_j, syn))

syn_df = pd.DataFrame(synergy_results, columns=['Aday', 'MI(A)', 'MI(B)', 'MI(birleşik)', 'Sinerji'])
syn_df['Karar'] = syn_df['Sinerji'].apply(lambda s: '✅ TUT' if s > 0.01 else '❌ ELE')
display(syn_df.style
    .format({'MI(A)': '{:.4f}', 'MI(B)': '{:.4f}', 'MI(birleşik)': '{:.4f}', 'Sinerji': '{:+.4f}'})
    .set_properties(**{'background-color': '#111827', 'color': '#f3f4f6'}))

# Eşiği geçenleri df_fe'ye ekle
accepted = syn_df[syn_df['Sinerji'] > 0.01]['Aday'].tolist()
print(f"\nKabul edilen sinerji feature'ları: {accepted}")
# (Kabul edilenleri df_fe'ye uygulama kodu — yalnızca 'accepted' listesindekiler eklenir)
```

**Markdown yorumu:**
```
Engineering tarafında sadece sum (Toplam Çalışma Yükü) korundu — bileşenlerinden
ayrı bir kapasite kavramı taşıyor. 4 sinerji adayı MI testi ile değerlendirildi;
sadece MI(birleşik) − max(MI(A), MI(B)) > 0.01 eşiğini geçen feature'lar eklendi.
Bu eşik literatürdeki "interaction effect" testlerinin pratik karşılığıdır:
parçaların toplamından kayda değer fazla bilgi geliyor mu?
```

```python
# Action logger
log_action("4.1 FE Cleanup", "7 redundant feature", "DROP", "Yüksek-MI değişkenlerden türev, kolinearite", "Düşük")
log_action("4.2 FE Keep", "Toplam Çalışma Yükü", "Korundu", "Sum, farklı kavram (kapasite)", "Düşük")
log_action("4.3 FE Test", f"{len(syn_df)} sinerji adayı", f"{len(accepted)} kabul", "MI sinerji eşiği +0.01", "Düşük")
log_action("4. FE", "GPA türev feature'ları", "REDDEDİLDİ", "GPA blok ablation'da recall'u düşürdü, drop edildi", "Yüksek")
```

---

## ADIM 4.5 — FEATURE KALİTE + IV FİLTRESİ (Hücre 3.2c)

> Her yeni feature için null inflation, leakage, redundancy, stability +
> **IV (Information Value)** kontrolü yapılır. IV binary classifier'da
> doğrudan feature gücü göstergesidir; literatür eşikleri:
>
> | IV | Karar |
> |----|-------|
> | < 0.02 | Faydasız — otomatik drop |
> | 0.02 – 0.10 | Zayıf — ablation'a kalsın |
> | 0.10 – 0.30 | Orta — tut |
> | 0.30 – 0.50 | Güçlü — tut |
> | > 0.50 | **Şüpheli — leakage check** |
>
> IV hesabı: `Σ (P(x|y=1) − P(x|y=0)) × WoE(x)` her bin için, toplamı feature IV'sidir.
> Kütüphane: `category_encoders.WoEEncoder` fit sonrası `.feature_names_` üzerinden
> manuel hesap veya `optbinning.BinningProcess` doğrudan IV verir.

```python
# 3.2b Feature Quality Check (classroom-agents Phase 6 standardı)
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder

print("=" * 50)
print("FEATURE KALİTE & LEAKAGE AUDIT")
print("=" * 50)

quality_checks = []

for col in ['Study_Ratio', 'Total_Study_Hours', 'High_AI_User']:
    null_rate = df_fe[col].isnull().mean() * 100
    # Target ile korelasyon (LabelEncoded target)
    le = LabelEncoder()
    y_enc = le.fit_transform(df_fe[TARGET].fillna('Unknown'))
    corr = abs(df_fe[col].corr(pd.Series(y_enc, index=df_fe.index)))
    # Leakage şüphesi: target ile korelasyon > 0.9
    leakage_flag = "⚠️ YÜK SEK — İNCELE" if corr > 0.9 else "Temiz"
    quality_checks.append({
        'Feature': col,
        'Null Oranı (%)': round(null_rate, 3),
        'Target Korelasyonu': round(corr, 3),
        'Leakage Durumu': leakage_flag
    })

quality_df = pd.DataFrame(quality_checks)
display(quality_df.style.set_properties(**{'background-color': '#111827', 'color': '#f3f4f6'}))

# VIF — Multicollinearity Kontrolü (sayısal feature'lar için)
try:
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    vif_cols = [c for c in NUMERIC_COLS_FE if c in df_fe.columns and df_fe[c].notna().all()]
    vif_data = df_fe[vif_cols].dropna()
    vif_vals = [variance_inflation_factor(vif_data.values, i) for i in range(len(vif_cols))]
    vif_df = pd.DataFrame({'Feature': vif_cols, 'VIF': [round(v, 2) for v in vif_vals]})
    vif_df['Durum'] = vif_df['VIF'].apply(lambda v: "⚠️ Yüksek Multicollinearity" if v > 10 else ("Orta" if v > 5 else "Temiz"))
    print("\nVIF Analizi:")
    display(vif_df.style.set_properties(**{'background-color': '#111827', 'color': '#f3f4f6'}))
    high_vif = vif_df[vif_df['VIF'] > 10]['Feature'].tolist()
    if high_vif:
        print(f"⚠️ VIF > 10 sütunlar: {high_vif} — Berkay'e bildir")
        log_action("4.5 VIF", str(high_vif), "Berkay'e bildirildi", "VIF>10 multicollinearity riski", "Orta")
    else:
        print("VIF kontrolü temiz — multicollinearity sorunu yok.")
        log_action("4.5 VIF", "Tüm sayısal feature'lar", "Temiz", "VIF<10, multicollinearity yok", "Düşük")
except ImportError:
    print("statsmodels kurulu değil — VIF atlandı. pip install statsmodels")

# Mutual Information — Feature önem sinyali (Berkay'e preview)
try:
    le2 = LabelEncoder()
    y_mi = le2.fit_transform(df_fe[TARGET])
    X_mi = df_fe[NUMERIC_COLS_FE].fillna(df_fe[NUMERIC_COLS_FE].median())
    mi_scores = mutual_info_classif(X_mi, y_mi, random_state=42)
    mi_df = pd.DataFrame({'Feature': NUMERIC_COLS_FE, 'MI Score': mi_scores}).sort_values('MI Score', ascending=False)
    print("\nMutual Information Skoru (Berkay için önizleme):")
    display(mi_df.style.set_properties(**{'background-color': '#111827', 'color': '#f3f4f6'})
            .bar(color='#a78bfa', subset=['MI Score']))
    add_handoff("Feature Importance Preview (MI)", "Hazır",
                f"En güçlü sinyal: {mi_df.iloc[0]['Feature']} (MI={mi_df.iloc[0]['MI Score']:.3f})")
except Exception as e:
    print(f"MI skoru hesaplanamadı: {e}")
```

**Markdown yorumu:**
```
VIF analizi sayısal feature'lar arasındaki çoklu doğrusallığı ölçer; VIF>10 olan değişkenler 
özellikle doğrusal modellerde (LR, SVM) baskı yaratır. Mutual Information skoru ise feature'ların 
target ile non-linear ilişki gücünü gösterir ve Berkay'e hangi feature'ların model için güçlü 
sinyal taşıdığını önceden bildirir.
```

---

## ADIM 5 — PIPELINE KURULUMU (Hücre 3.3) — Binary + WoE

```python
# 3.3 Sklearn Pipeline Kurulumu — Binary + WoE Revizyonu
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from category_encoders import WoEEncoder  # pip install category_encoders

# Sayısal pipeline: medyan imputation + StandardScaler
numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',  StandardScaler()),
])

# WoE pipeline (nominal kategorikler): mode imputation + WoEEncoder
# KRİTİK: WoEEncoder.fit(X, y) çağrıldığında y (binary) bekler.
# ColumnTransformer otomatik y'yi geçer; fit yalnızca train fold'unda
# çalıştığı sürece sızıntı olmaz.
woe_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', WoEEncoder(handle_unknown='value', handle_missing='value')),
])

# Ordinal pipeline (hiyerarşi korunur — WoE'ye geçirilmez)
ordinal_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OrdinalEncoder(
        categories=ORDINAL_ORDERS,
        handle_unknown='use_encoded_value',
        unknown_value=-1,
    )),
    ('scaler', StandardScaler()),  # sayısallarla aynı ölçeğe çek
])

# ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num',  numeric_pipeline, NUMERIC_COLS),  # + sinerji testinden geçen FE'ler
        ('woe',  woe_pipeline,     WOE_COLS),
        ('ord',  ordinal_pipeline, ORDINAL_COLS),
        ('bool', 'passthrough',    BOOL_COLS),
    ],
    remainder='drop',
)

print("Pipeline tanımlandı (Binary + WoE).")
print(f"  Sayısal transformer  : {len(NUMERIC_COLS)} sütun (+ FE)")
print(f"  WoE transformer      : {len(WOE_COLS)} sütun")
print(f"  Ordinal transformer  : {len(ORDINAL_COLS)} sütun")
print(f"  Boolean passthrough  : {len(BOOL_COLS)} sütun")
print(f"  Beklenen toplam      : {len(NUMERIC_COLS) + len(WOE_COLS) + len(ORDINAL_COLS) + len(BOOL_COLS)} sütun (+ FE)")
```

**Markdown yorumu:**
```
Pipeline binary classifier için WoE geçişine uyarlandı: nominal kategorikler
(Major_Category, Primary_Use_Case, Institutional_Policy) WoEEncoder ile tek
sütuna sıkıştırılarak hem boyut hem IV-tabanlı feature selection avantajı
kazanıldı. Ordinal sütunlar (Prompt_Engineering_Skill, Year_of_Study) sıra
bilgisini korumak için OrdinalEncoder'da bırakıldı. fit() yalnızca train
setinde çağrılır → WoE log-odds'ları train hedefinden öğrenilir, test
sızıntısı yok.
```

> **Kütüphane gereksinimi:** `requirements.txt`'e `category_encoders>=2.6.0`
> eklenmeli. Kurulum: `pip install category_encoders`.

---

## ADIM 6 — STRATİFİED TRAIN-TEST SPLIT (Hücre 3.4)

```python
# 3.4 Stratified Train-Test Split
from sklearn.model_selection import train_test_split

X = df_fe.drop(columns=[TARGET])
y = df_fe[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print(f"Train set : {X_train.shape[0]:,} satır  |  Test set : {X_test.shape[0]:,} satır")
print(f"\nHedef dağılımı (train):\n{y_train.value_counts(normalize=True).round(3)}")
print(f"\nHedef dağılımı (test):\n{y_test.value_counts(normalize=True).round(3)}")

# Stratified kontrolü — train ve test oranları birbirine yakın mı?
train_dist = y_train.value_counts(normalize=True)
test_dist  = y_test.value_counts(normalize=True)
max_diff = (train_dist - test_dist).abs().max()
assert max_diff < 0.01, f"Stratification başarısız! Max fark: {max_diff:.4f}"
print(f"\nStratification doğrulandı (max sınıf oranı farkı: {max_diff:.4f} < 0.01) ✓")
```

**Markdown yorumu:**
```
Stratified split, her sınıfın (Low/Medium/High) aynı oranla hem train hem test setinde 
temsil edilmesini garanti eder. Bu özellikle High sınıfının yalnızca %25 oranında bulunduğu 
veri setimizde kritiktir; rastgele split'te High sınıfı test setinden dışarıda kalabilirdi.
```

```python
# 3.5b Sınıf Dengesizliği Karar Motoru (classroom-agents standardı)
# Split sonrası train üzerinde çalışır — leakage yok
dominant_ratio = y_train.value_counts(normalize=True).max() * 100
minority_class = y_train.value_counts().idxmin()
minority_ratio = y_train.value_counts(normalize=True).min() * 100

print(f"\nDominant sınıf oranı: %{dominant_ratio:.1f}")
print(f"Azınlık sınıfı: '{minority_class}' — %{minority_ratio:.1f}")

if dominant_ratio > 85:
    imbalance_decision = "SMOTE veya class_weight='balanced' güçlü aday"
    imbalance_risk = "Yüksek"
elif dominant_ratio > 70:
    imbalance_decision = "class_weight='balanced' veya hafif SMOTE değerlendir"
    imbalance_risk = "Orta"
else:
    imbalance_decision = "Doğrudan müdahale gerekmeyebilir — class_weight='balanced' ekle"
    imbalance_risk = "Düşük"

print(f"Karar: {imbalance_decision}")
log_action("3.4b Imbalance", f"Dominant=%{dominant_ratio:.1f}, Azınlık='{minority_class}' %{minority_ratio:.1f}",
           imbalance_decision, "Dominant ratio karar motoruna göre belirlendi", imbalance_risk)
add_handoff("Sınıf Dengesizliği", f"Dominant=%{dominant_ratio:.1f}",
            f"{imbalance_decision}. Recall(High)>0.85 hedefi için kritik.")
```

---

## ADIM 7 — PIPELINE FİT & TRANSFORM (Hücre 3.5)

```python
# 3.5 Pipeline Fit & Transform
# KRİTİK: fit() SADECE train setinde çağrılır.
# Test seti sadece transform() ile dönüştürülür.

preprocessor.fit(X_train)   # ← SADECE TRAIN

X_train_proc = preprocessor.transform(X_train)
X_test_proc  = preprocessor.transform(X_test)

print(f"İşlenmiş train shape : {X_train_proc.shape}")
print(f"İşlenmiş test shape  : {X_test_proc.shape}")

# Feature isimlerini çıkar (model açıklanabilirliği için)
try:
    feature_names = preprocessor.get_feature_names_out()
    print(f"\nToplam feature sayısı: {len(feature_names)}")
    print("İlk 10 feature:", list(feature_names[:10]))
except Exception as e:
    print(f"Feature isimleri alınamadı: {e}")

# Eksik değer kalmadı mı?
assert not pd.isnull(X_train_proc).any(), "Train setinde hâlâ eksik değer var!"
assert not pd.isnull(X_test_proc).any(), "Test setinde hâlâ eksik değer var!"
print("\nEksik değer kontrolü geçti ✓")
```

**Markdown yorumu:**
```
preprocessor.fit() yalnızca X_train üzerinde çağrıldı. Bu, imputer'ların medyanını, scaler'ların 
ortalama/std'sini ve encoder'ların kategori setlerini yalnızca train verisinden öğrenmesini 
sağlar. Test setine transform uygulamak bu öğrenilmiş parametreleri kullanır — böylece test 
verisi hiçbir şekilde eğitime sızmaz (data leakage yok).
```

---

## ADIM 8 — VERİLERİ KAYDET (Hücre 3.6)

```python
# 3.6 İşlenmiş Verileri Kaydet
import os

os.makedirs(DATA_PROCESSED, exist_ok=True)

# Feature isimlerini al (varsa)
try:
    feat_names = preprocessor.get_feature_names_out()
except:
    feat_names = [f'feature_{i}' for i in range(X_train_proc.shape[1])]

# DataFrame'e çevir ve kaydet
X_train_df = pd.DataFrame(X_train_proc, columns=feat_names)
X_test_df  = pd.DataFrame(X_test_proc,  columns=feat_names)
y_train_df = y_train.reset_index(drop=True)
y_test_df  = y_test.reset_index(drop=True)

X_train_df.to_csv(DATA_PROCESSED + 'X_train.csv', index=False)
X_test_df.to_csv(DATA_PROCESSED  + 'X_test.csv',  index=False)
y_train_df.to_csv(DATA_PROCESSED + 'y_train.csv', index=False)
y_test_df.to_csv(DATA_PROCESSED  + 'y_test.csv',  index=False)

print("İşlenmiş veriler kaydedildi:")
print(f"  X_train.csv  → {X_train_df.shape}")
print(f"  X_test.csv   → {X_test_df.shape}")
print(f"  y_train.csv  → {y_train_df.shape}")
print(f"  y_test.csv   → {y_test_df.shape}")
```

---

## ADIM 9 — PİPELINE'I KAYDET (Hücre 3.7)

```python
# 3.7 Pipeline Kaydet
import joblib
import os

os.makedirs(MODELS_DIR, exist_ok=True)
joblib.dump(preprocessor, MODELS_DIR + 'pipeline.joblib')
print(f"Pipeline kaydedildi: {MODELS_DIR}pipeline.joblib")

# Yeniden yükleyerek kontrol et
pipeline_loaded = joblib.load(MODELS_DIR + 'pipeline.joblib')
test_output = pipeline_loaded.transform(X_test.iloc[:3])
assert test_output.shape[1] == X_test_proc.shape[1], "Yüklenen pipeline çıktısı uyumsuz!"
print("Pipeline yüklenme ve çıktı testi geçti ✓")
```

---

## ADIM 10 — PİPELINE GÖRSELLEŞTİRMESİ (Hücre 3.8)

```python
# 3.8 Pipeline Adımları Görsel Şeması
steps_data = [
    {'Adım': '1. Drop', 'Kapsam': 'Student_ID, Pre/Post_Semester_GPA', 'Yöntem': 'df.drop', 'Gerekçe': 'ID + GPA blok (recall-hostile)'},
    {'Adım': '2. FE Keep', 'Kapsam': 'Toplam Çalışma Yükü', 'Yöntem': 'AI + Geleneksel sum', 'Gerekçe': 'Kapasite kavramı (ratio değil)'},
    {'Adım': '2b. FE Sinerji', 'Kapsam': 'MI eşiği +0.01 geçen adaylar', 'Yöntem': 'MI(birleşik) − max(MI(A),MI(B))', 'Gerekçe': 'Sinerji testi'},
    {'Adım': '3. Split', 'Kapsam': 'Tüm veri', 'Yöntem': 'StratifiedShuffleSplit (80/20, seed=42)', 'Gerekçe': 'Binary sınıf dengesi'},
    {'Adım': '4a. Num. Impute', 'Kapsam': ', '.join(NUMERIC_COLS), 'Yöntem': 'SimpleImputer(median)', 'Gerekçe': '<%5 eksik → median yeterli'},
    {'Adım': '4b. Scaling', 'Kapsam': 'Tüm sayısal + ordinal', 'Yöntem': 'StandardScaler', 'Gerekçe': 'SVM/KNN/LR uyumu, aykırı düşük'},
    {'Adım': '5a. WoE Impute', 'Kapsam': ', '.join(WOE_COLS), 'Yöntem': 'SimpleImputer(most_frequent)', 'Gerekçe': 'Mod ile doldur'},
    {'Adım': '5b. WoE Encode', 'Kapsam': ', '.join(WOE_COLS), 'Yöntem': 'WoEEncoder (binary log-odds)', 'Gerekçe': 'Nominal → 1 sütun, IV bonus'},
    {'Adım': '6a. Ord. Impute', 'Kapsam': ', '.join(ORDINAL_COLS), 'Yöntem': 'SimpleImputer(most_frequent)', 'Gerekçe': 'Mod ile doldur'},
    {'Adım': '6b. Ord. Encode', 'Kapsam': ', '.join(ORDINAL_COLS), 'Yöntem': 'OrdinalEncoder(sıralı)', 'Gerekçe': 'Hiyerarşi korunur (WoE değil)'},
    {'Adım': '7. Bool Cast', 'Kapsam': 'Paid_Subscription', 'Yöntem': 'astype(int)', 'Gerekçe': 'Zaten 0/1'},
    {'Adım': '8. IV Filtre', 'Kapsam': 'Tüm feature\'lar (WoE sonrası)', 'Yöntem': 'IV<0.02 drop, IV>0.5 leakage check', 'Gerekçe': 'Binary feature selection prefilter'},
]

pipeline_df = pd.DataFrame(steps_data)

styled_pipeline = (
    pipeline_df.style
    .set_properties(**{'background-color': '#111827', 'color': '#f3f4f6', 'border': '1px solid #374151'})
    .set_table_styles([
        {'selector': 'th', 'props': [('background-color', '#1f2937'), ('color', '#4ade80'), ('font-weight', 'bold'), ('padding', '8px 14px')]},
        {'selector': 'td', 'props': [('padding', '7px 14px')]},
    ])
    .apply(lambda x: ['background-color: #052e16' if i % 2 == 0 else 'background-color: #111827'
                      for i in range(len(x))], axis=0)
)

display(styled_pipeline)
```

**Markdown yorumu:**
```
Pipeline şeması, veri hazırlama sürecinin her adımını, kapsamını ve gerekçesini bir arada 
göstermektedir. Bu tablo Berkay'e devir aşamasında referans belgesi görevi görür: hangi 
feature'ların hangi dönüşümden geçtiği, train-only fit garantisi ve leakage kararları buradan 
izlenebilir.
```

---

## ADIM 11 — AYKIRI DEĞER STRATEJİSİ (Hücre 3.9)

```python
# 3.9 Aykırı Değer Stratejisi
# EDA bulgusu: Aykırı değer oranı tüm sayısal sütunlarda < %0.5
# Karar: Pipeline'da StandardScaler kullanıldığından ağır kırpma yapmıyoruz.
# Ancak Train setinde IQR sınır dışına çıkan gözlemleri raporla.

outlier_report = []
for col in NUMERIC_COLS:
    if col not in X_train.columns:
        continue
    q1 = X_train[col].quantile(0.25)
    q3 = X_train[col].quantile(0.75)
    iqr = q3 - q1
    n_out = ((X_train[col] < q1 - 1.5*iqr) | (X_train[col] > q3 + 1.5*iqr)).sum()
    outlier_report.append({'Sütun': col, 'Aykırı Sayısı (Train)': n_out,
                           'Oran (%)': round(n_out/len(X_train)*100, 3)})

outlier_rep_df = pd.DataFrame(outlier_report)
display(outlier_rep_df.style.set_properties(**{'background-color': '#111827', 'color': '#f3f4f6'}))
print("Aykırı değer oranları %0.5'in altında → kırpma uygulanmadı.")
```

---

## ADIM 12 — BERKAY'E HANDOFF (Hücre 3.10)

Bu adım iki parçadan oluşur: önce action log tablosu çıktısı (kod hücresi), ardından yapısal handoff raporu (Markdown hücresi).

### 12a — Action Log Çıktısı (Kod Hücresi)

```python
# 3.10a Tüm DataPrep Kararları — Action Log
print("=" * 60)
print("PHASE 3 ACTION LOG")
print("=" * 60)
actions_df = pd.DataFrame(dataprep_actions)
display(actions_df.style
    .set_properties(**{'background-color': '#111827', 'color': '#f3f4f6', 'border': '1px solid #374151'})
    .set_table_styles([
        {'selector': 'th', 'props': [('background-color', '#1f2937'), ('color', '#4ade80'), ('font-weight', 'bold'), ('padding', '8px 14px')]},
    ])
    .applymap(lambda v: 'color: #ef4444; font-weight: bold' if 'Yüksek' in str(v) else
                        ('color: #f59e0b' if 'Orta' in str(v) else ''), subset=['Risk'])
)
print(f"\nToplam karar sayısı: {len(dataprep_actions)}")
high_risk = [a for a in dataprep_actions if a['Risk'] == 'Yüksek']
if high_risk:
    print(f"⚠️ Yüksek riskli karar sayısı: {len(high_risk)}")
```

### 12b — Yapısal Handoff Raporu (Markdown Hücresi)

Notebook'a şu Markdown hücresini ekle:

```markdown
### 3.10b Model Expert Handoff Raporu (Berkay için)

> classroom-agents/dataprep-expert-agent.md MODEL EXPERT HANDOFF FORMAT standardına uygundur.

---

**Veri Durumu:** Temiz · **Target:** Binary (`Yüksek` vs diğer, ~%25 pozitif)

**Missing Value Strategy:**
Tüm eksik oranları <%5 → basit imputation yeterli. Sayısal sütunlar (Weekly_GenAI_Hours,
Tool_Diversity, Traditional_Study_Hours, Perceived_AI_Dependency, Anxiety_Level_During_Exams,
Skill_Retention_Score) → Median. Kategorikler → Mode. Pipeline içinde train-only fit.

**Encoding Strategy (Binary + WoE):**
- **WoEEncoder** (nominal → 1 sütun log-odds): Major_Category, Primary_Use_Case, Institutional_Policy
- **OrdinalEncoder** (sıra korunur): Prompt_Engineering_Skill (Beginner→Advanced), Year_of_Study (Freshman→Senior)
- **astype(int)**: Paid_Subscription

**Scaling Strategy:**
StandardScaler — tüm sayısal + ordinal + WoE çıktıları. Aykırı oranı <%0.5 olduğu için
RobustScaler şart değil. SVM/KNN/LR direkt kullanılabilir.

**Imbalance Strategy:**
Binary target ~%25 pozitif. `class_weight='balanced'` veya `scale_pos_weight=3` öner.
SMOTE yalnızca train CV loop içinde uygula — asla split öncesi.

**Feature Engineering:**
- `Toplam Çalışma Yükü` = Weekly_GenAI_Hours + Traditional_Study_Hours (sum, kapasite kavramı)
- Sinerji adaylarından MI eşiğini (+0.01) geçenler: [test sonucu listele]
- 7 eski redundant feature (AI Bağımlılık Yükü, Sınav Kaygısı × AI Saati, vb.) → DROP (kolinearite)
- Tüm GPA feature'ları (Pre/Post/Ortalama/Değişim) → DROP (recall-hostile, ablation kanıtı)

**Feature Selection:**
- IV filtresi: IV<0.02 otomatik drop, IV>0.5 leakage check yapıldı
- Compact Core (eski 21 feature) yerine **Lean WoE Core** üretildi
- Ablation karşılaştırması Berkay'a bırakıldı (eski Compact Core vs yeni Lean Core)

**Leakage Status:** Temiz
- Pre/Post_Semester_GPA + Student_ID drop edildi
- WoE fit yalnızca train'de — log-odds train hedefinden öğrenildi
- Pipeline sadece train'de fit, test sadece transform

**Önerilen Model Türleri:**
Tree-based & Ensemble öncelikli (RF, XGBoost, LightGBM) — korelasyonlar çok zayıf, non-linear.
Distance-based (SVM, KNN) ikincil — veri ölçekli, kullanılabilir.
Linear (LR) üçüncül — baseline için.

**Kritik Uyarılar:**
1. Recall(High) > 0.85 hedefi — class_weight='balanced' veya threshold optimizasyonu şart.
2. VIF kontrolü geçti — multicollinearity sorunu yok.
3. Skill_Retention_Score'un feature importance'ı izle; beklenmedik yüksek önem = leakage sinyali.
4. X_train.csv'de OHE sonrası sütun sayısı ham veriden fazla — feature_names_in_ ile kontrol et.

**Yükleme:**
```python
import joblib, pandas as pd
preprocessor = joblib.load('../models/pipeline.joblib')
X_train = pd.read_csv('../data/processed/X_train.csv')
X_test  = pd.read_csv('../data/processed/X_test.csv')
y_train = pd.read_csv('../data/processed/y_train.csv').squeeze()
y_test  = pd.read_csv('../data/processed/y_test.csv').squeeze()
```
```

---

## ADIM 13 — DOĞRULAMA TESTLERİ (Hücre 3.11 — ZORUNLU)

Bu hücreyi çalıştırmadan Phase 3 tamamlanmış sayılmaz. Her assert geçmeli.

```python
# ══════════════════════════════════════════════════════
# PHASE 3 DOĞRULAMA TESTLERİ — TÜM SATIRLAR GEÇMELİ
# ══════════════════════════════════════════════════════
import os, joblib, pandas as pd, numpy as np

print("=" * 55)
print("PHASE 3 DOĞRULAMA TESTLERİ BAŞLIYOR")
print("=" * 55)

errors = []

# TEST 1: Dosya varlığı
for fname in ['X_train.csv', 'X_test.csv', 'y_train.csv', 'y_test.csv']:
    path = DATA_PROCESSED + fname
    if not os.path.exists(path):
        errors.append(f"FAIL — {fname} bulunamadı")
    else:
        print(f"[OK] {fname} mevcut")

if not os.path.exists(MODELS_DIR + 'pipeline.joblib'):
    errors.append("FAIL — pipeline.joblib bulunamadı")
else:
    print("[OK] pipeline.joblib mevcut")

# TEST 2: Shape tutarlılığı
Xtr = pd.read_csv(DATA_PROCESSED + 'X_train.csv')
Xte = pd.read_csv(DATA_PROCESSED + 'X_test.csv')
ytr = pd.read_csv(DATA_PROCESSED + 'y_train.csv').squeeze()
yte = pd.read_csv(DATA_PROCESSED + 'y_test.csv').squeeze()

if Xtr.shape[0] != len(ytr):
    errors.append(f"FAIL — X_train satır sayısı ({Xtr.shape[0]}) y_train'le uyumsuz ({len(ytr)})")
else:
    print(f"[OK] X_train shape: {Xtr.shape}, y_train: {len(ytr)}")

if Xte.shape[0] != len(yte):
    errors.append(f"FAIL — X_test satır sayısı ({Xte.shape[0]}) y_test'le uyumsuz ({len(yte)})")
else:
    print(f"[OK] X_test shape: {Xte.shape}, y_test: {len(yte)}")

if Xtr.shape[1] != Xte.shape[1]:
    errors.append(f"FAIL — X_train ({Xtr.shape[1]}) ve X_test ({Xte.shape[1]}) sütun sayısı farklı!")
else:
    print(f"[OK] Train/test sütun sayısı eşit: {Xtr.shape[1]}")

# TEST 3: 80/20 oranı
total = len(ytr) + len(yte)
train_ratio = len(ytr) / total
if abs(train_ratio - 0.8) > 0.01:
    errors.append(f"FAIL — Train oranı 0.8 değil: {train_ratio:.3f}")
else:
    print(f"[OK] Train/test oranı: {train_ratio:.3f} / {1-train_ratio:.3f}")

# TEST 4: Eksik değer yok
if Xtr.isnull().any().any():
    errors.append("FAIL — X_train'de eksik değer var!")
else:
    print("[OK] X_train'de eksik değer yok")
if Xte.isnull().any().any():
    errors.append("FAIL — X_test'de eksik değer var!")
else:
    print("[OK] X_test'de eksik değer yok")

# TEST 5: Stratification — sınıf oranları 0.02'den fazla sapmamalı
for cls in ytr.unique():
    tr_r = (ytr == cls).mean()
    te_r = (yte == cls).mean()
    diff = abs(tr_r - te_r)
    if diff > 0.02:
        errors.append(f"FAIL — '{cls}' sınıfı stratification sapması çok yüksek: {diff:.3f}")
    else:
        print(f"[OK] Stratification '{cls}': train={tr_r:.3f}, test={te_r:.3f}, diff={diff:.4f}")

# TEST 6: GPA blok tamamen çıkmış olmalı (recall-hostile karar)
gpa_cols = [c for c in Xtr.columns if 'GPA' in c or 'GNO' in c or 'Semester' in c]
if gpa_cols:
    errors.append(f"FAIL — GPA sütunları işlenmiş veride tespit edildi: {gpa_cols}")
else:
    print("[OK] GPA blok tamamen çıkarıldı (Pre/Post/GNO yok)")

# TEST 7: Student_ID işlenmiş veride olmamalı
id_cols = [c for c in Xtr.columns if 'Student_ID' in c or 'student_id' in c.lower()]
if id_cols:
    errors.append(f"FAIL — Student_ID işlenmiş veride bulundu: {id_cols}")
else:
    print("[OK] Student_ID işlenmiş veride yok")

# TEST 8: Pipeline yüklenip çalışıyor mu?
try:
    pipe_check = joblib.load(MODELS_DIR + 'pipeline.joblib')
    sample_out = pipe_check.transform(X_test.iloc[:5])
    assert sample_out.shape[0] == 5
    print(f"[OK] Pipeline yüklenip çalıştırıldı, çıktı shape: {sample_out.shape}")
except Exception as e:
    errors.append(f"FAIL — Pipeline yüklenme/çalıştırma hatası: {e}")

# TEST 9: Korunan FE feature'ı (Toplam Çalışma Yükü) ham X'te var mı?
if 'Toplam Çalışma Yükü' not in X_train.columns:
    errors.append("FAIL — 'Toplam Çalışma Yükü' X_train'de yok")
else:
    print("[OK] Toplam Çalışma Yükü mevcut")

# TEST 9b: Elenmesi gereken 7 redundant feature pipeline çıktısında olmamalı
banned_fe = [
    'AI Bağımlılık Yükü', 'Sınav Kaygısı × AI Saati', 'AI × Kaygı × Bağımlılık',
    'AI Çalışma Payı', 'Kalıcılık / AI Saati', 'AI × Düşük Kalıcılık',
    'Bağımlılık / Kalıcılık',
]
leaked_fe = [c for c in banned_fe if c in X_train.columns or c in Xtr.columns]
if leaked_fe:
    errors.append(f"FAIL — Elenmesi gereken redundant FE'ler hâlâ var: {leaked_fe}")
else:
    print(f"[OK] 7 redundant FE temiz şekilde elendi")

# TEST 10: Target sütunu işlenmiş feature setinde olmamalı (hedef sızıntısı)
if TARGET in Xtr.columns or TARGET in Xte.columns:
    errors.append(f"FAIL — Target sütunu '{TARGET}' X matrisine sızdı!")
else:
    print(f"[OK] Target '{TARGET}' X matrisinde yok")

# ── SONUÇ
print("\n" + "=" * 55)
if errors:
    print(f"❌ {len(errors)} HATA BULUNDU:")
    for e in errors:
        print(f"   {e}")
    raise AssertionError("Phase 3 doğrulama başarısız. Hataları düzeltin ve yeniden çalıştırın.")
else:
    print("✅ TÜM TESTLER GEÇTİ — Phase 3 tamamlandı.")
    print(f"   Train: {Xtr.shape} | Test: {Xte.shape} | Features: {Xtr.shape[1]}")
print("=" * 55)
```

---

## ADIM 14 — SECTION 3 HEADER'INI GÜNCELLE

Section 3'ün en üstündeki HTML header hücresini bul ve içini şu ile değiştir:

```python
display(HTML("""
<div style="background: linear-gradient(90deg, #0a1f12, #14532d); border-left: 5px solid #22c55e;
  padding: 20px 28px; border-radius: 8px; margin: 24px 0;">
  <h2 style="color: #86efac; font-size: 1.6em; margin: 0 0 4px;">Section 3 · Data Preparation + Pipeline</h2>
  <p style="color: #4ade80; margin: 0; font-size: 0.95em;">Cenker · CRISP-DM Faz 3 · 20 Puan</p>
</div>
"""))
```

---

## GENEL KURALLAR

1. **Plotly dark tema** tüm grafiklerde: `template='plotly_dark'`, `paper_bgcolor='#0f0e17'`
2. **Tablo çıktıları** `display(df.style...)` ile göster — ham print yasak
3. **Yorumlar Türkçe**, değişken/fonksiyon isimleri İngilizce
4. **Her hücre çalıştırılabilir olmalı** — bağımsız import olmayan satırlar üste taşı
5. **Doğrulama hücresi en son** — tüm adımlar tamamlanmadan çalıştırılmaz

---

## TAMAMLAMA KRİTERLERİ

Aşağıdakiler sağlanmadan Phase 3 tamamlanmış sayılmaz:

- [ ] `data/processed/X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv` mevcut
- [ ] `models/pipeline.joblib` mevcut ve yüklenip çalışıyor
- [ ] Doğrulama hücresindeki tüm 10 test geçiyor
- [ ] `Post_Semester_GPA` ve `Student_ID` işlenmiş veride yok
- [ ] Her code hücresinin altında Markdown yorumu var
- [ ] Berkay'e handoff notu Section 3.10'da mevcut
