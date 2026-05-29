# AI Student Impact — Faz 1 & 2 Uygulama Planı (CRISP-DM)

Bu doküman, **Feza**'nın sorumluluğunda olan **Faz 1 (Business Understanding)** ve **Faz 2 (Data Understanding / EDA)** adımlarının detaylı uygulama planını ve metodolojisini içermektedir. Projenin standartları gereği tüm görselleştirmeler Plotly ile yapılacak ve şablon notebook (`notebooks/final_analysis.ipynb`) doldurulacaktır.

---

## 1. Faz 1: Business Understanding (İş Anlamlandırma)

### 1.1. Hedef Değişken ve Problem Tanımı
* **Hedef Değişken (`Target`):** `Burnout_Risk_Level`
* **Sınıflar:** `Low` (Düşük Risk), `Medium` (Orta Risk), `High` (Yüksek Risk)
* **Problem Tipi:** 3 Sınıflı (Multi-class) Sınıflandırma.
* **Proje Amacı:** Öğrencilerin dönem başındaki akademik performansları, demografik bilgileri, yapay zeka kullanım alışkanlıkları ve kurumsal politikalar altındaki davranışlarından hareketle, dönem sonundaki akademik tükenmişlik risk seviyelerini erken aşamada tahmin etmektir. Proje, okulu bırakma veya okul terk oranlarıyla ilgili herhangi bir veri ya da analiz içermemektedir; sadece tükenmişlik seviyesini tespit etmeye odaklanmıştır.

### 1.2. Proje Başarı Metrikleri
Projenin doğası gereği başarıyı tek bir metrik yerine iş hedefleriyle uyumlu üçlü bir metrik setiyle değerlendireceğiz:
1. **Macro F1-Score (Birincil Metrik):** Sınıflar arası (Low, Medium, High) olası bir dağılım dengesizliğinde, modelin her üç sınıfta da dengeli bir performans göstermesi kritik önem taşır. Bu nedenle optimizasyon odağımız Macro F1 olacaktır. Hedef: **> 0.80**.
2. **Recall (High Risk Sınıfı için):** Yüksek tükenmişlik riski olan bir öğrencinin gözden kaçırılması (False Negative) kabul edilemez bir risktir. Akademik destek ve rehberlik servisinin bu öğrencileri kesinlikle yakalaması gerekir. Hedef: **> 0.85**.
3. **ROC-AUC (Genel Ayrım Gücü):** Modelin sınıfları birbirinden ayırt etme yeteneğini ölçmek amacıyla kullanılacaktır. Hedef: **> 0.88**.

### 1.3. Hata Maliyeti Analizi (Cost of Errors)

| Hata Tipi | Model Tahmini | Gerçek Durum | Akademik / Eğitimsel Etki ve Tahmini Maliyet |
|-----------|---------------|--------------|---------------------------------------------|
| **False Positive (Yanlış Alarm)** | High Risk | Low / Medium | Model, risk taşımayan bir öğrenciyi riskli olarak işaretler. Rehberlik birimi öğrenciyle proaktif bir görüşme yapar. **Maliyet: Düşük.** Sadece danışmanın zamanı harcanır. Öğrenci için hafif bir anksiyete yaratabilir ancak rehberlik kapsamında tolere edilebilir. |
| **False Negative (Kaçırılan Risk)** | Low / Medium | High Risk | Model, gerçekten tükenmişlik yaşayan ve yardıma ihtiyacı olan bir öğrenciyi sağlıklı tahmin eder. **Maliyet: Çok Yüksek.** Öğrenciye zamanında ulaşılamaz; akademik motivasyonu düşer, ders başarısı ciddi şekilde zarar görür ve psikolojik yıpranma önlenemez. |

* **Stratejik Karar:** Modelin eşik değerleri (thresholds) veya algoritma parametreleri False Negative sayısını en aza indirecek (yani Recall'u maksimize edecek) şekilde ayarlanmalıdır.

### 1.4. Sunum ve Eğitimsel Önemli Notlar (Presentation Highlights)
Bu projenin sunumunda dinleyicilere aktarılacak en önemli metodolojik detaylar şunlardır:
* **Erken Tahmin Yeteneği:** Tahminlerin dönem başı verileriyle yapılması, dönem ortası veya sonu gibi geç aşamalara gelinmeden akademik danışmanların önleyici destek planları hazırlamasına olanak tanır.
* **Eşik Değer Optimizasyonu:** Hata maliyeti tablosunda da görüldüğü üzere, False Negative hatasının eğitimsel maliyeti FP'den çok daha büyüktür. Bu yüzden model optimizasyonunda Recall (High) hedeflenmiştir.
* **Karar Destek Olasılıkları:** Sunumda modelin sadece statik bir sınıf tahmini vermediği, her öğrenci için bir risk olasılığı (%) hesapladığı ve danışmanların bu olasılık derecesine göre öğrencileri sıralayabileceği aktarılmalıdır.

### 1.5. Kaynak Değerlendirme ve Veri Özellikleri
Projenin tek ve en kritik veri kaynağı olan 50.000 gözlemli `ai_student_impact_dataset.csv` veri setindeki tüm 16 değişken (15 özellik ve 1 hedef değişken), veri hazırlama aşamasındaki veri tipi dönüşüm hedefleriyle birlikte aşağıda listelenmiştir:

| Değişken Adı | Mevcut -> Olması Gereken Tür / Birim | Değişken Açıklaması | Hedef Değişkene (Target) Göre Anlamı ve Önem Derecesi |
|---|---|---|---|
| **Student_ID** | `int64 (Sayısal / ID) -> Kaldırılacak (Ezber Engelleme)` | Öğrenciye özel benzersiz kimlik numarası (100001 - 150000). | Açıklayıcı değeri yoktur. Modelin ezber yapmasını engellemek için veri setinden düşürülecektir. |
| **Major_Category** | `object (Metin) -> Kategorik (One-Hot - Sayısal)` | Öğrencinin eğitim gördüğü ana akademik alan (Arts, Business, Humanities, Medical, STEM). | Bölümler arası risk dağılımı dengelidir. Farklı disiplinlerin YZ kullanım örüntülerini yakalamak için modele dahil edilecektir. |
| **Year_of_Study** | `object (Metin) -> Kategorik (Ordinal - Sayısal)` | Öğrencinin eğitim yılı/sınıf seviyesi (Freshman, Sophomore, Junior, Senior, Graduate). | Sınıf derecesinin yükselmesiyle birlikte (özellikle Graduate seviyesinde) akademik stres ve bitirme yükünün getirdiği tükenmişlik riski değişebilir. |
| **Pre_Semester_GPA** | `float64 (Sayısal / Not Ortalaması 0-4) -> Sayısal (Standartlaştırılmış GPA)` | Öğrencinin dönem başındaki akademik not ortalaması (1.18 - 4.00 arası). | Hafif negatif ilişki. Başlangıç GPA'i düşük öğrencilerin sınav stresine ve tükenmişlik riskine daha açık olduğu gözlemlenmiştir. |
| **Weekly_GenAI_Hours** | `float64 (Sayısal / Saat/Hafta) -> Sayısal (Standartlaştırılmış Saat)` | Haftalık üretken yapay zeka araçları kullanım süresi (0.0 - 40.0 saat arası). | **En Yüksek Pozitif Korelasyon (0.47).** YZ kullanım süresinin artması, artan stres ve tükenmişlik riskiyle doğrudan paraleldir. |
| **Primary_Use_Case** | `object (Metin) -> Kategorik (One-Hot - Sayısal)` | YZ araçlarının öncelikli kullanım amacı (Debugging/Troubleshooting, Ideation, Summarizing_Reading vb.). | Risk sınıfları üzerindeki oranları dengelidir (%25-%28 bandı). "Debugging/Troubleshooting" en yüksek hacimli gruptur. |
| **Prompt_Engineering_Skill** | `object (Metin) -> Kategorik (Ordinal - Sayısal)` | Öğrencinin prompt (komut) yazma beceri seviyesi (Beginner, Intermediate, Advanced). | Prompt becerisi "Beginner" olan ve ücretli aboneliği bulunmayan öğrencilerde stres ve tükenmişlik riski görece daha yüksektir. |
| **Tool_Diversity** | `float64 (Sayısal / Adet) -> Sayısal (Standartlaştırılmış Adet)` | Öğrencinin kullandığı farklı üretken YZ araçlarının sayısı (1.0 - 5.0 adet arası). | Hafif pozitif ilişki. Çok sayıda farklı aracın aynı anda kullanılması bilişsel adaptasyon yükünü artırarak tükenmişliği tetikleyebilir. |
| **Paid_Subscription** | `bool (Mantıksal) -> Sayısal (Binary 0/1)` | Öğrencinin herhangi bir ücretli YZ aracına aboneliği olup olmadığı (True / False). | Aboneliği olmayan başlangıç düzeyindeki öğrencilerde tükenmişlik riski hafif düzeyde daha yaygındır. |
| **Traditional_Study_Hours** | `float64 (Sayısal / Saat/Hafta) -> Sayısal (Standartlaştırılmış Saat)` | Kütüphane, kitap okuma, grup çalışması gibi geleneksel ders çalışma süresi (1.0 - 35.86 saat arası). | **Negatif Korelasyon (-0.14).** Geleneksel yöntemlerle çalışmaya daha fazla zaman ayıran öğrencilerin tükenmişlik riski azalmaktadır. |
| **Perceived_AI_Dependency** | `float64 (Sayısal / Skor 1-10) -> Sayısal (Standartlaştırılmış Skor)` | Öğrencinin YZ araçlarına olan bağımlılığını hissetme derecesi (1.0 - 10.0 arası). | **Güçlü Pozitif Korelasyon (0.37).** Kendini yapay zekaya bağımlı hisseden öğrencilerde tükenmişlik riski son derece belirgindir. |
| **Institutional_Policy** | `object (Metin) -> Kategorik (One-Hot - Sayısal)` | Okulun yapay zeka kullanım kuralları (Actively_Encouraged, Allowed_With_Citation, Strict_Ban). | Yasalara/kısıtlamalara tabi olan öğrencilerde YZ bağımlılığı ve akademik kaygı örüntüleri dolaylı etkiler yaratabilir. Oransal dağılım homojendir. |
| **Anxiety_Level_During_Exams** | `float64 (Sayısal / Skor 1-10) -> Sayısal (Standartlaştırılmış Kaygı Skoru)` | Sınav dönemlerinde hissedilen stres ve kaygı derecesi (1.0 - 10.0 arası). | **Pozitif Korelasyon (0.16).** Sınav kaygısı yüksek olan öğrencilerin High risk seviyesine girme ihtimali daha yüksektir. |
| **Post_Semester_GPA** | `float64 (Sayısal / Not Ortalaması 0-4) -> Kaldırılacak (Veri Sızıntısı Engelleme)` | Dönem sonundaki başarı notu ortalaması (1.00 - 4.00 arası). | **Kritik Veri Sızıntısı (Data Leakage)!** Dönem sonuna ait olduğu ve dönem başındaki öngörü modelinde bilinemeyeceği için kesinlikle çıkarılmalıdır. |
| **Skill_Retention_Score** | `float64 (Sayısal / Skor 0-100) -> Sayısal (Standartlaştırılmış Skor)` | Akademik bilgilerin zihinde kalıcılık derecesi (10.78 - 100.0 arası). | Hafif negatif ilişki. Hafıza kalıcılığı zayıf olan öğrencilerde tükenmişlik baskısı artabilir. Modellemede sızıntı takibi için yakından izlenecektir. |
| **Burnout_Risk_Level** | `object (Metin) -> Sayısal (Ordinal Sınıf Etiketi 0/1/2) - HEDEF` | Tahmin edilmeye çalışılan akademik tükenmişlik risk düzeyi (Low, Medium, High). | **Hedef Değişken (Target).** Model bu sınıfları tahmin etmek üzere eğitilmektedir. Sınıf dağılımı dengelidir. |

#### 1.5.1. Veri İçi Mantıksal Bağlamlar ve İş Odaklı Çıkarımlar
Veri setindeki değişkenlerin çapraz sorgulamaları sonucunda elde edilen ve akademik rehberlik stratejileri için karar destek sağlayan önemli örüntüler şunlardır:
* **STEM ve Kodlama Bağımlılığı:** STEM öğrencileri veri setinde en kalabalık grubu oluşturmakla kalmayıp (%30.1), haftalık ortalama YZ kullanımında **10.84 saat** ile liderdir. Bu durum, **3.79** bağımlılık algısı skoru ve en yüksek yüksek-riskli tükenmişlik oranı (**%30.0**) ile paralellik gösterir. STEM öğrencilerinin YZ'yi en çok **Debugging (%51.7)** için kullanması, kod hatalarını çözmede YZ'ye olan yüksek bağımlılık sarmalını açıklanmaktadır.
* **Humanities ve Yazım Odaklılık:** Humanities öğrencileri YZ'yi çoğunlukla **Copywriting/Drafting (%51.9)** amacıyla kullanmaktadır. Bu grupta YZ kullanım süresi (7.11 saat/hafta) ve bağımlılık algısı (3.24) en düşük düzeydedir. Buna bağlı olarak, yüksek tükenmişlik riski de bu grupta en düşük seviyededir (**%20.7**). Metin yazımı odaklı YZ kullanımının kodlama kadar yüksek bilişsel stres yaratmadığı söylenebilir.
* **Business ve Medical Uyumu:** Kullanım amaçları disiplinlerle örtüşmektedir. Business öğrencileri YZ'yi fikir üretmek için **Ideation (%47.9)** amacıyla kullanırken (yüksek risk %24.3), Medical öğrencileri yoğun ezber ve literatür yükünü yönetmek adına **Summarizing/Reading (%47.9)** amaçlı kullanmaktadır (yüksek risk %23.2).
* **Geleneksel Çalışma ve İkame Riskleri:** Geleneksel çalışma saatlerinin STEM öğrencilerinde en düşük (11.00 saat) olması, YZ kullanımının geleneksel çalışmanın yerini almaya başladığını gösterir. Bu ikame ilişkisi, sınav kaygısını ve tükenmişliği tetiklemektedir. Akademik rehberlik birimleri için YZ'yi bir ikame değil, destekleyici araç olarak konumlandırma stratejisi kritiktir.

---

## 2. Faz 2: Data Understanding (Keşifsel Veri Analizi - EDA)

### 2.2. Adım: Target Dağılımı ve Sınıf Dengesi
* **Aksiyon:** `Burnout_Risk_Level` dağılımı `Low -> Medium -> High` sıralaması korunarak incelenecek.
* **Grafikler (Plotly):** Yan yana iki grafik (`make_subplots`):
  1. *Bar Chart*: Her risk seviyesinin öğrenci sayısı.
  2. *Pie Chart*: Yüzdesel dağılım (Dengeli / Dengesiz dağılım analizi için).
* **Görsel Standart:** Yorum hücresi premium HTML kartı biçiminde yazılacaktır.

### 2.3. Adım: Eksik Değer Dağılım Analizi
* **Aksiyon:** 50.000 satırlık tüm veri seti taranarak eksik değer barındıran tüm 9 özellik (`Pre_Semester_GPA`, `Post_Semester_GPA`, `Weekly_GenAI_Hours`, `Skill_Retention_Score`, `Tool_Diversity`, `Perceived_AI_Dependency`, `Traditional_Study_Hours`, `Prompt_Engineering_Skill`, `Anxiety_Level_During_Exams`) oranları ve adetleri hesaplanacaktır.
* **Grafik (Plotly Bar Chart):** Sadece eksik değer içeren değişkenlerin eksiklik oranlarını (%) gösteren bar grafiği.
* **Analiz Yorumu:** Eksik değerlerin rastgele mi (MCAR) yoksa sistemsel anket doldurmama eğilimi gibi veri girişi hatalarından mı kaynaklandığı HTML kart formatında tartışılacaktır.

### 2.4. Adım: Sayısal Değişken Dağılımları (Univariate Analysis ve Hedef Kırılımı)
* **Aksiyon:** 7 sürekli sayısal değişken (`Pre_Semester_GPA`, `Weekly_GenAI_Hours`, `Traditional_Study_Hours`, `Skill_Retention_Score`, `Anxiety_Level_During_Exams`, `Perceived_AI_Dependency` ve `Post_Semester_GPA`) incelenecektir.
* **Grafik (Plotly):** Sürekli değişkenler için yan yana histogram (dağılımı görmek için) ve hedef kırılımlı box-plot (risk sınıfları bazında çeyreklik yayılım farklarını görmek için).
* **Sayım Değişkeni (`Tool_Diversity`):** Tool_Diversity (1-5 tam sayı) değişkeni için anlamsız olan box plot kaldırılacak, yerine hedef risk seviyeleri bazında gruplanmış bar grafiği çizilecektir.
* **Analiz Yorumu:** Sayısal özelliklerin hedef değişkenle ilişkisi (örn. YZ saati artışının risk grubunu yukarı taşıması) HTML kart formatında açıklanacaktır.

### 2.5. Adım: Kategorik Değişken Frekans Analizi
* **Aksiyon:** 6 kategorik özellik (`Major_Category`, `Year_of_Study`, `Primary_Use_Case`, `Prompt_Engineering_Skill`, `Paid_Subscription`, `Institutional_Policy`) incelenecektir.
* **Grafik (Plotly):** Her kategorik değişken için hedef kırılımlı gruplanmış bar grafikleri (`color=TARGET`, `barmode='group'`).
* **Analiz Yorumu:** Disiplin ve YZ kullanım amaçları kırılımları ile bunların hedef riski ne ölçüde beslediği (örn. STEM-Debugging ve Humanities-Copywriting ilişkisi) HTML kart formatında açıklanacaktır.

### 2.6. Adım: Korelasyon Matrisi (Multivariate Analysis)
* **Aksiyon:** Sayısal ve kategorik tüm 15 değişken (Student_ID hariç) matrise dahil edilecektir. Kategorik değişkenler mapped (ordinal/binary) ve label-encoded (nominal) şekilde sayısallaştırılacaktır.
* **Grafik (Plotly Heatmap):** Tüm 15 değişkenin Pearson korelasyon katsayılarını gösteren 15x15 matris grafiği.
* **Analiz Yorumu:** Değişkenlerin hedefle ilişkileri ve nominal özelliklerin düşük doğrusal katsayılarının doğrusal olmayan model gerekliliğine (LightGBM/XGBoost) etkisi HTML kart formatında açıklanacaktır.

### 2.7. Adım: İkili Değişken Analizleri (Bivariate Analysis)
* **Aksiyon:** Hedef değişkeni en çok etkileyen ve aralarında en güçlü pozitif/negatif bağlar bulunan değişkenlerin ikili ilişkileri incelenecektir.
* **Grafikler (Plotly):**
  1. *Weekly_GenAI_Hours vs. Traditional_Study_Hours:* YZ ikame etkisi ve risk seviyesi dağılımı (scatter plot).
  2. *Weekly_GenAI_Hours vs. Skill_Retention_Score:* Aşırı YZ kullanımının beceri kalıcılığına etkisi ve risk dağılımı (scatter plot).
  3. *Weekly_GenAI_Hours vs. Perceived_AI_Dependency:* Fiili kullanım saati ile bağımlılık algısı arasındaki doğrudan doğrusal ilişki ve risk sınıfları (ortalama çizgi grafiği).
* **Analiz Yorumu:** İkili etkileşimlerin akademik tükenmişlik riski üzerindeki yapısal etkileri HTML kart formatında açıklanacaktır.

### 2.8. Adım: Aykırı Değer (Outlier) Tespiti
* **Aksiyon:** 8 sayısal sütunda Tukey IQR yöntemiyle alt/üst sınırlar dışındaki aykırı değer sayılarının ve oranlarının hesaplanması.
* **Görsel:** Tekil analizlerdeki box-plotlarda da yer alan uç noktaların aykırı değer tablosuyla sunulması.
* **Analiz Yorumu:** Tespit edilen düşük aykırı oranlarının veri hatası değil doğal öğrenci davranışı olması ve veri hazırlama aşamasında `RobustScaler` / `clipping` kararlarının gerekçelendirilmesi (HTML kartı).

### 2.9. Adım: Veri Sızıntısı (Data Leakage) Değerlendirmesi
* **Aksiyon:** Sızıntının ne olduğu, eğitime/canlı performansa zararları detaylandırılacak ve alınan net kararlar vurgulanacaktır.
* **Görsel Standart:** Veri sızıntısının tanımı, `Post_Semester_GPA` ve `GPA_Change` üzerinden somut örnekleri ve sızıntıyı önlemek için bu özelliklerin drop edilmesi kararları premium kırmızı-yeşil HTML kart yapısıyla sunulacaktır.

### 2.10. Adım: EDA Bulguları ve Süreç Değerlendirme Raporu
* **Aksiyon:** Faz 3 (Veri Hazırlama) adımlarını yürütecek olan Cenker için tüm EDA sürecini özetleyen bir mini rapor kartı sunulacaktır.
* **İçerik:**
  * Süreç değerlendirmesi ve temel veri bulguları (ikame, sızıntı, bağımlılık sarmalı).
  * 9 değişkendeki eksikliklerin Median/Mode Imputer ile doldurulması, Ordinal/One-Hot encoding stratejileri, `Post_Semester_GPA` ve `Student_ID` drop kararları gibi net handoff talimatları.

---

## 3. Görsel ve Yazım Standartları
* **Kütüphaneler:** Tüm grafikler sadece **Plotly** (`plotly.express` veya `plotly.graph_objects`) ile oluşturulacaktır. `matplotlib` veya `seaborn` kullanılmayacaktır.
* **Tema:** Şablondaki modern koyu tema arka planı ve renk harmonisi (`ggplot2` veya özel dark temalar) korunacaktır.
* **Yorum Zorunluluğu:** Her grafiğin altında kalın (`**Veri Analisti Yorumu:**`) ile başlayan ve grafikten elde edilen iş odaklı 2-3 cümleyi içeren açıklama hücresi bulunacaktır.
* **İngilizce-Türkçe Uyumu:** Kod içi isimlendirmeler ve değişkenler orijinal veri setindeki gibi (İngilizce) bırakılacak, ancak analiz metinleri, başlıklar ve yorumlar tamamen Türkçe yazılacaktır.
