# PR / Session Summary — APEX-V2 Phase 1 & 2

Bu PR, APEX-V2 projesinin **Faz 1 (Business Understanding)** ve **Faz 2 (Data Understanding / EDA)** gereksinimlerini karşılamak amacıyla yapılan geliştirmeleri içermektedir.

---

## 🚀 Değişiklik Özeti (Session Summary)

### 1. İş Hedefleri, Metrikler & Karar Tasarımı (Faz 1)
* **Kart Tasarımı Entegrasyonu:** Section 1 (Business Understanding) paydaş sunumuna uygun görsel HTML kartları şeklinde yapılandırıldı. Mükerrer açıklamalar temizlendi.
* **Kaynak Değerlendirme, Veri Özellikleri ve Mantıksal Karar Bağlamları Kartı:** Projenin tek veri kaynağı olan veri setindeki tüm 16 değişkenin (15 özellik + 1 hedef) ham veri tipleri, birimleri, açıklamaları ve veri hazırlama aşamasındaki nihai tip dönüşüm hedefleri ok (->) gösterimiyle yeni bir HTML tablo kartı olarak Section 1'e dahil edildi. Ek olarak, veri setinden elde edilen disiplinler arası kullanım örüntüleri (STEM kodlama bağımlılığı, Humanities metin odağı vb.) ve geleneksel çalışma ikamesine dair mantıksal karar bağlamları karta entegre edilerek Business Understanding adımı güçlendirildi.
* **Tükenmişlik Seviyesi Odağı (Okul Terk Referansları Kaldırıldı):** Veri setinde yer almayan "okulu bırakma" (dropout) referansları projenin tamamından temizlendi; modelin sadece tükenmişlik riskini (`Burnout_Risk_Level`) tahmin ettiği netleştirildi.
* **Sunum ve Eğitimsel Önemli Notlar:** Akademik sunumlara yönelik olarak dönem başı tahminlemesinin proaktif rehberlikteki değeri, hata maliyeti mantığı ve Recall (High) metriğinin önceliklendirilme nedenleri açıklandı.
* **Başarı Metrikleri ve Hata Maliyeti:** Birincil metrik olarak Macro F1-Score (> 0.80), Recall (High Risk sınıfı için > 0.85) ve ROC-AUC (> 0.88) hedeflendi. False Positive / False Negative durumları eğitimsel etkileriyle açıklandı.

### 2. Temiz Keşifsel Veri Analizi - EDA (Faz 2) ve Kalite Denetim Düzeltmeleri
* **Yeniden Yazım:** `final_analysis.ipynb` dosyası hücre hücre sıfırdan, en temiz ve hatasız biçimde programatik olarak yeniden oluşturuldu.
* **Kalite Denetim Düzeltmeleri:**
  * **Veri Sızıntısı (Data Leakage) Değerlendirmesi:** Yeni bir bölüm (Section 2.8) eklenerek `Post_Semester_GPA` değişkeninin veri sızıntısı riski nedeniyle model dışı bırakılması gerekçelendirildi. İçinde `Post_Semester_GPA` barındıran `GPA_Change` feature engineering önerisi iptal edildi. `Skill_Retention_Score` için sızıntı takip protokolü tanımlandı.
  * **Eksik Değer Analizi:** Isı haritası yorumunda eksik değer içeren tüm 9 değişken (`Pre_Semester_GPA`, `Post_Semester_GPA`, `Weekly_GenAI_Hours`, `Skill_Retention_Score`, `Tool_Diversity`, `Perceived_AI_Dependency`, `Traditional_Study_Hours`, `Prompt_Engineering_Skill`, `Anxiety_Level_During_Exams`) listelendi.
  * **Sayısal Değişken Genişletmesi:** Sadece 5 değişken yerine veri setindeki tüm 8 sayısal değişken (`Tool_Diversity`, `Perceived_AI_Dependency`, `Post_Semester_GPA` dahil) dağılım ve box-plot analizlerine dahil edildi.
  * **Korelasyon Matrisi:** Hedef değişken `Burnout_Risk_Level` sayısal olarak (`Low`: 0, `Medium`: 1, `High`: 2) map edilerek korelasyon matrisine eklendi ve tüm 8 sayısal değişkenle olan doğrusal ilişkisi görselleştirildi.
  * **Primary Use Case Düzeltmesi:** Kullanım amacına göre risk oranlarının tüm sınıflarda dengeli olduğu (%25-%28) ve en yüksek hacimli grubun `Debugging/Troubleshooting` olduğu şekilde analist yorumu düzeltildi.
* **Görsel Standartlar:** Plotly (`plotly_dark` teması) kullanılarak kurumsal renk uyumuyla (Low: Emerald, Medium: Amber, High: Rose/Red) 17 ayrı görselleştirme hazırlandı. Grafikler `figures/` dizinine kaydedildi.
* **Yorum Temizliği ve Düzeltmeler:** Analiz yorumlarında bulunan tüm kopya ve mükerrer cümleler giderildi. Korelasyon yorumundaki `"poziyat"` yazım hatası `"pozitif"` olarak düzeltildi.

### 3. Çözülen Sistem Hataları
* **Windows Kaleido Deadlock Çözümü:** Plotly grafik kaydederken Windows sistemlerinde kilitlenmelere yol açan `py-kaleido==0.2.1` paketi, `kaleido>=1.3.0` sürümüne yükseltilerek çözüldü. `requirements.txt` dosyası güncellendi.

### 4. Dosya Düzeni & Katkı Günlüğü (Handoff Hazırlığı)
* **plans_and_agents/ Klasörü:** İş planı (`PHASE_1_2_PLAN.md`) ve agent yönergeleri (`PHASE_1_2_AGENT.md`) bu klasöre taşınarak kök dizin sadeleştirildi.
* **Faz Katılım Günlüğü:** Ortaklaşa güncellenebilir bir faz katılım günlüğü olan [plans_and_agents/phase_logs.md](file:///c:/Users/fezam/Desktop/apex-classification/apex-classification/plans_and_agents/phase_logs.md) dosyası oluşturuldu. Bu günlükte kimin, hangi fazda, neyi, neden ve nasıl yaptığı kayıt altına alındı. `CLAUDE.md` proje ağacına eklendi.

---

## 📊 Git Diff İstatistikleri (Git Diff Stats)

Aşağıda, upstream ana repo (`cenkergultekin/apex-classification`) ile yapılan değişikliklerin karşılaştırma istatistikleri yer almaktadır:

```text
 figures/eda_correlation_matrix.png              |  Bin 73890 -> 98614 bytes
 figures/eda_numeric_Perceived_AI_Dependency.png |  Bin 0 -> 47948 bytes
 figures/eda_numeric_Post_Semester_GPA.png       |  Bin 0 -> 58470 bytes
 figures/eda_numeric_Tool_Diversity.png          |  Bin 0 -> 46107 bytes
 notebooks/final_analysis.ipynb                  | 3539 ++++++++++++++++++++---
 plans_and_agents/PHASE_1_2_AGENT.md             |   65 +-
 plans_and_agents/PHASE_1_2_PLAN.md              |   26 +-
 plans_and_agents/phase_logs.md                  |   18 +-
 8 files changed, 3258 insertions(+), 390 deletions(-)
```

* **Toplam Etki:** 8 dosya değiştirildi, 3258 satır eklendi, 390 satır silindi.
* **Görsel Çıktılar:** `figures/` klasöründe toplam 17 adet PNG formatında Plotly çıktısı başarıyla oluşturuldu ve upstream reponun görsel gereksinimleri tamamlandı.
