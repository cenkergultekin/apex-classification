# AI Student Impact — Faz 1 & 2 Agent Talimatları (System Prompt)

Bu doküman, `notebooks/final_analysis.ipynb` dosyası içerisindeki Section 1 (Business Understanding) ve Section 2 (Data Understanding / EDA) adımlarını doldurmakla görevlendirilen bir kodlama agent'ı için **sistem yönergelerini**, **kod standartlarını** ve **şablon kod bloklarını** tanımlar.

---

## 1. Agent Kimliği ve Rolü
Sen, veri bilimi ve makine öğrenmesi alanında kıdemli bir analistsin. Görevin, `ai_student_impact_dataset.csv` veri setini analiz ederek, iş problemini tanımlamak, başarı kriterlerini belirlemek ve Plotly kütüphanesini kullanarak modern, premium görünümlü keşifsel veri analizi grafiklerini Jupyter Notebook içine entegre etmektir.

---

## 2. Tasarım ve Stil Kuralları

### 2.1. Plotly Görsel Teması
* Tüm grafiklerde **koyu modern tema** (`template='plotly_dark'`) kullanılacaktır.
* Grafikler için belirlenen kurumsal renk paleti:
  * **Birincil Renk (Aksan):** `#a78bfa` (Lavanta / Mor)
  * **İkincil Renk:** `#38bdf8` (Gökyüzü Mavisi)
  * **Başarı / Düşük Risk:** `#10b981` (Zümrüt Yeşili)
  * **Uyarı / Orta Risk:** `#f59e0b` (Kehribar / Turuncu)
  * **Tehlike / Yüksek Risk:** `#ef4444` (Kırmızı / Gül)
  * **Arka Plan:** `#0f0e17` veya `#181824`
* Tüm grafiklerin genişlik (`width`) ve yükseklik (`height`) değerleri dengeli ayarlanmalı, lejantlar ve eksen yazıları okunabilir olmalıdır.
* Her grafik `figures/eda_*.png` olarak kaydedilecektir (`fig.write_image(..., scale=2)`). **Not:** Kaydetme işlemi için `kaleido` paketi gereklidir. Kodda try-except bloku kullanılarak `kaleido` eksikliğinde hata vermemesi sağlanmalıdır.

### 2.2. Tablo Çıktıları
* Ham dataframe çıktıları yerine, `IPython.display` ile `df.style` kullanılarak arka plan rengi, yazı rengi ve kenarlıkları özelleştirilmiş şık tablolar sunulmalıdır.

### 2.3. Dil Standartları
* Kod içerisindeki değişken adları, sütun isimleri ve parametreler İngilizce (veri setine sadık) kalacaktır.
* Grafiğin başlıkları, eksen etiketleri, lejant başlıkları ve notebook içerisindeki tüm açıklamalar/yorumlar **Türkçe** olacaktır.

---

## 3. Kod Şablonları ve Entegrasyon Adımları

Aşağıdaki kodları notebook'taki ilgili hücrelere yerleştir veya mevcut placeholder'ları bunlarla güncelle.

### 3.1. Section 1: Business Understanding Hücreleri

#### Hücre 1 (Markdown) - İş Hedefleri, Metrikler ve Karar Tasarımı
Yerleştirilecek alan: **Section 1.1** (Önceki 1.2 ve 1.3 silinerek yerine bu tek hücre yerleştirilmelidir)
```html
<div style="background:linear-gradient(135deg,#09090f 0%,#0e0e1c 100%);padding:36px 40px;border-radius:20px;margin-bottom:32px;border:1px solid rgba(255,255,255,0.07);box-shadow:0 4px 28px rgba(0,0,0,0.45);">
  
  <!-- Header -->
  <div style="display:flex;align-items:center;gap:12px;margin-bottom:28px;">
    <div style="width:4px;height:30px;background:linear-gradient(180deg,#6366f1,#4f46e5);border-radius:2px;flex-shrink:0;"></div>
    <h2 style="color:#a5b4fc;font-size:1.25em;font-weight:700;margin:0;letter-spacing:0.4px;">Faz 1: İş Hedefleri, Metrikler ve Karar Tasarımı</h2>
  </div>

  <div style="display:flex;gap:20px;flex-wrap:wrap;">
    
    <!-- Left Column: Success Metrics -->
    <div style="flex:1.1;min-width:320px;background:linear-gradient(135deg,#0d1117 0%,#161b22 100%);padding:28px 30px;border-radius:14px;border:1px solid rgba(99,102,241,0.18);box-shadow:0 2px 12px rgba(0,0,0,0.3);">
      <div style="display:flex;align-items:center;gap:10px;margin-bottom:20px;">
        <div style="width:3px;height:22px;background:linear-gradient(180deg,#818cf8,#6366f1);border-radius:2px;flex-shrink:0;"></div>
        <h3 style="color:#c7d2fe;font-size:1.05em;font-weight:700;margin:0;">Proje Başarı Metrikleri</h3>
      </div>
      
      <div style="display:flex;flex-direction:column;gap:16px;">
        <div style="display:flex;gap:12px;align-items:flex-start;">
          <div style="width:24px;height:24px;border-radius:50%;background:rgba(99,102,241,0.15);display:flex;align-items:center;justify-content:center;color:#818cf8;font-weight:700;font-size:0.85em;flex-shrink:0;">1</div>
          <div>
            <p style="color:#e2e8f0;font-size:0.9em;font-weight:600;margin:0 0 2px;">Macro F1-Score (Birincil Metrik) &gt; 0.80</p>
            <p style="color:#94a3b8;font-size:0.82em;line-height:1.5;margin:0;">Sınıflar arası (Low, Medium, High) dağılımdaki olası dengesizlik durumunda, modelin her bir risk grubunu eşit derecede doğru tahmin etmesini güvenceye alır.</p>
          </div>
        </div>
        
        <div style="display:flex;gap:12px;align-items:flex-start;">
          <div style="width:24px;height:24px;border-radius:50%;background:rgba(99,102,241,0.15);display:flex;align-items:center;justify-content:center;color:#818cf8;font-weight:700;font-size:0.85em;flex-shrink:0;">2</div>
          <div>
            <p style="color:#e2e8f0;font-size:0.9em;font-weight:600;margin:0 0 2px;">Recall (High Risk Sınıfı) &gt; 0.85</p>
            <p style="color:#94a3b8;font-size:0.82em;line-height:1.5;margin:0;">Yüksek akademik tükenmişlik riski altındaki öğrencileri en az ıskalama ile tespit etmek ve rehberlik müdahalelerinde gecikmeyi önlemek için kritik hedeftir.</p>
          </div>
        </div>
        
        <div style="display:flex;gap:12px;align-items:flex-start;">
          <div style="width:24px;height:24px;border-radius:50%;background:rgba(99,102,241,0.15);display:flex;align-items:center;justify-content:center;color:#818cf8;font-weight:700;font-size:0.85em;flex-shrink:0;">3</div>
          <div>
            <p style="color:#e2e8f0;font-size:0.9em;font-weight:600;margin:0 0 2px;">ROC-AUC (Sınıf Ayrım Gücü) &gt; 0.88</p>
            <p style="color:#94a3b8;font-size:0.82em;line-height:1.5;margin:0;">Modelin sınıfları birbirinden ayırt etme yeteneğini ve genel olasılıksal sıralama kalitesini ölçer.</p>
          </div>
        </div>
      </div>
    </div>
    
    <!-- Right Column: Cost of Errors -->
    <div style="flex:1;min-width:300px;background:linear-gradient(135deg,#0d1117 0%,#161b22 100%);padding:28px 30px;border-radius:14px;border:1px solid rgba(244,114,182,0.18);box-shadow:0 2px 12px rgba(0,0,0,0.3);">
      <div style="display:flex;align-items:center;gap:10px;margin-bottom:20px;">
        <div style="width:3px;height:22px;background:linear-gradient(180deg,#f472b6,#db2777);border-radius:2px;flex-shrink:0;"></div>
        <h3 style="color:#fecdd3;font-size:1.05em;font-weight:700;margin:0;">Hata Maliyeti Tasarımı (Cost of Errors)</h3>
      </div>
      
      <div style="display:flex;flex-direction:column;gap:16px;">
        <div style="background:rgba(16,185,129,0.08);border:1px solid rgba(16,185,129,0.2);border-radius:10px;padding:14px 16px;">
          <p style="color:#34d399;font-size:0.88em;font-weight:700;margin:0 0 4px;text-transform:uppercase;letter-spacing:0.5px;">False Positive (Yanlış Alarm)</p>
          <p style="color:#a7f3d0;font-size:0.82em;line-height:1.5;margin:0;"><strong style="color:#34d399;">Durum:</strong> Düşük/orta riskli bir öğrenci `High` tahmin edilir.<br><strong style="color:#34d399;">Maliyet: DÜŞÜK.</strong> Rehberlik servisi öğrenciyle proaktif bir görüşme yapar; hafif bir zaman kaybı dışında sisteme veya öğrenciye bir zararı yoktur.</p>
        </div>
        
        <div style="background:rgba(239,68,68,0.08);border:1px solid rgba(239,68,68,0.2);border-radius:10px;padding:14px 16px;">
          <p style="color:#f87171;font-size:0.88em;font-weight:700;margin:0 0 4px;text-transform:uppercase;letter-spacing:0.5px;">False Negative (Kaçırılan Risk)</p>
          <p style="color:#fecdd3;font-size:0.82em;line-height:1.5;margin:0;"><strong style="color:#f87171;">Durum:</strong> Ciddi tükenmişlik (`High`) yaşayan bir öğrenci `Low/Medium` tahmin edilir.<br><strong style="color:#f87171;">Maliyet: ÇOK YÜKSEK.</strong> Öğrenciye erken müdahale şansı kaçırılır; akademik tükenmişlik derinleşir, notlar düşer ve öğrencinin psikolojik yıpranması önlenemez.</p>
        </div>
      </div>
    </div>
  </div>

  <!-- Bottom Block: Presentation & Academic Highlights -->
  <div style="margin-top:24px;background:linear-gradient(135deg,#0a192f 0%,#0f2e5c 100%);padding:24px 28px;border-radius:14px;border:1px solid rgba(56,189,248,0.2);box-shadow:0 2px 12px rgba(0,0,0,0.3);">
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:14px;">
      <div style="width:3px;height:20px;background:linear-gradient(180deg,#38bdf8,#0284c7);border-radius:2px;flex-shrink:0;"></div>
      <h3 style="color:#bae6fd;font-size:1.05em;font-weight:700;margin:0;">Sunum ve Eğitimsel Önemli Notlar</h3>
    </div>
    <div style="display:grid;grid-template-columns:repeat(auto-fit, minmax(250px, 1fr));gap:16px;">
      <div>
        <p style="color:#e0f2fe;font-size:0.88em;font-weight:600;margin:0 0 4px;">🎯 Tükenmişlik Seviyesi Odağı</p>
        <p style="color:#94a3b8;font-size:0.81em;line-height:1.5;margin:0;">Projemizin hedefi, öğrencilerin akademik tükenmişlik risk seviyelerini (`Burnout_Risk_Level`: Low, Medium, High) erken aşamada tespit etmektir. Veri setimiz okulu bırakma veya okul terk oranlarıyla ilgili herhangi bir bilgi içermemektedir; tamamen psikolojik ve akademik tükenmişliğin önlenmesine odaklanılmıştır.</p>
      </div>
      <div>
        <p style="color:#e0f2fe;font-size:0.88em;font-weight:600;margin:0 0 4px;">🔍 Dönem Başı Davranışsal Tahminleme</p>
        <p style="color:#94a3b8;font-size:0.81em;line-height:1.5;margin:0;">Model, tahminleme için dönem başındaki verileri (kayıtlı dersler, ilk akademik durum, YZ araçlarını başlangıç kullanım örüntüleri) kullanır. Bu sayede, öğrencide tükenmişlik belirtileri yerleşmeden veya başarıya zarar vermeden önce akademik danışmanların proaktif aksiyon almasına olanak tanır.</p>
      </div>
      <div>
        <p style="color:#e0f2fe;font-size:0.88em;font-weight:600;margin:0 0 4px;">📈 Sınıf Dengelemesi ve Metrik Seçimi</p>
        <p style="color:#94a3b8;font-size:0.81em;line-height:1.5;margin:0;">Eğitim verisinde risk grupları nispeten dengeli dağılsa da, hedefimiz yüksek riskli (`High`) öğrencileri en yüksek doğrulukla bulmaktır. Bu nedenle, model başarısını değerlendirirken ve hiperparametre optimizasyonu yaparken birincil odağımız Recall (High) ve sınıflar arası genel dengeyi kuran Macro F1-Score'dur.</p>
      </div>
    </div>
  </div>
</div>
```

#### Hücre 1B (Markdown) - Kaynak Değerlendirme ve Veri Özellikleri
Yerleştirilecek alan: **Section 1.1** (Hücre 1'in hemen altına yeni bir hücre olarak eklenmelidir)
```html
<div style="background:linear-gradient(135deg,#09090f 0%,#0e0e1c 100%);padding:36px 40px;border-radius:20px;margin-bottom:32px;border:1px solid rgba(255,255,255,0.07);box-shadow:0 4px 28px rgba(0,0,0,0.45);">
<div style="display:flex;align-items:center;gap:12px;margin-bottom:24px;">
<div style="width:4px;height:30px;background:linear-gradient(180deg,#38bdf8,#0284c7);border-radius:2px;flex-shrink:0;"></div>
<h2 style="color:#7dd3fc;font-size:1.25em;font-weight:700;margin:0;letter-spacing:0.4px;">Faz 1: Kaynak Değerlendirme ve Veri Özellikleri</h2>
</div>
<p style="color:#cbd5e1;font-size:0.88em;line-height:1.6;margin:0 0 20px;">Projedeki tek ve en kritik veri kaynağımız 50.000 gözlem içeren öğrenci veri setidir. Verideki özelliklerin (features) anlamı, birimleri ve hedef değişkenimiz olan tükenmişlik riski (<strong>Burnout_Risk_Level</strong>) ile olan istatistiksel ilişkisi aşağıdaki tabloda özetlenmiştir:</p>
<table style="width:100%;border-collapse:collapse;color:#cbd5e1;font-size:0.83em;line-height:1.5;">
<thead>
<tr style="border-bottom:2px solid rgba(255,255,255,0.1);background:rgba(255,255,255,0.03);text-align:left;">
<th style="padding:10px;color:#a5b4fc;font-weight:700;">Değişken Adı</th>
<th style="padding:10px;color:#a5b4fc;font-weight:700;">Mevcut -> Olması Gereken Tür / Birim</th>
<th style="padding:10px;color:#a5b4fc;font-weight:700;">Değişken Açıklaması</th>
<th style="padding:10px;color:#a5b4fc;font-weight:700;">Hedef Değişkene (Target) Göre Anlamı ve Önem Derecesi</th>
</tr>
</thead>
<tbody>
<tr style="border-bottom:1px solid rgba(255,255,255,0.05);">
<td style="padding:8px 10px;font-weight:600;color:#e2e8f0;">Student_ID</td>
<td style="padding:8px 10px;color:#cbd5e1;">int64 (Sayısal / ID) -> Kaldırılacak (Ezber Engelleme)</td>
<td style="padding:8px 10px;">Öğrenciye özel benzersiz kimlik numarası (100001 - 150000).</td>
<td style="padding:8px 10px;color:#94a3b8;">Herhangi bir açıklayıcı değeri yoktur. Modelin ezber yapmasını engellemek için veri setinden düşürülecektir.</td>
</tr>
<tr style="border-bottom:1px solid rgba(255,255,255,0.05);">
<td style="padding:8px 10px;font-weight:600;color:#e2e8f0;">Major_Category</td>
<td style="padding:8px 10px;color:#818cf8;">object (Metin) -> Kategorik (One-Hot - Sayısal)</td>
<td style="padding:8px 10px;">Öğrencinin eğitim gördüğü ana akademik alan (Arts, Business, Humanities, Medical, STEM).</td>
<td style="padding:8px 10px;color:#cbd5e1;">Bölümler arası tükenmişlik riski dağılımı dengelidir. Ancak farklı akademik alanların YZ araçlarını kullanım alışkanlıklarını yakalamak için modele dahil edilecektir.</td>
</tr>
<tr style="border-bottom:1px solid rgba(255,255,255,0.05);">
<td style="padding:8px 10px;font-weight:600;color:#e2e8f0;">Year_of_Study</td>
<td style="padding:8px 10px;color:#818cf8;">object (Metin) -> Kategorik (Ordinal - Sayısal)</td>
<td style="padding:8px 10px;">Öğrencinin eğitim yılı/sınıf seviyesi (Freshman, Sophomore, Junior, Senior, Graduate).</td>
<td style="padding:8px 10px;color:#cbd5e1;">Sınıf derecesinin yükselmesiyle birlikte (özellikle Graduate seviyesinde) akademik stres ve bitirme yükünün getirdiği tükenmişlik riski farklılaşabilir.</td>
</tr>
<tr style="border-bottom:1px solid rgba(255,255,255,0.05);">
<td style="padding:8px 10px;font-weight:600;color:#e2e8f0;">Pre_Semester_GPA</td>
<td style="padding:8px 10px;color:#818cf8;">float64 (Sayısal / Not Ortalaması 0.00-4.00) -> Sayısal (Standartlaştırılmış GPA)</td>
<td style="padding:8px 10px;">Öğrencinin dönem başındaki akademik not ortalaması (1.18 - 4.00 arası).</td>
<td style="padding:8px 10px;color:#cbd5e1;">Hafif negatif ilişki. Dönem başı not ortalaması görece düşük olan öğrencilerin sınav stresine ve tükenmişlik riskine daha açık olduğu gözlemlenmektedir.</td>
</tr>
<tr style="border-bottom:1px solid rgba(255,255,255,0.05);">
<td style="padding:8px 10px;font-weight:600;color:#e2e8f0;">Weekly_GenAI_Hours</td>
<td style="padding:8px 10px;color:#818cf8;">float64 (Sayısal / Saat/Hafta) -> Sayısal (Standartlaştırılmış Saat)</td>
<td style="padding:8px 10px;">Haftalık üretken yapay zeka araçları kullanım süresi (0.0 - 40.0 saat arası).</td>
<td style="padding:8px 10px;color:#f87171;font-weight:600;">En Yüksek Pozitif Korelasyon (0.47). YZ araçlarında geçirilen sürenin artması, artan stres ve tükenmişlik riskiyle doğrudan paraleldir.</td>
</tr>
<tr style="border-bottom:1px solid rgba(255,255,255,0.05);">
<td style="padding:8px 10px;font-weight:600;color:#e2e8f0;">Primary_Use_Case</td>
<td style="padding:8px 10px;color:#818cf8;">object (Metin) -> Kategorik (One-Hot - Sayısal)</td>
<td style="padding:8px 10px;">YZ araçlarının öncelikli kullanım amacı (Debugging/Troubleshooting, Ideation, Summarizing_Reading vb.).</td>
<td style="padding:8px 10px;color:#cbd5e1;">Kullanım amaçlarının risk sınıfları üzerindeki oranları dengelidir (%25-%28 bandı). "Debugging/Troubleshooting" en yüksek hacimli gruptur.</td>
</tr>
<tr style="border-bottom:1px solid rgba(255,255,255,0.05);">
<td style="padding:8px 10px;font-weight:600;color:#e2e8f0;">Prompt_Engineering_Skill</td>
<td style="padding:8px 10px;color:#818cf8;">object (Metin) -> Kategorik (Ordinal - Sayısal)</td>
<td style="padding:8px 10px;">Öğrencinin prompt (komut) yazma beceri seviyesi (Beginner, Intermediate, Advanced).</td>
<td style="padding:8px 10px;color:#cbd5e1;">Prompt becerisi "Beginner" olan ve ücretli aboneliği bulunmayan öğrencilerde stres ve tükenmişlik riski görece daha yüksek seyretmektedir.</td>
</tr>
<tr style="border-bottom:1px solid rgba(255,255,255,0.05);">
<td style="padding:8px 10px;font-weight:600;color:#e2e8f0;">Tool_Diversity</td>
<td style="padding:8px 10px;color:#818cf8;">float64 (Sayısal / Adet) -> Sayısal (Standartlaştırılmış Adet)</td>
<td style="padding:8px 10px;">Öğrencinin kullandığı farklı üretken YZ araçlarının sayısı (1.0 - 5.0 adet arası).</td>
<td style="padding:8px 10px;color:#cbd5e1;">Hafif pozitif ilişki. Çok sayıda farklı aracın aynı anda kullanılması bilişsel adaptasyon yükünü artırarak tükenmişliği tetikleyebilir.</td>
</tr>
<tr style="border-bottom:1px solid rgba(255,255,255,0.05);">
<td style="padding:8px 10px;font-weight:600;color:#e2e8f0;">Paid_Subscription</td>
<td style="padding:8px 10px;color:#818cf8;">bool (Mantıksal) -> Sayısal (Binary 0/1)</td>
<td style="padding:8px 10px;">Öğrencinin herhangi bir ücretli YZ aracına aboneliği olup olmadığı (True / False).</td>
<td style="padding:8px 10px;color:#cbd5e1;">Aboneliği olmayan başlangıç düzeyindeki öğrencilerde tükenmişlik riski hafif düzeyde daha yaygındır.</td>
</tr>
<tr style="border-bottom:1px solid rgba(255,255,255,0.05);">
<td style="padding:8px 10px;font-weight:600;color:#e2e8f0;">Traditional_Study_Hours</td>
<td style="padding:8px 10px;color:#818cf8;">float64 (Sayısal / Saat/Hafta) -> Sayısal (Standartlaştırılmış Saat)</td>
<td style="padding:8px 10px;">Kütüphane, kitap okuma, grup çalışması gibi geleneksel ders çalışma süresi (1.0 - 35.86 saat arası).</td>
<td style="padding:8px 10px;color:#34d399;font-weight:600;">Negatif Korelasyon (-0.14). Geleneksel yöntemlerle çalışmaya daha fazla zaman ayıran öğrencilerin tükenmişlik riski azalmaktadır.</td>
</tr>
<tr style="border-bottom:1px solid rgba(255,255,255,0.05);">
<td style="padding:8px 10px;font-weight:600;color:#e2e8f0;">Perceived_AI_Dependency</td>
<td style="padding:8px 10px;color:#818cf8;">float64 (Sayısal / Skor 1.0-10.0) -> Sayısal (Standartlaştırılmış Skor)</td>
<td style="padding:8px 10px;">Öğrencinin YZ araçlarına olan bağımlılığını hissetme derecesi (1.0 - 10.0 arası).</td>
<td style="padding:8px 10px;color:#f87171;font-weight:600;">Güçlü Pozitif Korelasyon (0.37). Kendini yapay zekaya bağımlı hisseden öğrencilerde tükenmişlik riski son derece belirgindir.</td>
</tr>
<tr style="border-bottom:1px solid rgba(255,255,255,0.05);">
<td style="padding:8px 10px;font-weight:600;color:#e2e8f0;">Institutional_Policy</td>
<td style="padding:8px 10px;color:#818cf8;">object (Metin) -> Kategorik (One-Hot - Sayısal)</td>
<td style="padding:8px 10px;">Okulun yapay zeka kullanım kuralları (Actively_Encouraged, Allowed_With_Citation, Strict_Ban).</td>
<td style="padding:8px 10px;color:#cbd5e1;">Yasalara/kısıtlamalara tabi olan öğrencilerde YZ bağımlılığı ve akademik kaygı örüntüleri dolaylı etkiler yaratabilir. Oransal dağılım homojendir.</td>
</tr>
<tr style="border-bottom:1px solid rgba(255,255,255,0.05);">
<td style="padding:8px 10px;font-weight:600;color:#e2e8f0;">Anxiety_Level_During_Exams</td>
<td style="padding:8px 10px;color:#818cf8;">float64 (Sayısal / Skor 1.0-10.0) -> Sayısal (Standartlaştırılmış Kaygı Skoru)</td>
<td style="padding:8px 10px;">Sınav dönemlerinde hissedilen stres ve kaygı derecesi (1.0 - 10.0 arası).</td>
<td style="padding:8px 10px;color:#fda4af;">Pozitif Korelasyon (0.16). Sınav kaygısı yüksek olan öğrencilerin High risk seviyesine girme ihtimali daha yüksektir.</td>
</tr>
<tr style="border-bottom:1px solid rgba(255,255,255,0.05);">
<td style="padding:8px 10px;font-weight:600;color:#ef4444;">Post_Semester_GPA</td>
<td style="padding:8px 10px;color:#ef4444;font-weight:600;">float64 (Sayısal / Not Ortalaması 0.00-4.00) -> Kaldırılacak (Veri Sızıntısı Engelleme)</td>
<td style="padding:8px 10px;">Dönem sonundaki başarı notu ortalaması (1.00 - 4.00 arası).</td>
<td style="padding:8px 10px;color:#ef4444;font-weight:600;">Kritik Veri Sızıntısı (Data Leakage)! Dönem sonuna ait olduğu ve dönem başındaki öngörü modelinde bilinemeyeceği için kesinlikle çıkarılmalıdır.</td>
</tr>
<tr style="border-bottom:1px solid rgba(255,255,255,0.05);">
<td style="padding:8px 10px;font-weight:600;color:#e2e8f0;">Skill_Retention_Score</td>
<td style="padding:8px 10px;color:#818cf8;">float64 (Sayısal / Skor 0.0-100.0) -> Sayısal (Standartlaştırılmış Skor)</td>
<td style="padding:8px 10px;">Akademik bilgilerin zihinde kalıcılık derecesi (10.78 - 100.0 arası).</td>
<td style="padding:8px 10px;color:#cbd5e1;">Hafif negatif ilişki. Hafıza kalıcılığı zayıf olan öğrencilerde tükenmişlik baskısı artabilir. Modellemede sızıntı takibi için yakından izlenecektir.</td>
</tr>
<tr style="border-bottom:1px solid rgba(255,255,255,0.05);">
<td style="padding:8px 10px;font-weight:600;color:#a5b4fc;">Burnout_Risk_Level</td>
<td style="padding:8px 10px;color:#a5b4fc;font-weight:600;">object (Metin) -> Sayısal (Ordinal Sınıf Etiketi 0/1/2) - HEDEF</td>
<td style="padding:8px 10px;">Tahmin edilmeye çalışılan akademik tükenmişlik risk düzeyi (Low, Medium, High).</td>
<td style="padding:8px 10px;color:#a5b4fc;font-weight:600;">Hedef Değişken (Target). Model bu sınıfları tahmin etmek üzere eğitilmektedir. Sınıf dağılımı dengelidir.</td>
</tr>
</tbody>
</table>
<div style="margin-top:24px;background:linear-gradient(135deg,#0a192f 0%,#0f2e5c 100%);padding:24px 28px;border-radius:14px;border:1px solid rgba(56,189,248,0.2);box-shadow:0 2px 12px rgba(0,0,0,0.3);">
<div style="display:flex;align-items:center;gap:10px;margin-bottom:14px;">
<div style="width:3px;height:20px;background:linear-gradient(180deg,#38bdf8,#0284c7);border-radius:2px;flex-shrink:0;"></div>
<h3 style="color:#bae6fd;font-size:1.05em;font-weight:700;margin:0;">Veri İçi Mantıksal Bağlamlar ve İş Odaklı Çıkarımlar</h3>
</div>
<div style="display:grid;grid-template-columns:repeat(auto-fit, minmax(250px, 1fr));gap:16px;">
<div>
<p style="color:#e0f2fe;font-size:0.88em;font-weight:600;margin:0 0 4px;">💻 STEM ve Kodlama Bağımlılığı</p>
<p style="color:#94a3b8;font-size:0.81em;line-height:1.5;margin:0;">STEM öğrencileri en kalabalık grubu oluşturmakla kalmayıp (%30.1), haftalık ortalama YZ kullanımında <strong>10.84 saat</strong> ile açık ara liderdir. Bu yoğun kullanım, <strong>3.79</strong> bağımlılık algısı skoru ve en yüksek yüksek-riskli tükenmişlik oranı (<strong>%30.0</strong>) ile doğrudan ilişkilidir. STEM öğrencilerinin YZ'yi en çok <strong>Debugging (%51.7)</strong> için kullanması, kod hatalarını çözmede YZ'ye olan bağımlılıklarını açıklamaktadır.</p>
</div>
<div>
<p style="color:#e0f2fe;font-size:0.88em;font-weight:600;margin:0 0 4px;">📝 Humanities ve Yazım Odaklılık</p>
<p style="color:#94a3b8;font-size:0.81em;line-height:1.5;margin:0;">Humanities öğrencileri YZ'yi çoğunlukla <strong>Copywriting/Drafting (%51.9)</strong> amacıyla kullanmaktadır. Bu grupta YZ kullanım süresi (7.11 saat/hafta) ve bağımlılık algısı (3.24) en düşük seviyededir. Buna paralel olarak, yüksek tükenmişlik riski de bu grupta en düşük düzeydedir (<strong>%20.7</strong>). Bu durum, metin yazma odaklı YZ kullanımının kodlama hatalarını giderme kadar yüksek stres ve yıpranma yaratmadığını gösterir.</p>
</div>
<div>
<p style="color:#e0f2fe;font-size:0.88em;font-weight:600;margin:0 0 4px;">📊 Business (Ideation) ve Medical (Summarizing)</p>
<p style="color:#94a3b8;font-size:0.81em;line-height:1.5;margin:0;">Kullanım amaçları akademik disiplinlerle son derece tutarlıdır: Business öğrencileri YZ'yi ezici bir oranla <strong>Ideation (%47.9)</strong> (fikir geliştirme) için kullanırken (yüksek risk %24.3), Medical öğrencileri tıp eğitiminin yoğun bilgi yükünü yönetmek için <strong>Summarizing/Reading (%47.9)</strong> amacıyla kullanmaktadır (yüksek risk %23.2).</p>
</div>
<div>
<p style="color:#e0f2fe;font-size:0.88em;font-weight:600;margin:0 0 4px;">🔄 Geleneksel Çalışma İkamesi</p>
<p style="color:#94a3b8;font-size:0.81em;line-height:1.5;margin:0;">Geleneksel çalışma saatlerinin STEM öğrencilerinde en düşük seviyede (11.00 saat) olması, YZ kullanımının geleneksel çalışmanın yerini almaya başladığını göstermektedir. Bu ikame ilişkisi, sınav anksiyetesini ve tükenmişliği artıran en büyük faktörlerden biridir. Akademik rehberlik birimleri için bu durum, YZ'yi bir ikame değil, destekleyici araç olarak konumlandırma stratejisinin önemini kanıtlamaktadır.</p>
</div>
</div>
</div>
</div>
```


---

### 3.2. Section 2: Data Understanding (EDA) Hücreleri

#### Hücre 1 (Code) - Genel Bilgiler
Yerleştirilecek alan: **Section 2.2**
```python
# 2.2 Genel Bilgi ve Yapısal Profilleme
print(f"Gözlem Sayısı (Satır): {df.shape[0]}")
print(f"Özellik Sayısı (Sütun): {df.shape[1]}")

# dtypes ve eksik değerleri içeren şık bir tablo hazırlayalım
info_df = pd.DataFrame({
    'Veri Tipi': df.dtypes,
    'Eksik Değer Sayısı': df.isnull().sum(),
    'Eksik Değer Oranı (%)': (df.isnull().sum() / len(df) * 100).round(2),
    'Benzersiz Değer Sayısı': df.nunique()
})

# display styling
styled_info = info_df.style.background_gradient(cmap='Blues', subset=['Eksik Değer Oranı (%)'])\
    .set_properties(**{'background-color': '#111827', 'color': '#f3f4f6', 'border-color': '#374151'})\
    .set_table_styles([{'selector': 'th', 'props': [('background-color': '#1f2937'), ('color': '#a78bfa'), ('font-weight', 'bold')]}])

display(styled_info)
display(df.describe().T.style.set_properties(**{'background-color': '#111827', 'color': '#f3f4f6'}))
```

#### Hücre 2 (Code) - Target Değişken Dağılımı
Yerleştirilecek alan: **Section 2.3**
```python
# 2.3 Target Değişken Dağılımı
TARGET = 'Burnout_Risk_Level'

# Hedef değişkeni kategorik yaparak sıralamayı (Low -> Medium -> High) olarak sabitliyoruz
df[TARGET] = pd.Categorical(df[TARGET], categories=['Low', 'Medium', 'High'], ordered=True)

target_counts = df[TARGET].value_counts(sort=False).reset_index()
target_counts.columns = ['Risk Seviyesi', 'Öğrenci Sayısı']

fig = make_subplots(rows=1, cols=2, specs=[[{"type": "bar"}, {"type": "pie"}]],
                    subplot_titles=("Öğrenci Sayısı Dağılımı", "Yüzdesel Dağılım"))

colors = {'Low': '#10b981', 'Medium': '#f59e0b', 'High': '#ef4444'}

# Bar chart
fig.add_trace(
    go.Bar(
        x=target_counts['Risk Seviyesi'], 
        y=target_counts['Öğrenci Sayısı'],
        marker_color=[colors[r] for r in target_counts['Risk Seviyesi']],
        text=target_counts['Öğrenci Sayısı'],
        textposition='auto',
        name='Sayı'
    ),
    row=1, col=1
)

# Pie chart
fig.add_trace(
    go.Pie(
        labels=target_counts['Risk Seviyesi'], 
        values=target_counts['Öğrenci Sayısı'],
        marker=dict(colors=[colors[l] for l in target_counts['Risk Seviyesi']]),
        hole=.4,
        name='Yüzde',
        sort=False
    ),
    row=1, col=2
)

fig.update_layout(
    title_text="Hedef Değişken (Burnout Risk Level) Dağılımı",
    template="plotly_dark",
    paper_bgcolor='#0f0e17',
    plot_bgcolor='#0f0e17',
    showlegend=True
)
fig.show()

try:
    fig.write_image(f"{FIGURES_DIR}eda_target_distribution.png", scale=2)
    print("Grafik kaydedildi: eda_target_distribution.png")
except Exception as e:
    print(f"Kaydetme başarısız: {e}")
```
*Not: Bu hücrenin hemen altına bir Markdown hücresi açıp analist yorumunu ekle (HTML formatında flat ve unindented):*
```html
<div style="background:linear-gradient(135deg,#0d1117 0%,#161b22 100%);padding:20px 24px;border-radius:12px;border:1px solid rgba(167,139,250,0.15);margin-top:12px;box-shadow:0 4px 20px rgba(0,0,0,0.3);">
<div style="display:flex;align-items:center;gap:8px;margin-bottom:10px;">
<div style="width:3px;height:18px;background:linear-gradient(180deg,#a78bfa,#7c3aed);border-radius:1.5px;flex-shrink:0;"></div>
<h4 style="color:#c7d2fe;font-size:0.95em;font-weight:700;margin:0;">Veri Analisti Yorumu</h4>
</div>
<p style="color:#cbd5e1;font-size:0.83em;line-height:1.5;margin:0;">
Hedef değişkenimiz olan <code>Burnout_Risk_Level</code> sınıflarının (Low, Medium, High) dağılımı incelendiğinde, dengeli bir yapı olduğu görülmektedir (Low: %32.7, Medium: %42.3, High: %25.0). Sınıf dengesizliği (class imbalance) derin bir problem olmamakla birlikte, yüksek riskli öğrencileri erken aşamada yakalamak akademik rehberlik açısından kritik öneme sahiptir. Bu doğrultuda, model başarısını değerlendirirken birincil hedeflerimiz <strong>Recall (High) &gt; 0.85</strong> ve genel dengeyi koruyan <strong>Macro F1-Score &gt; 0.80</strong> olacaktır.
</p>
</div>
```

#### Hücre 3 (Code) - Eksik Değer Dağılım Analizi
Yerleştirilecek alan: **Section 2.4**
```python
# 2.4 Eksik Değer Dağılım Analizi
missing_counts = df.isnull().sum()
missing_pct = (df.isnull().sum() / len(df) * 100).round(2)

missing_df = pd.DataFrame({
    'Değişken': df.columns,
    'Eksik Değer Sayısı': missing_counts,
    'Eksiklik Oranı (%)': missing_pct
}).sort_values(by='Eksik Değer Sayısı', ascending=False)

# Sadece eksik değer içeren özellikleri görselleştirelim
missing_df_plot = missing_df[missing_df['Eksik Değer Sayısı'] > 0]

fig = px.bar(
    missing_df_plot,
    x='Değişken',
    y='Eksiklik Oranı (%)',
    text='Eksik Değer Sayısı',
    title='Özelliklere Göre Eksik Değer Oranları (%) ve Sayıları (Tüm 50.000 Gözlem)',
    color='Eksiklik Oranı (%)',
    color_continuous_scale='Reds',
    labels={'Eksiklik Oranı (%)': 'Oran (%)', 'Değişken': 'Değişken Adı'}
)

fig.update_traces(textposition='outside')
fig.update_layout(
    template='plotly_dark',
    paper_bgcolor='#0f0e17',
    plot_bgcolor='#0f0e17',
    yaxis_range=[0, 10],
    height=500
)
fig.show()

try:
    fig.write_image(f"{FIGURES_DIR}eda_missing_value_heatmap.png", scale=2)
    print("Grafik kaydedildi: eda_missing_value_heatmap.png (Eksik değer bar grafiği olarak güncellendi)")
except Exception as e:
    print(e)
```
*Altına eklenecek analist yorumu:*
```html
<div style="background:linear-gradient(135deg,#0d1117 0%,#161b22 100%);padding:20px 24px;border-radius:12px;border:1px solid rgba(167,139,250,0.15);margin-top:12px;box-shadow:0 4px 20px rgba(0,0,0,0.3);">
<div style="display:flex;align-items:center;gap:8px;margin-bottom:10px;">
<div style="width:3px;height:18px;background:linear-gradient(180deg,#a78bfa,#7c3aed);border-radius:1.5px;flex-shrink:0;"></div>
<h4 style="color:#c7d2fe;font-size:0.95em;font-weight:700;margin:0;">Veri Analisti Yorumu</h4>
</div>
<p style="color:#cbd5e1;font-size:0.83em;line-height:1.5;margin:0;">
Eksik değerlerin dağılım analizi, veri setindeki 9 değişkenin tamamında (Pre_Semester_GPA, Post_Semester_GPA, Weekly_GenAI_Hours, Skill_Retention_Score, Tool_Diversity, Paid_Subscription, Traditional_Study_Hours, Perceived_AI_Dependency, Anxiety_Level_During_Exams) %1 ile %6 arasında değişen hafif eksiklikler olduğunu göstermektedir. Bu dağınık ve düşük oranlı eksik değer yapısı, veri toplama sürecindeki sistemsel kayıt kesintileri veya anket doldurmama eğilimi gibi <strong>rastgele hatalara (MCAR - Missing Completely at Random)</strong> işaret etmektedir. Bu durum veri setimizin temsil gücünü olumsuz etkilemez. Modelleme öncesi veri kaybını önlemek için satır silme yerine, sayısal değişkenler için <strong>Median Imputer</strong>, kategorik değişkenler için ise <strong>Mode Imputer</strong> yöntemleri uygulanacaktır.
</p>
</div>
```

#### Hücre 4 (Code) - Sayısal ve Sayım Değişkenleri Dağılımları
Yerleştirilecek alan: **Section 2.5**
```python
# 2.5 Sayısal ve Sayım Değişkenleri Dağılımları
TARGET = 'Burnout_Risk_Level'
num_cols = [
    'Pre_Semester_GPA', 'Weekly_GenAI_Hours', 'Traditional_Study_Hours', 
    'Skill_Retention_Score', 'Anxiety_Level_During_Exams', 
    'Perceived_AI_Dependency', 'Post_Semester_GPA'
]
colors = {'Low': '#10b981', 'Medium': '#f59e0b', 'High': '#ef4444'}

# 1. Sürekli Sayısal Değişkenler (Histogram + Box-Plot)
for col in num_cols:
    fig = make_subplots(rows=1, cols=2, subplot_titles=(f'{col} Dağılımı', f'{col} Kutu Grafiği (Risk Grubu Bazlı)'))
    
    # Histogram
    fig.add_trace(
        go.Histogram(x=df[col], marker_color='#a78bfa', name=col, nbinsx=30),
        row=1, col=1
    )
    
    # Box plot (Risk grubuna göre kırılımlı)
    for risk, color in colors.items():
        sub_df = df[df[TARGET] == risk]
        fig.add_trace(
            go.Box(y=sub_df[col], name=risk, marker_color=color, boxpoints='outliers'),
            row=1, col=2
        )
        
    fig.update_layout(
        title_text=f'{col} Değişken Analizi',
        template='plotly_dark',
        paper_bgcolor='#0f0e17',
        plot_bgcolor='#0f0e17',
        showlegend=False
    )
    fig.show()
    
    try:
        fig.write_image(f'{FIGURES_DIR}eda_numeric_{col}.png', scale=2)
    except Exception as e:
        pass

# 2. Sayım / Kesikli Değişken: Tool_Diversity
col = 'Tool_Diversity'
temp_df = df.groupby([col, TARGET]).size().reset_index(name='Öğrenci Sayısı')
fig = px.bar(
    temp_df,
    x=col,
    y='Öğrenci Sayısı',
    color=TARGET,
    color_discrete_map=colors,
    barmode='group',
    title=f'{col} Değişkeninin Tükenmişlik Riski Kırılımlı Dağılımı',
    labels={col: 'Kullanılan YZ Araç Sayısı', 'Öğrenci Sayısı': 'Öğrenci Sayısı', TARGET: 'Tükenmişlik Riski'}
)
fig.update_layout(
    template='plotly_dark',
    paper_bgcolor='#0f0e17',
    plot_bgcolor='#0f0e17'
)
fig.show()

try:
    fig.write_image(f'{FIGURES_DIR}eda_numeric_{col}.png', scale=2)
except Exception as e:
    pass
```
*Altına eklenecek analist yorumu:*
```html
<div style="background:linear-gradient(135deg,#0d1117 0%,#161b22 100%);padding:20px 24px;border-radius:12px;border:1px solid rgba(167,139,250,0.15);margin-top:12px;box-shadow:0 4px 20px rgba(0,0,0,0.3);">
<div style="display:flex;align-items:center;gap:8px;margin-bottom:10px;">
<div style="width:3px;height:18px;background:linear-gradient(180deg,#a78bfa,#7c3aed);border-radius:1.5px;flex-shrink:0;"></div>
<h4 style="color:#c7d2fe;font-size:0.95em;font-weight:700;margin:0;">Veri Analisti Yorumu (Tekil Değişkenlerin Hedef Değişken ile Olan İlişkisi)</h4>
</div>
<p style="color:#cbd5e1;font-size:0.83em;line-height:1.5;margin:0;">
Sayısal değişkenlerin hedef risk seviyeleri kırılımındaki kutu grafikleri (box-plot) ve dağılımları incelendiğinde son derece önemli çıkarımlar elde edilmiştir:
<br>1. <strong>Weekly_GenAI_Hours & Perceived_AI_Dependency:</strong> Haftalık YZ kullanım saatleri arttıkça ve bağımlılık algısı yükseldikçe, öğrencilerin ezici çoğunluğunun <strong>High</strong> risk grubunda toplandığı görülmektedir. Bu iki değişken en güçlü risk belirleyicileridir.
<br>2. <strong>Traditional_Study_Hours:</strong> Geleneksel yöntemlerle çalışmaya daha fazla zaman ayıran öğrencilerin tükenmişlik riskinin azaldığı (negatif ilişki) gözlenmektedir. Bu durum geleneksel ders çalışmanın koruyucu etkisini doğrular.
<br>3. <strong>Anxiety_Level_During_Exams:</strong> Sınav anksiyetesi yüksek olan öğrencilerin High risk seviyesine girme eğilimi çok daha yüksektir.
<br>4. <strong>Tool_Diversity:</strong> Çok sayıda farklı YZ aracının aynı anda kullanılması (4 veya 5 araç), box plot yerine bar grafiklerinden de görüleceği üzere bilişsel adaptasyon yükünü artırarak risk düzeyini hafifçe yukarı taşımaktadır.
<br>5. <strong>Pre_Semester_GPA & Skill_Retention_Score:</strong> Düşük başlangıç GPA'ine ve zayıf hafıza kalıcılık skoruna sahip öğrencilerde tükenmişlik riski daha yaygındır.
</p>
</div>
```

#### Hücre 5 (Code) - Kategorik Değişkenlerin Dağılımları
Yerleştirilecek alan: **Section 2.6**
```python
# 2.6 Kategorik Değişkenlerin Dağılımları
cat_cols = ['Major_Category', 'Year_of_Study', 'Primary_Use_Case', 'Prompt_Engineering_Skill', 'Paid_Subscription', 'Institutional_Policy']
colors = {'Low': '#10b981', 'Medium': '#f59e0b', 'High': '#ef4444'}

for col in cat_cols:
    temp_df = df.copy()
    if temp_df[col].dtype == 'bool':
        temp_df[col] = temp_df[col].map({True: 'Abonelik Var (True)', False: 'Abonelik Yok (False)'})
        
    plot_data = temp_df.groupby([col, TARGET]).size().reset_index(name='Öğrenci Sayısı')
    
    fig = px.bar(
        plot_data, 
        x=col, 
        y='Öğrenci Sayısı', 
        color=TARGET,
        color_discrete_map=colors,
        title=f"{col} Değişkeninin Tükenmişlik Riski Kırılımlı Dağılımı",
        barmode='group'
    )
    
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='#0f0e17',
        plot_bgcolor='#0f0e17',
        xaxis_title=col,
        yaxis_title="Öğrenci Sayısı"
    )
    fig.show()
    
    try:
        fig.write_image(f"{FIGURES_DIR}eda_categorical_{col}.png", scale=2)
    except Exception as e:
        pass
```
*Altına eklenecek analist yorumu:*
```html
<div style="background:linear-gradient(135deg,#0d1117 0%,#161b22 100%);padding:20px 24px;border-radius:12px;border:1px solid rgba(167,139,250,0.15);margin-top:12px;box-shadow:0 4px 20px rgba(0,0,0,0.3);">
<div style="display:flex;align-items:center;gap:8px;margin-bottom:10px;">
<div style="width:3px;height:18px;background:linear-gradient(180deg,#a78bfa,#7c3aed);border-radius:1.5px;flex-shrink:0;"></div>
<h4 style="color:#c7d2fe;font-size:0.95em;font-weight:700;margin:0;">Veri Analisti Yorumu (Kategorik Değişkenlerin Hedef Değişken ile Olan İlişkisi)</h4>
</div>
<p style="color:#cbd5e1;font-size:0.83em;line-height:1.5;margin:0;">
Kategorik değişkenlerin hedef değişken kırılımındaki dağılımları disiplinler arası önemli eğilimleri göstermektedir:
<br>1. <strong>Bölümler ve YZ Kullanım Amaçları:</strong> STEM öğrencileri en yoğun YZ kullanıcıları olup çoğunlukla <code>Debugging/Troubleshooting</code> (%51.7) amacıyla YZ'yi tercih etmektedir ve yüksek risk oranları %30 ile zirvededir. Buna karşın, Humanities öğrencileri YZ'yi <code>Copywriting/Drafting</code> (%51.9) amacıyla kullanmakta olup tükenmişlik riski en düşük gruptur (%20.7).
<br>2. <strong>Prompt Yetkinliği & Paid_Subscription:</strong> Prompt yazma becerisi başlangıç düzeyinde olan (Beginner) ve premium aboneliği bulunmayan öğrencilerde stres ve tükenmişlik riski hafif düzeyde daha yaygındır.
<br>3. <strong>Okul Politikaları:</strong> Yapay zekayı tamamen yasaklayan (Strict_Ban) veya serbest bırakan okul politikalarında tükenmişlik riskinin dağılımı homojendir, ancak yasakçı yaklaşımların anksiyete üzerindeki dolaylı etkileri incelenmelidir.
</p>
</div>
```

#### Hücre 6 (Code) - Korelasyon Matrisi (Tüm 15 Değişken)
Yerleştirilecek alan: **Section 2.7**
```python
# 2.7 Korelasyon Matrisi (Tüm 15 Değişken)
TARGET = 'Burnout_Risk_Level'

# Korelasyon matrisinde Student_ID dışındaki tüm 15 değişkeni göstermek için kopyalıyoruz ve encode ediyoruz
corr_df = df.copy().drop('Student_ID', axis=1, errors='ignore')

# Ordinal değişkenleri map ediyoruz
corr_df[TARGET] = corr_df[TARGET].map({'Low': 0, 'Medium': 1, 'High': 2})
corr_df['Prompt_Engineering_Skill'] = corr_df['Prompt_Engineering_Skill'].map({'Beginner': 0, 'Intermediate': 1, 'Advanced': 2})
corr_df['Year_of_Study'] = corr_df['Year_of_Study'].map({'Freshman': 0, 'Sophomore': 1, 'Junior': 2, 'Senior': 3, 'Graduate': 4})
corr_df['Paid_Subscription'] = corr_df['Paid_Subscription'].astype(float)

# Nominal değişkenleri label-encode ediyoruz
for col in ['Major_Category', 'Primary_Use_Case', 'Institutional_Policy']:
    corr_df[col] = corr_df[col].astype('category').cat.codes

corr_matrix = corr_df.corr()

fig = px.imshow(
    corr_matrix,
    text_auto='.2f',
    color_continuous_scale='Viridis',
    labels=dict(color='Korelasyon'),
    width=900,
    height=800
)

fig.update_layout(
    title='Sayısal, Kategorik ve Hedef Değişken Genel Korelasyon Matrisi',
    template='plotly_dark',
    paper_bgcolor='#0f0e17',
    plot_bgcolor='#0f0e17'
)
fig.show()

try:
    fig.write_image(f'{FIGURES_DIR}eda_correlation_matrix.png', scale=2)
    print('Grafik kaydedildi: eda_correlation_matrix.png')
except Exception as e:
    print(e)
```
*Altına eklenecek analist yorumu:*
```html
<div style="background:linear-gradient(135deg,#0d1117 0%,#161b22 100%);padding:20px 24px;border-radius:12px;border:1px solid rgba(167,139,250,0.15);margin-top:12px;box-shadow:0 4px 20px rgba(0,0,0,0.3);">
<div style="display:flex;align-items:center;gap:8px;margin-bottom:10px;">
<div style="width:3px;height:18px;background:linear-gradient(180deg,#a78bfa,#7c3aed);border-radius:1.5px;flex-shrink:0;"></div>
<h4 style="color:#c7d2fe;font-size:0.95em;font-weight:700;margin:0;">Veri Analisti Yorumu (Korelasyon Matrisi Çıkarımları)</h4>
</div>
<p style="color:#cbd5e1;font-size:0.83em;line-height:1.5;margin:0;">
Tüm 15 değişkeni (Student_ID hariç) kapsayan genişletilmiş korelasyon matrisi, doğrusal ve sıra bazlı ilişkileri netleştirmektedir:
<br>1. <strong>Hedef Değişkenle En Güçlü Pozitif Korelasyonlar:</strong> <code>Weekly_GenAI_Hours</code> (0.47) ve <code>Perceived_AI_Dependency</code> (0.37) değişkenlerindedir. Bu durum, fiili kullanım ve bağımlılık hissinin risk düzeyinin en birincil doğrusal öngörücüleri olduğunu onaylar.
<br>2. <strong>Negatif Korelasyonlar:</strong> Geleneksel çalışma saatleri (Traditional_Study_Hours, -0.14) ve Pre_Semester_GPA (-0.10) ile risk arasında negatif korelasyon vardır; yani bu iki faktör koruyucu etki göstermektedir.
<br>3. <strong>Disiplin ve Politika İlişkileri:</strong> Nominal değişkenlerin (Major_Category, Primary_Use_Case) katsayılarının düşüklüğü, bu değişkenlerin etkilerinin doğrusal olmadığını ve modelleme aşamasında ağaç tabanlı veya doğrusal olmayan algoritmaların (örn. LightGBM, Random Forest, XGBoost) tercih edilmesi gerektiğini göstermektedir.
</p>
</div>
```

#### Hücre 7 (Code) - İkili Değişken Analizleri (Bivariate Analysis)
Yerleştirilecek alan: **Section 2.8**
```python
# 2.8 İkili Değişken Analizleri (Bivariate Analysis)
colors = {'Low': '#10b981', 'Medium': '#f59e0b', 'High': '#ef4444'}

# 1. Weekly_GenAI_Hours vs Traditional_Study_Hours (İkame Etkisi)
fig = px.scatter(
    df,
    x='Weekly_GenAI_Hours',
    y='Traditional_Study_Hours',
    color=TARGET,
    color_discrete_map=colors,
    opacity=0.6,
    title='Haftalık YZ Kullanım Saati vs Geleneksel Çalışma Saati (İkame Etkisi)',
    labels={'Weekly_GenAI_Hours': 'Haftalık YZ Kullanım Saati', 'Traditional_Study_Hours': 'Geleneksel Çalışma Saati', TARGET: 'Tükenmişlik Riski'}
)
fig.update_layout(template='plotly_dark', paper_bgcolor='#0f0e17', plot_bgcolor='#0f0e17')
fig.show()
try:
    fig.write_image(f'{FIGURES_DIR}eda_bivariate_genai_vs_traditional.png', scale=2)
except Exception as e:
    pass

# 2. Weekly_GenAI_Hours vs Skill_Retention_Score (Öğrenme Kaybı)
fig = px.scatter(
    df,
    x='Weekly_GenAI_Hours',
    y='Skill_Retention_Score',
    color=TARGET,
    color_discrete_map=colors,
    opacity=0.6,
    title='Haftalık YZ Kullanım Saati vs Beceri Kalıcılık Skoru (Öğrenme Kaybı)',
    labels={'Weekly_GenAI_Hours': 'Haftalık YZ Kullanım Saati', 'Skill_Retention_Score': 'Beceri Kalıcılık Skoru', TARGET: 'Tükenmişlik Riski'}
)
fig.update_layout(template='plotly_dark', paper_bgcolor='#0f0e17', plot_bgcolor='#0f0e17')
fig.show()
try:
    fig.write_image(f'{FIGURES_DIR}eda_bivariate_genai_vs_retention.png', scale=2)
except Exception as e:
    pass

# 3. Weekly_GenAI_Hours vs Perceived_AI_Dependency (Ortalama Bağımlılık Gelişimi)
avg_dep = df.groupby(['Perceived_AI_Dependency', TARGET], observed=True)['Weekly_GenAI_Hours'].mean().reset_index()
fig = px.line(
    avg_dep,
    x='Perceived_AI_Dependency',
    y='Weekly_GenAI_Hours',
    color=TARGET,
    color_discrete_map=colors,
    markers=True,
    title='Algılanan YZ Bağımlılık Skoru vs Ortalama Haftalık YZ Kullanım Saati',
    labels={'Weekly_GenAI_Hours': 'Ortalama Haftalık YZ Saati', 'Perceived_AI_Dependency': 'Algılanan YZ Bağımlılık Skoru (1-10)', TARGET: 'Tükenmişlik Riski'}
)
fig.update_layout(template='plotly_dark', paper_bgcolor='#0f0e17', plot_bgcolor='#0f0e17')
fig.show()
try:
    fig.write_image(f'{FIGURES_DIR}eda_bivariate_genai_vs_dependency.png', scale=2)
except Exception as e:
    pass
```
*Altına eklenecek analist yorumu:*
```html
<div style="background:linear-gradient(135deg,#0d1117 0%,#161b22 100%);padding:20px 24px;border-radius:12px;border:1px solid rgba(167,139,250,0.15);margin-top:12px;box-shadow:0 4px 20px rgba(0,0,0,0.3);">
<div style="display:flex;align-items:center;gap:8px;margin-bottom:10px;">
<div style="width:3px;height:18px;background:linear-gradient(180deg,#a78bfa,#7c3aed);border-radius:1.5px;flex-shrink:0;"></div>
<h4 style="color:#c7d2fe;font-size:0.95em;font-weight:700;margin:0;">Veri Analisti Yorumu (İkili Değişken Analizleri ve Hedef İlişkisi)</h4>
</div>
<p style="color:#cbd5e1;font-size:0.83em;line-height:1.5;margin:0;">
İkili değişken analizleri (bivariate analysis) sonucunda ortaya koyulan kritik akademik ve davranışsal kesişimler şunlardır:
<br>1. <strong>YZ Kullanımı ve Geleneksel Çalışma (İkame Etkisi):</strong> Geleneksel çalışma saatleri ile YZ kullanım saatleri arasındaki negatif ilişki (-0.16) ve scatter plot grafiği incelendiğinde; haftalık YZ kullanım saati 15 saati geçen öğrencilerin geleneksel çalışma sürelerinin dramatik olarak azaldığı ve bu bölgedeki öğrencilerin neredeyse tamamının <strong>High</strong> risk sınıfına girdiğini görmekteyiz. YZ, çalışmayı desteklemekten ziyade ikame ettiğinde tükenmişlik tetiklenmektedir.
<br>2. <strong>YZ Kullanımı ve Beceri Kalıcılığı (Öğrenme Kaybı):</strong> <code>Weekly_GenAI_Hours</code> ve <code>Skill_Retention_Score</code> arasındaki negatif ilişki (-0.12) öğrenme süreçlerine dair alarm vermektedir. Haftalık 20 saatin üzerinde YZ kullanan öğrencilerin beceri kalıcılık skorlarının düştüğü ve bu bölgede High tükenmişliğin yoğunlaştığı görülmektedir. Aşırı kullanım, bilginin derinlemesine işlenmesini engellemektedir.
<br>3. <strong>Kullanım ve Bağımlılık Algısı:</strong> Haftalık YZ saati arttıkça bağımlılık algısının (1-10) doğrusal olarak arttığı (korelasyon: 0.67) ve bağımlılık algısı 7'yi geçen öğrencilerin ezici bir çoğunluğunun High riskli olduğu net bir şekilde doğrulanmıştır.
</p>
</div>
```

#### Hücre 8 (Code) - Aykırı Değer Analizi (IQR Yöntemi)
Yerleştirilecek alan: **Section 2.9**
```python
# 2.9 Aykırı Değer Analizi (Tukey IQR Yöntemi)
outlier_summary = []
num_cols = [
    'Pre_Semester_GPA', 'Weekly_GenAI_Hours', 'Traditional_Study_Hours', 
    'Skill_Retention_Score', 'Anxiety_Level_During_Exams', 
    'Tool_Diversity', 'Perceived_AI_Dependency', 'Post_Semester_GPA'
]

for col in num_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    outlier_summary.append({
        'Değişken': col,
        'Alt Sınır': round(lower_bound, 3),
        'Üst Sınır': round(upper_bound, 3),
        'Aykırı Değer Sayısı': len(outliers),
        'Aykırı Değer Oranı (%)': round(len(outliers) / len(df) * 100, 3)
    })

outlier_df = pd.DataFrame(outlier_summary)
display(outlier_df.style.set_properties(**{'background-color': '#111827', 'color': '#f3f4f6', 'border-color': '#374151'})\
    .set_table_styles([{'selector': 'th', 'props': [('background-color', '#1f2937'), ('color', '#a78bfa'), ('font-weight', 'bold')]}]))
print('Aykırı değer tablosu oluşturuldu.')
```
*Altına eklenecek analist yorumu:*
```html
<div style="background:linear-gradient(135deg,#0d1117 0%,#161b22 100%);padding:20px 24px;border-radius:12px;border:1px solid rgba(167,139,250,0.15);margin-top:12px;box-shadow:0 4px 20px rgba(0,0,0,0.3);">
<div style="display:flex;align-items:center;gap:8px;margin-bottom:10px;">
<div style="width:3px;height:18px;background:linear-gradient(180deg,#a78bfa,#7c3aed);border-radius:1.5px;flex-shrink:0;"></div>
<h4 style="color:#c7d2fe;font-size:0.95em;font-weight:700;margin:0;">Veri Analisti Yorumu</h4>
</div>
<p style="color:#cbd5e1;font-size:0.83em;line-height:1.5;margin:0;">
Tukey (IQR) yöntemiyle yapılan aykırı değer analizi, veri setindeki uç değerlerin oranının son derece düşük olduğunu göstermektedir (tüm sayısal sütunlar için oran %0.1'in altındadır). Bu durum veri toplama kalitesinin son derece yüksek olduğunu ve veride gürültü/bozuk veri bulunmadığını kanıtlar. Box-plot grafiklerinde de görülen bu uç değerler, veri hatası değil, öğrencilerin doğal davranışsal uç varyasyonlarıdır (örneğin haftada 40 saat YZ kullanan veya 35 saat geleneksel çalışan öğrenciler). Modelleme aşamasında (Faz 3) bu uç değerlerin mesafe bazlı algoritmaları (KNN, SVM vb.) olumsuz etkilemesini önlemek adına <strong>RobustScaler</strong> ölçeklendirme yöntemi veya <strong>clipping (kırpma)</strong> tekniği uygulanacaktır.
</p>
</div>
```

#### Hücre 9 (Markdown) - Veri Sızıntısı Değerlendirmesi
Yerleştirilecek alan: **Section 2.10**
```html
<div style="background:linear-gradient(135deg,#09090f 0%,#0e0e1c 100%);padding:36px 40px;border-radius:20px;margin-bottom:32px;border:1px solid rgba(255,255,255,0.07);box-shadow:0 4px 28px rgba(0,0,0,0.45);">
<div style="display:flex;align-items:center;gap:12px;margin-bottom:24px;">
<div style="width:4px;height:30px;background:linear-gradient(180deg,#ef4444,#dc2626);border-radius:2px;flex-shrink:0;"></div>
<h2 style="color:#f87171;font-size:1.25em;font-weight:700;margin:0;letter-spacing:0.4px;">Faz 2: Veri Sızıntısı (Data Leakage) Değerlendirmesi</h2>
</div>
<div style="display:flex-direction:column;gap:20px;">
<div style="background:rgba(239,68,68,0.08);border:1px solid rgba(239,68,68,0.2);border-radius:10px;padding:16px 20px;">
<p style="color:#f87171;font-size:0.95em;font-weight:700;margin:0 0 6px;text-transform:uppercase;letter-spacing:0.5px;">Veri Sızıntısı Nedir ve Neye Sebep Olur?</p>
<p style="color:#fecdd3;font-size:0.83em;line-height:1.5;margin:0;">
Veri sızıntısı (data leakage), eğitim veri setinde modelin öğrenmemesi gereken, geleceğe dair veya tahmin anında (dönem başında) elde edilemeyecek bilgilerin yer alması durumudur. Bu durum, modelin doğrulama aşamasında yapay olarak **%100'e yakın gerçekçi olmayan bir başarı** göstermesine sebob olurken, canlı sisteme alındığında ise **tamamen başarısız olmasına** (aşırı ezberleme/overfitting) yol açar.
</p>
</div>
<div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.08);border-radius:10px;padding:16px 20px;">
<p style="color:#e2e8f0;font-size:0.9em;font-weight:600;margin:0 0 8px;">📊 Örnek Senaryo ve Somut Etkileri</p>
<p style="color:#cbd5e1;font-size:0.82em;line-height:1.5;margin:0;">
<code>Post_Semester_GPA</code> (Dönem Sonu Not Ortalaması) dönem sonunda belirlenir. Bizim amacımız ise dönem başında öğrencinin tükenmişlik riskini tahmin etmektir. Eğer <code>Post_Semester_GPA</code> model girdisi olarak tutulursa, model bu not ile <code>Pre_Semester_GPA</code> arasındaki düşüşleri görerek tükenmişlik riskini doğrudan sızdırılan bu gelecek bilgisinden ezberler. Ancak dönem başında elimizde dönem sonu notu olmayacağı için model canlıda tahmin yapamaz.
</p>
</div>
<div style="background:rgba(16,185,129,0.08);border:1px solid rgba(16,185,129,0.2);border-radius:10px;padding:16px 20px;">
<p style="color:#34d399;font-size:0.95em;font-weight:700;margin:0 0 6px;text-transform:uppercase;letter-spacing:0.5px;">Alınan Net Kararlar ve Aksiyonlar</p>
<ul style="color:#a7f3d0;font-size:0.82em;line-height:1.5;margin:0;padding-left:20px;">
<li><strong>Post_Semester_GPA Kesinlikle Çıkarıldı:</strong> Veri sızıntısını tamamen kesmek için <code>Post_Semester_GPA</code> değişkeni model girdilerinden çıkarılmıştır.</li>
<li><strong>GPA_Change Özelliği İptal Edildi:</strong> İçinde <code>Post_Semester_GPA</code> barındıran <code>GPA_Change</code> (Not Değişimi) feature engineering önerisi veri sızıntısı nedeniyle iptal edilmiştir.</li>
<li><strong>Skill_Retention_Score Yakın Takip Altında:</strong> Dönem içi bir sınav skoru olarak kabul edilmekle birlikte, modellemede sızıntı şüphesi yaratıp yaratmadığı feature importance katsayıları üzerinden izlenecektir.</li>
</ul>
</div>
</div>
</div>
```

#### Hücre 10 (Markdown) - EDA Sonu Raporu / Bulgular Raporu
Yerleştirilecek alan: **Section 2.11**
```html
<div style="background:linear-gradient(135deg,#09090f 0%,#0e0e1c 100%);padding:36px 40px;border-radius:20px;margin-bottom:32px;border:1px solid rgba(255,255,255,0.07);box-shadow:0 4px 28px rgba(0,0,0,0.45);">
<div style="display:flex;align-items:center;gap:12px;margin-bottom:24px;">
<div style="width:4px;height:30px;background:linear-gradient(180deg,#10b981,#059669);border-radius:2px;flex-shrink:0;"></div>
<h2 style="color:#a7f3d0;font-size:1.25em;font-weight:700;margin:0;letter-spacing:0.4px;">Faz 2: EDA Bulguları ve Süreç Değerlendirme Raporu</h2>
</div>
<div style="display:flex;flex-direction:column;gap:20px;">
<p style="color:#cbd5e1;font-size:0.88em;line-height:1.6;margin:0;">
50.000 gözlem içeren öğrenci veri setimiz üzerinde gerçekleştirilen Keşifsel Veri Analizi (EDA) süreci başarıyla tamamlanmıştır. Bulgular, veri kalitesi sorunları, sızıntı analizleri ve bir sonraki aşama (Faz 3 - Veri Hazırlama) için kararlaştırılan yönergeler bu raporda özetlenmiştir.
</p>
<div style="display:grid;grid-template-columns:repeat(auto-fit, minmax(280px, 1fr));gap:20px;">
<div style="background:linear-gradient(135deg,#0d1117 0%,#161b22 100%);padding:20px 24px;border-radius:12px;border:1px solid rgba(16,185,129,0.18);box-shadow:0 2px 12px rgba(0,0,0,0.3);">
<h3 style="color:#34d399;font-size:1.0em;font-weight:700;margin:0 0 12px;">📊 Temel EDA Çıktıları</h3>
<ul style="color:#94a3b8;font-size:0.82em;line-height:1.6;margin:0;padding-left:18px;">
<li><strong>Dengeli Dağılım:</strong> Hedef sınıfımız Low-Medium-High oranları dengeli olup, sınıf dengesizliği riski düşüktür.</li>
<li><strong>En Güçlü Belirleyiciler:</strong> Haftalık GenAI kullanım saati (0.47 korelasyon) ve bağımlılık algısı (0.37) tükenmişliği artıran en güçlü faktörlerdir.</li>
<li><strong>Geleneksel Çalışmanın Koruyucu Etkisi:</strong> Geleneksel ders çalışma saatleri (-0.14) tükenmişlik riskini azaltmaktadır.</li>
<li><strong>İkame ve Öğrenme Kaybı:</strong> Haftalık 15 saatin üzerinde YZ kullanımı, geleneksel çalışma saatlerini ikame etmekte ve 20 saatin üzerindeki aşırı kullanım, beceri kalıcılık skorunu (-0.12) düşürmektedir.</li>
</ul>
</div>
<div style="background:linear-gradient(135deg,#0d1117 0%,#161b22 100%);padding:20px 24px;border-radius:12px;border:1px solid rgba(99,102,241,0.18);box-shadow:0 2px 12px rgba(0,0,0,0.3);">
<h3 style="color:#818cf8;font-size:1.0em;font-weight:700;margin:0 0 12px;">🛠️ Veri Hazırlama (Faz 3) Yönergeleri</h3>
<ul style="color:#94a3b8;font-size:0.82em;line-height:1.6;margin:0;padding-left:18px;">
<li><strong>Eksik Değer Yönetimi:</strong> Sayısal değişkenler için <code>Median Imputer</code>, kategorik değişkenler için <code>Mode Imputer</code> kullanılacaktır (9 değişken etkilendi).</li>
<li><strong>Encoding Stratejisi:</strong> <code>Prompt_Engineering_Skill</code> ve <code>Year_of_Study</code> için <code>OrdinalEncoder</code>; nominal kategorikler (Major, Use Case, Policy) için <code>OneHotEncoder</code> uygulanacaktır.</li>
<li><strong>Ölçeklendirme:</strong> Mesafe hassasiyetini ve aykırı değerlerin etkisini yönetmek için tüm sayısallara <code>RobustScaler</code> veya <code>StandardScaler</code> uygulanacaktır.</li>
<li><strong>Sızıntı Önleme:</strong> Pipeline öncesi <code>Post_Semester_GPA</code> ve <code>Student_ID</code> sütunları veri setinden kesinlikle düşürülecektir (drop).</li>
</ul>
</div>
</div>
</div>
</div>
```

---

## 4. Entegrasyon Doğrulama Adımları
1. Kod bloklarını notebook'a yazdıktan sonra Jupyter kernel'ı restart edilerek tüm hücreler yukarıdan aşağıya çalıştırılmalıdır (`Restart & Run All`).
2. `figures/` klasörünün oluşturulduğundan ve tüm `eda_*.png` dosyalarının doğru kaydedildiğinden emin olunmalıdır.
3. Notebook'un kaydedildiğinden emin olunmalıdır.
