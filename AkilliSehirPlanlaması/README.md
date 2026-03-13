# MLE ile Akıllı Şehir Trafik Planlaması

**YZM212 Makine Öğrenmesi — 2. Laboratuvar Ödevi**

---

## Problem Tanımı

Bir şehrin belirli bir kavşağından 1 dakikada geçen araç sayısı gözlemlenmiştir.
Bu gözlemlerden hareketle Poisson dağılımının parametre tahmini yapılarak trafik akışı modellenecek ve modelin aykırı değerlere karşı duyarlılığı incelenecektir.

---

## Veri

| Özellik | Değer |
|---|---|
| Gözlem sayısı (n) | 14 |
| Veri | `[12, 15, 10, 8, 14, 11, 13, 16, 9, 12, 11, 14, 10, 15]` |
| Birim | araç / dakika |
| Outlier senaryosu | Veri setine `200` eklenerek n = 15 yapılmıştır |

---

## Yöntem

Trafik sayımı verisi **Poisson dağılımı** ile modellendi.

**Analitik türetme (Bölüm 1):** Likelihood fonksiyonu log alınarak basitleştirildi, türevi sıfıra eşitlenerek kapalı form çözümüne ulaşıldı:

$$\hat{\lambda}_{MLE} = \frac{1}{n}\sum_{i=1}^{n} k_i = \bar{k}$$

**Sayısal optimizasyon (Bölüm 2):** `scipy.optimize.minimize` ile Negatif Log-Likelihood minimize edilerek λ tahmin edildi. Sonucun analitik çözümle örtüştüğü doğrulandı.

**Görselleştirme (Bölüm 3):** Bulunan λ ile Poisson PMF grafiği çizildi ve gerçek verinin histogramı üzerine eklendi.

**Outlier analizi (Bölüm 4):** Veri setine yanlışlıkla girilen 200 araçlık kayıt eklenerek λ tahminine etkisi incelendi.

---

## Sonuçlar

| Senaryo | n | Σk | λ̂ MLE |
|---|---|---|---|
| Temiz veri | 14 | 170 | **12.1429** |
| Outlier ekli (200) | 15 | 370 | **24.6667** |

Tek bir aykırı gözlem, λ tahminini yaklaşık **%103 artırmıştır.**

---

## Yorum ve Tartışma

- MLE tahmini doğrudan verilerin ortalaması olduğundan **aykırı değerlere karşı hassastır.**
- 200 araçlık tek hatalı kayıt, modelin trafik yoğunluğunu iki kattan fazla abartmasına yol açmıştır.
- Bu hatalı λ ile belediye, gereksiz yol genişletme ya da kavşak yenileme kararı alabilir; bu da büyük ölçekli altyapı yatırımlarının israfına neden olur.
- **Çözüm önerisi:** Model eğitiminden önce IQR veya Z-skoru yöntemiyle aykırı değer temizliği yapılmalıdır.

---

## Dosya Yapısı

```
AkilliSehirPlanlaması/
├── mle_trafik_analizi.ipynb   # Bölüm 2 ve 3 — Python kodu (Jupyter Notebook)
├── generate_report.py         # PDF rapor üretici script
├── MLE_Rapor.pdf              # Bölüm 1 ve 4 — Teorik türetme + Outlier analizi raporu
└── README.md                  # Bu dosya
```

---

## Gereksinimler

```
numpy
scipy
matplotlib
```
