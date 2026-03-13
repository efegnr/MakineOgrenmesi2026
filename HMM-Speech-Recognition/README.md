# HMM-Speech-Recognition

## Problem Tanımı
Bu proje, **Gizli Markov Modelleri (HMM)** kullanarak basit bir kelime tanıma (Speech Classifier) sistemi simüle etmektedir. İki farklı kelime — **"EV"** ve **"OKUL"** — için ayrı HMM modelleri tanımlanmış ve dışarıdan gelen yeni bir gözlem dizisinin hangi kelimeye ait olduğu **Log-Likelihood** karşılaştırmasıyla belirlenmektedir.

## Veri
Her kelime için elle tanımlanmış parametreler (başlangıç olasılıkları, geçiş matrisi, emisyon matrisi) ve temsili gözlem dizileri kullanılmıştır. 3 farklı gözlem sembolü (indeks 0, 1, 2) mevcuttur.

| Parametre | EV Modeli | OKUL Modeli |
|---|---|---|
| Durum Sayısı | 2 | 2 |
| Gözlem Sembol Sayısı | 3 | 3 |
| Başlangıç Olasılığı | [1.0, 0.0] | [1.0, 0.0] |

## Yöntem
1. **hmmlearn** kütüphanesi ile `MultinomialHMM` modelleri oluşturulmuştur.
2. Her model için geçiş (transition) ve emisyon (emission) matrisleri manuel olarak belirlenmiştir.
3. Yeni bir gözlem dizisi geldiğinde her iki modelin `score()` metodu ile **Log-Likelihood** değeri hesaplanır.
4. Hangi model daha yüksek skor veriyorsa gözlem dizisi o kelimeye ait kabul edilir.

## Sonuçlar
Program çalıştırıldığında test verileri üzerinde sınıflandırma yapılır ve her test dizisi için modellerin skorları ve tahmin sonucu yazdırılır.

## Analiz ve Yorumlama

**1. Ses verisindeki "gürültü" (noise), HMM modelindeki Emisyon Olasılıklarını nasıl etkiler?**

Gürültü, gözlem sembollerinin gerçek değerlerinden sapmasına yol açarak emisyon olasılık dağılımını bulanıklaştırır. Bu durum, normalde belirli bir duruma ait olan gözlemlerin farklı durumlara atanma olasılığını artırır ve modelin doğru durum-gözlem eşleşmesini yapmasını zorlaştırır. Sonuç olarak Log-Likelihood skorları düşer ve sınıflandırma doğruluğu azalır.

**2. Gerçek bir sistemde binlerce kelime olduğunu düşünürsek, Viterbi yerine neden daha karmaşık yapılar (Deep Learning gibi) tercih edilmeye başlanmıştır?**

HMM ve Viterbi algoritması her kelime için ayrı model gerektirdiğinden, binlerce kelimelik bir sözlükte model sayısı ve hesaplama maliyeti çok yükselir. Deep Learning modelleri (RNN, Transformer vb.) tek bir uçtan uca model ile tüm kelime dağarcığını öğrenebilir, bağlamsal bilgiyi daha iyi yakalayabilir ve ham ses sinyalinden doğrudan özellik çıkarabilir. Bu nedenle büyük ölçekli gerçek dünya uygulamalarında derin öğrenme yaklaşımları çok daha ölçeklenebilir ve başarılıdır.

## Yorum / Tartışma
- Manuel olarak belirlenen parametreler, gerçek ses verisi yerine kavramsal bir gösterim sunmaktadır.
- Daha büyük ve gerçek veri kümeleriyle modeller eğitilerek (Baum-Welch algoritması) doğruluk artırılabilir.
- Gözlem sembollerinin sayısı artırılarak veya sürekli dağılımlı HMM (`GaussianHMM`) kullanılarak model geliştirilebilir.

## Kurulum ve Çalıştırma

```bash
pip install -r requirements.txt
python src/recognizer.py
```

## Dosya Yapısı
```
HMM-Speech-Recognition/
├── data/                # Varsa örnek ses verileri veya değer tabloları
├── src/
│   └── recognizer.py    # Ana Python kodu
├── report/
│   └── cozum_anahtari.pdf  # El hesaplamaları ve analizler
├── requirements.txt     # Gerekli kütüphaneler
└── README.md            # Proje açıklaması
```
