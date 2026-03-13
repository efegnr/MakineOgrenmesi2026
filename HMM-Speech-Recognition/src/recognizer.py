from hmmlearn import hmm
import numpy as np

# ============================================================
# 1) "EV" kelimesi için HMM modeli (2 durum, 3 gözlem sembolü)
# ============================================================
model_ev = hmm.CategoricalHMM(n_components=2, n_iter=100)
model_ev.startprob_ = np.array([1.0, 0.0])

# Geçiş (transition) matrisi  –  A
# Durum 0 -> Durum 0 : 0.7,  Durum 0 -> Durum 1 : 0.3
# Durum 1 -> Durum 0 : 0.4,  Durum 1 -> Durum 1 : 0.6
model_ev.transmat_ = np.array([
    [0.7, 0.3],
    [0.4, 0.6]
])

# Emisyon (emission) matrisi  –  B
# 3 farklı gözlem sembolü olduğunu varsayıyoruz (indeks 0, 1, 2)
# Durum 0 -> sembol 0: 0.5,  sembol 1: 0.4,  sembol 2: 0.1
# Durum 1 -> sembol 0: 0.1,  sembol 1: 0.3,  sembol 2: 0.6
model_ev.emissionprob_ = np.array([
    [0.5, 0.4, 0.1],
    [0.1, 0.3, 0.6]
])

# ============================================================
# 2) "OKUL" kelimesi için HMM modeli (2 durum, 3 gözlem sembolü)
# ============================================================
model_okul = hmm.CategoricalHMM(n_components=2, n_iter=100)
model_okul.startprob_ = np.array([1.0, 0.0])

# Geçiş (transition) matrisi  –  A
# Durum 0 -> Durum 0 : 0.6,  Durum 0 -> Durum 1 : 0.4
# Durum 1 -> Durum 0 : 0.3,  Durum 1 -> Durum 1 : 0.7
model_okul.transmat_ = np.array([
    [0.6, 0.4],
    [0.3, 0.7]
])

# Emisyon (emission) matrisi  –  B
# Durum 0 -> sembol 0: 0.1,  sembol 1: 0.2,  sembol 2: 0.7
# Durum 1 -> sembol 0: 0.6,  sembol 1: 0.3,  sembol 2: 0.1
model_okul.emissionprob_ = np.array([
    [0.1, 0.2, 0.7],
    [0.6, 0.3, 0.1]
])


# ============================================================
# 3) Her kelime için temsili eğitim verisi (Gözlem dizileri)
# ============================================================
# "EV" kelimesine ait örnek gözlem dizileri
train_ev = [
    np.array([[0, 1, 0, 1]]).T,
    np.array([[0, 0, 1, 1]]).T,
    np.array([[1, 0, 0, 1]]).T,
]

# "OKUL" kelimesine ait örnek gözlem dizileri
train_okul = [
    np.array([[2, 2, 1, 0]]).T,
    np.array([[2, 1, 2, 0]]).T,
    np.array([[2, 2, 0, 0]]).T,
]


# ============================================================
# 4) Sınıflandırma fonksiyonu
#    Dışarıdan gelen yeni bir gözlem dizisinin hangi modelde
#    daha yüksek Log-Likelihood değeri verdiğini hesaplar.
# ============================================================
def classify(observation_sequence):
    """
    Verilen gözlem dizisini (observation_sequence) her iki HMM
    modeline de göndererek Log-Likelihood skorlarını karşılaştırır.

    Parametreler
    ----------
    observation_sequence : numpy.ndarray
        Gözlem indekslerinden oluşan sütun vektörü.  Örn: np.array([[0,1,1]]).T

    Dönüş
    ------
    str
        "EV" veya "OKUL" — hangi modelin skoru daha yüksekse o kelime döner.
    """
    score_ev = model_ev.score(observation_sequence)
    score_okul = model_okul.score(observation_sequence)

    print(f"EV Modeli Puanı: {score_ev}")
    print(f"OKUL Modeli Puanı: {score_okul}")

    if score_ev >= score_okul:
        print("Sonuç: Bu ses kaydı 'EV' kelimesine daha yakın.\n")
        return "EV"
    else:
        print("Sonuç: Bu ses kaydı 'OKUL' kelimesine daha yakın.\n")
        return "OKUL"


# ============================================================
# Ana çalıştırma bloğu
# ============================================================
if __name__ == "__main__":
    # --- Eğitim verilerinin skorları (bilgi amaçlı) ---
    print("=" * 50)
    print("Eğitim Verileri Skorları")
    print("=" * 50)

    for i, seq in enumerate(train_ev):
        s_ev = model_ev.score(seq)
        s_ok = model_okul.score(seq)
        print(f"  EV eğitim dizisi {i+1} -> EV skoru: {s_ev:.4f}, OKUL skoru: {s_ok:.4f}")

    for i, seq in enumerate(train_okul):
        s_ev = model_ev.score(seq)
        s_ok = model_okul.score(seq)
        print(f"  OKUL eğitim dizisi {i+1} -> EV skoru: {s_ev:.4f}, OKUL skoru: {s_ok:.4f}")

    # --- Test verisi (Yeni bir ses kaydı geldiğini varsayalım) ---
    print("\n" + "=" * 50)
    print("Test / Sınıflandırma")
    print("=" * 50)

    test_data = np.array([[0, 1, 1]]).T  # Gözlemlerin indexleri

    # Hangi model daha yüksek puan veriyor?
    result = classify(test_data)

    # Ek test örnekleri
    test_data_2 = np.array([[2, 2, 1, 0]]).T
    result_2 = classify(test_data_2)

    test_data_3 = np.array([[0, 0, 1]]).T
    result_3 = classify(test_data_3)
