import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import poisson
import scipy.optimize as opt

# ─── Veri ve Hesaplamalar ────────────────────────────────────────────────────
traffic_data    = np.array([12, 15, 10, 8, 14, 11, 13, 16, 9, 12, 11, 14, 10, 15])
traffic_outlier = np.append(traffic_data, 200)

def negative_log_likelihood(lam, data):
    l = lam[0]
    return len(data) * l - np.sum(data) * np.log(l)

lambda_mle     = opt.minimize(negative_log_likelihood, [1.0],
                               args=(traffic_data,),    bounds=[(0.001, None)]).x[0]
lambda_outlier = opt.minimize(negative_log_likelihood, [1.0],
                               args=(traffic_outlier,), bounds=[(0.001, None)]).x[0]

SUM_K = int(np.sum(traffic_data))
N     = len(traffic_data)

# ─── Yardımcı: bölüm başlığı şeridi ─────────────────────────────────────────
def section_header(ax, x0, y, width, title):
    ax.add_patch(patches.FancyBboxPatch(
        (x0, y - 0.28), width, 0.48,
        boxstyle="round,pad=0.06",
        facecolor="#1a3a5c", edgecolor="none", zorder=2,
        transform=ax.transData
    ))
    ax.text(x0 + width / 2, y, title,
            fontsize=13, fontweight="bold", color="white",
            ha="center", va="center", zorder=3)

# ─────────────────────────────────────────────────────────────────────────────
# SAYFA 1 — KAPAK
# ─────────────────────────────────────────────────────────────────────────────
def page_kapak(pdf):
    fig = plt.figure(figsize=(8.27, 11.69))
    ax  = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    # Arka plan
    ax.add_patch(patches.Rectangle((0, 0), 1, 1,
                                   facecolor="#f5f7fa", transform=ax.transAxes, zorder=0))
    # Üst bant
    ax.add_patch(patches.Rectangle((0, 0.86), 1, 0.14,
                                   facecolor="#1a3a5c", transform=ax.transAxes, zorder=1))
    ax.text(0.5, 0.935, "YZM212 — Makine Öğrenmesi Dersi",
            transform=ax.transAxes, fontsize=13, color="white",
            ha="center", va="center", fontweight="bold")
    ax.text(0.5, 0.875, "2025–2026 Bahar Dönemi  |  2. Laboratuvar Ödevi",
            transform=ax.transAxes, fontsize=10, color="#a8c4e0",
            ha="center", va="center")

    # Ana başlık
    ax.text(0.5, 0.72, "MLE ile Akıllı Şehir Planlaması",
            transform=ax.transAxes, fontsize=21, color="#1a3a5c",
            ha="center", va="center", fontweight="bold")
    ax.text(0.5, 0.67, "Maximum Likelihood Estimation",
            transform=ax.transAxes, fontsize=13, color="#5a7a9a",
            ha="center", va="center", style="italic")

    # Ayırıcı çizgi
    ax.plot([0.18, 0.82], [0.625, 0.625],
            transform=ax.transAxes, color="#1a3a5c", linewidth=1.5)

    # Bilgi tablosu
    rows = [
        ("Ad – Soyad",      "Efe Güner"),
        ("Öğrenci No",      "24290192"),
        ("Teslim Tarihi",   "13.03.2026"),
        ("Ders Sorumlusu",  "Öğr. Gör. Ayşe Elif Yoldaş"),
    ]
    y0 = 0.57
    for label, val in rows:
        ax.text(0.36, y0, f"{label} :", transform=ax.transAxes,
                fontsize=11, color="#333", ha="right", fontweight="bold")
        ax.text(0.38, y0, val, transform=ax.transAxes,
                fontsize=11, color="#333", ha="left")
        y0 -= 0.055

    # Alt not
    ax.text(0.5, 0.08,
            "Bu rapor Python / Matplotlib kullanılarak otomatik oluşturulmuştur.",
            transform=ax.transAxes, fontsize=8.5, color="#888",
            ha="center", va="center", style="italic")

    pdf.savefig(fig); plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# SAYFA 2 — BÖLÜM 1: TEORİK TÜRETME
# ─────────────────────────────────────────────────────────────────────────────
def page_bolum1(pdf):
    fig = plt.figure(figsize=(8.27, 11.69))
    ax  = fig.add_axes([0.09, 0.04, 0.84, 0.90])
    ax.set_axis_off()
    ax.set_xlim(0, 10); ax.set_ylim(0, 26)
    y = 25.2

    section_header(ax, 0, y, 10, "Bölüm 1 — Teorik Türetme (Analitik Çözüm)")
    y -= 1.1

    # ── Adım 1 ──────────────────────────────────────────────────────────────
    ax.text(0, y, "Adım 1 — Poisson Olasılık Yoğunluğu",
            fontsize=11, fontweight="bold", color="#1a3a5c")
    y -= 0.55
    ax.text(0.3, y,
            "Trafik akışındaki araç sayısı Poisson dağılımı ile modellenir:",
            fontsize=10, color="#444")
    y -= 0.65
    ax.text(5, y,
            r"$P(k \mid \lambda) \;=\; \dfrac{e^{-\lambda}\,\lambda^{k}}{k!}$"
            r"$\qquad k = 0,1,2,\ldots\;,\quad \lambda > 0$",
            fontsize=13, ha="center", va="top")
    y -= 1.4   # dfrac: orta yükseklikte formül

    # ── Adım 2 ──────────────────────────────────────────────────────────────
    ax.text(0, y, "Adım 2 — Likelihood Fonksiyonu",
            fontsize=11, fontweight="bold", color="#1a3a5c")
    y -= 0.55
    ax.text(0.3, y,
            r"$n$ bağımsız gözlem $k_1, k_2, \ldots, k_n$ için bileşik olasılık:",
            fontsize=10, color="#444")
    y -= 0.75
    ax.text(5, y,
            r"$L(\lambda) \;=\; \prod_{i=1}^{n} \dfrac{e^{-\lambda}\,\lambda^{k_i}}{k_i!}"
            r"\;=\; e^{-n\lambda}\,\lambda^{\,\sum k_i}\,\prod_{i=1}^{n}\dfrac{1}{k_i!}$",
            fontsize=12, ha="center", va="top")
    y -= 1.85  # prod + iç dfrac: en yüksek formül

    # ── Adım 3 ──────────────────────────────────────────────────────────────
    ax.text(0, y, "Adım 3 — Log-Likelihood Fonksiyonu",
            fontsize=11, fontweight="bold", color="#1a3a5c")
    y -= 0.55
    ax.text(0.3, y,
            "Çarpım toplamdan kolay türetilir; iki tarafın logaritması alınır:",
            fontsize=10, color="#444")
    y -= 0.75
    ax.text(5, y,
            r"$\ell(\lambda) \;=\; \ln L(\lambda) \;=\;"
            r"\sum_{i=1}^{n}\left(-\lambda + k_i\ln\lambda - \ln(k_i!)\right)$",
            fontsize=11, ha="center", va="top")
    y -= 1.05  # sum ile parantezli ifade
    ax.text(5, y,
            r"$\ell(\lambda) \;=\; -n\lambda \;+\; \left(\sum_{i=1}^{n}k_i\right)\ln\lambda"
            r"\;-\; \sum_{i=1}^{n}\ln(k_i!)$",
            fontsize=11, ha="center", va="top")
    y -= 1.3

    # ── Adım 4 ──────────────────────────────────────────────────────────────
    ax.text(0, y, "Adım 4 — Türev ve Kritik Nokta",
            fontsize=11, fontweight="bold", color="#1a3a5c")
    y -= 0.55
    ax.text(0.3, y,
            r"$\ell(\lambda)$ fonksiyonunun $\lambda$'ya göre türevi alınır ve sıfıra eşitlenir:",
            fontsize=10, color="#444")
    y -= 0.75
    ax.text(5, y,
            r"$\dfrac{d\,\ell(\lambda)}{d\lambda} \;=\; -n \;+\; \dfrac{\sum_{i=1}^{n}k_i}{\lambda} \;=\; 0$",
            fontsize=13, ha="center", va="top")
    y -= 1.7   # çift dfrac: yüksek formül

    # ── Adım 5 ──────────────────────────────────────────────────────────────
    ax.text(0, y, "Adım 5 — λ için Çözüm",
            fontsize=11, fontweight="bold", color="#1a3a5c")
    y -= 0.55
    ax.text(0.3, y, r"Denklem $\lambda$ için çözülür:", fontsize=10, color="#444")
    y -= 0.75
    ax.text(5, y,
            r"$\lambda \;=\; \dfrac{\sum_{i=1}^{n}k_i}{n}$",
            fontsize=13, ha="center", va="top")
    y -= 2.4   # dfrac yüksekliği + kutu öncesi boşluk

    # ── Sonuç kutusu ────────────────────────────────────────────────────────
    ax.add_patch(patches.FancyBboxPatch(
        (0.3, y - 0.5), 9.4, 0.95,
        boxstyle="round,pad=0.1",
        facecolor="#e8f4fd", edgecolor="#1a3a5c", linewidth=1.8
    ))
    ax.text(5, y,
            r"$\hat{\lambda}_{MLE} \;=\; \dfrac{1}{n}\sum_{i=1}^{n}k_i \;=\; \bar{k}$"
            r"$\quad\Rightarrow\quad$ MLE tahmini, verilerin aritmetik ortalamasıdır.",
            fontsize=11, ha="center", va="center", color="#1a3a5c", fontweight="bold")
    y -= 1.5

    # ── Sayısal Doğrulama ────────────────────────────────────────────────────
    ax.text(0, y, "Sayısal Doğrulama",
            fontsize=11, fontweight="bold", color="#1a3a5c")
    y -= 0.55
    ax.text(0.3, y,
            rf"Veri : $[{', '.join(map(str, traffic_data))}]$   →   $n = {N}$,   $\Sigma k = {SUM_K}$",
            fontsize=10, color="#333")
    y -= 0.62
    ax.text(5, y,
            rf"$\hat{{\lambda}}_{{MLE}} \;=\; \dfrac{{{SUM_K}}}{{{N}}} \;=\; {lambda_mle:.4f}$",
            fontsize=13, ha="center", va="top")

    pdf.savefig(fig); plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# SAYFA 3 — BÖLÜM 4: OUTLIER ANALİZİ
# ─────────────────────────────────────────────────────────────────────────────
def page_bolum4(pdf):
    fig = plt.figure(figsize=(8.27, 11.69))

    # ── Üst metin paneli ──────────────────────────────────────────────────
    ax = fig.add_axes([0.09, 0.52, 0.84, 0.42])
    ax.set_axis_off()
    ax.set_xlim(0, 10); ax.set_ylim(0, 11)
    y = 10.6

    section_header(ax, 0, y, 10, "Bölüm 4 — Outlier (Aykırı Değer) Analizi")
    y -= 1.0

    # Senaryo
    ax.text(0, y, "Senaryo", fontsize=11, fontweight="bold", color="#1a3a5c")
    y -= 0.55
    ax.text(0.3, y,
            "Veri setine yanlışlıkla 200 araçlık hatalı bir gözlem (outlier) eklendi:",
            fontsize=10, color="#333")
    y -= 0.52
    ax.text(0.3, y,
            "traffic_outlier = [12, 15, 10, 8, 14, 11, 13, 16, 9, 12, 11, 14, 10, 15,  200]",
            fontsize=9.5, color="#555", family="monospace")
    y -= 0.9

    # Lambda karşılaştırma tablosu
    ax.text(0, y, "MLE Tahmini Karşılaştırması", fontsize=11, fontweight="bold", color="#1a3a5c")
    y -= 0.6

    headers = [("Veri Seti", 0.3, "center"),
               ("n", 4.2, "center"),
               ("Σk", 5.5, "center"),
               ("λ̂ MLE", 7.2, "center")]
    for txt, xpos, ha in headers:
        ax.text(xpos, y, txt, fontsize=10, fontweight="bold",
                color="white", ha=ha, va="center",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#1a3a5c"))

    y -= 0.58
    rows_data = [
        ("Temiz veri",    str(N),              str(SUM_K),
         f"{lambda_mle:.4f}",     "#2c7bb6"),
        ("Outlier ekli",  str(len(traffic_outlier)),
         str(int(np.sum(traffic_outlier))),
         f"{lambda_outlier:.4f}", "#c0392b"),
    ]
    for label, n_val, sum_val, lam_val, col in rows_data:
        ax.text(0.3, y, label,   fontsize=10, color=col, ha="center")
        ax.text(4.2, y, n_val,   fontsize=10, color=col, ha="center")
        ax.text(5.5, y, sum_val, fontsize=10, color=col, ha="center")
        ax.text(7.2, y, lam_val, fontsize=10, color=col, ha="center", fontweight="bold")
        y -= 0.52

    pct = (lambda_outlier - lambda_mle) / lambda_mle * 100
    y -= 0.15
    ax.add_patch(patches.FancyBboxPatch(
        (0.2, y - 0.28), 9.2, 0.5,
        boxstyle="round,pad=0.08",
        facecolor="#fdecea", edgecolor="#c0392b", linewidth=1.3
    ))
    ax.text(5, y,
            rf"Tek bir outlier, $\hat{{\lambda}}$ tahminini %{pct:.1f} artırdı  "
            rf"({lambda_mle:.2f} → {lambda_outlier:.2f})",
            fontsize=10.5, ha="center", va="center",
            color="#c0392b", fontweight="bold")
    y -= 0.85

    # Tartışma
    ax.text(0, y, "Tartışma ve Sonuç", fontsize=11, fontweight="bold", color="#1a3a5c")
    y -= 0.52
    bullets = [
        "MLE tahmini verinin ortalaması olduğundan aykırı değerlere karşı hassastır.",
        f"200 araçlık tek kayıt, λ tahminini {lambda_mle:.1f} → {lambda_outlier:.1f} "
        "düzeyine taşıdı.",
        "Yanlış λ ile belediye gereksiz yol kapasitesi planlar; milyonluk yatırım kararları",
        "yanıltıcı veriye dayandırılmış olur.",
        "Çözüm: IQR / Z-skoru yöntemiyle veri ön işleme uygulanmalıdır.",
    ]
    for b in bullets:
        ax.text(0.4, y, f"• {b}", fontsize=9.5, color="#333", va="top")
        y -= 0.46

    # ── Alt grafik paneli ─────────────────────────────────────────────────
    ax2 = fig.add_axes([0.10, 0.05, 0.84, 0.44])

    k_vals      = np.arange(0, 30)
    pmf_clean   = poisson.pmf(k_vals, lambda_mle)
    pmf_outlier = poisson.pmf(k_vals, lambda_outlier)

    ax2.bar(k_vals - 0.22, pmf_clean,   width=0.38, alpha=0.85,
            color="#2c7bb6",
            label=rf"Temiz veri  ($\hat{{\lambda}}$ = {lambda_mle:.2f})")
    ax2.bar(k_vals + 0.22, pmf_outlier, width=0.38, alpha=0.80,
            color="#e74c3c",
            label=rf"Outlier ekli ($\hat{{\lambda}}$ = {lambda_outlier:.2f})")

    bins = list(range(int(min(traffic_data)), int(max(traffic_data)) + 2))
    ax2.hist(traffic_data, bins=bins, density=True, alpha=0.35,
             color="#2c7bb6", align="left",
             label="Gerçek veri histogramı")

    ax2.axvline(lambda_mle,     color="#1a5276", linestyle="--",
                linewidth=1.6, alpha=0.9,
                label=rf"$\hat{{\lambda}}$ = {lambda_mle:.2f}")
    ax2.axvline(lambda_outlier, color="#922b21", linestyle="--",
                linewidth=1.6, alpha=0.9,
                label=rf"$\hat{{\lambda}}_{{out}}$ = {lambda_outlier:.2f}")

    ax2.set_xlabel("Araç Sayısı  (k)",  fontsize=11)
    ax2.set_ylabel("Olasılık",          fontsize=11)
    ax2.set_title("Outlier Etkisi — Poisson PMF Karşılaştırması",
                  fontsize=12, fontweight="bold")
    ax2.legend(fontsize=8.5, loc="upper right")
    ax2.set_xlim(0, 28)
    ax2.grid(axis="y", alpha=0.25)

    pdf.savefig(fig); plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# SAYFA 4 — KAYNAKÇA
# ─────────────────────────────────────────────────────────────────────────────
def page_kaynakca(pdf):
    fig = plt.figure(figsize=(8.27, 11.69))
    ax  = fig.add_axes([0.09, 0.70, 0.84, 0.22])
    ax.set_axis_off()
    ax.set_xlim(0, 10); ax.set_ylim(0, 4)

    ax.text(0, 3.7, "Kaynakça",
            fontsize=13, fontweight="bold", color="#1a3a5c")
    ax.plot([0, 10], [3.35, 3.35], color="#1a3a5c", linewidth=1.2)

    refs = [
        "[1] Bishop, C. M. (2006). Pattern Recognition and Machine Learning. Springer.",
        "[2] SciPy — scipy.optimize.minimize, https://docs.scipy.org/doc/scipy/",
        "[3] NumPy Documentation, https://numpy.org/doc/",
        "[4] Hunter, J. D. (2007). Matplotlib: A 2D Graphics Environment. "
        "Computing in Science & Engineering, 9(3), 90–95.",
    ]
    y = 3.0
    for r in refs:
        ax.text(0, y, r, fontsize=9.5, color="#333")
        y -= 0.58

    pdf.savefig(fig); plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# PDF ÇIKIŞI
# ─────────────────────────────────────────────────────────────────────────────
OUTPUT = r"c:/Users/EfeGuner/OneDrive/Desktop/AkilliSehirPlanlaması/MLE_Rapor.pdf"

with PdfPages(OUTPUT) as pdf:
    page_kapak(pdf)
    page_bolum1(pdf)
    page_bolum4(pdf)
    page_kaynakca(pdf)

    d = pdf.infodict()
    d["Title"]   = "MLE ile Akıllı Şehir Planlaması"
    d["Subject"] = "YZM212 Makine Öğrenmesi — 2. Lab Ödevi"

print(f"PDF olusturuldu: {OUTPUT}")
