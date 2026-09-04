"""Ayni tohum, ayni sonuc.

04.09.2026'da dort modelli olcum iki kez kosuldu. Panel bit bit ayniydi
(188.465 satir, 73 tarih), ridge ikisinde de ayni sonucu verdi -- ama mlp bir
kez +0.0321, bir kez +0.0181 dedi. Iki kosu arasindaki fark, MODELLER
arasindaki farklardan buyuktu. Yani o tablo bir mimari karsilastirmasi degil,
rastgele tohum kumarinin fotografiydi.

Iki kaynak vardi:

  1. Yigin sirasi kuresel numpy uretecinden karistiriliyordu; o uretec hicbir
     yerde tohumlanmiyordu.
  2. Daha sinsisi: torch.manual_seed _train_torch icinde cagriliyordu, ama ag
     ondan ONCE fit() icinde kuruluyordu. Yani egitim tohumluydu, agirlik
     BASLATMA degildi.

Bu dosya ucunu de kilitler:

  1. Ayni tohumla iki egitim BIT BIT ayni tahmini vermeli
  2. Farkli tohum farkli sonuc vermeli (tohum gercekten baglaniyor mu)
  3. Egitim, kuresel uretec durumundan etkilenmemeli

Calistir:  python tests/test_tekrarlanabilirlik.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import dataset as ds       # noqa: E402
from src import models as mz        # noqa: E402

fails = 0


def check(name: str, cond: bool, extra: str = "") -> None:
    global fails
    if cond:
        print(f"  OK    {name}" + (f"  {extra}" if extra else ""))
    else:
        print(f"  HATA  {name}" + (f"  {extra}" if extra else ""))
        fails += 1


if not mz.torch_available():
    print("torch kurulu degil — tekrarlanabilirlik testleri atlandi")
    raise SystemExit(0)

import torch                        # noqa: E402


def veri(n_days: int = 22, per_day: int = 60, n_f: int = 6, seed: int = 0):
    g = np.random.default_rng(seed)
    X = g.normal(0, 1, (n_days * per_day, n_f)).astype(np.float32)
    y = (X[:, 0] * 0.7 + g.normal(0, 1, n_days * per_day)).astype(np.float32)
    d = np.repeat([f"2026-01-{i + 1:02d}" for i in range(n_days)], per_day)
    return X, y, d


X, y, dates = veri()

# Dizi modeli PENCERELENMIS girdiyle egitilir; bu donusumu boru hatti
# (walk_forward) yapiyor, model degil. Testin gercek kullanimla ayni yolu
# izlemesi icin burada da ayni fonksiyon kullaniliyor.
_panel = ds.Panel(X=X, y=y, dates=dates,
                  tickers=np.array([f"T{i % 60:03d}" for i in range(len(y))]),
                  feature_names=[f"f{i}" for i in range(X.shape[1])], horizon=21)
SX, SY, SD, _ = ds.build_sequences(_panel, window=5)

MODELLER = [
    ("mlp", lambda s: mz.MLPRanker(epochs=20, patience=5, seed=s),
     (X, y, dates), X[:60]),
    ("seq", lambda s: mz.SeqRanker(epochs=20, patience=5, seed=s, window=5),
     (SX, SY, SD), SX[:60]),
    ("attn", lambda s: mz.AttnRanker(d_model=32, heads=4, layers=1,
                                     epochs=20, patience=5, seed=s),
     (X, y, dates), X[:60]),
]


def egit(yap, s, egitim, probe):
    m = yap(s)
    m.fit(*egitim)
    return m.predict(probe)


print("=" * 70)
print("1) AYNI TOHUM -> BIT BIT AYNI")
print("=" * 70)

taban = {}
for ad, yap, egitim, probe in MODELLER:
    p1 = egit(yap, 4, egitim, probe)
    p2 = egit(yap, 4, egitim, probe)
    taban[ad] = p1
    fark = float(np.abs(p1 - p2).max())
    check(f"{ad}: iki egitim ayni tahmini veriyor", fark == 0.0, f"azami fark {fark:.2e}")

print()
print("=" * 70)
print("2) FARKLI TOHUM -> FARKLI SONUC (tohum gercekten bagli mi)")
print("=" * 70)

for ad, yap, egitim, probe in MODELLER:
    p3 = egit(yap, 91, egitim, probe)
    fark = float(np.abs(taban[ad] - p3).max())
    check(f"{ad}: tohum degisince sonuc degisiyor", fark > 1e-4, f"azami fark {fark:.4f}")

print()
print("=" * 70)
print("3) KURESEL URETEC DURUMU SONUCU ETKILEMEMELI")
print("=" * 70)

# Egitimden once kuresel ureteclerin durumunu bozalim. Tohumlama dogru
# yapiliyorsa sonuc degismemeli. Eski kodda AG BASLATMA bu durumdan
# etkileniyordu ve bu, iki kosunun farkli cikmasinin asil sebebiydi.
for ad, yap, egitim, probe in MODELLER:
    np.random.seed(12345)
    np.random.random(1000)
    torch.manual_seed(999)
    _ = torch.randn(500)
    p4 = egit(yap, 4, egitim, probe)
    fark = float(np.abs(taban[ad] - p4).max())
    check(f"{ad}: bozulmus kuresel durumdan etkilenmiyor", fark == 0.0,
          f"azami fark {fark:.2e}")

print()
print("=" * 70)
print("4) TOHUMLAMA YARDIMCISI")
print("=" * 70)

mz._tohumla(7)
a = torch.randn(5)
mz._tohumla(7)
b = torch.randn(5)
check("_tohumla ayni diziyi veriyor", bool(torch.equal(a, b)))
mz._tohumla(8)
c = torch.randn(5)
check("farkli tohum farkli dizi", not bool(torch.equal(a, c)))

print()
if fails:
    print(f"{fails} KONTROL BASARISIZ")
    raise SystemExit(1)
print("TUM TEKRARLANABILIRLIK TESTLERI GECTI")
