"""Capraz kesitsel dikkat modeli (AttnRanker).

Bu model digerlerinden BIR SEYLE ayriliyor: bir hisseyi puanlarken ayni gunun
diger hisselerine bakiyor. Testler tam olarak bu iddiayi kovaliyor -- "egitim
patlamadan bitti" demek yeterli degil.

  1. Kume davranisi : ayni girdi, farkli komsularla FARKLI puan almali.
                      Almiyorsa dikkat katmani bosuna duruyordur.
  2. Permutasyon    : hisseleri karistirmak puanlari ayni sekilde
     esdegerligi      karistirmali, DEGISTIRMEMELI. Bir gunun hisseleri
                      sirasiz bir kumedir; modelin sira ogrenmemesi gerekir.
                      (Konum kodlamasi eklenirse bu test duser.)
  3. Gun yalitimi   : predict(dates=...) her gunu ayri islemeli. Gunler
                      birbirine karisirsa bir gunun siralamasi baska bir gunun
                      hisselerinden etkilenir -- sessiz ve ciddi bir hata.
  4. Kalicilik      : pickle turu sonrasi ayni tahmin.
  5. Ogrenme        : ogrenilebilir bir sinyalde taban cizgisini gecebilmeli.
                      (Bu SENTETIK veridir; gercek piyasada ustunluk iddiasi
                      DEGILDIR -- gercek olcum walk_forward ile yapilir.)

Calistir:  python tests/test_attn.py
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import models as mz              # noqa: E402

fails = 0


def check(name: str, cond: bool, extra: str = "") -> None:
    global fails
    if cond:
        print(f"  OK    {name}" + (f"  {extra}" if extra else ""))
    else:
        print(f"  HATA  {name}" + (f"  {extra}" if extra else ""))
        fails += 1


if not mz.torch_available():
    print("torch kurulu degil — AttnRanker testleri atlandi")
    raise SystemExit(0)

print("=" * 70)
print("0) KAYIT DEFTERI")
print("=" * 70)
check("attn kayitli", "attn" in mz.AVAILABLE)
check("needs_dates bayragi acik", getattr(mz.AttnRanker, "needs_dates", False))
check("dizi modeli DEGIL", not getattr(mz.AttnRanker, "needs_sequence", True))
try:
    mz.AttnRanker(d_model=10, heads=4)
    check("d_model/heads uyumsuzlugu yakalaniyor", False)
except ValueError:
    check("d_model/heads uyumsuzlugu yakalaniyor", True)


# ---------------------------------------------------------------------------
# Sentetik veri: hedef, gun ICINDEKI goreli konuma bagli.
# Yani "ham deger" degil, "o gun kacinci sirada oldugun" onemli. Dikkat
# modelinin mimarisiyle tam ortusen sinyal bu.
# ---------------------------------------------------------------------------
def synth(n_days: int = 40, per_day: int = 60, n_f: int = 6, seed: int = 0):
    rng = np.random.default_rng(seed)
    X, y, dates = [], [], []
    for d in range(n_days):
        # Gun bazli kayma: ham deger tek basina anlamsiz olsun
        shift = rng.normal(0, 3.0)
        f = rng.normal(shift, 1.0, size=(per_day, n_f)).astype(np.float32)
        # Hedef: ilk ozelligin GUN ICINDEKI yuzdeligi (+ gurultu)
        r = f[:, 0].argsort().argsort() / (per_day - 1)
        target = (r - 0.5) * 2 + rng.normal(0, 0.35, per_day)
        X.append(f)
        y.append(target.astype(np.float32))
        dates.append(np.full(per_day, f"2026-01-{d + 1:02d}"))
    return (np.vstack(X), np.concatenate(y), np.concatenate(dates))


X, y, dates = synth()

print()
print("=" * 70)
print("1) KUME DAVRANISI - komsular puani degistirmeli")
print("=" * 70)

m = mz.AttnRanker(d_model=32, heads=4, layers=2, epochs=40, patience=8, seed=3)
m.fit(X, y, dates)

probe = X[:40].copy()
p_full = m.predict(probe)                       # tek gun, 40 hisse
# Ayni ilk hisse, TAMAMEN farkli komsularla
alt = np.vstack([probe[:1], probe[20:40] * 0 + probe[20:40].mean(0) + 5.0])
p_alt = m.predict(alt)
check("ayni hisse farkli komsularla farkli puan aliyor",
      abs(float(p_full[0]) - float(p_alt[0])) > 1e-4,
      f"{float(p_full[0]):.5f} vs {float(p_alt[0]):.5f}")

# Tek basina bir hisse ile kalabalik icindeki ayni hisse
p_solo = m.predict(np.repeat(probe[:1], 8, axis=0))
check("tek basina puan, kalabalik icindekinden farkli",
      abs(float(p_solo[0]) - float(p_full[0])) > 1e-4)

print()
print("=" * 70)
print("2) PERMUTASYON ESDEGERLIGI - kume, dizi degil")
print("=" * 70)

rng = np.random.default_rng(42)
order = rng.permutation(len(probe))
p_perm = m.predict(probe[order])
# Karistirilmis girdinin ciktisi, orijinal ciktinin ayni sekilde karistirilmisi
# olmali. Konum kodlamasi olsaydi bu esitlik BOZULURDU.
max_diff = float(np.abs(p_perm - p_full[order]).max())
check("hisseleri karistirmak puanlari yalnizca yeniden siraliyor",
      max_diff < 2e-4, f"azami sapma {max_diff:.2e}")

# Ayrica: siralama da ayni olmali
r1 = p_full.argsort().argsort()
r2 = p_perm.argsort().argsort()
check("siralama permutasyondan etkilenmiyor",
      bool((r1[order] == r2).all()))

print()
print("=" * 70)
print("3) GUN YALITIMI - gunler birbirine karismamali")
print("=" * 70)

two = np.vstack([X[:30], X[60:90]])
dd = np.array(["2026-01-01"] * 30 + ["2026-01-02"] * 30)
p_together = m.predict(two, dd)
p_sep = np.concatenate([m.predict(X[:30]), m.predict(X[60:90])])
check("dates verilince her gun ayri isleniyor",
      float(np.abs(p_together - p_sep).max()) < 2e-4)

p_mixed = m.predict(two)          # dates YOK -> tek kume sayilir
check("dates verilmezse gunler karisiyor (belgelenmis davranis)",
      float(np.abs(p_mixed - p_sep).max()) > 1e-4)

print()
print("=" * 70)
print("4) KALICILIK - pickle turu")
print("=" * 70)

blob = pickle.dumps(m)
m2 = pickle.loads(blob)
check("pickle sonrasi ayni tahmin",
      float(np.abs(m2.predict(probe) - p_full).max()) < 1e-6)
check("agirliklar geri yuklendi", m2.net is not None)

print()
print("=" * 70)
print("5) OGRENME - sentetik sinyalde taban cizgisini gecebilmeli")
print("=" * 70)

Xtr, ytr, dtr = synth(n_days=40, per_day=60, seed=1)
Xte, yte, dte = synth(n_days=12, per_day=60, seed=2)

ridge = mz.RidgeRanker()
ridge.fit(Xtr, ytr, dtr)
attn = mz.AttnRanker(d_model=32, heads=4, layers=2, epochs=80, patience=12, seed=5)
attn.fit(Xtr, ytr, dtr)


def day_ic(pred: np.ndarray, truth: np.ndarray, dd: np.ndarray) -> float:
    ics = [mz.spearman(pred[dd == d], truth[dd == d]) for d in np.unique(dd)]
    ics = [i for i in ics if np.isfinite(i)]
    return float(np.mean(ics)) if ics else float("nan")


ic_r = day_ic(ridge.predict(Xte), yte, dte)
ic_a = day_ic(attn.predict(Xte, dte), yte, dte)
print(f"        ridge IC = {ic_r:+.4f}")
print(f"        attn  IC = {ic_a:+.4f}")
check("attn sentetik sinyalde pozitif IC uretiyor", ic_a > 0.05, f"{ic_a:+.4f}")
check("attn taban cizgisine yakin veya ustunde", ic_a > ic_r - 0.05,
      f"fark {ic_a - ic_r:+.4f}")

print()
print("NOT: 5. bolum SENTETIK veridir ve gercek piyasada ustunluk iddiasi")
print("     DEGILDIR. Gercek olcum: python run.py ml train --models attn")

print()
if fails:
    print(f"{fails} KONTROL BASARISIZ")
    raise SystemExit(1)
print("TUM DIKKAT MODELI TESTLERI GECTI")
