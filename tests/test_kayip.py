"""Kayip fonksiyonu: hedef siraya cevrilince ne degisiyor?

04.09.2026'ya kadar kayip, tahmin ile HAM getiri arasindaki korelasyondu ve
koda "Spearman vekili" diye yazilmisti. Ham getiriyle alinan korelasyon
Pearson'dur; ortalama/standart sapma duzeltmesi olcegi duzeltir, carpikligi
duzeltmez. Bu dosya farki olcuyor.

  1. Sira donusumu   : ciktilar [-1,1] araliginda esit araliklarla dagilmali
  2. Monoton         : hedefe monoton bir donusum uygulamak sira kaybini
     donusum            DEGISTIRMEMELI (Spearman'i taniyan ozellik).
                        Pearson kaybi degisir.
  3. Aykiri deger    : bir hissenin getirisini SIRASI degismeden buyutmek
                        Pearson kaybini oynatmali, sira kaybini HIC
                        oynatmamali
  4. Ne oldugu       : kayip = -Pearson(tahmin, hedefin sirasi). Tam Spearman
                        DEGIL: tahmin tarafi siraya cevrilmez, cunku argsort
                        turevlenemez ve gradyan olurdu.
  5. Egitim          : agir kuyruklu hedefte sira kaybi en az Pearson kadar
                        iyi olmali (asil iddia bu)

Calistir:  python tests/test_kayip.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import models as mz         # noqa: E402

fails = 0


def check(name: str, cond: bool, extra: str = "") -> None:
    global fails
    if cond:
        print(f"  OK    {name}" + (f"  {extra}" if extra else ""))
    else:
        print(f"  HATA  {name}" + (f"  {extra}" if extra else ""))
        fails += 1


if not mz.torch_available():
    print("torch kurulu degil — kayip testleri atlandi")
    raise SystemExit(0)

import torch                          # noqa: E402

rng = np.random.default_rng(11)


def T(a) -> "torch.Tensor":
    return torch.tensor(np.asarray(a, dtype=np.float32))


print("=" * 70)
print("1) SIRA DONUSUMU")
print("=" * 70)

x = T([5.0, -2.0, 9.0, 0.0])
r = mz._to_rank_tensor(x).numpy()
check("aralik [-1, 1]", abs(r.min() + 1) < 1e-6 and abs(r.max() - 1) < 1e-6, str(r))
check("sira, girdinin sirasiyla ayni",
      list(r.argsort()) == list(np.asarray([5.0, -2.0, 9.0, 0.0]).argsort()))
check("esit aralikli", np.allclose(np.diff(np.sort(r)), 2.0 / 3.0))
check("tek elemanda patlamiyor", float(mz._to_rank_tensor(T([1.0])).sum()) == 0.0)

print()
print("=" * 70)
print("2) MONOTON DONUSUM — Spearman'i taniyan ozellik")
print("=" * 70)

pred = T(rng.normal(0, 1, 60))
y = T(rng.normal(0, 1, 60))
# exp monoton artan: siralamayi korur, degerleri carpitir
y_exp = torch.exp(y * 1.5)

s1 = float(mz._rank_loss(pred, y, rank_target=True))
s2 = float(mz._rank_loss(pred, y_exp, rank_target=True))
check("sira kaybi monoton donusumden ETKILENMIYOR", abs(s1 - s2) < 1e-6,
      f"{s1:.6f} vs {s2:.6f}")

p1 = float(mz._rank_loss(pred, y, rank_target=False))
p2 = float(mz._rank_loss(pred, y_exp, rank_target=False))
check("Pearson kaybi ETKILENIYOR (eski davranis)", abs(p1 - p2) > 0.05,
      f"{p1:.4f} -> {p2:.4f}")

print()
print("=" * 70)
print("3) AYKIRI DEGER")
print("=" * 70)

# En yuksek getirili hisseyi daha da yukari cek: SIRALAMA ayni kalir.
taban = rng.normal(0, 0.02, 50).astype(np.float32)
pred = T(np.arange(50))
y_norm = taban.copy()
y_buyuk = taban.copy()
y_buyuk[int(taban.argmax())] = 3.0        # zaten en yuksekti, simdi %300

d_sira = abs(float(mz._rank_loss(pred, T(y_norm), True)) -
             float(mz._rank_loss(pred, T(y_buyuk), True)))
d_pear = abs(float(mz._rank_loss(pred, T(y_norm), False)) -
             float(mz._rank_loss(pred, T(y_buyuk), False)))
check("sira degismedigi icin sira kaybi HIC degismiyor", d_sira < 1e-6,
      f"{d_sira:.2e}")
check("ayni degisiklik Pearson kaybini ciddi oynatiyor", d_pear > 0.05,
      f"{d_pear:.4f}")

# Sirayi da degistiren bir sicrama: sira kaybi degisir ama az.
y_sic = np.concatenate([taban[:49], [3.0]]).astype(np.float32)
d_sira2 = abs(float(mz._rank_loss(pred, T(taban), True)) -
              float(mz._rank_loss(pred, T(y_sic), True)))
d_pear2 = abs(float(mz._rank_loss(pred, T(taban), False)) -
              float(mz._rank_loss(pred, T(y_sic), False)))
check("sira degisince bile Pearson daha cok oynuyor", d_pear2 > 2 * d_sira2,
      f"pearson {d_pear2:.4f} vs sira {d_sira2:.4f}")

print()
print("=" * 70)
print("4) NE OLDUGU — kayip = -Pearson(tahmin, hedefin sirasi)")
print("=" * 70)


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(a, b)[0, 1])


for seed in (1, 2, 3):
    g = np.random.default_rng(seed)
    a = g.normal(0, 1, 80).astype(np.float32)
    b = (0.6 * a + g.normal(0, 1, 80)).astype(np.float32)
    kayip = float(mz._rank_loss(T(b), T(a), rank_target=True))
    hedef_sira = mz._to_rank_tensor(T(a)).numpy()
    check(f"seed {seed}: kayip == -Pearson(tahmin, hedef sirasi)",
          abs(kayip + pearson(b, hedef_sira)) < 2e-3,
          f"{kayip:+.4f} vs {-pearson(b, hedef_sira):+.4f}")
    # Tam Spearman'a YAKIN ama esit degil -- bu bilerek boyle.
    check(f"seed {seed}: Spearman'a yakin ama ayni degil",
          abs(kayip + mz.spearman(b, a)) < 0.05,
          f"fark {abs(kayip + mz.spearman(b, a)):.4f}")

print()
print("=" * 70)
print("5) EGITIM — agir kuyruklu hedefte hangisi daha iyi ogreniyor")
print("=" * 70)


def agir_kuyruk(n_days=45, per_day=70, n_f=6, seed=0):
    """Sinyal temiz, GURULTU agir kuyruklu.

    Gercek piyasada gunluk getiri dagilimi normal degil: birkac hisse
    haber/bilanco ile %50-300 oynuyor. Pearson kaybi bu hisseleri aciklamaya
    calisir, sira kaybi onlari sadece 'en ustte' sayar.
    """
    g = np.random.default_rng(seed)
    X, y, dates = [], [], []
    for d in range(n_days):
        f = g.normal(0, 1, size=(per_day, n_f)).astype(np.float32)
        sinyal = f[:, 0] * 0.8 + f[:, 1] * 0.4
        gurultu = g.standard_t(df=2.0, size=per_day) * 0.9    # agir kuyruk
        sicrama = (g.random(per_day) < 0.03) * g.random(per_day) * 12.0
        X.append(f)
        y.append((sinyal + gurultu + sicrama).astype(np.float32))
        dates.append(np.full(per_day, f"2026-02-{d + 1:02d}"))
    return np.vstack(X), np.concatenate(y), np.concatenate(dates)


Xtr, ytr, dtr = agir_kuyruk(seed=1)
Xte, yte, dte = agir_kuyruk(n_days=15, seed=2)


def gun_ic(pred, truth, dd):
    ics = [mz.spearman(pred[dd == d], truth[dd == d]) for d in np.unique(dd)]
    ics = [i for i in ics if np.isfinite(i)]
    return float(np.mean(ics)) if ics else float("nan")


def egit_ve_olc(rank_target: bool, seed: int) -> float:
    eski = mz.RANK_TARGET
    mz.RANK_TARGET = rank_target
    try:
        m = mz.MLPRanker(epochs=90, patience=12, seed=seed)
        m.fit(Xtr, ytr, dtr)
        return gun_ic(m.predict(Xte), yte, dte)
    finally:
        mz.RANK_TARGET = eski


# Tek kosuda fark rastgele tohuma takilabilir; uc tohumun ortalamasi alinir.
sira_ic = [egit_ve_olc(True, s) for s in (3, 5, 7)]
pear_ic = [egit_ve_olc(False, s) for s in (3, 5, 7)]
print(f"        sira kaybi    IC: {np.mean(sira_ic):+.4f}   {[round(v, 4) for v in sira_ic]}")
print(f"        Pearson kaybi IC: {np.mean(pear_ic):+.4f}   {[round(v, 4) for v in pear_ic]}")

check("sira kaybi pozitif IC uretiyor", np.mean(sira_ic) > 0.05,
      f"{np.mean(sira_ic):+.4f}")
check("sira kaybi, Pearson kaybindan kotu DEGIL",
      np.mean(sira_ic) >= np.mean(pear_ic) - 0.01,
      f"fark {np.mean(sira_ic) - np.mean(pear_ic):+.4f}")

check("varsayilan sira kaybi", mz.RANK_TARGET is True)

print()
print("NOT: 5. bolum sentetiktir. Gercek panelde olcum:")
print("     python run.py ml train --pretrain --models ridge,mlp,seq,attn")

print()
if fails:
    print(f"{fails} KONTROL BASARISIZ")
    raise SystemExit(1)
print("TUM KAYIP TESTLERI GECTI")
