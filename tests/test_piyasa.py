"""Piyasa yonu olcumu — ekilmis onculugu bulur, uydurmaz.

Bu olcum bir sey bulamadi ve asil riski o: bulamadigina guvenebilmek icin,
ORTADA OLSA bulacagini gostermek gerekiyor. Yoksa "piyasa tahmin edilemiyor"
sonucu ile "olcum bozuk" sonucu ayni gorunur.

Ag istegi YAPILMAZ.

Calistir:  python tests/test_piyasa.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import piyasa as pz         # noqa: E402

fails = 0


def check(ad, kosul, ek=""):
    global fails
    ok = bool(kosul)
    if not ok:
        fails += 1
    print(f"  {'OK  ' if ok else 'HATA'}  {ad}" + (f"  {ek}" if ek else ""))


gun = pd.DatetimeIndex(pd.bdate_range("2016-01-01", periods=2500))
r = np.random.default_rng(9)

print("=" * 72)
print("1) OZELLIKLER GERIYE BAKAR")
print("=" * 72)

spy = pd.Series(100 * np.exp(np.cumsum(r.normal(0.0003, 0.01, len(gun)))),
                index=gun)
oz = pz.ozellikler(spy)
check("sutunlar uretildi", oz.shape[1] >= 8, str(oz.shape))
kes = 1800
kismi = pz.ozellikler(spy.iloc[:kes])
sapan = [c for c in kismi.columns
         if not np.allclose(oz[c].to_numpy()[:kes], kismi[c].to_numpy(),
                            equal_nan=True)]
check("seriyi kesmek onceki gunleri degistirmiyor", not sapan, str(sapan))

print()
print("=" * 72)
print("2) EKILMIS ONCULUK BULUNUYOR")
print("=" * 72)

# Ileri getiriyi GERCEKTEN tahmin eden bir ozellik kuruluyor: yarin baslayan
# 21 gunluk getirinin bilinen bir parcasi. Olcum bunu gormezse, hicbir seyi
# goremez.
ileri = spy.shift(-21) / spy - 1.0
sahte = pd.DataFrame(index=gun)
sahte["kahin"] = ileri + r.normal(0, 0.01, len(gun))
sahte["gurultu"] = r.normal(0, 1, len(gun))
s = pz.onculuk(spy, sahte, ufuk=21)
check("tablo uretildi", s.get("ok"), str(s.get("reason")))
d = {x["ozellik"]: x for x in s["ozellikler"]}
check("kahin bulunuyor", (d["kahin"]["t_nw"] or 0) > 4,
      f"t {d['kahin']['t_nw']}, rho {d['kahin']['rho']}")
check("gurultu bulunmuyor", abs(d["gurultu"]["t_nw"] or 0) < 2.5,
      f"t {d['gurultu']['t_nw']}")
check("bagimsiz donem sayisi raporlaniyor",
      100 < s["bagimsiz_donem"] < 130, str(s["bagimsiz_donem"]))

print()
print("=" * 72)
print("3) ORTUSME HESABA KATILIYOR")
print("=" * 72)

# Ardisik gunlerin 21 gunluk ileri pencereleri neredeyse tamamen ortusuyor.
# Bunu gormezden gelen bir t, ayni veriden cok daha buyuk cikar.
from src.faktor_zaman import newey_west_t      # noqa: E402

ortak = sahte.index.intersection(ileri.dropna().index)
x = sahte["kahin"].reindex(ortak).to_numpy()
y = ileri.reindex(ortak).to_numpy()
xs = (x - x.mean()) / x.std()
t1, _, _ = newey_west_t(xs * y, lag=1)
t21, _, _ = newey_west_t(xs * y, lag=21)
check("ortusme duzeltmesi t'yi kucultuyor", abs(t21) < abs(t1),
      f"lag1 {t1:.1f} -> lag21 {t21:.1f}")

print()
print("=" * 72)
print("4) GENISLIK")
print("=" * 72)

# Yarisi ortalamasinin uzerinde, yarisi altinda olan sentetik bir evren.
paket = {}
for i in range(40):
    egim = 0.002 if i < 10 else -0.002
    c = pd.Series(50 * np.exp(np.cumsum(r.normal(egim, 0.01, len(gun)))),
                  index=gun)
    paket[f"T{i}"] = {"history": pd.DataFrame({"Close": c})}
g = pz.genislik(paket, pencere=50)
check("genislik serisi uretildi", len(g) > 2000, str(len(g)))
check("0 ile 1 arasinda", bool(g.between(0, 1).all()))
check("ilk 50 gun yok (ortalama henuz tanimsiz)",
      g.index[0] >= gun[49], f"{g.index[0].date()} vs {gun[49].date()}")
check("yukselenlerin orani dogru buyuklukte",
      0.15 < float(g.tail(500).mean()) < 0.45,
      f"{float(g.tail(500).mean()):.2f}")

# Hic bar olmayan bir evren patlamamali.
check("bos evren bos seri doner", len(pz.genislik({}, pencere=50)) == 0)

print()
if fails:
    print(f"{fails} KONTROL BASARISIZ")
    raise SystemExit(1)
print("TUM PIYASA TESTLERI GECTI")
