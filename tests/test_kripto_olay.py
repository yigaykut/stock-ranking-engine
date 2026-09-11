"""Kriptoda listelenme yasi — ve kullanilmayan sutun.

Bu modulun tek gercek riski bir sey EKLEMEK degil, bir sey eklememek:
elimizde olu paritelerin tam omru var, yani "kac gun sonra kapanacak"
hesaplanabilir ve mukemmel bir tahminci olurdu. Tamamen ileriye bakis
olurdu. Test bunun sizmadigini de kontrol ediyor.

Calistir:  python tests/test_kripto_olay.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import kripto_olay as ko     # noqa: E402

fails = 0


def check(ad, kosul, ek=""):
    global fails
    ok = bool(kosul)
    if not ok:
        fails += 1
    print(f"  {'OK  ' if ok else 'HATA'}  {ad}" + (f"  {ek}" if ek else ""))


gun = pd.DatetimeIndex(pd.date_range("2020-01-01", periods=500))
f = ko.ozellikler(gun)

print("=" * 72)
print("1) YAS ILK BARDAN SAYILIYOR")
print("=" * 72)

check("ilk bar sifir yas", float(f["kripto_yas"].iloc[0]) == 0.0)
check("yas artiyor", bool(f["kripto_yas"].is_monotonic_increasing))
check("ilk 30 bar yeni", float(f["kripto_yeni30"].iloc[:30].min()) == 1.0
      and float(f["kripto_yeni30"].iloc[30]) == 0.0)
check("ilk 90 bar yeni90", float(f["kripto_yeni90"].iloc[:90].min()) == 1.0
      and float(f["kripto_yeni90"].iloc[90]) == 0.0)
# Log olcek: kesitte butun eski pariteleri tek uca yigmasin diye.
erken = float(f["kripto_yas"].iloc[40]) - float(f["kripto_yas"].iloc[10])
gec = float(f["kripto_yas"].iloc[430]) - float(f["kripto_yas"].iloc[400])
check("erken gunlerdeki fark gec gunlerdekinden buyuk", erken > 5 * gec,
      f"{erken:.3f} vs {gec:.3f}")
# Sutun float32; tam esitlik beklemek yuvarlamaya takilir.
check("yas tavanliyor",
      abs(float(f["kripto_yas"].iloc[-1]) - float(np.log1p(365))) < 1e-5,
      f"{float(f['kripto_yas'].iloc[-1]):.5f} vs {float(np.log1p(365)):.5f}")

print()
print("=" * 72)
print("2) GERIYE BAKIYOR")
print("=" * 72)

# Seriyi kesmek onceki barlari degistirmemeli: yas yalnizca GECMISE bagli.
kes = 300
kismi = ko.ozellikler(gun[:kes])
sapan = [c for c in f.columns
         if not np.allclose(f[c].to_numpy()[:kes], kismi[c].to_numpy())]
check("seriyi kesmek onceki barlari degistirmiyor", not sapan, str(sapan))

# ILERIYE BAKIS TUZAGI: olu paritelerin omru elimizde oldugu icin "kapanisa
# kac gun kaldi" hesaplanabilir. O sutun OLMAMALI.
ileri = [c for c in f.columns
         if any(k in c for k in ("kalan", "kapan", "olum", "son_"))]
check("kapanisa iliskin sutun YOK", not ileri, str(ileri))
# Ve dolayli olarak: seri uzunlugu degisince yas degismiyor demek, yasin
# toplam omru bilmedigi demek.
check("yas toplam omru bilmiyor",
      float(kismi["kripto_yas"].iloc[-1]) == float(f["kripto_yas"].iloc[kes - 1]))

print()
print("=" * 72)
print("3) SINIR DURUMLAR")
print("=" * 72)

bos = ko.ozellikler(pd.DatetimeIndex([]))
check("bos indeks patlamiyor", len(bos) == 0 and bos.shape[1] == 3)
tek = ko.ozellikler(pd.DatetimeIndex(["2021-05-05"]))
check("tek bar", len(tek) == 1 and float(tek["kripto_yeni30"].iloc[0]) == 1.0)
check("hicbiri NaN degil", int(f.isna().sum().sum()) == 0)

print()
if fails:
    print(f"{fails} KONTROL BASARISIZ")
    raise SystemExit(1)
print("TUM KRIPTO OLAY TESTLERI GECTI")
