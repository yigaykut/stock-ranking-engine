"""Kripto rejim etiketi — geriye bakiyor ve dogru yeri isaretliyor.

En buyuk tehlike, rejim etiketinin ILERIYE bakmasi. "Cokus rejimi"ni cokusu
gorduken sonra koyan bir etiket, rejime gore yapilan her olcumu anlamsiz
kilar: model cokusu bildigi icin degil, biz soyledigimiz icin iyi gorunur.

Calistir:  python tests/test_kripto_rejim.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import kripto_rejim as krj    # noqa: E402

fails = 0


def check(ad, kosul, ek=""):
    global fails
    ok = bool(kosul)
    if not ok:
        fails += 1
    print(f"  {'OK  ' if ok else 'HATA'}  {ad}" + (f"  {ek}" if ek else ""))


gun = pd.DatetimeIndex(pd.date_range("2020-01-01", periods=900))
# Once 400 gun duzenli yukselis, sonra 500 gun duzenli dusus.
yol = np.concatenate([np.linspace(100, 400, 400), np.linspace(400, 80, 500)])
btc = pd.Series(yol, index=gun)

print("=" * 72)
print("1) YUKSELIS BOGA, DUSUS AYI")
print("=" * 72)

r = krj.seri(btc)
check("ilk 200 gun bilinmiyor (ortalama tanimsiz)",
      (r["rejim"].iloc[:199] == "bilinmiyor").all())
# Yukselisin sonlarinda boga olmali.
check("yukselisin sonu boga", r["rejim"].iloc[380] == "boga",
      r["rejim"].iloc[380])
# Dususun ortasinda ayi olmali.
check("dususun ortasi ayi", r["rejim"].iloc[750] == "ayi",
      r["rejim"].iloc[750])
# Bir trend etiketi GECIKIR ve gecikmesi gerekir: tepede 200 gunluk ortalama
# hala yukari egimli ve fiyat hala ustunde, yani etiket bir sure daha boga
# der. Beklenen sey tepenin aninda yakalanmasi degil, bogadan ayiya DOGRUDAN
# atlanmamasi -- aradan gecmeden donmek, etiketin gelecege baktigi anlamina
# gelirdi.
dizi = [x for x in r["rejim"].tolist() if x != "bilinmiyor"]
gecis = [(a, b) for a, b in zip(dizi, dizi[1:]) if a != b]
check("bogadan ayiya dogrudan gecis yok",
      ("boga", "ayi") not in gecis and ("ayi", "boga") not in gecis,
      str(gecis[:6]))
check("arada karisik var", "karisik" in dizi)

print()
print("=" * 72)
print("2) GERIYE BAKIYOR")
print("=" * 72)

# Seriyi kesmek onceki gunlerin etiketini degistirmemeli. Degisiyorsa etiket
# gelecege bakiyordur ve butun rejim ayrimi anlamsizdir.
kes = 600
kismi = krj.seri(btc.iloc[:kes])
ayni = (r["rejim"].iloc[:kes].to_numpy() == kismi["rejim"].to_numpy())
check("seriyi kesmek onceki etiketleri degistirmiyor", bool(ayni.all()),
      f"{int((~ayni).sum())} gun degisti")
sapan = [c for c in ("btc_ma_uzaklik", "btc_ma_egim")
         if not np.allclose(r[c].to_numpy()[:kes], kismi[c].to_numpy(),
                            equal_nan=True)]
check("olcumler de degismiyor", not sapan, str(sapan))

print()
print("=" * 72)
print("3) GENISLIK UCUNCU KOSUL")
print("=" * 72)

# Endeks yukarida ama paritelerin yalnizca ucte biri kendi ortalamasinin
# ustundeyse, bu bir boga degil birkac ismin tasidigi yukselistir.
dar = pd.Series(0.20, index=gun)
r2 = krj.seri(btc, dar)
check("dar genislikte boga yok", (r2["rejim"] == "boga").sum() == 0,
      str(int((r2["rejim"] == "boga").sum())))
genis = pd.Series(0.80, index=gun)
r3 = krj.seri(btc, genis)
check("genis piyasada boga var", (r3["rejim"] == "boga").sum() > 100,
      str(int((r3["rejim"] == "boga").sum())))
check("genis piyasada ayi YOK (genislik kosulu tutmuyor)",
      (r3["rejim"] == "ayi").sum() == 0,
      "genislik %80 iken ayi demek celiskili olurdu")

print()
print("=" * 72)
print("4) ETIKET UYDURMUYOR")
print("=" * 72)

check("ortalama tanimsizken rejim 'bilinmiyor'",
      r["rejim"].iloc[0] == "bilinmiyor")
check("etiket kumesi sabit",
      set(r["rejim"].unique()) <= {"boga", "ayi", "karisik", "bilinmiyor"},
      str(sorted(set(r["rejim"].unique()))))
kisa = krj.seri(btc.iloc[:50])
check("cok kisa seride hepsi bilinmiyor",
      (kisa["rejim"] == "bilinmiyor").all())

print()
if fails:
    print(f"{fails} KONTROL BASARISIZ")
    raise SystemExit(1)
print("TUM KRIPTO REJIM TESTLERI GECTI")
