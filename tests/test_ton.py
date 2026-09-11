"""Bildirim tonu — metin dogru yerden cikiyor ve puan dogru olcuyor.

Iki tehlike var ve ikisi de sessiz:

  1. Yanlis belgeyi okumak. 8-K'nin birincil belgesi cogunlukla XBRL
     etiketinden ibaret; oradan bir "ton" hesaplanirsa sayi uretilir ama
     olculen sey metin degil, etiket sayisi olur.
  2. Puani metnin UZUNLUGUNA baglamak. Uzun bir bulten her turden daha cok
     kelime icerir; toplam kelimeye bolmek tonu boyutun fonksiyonu yapar.

Ag istegi YAPILMAZ.

Calistir:  python tests/test_ton.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import ton                    # noqa: E402

fails = 0


def check(ad, kosul, ek=""):
    global fails
    ok = bool(kosul)
    if not ok:
        fails += 1
    print(f"  {'OK  ' if ok else 'HATA'}  {ad}" + (f"  {ek}" if ek else ""))


GONDERI = """
<SEC-DOCUMENT>0000000000-26-000001.txt
<DOCUMENT>
<TYPE>8-K
<FILENAME>birincil.htm
<TEXT>
<html><body><ix:header>0000320193 2026-01-01 us-gaap:CommonStockMember
</ix:header></body></html>
</TEXT>
</DOCUMENT>
<DOCUMENT>
<TYPE>EX-99.1
<FILENAME>basin.htm
<TEXT>
<html><body><p>Acme reports record revenue and strong growth.</p>
<p>Margins improved and the company exceeded guidance.</p></body></html>
</TEXT>
</DOCUMENT>
<DOCUMENT>
<TYPE>GRAPHIC
<FILENAME>logo.jpg
<TEXT>
binary-noise-should-not-appear
</TEXT>
</DOCUMENT>
"""

print("=" * 72)
print("1) METIN EX-99'DAN CIKIYOR")
print("=" * 72)

m = ton._metni_ayikla(GONDERI)
check("basin bulteni bulundu", "record revenue" in m, m[:60])
# Birincil belge XBRL etiketi; oradan gelen hicbir sey metne karismamali.
check("birincil belgedeki XBRL girmiyor", "CommonStockMember" not in m)
check("ikili ek girmiyor", "binary-noise" not in m)
check("html etiketleri temizlendi", "<p>" not in m and "<html>" not in m)

# EX-99 hic yoksa BOS donmeli; birincil belgeye dusmemeli, cunku orasi
# anlati degil etiket ve oradan uretilen bir ton baska bir seyi olcer.
yalniz_birincil = GONDERI.split("<DOCUMENT>")[1]
check("EX-99 yoksa bos doner",
      ton._metni_ayikla("<DOCUMENT>" + yalniz_birincil) == "")

print()
print("=" * 72)
print("2) PUAN DOGRU YONU GOSTERIYOR")
print("=" * 72)

dolgu = " the and of to in for with on at by from as that this " * 20
iyi = ("record growth increased strong improved exceeded profitable "
       "expansion successful achieved milestone gains " + dolgu)
kotu = ("loss declined weakness impairment litigation investigation default "
        "resigned terminated downgrade adverse shortfall " + dolgu)
notr = dolgu * 2

p_iyi, p_kotu, p_notr = ton.puan(iyi), ton.puan(kotu), ton.puan(notr)
check("olumlu metin pozitif", p_iyi["ton"] > 0.5, f"{p_iyi['ton']:.2f}")
check("olumsuz metin negatif", p_kotu["ton"] < -0.5, f"{p_kotu['ton']:.2f}")
check("notr metin puanlanmiyor", not np.isfinite(p_notr["ton"]),
      "duygu yuklu kelime yok, sayi uydurulmamali")
check("olumsuz oran olumsuz metinde yuksek",
      p_kotu["olumsuz_oran"] > p_iyi["olumsuz_oran"],
      f"{p_kotu['olumsuz_oran']:.4f} vs {p_iyi['olumsuz_oran']:.4f}")

print()
print("=" * 72)
print("3) PUAN METNIN UZUNLUGUNA BAGLI DEGIL")
print("=" * 72)

# Ayni metin iki kez: her turden kelime iki katina cikar, ORAN degismemeli.
tek, cift = ton.puan(iyi), ton.puan(iyi + " " + iyi)
check("metni iki katina cikarmak tonu degistirmiyor",
      abs(tek["ton"] - cift["ton"]) < 1e-9,
      f"{tek['ton']:.4f} vs {cift['ton']:.4f}")
check("kelime sayisi gercekten iki katina cikti",
      abs(cift["kelime"] - 2 * tek["kelime"]) <= 1,
      f"{tek['kelime']} -> {cift['kelime']}")
# Olumsuz oran da bir ORAN; o da degismemeli.
check("olumsuz oran da degismiyor",
      abs(tek["olumsuz_oran"] - cift["olumsuz_oran"]) < 1e-9)

print()
print("=" * 72)
print("4) ELDE OLMAYAN METIN")
print("=" * 72)

check("bos metin NaN doner", not np.isfinite(ton.puan("")["ton"]))
check("None NaN doner", not np.isfinite(ton.puan(None)["ton"]))
check("cok kisa metin puanlanmiyor",
      not np.isfinite(ton.puan("record growth")["ton"]),
      "elli kelimeden kisa bir metinden ton cikarilmaz")

print()
if fails:
    print(f"{fails} KONTROL BASARISIZ")
    raise SystemExit(1)
print("TUM TON TESTLERI GECTI")
