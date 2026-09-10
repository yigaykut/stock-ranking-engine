"""Hayatta kalma olcumu — kovalarin ve oranlarin dogrulugu.

Bu tablo bir sonuc degil bir UYARI uretiyor: modelin gordugu evrenin ne
kadarinin eksik oldugunu soyluyor. Yanlis hesaplanirsa uyari da yanlis olur,
ve bu projede her pozitif sonucun tek ciddi rakibi bu yanlilik.

Ag istegi YAPILMAZ: cerceve cekimi sentetik veriyle degistiriliyor.

Calistir:  python tests/test_hayatta_kalma.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import hayatta_kalma as hk    # noqa: E402

fails = 0


def check(ad, kosul, ek=""):
    global fails
    ok = bool(kosul)
    if not ok:
        fails += 1
    print(f"  {'OK  ' if ok else 'HATA'}  {ad}" + (f"  {ek}" if ek else ""))


print("=" * 72)
print("1) KUCUKLER OLUYOR, BUYUKLER KALIYOR")
print("=" * 72)

# 1000 sirket. Varlik 1e5'ten 1e11'e; hayatta kalma varliga bagli olarak
# %10'dan %90'a cikiyor. Tablo bu gradyani bulmali.
n = 1000
cikler = list(range(1, n + 1))
varlik = [10 ** (5 + 6 * i / n) for i in range(n)]
yasayan = [c for i, c in enumerate(cikler) if (i % 10) < (1 + 8 * i // n)]

gercek = hk._cerceve
try:
    def sahte(donem, yenile=False):
        if donem == "A":
            return pd.DataFrame({"cik": cikler, "ad": [""] * n,
                                 "varlik": varlik})
        return pd.DataFrame({"cik": yasayan, "ad": [""] * len(yasayan),
                             "varlik": [1.0] * len(yasayan)})
    hk._cerceve = sahte
    hk.cik_haritasi = lambda: {}
    r = hk.kohort(baslangic="A", bitis="B", kova=10)
finally:
    hk._cerceve = gercek

check("tablo uretildi", r.get("ok"), str(r.get("reason")))
check("on kova var", len(r["kovalar"]) == 10, str(len(r["kovalar"])))
check("kohort tam sayildi", r["kohort"] == n, str(r["kohort"]))

alt, ust = r["kovalar"][0], r["kovalar"][-1]
check("en kucuk kovada hayatta kalma dusuk", alt["yasayan"] < 0.3,
      f"%{100 * alt['yasayan']:.0f}")
check("en buyuk kovada yuksek", ust["yasayan"] > 0.7,
      f"%{100 * ust['yasayan']:.0f}")
check("kovalar varliga gore siralanmis",
      all(r["kovalar"][i]["ortanca_varlik"] < r["kovalar"][i + 1]["ortanca_varlik"]
          for i in range(9)))
check("gradyan tek yonlu",
      alt["yasayan"] < r["kovalar"][5]["yasayan"] < ust["yasayan"])

print()
print("=" * 72)
print("2) GRADYAN YOKKA TABLO GRADYAN UYDURMUYOR")
print("=" * 72)

# Ayni varlik dagilimi, ama hayatta kalma buyuklukten BAGIMSIZ.
duz = [c for i, c in enumerate(cikler) if i % 2 == 0]
try:
    def sahte2(donem, yenile=False):
        if donem == "A":
            return pd.DataFrame({"cik": cikler, "ad": [""] * n,
                                 "varlik": varlik})
        return pd.DataFrame({"cik": duz, "ad": [""] * len(duz),
                             "varlik": [1.0] * len(duz)})
    hk._cerceve = sahte2
    r2 = hk.kohort(baslangic="A", bitis="B", kova=10)
finally:
    hk._cerceve = gercek

oranlar = [x["yasayan"] for x in r2["kovalar"]]
check("her kovada ayni oran", max(oranlar) - min(oranlar) < 0.05,
      f"{min(oranlar):.2f} .. {max(oranlar):.2f}")
check("genel oran dogru", abs(r2["yasayan_oran"] - 0.5) < 0.02,
      f"{r2['yasayan_oran']}")

print()
print("=" * 72)
print("3) SIFIR VE EKSI VARLIK ELENIYOR")
print("=" * 72)

try:
    def sahte3(donem, yenile=False):
        if donem == "A":
            return pd.DataFrame({"cik": [1, 2, 3, 4] * 50,
                                 "ad": [""] * 200,
                                 "varlik": [0.0, -5.0, 1e6, 1e9] * 50})
        return pd.DataFrame({"cik": [3], "ad": [""], "varlik": [1.0]})
    hk._cerceve = sahte3
    r3 = hk.kohort(baslangic="A", bitis="B", kova=2)
finally:
    hk._cerceve = gercek

check("varligi sifir/eksi olanlar kohorta girmiyor", r3["kohort"] == 100,
      str(r3["kohort"]))

print()
if fails:
    print(f"{fails} KONTROL BASARISIZ")
    raise SystemExit(1)
print("TUM HAYATTA KALMA TESTLERI GECTI")
