"""Kripto evreni — bu modulun tek varlik sebebi korunuyor mu.

Modul, hayatta kalma yanliligi OLMAYAN bir evren kurmak icin yazildi. En
buyuk risk, o ozelligi sessizce kaybetmek: bir yerde `status == TRADING`
suzgeci, bir yerde "veri gelmedi, gec" -- ve elde yine yalnizca hayatta
kalanlar kalir. Testlerin yarisi bunu bekliyor.

Ag istegi YAPILMAZ: hepsi sentetik cevaplarla.

Calistir:  python tests/test_kripto.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import kripto as kr          # noqa: E402

fails = 0


def check(ad, kosul, ek=""):
    global fails
    ok = bool(kosul)
    if not ok:
        fails += 1
    print(f"  {'OK  ' if ok else 'HATA'}  {ad}" + (f"  {ek}" if ek else ""))


def kline(gun, kapanis, hacim=1e6):
    ms = int(pd.Timestamp(gun).timestamp() * 1000)
    return [ms, kapanis * 0.99, kapanis * 1.01, kapanis * 0.98, kapanis,
            hacim / kapanis, ms + 86_399_999, hacim, 100, 0, 0, 0]


print("=" * 72)
print("1) KAPANMIS PARITELER EVRENDE KALIYOR")
print("=" * 72)

BORSA = {"symbols": [
    {"symbol": "BTCUSDT", "baseAsset": "BTC", "quoteAsset": "USDT",
     "status": "TRADING"},
    {"symbol": "BCCUSDT", "baseAsset": "BCC", "quoteAsset": "USDT",
     "status": "BREAK"},                      # kapanmis — KALMALI
    {"symbol": "LUNAUSDT", "baseAsset": "LUNA", "quoteAsset": "USDT",
     "status": "HALT"},                       # cokmus — KALMALI
    {"symbol": "BTCUPUSDT", "baseAsset": "BTCUP", "quoteAsset": "USDT",
     "status": "TRADING"},                    # kaldiracli — ELENMELI
    {"symbol": "ETHDOWNUSDT", "baseAsset": "ETHDOWN", "quoteAsset": "USDT",
     "status": "TRADING"},                    # kaldiracli — ELENMELI
    {"symbol": "USDCUSDT", "baseAsset": "USDC", "quoteAsset": "USDT",
     "status": "TRADING"},                    # sabit para — ELENMELI
    {"symbol": "JUPUSDT", "baseAsset": "JUP", "quoteAsset": "USDT",
     "status": "TRADING"},                    # UP ile bitiyor ama mesru
    {"symbol": "BTCBUSD", "baseAsset": "BTC", "quoteAsset": "BUSD",
     "status": "TRADING"},                    # USDT degil — ELENMELI
]}

gercek_istek, gercek_peek, gercek_put = kr._istek, kr._cache.peek, kr._cache.put
try:
    kr._istek = lambda yol, params, zaman_asimi=25: BORSA
    kr._cache.peek = lambda ns, ident: None
    kr._cache.put = lambda ns, ident, val: None
    s = kr.semboller()
finally:
    kr._istek, kr._cache.peek, kr._cache.put = (gercek_istek, gercek_peek,
                                                gercek_put)

check("kapanmis parite evrende", "BCCUSDT" in s)
check("cokmus parite evrende", "LUNAUSDT" in s)
check("ve kapanmis olarak isaretli",
      s.get("BCCUSDT", {}).get("aktif") is False)
check("kaldiracli token elendi",
      "BTCUPUSDT" not in s and "ETHDOWNUSDT" not in s)
check("sabit para elendi", "USDCUSDT" not in s)
check("USDT disi parite elendi", "BTCBUSD" not in s)
check("UP ile biten mesru coin ELENMEDI", "JUPUSDT" in s,
      "JUP kaldiracli degil, kisa bir isim")
check("evren dogru boyutta", len(s) == 4, str(sorted(s)))

print()
print("=" * 72)
print("2) SAYFALAMA BOSLUK BIRAKMIYOR")
print("=" * 72)

# Iki tam sayfa + kisa bir ucuncu. Sinirda bir gun kaybolmamali, ayni gun
# iki kez de gorunmemeli.
gunler = pd.bdate_range("2020-01-01", periods=2300, freq="D")
TUM = [kline(g, 100 + i * 0.1) for i, g in enumerate(gunler)]


def sahte_klines(yol, params, zaman_asimi=25):
    bas = int(params.get("startTime", 0))
    kalan = [k for k in TUM if k[0] >= bas]
    return kalan[:int(params.get("limit", 1000))]


try:
    kr._istek = sahte_klines
    d = kr.cek_bir("XUSDT", 0, bekle=0.0)
finally:
    kr._istek = gercek_istek

check("butun sayfalar birlestirildi", len(d) == len(TUM),
      f"{len(d)} / {len(TUM)}")
check("gun tekrari yok", not d.index.duplicated().any())
check("sira dogru", d.index.is_monotonic_increasing)
check("ilk ve son gun yerinde",
      d.index[0] == gunler[0].normalize() and d.index[-1] == gunler[-1].normalize())
check("dolar hacmi quote hacminden geliyor",
      abs(float(d["dolar_hacim"].iloc[0]) - 1e6) < 1.0)

print()
print("=" * 72)
print("3) SONDAKI HACIMSIZ BARLAR KESILIYOR")
print("=" * 72)

# Kapanan bir parite son gunlerinde sifir hacimle, donmus fiyatla
# gorunebiliyor. O barlari birakmak, olen bir coini tutmanin sonucunu
# "fiyat degismedi" diye okutur.
OLU = ([kline(g, 50.0) for g in pd.bdate_range("2018-01-01", periods=300)]
       + [kline(g, 50.0, hacim=0.0)
          for g in pd.bdate_range("2019-02-25", periods=3)])
try:
    kr._istek = lambda yol, params, zaman_asimi=25: OLU
    d2 = kr.cek_bir("OLUUSDT", 0, bekle=0.0)
finally:
    kr._istek = gercek_istek

check("hacimsiz kuyruk kesildi", len(d2) == 300, str(len(d2)))
check("son bar gercekten islem gormus", float(d2["dolar_hacim"].iloc[-1]) > 0)
# Ortadaki bir duraklama KESILMEMELI: yalnizca kuyruk.
ORTA = ([kline(g, 50.0) for g in pd.bdate_range("2018-01-01", periods=100)]
        + [kline(g, 50.0, hacim=0.0)
           for g in pd.bdate_range("2018-05-21", periods=2)]
        + [kline(g, 51.0) for g in pd.bdate_range("2018-05-23", periods=100)])
try:
    kr._istek = lambda yol, params, zaman_asimi=25: ORTA
    d3 = kr.cek_bir("DURUSDT", 0, bekle=0.0)
finally:
    kr._istek = gercek_istek
check("ortadaki duraklama korunuyor", len(d3) == 202, str(len(d3)))

print()
print("=" * 72)
print("4) HIZ SINIRI PASI BITIRIYOR, SAVASMIYOR")
print("=" * 72)


def patlat(yol, params, zaman_asimi=25):
    raise kr.HizSiniri("Binance 429")


try:
    kr._istek = patlat
    kr._cache.peek = lambda ns, ident: None
    r = kr.cek(["A", "B", "C"], bekle=0.0)
finally:
    kr._istek, kr._cache.peek = gercek_istek, gercek_peek

check("pas durduruldu", r["durduruldu"] is not None, str(r["durduruldu"]))
check("hicbir sey yazildi denmiyor", r["yazildi"] == 0)

# Onbellektekiler atlanir -> tekrar calistirmak ucuz.
try:
    kr._cache.peek = lambda ns, ident: ({"history": pd.DataFrame()}, 0.0)
    r2 = kr.cek(["A", "B"], bekle=0.0)
finally:
    kr._cache.peek = gercek_peek
check("onbellektekiler atlaniyor", r2["atlandi"] == 2, str(r2["atlandi"]))

print()
print("=" * 72)
print("5) KISA GECMIS YAZILMIYOR AMA SAYILIYOR")
print("=" * 72)

KISA = [kline(g, 10.0) for g in pd.bdate_range("2024-01-01", periods=30)]
yazilanlar = []


def uca_gore(yol, params, zaman_asimi=25):
    """cek() once sembol listesini, sonra barlari istiyor."""
    return BORSA if yol == "exchangeInfo" else KISA


try:
    kr._istek = uca_gore
    kr._cache.peek = lambda ns, ident: None
    kr._cache.put = lambda ns, ident, val: yazilanlar.append(ident)
    r3 = kr.cek(["KISAUSDT"], bekle=0.0, en_az_bar=200)
finally:
    kr._istek, kr._cache.peek, kr._cache.put = (gercek_istek, gercek_peek,
                                                gercek_put)
# Sembol haritasinin onbellege alinmasi normal; bakilan sey BAR yazilip
# yazilmadigi ve bar anahtarlari "SEMBOL:baslangic" seklinde.
bar_yazimi = [x for x in yazilanlar if ":" in x]
check("kisa gecmis yazilmadi", not bar_yazimi, str(bar_yazimi))
check("ama raporlandi", r3["kisa"] == 1, str(r3))

print()
if fails:
    print(f"{fails} KONTROL BASARISIZ")
    raise SystemExit(1)
print("TUM KRIPTO TESTLERI GECTI")
