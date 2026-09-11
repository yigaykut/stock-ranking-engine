"""Binance gunluk barlari — hayatta kalma yanliligi OLMAYAN bir evren.

NEDEN BU MODUL VAR
------------------
Bu projede olculen her pozitif sonuc ayni yere ciktı: kenar en ince
isimlerde, ve dolar hacmi eksenini cikarinca kenar yok oluyor. Sebep
olculdu: fiyat onbellegi yalnizca BUGUN kote olan sirketleri tutuyor ve
2016 kohortunun en kucuk onda birinde bugun onbellekte olan %1. Yani
mikro-kap bandinda "kenar" diye gorulen sey buyuk olcude hayatta
kalanlarin kendisi.

Ucretsiz hisse verisiyle bu duzeltilemiyor: Yahoo kote disi sembollere 404
donuyor ve 200 donenler yeniden kullanilmis semboller (SBNY 2024'te,
BBBY 2026'da baska sirket).

Binance'te durum tersine. Kapanmis pariteler API'de duruyor ve gecmisleri
eksiksiz geliyor: BCCUSDT'nin 371 gunluk tum omru (2017-11 .. 2018-11),
LUNAUSDT'nin cokusu, FTTUSDT'nin FTX ile batisi. 490 aktif paritenin
yaninda 252 kapanmis parite var ve ONLARI DA ALIYORUZ. Bu modulun varlik
sebebi tam olarak bu; `status != TRADING` olanlari elemek, kacmaya
calistigimiz yanliligi elle geri koymak olurdu.

Ayrica: 2017-08'den bu yana gunluk bar, istek basina 1000 kayit,
sayfalanabiliyor, anahtar gerekmiyor.

NE ELENIYOR VE NEDEN
--------------------
  - Kaldiracli tokenlar (BTCUP, ETHDOWN, XRPBULL...). Bunlar ayri bir
    varlik degil, ayni varligin turevi; kesitte ayni hareketi birkac kez
    saydirirlar ve kaldiracli olduklari icin oynaklik siralamasinin tepesini
    hep onlar tutar.
  - Sabit para pariteleri (USDCUSDT, BUSDUSDT, EURUSDT...). Neredeyse sifir
    oynaklikli ve bir kesitsel siralamanin "en sakin" ucunu tek baslarina
    doldururlar.

MALIYET — BURASI HISSEDEN SERT
------------------------------
Binance taker ucreti %0.1, yani tek yonde 10bp ve gidis-donus 20bp. Hisse
tarafinda 10bp kullaniyorduk. Kripto olcumlerinde varsayilan 20/40/60bp
olmali; 1 gunluk bir ufukta bu yilda 250 tur eder ve cogu kenari tek basina
yer.
"""
from __future__ import annotations

import time

import pandas as pd
import requests

from .providers import cache as _cache

UC = "https://api.binance.com/api/v3"
ALAN = "kripto"
BEKLE = 0.15                    # ~7 istek/saniye; Binance agirlik siniri bol
EN_AZ_BAR = 200                 # bundan kisa gecmis kesitte tasiyamaz
GUN_MS = 86_400_000

# Sabit para ve itibari para bazlari: kesitte oynaklik sifir, siralamayi
# bozarlar.
SABIT = {
    "USDC", "BUSD", "TUSD", "DAI", "USDP", "FDUSD", "USDS", "SUSD", "PAX",
    "EUR", "GBP", "AUD", "TRY", "BRL", "RUB", "JPY", "ZAR", "IDRT", "NGN",
    "UAH", "BIDR", "VAI", "USTC",
}
KALDIRAC_SONU = ("UP", "DOWN", "BULL", "BEAR")


class HizSiniri(RuntimeError):
    """Binance 429/418 dondurdu — pas bitirilir, kaldigi yerden devam edilir."""


def _istek(yol: str, params: dict, zaman_asimi: int = 25):
    r = requests.get(f"{UC}/{yol}", params=params, timeout=zaman_asimi,
                     headers={"User-Agent": "invest-research"})
    if r.status_code in (429, 418):
        raise HizSiniri(f"Binance {r.status_code}")
    r.raise_for_status()
    return r.json()


def _kaldiracli(baz: str) -> bool:
    """BTCUP, ETHDOWN, XRPBULL gibi turev tokenlar.

    Sondaki eki kesip geriye anlamli bir baz kaliyorsa turevdir. Uzunluk
    sarti, UP ile biten mesru bir coini (ornegin "JUP") elemekten korur.
    """
    for ek in KALDIRAC_SONU:
        if baz.endswith(ek) and len(baz) > len(ek) + 2:
            return True
    return False


def semboller(yenile: bool = False) -> dict:
    """USDT pariteleri — KAPANMISLAR DAHIL.

    Doner: {sembol: {"baz": ..., "durum": ..., "aktif": bool}}

    `aktif` yalnizca bilgi icindir; SUZGEC DEGILDIR. Kapanmis pariteleri
    atmak bu modulun butun amacini ortadan kaldirir.
    """
    if not yenile:
        hit = _cache.peek(ALAN, "semboller")
        if hit and hit[0]:
            return hit[0]
    d = _istek("exchangeInfo", {})
    out = {}
    for s in d.get("symbols", []):
        if s.get("quoteAsset") != "USDT":
            continue
        baz = s.get("baseAsset", "")
        if baz in SABIT or _kaldiracli(baz):
            continue
        out[s["symbol"]] = {"baz": baz, "durum": s.get("status", ""),
                            "aktif": s.get("status") == "TRADING"}
    if out:
        _cache.put(ALAN, "semboller", out)
    return out


def _cerceve(kayitlar) -> pd.DataFrame | None:
    """Binance klines listesini OHLCV cercevesine cevirir.

    Bolunme duzeltmesi YOK ve gerekmiyor: kriptoda hisse bolunmesi diye bir
    sey yok. Nadir token birlesmelerinde (redenomination) fiyat serisinde
    sicrama olur; bunu kesit tarafindaki gunluk budama zaten yakaliyor.
    """
    if not kayitlar:
        return None
    d = pd.DataFrame(kayitlar, columns=[
        "acilis_ms", "Open", "High", "Low", "Close", "Volume", "kapanis_ms",
        "quote_hacim", "islem", "taker_al", "taker_al_quote", "bos"])
    d = d[["acilis_ms", "Open", "High", "Low", "Close", "Volume",
           "quote_hacim", "islem"]]
    for c in d.columns:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d.index = pd.to_datetime(d["acilis_ms"], unit="ms").dt.normalize()
    d = d.drop(columns=["acilis_ms"])
    d = d[~d.index.duplicated(keep="last")].sort_index()
    # Hacmi coin cinsinden degil DOLAR cinsinden tutuyoruz: kesitte
    # karsilastirilan sey buyukluk ve 1 BTC ile 1 DOGE ayni sey degil.
    d["dolar_hacim"] = d["quote_hacim"]
    d = d.dropna(subset=["Close"])
    # Sondaki hacimsiz barlari at.
    #
    # Kapanan bir parite bazen son gununde sifir hacimle ve donmus fiyatla
    # gorunuyor. O bar islem yapilabilir bir gun degil; birakirsak olen bir
    # coini tutmanin sonucu "fiyat degismedi" diye okunur ve model olu
    # pariteyi elde tutmanin cezasini hic gormez. Nadir (bakilan sekiz olu
    # pariteden birinde tek bar) ama yonu tek tarafli.
    hacim = d["dolar_hacim"].to_numpy()
    kes = len(hacim)
    while kes > 0 and not hacim[kes - 1] > 0:
        kes -= 1
    return d.iloc[:kes]


def cek_bir(sembol: str, baslangic_ms: int = 0, bekle: float = BEKLE
            ) -> pd.DataFrame | None:
    """Bir paritenin tum gunluk gecmisi, sayfalanarak.

    Binance istek basina en fazla 1000 bar veriyor. Son sayfa 1000'den
    kisaysa seri bitmistir; kapanmis bir paritede bu, paritenin oldugu
    gundur ve o gun bizim icin degerli bilgidir.
    """
    hepsi = []
    t = int(baslangic_ms)
    while True:
        k = _istek("klines", {"symbol": sembol, "interval": "1d",
                              "limit": 1000, "startTime": t})
        if not k:
            break
        hepsi.extend(k)
        if len(k) < 1000:
            break
        t = int(k[-1][0]) + GUN_MS
        time.sleep(bekle)
    return _cerceve(hepsi)


def cek(liste=None, baslangic: str = "2017-08-01", yenile: bool = False,
        bekle: float = BEKLE, en_az_bar: int = EN_AZ_BAR,
        ilerleme=None) -> dict:
    """Evrenin gunluk barlari. Onbellektekini atlar, yani tekrar calisir."""
    # Sembol listesi de bir ag istegi ve o da hiz sinirina carpabilir.
    # Carparsa pas biter; disari istisna firlatmak, cagiranin elindeki
    # ilerlemeyi de goturur.
    try:
        sembol_bilgi = semboller()
    except HizSiniri as exc:
        return {"istenen": 0, "yazildi": 0, "atlandi": 0, "kisa": 0,
                "hatali": 0, "olu_yazildi": 0, "durduruldu": str(exc)}
    istenen = list(liste or sembol_bilgi)
    bas_ms = int(pd.Timestamp(baslangic).timestamp() * 1000)
    yazildi = atlandi = kisa = hatali = olu_yazildi = 0
    durduruldu = None

    for i, sym in enumerate(istenen):
        anahtar = f"{sym}:{baslangic}"
        if not yenile and _cache.peek(ALAN, anahtar):
            atlandi += 1
            continue
        try:
            d = cek_bir(sym, bas_ms, bekle=bekle)
        except HizSiniri as exc:
            durduruldu = str(exc)
            break
        except Exception:
            hatali += 1
            time.sleep(bekle)
            continue
        if d is None or len(d) < en_az_bar:
            # Kisa gecmisli pariteyi YAZMIYORUZ ama saymiyoruz da degil:
            # kac tanesinin elendigi raporlaniyor, cunku bu esik sessizce
            # bir evren suzgecidir.
            kisa += 1
            time.sleep(bekle)
            continue
        _cache.put(ALAN, anahtar, {"history": d})
        yazildi += 1
        if not sembol_bilgi.get(sym, {}).get("aktif", True):
            olu_yazildi += 1
        time.sleep(bekle)
        if ilerleme and (i + 1) % 50 == 0:
            ilerleme(i + 1, len(istenen), yazildi, atlandi)

    return {"istenen": len(istenen), "yazildi": yazildi, "atlandi": atlandi,
            "kisa": kisa, "hatali": hatali, "olu_yazildi": olu_yazildi,
            "durduruldu": durduruldu}


def oku(sembol: str, baslangic: str = "2017-08-01") -> pd.DataFrame | None:
    """Onbellekten bir paritenin barlari. Ag istegi YOK."""
    hit = _cache.peek(ALAN, f"{sembol}:{baslangic}")
    if not hit or not hit[0]:
        return None
    return (hit[0] or {}).get("history")


def paketler(baslangic: str = "2017-08-01", liste=None) -> dict:
    """kalibrasyon.panel()'in bekledigi {sembol: {"history": df}} sozlugu."""
    out = {}
    for sym in (liste or semboller()):
        d = oku(sym, baslangic)
        if d is not None and len(d):
            out[sym] = {"history": d}
    return out


def kapsam(baslangic: str = "2017-08-01") -> dict:
    """Onbellekte ne var — ve kaci OLU parite.

    Olu sayisi bu raporun en onemli satiri. Sifirsa hayatta kalma yanliligi
    geri gelmis demektir ve butun alistirmanin anlami kalmaz.
    """
    bilgi = semboller()
    n = olu = bar = 0
    ilk = son = None
    for sym, b in bilgi.items():
        d = oku(sym, baslangic)
        if d is None or not len(d):
            continue
        n += 1
        bar += len(d)
        if not b.get("aktif", True):
            olu += 1
        a, z = d.index.min(), d.index.max()
        ilk = a if ilk is None or a < ilk else ilk
        son = z if son is None or z > son else son
    return {"ok": n > 0, "parite": n, "olu": olu, "bar": bar,
            "bar_ortalama": round(bar / n, 1) if n else 0.0,
            "evren": len(bilgi),
            "ilk": str(ilk)[:10] if ilk is not None else None,
            "son": str(son)[:10] if son is not None else None}
