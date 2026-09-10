"""Toplu gunluk bar indirme — uzun gecmis icin.

NEDEN AYRI BIR MODUL

Onbellekteki gunluk barlar iki yillik. 21 gunluk bir ufukta bu, ortusmeyen
yaklasik 24 donem demek: bir modelin ogrenmesi icin az, bir olcumun sifirdan
ayirt etmesi icin daha da az. Ufku uzattikca durum kotulesiyor.

Uzun gecmisi mevcut yoldan cekmek pratik degil. yahoo.fetch her sembol icin
fiyatla birlikte bilanco, nakit akisi, tahmin revizyonlari ve daha bir dizi
seyi de aliyor -- 2700 sembol icin bu on binlerce istek eder. Panel ise
bundle'dan yalnizca "history" okuyor.

NEDEN yfinance DEGIL

Bu modul yazildiginda (06.09.2026) yfinance 1.0'in butun fiyat yollari --
yf.download da, Ticker.history de -- "'NoneType' object is not subscriptable"
ile dusuyordu; kutuphanenin cerez/crumb adimi None donuyordu. Yahoo'nun chart
ucu ayni anda duz bir HTTP istegine cevap veriyordu, yani engellenmis
degildik; arizali olan aradaki kutuphaneydi.

10.09.2026'da kutuphane 1.7.0'a yukseltildi ve fiyat yollari yeniden
calisiyor. Bu modul yine de duruyor ve toplu gecmis icin tercih edilen yol:
sembol basina tek istek, kendi hiz siniri, kendi bolunme duzeltmesi ve
aradaki katman kadar az hareketli parca. On yillik onbellegin tamami bununla
kuruldu; ayrica ayni ariza bir daha oldugunda beklemeye gerek kalmiyor.

Bu modul o yuzden ucu dogrudan konusuyor. Sembol basina tek istek, aralarda
bekleme, ve sonuc mevcut onbellek anahtarina yaziliyor ki panel hicbir sey
degismemis gibi okuyabilsin. Temel veriler bu yolla GUNCELLENMEZ; onlar kendi
akislarinda kaliyor.
"""

from __future__ import annotations

import json
import random
import time
import urllib.error
import urllib.request

import pandas as pd

from .providers import cache as _cache

UC = "https://query1.finance.yahoo.com/v8/finance/chart/"
BASLIK = {"User-Agent": "Mozilla/5.0"}
BEKLE = 0.35
EN_AZ_BAR = 250


class HizSiniri(RuntimeError):
    """429 — istemeden degil, kasten yavaslatiliyoruz."""


def _istek(ticker: str, aralik: str, zaman_asimi: int = 20) -> dict | None:
    url = f"{UC}{ticker}?range={aralik}&interval=1d&events=div%2Csplit"
    r = urllib.request.Request(url, headers=BASLIK)
    try:
        with urllib.request.urlopen(r, timeout=zaman_asimi) as f:
            return json.loads(f.read())
    except urllib.error.HTTPError as exc:
        if exc.code == 429:
            raise HizSiniri(str(exc)[:100]) from exc
        return None
    except Exception:
        return None


def _cerceve(veri: dict) -> pd.DataFrame | None:
    """Chart cevabini OHLCV tablosuna cevirir, bolunmelere gore duzeltir.

    Yahoo ham OHLC'yi duzeltmeden verir, yaninda adjclose'u ayri koyar.
    yfinance'in auto_adjust=True davranisi ile ayni sey yapiliyor: butun
    fiyat sutunlari adjclose/close orani ile olcekleniyor. Bu olmadan her
    bolunme, gostergelerin gercek sandigi devasa bir bosluk olur.
    """
    try:
        r = veri["chart"]["result"][0]
        ts = r["timestamp"]
        q = r["indicators"]["quote"][0]
    except (KeyError, IndexError, TypeError):
        return None

    df = pd.DataFrame({
        "Open": q.get("open"), "High": q.get("high"), "Low": q.get("low"),
        "Close": q.get("close"), "Volume": q.get("volume"),
    }, index=pd.to_datetime(ts, unit="s"))
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    if df.empty:
        return None

    try:
        adj = pd.Series(r["indicators"]["adjclose"][0]["adjclose"],
                        index=pd.to_datetime(ts, unit="s")).reindex(df.index)
        oran = (adj / df["Close"]).where(lambda s: s.notna() & (s > 0), 1.0)
        for c in ("Open", "High", "Low", "Close"):
            df[c] = df[c] * oran
    except (KeyError, IndexError, TypeError):
        pass

    df.index = df.index.tz_localize(None).normalize()
    return df[~df.index.duplicated(keep="last")].sort_index()


def cek(semboller, period: str = "10y", yenile: bool = False,
        bekle: float = BEKLE, en_az_bar: int = EN_AZ_BAR,
        ilerleme=None) -> dict:
    """Barlari ceker ve onbellege yazar.

    Zaten onbellekte olan sembol atlanir (yenile=True degilse), boylece
    yarida kalan bir cekim kaldigi yerden devam eder. 2700 sembol tek
    oturumda bitmeyebilir ve bastan baslamak hiz sinirini bosa harcar.

    Hiz sinirina carpilirsa DURULUR. Israr etmek yasagi uzatiyor; sistemin
    dort gunluk kesintisi bir keresinde tam olarak bundan cikmisti.
    """
    semboller = list(dict.fromkeys(str(s) for s in semboller if s))
    kalan, atlandi = [], 0
    for tk in semboller:
        if not yenile and _cache.peek("yahoo", f"{tk}:{period}"):
            atlandi += 1
        else:
            kalan.append(tk)

    yazildi = hatali = kisa = 0
    durduruldu = None
    for i, tk in enumerate(kalan, 1):
        try:
            veri = _istek(tk, period)
        except HizSiniri as exc:
            durduruldu = f"hiz siniri: {exc}"
            break

        h = _cerceve(veri) if veri else None
        if h is None:
            hatali += 1
        elif len(h) < en_az_bar:
            # Yeni halka arzlar ve kisa gecmisli semboller. Yazmiyoruz:
            # 2 yillik anahtarin uzerine daha kisa bir gecmis yazmak, sessizce
            # veri kaybetmek olur.
            kisa += 1
        else:
            _cache.put("yahoo", f"{tk}:{period}", {"history": h})
            yazildi += 1

        if ilerleme and i % 100 == 0:
            ilerleme(i, len(kalan), yazildi, hatali + kisa)
        time.sleep(bekle + random.random() * bekle)

    return {"ok": yazildi > 0, "period": period, "istenen": len(semboller),
            "atlandi": atlandi, "yazildi": yazildi, "hatali": hatali,
            "kisa": kisa, "durduruldu": durduruldu}


def endeks(sembol: str = "SPY", period: str = "10y",
           yenile: bool = False) -> pd.DataFrame | None:
    """Karsilastirma endeksini ayni yoldan ceker.

    Etiket 'endeksten iyi mi' sorusunu soruyor, yani endeksin gecmisi
    hisselerinkiyle ayni uzunlukta olmali. Endeks kisa kalirsa etiketsiz
    satirlar sessizce dusuyor -- gun ici tarafta bir kez tam bu oldu ve
    satirlarin ucte biri kayboldu.

    Ayri bir onbellek alaninda duruyor (yahoo_bench), cunku onu okuyan taraf
    bundle degil dogrudan tabloyu bekliyor.
    """
    anahtar = f"{sembol}:{period}"
    if not yenile:
        hit = _cache.peek("yahoo_bench", anahtar)
        if hit and hit[0] is not None and len(hit[0]):
            return hit[0]
    h = _cerceve(_istek(sembol, period) or {})
    if h is None or len(h) < EN_AZ_BAR:
        return None
    _cache.put("yahoo_bench", anahtar, h)
    return h


def kapsam(semboller, period: str = "10y") -> dict:
    """Onbellekte bu period icin ne kadar gecmis var?"""
    barlar = []
    ilk = son = None
    for tk in dict.fromkeys(str(s) for s in semboller if s):
        hit = _cache.peek("yahoo", f"{tk}:{period}")
        if not hit:
            continue
        h = (hit[0] or {}).get("history")
        if h is None or not len(h):
            continue
        barlar.append(len(h))
        idx = pd.to_datetime(h.index)
        ilk = idx.min() if ilk is None else min(ilk, idx.min())
        son = idx.max() if son is None else max(son, idx.max())

    if not barlar:
        return {"ok": False, "period": period, "hisse": 0}
    s = pd.Series(barlar)
    return {"ok": True, "period": period, "hisse": len(barlar),
            "bar_ortanca": int(s.median()), "bar_en_az": int(s.min()),
            "bar_en_cok": int(s.max()),
            "ilk": str(ilk)[:10], "son": str(son)[:10]}
