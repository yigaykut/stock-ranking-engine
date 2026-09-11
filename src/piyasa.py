"""Piyasanin kendisi: yon tahmin edilebiliyor mu.

NEDEN AYRI BIR SORU
-------------------
Kesitsel modelin etiketi akran-gorelidir, yani piyasayi TANIM GEREGI aradan
cikarir. "Sektorunden %2 iyi" ile "hesabin buyudu" arasindaki fark tam olarak
budur: sektor %12 dustuyse, akranini %2 gecen bir portfoy %10 kaybeder.

Bu modul o eksik parcayi olcer. Skor uretmez, hisse secmez; yalnizca sunu
sorar: elimizdeki bilgiyle endeksin onumuzdeki ayi hakkinda olculebilir bir
sey soyleyebiliyor muyuz.

`regime.py` ile karistirilmamali. O modul bugunun rejimini ETIKETLER ve
bilerek tahmin yapmaz ("piyasayi zamanlamak icin degil, baglam vermek icin").
Burasi tahminin mumkun olup olmadigini olcuyor.

ORNEKLEM SORUNU — ONCE BU
-------------------------
On yil, 21 gunluk ortusmeyen donem cinsinden yaklasik 120 gozlem demek. Bu
kadari, kesitte iki bin hisse x bes yuz gunle calismaya alismis bir olcum
disiplini icin cok azdir ve bir daha artmaz: piyasa zamanlamasinda orneklem
takvimle sinirlidir, evreni buyutmek ise yaramaz.

Pratikte anlami su: |t| = 2 civari bir sonuc burada kanit degildir. Gunluk
gozlemler 2500 tane ama bagimsiz degil, ve Newey-West duzeltmesi bunu
hesaba katiyor -- katmasaydi her sey anlamli gorunurdu.

GENISLIK NEDEN ONEMLI
---------------------
Endeks birkac dev hisseyle ayakta durabilir; hisselerin yuzde kaci kendi
ortalamasinin ustunde sorusu duramaz. Kucuk sirket avinda genislik endeksten
daha anlamlidir ve elimizde 2600 hissenin on yillik bari var, yani bu sayi
gecmise donuk olarak hesaplanabiliyor.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import requests

from .providers import cache as _cache

UC = "https://query1.finance.yahoo.com/v8/finance/chart/"
ALAN = "piyasa"


def _seri(sembol: str, period: str = "10y", yenile: bool = False
          ) -> pd.Series | None:
    """Bir endeksin kapanis serisi. Onbellekten, yoksa chart ucundan."""
    anahtar = f"{sembol}:{period}"
    if not yenile:
        hit = _cache.peek(ALAN, anahtar)
        if hit and hit[0] is not None:
            d = hit[0]
            return pd.Series(d["deger"], index=pd.DatetimeIndex(d["gun"]))
    r = requests.get(UC + sembol.replace("^", "%5E"),
                     headers={"User-Agent": "Mozilla/5.0"},
                     params={"range": period, "interval": "1d"}, timeout=30)
    r.raise_for_status()
    res = (r.json().get("chart") or {}).get("result") or []
    if not res:
        return None
    ts = res[0].get("timestamp") or []
    kapanis = (((res[0].get("indicators") or {}).get("quote") or [{}])[0]
               .get("close") or [])
    s = pd.Series(kapanis, index=pd.to_datetime(ts, unit="s")).dropna()
    s.index = s.index.tz_localize(None).normalize()
    s = s[~s.index.duplicated(keep="last")]
    _cache.put(ALAN, anahtar, {"gun": [str(x) for x in s.index],
                               "deger": s.to_numpy().tolist()})
    return s


def genislik(bundles, pencere: int = 50, period: str = "10y") -> pd.Series:
    """Gun bazinda, hisselerin yuzde kaci kendi N gunluk ortalamasinin ustunde.

    Hisse hisse okunup gun bazinda SAYAC biriktiriliyor; (gun x hisse) bir
    matris kurulmuyor. 2600 hisse x 2500 gun boolean bile birkac yuz megabayt
    eder ve bu makinede tepe bellek en dar kaynak.
    """
    ust = {}
    toplam = {}
    for tk in sorted(bundles):
        h = (bundles[tk] or {}).get("history")
        if h is None or len(h) < pencere + 5:
            continue
        c = h["Close"].astype(float)
        idx = pd.DatetimeIndex(c.index)
        try:
            idx = idx.tz_localize(None) if idx.tz is None else idx.tz_convert(None)
        except (TypeError, AttributeError):
            pass
        c.index = idx.normalize()
        ma = c.rolling(pencere, min_periods=pencere).mean()
        gecerli = ma.notna()
        if not gecerli.any():
            continue
        uz = (c > ma)[gecerli]
        for g, v in zip(uz.index.to_numpy(), uz.to_numpy()):
            toplam[g] = toplam.get(g, 0) + 1
            if v:
                ust[g] = ust.get(g, 0) + 1
    if not toplam:
        return pd.Series(dtype=float)
    gunler = pd.DatetimeIndex(sorted(toplam))
    return pd.Series([ust.get(g, 0) / toplam[g] for g in gunler.to_numpy()],
                     index=gunler)


def ozellikler(spy: pd.Series, vix: "pd.Series | None" = None,
               gen: "pd.Series | None" = None) -> pd.DataFrame:
    """Gun bazinda piyasa ozellikleri. Hepsi GERIYE bakar."""
    d = pd.DataFrame(index=spy.index)
    ma50 = spy.rolling(50, min_periods=50).mean()
    ma200 = spy.rolling(200, min_periods=200).mean()
    d["ma50_uzaklik"] = spy / ma50 - 1.0
    d["ma200_uzaklik"] = spy / ma200 - 1.0
    d["ma200_egim"] = ma200 / ma200.shift(21) - 1.0
    getiri = spy.pct_change()
    d["getiri21"] = spy / spy.shift(21) - 1.0
    d["getiri63"] = spy / spy.shift(63) - 1.0
    d["oynaklik20"] = getiri.rolling(20, min_periods=20).std()
    d["oynaklik_yuzdelik"] = (d["oynaklik20"]
                              .rolling(252, min_periods=120).rank(pct=True))
    # Zirveden geri cekilme: dususu tarif eden en dogrudan sey.
    d["tepe_uzaklik"] = spy / spy.rolling(252, min_periods=120).max() - 1.0

    if vix is not None:
        v = vix.reindex(d.index).ffill()
        d["vix"] = v
        d["vix_yuzdelik"] = v.rolling(252, min_periods=120).rank(pct=True)
        d["vix_degisim5"] = v / v.shift(5) - 1.0
    if gen is not None:
        g = gen.reindex(d.index).ffill()
        d["genislik"] = g
        d["genislik_degisim10"] = g - g.shift(10)
        # Endeks yukarida ama genislik dusukse, yukselisi birkac hisse
        # tasiyor demektir. Ikisinin arasindaki acikligi tek sayiya indiriyor.
        d["genislik_ayrisma"] = g - (d["ma50_uzaklik"] > 0).astype(float)
    return d


def onculuk(spy: pd.Series, oz: pd.DataFrame, ufuk: int = 21,
            en_az: int = 250) -> dict:
    """Her ozellik, endeksin ILERI getirisini tahmin ediyor mu.

    Spearman korelasyonu ve Newey-West duzeltmeli t. Gecikme ufuk kadar:
    ardisik gunlerin ileri pencereleri neredeyse tamamen ortusuyor ve bunu
    hesaba katmayan bir t degeri burada dort-bes kat sisik cikar.

    Ayrica DUSUS ozel olarak soruluyor: onumuzdeki ufukta endeks %5'ten
    fazla dusecek mi. Ortalama getiriyi tahmin etmekle, kotu ayi yakalamak
    ayni sey degil ve kullanici icin ikincisi daha onemli.
    """
    from .faktor_zaman import newey_west_t
    from scipy.stats import spearmanr

    ileri = spy.shift(-ufuk) / spy - 1.0
    ortak = oz.index.intersection(ileri.dropna().index)
    if len(ortak) < en_az:
        return {"ok": False, "reason": f"{len(ortak)} gun"}
    y = ileri.reindex(ortak).to_numpy()
    dusus = (y < -0.05).astype(float)

    out = []
    for c in oz.columns:
        x = oz[c].reindex(ortak).to_numpy(dtype=float)
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < en_az or np.nanstd(x[m]) < 1e-12:
            continue
        rho = float(spearmanr(x[m], y[m]).statistic)
        # t, gunluk katki serisinin ortalamasindan: standartlastirilmis
        # ozellik carpi getiri, yani basit bir tek-faktor portfoyunun gunluk
        # getirisi. Bu seri uzerinden Newey-West, korelasyonun kendisine
        # dogrudan t vermekten daha durust: ortusmeyi tasiyan sey bu.
        xs = (x[m] - np.nanmean(x[m])) / (np.nanstd(x[m]) + 1e-12)
        katki = xs * y[m]
        tv, _, _ = newey_west_t(katki, lag=max(1, int(ufuk)))
        rho_d = float(spearmanr(x[m], dusus[m]).statistic)
        out.append({
            "ozellik": c, "n": int(m.sum()),
            "rho": round(rho, 4),
            "t_nw": None if not np.isfinite(tv) else round(float(tv), 2),
            "rho_dusus": round(rho_d, 4),
        })
    out.sort(key=lambda r: -abs(r["t_nw"] or 0))
    bagimsiz = int(len(ortak) / max(ufuk, 1))
    return {"ok": True, "ufuk": ufuk, "gun": int(len(ortak)),
            "bagimsiz_donem": bagimsiz,
            "dusus_orani": round(float(dusus.mean()), 4),
            "ozellikler": out}
