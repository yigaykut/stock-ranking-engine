"""Yedek fiyat kaynagi (Nasdaq) — Yahoo hiz siniri devredeyken devreye girer.

NEDEN
-----
Olculen gercek: gunluk taramada 800 sembolun yalnizca ~440'i (%55) Yahoo'dan
cekilebiliyor. Kalani onbellekteki BAYAT fiyatla skorlaniyor. Toplam agirligin
~%40'i fiyat serisinden hesaplandigi icin bu, siralamanin yarisinin dunun
verisiyle uretilmesi demek.

NEDEN BU KAYNAK
---------------
api.nasdaq.com zaten kotasyon listesi icin kullaniliyor ve calisiyor: anahtar
istemiyor, ayri bir hesap gerektirmiyor, Yahoo'dan BAGIMSIZ bir uc. Denenen
diger ucretsiz secenek (Stooq) artik JavaScript tabanli bot dogrulamasi
istiyor; bunu asmak hem dogru degil hem de kirilgan olurdu, o yuzden
kullanilmadi.

ONEMLI FARK — DUZELTILMEMIS SERI
--------------------------------
Yahoo serisi `auto_adjust=True` ile geliyor: temettu ve bolunmelere gore
duzeltilmis. Nasdaq ucu HAM fiyat veriyor. Bolunme, seride yapay bir sicrama
birakir ve momentum faktorlerini yanlis hesaplatir.

Bu yuzden iki savunma var:
  1. Seride tek gunde %35'ten buyuk bir sicrama varsa (tipik bolunme izi)
     paket KULLANILMAZ -- bayat ama tutarli Yahoo verisi tercih edilir.
  2. Paket icinde `_price_source` isaretlenir; teshis ciktisi kac hissenin
     yedek kaynaktan geldigini gosterir.

Temettu duzeltmesinin yoklugu 1-2 yillik pencerede yuksek temettu odeyen
hisselerde birkac yuzde puanlik fark yaratir. Bu, HIC VERI OLMAMASINDAN veya
bir hafta eski veriden daha iyidir, ama bedava degildir.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from .cache import get_or_fetch

BASE = ("https://api.nasdaq.com/api/quote/{sym}/historical"
        "?assetclass=stocks&fromdate={frm}&todate={to}&limit=9999")

_PERIOD_DAYS = {"6mo": 190, "1y": 380, "2y": 760, "5y": 1900, "max": 7300}

# Tek gunde bu orandan buyuk bir sicrama, duzeltilmemis bolunme demektir.
SPLIT_JUMP = 0.35

# Yahoo gecmisini bu dilimle veriyor; yedek kaynak da ayni dilime cevrilmezse
# iki seri birbirine hizalanamiyor (bkz. _to_frame icindeki not).
_EXCHANGE_TZ = "America/New_York"


class _Unusable(Exception):
    pass


def _money(v: Any) -> float | None:
    if v is None:
        return None
    s = str(v).replace("$", "").replace(",", "").strip()
    if not s or s in ("N/A", "--"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _parse(payload: dict) -> pd.DataFrame | None:
    rows = ((payload or {}).get("data") or {}).get("tradesTable", {}).get("rows")
    if not rows:
        return None
    recs = []
    for r in rows:
        try:
            d = pd.Timestamp(r["date"])
        except Exception:
            continue
        rec = {"Date": d, "Open": _money(r.get("open")), "High": _money(r.get("high")),
               "Low": _money(r.get("low")), "Close": _money(r.get("close")),
               "Volume": _money(r.get("volume"))}
        if rec["Close"] is None:
            continue
        recs.append(rec)
    if not recs:
        return None
    df = pd.DataFrame(recs).set_index("Date").sort_index()
    # Ayni gun birden fazla satir gelebiliyor
    df = df[~df.index.duplicated(keep="last")]
    if not len(df):
        return None

    # SAAT DILIMI: Yahoo gecmisi America/New_York ile DILIMLI gelir, buradaki
    # tarihler ise dilimsiz. Ikisi karisinca goreli guc hesabi endeksle
    # hizalanirken "Cannot join tz-naive with tz-aware DatetimeIndex" atiyor ve
    # hisse faktor asamasinda DUSUYOR. Yani yedek kaynak veriyi basariyla
    # kurtariyor, sonra kurtardigi her hisse sessizce eleniyordu (31.08.2026
    # taramasinda kurtarilan 248 hissenin tamami). Borsa saatiyle yerellestirmek
    # dogru olani: bunlar ABD borsasinin gunluk barlari.
    if df.index.tz is None:
        df.index = df.index.tz_localize(_EXCHANGE_TZ)
    else:
        df.index = df.index.tz_convert(_EXCHANGE_TZ)
    return df


def _has_split_jump(close: pd.Series) -> bool:
    """Duzeltilmemis bolunme izi var mi?"""
    r = close.pct_change().abs().dropna()
    return bool(len(r) and (r > SPLIT_JUMP).any())


def fetch_history(ticker: str, period: str = "2y", use_cache: bool = True,
                  ttl_seconds: int = 6 * 3600, timeout: int = 25
                  ) -> pd.DataFrame | None:
    """Gunluk OHLCV. Bulunamaz veya guvenilmezse None (istisna atmaz)."""
    def _do() -> pd.DataFrame | None:
        import requests
        days = _PERIOD_DAYS.get(period, 760)
        to = pd.Timestamp.utcnow().normalize()
        frm = to - pd.Timedelta(days=days)
        url = BASE.format(sym=str(ticker).upper().replace("-", "."),
                          frm=frm.strftime("%Y-%m-%d"), to=to.strftime("%Y-%m-%d"))
        try:
            r = requests.get(url, timeout=timeout, headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
                "Accept": "application/json"})
            if r.status_code != 200:
                return None
            return _parse(r.json())
        except Exception:
            return None

    df = get_or_fetch("nasdaq", f"{ticker}:{period}", _do,
                      ttl_seconds=ttl_seconds, enabled=use_cache,
                      # Bos/kisa sonuc ONBELLEGE YAZILMAZ: gecici bir hata
                      # aksi halde TTL boyunca "veri yok" olarak donerdi.
                      should_cache=lambda v: v is not None and len(v) >= 60)
    if df is None or len(df) < 60:
        return None
    if _has_split_jump(df["Close"]):
        return None                    # duzeltilmemis bolunme -> kullanma
    return df


def as_bundle(ticker: str, period: str = "2y",
              base: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """Seriyi tarama paketine cevirir / mevcut paketin fiyatini tazeler.

    base verilirse (onbellekteki eski Yahoo paketi) temel veri alanlari
    KORUNUR, yalnizca `history` degistirilir. Sonuc: taze teknik + birkac gun
    eski temel veri. Ikisinin de bayat olmasindan iyidir.
    """
    df = fetch_history(ticker, period)
    if df is None:
        return None

    # Onbellekte dilim duzeltmesinden ONCE yazilmis seriler var; onlar okununca
    # yine dilimsiz geliyor. Duzeltmeyi burada da uyguluyoruz ki eski kayitlar
    # TTL dolana kadar hisseyi elemeye devam etmesin.
    try:
        if getattr(df.index, "tz", None) is None:
            df = df.copy()
            df.index = df.index.tz_localize(_EXCHANGE_TZ)
    except (TypeError, AttributeError):
        return None

    bundle: dict[str, Any] = dict(base) if base else {
        "info": {}, "recommendations": None, "eps_trend": None,
        "eps_revisions": None, "earnings_history": None,
        "growth_estimates": None, "price_targets": {}, "calendar": None,
        "cashflow": None, "balance_sheet": None, "income": None,
    }
    bundle["history"] = df
    bundle["_price_source"] = "nasdaq"
    bundle["_fundamentals_source"] = "yahoo_onbellek" if base else "yok"
    return bundle
