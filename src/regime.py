"""Piyasa rejimi — siralamanin hangi ortamda uretildigi.

NEDEN
-----
Toplam agirligin buyuk bolumu trend/momentum egilimli. Momentum, dusus
rejiminde ve donus noktalarinda en kotu performansini verir; bu, literaturun
en iyi belgelenmis bulgularindan biridir. Sistem su ana kadar rejimden
HABERSIZDI ve her gun ayni ozguvenle liste uretiyordu.

Bu modul piyasayi ZAMANLAMAK icin degil, BAGLAM vermek icin var. Hicbir skoru
degistirmez, hicbir hisseyi elemez. Iki isi vardir:

  1. Panoda tek satirlik bir uyari: "bu siralama dusus rejiminde uretildi".
  2. Her anlik goruntuye rejim etiketi yazmak. Alti ay sonra IC'yi rejime
     GORE olcebilmenin tek yolu, o gunun rejimini o gun kaydetmis olmaktir.
     Sonradan yeniden uretilemez cunku evren ve kapsama degisir.

OLCULEN UC SEY
--------------
  trend    : endeks kendi 50 ve 200 gunluk ortalamasinin neresinde
  genislik : taranan hisselerin yuzde kaci kendi 50 gunluk ortalamasinin uzerinde
             (endeks birkac dev hisseyle ayakta duruyorsa bu sayi dusuk cikar --
             kucuk sirket avinda endeksten daha anlamlidir)
  oynaklik : endeksin 20 gunluk gerceklesmis oynakligi, kendi 1 yillik
             dagilimindaki yuzdelik dilimi
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"
HISTORY = DATA / "rejim_gecmisi.json"


def _pct(a: float, b: float) -> float | None:
    if not b:
        return None
    return round(100.0 * (a / b - 1.0), 2)


def _olcumler(s: pd.Series) -> dict | None:
    """Serinin SON gunune ait rejim olcumleri. Yalnizca gecmise bakar.

    compute() ile labels_for_dates() ayni kurali kullansin diye ayri duruyor.
    Iki ayri kopya olsaydi biri degistiginde gecmis olcum bugunkuyle
    kiyaslanamaz hale gelirdi -- ve bunu fark etmek imkansiz olurdu.
    """
    if s is None or len(s) < 210:
        return None
    px = float(s.iloc[-1])
    ma50 = float(s.rolling(50).mean().iloc[-1])
    ma200 = float(s.rolling(200).mean().iloc[-1])
    ma200_prev = float(s.rolling(200).mean().iloc[-21])

    ret = s.pct_change().dropna()
    vol20 = float(ret.tail(20).std() * np.sqrt(252))
    vol_series = ret.rolling(20).std().dropna().tail(252) * np.sqrt(252)
    vol_pct = float((vol_series <= vol20).mean() * 100) if len(vol_series) > 30 else None

    m = {
        "price": round(px, 2),
        "vs_ma50_pct": _pct(px, ma50),
        "vs_ma200_pct": _pct(px, ma200),
        "ma200_slope_pct": _pct(ma200, ma200_prev),
        "vol20_annual_pct": round(100 * vol20, 1),
        "vol_percentile": round(vol_pct, 0) if vol_pct is not None else None,
    }
    m["label"] = _etiket(px > ma50, px > ma200, (m["ma200_slope_pct"] or 0) > 0)
    m["stressed"] = bool(vol_pct is not None and vol_pct >= 80)
    return m


def _etiket(above50: bool, above200: bool, rising: bool) -> str:
    """Rejim kurali. Tek yerde durur; genislik (breadth) etiketi DEGISTIRMEZ,
    yalnizca aciklama metnine uyari ekler -- bu yuzden etiket, endeks fiyat
    gecmisinden gecmise donuk olarak da uretilebilir."""
    if above50 and above200 and rising:
        return "YUKSELIS"
    if not above200 and not rising:
        return "DUSUS"
    return "GECIS"


def compute(bench_close: pd.Series | None,
            breadth_pct: float | None = None) -> dict:
    """Rejim ozeti. bench_close: endeksin kapanis serisi."""
    out: dict = {"date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                 "breadth_pct": breadth_pct}

    s = (pd.to_numeric(bench_close, errors="coerce").dropna()
         if bench_close is not None else None)
    m = _olcumler(s) if s is not None else None
    if m is None:
        out.update(label="BILINMIYOR", label_tr="Rejim olculemedi",
                   detail_tr="Endeks gecmisi yetersiz.")
        return out

    label = m.pop("label")
    stressed = m.pop("stressed")
    vol_pct = m["vol_percentile"]
    out.update(m)

    if label == "YUKSELIS":
        tr = "Yukselis rejimi"
        detail = ("Endeks 50 ve 200 gunluk ortalamasinin uzerinde, uzun ortalama "
                  "yukseliyor. Momentum egilimli siralamanin tarihsel olarak en "
                  "iyi calistigi ortam.")
    elif label == "DUSUS":
        tr = "Dusus rejimi"
        detail = ("Endeks 200 gunluk ortalamasinin ALTINDA ve ortalama dusuyor. "
                  "Momentum/trend agirlikli siralamalar bu ortamda tarihsel "
                  "olarak en kotu sonucu verir. Liste yine uretilir, ama "
                  "guveni dusuk okunmalidir.")
    else:
        tr = "Gecis / yatay rejim"
        detail = ("Endeks ortalamalarin arasinda: yon belirsiz. Donus "
                  "noktalarinda trend sinyalleri en cok yaniltir.")

    if stressed:
        tr += " (yuksek oynaklik)"
        detail += (f" Ayrica oynaklik son bir yilin ust %{100 - (vol_pct or 0):.0f}'lik "
                   f"diliminde -- gunluk sira degisimleri normalden buyuk olacaktir.")

    if breadth_pct is not None and breadth_pct < 40 and label == "YUKSELIS":
        detail += (f" DIKKAT: hisselerin yalnizca %{breadth_pct:.0f}'i kendi 50 gunluk "
                   f"ortalamasinin uzerinde. Endeks birkac buyuk hisseyle ayakta; "
                   f"kucuk sirket tarafinda ortam gorundugu kadar iyi degil.")

    out.update(label=label, label_tr=tr, detail_tr=detail)
    return out


def breadth(bundles: dict, window: int = 50) -> float | None:
    """Taranan hisselerin yuzde kaci kendi N gunluk ortalamasinin uzerinde."""
    hits = total = 0
    for b in bundles.values():
        h = (b or {}).get("history")
        if h is None or len(h) < window + 5 or "Close" not in h:
            continue
        c = pd.to_numeric(h["Close"], errors="coerce").dropna()
        if len(c) < window + 1:
            continue
        ma = c.rolling(window).mean().iloc[-1]
        if ma != ma:
            continue
        total += 1
        hits += int(float(c.iloc[-1]) > float(ma))
    return round(100.0 * hits / total, 1) if total >= 50 else None


def record(state: dict) -> None:
    """Gunun rejimini gecmise yazar (sonradan yeniden uretilemez)."""
    hist = {}
    if HISTORY.exists():
        try:
            hist = json.loads(HISTORY.read_text(encoding="utf-8")) or {}
        except Exception:
            hist = {}
    hist[state.get("date") or datetime.now(timezone.utc).strftime("%Y-%m-%d")] = {
        k: state.get(k) for k in
        ("label", "vs_ma50_pct", "vs_ma200_pct", "ma200_slope_pct",
         "vol20_annual_pct", "vol_percentile", "breadth_pct")
    }
    DATA.mkdir(parents=True, exist_ok=True)
    HISTORY.write_text(json.dumps(hist, ensure_ascii=False, indent=0, sort_keys=True),
                       encoding="utf-8")


def history() -> dict:
    if not HISTORY.exists():
        return {}
    try:
        return json.loads(HISTORY.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def labels_for_dates(bench_close: pd.Series | None,
                     dates: "list[str]") -> dict[str, str]:
    """Verilen tarihlerin her biri icin GECMISE DONUK rejim etiketi.

    NEDEN YAPILABILIYOR: etiket kurali (_etiket) yalnizca endeksin kendi fiyat
    gecmisine bakar -- 50/200 gunluk ortalama ve 200 gunluk egim. Genislik
    (breadth) etikete girmez, sadece aciklama metnine uyari ekler. Bu yuzden
    gecmise donuk panelin 73 tarihi de bugunku canli hesapla AYNI kuralla
    etiketlenebilir. Modul basindaki "rejim sonradan uretilemez" notu
    genislik icin dogru, etiket icin degil.

    ILERIYE BAKIS YOK: her tarih icin seri o tarihte KESILIR.

    Doner: {'2026-01-15': 'YUKSELIS', ...}  (olculemeyen tarih atlanir)
    """
    if bench_close is None or not len(bench_close):
        return {}
    s = pd.to_numeric(bench_close, errors="coerce").dropna()
    if s.empty:
        return {}

    # Endeks serisi dilimli (America/New_York), anlik goruntu tarihleri duz
    # metin. Kiyaslama icin indeks dilimsiz gune indirilir.
    idx = pd.to_datetime(s.index)
    try:
        idx = idx.tz_localize(None) if idx.tz is None else idx.tz_convert(None)
    except (TypeError, AttributeError):
        pass
    s = pd.Series(s.to_numpy(), index=pd.DatetimeIndex(idx).normalize())
    s = s[~s.index.duplicated(keep="last")].sort_index()

    out: dict[str, str] = {}
    for d in dates:
        ts = pd.to_datetime(str(d), errors="coerce")
        if ts is pd.NaT:
            continue
        kesit = s[s.index <= ts.normalize()]
        m = _olcumler(kesit)
        if m:
            out[str(d)] = m["label"]
    return out
