"""Temel verinin gunluk arsivi — gelecekteki egitimin hammaddesi.

SORUN
-----
Yahoo `info` alanlari yalnizca BUGUNUN degerini verir. Gecmise donuk temel
veri hicbir sekilde alinamaz. Bu yuzden gecmise donuk panel (backfill) sadece
11 FIYAT faktoruyle uretilebildi; degerleme, karlilik, analist ve buyume
faktorlerinin -- yani agirligin yarisindan fazlasinin -- gecmisi yok.

COZUM
-----
Veri zaten her taramada cekiliyor, kullaniliyor ve ATILIYOR. Burada tarih
damgalanip saklaniyor. Bugun baslanirsa alti ay sonra 28 faktorun TAMAMIYLA
egitim yapilabilir; baslanmazsa alti ay sonra da ayni yerde olunur. Erteleme
maliyeti en yuksek olan is budur ve maliyeti gunde ~100 KB'dir.

NEDEN FAKTOR SKORU DEGIL, HAM ALAN
----------------------------------
Feature store zaten `raw_<faktor>` sutunlarini tutuyor. Ama bir faktorun
FORMULU degisirse (ki degisecek -- IC olcumleri geldiginde tanimlar
guncellenecek) o sutunlar yeniden hesaplanamaz. Ham alanlar saklanirsa yeni
tanim gecmise de uygulanabilir. Ayrica alan bazinda saklamak, bir faktorun
hangi girdisinden dolayi bozuldugunu gormeyi mumkun kilar.

DOSYA DUZENI
------------
    data/fundamentals/temel_YYYY-MM-DD.csv.gz

Gzip zorunlu degil ama fark buyuk: ~2400 satir x 45 sutun sikistirilmadan
gunde ~1 MB, sikistirilmis ~90 KB. Yilda 350 MB ile 30 MB arasindaki fark.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STORE = ROOT / "data" / "fundamentals"

# Faktorlerin gercekten okudugu alanlar (src/factors.py taranarak cikarildi).
# Yeni bir faktor yeni bir alan kullanmaya baslarsa buraya da eklenmeli;
# aksi halde o alanin gecmisi birikmez.
INFO_FIELDS = (
    # kimlik / olcek
    "sector", "industry", "longName", "currency", "marketCap", "currentPrice",
    # degerleme
    "trailingPE", "forwardPE", "priceToBook", "priceToSalesTrailing12Months",
    "enterpriseToEbitda", "trailingPegRatio",
    # karlilik
    "grossMargins", "operatingMargins", "profitMargins",
    "returnOnEquity", "returnOnAssets", "netIncomeToCommon",
    # buyume
    "revenueGrowth", "earningsGrowth", "earningsQuarterlyGrowth", "totalRevenue",
    # bilanco / nakit
    "totalCash", "totalDebt", "debtToEquity", "currentRatio", "quickRatio",
    "freeCashflow", "operatingCashflow",
    # analist
    "recommendationMean", "recommendationKey", "numberOfAnalystOpinions",
    # sahiplik / acik pozisyon
    "heldPercentInsiders", "heldPercentInstitutions",
    "shortPercentOfFloat", "shortRatio",
)

# price_targets sozlugundeki alanlar
TARGET_FIELDS = ("current", "low", "high", "mean", "median")


def _num(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if f != f else f          # NaN'i None'a cevir


def _earnings_date(bundle: dict) -> str | None:
    """Bir sonraki bilanco tarihi (varsa).

    Bu alan iki ise yarar: (1) 21 gunluk ufukta getiriyi en cok belirleyen tek
    olay budur, modelin gormesi gerekir; (2) kart uzerinde "bilancoya N gun"
    yazabilmek icin gunluk kaydi gerekir.
    """
    cal = bundle.get("calendar") or {}
    for key in ("Earnings Date", "earningsDate"):
        v = cal.get(key)
        if isinstance(v, (list, tuple)) and v:
            v = v[0]
        if v is None:
            continue
        try:
            return pd.Timestamp(v).strftime("%Y-%m-%d")
        except Exception:
            continue
    return None


def extract(ticker: str, bundle: dict) -> dict:
    """Bir hissenin o gunku temel verisini duz bir satira cevirir."""
    info = bundle.get("info") or {}
    row: dict[str, Any] = {"ticker": ticker}

    for f in INFO_FIELDS:
        v = info.get(f)
        row[f] = v if isinstance(v, str) else _num(v)

    tgt = bundle.get("price_targets") or {}
    for f in TARGET_FIELDS:
        row[f"target_{f}"] = _num(tgt.get(f))

    row["earnings_date"] = _earnings_date(bundle)

    # Fiyat serisinden iki dayanak: kapanis ve 30 gunluk dolar hacmi. Temel
    # veriyle birlikte saklanmasi, arsivin tek basina kullanilabilmesini saglar.
    h = bundle.get("history")
    if h is not None and len(h):
        try:
            row["close"] = _num(h["Close"].iloc[-1])
            row["dollar_volume_30d"] = _num(
                (h["Close"] * h["Volume"]).tail(30).mean())
            row["bars"] = int(len(h))
        except Exception:
            pass
    return row


def save_snapshot(rows: list[dict], date: str | None = None) -> Path | None:
    """Gunun temel veri arsivini yazar.

    ml.save_snapshot ile ayni kural: ayni gun ikinci tarama UZERINE YAZMAZ,
    BIRLESTIRIR. Donusumlu tarama her turda evrenin farkli bir dilimini
    gezdigi icin uzerine yazmak, o gun daha once toplanan hisseleri silerdi.
    """
    rows = [r for r in rows if r and r.get("ticker")]
    if not rows:
        return None

    STORE.mkdir(parents=True, exist_ok=True)
    date = date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    path = STORE / f"temel_{date}.csv.gz"

    df = pd.DataFrame(rows)
    df.insert(0, "snapshot_date", date)

    if path.exists():
        try:
            old = pd.read_csv(path)
            keep = old[~old["ticker"].isin(df["ticker"])]
            df = pd.concat([df, keep], ignore_index=True, sort=False)
        except Exception:
            pass

    df = df.sort_values("ticker")
    df.to_csv(path, index=False, encoding="utf-8", compression="gzip")
    return path


def info() -> dict:
    """Arsivin durumu — pano ve durum betigi icin."""
    if not STORE.exists():
        return {"snapshots": 0, "rows": 0}
    files = sorted(STORE.glob("temel_*.csv.gz"))
    if not files:
        return {"snapshots": 0, "rows": 0}
    days = [p.stem.replace("temel_", "").replace(".csv", "") for p in files]
    size = sum(p.stat().st_size for p in files)
    rows = 0
    try:
        rows = len(pd.read_csv(files[-1]))
    except Exception:
        pass
    return {
        "snapshots": len(files),
        "first_date": days[0],
        "last_date": days[-1],
        "rows_last": rows,
        "mb": round(size / 1024 / 1024, 1),
        "fields": len(INFO_FIELDS) + len(TARGET_FIELDS) + 4,
    }


def load_all() -> pd.DataFrame:
    """Tum arsivi tek panel olarak okur (ileride egitim icin)."""
    if not STORE.exists():
        return pd.DataFrame()
    parts = []
    for p in sorted(STORE.glob("temel_*.csv.gz")):
        try:
            parts.append(pd.read_csv(p))
        except Exception:
            continue
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def days_to_earnings(ticker_row: dict, as_of: str | None = None) -> int | None:
    """Bilancoya kalan gun (negatifse gecmis)."""
    ed = ticker_row.get("earnings_date")
    if not ed:
        return None
    try:
        d0 = pd.Timestamp(as_of or datetime.now(timezone.utc).strftime("%Y-%m-%d"))
        return int((pd.Timestamp(ed) - d0).days)
    except Exception:
        return None
