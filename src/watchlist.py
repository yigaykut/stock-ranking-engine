"""Izleme listesi / pozisyon takibi.

Depolama: data/watchlist.json  (insan tarafindan okunabilir, elle duzenlenebilir)

Her kayit:
    ticker          sembol
    added_date      listeye eklenme tarihi
    entry_price     alis fiyati (None ise sadece izleniyor, pozisyon yok)
    quantity        adet (istege bagli)
    score_at_entry  eklendigi andaki toplam etki puani (bozulma takibi icin)
    note            serbest not

Ayrica gunluk anlik goruntuler data/watch_history.csv dosyasinda birikir;
boylece fiyat, risk seviyesi ve stop'un zaman icindeki seyri izlenebilir
ve ileride model egitimine girdi olur.
"""
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DATA = Path(__file__).resolve().parents[1] / "data"
WATCHLIST = DATA / "watchlist.json"
HISTORY = DATA / "watch_history.csv"


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


# =============================================================================
#  Depolama
# =============================================================================
def load() -> list[dict]:
    if not WATCHLIST.exists():
        return []
    try:
        data = json.loads(WATCHLIST.read_text(encoding="utf-8"))
        return data.get("positions", []) if isinstance(data, dict) else list(data)
    except Exception:
        return []


def save(positions: list[dict]) -> Path:
    DATA.mkdir(parents=True, exist_ok=True)
    payload = {"updated_at": datetime.now(timezone.utc).isoformat(),
               "positions": positions}
    WATCHLIST.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                         encoding="utf-8")
    return WATCHLIST


def add(ticker: str, entry_price: float | None = None, quantity: float | None = None,
        note: str = "", score_at_entry: float | None = None,
        added_date: str | None = None) -> tuple[list[dict], bool]:
    """Hisseyi listeye ekler. Zaten varsa alanlarini gunceller.

    Doner: (yeni liste, yeni_kayit_mi)
    """
    ticker = ticker.strip().upper()
    positions = load()

    for p in positions:
        if p["ticker"] == ticker:
            if entry_price is not None:
                p["entry_price"] = float(entry_price)
            if quantity is not None:
                p["quantity"] = float(quantity)
            if note:
                p["note"] = note
            if score_at_entry is not None:
                p["score_at_entry"] = float(score_at_entry)
            save(positions)
            return positions, False

    positions.append({
        "ticker": ticker,
        "added_date": added_date or _today(),
        "entry_price": float(entry_price) if entry_price is not None else None,
        "quantity": float(quantity) if quantity is not None else None,
        "score_at_entry": float(score_at_entry) if score_at_entry is not None else None,
        "note": note,
    })
    save(positions)
    return positions, True


def remove(ticker: str) -> bool:
    ticker = ticker.strip().upper()
    positions = load()
    kept = [p for p in positions if p["ticker"] != ticker]
    if len(kept) == len(positions):
        return False
    save(kept)
    return True


# =============================================================================
#  Gunluk gecmis (zaman serisi)
# =============================================================================
_HIST_COLS = ["date", "ticker", "price", "entry_price", "pnl_pct",
              "risk_level", "signal_count", "active_stop", "stop_pnl_pct",
              "short_target", "short_upside_pct", "long_target", "long_upside_pct",
              "total_score", "rsi14", "atr_pct"]


def append_history(rows: list[dict]) -> Path:
    """Gunluk anlik goruntuyu ekler. Ayni gun tekrar calisirsa gunu tazeler."""
    DATA.mkdir(parents=True, exist_ok=True)
    today = _today()

    existing: list[dict] = []
    if HISTORY.exists():
        try:
            with HISTORY.open(encoding="utf-8-sig", newline="") as fh:
                existing = [r for r in csv.DictReader(fh) if r.get("date") != today]
        except Exception:
            existing = []

    with HISTORY.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=_HIST_COLS, extrasaction="ignore")
        w.writeheader()
        for r in existing + rows:
            w.writerow(r)
    return HISTORY


def load_history() -> list[dict]:
    if not HISTORY.exists():
        return []
    try:
        with HISTORY.open(encoding="utf-8-sig", newline="") as fh:
            return list(csv.DictReader(fh))
    except Exception:
        return []


def history_for(ticker: str) -> list[dict]:
    return [r for r in load_history() if r.get("ticker") == ticker]


# =============================================================================
#  Gunluk guncelleme
# =============================================================================
def update(positions: list[dict] | None = None, use_cache: bool = False,
           period: str = "2y") -> list[dict[str, Any]]:
    """Listedeki her hisse icin taze veri ceker ve tam analiz uretir.

    use_cache varsayilan olarak KAPALIDIR: gunluk takipte bayat fiyat
    kabul edilemez.
    """
    from . import targets
    from .providers import yahoo

    positions = positions if positions is not None else load()
    out: list[dict[str, Any]] = []

    for pos in positions:
        tk = pos["ticker"]
        try:
            bundle = yahoo.fetch(tk, period=period, use_cache=use_cache)
        except Exception as exc:
            out.append({"ticker": tk, "ok": False,
                        "reason": f"{type(exc).__name__}: {exc}", "position": pos})
            continue

        df = bundle.get("history")
        if df is None or len(df) < 60:
            out.append({"ticker": tk, "ok": False,
                        "reason": "yetersiz fiyat gecmisi", "position": pos})
            continue

        info = bundle.get("info") or {}
        analysis = targets.analyze(
            df.dropna(subset=["Close"]), bundle,
            entry_price=pos.get("entry_price"),
            score_now=pos.get("score_now"),
            score_at_entry=pos.get("score_at_entry"),
        )
        if not analysis.get("available"):
            out.append({"ticker": tk, "ok": False,
                        "reason": analysis.get("reason", "analiz yapilamadi"),
                        "position": pos})
            continue

        out.append({
            "ticker": tk,
            "ok": True,
            "name": info.get("shortName") or info.get("longName") or tk,
            "sector": info.get("sector") or "Bilinmiyor",
            "currency": info.get("currency", "USD"),
            "market_cap": info.get("marketCap"),
            "position": pos,
            "analysis": analysis,
        })

    # En riskliler once
    out.sort(key=lambda r: -(r.get("analysis", {}).get("risk_index", -1)))
    return out


def to_history_rows(results: list[dict]) -> list[dict]:
    """Guncelleme sonucunu gecmis CSV satirlarina cevirir."""
    today = _today()
    rows = []
    for r in results:
        if not r.get("ok"):
            continue
        a = r["analysis"]
        st, lt = a.get("short_term", {}), a.get("long_term", {})
        rows.append({
            "date": today,
            "ticker": r["ticker"],
            "price": a.get("price"),
            "entry_price": a.get("entry_price"),
            "pnl_pct": a.get("pnl_pct"),
            "risk_level": a.get("risk_level"),
            "signal_count": a.get("signal_count"),
            "active_stop": (a.get("stops") or {}).get("active_stop"),
            "stop_pnl_pct": a.get("stop_pnl_pct"),
            "short_target": st.get("target") if st.get("available") else None,
            "short_upside_pct": st.get("upside_pct") if st.get("available") else None,
            "long_target": lt.get("target") if lt.get("available") else None,
            "long_upside_pct": lt.get("upside_pct") if lt.get("available") else None,
            "total_score": r["position"].get("score_now"),
            "rsi14": (a.get("technical") or {}).get("rsi14"),
            "atr_pct": (a.get("technical") or {}).get("atr_pct"),
        })
    return rows
