"""Tarama gecmisi — hangi sembol ne zaman basariyla tarandi.

Iki denetim bulgusunu birden cozer:

K3 (evrenin yarisi hic gorulmuyor): Yahoo hiz siniri yuzunden tek seferde
    2800 hisse cekilemiyor. Evren HER GUN AYNI SIRADA taranirsa, listenin
    sonundakiler sistematik olarak hic gorulmez. Bu kayit sayesinde evren
    "en uzun suredir taranmamis once" sirasina gore dizilir; boylece birkac
    gunde tum evren dolasilir.

Y3 (hayatta kalan yanliligi): Her gunun sembol listesi saklanir. Bir sembol
    listeden kalici olarak kaybolduysa muhtemelen kote disi kalmistir; ileride
    yapilacak egitimde bu bilgi olmadan sonuclar yukari yanli olur.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"
LOG = DATA / "scan_log.json"
UNIVERSE_LOG = DATA / "universe_history.json"


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


# =============================================================================
#  Sembol bazli son basarili tarama tarihi
# =============================================================================
def load() -> dict[str, str]:
    if not LOG.exists():
        return {}
    try:
        return json.loads(LOG.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def record(tickers: list[str] | set[str]) -> None:
    """Basariyla taranan sembolleri bugunun tarihiyle isaretler."""
    log = load()
    today = _today()
    for t in tickers:
        log[str(t).upper()] = today
    DATA.mkdir(parents=True, exist_ok=True)
    LOG.write_text(json.dumps(log, ensure_ascii=False, indent=0, sort_keys=True),
                   encoding="utf-8")


def order_by_staleness(tickers: list[str], pinned: set[str] | None = None) -> list[str]:
    """Evreni "en uzun suredir taranmamis once" sirasina dizer.

    Sirasiyla:
      1. Izleme listesi (her zaman, kosulsuz)
      2. Hic taranmamis semboller
      3. En eski tarihten en yeniye
    """
    pinned = pinned or set()
    log = load()

    def key(t: str) -> tuple:
        if t in pinned:
            return (0, "")
        seen = log.get(t)
        return (1, "") if seen is None else (2, seen)

    return sorted(tickers, key=key)


def coverage_stats(tickers: list[str]) -> dict:
    """Evrenin ne kadarinin ne zamandir taranmadigini ozetler."""
    log = load()
    never = [t for t in tickers if t not in log]
    dates = [log[t] for t in tickers if t in log]
    by_date: dict[str, int] = {}
    for d in dates:
        by_date[d] = by_date.get(d, 0) + 1
    return {
        "universe": len(tickers),
        "never_scanned": len(never),
        "ever_scanned": len(dates),
        "coverage_pct": round(100 * len(dates) / max(1, len(tickers)), 1),
        "by_date": dict(sorted(by_date.items(), reverse=True)[:7]),
    }


# =============================================================================
#  Gunluk evren anlik goruntusu (kote disi tespiti icin)
# =============================================================================
def record_universe(tickers: list[str]) -> dict:
    """Bugunun evren listesini saklar ve kaybolan sembolleri raporlar."""
    hist: dict[str, list[str]] = {}
    if UNIVERSE_LOG.exists():
        try:
            hist = json.loads(UNIVERSE_LOG.read_text(encoding="utf-8")) or {}
        except Exception:
            hist = {}

    today = _today()
    prev_days = sorted([d for d in hist if d < today], reverse=True)
    disappeared: list[str] = []
    if prev_days:
        prev = set(hist[prev_days[0]])
        disappeared = sorted(prev - set(tickers))

    hist[today] = sorted(tickers)
    # Son 60 gun yeter; dosya sonsuz buyumesin
    for d in sorted(hist)[:-60]:
        hist.pop(d, None)

    DATA.mkdir(parents=True, exist_ok=True)
    UNIVERSE_LOG.write_text(json.dumps(hist, ensure_ascii=False), encoding="utf-8")

    return {
        "recorded_days": len(hist),
        "disappeared_since_last": disappeared[:50],
        "disappeared_count": len(disappeared),
        "compared_to": prev_days[0] if prev_days else None,
    }
