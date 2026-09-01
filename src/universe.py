"""Taranacak hisse evreninin olusturulmasi.

Kaynaklar birlestirilebilir:
  sp500   -> Wikipedia'dan S&P 500 bilesenleri
  nasdaq100 -> Wikipedia'dan Nasdaq-100
  wsb     -> Reddit r/wallstreetbets'te en cok anilanlar
  file    -> kendi listen (her satirda bir sembol)
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from .providers import cache, reddit_wsb
from .providers.cache import get_or_fetch

# ETF / endeks / kripto gibi hisse olmayan semboller elenir
_NON_STOCK = {
    "SPY", "QQQ", "IWM", "DIA", "VOO", "VTI", "VXX", "UVXY", "SQQQ", "TQQQ",
    "SOXL", "SOXS", "SPXU", "SPXS", "TNA", "TZA", "GLD", "SLV", "USO", "TLT",
    "HYG", "ARKK", "XLF", "XLE", "XLK", "XLV", "XLU", "XLI", "XLP", "XLY",
    "BTC", "ETH", "DOGE", "BTCUSD", "ES", "NQ", "VIX", "SPX", "NDX", "RUT",
    "IBIT", "GBTC", "MSTU", "MSTZ", "YOLO", "WSB", "DD", "CEO", "USA", "IT",
    "EPS", "IRA", "ATH", "FOMO", "YOLO", "IMO", "PSA", "GDP", "CPI", "FED",
}

_TICKER_RE = re.compile(r"^[A-Z][A-Z.\-]{0,6}$")

# Wikipedia sayfa duzenleri zaman zaman degisir; birden fazla aday URL denenir.
_WIKI = {
    "sp500": ["https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"],
    "nasdaq100": ["https://en.wikipedia.org/wiki/Nasdaq-100",
                  "https://en.wikipedia.org/wiki/NASDAQ-100"],
}


def _tickers_from_tables(tables: list[pd.DataFrame], min_count: int = 20) -> list[str]:
    """Sutun ADINA guvenmeden, sembol GORUNUMLU sutunu bulur.

    Wikipedia tablo basliklarini degistirdiginde ('Symbol' -> 'Ticker' -> yok)
    kirilmamasi icin: her sutunu tarayip cogunlugu ticker desenine uyan
    sutunu secer.
    """
    best: list[str] = []
    for t in tables:
        for c in t.columns:
            try:
                vals = t[c].astype(str).str.strip().str.upper()
            except Exception:
                continue
            syms = [v.replace(".", "-") for v in vals if _TICKER_RE.match(v)]
            # sutunun cogunlugu ticker olmali (yil/sayi sutunlarini eleme)
            if len(syms) >= min_count and len(syms) >= 0.8 * len(vals) and len(syms) > len(best):
                best = syms
    return best


def _from_wikipedia(name: str) -> list[str]:
    def _fetch() -> list[str]:
        import io

        import requests

        for url in _WIKI[name]:
            try:
                # requests ile cekip StringIO'ya vermek pandas'in kendi
                # indiricisinden daha guvenilir (User-Agent gonderebiliyoruz).
                html = requests.get(url, headers={"User-Agent": "Mozilla/5.0"},
                                    timeout=30).text
                syms = _tickers_from_tables(pd.read_html(io.StringIO(html)))
                if syms:
                    return syms
            except Exception:
                continue
        return []

    try:
        # Bos sonucu onbellege yazma — sayfa gecici olarak erisilemez olabilir.
        return get_or_fetch("universe", name, _fetch, ttl_seconds=7 * 24 * 3600,
                            should_cache=lambda v: bool(v))
    except Exception:
        return []


_NASDAQ_SCREENER = ("https://api.nasdaq.com/api/screener/stocks"
                    "?tableonly=true&limit=10000&offset=0&exchange={ex}")


def _mcap(row: dict) -> float:
    try:
        return float(str(row.get("marketCap", "")).replace(",", "").strip() or 0)
    except (TypeError, ValueError):
        return 0.0


# Kotasyon listesi gunde bir kez cekilir; gun icinde birkac sembol degismez.
_LISTINGS_TTL = 24 * 3600
# Bu sayinin altinda donen bir cekim BASARISIZ sayilir. ABD borsalarinda
# piyasa degeri bilinen binlerce hisse var; 500'un altina dusmesinin tek
# makul aciklamasi ucun erisilemez olmasidir.
_LISTINGS_MIN = 500
_LISTINGS_KEY = "us_listings_v2"

# Son cagriya ait durum — build() raporlayabilsin diye.
LAST_LISTINGS_STATE: dict = {}


def _fetch_us_listings(exchanges: tuple[str, ...]) -> list[tuple[str, float]]:
    """api.nasdaq.com'dan (sembol, piyasa degeri) ciftleri. Suzme yok."""
    import requests

    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
               "Accept": "application/json"}
    rows: list[dict] = []
    for ex in exchanges:
        try:
            r = requests.get(_NASDAQ_SCREENER.format(ex=ex), headers=headers, timeout=60)
            r.raise_for_status()
            rows += r.json()["data"]["table"]["rows"] or []
        except Exception:
            continue

    out: list[tuple[str, float]] = []
    seen: set[str] = set()
    for row in rows:
        sym = str(row.get("symbol", "")).strip().upper().replace(".", "-")
        if not sym or sym in seen or not _TICKER_RE.match(sym):
            continue
        m = _mcap(row)
        if m <= 0:                      # piyasa degeri bilinmiyor -> atla
            continue
        seen.add(sym)
        out.append((sym, m))
    return out


def us_listings(exchanges: tuple[str, ...] = ("NASDAQ", "NYSE", "AMEX")
                ) -> tuple[list[tuple[str, float]], dict]:
    """Kotasyon listesi + nereden geldigi bilgisi.

    Onbellek katmani burada KRITIKTIR. Eskiden her cagri dogrudan aga cikiyordu
    ve ag hatasi sessizce bos liste donduruyordu; evren cokuyor, tarama sadece
    izleme listesindeki birkac hisseyi skorluyor ve pano O HALIYLE yeniden
    yaziliyordu. Kullanicinin sitesinde "sadece kendi ekledigim hisseler
    gorunuyor" sikayetinin sebebi buydu.

    Sira: taze onbellek -> ag -> BAYAT onbellek -> bos.
    """
    global LAST_LISTINGS_STATE
    ident = f"{_LISTINGS_KEY}:{'+'.join(exchanges)}"
    cached = cache.peek("universe", ident)

    if cached and cached[1] < _LISTINGS_TTL and len(cached[0]) >= _LISTINGS_MIN:
        LAST_LISTINGS_STATE = {"source": "onbellek", "count": len(cached[0]),
                               "age_hours": round(cached[1] / 3600, 1), "ok": True}
        return cached[0], LAST_LISTINGS_STATE

    fresh = _fetch_us_listings(exchanges)
    if len(fresh) >= _LISTINGS_MIN:
        cache.put("universe", ident, fresh)
        LAST_LISTINGS_STATE = {"source": "ag", "count": len(fresh),
                               "age_hours": 0.0, "ok": True}
        return fresh, LAST_LISTINGS_STATE

    if cached and len(cached[0]) >= _LISTINGS_MIN:
        LAST_LISTINGS_STATE = {"source": "bayat_onbellek", "count": len(cached[0]),
                               "age_hours": round(cached[1] / 3600, 1), "ok": True,
                               "fetched_now": len(fresh)}
        return cached[0], LAST_LISTINGS_STATE

    LAST_LISTINGS_STATE = {"source": "basarisiz", "count": len(fresh),
                           "age_hours": None, "ok": False}
    return fresh, LAST_LISTINGS_STATE


def _from_us_listings(min_mcap: float, max_mcap: float,
                      exchanges: tuple[str, ...] = ("NASDAQ", "NYSE", "AMEX")) -> list[str]:
    """Tum ABD borsalarina kote hisseler, PIYASA DEGERI BANDINA gore suzulur.

    Bu kaynak sistemin "gelecek vadeden hisse" arayisinin temelidir: S&P 500
    tanimi geregi olgunlasmis devleri icerir; yukselen sirketler kucuk ve orta
    olcekte yasar.
    """
    rows, _ = us_listings(exchanges)
    picked = [(s, m) for s, m in rows if min_mcap <= m <= max_mcap]
    # Buyukten kucuge: likidite ve veri kalitesi ust bantta daha iyi
    picked.sort(key=lambda x: -x[1])
    return [s for s, _ in picked]


def _from_wsb(top_n: int, use_cache: bool = True) -> list[str]:
    data = reddit_wsb.load(use_cache=use_cache)
    ranked = sorted(
        ((tk, rec) for tk, rec in data.items() if rec.get("mentions")),
        key=lambda kv: -float(kv[1].get("mentions") or 0),
    )
    return [tk for tk, _ in ranked[:top_n]]


def _from_file(path: str) -> list[str]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Sembol dosyasi bulunamadi: {path}")
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        s = line.split("#")[0].split(",")[0].strip().upper()
        if s and _TICKER_RE.match(s):
            out.append(s)
    return out


# Piyasa degeri bandi on ayarlari (USD)
PRESETS: dict[str, tuple[float, float]] = {
    "micro":    (5e7,  3e8),     # 50M  - 300M   cok riskli, cok yuksek potansiyel
    "smallcap": (3e8,  2e9),     # 300M - 2Mr    klasik "yukselen sirket" bandi
    "midcap":   (2e9,  1e10),    # 2Mr  - 10Mr   kanitlanmis ama hala buyuyebilir
    "emerging": (2e8,  1e10),    # 200M - 10Mr   micro+small+mid: ana av sahasi
    "largecap": (1e10, 2e11),
    "us":       (5e7,  1e15),    # tumu
}


def build(sources: list[str], wsb_top: int = 60, symbols_file: str | None = None,
          limit: int | None = None, keep_etfs: bool = False,
          min_mcap: float | None = None,
          max_mcap: float | None = None) -> tuple[list[str], dict]:
    """Evreni olusturur. Doner: (semboller, kaynak dagilimi).

    min_mcap/max_mcap verilirse on ayar bandini ezer.
    """
    collected: dict[str, list[str]] = {}

    for src in sources:
        src = src.strip().lower()
        if src in _WIKI:
            collected[src] = _from_wikipedia(src)
        elif src in PRESETS:
            lo, hi = PRESETS[src]
            lo = min_mcap if min_mcap is not None else lo
            hi = max_mcap if max_mcap is not None else hi
            collected[src] = _from_us_listings(lo, hi)
        elif src == "wsb":
            collected["wsb"] = _from_wsb(wsb_top)
        elif src == "file":
            collected["file"] = _from_file(symbols_file) if symbols_file else []

    seen, ordered = set(), []
    for src, syms in collected.items():
        for s in syms:
            if s in seen:
                continue
            if not keep_etfs and s in _NON_STOCK:
                continue
            seen.add(s)
            ordered.append(s)

    if limit and limit < len(ordered):
        # Bastan kesmek yerine esit araliklarla ornekle: liste piyasa degerine
        # gore sirali oldugu icin bastan kesmek evreni sadece en buyuklere
        # daraltir ve "yukselen sirket" arayisini bosa cikarirdi.
        step = len(ordered) / limit
        ordered = [ordered[int(i * step)] for i in range(limit)]

    breakdown = {src: len(syms) for src, syms in collected.items()}
    breakdown["final_unique"] = len(ordered)

    # Kotasyon listesi kullanildiysa nereden geldigini de bildir: cagiran taraf
    # (cmd_scan) evrenin cokup cokmedigine buna bakarak karar veriyor.
    if any(src.strip().lower() in PRESETS for src in sources):
        breakdown["_listings"] = dict(LAST_LISTINGS_STATE)
    return ordered, breakdown
