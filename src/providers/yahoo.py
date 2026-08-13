"""Yahoo Finance veri saglayicisi (yfinance).

Cektikleri:
  * gunluk OHLCV gecmisi (teknik gostergeler icin)
  * temel veriler (info): degerleme, karlilik, borc, sahiplik
  * analist tavsiye dagilimi ve ZAMAN SERISI (revizyon momentumu icin kritik)
  * EPS tahmin trendi / revizyon sayilari  (Zacks tarzi sinyal)
  * bilanco surpriz gecmisi (PEAD icin)
"""
from __future__ import annotations

import os
import random
import time
import warnings
from pathlib import Path
from typing import Any

import pandas as pd

from .cache import get_or_fetch

warnings.filterwarnings("ignore")


def ensure_ssl_env() -> None:
    """Windows'ta ASCII olmayan kullanici adi (orn. 'MSI' icindeki I) curl'un
    CA sertifika dosyasini acamamasina yol acar. Sertifikayi ASCII bir yola
    kopyalayip ilgili ortam degiskenlerini isaret ediyoruz."""
    if os.environ.get("CURL_CA_BUNDLE") and os.environ.get("SSL_CERT_FILE"):
        return
    try:
        import certifi
        src = Path(certifi.where())
        if src.exists() and src.as_posix().isascii():
            os.environ.setdefault("CURL_CA_BUNDLE", str(src))
            os.environ.setdefault("SSL_CERT_FILE", str(src))
            return
        dst = Path(os.environ.get("SystemDrive", "C:") + "/claude-certs/cacert.pem")
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists() or dst.stat().st_size != src.stat().st_size:
            dst.write_bytes(src.read_bytes())
        os.environ["CURL_CA_BUNDLE"] = str(dst)
        os.environ["SSL_CERT_FILE"] = str(dst)
    except Exception:
        pass


ensure_ssl_env()
import yfinance as yf  # noqa: E402  (ensure_ssl_env oncesinde import edilmemeli)


def _df_or_none(obj) -> pd.DataFrame | None:
    return obj if isinstance(obj, pd.DataFrame) and not obj.empty else None


class RateLimited(RuntimeError):
    """Yahoo hiz siniri uyguluyor — kisa vadede tekrar denemek anlamsiz."""


def _is_rate_limit(exc: Exception) -> bool:
    name = type(exc).__name__.lower()
    text = str(exc).lower()
    return ("ratelimit" in name or "too many requests" in text
            or "429" in text or "rate limited" in text)


def _fetch_history(t, period: str, attempts: int = 3) -> pd.DataFrame | None:
    """Fiyat gecmisi — gecici hatalara karsi geri cekilmeli tekrar deneme.

    Yahoo yogun paralel istekte 401 'Invalid Crumb' donebiliyor; bu GECICIDIR
    ve tekrar denemeden vazgecersek hisse haksiz yere elenir.

    Ama 429 / YFRateLimitError farklidir: sunucu bizi kasten yavaslatiyordur.
    O durumda tekrar denemek sadece yasagi uzatir, bu yuzden RateLimited
    firlatilir ve cagiran taraf taramayi durdurabilir.
    """
    for i in range(attempts):
        try:
            h = t.history(period=period, auto_adjust=True)
            if isinstance(h, pd.DataFrame) and len(h):
                return h
        except Exception as exc:
            if _is_rate_limit(exc):
                raise RateLimited(str(exc)[:120]) from exc
        if i < attempts - 1:
            time.sleep(1.5 * (2 ** i) + random.random())
    return None


def _fetch_bundle(ticker: str, period: str) -> dict[str, Any]:
    t = yf.Ticker(ticker)
    out: dict[str, Any] = {"ticker": ticker}
    out["history"] = _fetch_history(t, period)

    try:
        out["info"] = dict(t.info or {})
    except Exception:
        out["info"] = {}

    for attr in ("recommendations", "eps_trend", "eps_revisions",
                 "earnings_history", "growth_estimates"):
        try:
            out[attr] = _df_or_none(getattr(t, attr))
        except Exception:
            out[attr] = None

    try:
        out["price_targets"] = dict(t.analyst_price_targets or {})
    except Exception:
        out["price_targets"] = {}

    try:
        cal = t.calendar
        out["calendar"] = dict(cal) if isinstance(cal, dict) else None
    except Exception:
        out["calendar"] = None

    try:
        out["cashflow"] = _df_or_none(t.quarterly_cashflow)
    except Exception:
        out["cashflow"] = None

    try:
        out["balance_sheet"] = _df_or_none(t.quarterly_balance_sheet)
    except Exception:
        out["balance_sheet"] = None

    try:
        out["income"] = _df_or_none(t.quarterly_income_stmt)
    except Exception:
        out["income"] = None

    return out


def _worth_caching(bundle: dict[str, Any]) -> bool:
    """Fiyat gecmisi yoksa cekim basarisiz sayilir -> onbellege YAZILMAZ."""
    h = bundle.get("history")
    return isinstance(h, pd.DataFrame) and len(h) >= 30


def fetch(ticker: str, period: str = "2y", use_cache: bool = True,
          ttl_seconds: int = 6 * 3600) -> dict[str, Any]:
    """Tek hisse icin tum veri paketini getirir (onbellekli)."""
    return get_or_fetch("yahoo", f"{ticker}:{period}",
                        lambda: _fetch_bundle(ticker, period),
                        ttl_seconds=ttl_seconds, enabled=use_cache,
                        should_cache=_worth_caching)


def fetch_cached(ticker: str, period: str = "2y",
                 max_age_seconds: int = 5 * 24 * 3600) -> dict[str, Any] | None:
    """SADECE onbellekten okur — ag istegi yapmaz.

    Donusumlu tarama her turda evrenin bir dilimini ceker. Yalnizca o dilim
    skorlansaydi siralama evrenin kucuk bir parcasini kapsardi. Bu fonksiyon,
    daha once cekilmis hisseleri bedavaya (ag istegi olmadan) skorlamaya dahil
    eder; boylece siralama her turda genisler.

    TTL, canli taramadan uzun tutulur: bir kac gunluk temel veri, hisseyi
    siralamadan tamamen cikarmaktan iyidir. Fiyat verisi eskidiginde satir
    'bayat' olarak isaretlenir.
    """
    def _none():
        return None

    return get_or_fetch("yahoo", f"{ticker}:{period}", _none,
                        ttl_seconds=max_age_seconds, enabled=True)


def fetch_benchmark(symbol: str = "SPY", period: str = "2y",
                    use_cache: bool = True) -> pd.DataFrame | None:
    """Goreli guc hesabi icin endeks getirisi."""
    # Tekrar denemeli; basarisiz cekim ONBELLEGE YAZILMAZ. Aksi halde tek bir
    # gecici hata 'goreli guc' faktorunu (agirlik 10) saatlerce yok ederdi.
    return get_or_fetch("yahoo_bench", f"{symbol}:{period}",
                        lambda: _fetch_history(yf.Ticker(symbol), period, attempts=4),
                        ttl_seconds=6 * 3600, enabled=use_cache,
                        should_cache=lambda h: h is not None and len(h) >= 30)
