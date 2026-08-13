"""r/wallstreetbets anma (mention) ve duygu (sentiment) verisi.

Iki ucretsiz, anahtar gerektirmeyen kaynak birlestirilir:
  * ApeWisdom  -> anma sayisi, siralama, 24 saat oncesine gore DEGISIM
  * Tradestie  -> yorum sayisi ve duygu skoru

Onemli tasarim notu: "anma SAYISI" degil "anma IVMESI" ana sinyaldir.
GME'nin her gun 200 kez anilmasi bilgi tasimaz; 20'den 200'e cikmasi tasir.
"""
from __future__ import annotations

import math
from typing import Any

import requests

from .cache import get_or_fetch

APEWISDOM_URL = "https://apewisdom.io/api/v1.0/filter/{sub}/page/{page}"
TRADESTIE_URL = "https://tradestie.com/api/v1/apps/reddit"
_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; stock-screener/1.0)"}


def _fetch_apewisdom(sub: str, max_pages: int) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for page in range(1, max_pages + 1):
        try:
            r = requests.get(APEWISDOM_URL.format(sub=sub, page=page), headers=_HEADERS, timeout=25)
            r.raise_for_status()
            payload = r.json()
        except Exception:
            break

        results = payload.get("results") or []
        if not results:
            break

        for row in results:
            tk = str(row.get("ticker", "")).upper().strip()
            if not tk:
                continue
            mentions = float(row.get("mentions") or 0)
            prev = row.get("mentions_24h_ago")
            prev = float(prev) if prev not in (None, "") else None
            rank = row.get("rank")
            rank_prev = row.get("rank_24h_ago")

            out[tk] = {
                "mentions": mentions,
                "mentions_prev": prev,
                "upvotes": float(row.get("upvotes") or 0),
                "rank": int(rank) if rank else None,
                "rank_prev": int(rank_prev) if rank_prev else None,
                "name": row.get("name"),
            }

        if page >= int(payload.get("pages", 1)):
            break
    return out


def _fetch_tradestie() -> dict[str, dict]:
    try:
        r = requests.get(TRADESTIE_URL, headers=_HEADERS, timeout=25)
        r.raise_for_status()
        rows = r.json()
    except Exception:
        return {}

    out = {}
    for row in rows or []:
        tk = str(row.get("ticker", "")).upper().strip()
        if tk:
            out[tk] = {
                "sentiment": row.get("sentiment"),
                "sentiment_score": float(row.get("sentiment_score") or 0.0),
                "comments": float(row.get("no_of_comments") or 0),
            }
    return out


def load(subreddit: str = "wallstreetbets", max_pages: int = 4,
         use_cache: bool = True) -> dict[str, dict]:
    """Ticker -> WSB metrikleri sozlugu."""
    ape = get_or_fetch("wsb_ape", f"{subreddit}:{max_pages}",
                       lambda: _fetch_apewisdom(subreddit, max_pages),
                       ttl_seconds=3 * 3600, enabled=use_cache)
    tra = get_or_fetch("wsb_tradestie", "daily", _fetch_tradestie,
                       ttl_seconds=3 * 3600, enabled=use_cache)

    merged: dict[str, dict] = {}
    for tk in set(ape) | set(tra):
        rec = dict(ape.get(tk, {}))
        rec.update(tra.get(tk, {}))
        merged[tk] = rec
    return merged


def score_ticker(rec: dict[str, Any] | None, universe_size: int = 300) -> dict[str, Any]:
    """WSB kaydini 0-100 skora cevirir.

    Bilesenler:
      ivme (%50) : anmalarin 24s degisimi — asil bilgi tasiyan kisim
      seviye(%30) : logaritmik anma sayisi (doygunlasan fayda)
      duygu (%20) : yorum duygu skoru
    Faktor mevcut degilse None doner -> agirlik yeniden dagitilir.
    """
    if not rec or not rec.get("mentions"):
        return {"available": False, "score": None}

    mentions = float(rec.get("mentions") or 0)
    prev = rec.get("mentions_prev")

    # --- seviye: log olcek, ~200 anmada doygunluk
    level = 100.0 * min(1.0, math.log1p(mentions) / math.log1p(200.0))

    # --- ivme: 24 saatlik degisim orani
    if prev is not None and prev > 0:
        growth = (mentions - prev) / prev
        # -%50 -> 0 puan, 0% -> 50 puan, +%200 -> 100 puan
        accel = 50.0 + 50.0 * max(-1.0, min(1.0, growth / 2.0))
    elif prev == 0 and mentions > 0:
        accel = 90.0  # sifirdan gorunurluge cikis
    else:
        accel = 50.0

    # --- duygu
    ss = rec.get("sentiment_score")
    sentiment = 50.0 + 50.0 * max(-1.0, min(1.0, float(ss) / 0.3)) if ss is not None else 50.0

    score = 0.50 * accel + 0.30 * level + 0.20 * sentiment

    rank = rec.get("rank")
    rank_prev = rec.get("rank_prev")
    return {
        "available": True,
        "score": float(max(0.0, min(100.0, score))),
        "mentions": mentions,
        "mentions_prev": prev,
        "rank": rank,
        "rank_change": (rank_prev - rank) if (rank and rank_prev) else None,
        "sentiment_score": ss,
        "components": {"accel": round(accel, 1), "level": round(level, 1),
                       "sentiment": round(sentiment, 1)},
        # hype ceza kurali icin
        "is_top10": bool(rank and rank <= 10),
    }
