"""Turns the bars before a signal into a fixed-size window the model can read.

The scalar features we had (ATR%, RSI, volume ratio, distance from the moving
averages) describe the moment a setup fires. They say nothing about how price
got there — whether it drifted up quietly or spiked and stalled, whether volume
was building or drying out. That path is most of what a chart reader claims to
be looking at, and none of it was reaching the model.

So each signal now also gets the last N bars leading up to it, encoded as
channels that mean the same thing for a $12 stock and a $300 one.

Everything here looks backwards only. The window ends on the signal bar; the
label starts after it.

News is not modelled. An earnings surprise or a downgrade will move price in
ways no amount of chart history explains, and pulling that in reliably is a
much bigger job. That caps how far this can go, and it's a deliberate choice
rather than an oversight.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# How many bars of history each signal carries. On hourly bars 24 is about
# three and a half trading days — long enough to hold a pullback and the move
# it pulled back from, short enough that most signals actually have it.
PENCERE = 24

KANALLAR = ("getiri", "govde", "ust_fitil", "alt_fitil", "menzil", "hacim",
            "konum")


def _kanallar(df: pd.DataFrame) -> np.ndarray:
    """Bars in, scale-free channels out, one row per bar.

    Prices themselves are useless across stocks, so nothing here is a price.
    Each channel is either a return, a fraction of the bar's own range, or a
    ratio to a rolling median.
    """
    o = df["Open"].to_numpy(float)
    h = df["High"].to_numpy(float)
    l = df["Low"].to_numpy(float)
    c = df["Close"].to_numpy(float)
    v = df["Volume"].to_numpy(float)

    menzil = np.maximum(h - l, 1e-12)
    onceki = np.concatenate([[c[0]], c[:-1]])

    getiri = np.log(np.maximum(c, 1e-12) / np.maximum(onceki, 1e-12))
    govde = (c - o) / menzil
    ust = (h - np.maximum(o, c)) / menzil
    alt = (np.minimum(o, c) - l) / menzil
    # Range relative to the stock's own recent range, so a quiet name and a
    # volatile one look the same when they're both having a big bar.
    menzil_med = pd.Series(menzil / np.maximum(c, 1e-12)).rolling(
        50, min_periods=10).median().to_numpy()
    menzil_orani = np.log((menzil / np.maximum(c, 1e-12))
                          / np.maximum(menzil_med, 1e-9))
    hacim_med = pd.Series(v).rolling(50, min_periods=10).median().to_numpy()
    hacim = np.log(np.maximum(v, 1.0) / np.maximum(hacim_med, 1.0))
    # Where the close sits inside the last 20 bars.
    en_dusuk = pd.Series(l).rolling(20, min_periods=5).min().to_numpy()
    en_yuksek = pd.Series(h).rolling(20, min_periods=5).max().to_numpy()
    konum = (c - en_dusuk) / np.maximum(en_yuksek - en_dusuk, 1e-12)

    X = np.column_stack([getiri, govde, ust, alt, menzil_orani, hacim, konum])
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def pencereler(bars: dict, anahtarlar: "list[tuple[str, pd.Timestamp]]",
               pencere: int = PENCERE) -> tuple[np.ndarray, np.ndarray]:
    """Build one window per (ticker, timestamp).

    Returns (X, bulundu). Rows we couldn't build — symbol missing, or the
    signal sits too close to the start of the series — come back as zeros and
    are flagged False so the caller can drop them. Silently returning zeros
    would train the model on blank history.
    """
    n = len(anahtarlar)
    X = np.zeros((n, pencere, len(KANALLAR)), dtype=np.float32)
    bulundu = np.zeros(n, dtype=bool)

    # Group by symbol so each series is converted to channels once.
    sirali: dict[str, list[int]] = {}
    for i, (tk, _) in enumerate(anahtarlar):
        sirali.setdefault(tk, []).append(i)

    for tk, idxs in sirali.items():
        df = bars.get(tk)
        if df is None or len(df) < pencere + 5:
            continue
        try:
            kanal = _kanallar(df)
        except Exception:
            continue
        zaman = pd.DatetimeIndex(df.index)
        try:
            zaman = (zaman.tz_localize(None) if zaman.tz is None
                     else zaman.tz_convert(None))
        except (TypeError, AttributeError):
            pass
        yer = pd.Series(np.arange(len(zaman)), index=zaman)
        yer = yer[~yer.index.duplicated(keep="last")]

        for i in idxs:
            t = anahtarlar[i][1]
            p = yer.get(t)
            if p is None or p < pencere - 1:
                continue
            X[i] = kanal[p - pencere + 1:p + 1]
            bulundu[i] = True

    return X, bulundu


def ozet(X: np.ndarray, bulundu: np.ndarray) -> dict:
    return {
        "satir": int(len(bulundu)),
        "pencereli": int(bulundu.sum()),
        "oran": round(float(bulundu.mean()), 4) if len(bulundu) else 0.0,
        "pencere": int(X.shape[1]) if X.ndim == 3 else 0,
        "kanal": list(KANALLAR),
    }
