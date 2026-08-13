"""Investing.com "Teknik Ozet" tablosunun yerel yeniden uretimi.

Investing.com'un ozeti deterministik bir OY SAYIMIDIR:
  * 12 hareketli ortalama : SMA ve EMA x (5, 10, 20, 50, 100, 200)
        fiyat > MA  -> Al,  fiyat < MA -> Sat
  * 9 osilator : RSI(14), STOCH(9,6), STOCHRSI(14), MACD(12,26), ADX(14),
        Williams %R, CCI(14), ROC, Ultimate(7,14,28)
        her biri kendi esigine gore Al / Sat / Notr

Sonuc: (Al - Sat) / toplam  ->  Guclu Al ... Guclu Sat

Neden scraping degil de yeniden uretim?
  - Investing.com'un HTML'i sik degisir ve bot korumasi vardir; scraper kirilir.
  - Ayni girdilerle ayni cikti uretilir, ek olarak ARA DEGERLER de elimizde
    kalir (ML modeline aktarilabilir ozellikler).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from . import indicators as ta

MA_PERIODS = (5, 10, 20, 50, 100, 200)

# (Al - Sat) / toplam oy oranindan etiket
_LABEL_BANDS = [
    (0.50, "Guclu Al", "STRONG_BUY", 100.0),
    (0.15, "Al", "BUY", 78.0),
    (-0.15, "Notr", "NEUTRAL", 50.0),
    (-0.50, "Sat", "SELL", 22.0),
    (-1.01, "Guclu Sat", "STRONG_SELL", 0.0),
]


def _label(ratio: float) -> tuple[str, str, float]:
    for thr, tr, en, base in _LABEL_BANDS:
        if ratio >= thr:
            return tr, en, base
    return "Guclu Sat", "STRONG_SELL", 0.0


def _vote_ma(price: float, ma_val: float) -> int:
    if not np.isfinite(ma_val) or not np.isfinite(price):
        return 0
    return 1 if price > ma_val else -1


def _last(s: pd.Series) -> float:
    s = s.dropna()
    return float(s.iloc[-1]) if len(s) else float("nan")


def compute(df: pd.DataFrame) -> dict[str, Any]:
    """df: Open/High/Low/Close/Volume sutunlu gunluk OHLCV.

    Doner: oy detaylari + 0-100 arasi 'score' + Investing tarzi etiket.
    """
    if df is None or len(df) < 30:
        return {"available": False}

    close, high, low = df["Close"], df["High"], df["Low"]
    price = _last(close)
    if not np.isfinite(price):
        return {"available": False}

    ma_votes: dict[str, int] = {}
    for n in MA_PERIODS:
        if len(close.dropna()) >= n // 2:
            ma_votes[f"SMA{n}"] = _vote_ma(price, _last(ta.sma(close, n)))
            ma_votes[f"EMA{n}"] = _vote_ma(price, _last(ta.ema(close, n)))

    osc: dict[str, int] = {}
    osc_values: dict[str, float] = {}

    # RSI(14): >70 asiri alim -> Sat, <30 asiri satim -> Al
    v = _last(ta.rsi(close, 14))
    osc_values["RSI14"] = v
    osc["RSI14"] = 0 if not np.isfinite(v) else (-1 if v > 70 else (1 if v < 30 else 0))

    # STOCH %K(9,6): >80 -> Sat, <20 -> Al
    k, d = ta.stoch(high, low, close, k=9, d=6, smooth=1)
    v = _last(k)
    osc_values["STOCH9"] = v
    osc["STOCH9"] = 0 if not np.isfinite(v) else (-1 if v > 80 else (1 if v < 20 else 0))

    # STOCHRSI(14)
    v = _last(ta.stoch_rsi(close, 14))
    osc_values["STOCHRSI14"] = v
    osc["STOCHRSI14"] = 0 if not np.isfinite(v) else (-1 if v > 80 else (1 if v < 20 else 0))

    # MACD(12,26): line > signal -> Al
    line, sig, _ = ta.macd(close)
    lv, sv = _last(line), _last(sig)
    osc_values["MACD"] = lv - sv if np.isfinite(lv) and np.isfinite(sv) else float("nan")
    osc["MACD"] = 0 if not (np.isfinite(lv) and np.isfinite(sv)) else (1 if lv > sv else -1)

    # ADX(14): trend gucu > 20 ise yonu +DI/-DI belirler
    adx_v, pdi, mdi = ta.adx(high, low, close, 14)
    a, p, m = _last(adx_v), _last(pdi), _last(mdi)
    osc_values["ADX14"] = a
    osc_values["PLUS_DI"] = p
    osc_values["MINUS_DI"] = m
    if np.isfinite(a) and a > 20 and np.isfinite(p) and np.isfinite(m):
        osc["ADX14"] = 1 if p > m else -1
    else:
        osc["ADX14"] = 0

    # Williams %R: < -80 -> Al (asiri satim), > -20 -> Sat
    v = _last(ta.williams_r(high, low, close, 14))
    osc_values["WILLR14"] = v
    osc["WILLR14"] = 0 if not np.isfinite(v) else (1 if v < -80 else (-1 if v > -20 else 0))

    # CCI(14): > +100 -> Al (momentum), < -100 -> Sat
    v = _last(ta.cci(high, low, close, 14))
    osc_values["CCI14"] = v
    osc["CCI14"] = 0 if not np.isfinite(v) else (1 if v > 100 else (-1 if v < -100 else 0))

    # ROC: > 0 -> Al
    v = _last(ta.roc(close, 12))
    osc_values["ROC12"] = v
    osc["ROC12"] = 0 if not np.isfinite(v) else (1 if v > 0 else -1)

    # Ultimate Oscillator: > 70 -> Sat, < 30 -> Al
    v = _last(ta.ultimate_oscillator(high, low, close))
    osc_values["ULTOSC"] = v
    osc["ULTOSC"] = 0 if not np.isfinite(v) else (-1 if v > 70 else (1 if v < 30 else 0))

    # --------------------------------------------------------------------
    # GENISLETILMIS BLOK — Investing.com'un standart tablosunun otesinde.
    # Trend takibi ve kirilim kurulumu icin 9 ek sinyal.
    # --------------------------------------------------------------------
    ext: dict[str, int] = {}
    volume = df["Volume"] if "Volume" in df else None

    # Bollinger %B: bandin ust yarisinda olmak trend gucudur, ama >1 asiri uzama
    _, _, _, pct_b, bb_width = ta.bollinger(close, 20, 2.0)
    v = _last(pct_b)
    osc_values["BB_PCT_B"] = v
    osc_values["BB_WIDTH"] = _last(bb_width)
    ext["BOLLINGER"] = 0 if not np.isfinite(v) else (-1 if v > 1.05 else (1 if 0.5 < v <= 1.05 else (-1 if v < 0 else 0)))

    # Aroon: trendin tazeligi
    a_up, a_dn = ta.aroon(high, low, 25)
    au, ad_ = _last(a_up), _last(a_dn)
    osc_values["AROON_UP"] = au
    osc_values["AROON_DOWN"] = ad_
    if np.isfinite(au) and np.isfinite(ad_):
        ext["AROON"] = 1 if (au > 70 and au > ad_) else (-1 if (ad_ > 70 and ad_ > au) else 0)
    else:
        ext["AROON"] = 0

    # MFI: hacim agirlikli asiri alim/satim
    if volume is not None:
        v = _last(ta.mfi(high, low, close, volume, 14))
        osc_values["MFI14"] = v
        ext["MFI"] = 0 if not np.isfinite(v) else (-1 if v > 80 else (1 if v < 20 else 0))

    # Donchian konumu: 20 gunluk kanalin neresinde
    v = _last(ta.donchian_position(high, low, close, 20))
    osc_values["DONCHIAN_POS"] = v
    ext["DONCHIAN"] = 0 if not np.isfinite(v) else (1 if v > 0.80 else (-1 if v < 0.20 else 0))

    # Supertrend yonu
    try:
        v = _last(ta.supertrend_dir(high, low, close, 10, 3.0))
        osc_values["SUPERTREND"] = v
        ext["SUPERTREND"] = 0 if not np.isfinite(v) else int(np.sign(v))
    except Exception:
        ext["SUPERTREND"] = 0

    # Ichimoku bulut konumu
    try:
        v = _last(ta.ichimoku_position(high, low, close))
        osc_values["ICHIMOKU"] = v
        ext["ICHIMOKU"] = 0 if not np.isfinite(v) else int(np.sign(v))
    except Exception:
        ext["ICHIMOKU"] = 0

    # MA dizilimi (golden/death cross yapisi)
    ma50, ma200 = _last(ta.sma(close, 50)), _last(ta.sma(close, 200))
    if np.isfinite(ma50) and np.isfinite(ma200):
        ext["MA_CROSS"] = 1 if ma50 > ma200 else -1
        osc_values["MA50_OVER_MA200"] = ma50 / ma200 - 1
    else:
        ext["MA_CROSS"] = 0

    # Trend duzgunlugu (R²) — teknik analize uygunlugun dogrudan olcusu
    r2 = ta.trend_r_squared(close, 120)
    osc_values["TREND_R2"] = r2
    ext["TREND_QUALITY"] = 0 if not np.isfinite(r2) else (1 if r2 > 0.55 else (-1 if r2 < -0.35 else 0))

    # Hacim trendi onayi
    if volume is not None and len(volume.dropna()) > 60:
        v20 = float(volume.tail(20).mean())
        v60 = float(volume.tail(60).mean())
        ratio = v20 / v60 if v60 > 0 else np.nan
        osc_values["VOL_RATIO"] = ratio
        up = _last(close) > _last(ta.sma(close, 20))
        if np.isfinite(ratio):
            ext["VOLUME_CONFIRM"] = 1 if (ratio > 1.15 and up) else (-1 if (ratio > 1.15 and not up) else 0)
        else:
            ext["VOLUME_CONFIRM"] = 0

    osc.update(ext)
    all_votes = list(ma_votes.values()) + list(osc.values())
    buys = sum(1 for x in all_votes if x > 0)
    sells = sum(1 for x in all_votes if x < 0)
    neutrals = sum(1 for x in all_votes if x == 0)
    total = max(1, buys + sells + neutrals)

    ratio = (buys - sells) / total
    tr_label, en_label, base = _label(ratio)

    # Etiket bandinin icinde de ayrisim olsun diye orani skora karistiriyoruz.
    score = float(np.clip(50 + 50 * ratio, 0, 100))

    ma_buys = sum(1 for x in ma_votes.values() if x > 0)
    ma_sells = sum(1 for x in ma_votes.values() if x < 0)

    return {
        "available": True,
        "score": score,
        "label_tr": tr_label,
        "label": en_label,
        "band_score": base,
        "vote_ratio": float(ratio),
        "buy_votes": buys,
        "sell_votes": sells,
        "neutral_votes": neutrals,
        "ma_buy": ma_buys,
        "ma_sell": ma_sells,
        "osc_buy": sum(1 for x in osc.values() if x > 0),
        "osc_sell": sum(1 for x in osc.values() if x < 0),
        "signal_count": len(all_votes),
        "votes": {k: int(v) for k, v in {**ma_votes, **osc}.items()},
        "oscillator_values": {k: (None if not np.isfinite(v) else round(float(v), 4))
                              for k, v in osc_values.items()},
    }
