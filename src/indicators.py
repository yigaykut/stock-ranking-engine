"""Teknik gosterge hesaplamalari — saf pandas/numpy, harici TA kutuphanesi yok.

Investing.com'un "Teknik Ozet" tablosunu birebir yeniden uretmek icin gereken
tum gostergeler burada. Boylece scraping'e bagimli kalmadan ayni ciktiya
ulasiyoruz (site HTML'i degisince sistem bozulmaz).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------- ortalamalar
def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=max(2, n // 2)).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=max(2, n // 2)).mean()


# ---------------------------------------------------------------- osilatorler
def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0.0)
    dn = (-d).clip(lower=0.0)
    # Wilder yumusatmasi
    au = up.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    ad = dn.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = au / ad.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50.0) if ad.notna().any() else out


def stoch(high, low, close, k: int = 14, d: int = 3, smooth: int = 3):
    ll = low.rolling(k, min_periods=k // 2).min()
    hh = high.rolling(k, min_periods=k // 2).max()
    rng = (hh - ll).replace(0, np.nan)
    raw_k = 100 * (close - ll) / rng
    k_line = raw_k.rolling(smooth, min_periods=1).mean()
    d_line = k_line.rolling(d, min_periods=1).mean()
    return k_line, d_line


def stoch_rsi(close: pd.Series, n: int = 14) -> pd.Series:
    r = rsi(close, n)
    ll = r.rolling(n, min_periods=n // 2).min()
    hh = r.rolling(n, min_periods=n // 2).max()
    rng = (hh - ll).replace(0, np.nan)
    return (100 * (r - ll) / rng).clip(0, 100)


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    line = ema(close, fast) - ema(close, slow)
    sig = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return line, sig, line - sig


def true_range(high, low, close) -> pd.Series:
    pc = close.shift(1)
    return pd.concat([high - low, (high - pc).abs(), (low - pc).abs()], axis=1).max(axis=1)


def atr(high, low, close, n: int = 14) -> pd.Series:
    return true_range(high, low, close).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def adx(high, low, close, n: int = 14):
    """ADX + yon gostergeleri. Investing.com ADX(14) kullanir."""
    up_move = high.diff()
    dn_move = -low.diff()
    plus_dm = pd.Series(np.where((up_move > dn_move) & (up_move > 0), up_move, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((dn_move > up_move) & (dn_move > 0), dn_move, 0.0), index=high.index)

    tr_n = true_range(high, low, close).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    tr_n = tr_n.replace(0, np.nan)
    plus_di = 100 * plus_dm.ewm(alpha=1 / n, adjust=False, min_periods=n).mean() / tr_n
    minus_di = 100 * minus_dm.ewm(alpha=1 / n, adjust=False, min_periods=n).mean() / tr_n

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False, min_periods=n).mean(), plus_di, minus_di


def williams_r(high, low, close, n: int = 14) -> pd.Series:
    hh = high.rolling(n, min_periods=n // 2).max()
    ll = low.rolling(n, min_periods=n // 2).min()
    rng = (hh - ll).replace(0, np.nan)
    return -100 * (hh - close) / rng


def cci(high, low, close, n: int = 20) -> pd.Series:
    tp = (high + low + close) / 3
    ma = tp.rolling(n, min_periods=n // 2).mean()
    md = (tp - ma).abs().rolling(n, min_periods=n // 2).mean().replace(0, np.nan)
    return (tp - ma) / (0.015 * md)


def roc(close: pd.Series, n: int = 12) -> pd.Series:
    return 100 * (close / close.shift(n) - 1)


def ultimate_oscillator(high, low, close, s: int = 7, m: int = 14, l: int = 28) -> pd.Series:
    pc = close.shift(1)
    bp = close - pd.concat([low, pc], axis=1).min(axis=1)
    tr = true_range(high, low, close)

    def avg(n):
        return bp.rolling(n, min_periods=n // 2).sum() / tr.rolling(n, min_periods=n // 2).sum().replace(0, np.nan)

    return 100 * (4 * avg(s) + 2 * avg(m) + avg(l)) / 7


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    return (np.sign(close.diff().fillna(0.0)) * volume).fillna(0.0).cumsum()


# ------------------------------------------------- genisletilmis gostergeler
def bollinger(close: pd.Series, n: int = 20, k: float = 2.0):
    """Doner: (ust, orta, alt, %B, bant genisligi).

    Bant genisligi 'volatilite sikismasi' (squeeze) tespitinin temelidir —
    daralan bantlar biriken potansiyel enerjiyi gosterir.
    """
    mid = sma(close, n)
    sd = close.rolling(n, min_periods=max(2, n // 2)).std(ddof=0)
    upper, lower = mid + k * sd, mid - k * sd
    rng = (upper - lower).replace(0, np.nan)
    pct_b = (close - lower) / rng
    width = rng / mid.replace(0, np.nan)
    return upper, mid, lower, pct_b, width


def aroon(high: pd.Series, low: pd.Series, n: int = 25):
    """Trendin YASINI olcer: son zirve/dip ne kadar yakin zamanda olustu."""
    def _since_max(x):
        return float(len(x) - 1 - int(np.argmax(x)))

    def _since_min(x):
        return float(len(x) - 1 - int(np.argmin(x)))

    hh = high.rolling(n, min_periods=n).apply(_since_max, raw=True)
    ll = low.rolling(n, min_periods=n).apply(_since_min, raw=True)
    return 100 * (n - hh) / n, 100 * (n - ll) / n


def mfi(high, low, close, volume, n: int = 14) -> pd.Series:
    """Para Akisi Endeksi — hacim agirlikli RSI."""
    tp = (high + low + close) / 3
    flow = tp * volume
    d = tp.diff()
    pos = flow.where(d > 0, 0.0).rolling(n, min_periods=n // 2).sum()
    neg = flow.where(d < 0, 0.0).rolling(n, min_periods=n // 2).sum()
    ratio = pos / neg.replace(0, np.nan)
    return 100 - (100 / (1 + ratio))


def donchian_position(high, low, close, n: int = 20) -> pd.Series:
    """Fiyatin n-gunluk kanal icindeki konumu (0=dip, 1=zirve)."""
    hh = high.rolling(n, min_periods=n // 2).max()
    ll = low.rolling(n, min_periods=n // 2).min()
    return (close - ll) / (hh - ll).replace(0, np.nan)


def supertrend_dir(high, low, close, n: int = 10, mult: float = 3.0) -> pd.Series:
    """Supertrend yonu (+1 yukari, -1 asagi) — ATR tabanli takip eden durdurma."""
    atr_v = atr(high, low, close, n)
    hl2 = (high + low) / 2
    upper = hl2 + mult * atr_v
    lower = hl2 - mult * atr_v

    direction = pd.Series(index=close.index, dtype=float)
    prev_dir = 1.0
    prev_up, prev_lo = np.nan, np.nan

    for i in range(len(close)):
        c = float(close.iloc[i])
        u, lo = float(upper.iloc[i]), float(lower.iloc[i])
        if not np.isfinite(u) or not np.isfinite(lo):
            direction.iloc[i] = np.nan
            continue
        if np.isfinite(prev_up):
            u = min(u, prev_up) if prev_dir > 0 else u
            lo = max(lo, prev_lo) if prev_dir < 0 else lo
        if c > (prev_up if np.isfinite(prev_up) else u):
            prev_dir = 1.0
        elif c < (prev_lo if np.isfinite(prev_lo) else lo):
            prev_dir = -1.0
        direction.iloc[i] = prev_dir
        prev_up, prev_lo = u, lo
    return direction


def ichimoku_position(high, low, close) -> pd.Series:
    """Fiyatin Ichimoku bulutuna gore konumu: +1 ustunde, 0 icinde, -1 altinda."""
    def mid_ch(n):
        return (high.rolling(n, min_periods=n // 2).max() +
                low.rolling(n, min_periods=n // 2).min()) / 2

    span_a = ((mid_ch(9) + mid_ch(26)) / 2).shift(26)
    span_b = mid_ch(52).shift(26)
    top = pd.concat([span_a, span_b], axis=1).max(axis=1)
    bot = pd.concat([span_a, span_b], axis=1).min(axis=1)
    return pd.Series(np.where(close > top, 1.0, np.where(close < bot, -1.0, 0.0)),
                     index=close.index).where(top.notna())


def trend_r_squared(close: pd.Series, n: int = 120) -> float:
    """Log-fiyatin dogrusal trende UYUM iyiligi (R²).

    Yuksek R² = duzgun, istikrarli, TEKNIK ANALIZE UYGUN trend.
    Dusuk R² = testere disi, tahmin edilemez hareket.
    """
    y = np.log(close.dropna().tail(n).to_numpy(dtype=float))
    if len(y) < max(20, n // 3) or not np.all(np.isfinite(y)):
        return float("nan")
    x = np.arange(len(y), dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (slope * x + intercept)
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    if ss_tot <= 0:
        return float("nan")
    r2 = 1 - ss_res / ss_tot
    # Yonu koru: dususte duzgun trend "iyi" sayilmamali
    return float(r2 if slope > 0 else -r2)


# ------------------------------------------------------------------ yardimci
def slope_pct(s: pd.Series, n: int) -> float:
    """Son n barin dogrusal regresyon egimi, seviyeye gore yuzde olarak."""
    y = s.dropna().tail(n)
    if len(y) < max(3, n // 2):
        return float("nan")
    x = np.arange(len(y), dtype=float)
    beta = np.polyfit(x, y.to_numpy(dtype=float), 1)[0]
    base = float(np.abs(y.mean()))
    return float(100 * beta / base) if base > 0 else float("nan")


def annualized_vol(close: pd.Series, n: int = 252) -> float:
    r = close.pct_change().dropna().tail(n)
    return float(r.std() * np.sqrt(252)) if len(r) > 20 else float("nan")


def max_drawdown(close: pd.Series, n: int = 252) -> float:
    c = close.dropna().tail(n)
    if len(c) < 20:
        return float("nan")
    return float((c / c.cummax() - 1).min())
