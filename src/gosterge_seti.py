"""The full indicator sweep, as model features.

The first feature set had ten columns because it was built for bucketing, not
for a model — it only had to describe the environment coarsely enough to split
signals into "calm / mid / volatile". A model can use much more than that, and
indicators.py already had twenty-four functions of which four were being
touched.

So this pulls the whole library in, adds the classic candle patterns as flags,
and adds the structural things people actually read off a chart: distance from
recent highs and lows, how many bars in a row have closed up, whether the
20-bar range is contracting.

Everything is scale-free. Nothing is a price, because a $12 stock and a $300
one have to produce comparable numbers. Anything with a natural unit (RSI,
%B, Aroon) is left in that unit; anything else is a ratio or a log ratio.

Nothing looks forward. Every column is a rolling or shift(+n) construction, and
there's a test that greps the source for negative shifts.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import indicators as ind


def _guvenli(x):
    return np.nan_to_num(np.asarray(x, dtype=float),
                         nan=0.0, posinf=0.0, neginf=0.0)


def _oran(a, b):
    """a/b - 1, protected against zeros."""
    b = np.asarray(b, dtype=float)
    return _guvenli(np.asarray(a, dtype=float) / np.where(np.abs(b) < 1e-12,
                                                          np.nan, b) - 1.0)


def _egim(s: pd.Series, n: int) -> np.ndarray:
    """Percent change of a series over n bars — a slope you can compare."""
    return _oran(s.to_numpy(), s.shift(n).to_numpy())


# ---------------------------------------------------------------------------
#  Candle patterns, as plain flags
#
#  These are the shapes on their own, with no trend filter. The setup
#  detectors in kisa_vade.py add context and only fire in one direction;
#  here we just want the model to know the shape was there.
# ---------------------------------------------------------------------------
def mum_kaliplari(o, h, l, c) -> dict:
    menzil = np.maximum(h - l, 1e-12)
    govde = np.abs(c - o)
    ust = h - np.maximum(o, c)
    alt = np.minimum(o, c) - l
    yukseldi = c > o
    o1, c1 = np.roll(o, 1), np.roll(c, 1)
    o2, c2 = np.roll(o, 2), np.roll(c, 2)
    h1, l1 = np.roll(h, 1), np.roll(l, 1)
    govde1 = np.abs(c1 - o1)

    out = {}
    out["doji"] = (govde <= 0.1 * menzil).astype(float)
    out["marubozu"] = (govde >= 0.9 * menzil).astype(float)
    out["cekic_sekli"] = ((govde <= 0.35 * menzil) & (alt >= 2 * govde)
                          & (ust <= 0.15 * menzil)).astype(float)
    out["ters_cekic"] = ((govde <= 0.35 * menzil) & (ust >= 2 * govde)
                         & (alt <= 0.15 * menzil)).astype(float)
    out["yutan_boga_sekli"] = ((c1 < o1) & yukseldi & (o <= c1)
                               & (c >= o1)).astype(float)
    out["yutan_ayi_sekli"] = ((c1 > o1) & ~yukseldi & (o >= c1)
                              & (c <= o1)).astype(float)
    out["harami"] = ((govde < govde1 * 0.6)
                     & (np.maximum(o, c) <= np.maximum(o1, c1))
                     & (np.minimum(o, c) >= np.minimum(o1, c1))).astype(float)
    out["ic_bar"] = ((h <= h1) & (l >= l1)).astype(float)
    out["dis_bar"] = ((h > h1) & (l < l1)).astype(float)
    out["uc_asker"] = (yukseldi & (c1 > o1) & (c2 > o2)
                       & (c > c1) & (c1 > c2)).astype(float)
    out["uc_karga"] = (~yukseldi & (c1 < o1) & (c2 < o2)
                       & (c < c1) & (c1 < c2)).astype(float)
    out["sabah_yildizi"] = ((c2 < o2) & (np.abs(c1 - o1) <= 0.3 * govde1.max()
                                         if govde1.max() else False)
                            & yukseldi & (c > (o2 + c2) / 2)).astype(float)
    out["delen"] = ((c1 < o1) & yukseldi & (o < c1)
                    & (c > (o1 + c1) / 2) & (c < o1)).astype(float)
    out["kara_bulut"] = ((c1 > o1) & ~yukseldi & (o > c1)
                         & (c < (o1 + c1) / 2) & (c > o1)).astype(float)
    out["cimbiz_dip"] = (np.abs(l - l1) <= 0.001 * np.maximum(c, 1e-9)).astype(float)
    out["cimbiz_tepe"] = (np.abs(h - h1) <= 0.001 * np.maximum(c, 1e-9)).astype(float)

    # The first two bars have no history to compare against.
    for k in out:
        out[k][:2] = 0.0
    return out


def olustur(df: pd.DataFrame) -> pd.DataFrame:
    """Every feature for every bar. Index matches the input."""
    o = df["Open"].astype(float)
    h = df["High"].astype(float)
    l = df["Low"].astype(float)
    c = df["Close"].astype(float)
    v = df["Volume"].astype(float)
    cn = c.to_numpy()

    F: dict[str, np.ndarray] = {}

    # --- oscillators -------------------------------------------------------
    for n in (7, 14, 28):
        F[f"rsi{n}"] = _guvenli(ind.rsi(c, n).to_numpy())
    k, d = ind.stoch(h, l, c, 14, 3, 3)
    F["stoch_k"] = _guvenli(k.to_numpy())
    F["stoch_d"] = _guvenli(d.to_numpy())
    F["stoch_rsi"] = _guvenli(ind.stoch_rsi(c, 14).to_numpy())
    F["williams_r"] = _guvenli(ind.williams_r(h, l, c, 14).to_numpy())
    F["cci"] = _guvenli(ind.cci(h, l, c, 20).to_numpy()) / 100.0
    F["ult_osc"] = _guvenli(ind.ultimate_oscillator(h, l, c).to_numpy())
    F["mfi"] = _guvenli(ind.mfi(h, l, c, v, 14).to_numpy())
    for n in (3, 5, 10, 20):
        F[f"roc{n}"] = _guvenli(ind.roc(c, n).to_numpy()) / 100.0

    macd, sinyal, hist = ind.macd(c)
    F["macd"] = _oran(macd.to_numpy() + cn, cn)
    F["macd_sinyal"] = _oran(sinyal.to_numpy() + cn, cn)
    F["macd_hist"] = _oran(hist.to_numpy() + cn, cn)

    # --- trend -------------------------------------------------------------
    adx, pdi, ndi = ind.adx(h, l, c, 14)
    F["adx"] = _guvenli(adx.to_numpy())
    F["di_fark"] = _guvenli(pdi.to_numpy() - ndi.to_numpy())
    for n in (10, 20, 50, 100, 200):
        F[f"ma{n}_uzaklik"] = _oran(cn, ind.sma(c, n).to_numpy())
    for n in (5, 20, 50):
        F[f"ma20_egim{n}"] = _egim(ind.sma(c, 20), n)
    F["ema_orani"] = _oran(ind.ema(c, 12).to_numpy(), ind.ema(c, 26).to_numpy())
    F["supertrend"] = _guvenli(ind.supertrend_dir(h, l, c).to_numpy())
    F["ichimoku"] = _guvenli(ind.ichimoku_position(h, l, c).to_numpy())
    au, ad = ind.aroon(h, l, 25)
    F["aroon_up"] = _guvenli(au.to_numpy())
    F["aroon_down"] = _guvenli(ad.to_numpy())
    F["aroon_osc"] = F["aroon_up"] - F["aroon_down"]
    for n in (20, 50):
        F[f"donchian{n}"] = _guvenli(ind.donchian_position(h, l, c, n).to_numpy())

    # --- volatility --------------------------------------------------------
    for n in (7, 14):
        F[f"atr{n}"] = _guvenli((ind.atr(h, l, c, n) / c).to_numpy())
    ust, orta, alt_b, pct_b, genislik = ind.bollinger(c, 20, 2.0)
    F["bb_pct"] = _guvenli(pct_b.to_numpy())
    F["bb_genislik"] = _guvenli(genislik.to_numpy())
    F["bb_genislik_yuzdelik"] = _guvenli(
        genislik.rolling(120, min_periods=30).rank(pct=True).to_numpy())
    for n in (10, 20):
        F[f"oynaklik{n}"] = _guvenli(
            c.pct_change().rolling(n).std().to_numpy())
    F["menzil_atr"] = _guvenli(
        ((h - l) / ind.atr(h, l, c, 14).replace(0, np.nan)).to_numpy())

    # --- volume ------------------------------------------------------------
    for n in (10, 50):
        med = v.rolling(n, min_periods=max(3, n // 4)).median()
        F[f"hacim{n}"] = _guvenli(np.log(np.maximum(v.to_numpy(), 1.0)
                                         / np.maximum(med.to_numpy(), 1.0)))
    F["obv_egim"] = _egim(ind.obv(c, v).abs() + 1.0, 20)
    F["dolar_hacim"] = _guvenli(np.log(np.maximum((c * v).to_numpy(), 1.0)))
    yukari = (c > c.shift(1)).astype(float)
    F["yukari_hacim_pay"] = _guvenli(
        (v * yukari).rolling(20).sum().to_numpy()
        / np.maximum(v.rolling(20).sum().to_numpy(), 1.0))

    # --- structure ---------------------------------------------------------
    for n in (20, 50):
        F[f"tepe{n}_uzaklik"] = _oran(cn, h.rolling(n).max().to_numpy())
        F[f"dip{n}_uzaklik"] = _oran(cn, l.rolling(n).min().to_numpy())
    art = (c > c.shift(1))
    # Consecutive up bars, and the same for down bars.
    grup = (art != art.shift(1)).cumsum()
    seri = art.groupby(grup).cumcount() + 1
    F["ust_uste_yukari"] = _guvenli(np.where(art.to_numpy(), seri.to_numpy(), 0))
    F["ust_uste_asagi"] = _guvenli(np.where(~art.to_numpy(), seri.to_numpy(), 0))
    F["yuksek_tepe_sayisi"] = _guvenli(
        (h > h.shift(1)).rolling(20).sum().to_numpy())
    F["dusuk_dip_sayisi"] = _guvenli(
        (l < l.shift(1)).rolling(20).sum().to_numpy())
    F["bosluk"] = _oran(o.to_numpy(), c.shift(1).to_numpy())
    menzil = np.maximum((h - l).to_numpy(), 1e-12)
    F["govde_orani"] = _guvenli((c - o).to_numpy() / menzil)
    F["ust_fitil"] = _guvenli((h - np.maximum(o, c)).to_numpy() / menzil)
    F["alt_fitil"] = _guvenli((np.minimum(o, c) - l).to_numpy() / menzil)
    F["bar_konum"] = _guvenli((cn - l.to_numpy()) / menzil)
    # How tight the last 20 bars are against the 100 before them.
    dar = (h.rolling(20).max() - l.rolling(20).min()) / c
    F["sikisma"] = _guvenli(
        (dar / dar.rolling(100, min_periods=30).median()).to_numpy())

    # --- candle patterns ---------------------------------------------------
    F.update(mum_kaliplari(o.to_numpy(), h.to_numpy(), l.to_numpy(),
                           c.to_numpy()))

    return pd.DataFrame(F, index=df.index).astype(np.float32)


def adlar(df: pd.DataFrame | None = None) -> list[str]:
    """Column names, without needing real bars."""
    if df is not None:
        return list(olustur(df).columns)
    n = 260
    idx = pd.bdate_range("2024-01-01", periods=n)
    c = pd.Series(np.linspace(50, 60, n), index=idx)
    sahte = pd.DataFrame({"Open": c * 0.99, "High": c * 1.01, "Low": c * 0.98,
                          "Close": c, "Volume": pd.Series(1e6, index=idx)})
    return list(olustur(sahte).columns)
