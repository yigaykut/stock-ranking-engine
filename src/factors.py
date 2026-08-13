"""Her hisse icin HAM faktor degerlerini uretir.

Cikti sozlugu iki bolumden olusur:
  raw[<factor_id>]  -> ham sayisal deger (None = veri yok)
  meta[...]         -> aciklama, ara degerler, ceza bayraklari

Normalizasyon ve agirliklandirma burada YAPILMAZ; o is scoring.py'de,
tum evren toplandiktan sonra (capraz kesitsel siralama icin) yapilir.
Bu ayrim ayni zamanda ML icin temiz bir ozellik matrisi verir.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from . import indicators as ta
from . import investing_summary
from .providers import reddit_wsb


def _f(x: Any) -> float:
    """Guvenli float donusumu; gecersizse nan."""
    try:
        if x is None:
            return float("nan")
        v = float(x)
        return v if math.isfinite(v) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def _ok(x: float) -> bool:
    return isinstance(x, float) and math.isfinite(x)


def _n(x: float) -> float | None:
    """nan -> None (JSON uyumlu)."""
    return None if not _ok(_f(x)) else float(x)


def _ret(close: pd.Series, days: int) -> float:
    c = close.dropna()
    if len(c) <= days:
        return float("nan")
    return float(c.iloc[-1] / c.iloc[-1 - days] - 1)


# =============================================================================
#  MOMENTUM
# =============================================================================
def f_price_momentum_12_1(close: pd.Series) -> tuple[float, dict]:
    """12 aylik getiri, son 1 ay haric (kisa vadeli reversal temizligi)."""
    c = close.dropna()
    if len(c) < 260:
        return float("nan"), {}
    p_now, p_1m, p_12m = c.iloc[-22], c.iloc[-1], c.iloc[-252]
    mom = float(p_now / p_12m - 1)
    return mom, {"ret_12m": _n(float(p_1m / p_12m - 1)), "ret_12_1": _n(mom)}


def f_relative_strength(close: pd.Series, bench: pd.Series | None) -> tuple[float, dict]:
    """IBD tarzi agirlikli goreli guc: 3ay x2 + 6ay + 9ay + 12ay."""
    if bench is None or len(bench.dropna()) < 60:
        return float("nan"), {}

    # Ortak takvime hizala (farkli borsa tatilleri)
    df = pd.concat([close.rename("s"), bench.rename("b")], axis=1).dropna()
    if len(df) < 70:
        return float("nan"), {}

    parts, weights, detail = [], [], {}
    for days, w, key in ((63, 2.0, "3m"), (126, 1.0, "6m"), (189, 1.0, "9m"), (252, 1.0, "12m")):
        if len(df) > days:
            rs = float(df["s"].iloc[-1] / df["s"].iloc[-1 - days]) - \
                 float(df["b"].iloc[-1] / df["b"].iloc[-1 - days])
            parts.append(rs * w)
            weights.append(w)
            detail[f"rs_{key}"] = round(rs, 4)

    if not parts:
        return float("nan"), {}
    return float(sum(parts) / sum(weights)), detail


# =============================================================================
#  TEKNIK
# =============================================================================
def f_trend_structure(df: pd.DataFrame) -> tuple[float, dict]:
    """Minervini "Trend Template" — 8 kosullu 0-100 puan.

    Kullanicinin "borsasi teknik analize uygun olsun" istegi bu faktorde:
    net, duzenli, analiz edilebilir bir yukselis trendi yuksek puan alir.
    """
    close = df["Close"].dropna()
    if len(close) < 150:
        return float("nan"), {}

    price = float(close.iloc[-1])
    ma50 = ta.sma(close, 50).iloc[-1]
    ma150 = ta.sma(close, 150).iloc[-1]
    ma200 = ta.sma(close, 200).iloc[-1] if len(close) >= 200 else np.nan
    if not _ok(_f(ma200)):
        ma200 = ta.sma(close, min(200, len(close) - 1)).iloc[-1]

    lo52 = float(close.tail(252).min())
    hi52 = float(close.tail(252).max())
    ma200_slope = ta.slope_pct(ta.sma(close, 200), 25)

    checks = {
        "price_above_ma150_ma200": price > _f(ma150) and price > _f(ma200),
        "ma150_above_ma200": _f(ma150) > _f(ma200),
        "ma200_trending_up": _ok(ma200_slope) and ma200_slope > 0,
        "ma50_above_ma150_ma200": _f(ma50) > _f(ma150) and _f(ma50) > _f(ma200),
        "price_above_ma50": price > _f(ma50),
        "price_30pct_above_52w_low": lo52 > 0 and price >= lo52 * 1.30,
        "price_within_25pct_of_52w_high": hi52 > 0 and price >= hi52 * 0.75,
        "ma50_trending_up": _ok(ta.slope_pct(ta.sma(close, 50), 15)) and ta.slope_pct(ta.sma(close, 50), 15) > 0,
    }
    passed = sum(1 for v in checks.values() if v)
    score = 100.0 * passed / len(checks)

    meta = {
        "checks": {k: bool(v) for k, v in checks.items()},
        "passed": passed,
        "total": len(checks),
        "pct_from_52w_high": _n(100 * (price / hi52 - 1)) if hi52 > 0 else None,
        "pct_above_52w_low": _n(100 * (price / lo52 - 1)) if lo52 > 0 else None,
        "pct_above_ma50": _n(100 * (price / _f(ma50) - 1)) if _ok(_f(ma50)) else None,
        "ma50": _n(_f(ma50)), "ma150": _n(_f(ma150)), "ma200": _n(_f(ma200)),
    }
    return score, meta


def f_breakout_setup(df: pd.DataFrame) -> tuple[float, dict]:
    """POTANSIYEL ENERJI: sikisma / baz olusumu / kirilim kurulumu.

    Bu faktor "zaten kosmus" hisseyi degil, "kosmaya HAZIR" hisseyi arar —
    aradigimiz sey potansiyel oldugu icin sistemin merkezinde.

    Bilesenler:
      * Volatilite sikismasi (VCP): Bollinger bant genisligi kendi 6 aylik
        dagiliminin neresinde — dar bant = birikmis enerji
      * Baz sikiligi: son 6 haftalik islem araligi ne kadar dar
      * Kirilim yakinligi: fiyat son 3 aylik zirveye ne kadar yakin
      * ATR daralmasi: kisa vadeli oynaklik uzun vadeliye gore dusuyor mu
    """
    close, high, low = df["Close"], df["High"], df["Low"]
    if len(close.dropna()) < 130:
        return float("nan"), {}

    parts, meta = [], {}

    # --- 1) Bollinger sikismasi (kendi gecmisine gore yuzdelik)
    _, _, _, _, width = ta.bollinger(close, 20, 2.0)
    w = width.dropna().tail(126)
    if len(w) > 40:
        cur = float(w.iloc[-1])
        pct = float((w < cur).mean())        # 0 = en dar, 1 = en genis
        squeeze = 1.0 - 2.0 * pct            # dar -> +1, genis -> -1
        parts.append((1.5, squeeze))
        meta["bb_width_percentile"] = round(pct, 3)
        meta["bb_width"] = round(cur, 4)

    # --- 2) Baz sikiligi: son 30 gunun araligi / fiyat
    c30 = close.tail(30)
    if len(c30) >= 20:
        rng = float((c30.max() - c30.min()) / c30.mean())
        # %8 cok siki -> +1 ; %35 genis -> -1
        tight = max(-1.0, min(1.0, (0.22 - rng) / 0.14))
        parts.append((1.2, tight))
        meta["base_range_30d_pct"] = round(rng * 100, 2)

    # --- 3) Kirilim yakinligi: 3 aylik zirveye mesafe
    hi63 = float(close.tail(63).max())
    px = float(close.iloc[-1])
    if hi63 > 0:
        dist = px / hi63 - 1                  # 0 = zirvede, negatif = altinda
        # zirveye cok yakin (%0-4) ideal kurulum
        near = max(-1.0, min(1.0, (dist + 0.04) / 0.04)) if dist <= 0 else 0.4
        parts.append((1.3, near))
        meta["dist_to_3m_high_pct"] = round(dist * 100, 2)

    # --- 4) ATR daralmasi
    atr14 = ta.atr(high, low, close, 14).dropna()
    if len(atr14) > 90:
        short_atr = float(atr14.tail(14).mean())
        long_atr = float(atr14.tail(90).mean())
        if long_atr > 0:
            contraction = 1.0 - short_atr / long_atr   # pozitif = daraliyor
            parts.append((1.0, max(-1.0, min(1.0, contraction / 0.25))))
            meta["atr_contraction"] = round(contraction, 3)

    if not parts:
        return float("nan"), {}

    tw = sum(w for w, _ in parts)
    val = sum(w * v for w, v in parts) / tw
    meta["components_used"] = len(parts)
    return float(val), meta


def f_size_opportunity(bundle: dict) -> tuple[float, dict]:
    """OLCEK FIRSATI: kucuk sirketin katlanma alani buyuktur.

    500 milyar dolarlik bir sirket 10 katina cikamaz; 1 milyarlik cikabilir.
    Ters-U puanlama (log olcek):
      * < 150M  -> cok riskli, likidite yok, manipulasyona acik (ceza)
      * 400M-6Mr -> tatli nokta: kurumsal radara girecek kadar buyuk,
                    katlanacak kadar kucuk
      * > 50Mr  -> katlanma potansiyeli yapisal olarak sinirli
    """
    info = bundle.get("info") or {}
    mcap = _f(info.get("marketCap"))
    if not _ok(mcap) or mcap <= 0:
        return float("nan"), {}

    lg = math.log10(mcap)
    # tepe ~1.5Mr (10^9.18); genislik ~0.95 dekad
    peak, width = 9.18, 0.95
    score = 1.0 - ((lg - peak) / width) ** 2

    # cok kucukleri ayrica cezalandir (veri kalitesi + likidite riski)
    if mcap < 1.5e8:
        score -= 0.6

    return float(max(-1.5, min(1.0, score))), {
        "market_cap": _n(mcap),
        "market_cap_log10": round(lg, 2),
        "band": ("micro" if mcap < 3e8 else "small" if mcap < 2e9 else
                 "mid" if mcap < 1e10 else "large" if mcap < 5e10 else "mega"),
    }


def f_revenue_scaling(bundle: dict) -> tuple[float, dict]:
    """CIRO BUYUMESI ve IVMESI — henuz kar etmeyen sirketler icin dogru olcu.

    Yukselen sirketlerin cogu kar etmez ama cirosu hizla buyur. EPS'e bakan
    bir sistem bu sirketleri haksiz yere eler. Bu faktor ciroyu merkeze alir.
    """
    info = bundle.get("info") or {}
    rg = _f(info.get("revenueGrowth"))
    parts, meta = [], {"revenue_growth": _n(rg)}

    if _ok(rg):
        # %25 iyi, %60+ mukemmel
        parts.append((2.0, max(-1.5, min(1.5, rg / 0.25))))

    # Ceyreklik gelir tablosundan ivme: son ceyrek buyumesi > yillik buyume mu
    inc = bundle.get("income")
    if isinstance(inc, pd.DataFrame) and not inc.empty:
        for key in ("Total Revenue", "TotalRevenue", "OperatingRevenue"):
            if key in inc.index:
                rev = pd.to_numeric(inc.loc[key], errors="coerce").dropna()
                # sutunlar yeniden eskiye dogru
                if len(rev) >= 5:
                    latest, year_ago = float(rev.iloc[0]), float(rev.iloc[4])
                    prev = float(rev.iloc[1])
                    prev_year = float(rev.iloc[5]) if len(rev) >= 6 else None
                    if year_ago > 0:
                        yoy = latest / year_ago - 1
                        meta["revenue_yoy_latest_q"] = round(yoy, 4)
                        parts.append((1.2, max(-1.5, min(1.5, yoy / 0.25))))
                        if prev_year and prev_year > 0:
                            yoy_prev = prev / prev_year - 1
                            accel = yoy - yoy_prev
                            meta["revenue_growth_acceleration"] = round(accel, 4)
                            # HIZLANMA seviyeden daha bilgilendirici
                            parts.append((1.5, max(-1.0, min(1.0, accel / 0.10))))
                break

    # Ileriye donuk ciro beklentisi
    ge = bundle.get("growth_estimates")
    if isinstance(ge, pd.DataFrame) and not ge.empty and "stockTrend" in ge.columns:
        if "+1y" in ge.index:
            nxt = _f(ge.loc["+1y", "stockTrend"])
            if _ok(nxt):
                parts.append((0.8, max(-1.0, min(1.5, nxt / 0.25))))
                meta["growth_est_next_year"] = round(nxt, 4)

    if not parts:
        return float("nan"), meta
    tw = sum(w for w, _ in parts)
    return float(sum(w * v for w, v in parts) / tw), meta


def f_rule_of_40(bundle: dict) -> tuple[float, dict]:
    """40 KURALI: buyume% + marj% >= 40.

    Buyume sirketlerinin standart kalite barajı. %60 buyuyup -%20 marj eden
    bir sirket de, %10 buyuyup %30 marj eden de baraji gecer. Boylece
    kar etmeyen ama hizli buyuyen sirketler adil degerlendirilir —
    klasik karlilik faktorlerinin yapamadigi sey budur.
    """
    info = bundle.get("info") or {}
    rg = _f(info.get("revenueGrowth"))
    if not _ok(rg):
        return float("nan"), {}

    # Marj olarak once FCF marji, yoksa faaliyet marji, yoksa net marj
    margin, margin_kind = float("nan"), None
    fcf, rev = _f(info.get("freeCashflow")), _f(info.get("totalRevenue"))
    if _ok(fcf) and _ok(rev) and rev > 0:
        margin, margin_kind = fcf / rev, "fcf"
    else:
        for key, kind in (("operatingMargins", "operating"), ("profitMargins", "net"),
                          ("ebitdaMargins", "ebitda")):
            v = _f(info.get(key))
            if _ok(v):
                margin, margin_kind = v, kind
                break

    if not _ok(margin):
        return float("nan"), {}

    score40 = 100 * (rg + margin)
    # 40 baraj, 70+ olaganustu, 10 zayif
    val = (score40 - 40.0) / 30.0
    return float(max(-1.5, min(1.5, val))), {
        "rule_of_40_score": round(score40, 1),
        "revenue_growth_pct": round(rg * 100, 1),
        "margin_pct": round(margin * 100, 1),
        "margin_type": margin_kind,
        "passes_rule_of_40": bool(score40 >= 40),
    }


def f_undiscovered(bundle: dict) -> tuple[float, dict]:
    """HENUZ KESFEDILMEMIS: dusuk analist takibi + kurumsallara alan.

    Bir hisse 40 analist tarafindan takip ediliyor ve kurumlarin %90'i
    elindeyse, iyi haber zaten fiyatta demektir. Asil potansiyel, kurumsal
    parayi HENUZ cekmemis ama cekmeye aday sirkettedir.

    DIKKAT: sifir takip de kotudur (bilgi yok, likidite yok). Tatli nokta
    az ama sifir olmayan takip + %25-65 kurumsal sahiplik.
    """
    info = bundle.get("info") or {}
    n_an = _f(info.get("numberOfAnalystOpinions"))
    inst = _f(info.get("heldPercentInstitutions"))
    ins = _f(info.get("heldPercentInsiders"))

    parts, meta = [], {}

    if _ok(n_an):
        meta["analyst_count"] = _n(n_an)
        # 3-12 analist ideal; 0 kotu, 35+ = doymus
        if n_an <= 0:
            parts.append((1.0, -0.7))
        elif n_an <= 12:
            parts.append((1.2, 1.0 - abs(n_an - 7.0) / 9.0))
        else:
            parts.append((1.2, max(-1.0, 0.55 - (n_an - 12) / 22.0)))

    if _ok(inst):
        meta["held_percent_institutions"] = _n(inst)
        # %25-65 = kurumlar farketmis ama doymamis; alan var
        if inst < 0.05:
            parts.append((1.0, -0.5))          # hic kurumsal ilgi yok
        else:
            parts.append((1.3, 1.0 - abs(min(inst, 1.0) - 0.45) / 0.45))

    if _ok(ins):
        meta["held_percent_insiders"] = _n(ins)
        # iceriden sahiplik = kurucu hala oyunda (cikar birligi)
        parts.append((0.7, max(-0.5, min(1.0, ins / 0.15))))

    if not parts:
        return float("nan"), meta
    tw = sum(w for w, _ in parts)
    return float(sum(w * v for w, v in parts) / tw), meta


def f_cash_runway(bundle: dict) -> tuple[float, dict]:
    """NAKIT OMRU: kar etmeyen sirket ne kadar dayanabilir?

    Kucuk sirket avinda hayati onem tasir. Nakit yakan ve nakdi tukenen sirket
    ya batar ya da hisse basina degeri eriten sermaye artirimina gider.
    Kendi kendini finanse eden (pozitif nakit akisi) sirket en yuksek puani alir.
    """
    info = bundle.get("info") or {}
    fcf = _f(info.get("freeCashflow"))
    ocf = _f(info.get("operatingCashflow"))
    cash = _f(info.get("totalCash"))
    debt = _f(info.get("totalDebt"))

    flow = fcf if _ok(fcf) else ocf
    if not _ok(flow):
        return float("nan"), {}

    meta = {"free_cash_flow": _n(fcf), "operating_cash_flow": _n(ocf),
            "total_cash": _n(cash), "total_debt": _n(debt)}

    if flow > 0:
        # Kendini finanse ediyor. Nakit/borc dengesi ile ince ayar.
        val = 1.0
        if _ok(cash) and _ok(debt) and debt > 0:
            ratio = cash / debt
            meta["cash_to_debt"] = round(ratio, 2)
            val = min(1.5, 0.7 + 0.5 * min(2.0, ratio))
        meta["self_funding"] = True
        meta["runway_years"] = None
    else:
        meta["self_funding"] = False
        if not _ok(cash) or cash <= 0:
            return -1.2, meta          # yakiyor ve nakit yok
        years = cash / abs(flow)
        meta["runway_years"] = round(years, 2)
        # 1 yil kritik, 3 yil rahat
        val = max(-1.5, min(0.8, (years - 1.5) / 1.5))

    return float(val), meta


def f_stage2_breakout(df: pd.DataFrame) -> tuple[float, dict]:
    """WEINSTEIN ASAMA ANALIZI: uzun bazdan YENI cikmis olmak.

    Aradigimiz "gelecek vadeden" tanimi teknik olarak budur:
      Asama 1 = uzun yatay baz (birikim)
      Asama 2 = kirilim ve yukselis  <-- ERKEN asamasi en yuksek potansiyel
      Asama 3 = tepe / dagitim
      Asama 4 = dusus

    Bu faktor Asama 2'nin BASINDA olan hisseleri odullendirir; uzun suredir
    kosanlari (gec Asama 2 / Asama 3) odullendirmez.
    """
    close = df["Close"].dropna()
    if len(close) < 200:
        return float("nan"), {}

    ma150 = ta.sma(close, 150)
    px = float(close.iloc[-1])
    parts, meta = [], {}

    # --- 1) MA150 uzerinde mi ve MA150 yukari egimli mi (Asama 2 sarti)
    m150 = _f(ma150.iloc[-1])
    slope150 = ta.slope_pct(ma150, 30)
    above = _ok(m150) and px > m150
    rising = _ok(slope150) and slope150 > 0
    meta["above_ma150"] = bool(above)
    meta["ma150_rising"] = bool(rising)

    if not above:
        return -1.0, meta          # Asama 4 veya 1 -> aradigimiz sey degil
    parts.append((1.0, 1.0 if rising else -0.2))

    # --- 2) NE ZAMANDIR MA150 uzerinde? (erken = iyi)
    rel = (close - ma150).dropna()
    below = rel[rel <= 0]
    if len(below):
        last_below = below.index[-1]
        days_above = int((close.index > last_below).sum())
    else:
        days_above = len(rel)
    meta["days_above_ma150"] = days_above

    # 20-160 gun = erken/orta Asama 2 (ideal ~60 gun)
    # <15 gun = henuz teyit edilmemis;  >260 gun = gec asama
    if days_above < 15:
        parts.append((1.4, 0.15))
    elif days_above <= 160:
        parts.append((1.4, 1.0 - abs(days_above - 70) / 130.0))
    else:
        parts.append((1.4, max(-1.0, 0.45 - (days_above - 160) / 220.0)))

    # --- 3) Kirilim oncesi baz NE KADAR UZUNDU? (uzun baz = guclu yay)
    #     Kirilimdan onceki 200 gunde fiyat araliginin darligina bakiyoruz.
    if days_above < len(close) - 60:
        pre = close.iloc[max(0, len(close) - days_above - 200): len(close) - days_above]
        if len(pre) >= 60:
            rng = float((pre.max() - pre.min()) / pre.mean())
            meta["prior_base_range_pct"] = round(rng * 100, 1)
            meta["prior_base_days"] = int(len(pre))
            # %35 dar baz -> +1 ; %90 genis -> -1
            parts.append((1.2, max(-1.0, min(1.0, (0.62 - rng) / 0.27))))

    # --- 4) 52 haftalik dipten yukselis ama zirveden asiri uzak degil
    win = close.tail(252)
    lo, hi = float(win.min()), float(win.max())
    if lo > 0 and hi > lo:
        from_low = px / lo - 1
        meta["pct_above_52w_low"] = round(from_low * 100, 1)
        # %20-120 saglikli Asama 2; %300+ gec kalinmis
        parts.append((0.9, max(-1.0, min(1.0, (1.30 - from_low) / 0.9))))

    tw = sum(w for w, _ in parts)
    return float(sum(w * v for w, v in parts) / tw), meta


def f_chart_position(df: pd.DataFrame, bundle: dict) -> tuple[float, dict]:
    """YUKARI ALAN: fiyatin yillik araligindaki konumu ve tavana mesafesi.

    Potansiyel arayisinin ikinci yarisi. Ters-U puanlama:
      * dipte surunen hisse (dusen bicak) -> dusuk puan
      * zirvenin cok uzerinde uzamis hisse -> dusuk puan (yukari alan kalmamis)
      * zirvenin hemen altinda, saglam bolgede -> yuksek puan
    Ayrica analist hedef fiyatina gore kalan yol da hesaba katilir.
    """
    close = df["Close"].dropna()
    if len(close) < 130:
        return float("nan"), {}

    px = float(close.iloc[-1])
    win = close.tail(252)
    lo52, hi52 = float(win.min()), float(win.max())
    if hi52 <= lo52:
        return float("nan"), {}

    pos = (px - lo52) / (hi52 - lo52)        # 0 = 52h dip, 1 = 52h zirve
    # Ters-U: en iyi bolge 0.62-0.92 (guclu ama tukenmemis)
    ideal = 0.77
    band = 0.30
    position_score = 1.0 - abs(pos - ideal) / band
    position_score = max(-1.2, min(1.0, position_score))

    parts = [(1.5, position_score)]
    meta = {
        "range_position": round(pos, 3),
        "pct_below_52w_high": round((px / hi52 - 1) * 100, 2),
        "pct_above_52w_low": round((px / lo52 - 1) * 100, 2),
    }

    # --- tarihi zirveye gore kalan alan (mevcut gecmis icinde)
    all_hi = float(close.max())
    if all_hi > 0:
        room = all_hi / px - 1               # zirvenin altindaysa pozitif
        meta["room_to_period_high_pct"] = round(room * 100, 2)
        # %0-25 arasi kalan yol saglikli; %60+ ise yapisal sorun sinyali
        parts.append((0.8, max(-1.0, min(1.0, (0.30 - abs(room - 0.12)) / 0.25))))

    return float(sum(w * v for w, v in parts) / sum(w for w, _ in parts)), meta


def f_analyst_upside(bundle: dict) -> tuple[float, dict]:
    """POTANSIYEL: analist hedef fiyatina gore kalan yukari alan.

    Dogrudan "ne kadar yukselebilir" sorusunu olcer. Asiri iyimser hedefler
    (>%80) guvenilirligini kaybettigi icin kirpilir.
    """
    tgt = bundle.get("price_targets") or {}
    info = bundle.get("info") or {}

    cur = _f(tgt.get("current")) or _f(info.get("currentPrice"))
    mean = _f(tgt.get("mean"))
    high = _f(tgt.get("high"))
    low = _f(tgt.get("low"))

    if not _ok(cur) or cur <= 0 or not _ok(mean):
        return float("nan"), {}

    upside = mean / cur - 1
    parts = [(1.5, max(-1.0, min(1.5, upside / 0.25)))]
    meta = {
        "target_mean": round(mean, 2),
        "upside_to_mean_pct": round(upside * 100, 2),
    }

    # Hedef araliginin asimetrisi: yukari alan asagi riskten buyuk mu
    if _ok(high) and _ok(low) and high > low:
        up_room = high / cur - 1
        down_room = 1 - low / cur
        meta["upside_to_high_pct"] = round(up_room * 100, 2)
        meta["downside_to_low_pct"] = round(down_room * 100, 2)
        if down_room > 0:
            skew = (up_room - down_room) / (up_room + down_room)
            parts.append((1.0, max(-1.0, min(1.0, skew / 0.4))))
            meta["risk_reward_skew"] = round(skew, 3)

    return float(sum(w * v for w, v in parts) / sum(w for w, _ in parts)), meta


def f_momentum_persistence(df: pd.DataFrame) -> tuple[float, dict]:
    """TEKNIK ANALIZE UYGUNLUK: trendin duzgunlugu ve tutarliligi.

    Kullanicinin "borsasi teknik analize uygun olsun" istegini dogrudan olcer.
    Testere disi, bosluklu, tahmin edilemez hisseler dusuk puan alir; duzgun,
    okunabilir trendler yuksek.
    """
    close = df["Close"].dropna()
    if len(close) < 130:
        return float("nan"), {}

    parts, meta = [], {}

    # --- 1) Log-fiyat trendine uyum (R²)
    r2 = ta.trend_r_squared(close, 120)
    if _ok(r2):
        parts.append((1.6, max(-1.0, min(1.0, r2 / 0.6))))
        meta["trend_r_squared"] = round(r2, 3)

    # --- 2) Pozitif gun orani
    rets = close.pct_change().dropna().tail(120)
    if len(rets) > 60:
        win_rate = float((rets > 0).mean())
        parts.append((0.8, max(-1.0, min(1.0, (win_rate - 0.50) / 0.06))))
        meta["positive_day_rate"] = round(win_rate, 3)

    # --- 3) Bosluk / sicrama cezasi: asiri buyuk gunluk hareketler okunabilirligi bozar
    if len(rets) > 60:
        jump_rate = float((rets.abs() > 0.07).mean())
        parts.append((0.7, max(-1.0, 1.0 - jump_rate / 0.06)))
        meta["large_gap_rate"] = round(jump_rate, 4)

    # --- 4) Ust ust yeni zirve yapabilme (yapisal yukselis trendi)
    roll_max = close.rolling(60, min_periods=30).max()
    if roll_max.notna().sum() > 60:
        new_high_rate = float((close.tail(120) >= roll_max.tail(120) * 0.999).mean())
        parts.append((0.9, max(-1.0, min(1.0, (new_high_rate - 0.06) / 0.12))))
        meta["new_high_frequency"] = round(new_high_rate, 3)

    if not parts:
        return float("nan"), {}
    return float(sum(w * v for w, v in parts) / sum(w for w, _ in parts)), meta


def f_volume_accumulation(df: pd.DataFrame) -> tuple[float, dict]:
    """OBV egimi + hacim genislemesi = kurumsal toplama gostergesi."""
    if "Volume" not in df or len(df) < 60:
        return float("nan"), {}

    close, vol = df["Close"], df["Volume"]
    obv_series = ta.obv(close, vol)
    obv_slope = ta.slope_pct(obv_series, 40)

    v20 = float(vol.tail(20).mean())
    v60 = float(vol.tail(60).mean())
    vol_ratio = v20 / v60 if v60 > 0 else float("nan")

    price_slope = ta.slope_pct(close, 40)

    if not _ok(obv_slope) or not _ok(vol_ratio):
        return float("nan"), {}

    # OBV egimi ana bilesen, hacim genislemesi destekleyici
    raw = obv_slope + 8.0 * (vol_ratio - 1.0)

    # Negatif uyumsuzluk (dagitim): fiyat yukari, OBV asagi
    divergence = _ok(price_slope) and price_slope > 0.05 and obv_slope < -0.05

    return float(raw), {
        "obv_slope_pct": round(obv_slope, 4),
        "volume_ratio_20_60": round(vol_ratio, 3),
        "price_slope_pct": _n(price_slope),
        "distribution_divergence": bool(divergence),
    }


# =============================================================================
#  ANALIST / REVIZYON  (sistemin en agir ailesi)
# =============================================================================
def f_eps_revision_momentum(bundle: dict) -> tuple[float, dict]:
    """Zacks tarzi sinyal: EPS tahminleri son 30-90 gunde yukari mi gitti?

    Ucu birlestirilir:
      1. eps_trend  -> cari yil tahmininin 30/90 gun once ile karsilastirmasi
      2. eps_revisions -> yukari/asagi revizyon sayisi dengesi
      3. recommendations -> tavsiye dagiliminin 3 ayda kaymasi
    """
    parts, meta = [], {}

    # --- 1) EPS tahmin trendi
    tr = bundle.get("eps_trend")
    if isinstance(tr, pd.DataFrame) and not tr.empty:
        for period in ("0y", "+1y"):
            if period not in tr.index:
                continue
            row = tr.loc[period]
            cur = _f(row.get("current"))
            for col, w, tag in (("30daysAgo", 1.0, "30d"), ("90daysAgo", 0.6, "90d")):
                old = _f(row.get(col))
                if _ok(cur) and _ok(old) and abs(old) > 1e-6:
                    chg = (cur - old) / abs(old)
                    # +-%10'da doygunluk -> tek bir aykiri deger sonucu ezmesin
                    parts.append(w * max(-1.0, min(1.0, chg / 0.10)))
                    meta[f"eps_{period}_chg_{tag}"] = round(chg, 4)

    # --- 2) Yukari / asagi revizyon sayilari
    rev = bundle.get("eps_revisions")
    if isinstance(rev, pd.DataFrame) and not rev.empty:
        up = dn = 0.0
        for period in ("0y", "+1y", "0q", "+1q"):
            if period in rev.index:
                r = rev.loc[period]
                up += _f(r.get("upLast30days")) if _ok(_f(r.get("upLast30days"))) else 0.0
                dn += _f(r.get("downLast30days")) if _ok(_f(r.get("downLast30days"))) else 0.0
        if up + dn > 0:
            balance = (up - dn) / (up + dn)
            parts.append(1.4 * balance)
            meta.update({"revisions_up_30d": up, "revisions_down_30d": dn,
                         "revision_balance": round(balance, 3)})

    # --- 3) Tavsiye dagiliminin kaymasi (3 ay)
    rec = bundle.get("recommendations")
    if isinstance(rec, pd.DataFrame) and not rec.empty and "period" in rec.columns:
        r = rec.set_index("period")

        def mean_rating(period: str) -> float:
            if period not in r.index:
                return float("nan")
            row = r.loc[period]
            counts = [_f(row.get(c)) for c in ("strongBuy", "buy", "hold", "sell", "strongSell")]
            counts = [0.0 if not _ok(c) else c for c in counts]
            tot = sum(counts)
            if tot <= 0:
                return float("nan")
            return sum(c * (i + 1) for i, c in enumerate(counts)) / tot

        now, old = mean_rating("0m"), mean_rating("-3m")
        if _ok(now) and _ok(old):
            drift = old - now  # dusen ortalama = iyilesme (1=Strong Buy)
            parts.append(1.2 * max(-1.0, min(1.0, drift / 0.30)))
            meta.update({"rating_mean_now": round(now, 3), "rating_mean_3m_ago": round(old, 3),
                         "rating_improvement": round(drift, 3)})

    if not parts:
        return float("nan"), meta
    return float(sum(parts) / len(parts)), meta


def f_analyst_consensus(bundle: dict) -> tuple[float, dict]:
    """Investing.com'daki "Al / Guclu Al" etiketinin muadili.

    recommendationMean: 1.0=Guclu Al ... 5.0=Guclu Sat
    Az analist takibi varsa sinyal notre cekilir (guven duzeltmesi).
    """
    info = bundle.get("info") or {}
    mean = _f(info.get("recommendationMean"))
    n_an = _f(info.get("numberOfAnalystOpinions"))
    if not _ok(mean):
        return float("nan"), {}

    n = 0.0 if not _ok(n_an) else n_an
    # 5+ analist tam guven; daha azinda notre (3.0) dogru kaydir
    confidence = min(1.0, n / 5.0) if n > 0 else 0.4
    adjusted = 3.0 + (mean - 3.0) * confidence

    tgt = bundle.get("price_targets") or {}
    cur, tmean = _f(tgt.get("current")), _f(tgt.get("mean"))
    upside = _n(100 * (tmean / cur - 1)) if _ok(cur) and _ok(tmean) and cur > 0 else None

    return float(adjusted), {
        "recommendation_mean": round(mean, 3),
        "recommendation_key": info.get("recommendationKey"),
        "analyst_count": _n(n_an),
        "confidence_adjusted": round(adjusted, 3),
        "target_upside_pct": upside,
        "target_mean": _n(tmean),
    }


def f_earnings_surprise(bundle: dict) -> tuple[float, dict]:
    """Son 4 ceyregin surpriz yuzdesi, yakina daha agirlikli (PEAD)."""
    eh = bundle.get("earnings_history")
    if not isinstance(eh, pd.DataFrame) or eh.empty or "surprisePercent" not in eh.columns:
        return float("nan"), {}

    s = pd.to_numeric(eh["surprisePercent"], errors="coerce").dropna().tail(4)
    if s.empty:
        return float("nan"), {}

    w = np.linspace(1.0, 2.0, len(s))
    # tek ceyrekte %500 surpriz sonucu ezmesin -> +-%25'te kirp
    clipped = s.clip(-0.25, 0.25).to_numpy(dtype=float)
    val = float(np.average(clipped, weights=w))
    beats = int((s > 0).sum())

    return val, {
        "surprises": [round(float(x), 4) for x in s.tolist()],
        "beat_count": beats,
        "quarters": int(len(s)),
        "last_surprise_pct": round(float(s.iloc[-1]) * 100, 2),
    }


# =============================================================================
#  TEMEL (KALITE / DEGERLEME / BUYUME / SAGLIK)
# =============================================================================
def f_quality_profitability(bundle: dict) -> tuple[float, dict]:
    """Novy-Marx / QMJ tarzi karlilik bilesigi (z-skorlarin ortalamasi)."""
    info = bundle.get("info") or {}
    roe = _f(info.get("returnOnEquity"))
    roa = _f(info.get("returnOnAssets"))
    gm = _f(info.get("grossMargins"))
    om = _f(info.get("operatingMargins"))
    pm = _f(info.get("profitMargins"))

    # (deger, tipik iyi seviye) — kabaca 0-1 arasi olceklenir
    specs = [(roe, 0.20), (roa, 0.08), (gm, 0.40), (om, 0.15), (pm, 0.10)]
    vals = []
    for v, good in specs:
        if _ok(v):
            # asiri ROE (orn. negatif ozkaynak kaynakli 1.48) kirpilir
            vals.append(max(-1.0, min(2.0, v / good)))

    if not vals:
        return float("nan"), {}

    return float(np.mean(vals)), {
        "roe": _n(roe), "roa": _n(roa), "gross_margin": _n(gm),
        "operating_margin": _n(om), "profit_margin": _n(pm),
        "components_used": len(vals),
    }


def f_valuation_composite(bundle: dict) -> tuple[float, dict]:
    """UCUZLUK bilesigi — yuksek = ucuz/cazip.

    Kullanicinin "fiyat cok yuksek olmasin" kriterinin istatistiksel olarak
    dogru karsiligi budur (nominal fiyat degil).
    Her carpanin tersi alinir; boylece buyuk = ucuz olur.
    """
    info = bundle.get("info") or {}
    pe = _f(info.get("trailingPE"))
    fpe = _f(info.get("forwardPE"))
    pb = _f(info.get("priceToBook"))
    ps = _f(info.get("priceToSalesTrailing12Months"))
    ev_ebitda = _f(info.get("enterpriseToEbitda"))
    peg = _f(info.get("trailingPegRatio"))

    parts = []
    # (deger, tipik "adil" seviye, agirlik)
    for v, fair, w in ((fpe, 18.0, 1.4), (pe, 20.0, 1.0), (ev_ebitda, 12.0, 1.2),
                       (ps, 2.5, 0.8), (pb, 3.0, 0.6), (peg, 1.5, 1.0)):
        if _ok(v) and v > 0:
            parts.append((w, max(-1.0, min(2.0, fair / v - 1.0))))

    # Serbest nakit akisi getirisi — en gurbuz degerleme olcusu
    fcf = _f(info.get("freeCashflow"))
    mcap = _f(info.get("marketCap"))
    fcf_yield = None
    if _ok(fcf) and _ok(mcap) and mcap > 0:
        fcf_yield = fcf / mcap
        parts.append((1.5, max(-1.0, min(2.0, fcf_yield / 0.05))))

    if not parts:
        return float("nan"), {}

    total_w = sum(w for w, _ in parts)
    val = sum(w * v for w, v in parts) / total_w

    return float(val), {
        "trailing_pe": _n(pe), "forward_pe": _n(fpe), "price_to_book": _n(pb),
        "price_to_sales": _n(ps), "ev_to_ebitda": _n(ev_ebitda), "peg": _n(peg),
        "fcf_yield": None if fcf_yield is None else round(fcf_yield, 4),
        "metrics_used": len(parts),
    }


def f_financial_health(bundle: dict) -> tuple[float, dict]:
    """Piotroski F-Score ruhunda 0-100 saglik puani + Altman Z benzeri unsurlar."""
    info = bundle.get("info") or {}
    checks: dict[str, bool | None] = {}

    cr = _f(info.get("currentRatio"))
    checks["current_ratio_gt_1_2"] = (cr > 1.2) if _ok(cr) else None

    de = _f(info.get("debtToEquity"))          # yfinance yuzde olarak verir
    checks["debt_to_equity_lt_100"] = (de < 100) if _ok(de) else None

    fcf = _f(info.get("freeCashflow"))
    checks["positive_free_cash_flow"] = (fcf > 0) if _ok(fcf) else None

    ni = _f(info.get("netIncomeToCommon"))
    checks["positive_net_income"] = (ni > 0) if _ok(ni) else None

    om = _f(info.get("operatingMargins"))
    checks["positive_operating_margin"] = (om > 0) if _ok(om) else None

    # Tahakkuk kalitesi: nakit akisi net kardan buyuk olmali
    ocf = _f(info.get("operatingCashflow"))
    checks["ocf_exceeds_net_income"] = (ocf > ni) if (_ok(ocf) and _ok(ni)) else None

    qr = _f(info.get("quickRatio"))
    checks["quick_ratio_gt_1"] = (qr > 1.0) if _ok(qr) else None

    # Nakit borcu karsiliyor mu
    cash, debt = _f(info.get("totalCash")), _f(info.get("totalDebt"))
    checks["cash_covers_half_debt"] = (cash > 0.5 * debt) if (_ok(cash) and _ok(debt) and debt > 0) else None

    rg = _f(info.get("revenueGrowth"))
    checks["revenue_growing"] = (rg > 0) if _ok(rg) else None

    known = {k: v for k, v in checks.items() if v is not None}
    if len(known) < 3:
        return float("nan"), {"checks": checks}

    score = 100.0 * sum(1 for v in known.values() if v) / len(known)
    return score, {
        "checks": checks,
        "passed": sum(1 for v in known.values() if v),
        "evaluated": len(known),
        "debt_to_equity": _n(de),
        "current_ratio": _n(cr),
        "free_cash_flow": _n(fcf),
        # ceza kurali icin
        "negative_fcf_high_debt": bool(_ok(fcf) and fcf < 0 and _ok(de) and de > 150),
    }


def f_growth_quality(bundle: dict) -> tuple[float, dict]:
    """Buyume SEVIYESI + IVMESI (hizlanma seviyeden daha bilgilendiricidir)."""
    info = bundle.get("info") or {}
    rg = _f(info.get("revenueGrowth"))
    eg = _f(info.get("earningsGrowth"))
    eqg = _f(info.get("earningsQuarterlyGrowth"))

    parts, meta = [], {"revenue_growth": _n(rg), "earnings_growth": _n(eg),
                       "earnings_quarterly_growth": _n(eqg)}

    for v, good, w in ((rg, 0.15, 1.0), (eg, 0.20, 1.2), (eqg, 0.20, 0.8)):
        if _ok(v):
            parts.append((w, max(-1.5, min(2.0, v / good))))

    # Ileriye donuk buyume beklentisi + hizlanma
    ge = bundle.get("growth_estimates")
    if isinstance(ge, pd.DataFrame) and not ge.empty and "stockTrend" in ge.columns:
        cur_y = _f(ge.loc["0y", "stockTrend"]) if "0y" in ge.index else float("nan")
        nxt_y = _f(ge.loc["+1y", "stockTrend"]) if "+1y" in ge.index else float("nan")
        if _ok(nxt_y):
            parts.append((1.0, max(-1.5, min(2.0, nxt_y / 0.15))))
            meta["growth_est_next_year"] = round(nxt_y, 4)
        if _ok(cur_y) and _ok(nxt_y):
            meta["growth_acceleration"] = round(nxt_y - cur_y, 4)
            parts.append((0.8, max(-1.0, min(1.0, (nxt_y - cur_y) / 0.10))))

    if not parts:
        return float("nan"), meta
    tw = sum(w for w, _ in parts)
    return float(sum(w * v for w, v in parts) / tw), meta


# =============================================================================
#  RISK / LIKIDITE / SAHIPLIK
# =============================================================================
def f_risk_drawdown(df: pd.DataFrame, bundle: dict) -> tuple[float, dict]:
    """Dusuk risk -> YUKSEK puan (yon cevrilmis)."""
    close = df["Close"]
    vol = ta.annualized_vol(close)
    mdd = ta.max_drawdown(close)
    beta = _f((bundle.get("info") or {}).get("beta"))

    parts = []
    if _ok(vol):
        # %25 yillik oynaklik notr; %60+ cok riskli
        parts.append(max(-1.5, min(1.5, (0.25 - vol) / 0.20)))
    if _ok(mdd):
        # -%25 notr; -%60 cok kotu  (mdd negatif)
        parts.append(max(-1.5, min(1.5, (mdd + 0.25) / 0.20)))
    if _ok(beta):
        parts.append(max(-1.5, min(1.5, (1.10 - beta) / 0.45)))

    if not parts:
        return float("nan"), {}

    return float(np.mean(parts)), {
        "annualized_volatility": _n(vol),
        "max_drawdown_1y": _n(mdd),
        "beta": _n(beta),
    }


def f_liquidity(df: pd.DataFrame, bundle: dict) -> tuple[float, dict]:
    """Islem gorebilirlik — BUYUKLUKTEN ARINDIRILMIS.

    DENETIM DUZELTMESI (Y1): eskiden ham dolar hacmi + piyasa degeri
    karisimiydi ve bu yuzden `size_opportunity` ile r=0.65 korelasyonluydu —
    kucuk sirketler hem "kucuk oldugu icin iyi" hem "hacmi dusuk oldugu icin
    kotu" puan aliyor, ayni bilgi iki kez sayiliyordu.

    Simdi asil olculen DEVIR HIZI (turnover): gunluk dolar hacmi / piyasa
    degeri. Bu oran sirket buyuklugunden bagimsizdir — 500M'lik bir sirket de
    50Mr'lik bir sirket de yuksek devir hizina sahip olabilir.

    Ayrica mutlak bir islenebilirlik tabani korunur: gunde 250 bin dolarin
    altinda islem goren hisse, devir hizi ne olursa olsun alinip satilamaz.
    """
    info = bundle.get("info") or {}
    close = df["Close"].dropna()
    if "Volume" not in df or close.empty:
        return float("nan"), {}

    px = float(close.iloc[-1])
    avg_vol = float(df["Volume"].tail(30).mean())
    dollar_vol = px * avg_vol if avg_vol > 0 else float("nan")
    mcap = _f(info.get("marketCap"))

    if not _ok(dollar_vol) or dollar_vol <= 0:
        return float("nan"), {}

    parts = []
    turnover = None
    if _ok(mcap) and mcap > 0:
        turnover = dollar_vol / mcap            # gunluk devir hizi
        # %0.5 tipik, %2+ cok canli
        parts.append((1.5, max(-1.5, min(1.5, (turnover - 0.005) / 0.006))))

    # islenebilirlik tabani: 250K altinda sert ceza, 5M ustunde ek fayda yok
    floor_score = (math.log10(dollar_vol) - math.log10(250_000)) / \
                  (math.log10(5_000_000) - math.log10(250_000))
    parts.append((1.0, max(-1.5, min(1.0, floor_score))))

    val = sum(w * v for w, v in parts) / sum(w for w, _ in parts)

    return float(val), {
        "avg_dollar_volume_30d": _n(dollar_vol),
        "avg_volume_30d": _n(avg_vol),
        "market_cap": _n(mcap),
        "turnover_daily": None if turnover is None else round(turnover, 5),
        # Pozisyon boyutu icin: gunluk hacmin %5'i, piyasayi bozmadan
        # girilebilecek kaba ust sinirdir (denetim bulgusu O1).
        "max_position_usd": _n(dollar_vol * 0.05),
    }


def f_short_squeeze(bundle: dict) -> tuple[float, dict]:
    """Yuksek short float + pozitif momentum = squeeze yakiti."""
    info = bundle.get("info") or {}
    spf = _f(info.get("shortPercentOfFloat"))
    sr = _f(info.get("shortRatio"))  # days-to-cover

    if not _ok(spf) and not _ok(sr):
        return float("nan"), {}

    parts = []
    if _ok(spf):
        parts.append(min(1.5, spf / 0.10))     # %10 float short = referans
    if _ok(sr):
        parts.append(min(1.5, sr / 5.0))       # 5 gun = referans

    return float(np.mean(parts)), {
        "short_percent_of_float": _n(spf),
        "short_ratio_days": _n(sr),
    }


def f_institutional_ownership(bundle: dict) -> tuple[float, dict]:
    """Ters-U: kurumsal destek iyi, ama %95+ doygunluk yukari alani kisitlar."""
    info = bundle.get("info") or {}
    inst = _f(info.get("heldPercentInstitutions"))
    ins = _f(info.get("heldPercentInsiders"))

    if not _ok(inst) and not _ok(ins):
        return float("nan"), {}

    val = 0.0
    if _ok(inst):
        # 0.55-0.80 arasi tatli nokta
        val += 1.0 - abs(min(inst, 1.0) - 0.675) / 0.675
    if _ok(ins):
        val += 0.5 * min(1.0, ins / 0.10)   # iceriden sahiplik cikar birligi

    return float(val), {
        "held_percent_institutions": _n(inst),
        "held_percent_insiders": _n(ins),
    }


def f_nominal_price_fit(df: pd.DataFrame) -> tuple[float, dict]:
    close = df["Close"].dropna()
    if close.empty:
        return float("nan"), {}
    return float(close.iloc[-1]), {"price": round(float(close.iloc[-1]), 4)}


# =============================================================================
#  ANA GIRIS NOKTASI
# =============================================================================
def compute_all(ticker: str, bundle: dict, bench_close: pd.Series | None,
                wsb_record: dict | None) -> dict[str, Any]:
    """Bir hisse icin tum ham faktorleri + meta veriyi uretir."""
    df = bundle.get("history")
    info = bundle.get("info") or {}

    if not isinstance(df, pd.DataFrame) or len(df) < 30 or "Close" not in df:
        return {"ticker": ticker, "ok": False, "reason": "yetersiz fiyat gecmisi"}

    df = df.dropna(subset=["Close"])

    # DENETIM DUZELTMESI (O5): piyasa acikken calistirildiginda gunun YARIM bari
    # gelir; RSI/ATR/MA bundan etkilenir ve ayni gun iki kez calistirmak farkli
    # siralama uretir. Bugune ait tamamlanmamis bar atilir.
    partial_bar_dropped = False
    try:
        last_ts = pd.Timestamp(df.index[-1])
        now = pd.Timestamp.now(tz=last_ts.tz) if last_ts.tz else pd.Timestamp.now()
        if last_ts.date() == now.date() and len(df) > 30:
            df = df.iloc[:-1]
            partial_bar_dropped = True
    except Exception:
        pass
    close = df["Close"]
    raw: dict[str, float | None] = {}
    meta: dict[str, Any] = {}

    def put(fid: str, result: tuple[float, dict]) -> None:
        val, m = result
        raw[fid] = _n(val)
        if m:
            meta[fid] = m

    put("price_momentum_12_1", f_price_momentum_12_1(close))
    put("relative_strength", f_relative_strength(close, bench_close))
    put("trend_structure", f_trend_structure(df))
    put("breakout_setup", f_breakout_setup(df))
    put("stage2_breakout", f_stage2_breakout(df))
    put("chart_position", f_chart_position(df, bundle))
    put("momentum_persistence", f_momentum_persistence(df))
    put("analyst_upside", f_analyst_upside(bundle))
    put("volume_accumulation", f_volume_accumulation(df))
    put("size_opportunity", f_size_opportunity(bundle))
    put("revenue_scaling", f_revenue_scaling(bundle))
    put("rule_of_40", f_rule_of_40(bundle))
    put("undiscovered", f_undiscovered(bundle))
    put("cash_runway", f_cash_runway(bundle))
    put("eps_revision_momentum", f_eps_revision_momentum(bundle))
    put("analyst_consensus", f_analyst_consensus(bundle))
    put("earnings_surprise", f_earnings_surprise(bundle))
    put("quality_profitability", f_quality_profitability(bundle))
    put("valuation_composite", f_valuation_composite(bundle))
    put("financial_health", f_financial_health(bundle))
    put("growth_quality", f_growth_quality(bundle))
    put("risk_drawdown", f_risk_drawdown(df, bundle))
    put("liquidity", f_liquidity(df, bundle))
    put("short_squeeze", f_short_squeeze(bundle))
    put("institutional_ownership", f_institutional_ownership(bundle))
    put("nominal_price_fit", f_nominal_price_fit(df))

    # --- Investing.com tarzi teknik ozet
    tech = investing_summary.compute(df)
    if tech.get("available"):
        raw["technical_oscillators"] = tech["score"]
        meta["technical_oscillators"] = tech
    else:
        raw["technical_oscillators"] = None

    # --- Reddit WSB
    wsb = reddit_wsb.score_ticker(wsb_record)
    raw["reddit_wsb_attention"] = wsb["score"] if wsb.get("available") else None
    if wsb.get("available"):
        meta["reddit_wsb_attention"] = wsb

    # --- Ceza bayraklari
    price = float(close.iloc[-1])
    rsi14 = _f(ta.rsi(close, 14).iloc[-1])
    ma50 = _f(ta.sma(close, 50).iloc[-1]) if len(close) >= 50 else float("nan")
    ext_pct = (100 * (price / ma50 - 1)) if _ok(ma50) and ma50 > 0 else float("nan")

    days_to_earnings = None
    cal = bundle.get("calendar") or {}
    ed = cal.get("Earnings Date")
    if isinstance(ed, (list, tuple)) and ed:
        ed = ed[0]
    if ed is not None:
        try:
            dt = pd.Timestamp(ed)
            if dt.tzinfo is None:
                dt = dt.tz_localize("UTC")
            days_to_earnings = int((dt - pd.Timestamp.now(tz=timezone.utc)).days)
        except Exception:
            days_to_earnings = None

    fh_meta = meta.get("financial_health", {})
    va_meta = meta.get("volume_accumulation", {})

    penalties = {
        "parabolic_extension": bool(_ok(ext_pct) and ext_pct > 40),
        "hype_reversal_risk": bool(_ok(rsi14) and rsi14 > 75 and wsb.get("is_top10")),
        "earnings_imminent": bool(days_to_earnings is not None and 0 <= days_to_earnings <= 7),
        "negative_fcf_high_debt": bool(fh_meta.get("negative_fcf_high_debt")),
        "distribution_pattern": bool(va_meta.get("distribution_divergence")),
    }

    return {
        "ticker": ticker,
        "ok": True,
        "name": info.get("shortName") or info.get("longName") or ticker,
        "sector": info.get("sector") or "Bilinmiyor",
        "industry": info.get("industry") or "Bilinmiyor",
        "price": round(price, 4),
        "currency": info.get("currency", "USD"),
        "market_cap": _n(_f(info.get("marketCap"))),
        "avg_dollar_volume": meta.get("liquidity", {}).get("avg_dollar_volume_30d"),
        "max_position_usd": meta.get("liquidity", {}).get("max_position_usd"),
        "turnover_daily": meta.get("liquidity", {}).get("turnover_daily"),
        "partial_bar_dropped": partial_bar_dropped,
        "rsi14": _n(rsi14),
        "pct_extended_from_ma50": _n(ext_pct),
        "days_to_earnings": days_to_earnings,
        "returns": {
            "1m": _n(_ret(close, 21)), "3m": _n(_ret(close, 63)),
            "6m": _n(_ret(close, 126)), "12m": _n(_ret(close, 252)),
        },
        "raw": raw,
        "meta": meta,
        "penalty_flags": penalties,
        "snapshot_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
    }
