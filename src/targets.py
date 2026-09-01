"""Fiyat hedefleri, stop seviyeleri ve satis sinyalleri.

Bu modul, izleme listesindeki hisseler icin uc soruyu cevaplar:
  1. NEREYE kadar gidebilir?   -> kisa vadeli (teknik) + uzun vadeli (temel) hedef
  2. NEREDE yanlisim?          -> stop seviyesi (baslangic + takip eden)
  3. NE ZAMAN cikmaliyim?      -> satis sinyalleri ve risk seviyesi

Tasarim ilkesi: her sayinin bir GEREKCESI vardir. Hicbir hedef "tahmin" degil;
her biri gozlemlenebilir bir seviyeden (direnc, ATR, analist hedefi, degerleme
carpani) turetilir ve hangi yontemle bulundugu ciktida belirtilir.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from . import indicators as ta

# --- Risk seviyeleri (dusukten yuksege) -------------------------------------
RISK_LEVELS = ["GUVENLI", "IZLE", "DIKKAT", "YUKSEK_RISK", "SAT"]

RISK_TR = {
    "GUVENLI":     "Guvenli — trend saglam",
    "IZLE":        "Izle — kucuk zayiflama isaretleri",
    "DIKKAT":      "Dikkat — trend bozulmaya basladi",
    "YUKSEK_RISK": "Yuksek risk — cikis planla",
    "SAT":         "SAT — cikis sarti olustu",
}


def _f(x: Any) -> float:
    try:
        if x is None:
            return float("nan")
        v = float(x)
        return v if math.isfinite(v) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def _ok(x: float) -> bool:
    return isinstance(x, float) and math.isfinite(x)


def _n(x) -> float | None:
    v = _f(x)
    return None if not _ok(v) else round(float(v), 4)


# =============================================================================
#  Destek / direnc seviyeleri
# =============================================================================
def swing_levels(df: pd.DataFrame, window: int = 10, lookback: int = 252
                 ) -> tuple[list[float], list[float]]:
    """Yerel tepe ve dip noktalari (swing high/low).

    Bir bar, her iki yaninda `window` bar boyunca en yuksekse tepe sayilir.
    Bu noktalar piyasanin gercekten islem gordugu seviyelerdir — bu yuzden
    direnc/destek olarak uydurma cizgilerden daha anlamlidir.
    """
    d = df.tail(lookback)
    if len(d) < 3 * window:
        return [], []

    high, low = d["High"].to_numpy(float), d["Low"].to_numpy(float)
    highs, lows = [], []
    for i in range(window, len(d) - window):
        seg_h = high[i - window: i + window + 1]
        seg_l = low[i - window: i + window + 1]
        if high[i] == seg_h.max():
            highs.append(float(high[i]))
        if low[i] == seg_l.min():
            lows.append(float(low[i]))
    return highs, lows


def _cluster(levels: list[float], tol: float = 0.02) -> list[float]:
    """Birbirine %2'den yakin seviyeleri tek seviyede birlestirir."""
    if not levels:
        return []
    out: list[float] = []
    for lv in sorted(levels):
        if out and abs(lv - out[-1]) / max(out[-1], 1e-9) <= tol:
            out[-1] = (out[-1] + lv) / 2
        else:
            out.append(lv)
    return out


# =============================================================================
#  Hedefler
# =============================================================================
def _dispersion(values: list[float], target: float) -> dict[str, Any]:
    """Yontemler arasi dagilim = belirsizligin dogrudan olcusu (bulgu O2).

    Hedef tek bir sayi olarak sunuldugunda sahte kesinlik uretiyordu: uc ayri
    yontem 53 ve 63 dedigi halde cikti "58" oluyor ve aradaki %19'luk
    anlasmazlik kayboluyordu. Yontemler birbirine yakinsa hedef guvenilir;
    uzaksa hedefin kendisi degil, ARALIK bilgidir.
    """
    vals = [v for v in values if _ok(v) and v > 0]
    if not vals or not target:
        return {}
    lo, hi = min(vals), max(vals)
    spread = 100.0 * (hi - lo) / target

    if len(vals) < 2:
        conf, conf_tr = "dogrulanmadi", "Tek yontem - baska bir yontemle dogrulanmadi"
    elif spread < 10:
        conf, conf_tr = "yuksek", "Yontemler birbirini dogruluyor"
    elif spread < 25:
        conf, conf_tr = "orta", "Yontemler arasinda kayda deger fark var"
    else:
        conf, conf_tr = "dusuk", "Yontemler ciddi bicimde ayrisiyor - hedef degil ARALIK okunmali"

    return {
        "range_low": round(lo, 4),
        "range_high": round(hi, 4),
        "spread_pct": round(spread, 1),
        "methods_n": len(vals),
        "confidence": conf,
        "confidence_tr": conf_tr,
    }


# =============================================================================
#  Islem maliyeti (bulgu D3)
# =============================================================================
# Evren kucuk sirket agirlikli ve filtre esigi gunluk yalnizca 1M USD. Boyle
# bir hissede alis-satis makasi %1-3 olabilir. "%8 hedef" brut bir sayidir;
# makasin %3'u odenirse gercek hedef %5'tir. Butun hedef/stop hesaplari bugune
# kadar brutu gosteriyordu.
#
# Makas dogrudan olculemiyor (Yahoo ucu bid/ask gecmisi vermiyor), bu yuzden
# LIKIDITEDEN tahmin ediliyor. Model kaba ama yonu dogru: hacim dustukce makas
# buyur, fiyat dustukce kurus adimi orani buyur.
COMMISSION_PCT = 0.0          # ABD'de cogu aracida sifir
MIN_SPREAD_PCT = 0.05
MAX_SPREAD_PCT = 3.00


def estimate_costs(price: float, avg_dollar_volume: float | None,
                   position_usd: float | None = None) -> dict[str, Any]:
    """Gidis-donus islem maliyeti tahmini (yuzde)."""
    if not price or price <= 0:
        return {"available": False}

    # 1) Kurus adimi tabani: 5 dolarlik hissede 1 sentlik adim %0.2'dir.
    tick_pct = 100.0 * 0.01 / price

    # 2) Likidite bileseni: gunluk hacim milyon dolar cinsinden.
    dv_m = (float(avg_dollar_volume) / 1e6) if avg_dollar_volume else 0.5
    dv_m = max(0.05, dv_m)
    liq_pct = 0.55 / (dv_m ** 0.5)

    spread_pct = min(MAX_SPREAD_PCT, max(MIN_SPREAD_PCT, tick_pct + liq_pct))

    # 3) Piyasa etkisi: pozisyon gunluk hacmin ne kadari? Verilmezse "gunluk
    #    hacmin %5'i" varsayilir (panoda gosterilen girilebilir azami pozisyon).
    adv = float(avg_dollar_volume) if avg_dollar_volume else 0.0
    part = (float(position_usd) / adv) if (position_usd and adv > 0) else 0.05
    impact_pct = 10.0 * (max(0.0, part) ** 0.5) * (spread_pct / 100.0) * 100.0
    impact_pct = min(2.0, impact_pct)

    one_way = spread_pct / 2.0 + impact_pct + COMMISSION_PCT
    return {
        "available": True,
        "spread_pct": round(spread_pct, 3),
        "impact_pct": round(impact_pct, 3),
        "commission_pct": COMMISSION_PCT,
        "one_way_pct": round(one_way, 3),
        "round_trip_pct": round(2 * one_way, 3),
        "assumed_participation": round(part, 4),
        "note_tr": ("Makas dogrudan olculemedigi icin likiditeden tahmin "
                    "edildi; gercek maliyet farkli olabilir."),
    }


def short_term_target(df: pd.DataFrame, price: float) -> dict[str, Any]:
    """KISA VADE (yaklasik 1-3 ay) — teknik hedef.

    Uc bagimsiz yontem hesaplanir ve en makul olani secilir:
      A) Bir ustteki direnc kumesi        (piyasanin hafizasi)
      B) ATR projeksiyonu: fiyat + 3xATR  (tipik dalga boyu)
      C) Olculu hareket: son bazin yuksekligi kirilim noktasina eklenir
    """
    atr14 = _f(ta.atr(df["High"], df["Low"], df["Close"], 14).dropna().iloc[-1]) \
        if len(df) > 20 else float("nan")

    methods: list[tuple[str, float, str]] = []

    # --- A) ustteki direnc
    highs, _ = swing_levels(df)
    res = [h for h in _cluster(highs) if h > price * 1.01]
    hi52 = float(df["Close"].tail(252).max())
    if hi52 > price * 1.01:
        res.append(hi52)
    res = sorted(set(round(r, 4) for r in res))
    if res:
        methods.append(("direnc", res[0], "Bir ustteki direnc kumesi"))

    # --- B) ATR projeksiyonu
    if _ok(atr14) and atr14 > 0:
        methods.append(("atr", price + 3.0 * atr14, "Fiyat + 3xATR (tipik dalga boyu)"))

    # --- C) olculu hareket (son 6 ayin baz yuksekligi)
    win = df["Close"].tail(126)
    if len(win) > 60:
        base_h = float(win.max() - win.min())
        if base_h > 0:
            methods.append(("olculu_hareket", price + base_h * 0.6,
                            "Baz yuksekliginin %60'i kadar olculu hareket"))

    if not methods:
        return {"available": False}

    # Anlamli hedef esigi: fiyata cok yakin bir direnc "hedef" degildir.
    # En az %4 veya 1xATR uzaklik ariyoruz (hangisi buyukse).
    min_gap = max(price * 0.04, atr14 if _ok(atr14) else 0.0)
    meaningful = [r for r in res if r >= price + min_gap]

    vals = [v for _, v, _ in methods]
    # Ortanca: tek bir asiri yontem hedefi bozmasin
    target = float(np.median(vals))

    # Piyasa gercek direnclerde duraklar: ortancaya yakin ANLAMLI bir direnc
    # varsa ona hizala. Yakinligi tetikleyen onemsiz direncler elendi.
    chosen = min(methods, key=lambda m: abs(m[1] - target))[0]
    if meaningful and abs(meaningful[0] - target) / target < 0.15:
        target, chosen = meaningful[0], "direnc"

    # Taban: hedef en az 1.5xATR uzakta olmali, yoksa islem edilebilir degil
    if _ok(atr14) and atr14 > 0:
        floor = price + 1.5 * atr14
        if target < floor:
            target, chosen = floor, "atr_taban"

    return {
        "available": True,
        "target": round(target, 4),
        "upside_pct": round((target / price - 1) * 100, 2),
        "method": chosen,
        "horizon_tr": "1-3 ay",
        "candidates": [{"yontem": k, "hedef": round(v, 4), "aciklama": d}
                       for k, v, d in methods],
        "next_resistances": [round(r, 4) for r in res[:3]],
        **_dispersion(vals, target),
    }


def long_term_target(df: pd.DataFrame, bundle: dict, price: float) -> dict[str, Any]:
    """UZUN VADE (yaklasik 12 ay) — temel hedef.

    Iki yontem harmanlanir:
      A) Analist ortalama hedef fiyati
      B) Degerleme: ileri EPS x buyumeye gore makul F/K (PEG ~1.5 mantigi)
    Ikisi de varsa agirlikli ortalama alinir; biri varsa o kullanilir.
    """
    info = bundle.get("info") or {}
    tgt = bundle.get("price_targets") or {}
    methods: list[tuple[str, float, float, str]] = []   # (ad, deger, agirlik, aciklama)

    # --- A) analist hedefi
    mean = _f(tgt.get("mean"))
    n_an = _f(info.get("numberOfAnalystOpinions"))
    if _ok(mean) and mean > 0:
        # Az analist -> daha dusuk guven
        w = 1.6 if (_ok(n_an) and n_an >= 5) else 0.9
        methods.append(("analist", mean, w,
                        f"{int(n_an) if _ok(n_an) else '?'} analistin ortalama hedefi"))

    # --- B) degerleme tabanli
    fwd_eps = _f(info.get("forwardEps"))
    growth = _f(info.get("earningsGrowth"))
    if not _ok(growth):
        growth = _f(info.get("revenueGrowth"))

    if _ok(fwd_eps) and fwd_eps > 0:
        if _ok(growth) and growth > 0:
            growth_pe = max(10.0, min(35.0, growth * 100 * 1.5))   # PEG ~1.5
        else:
            growth_pe = 15.0

        # Hissenin KENDI carpanina yari yolda bulusturuyoruz. 12 ayda F/K 7'den
        # 15'e tam yeniden fiyatlama bir taban senaryo degildir; carpan
        # genislemesinin yarisini varsaymak cok daha savunulabilir.
        cur_pe = _f(info.get("forwardPE"))
        if not _ok(cur_pe) or cur_pe <= 0:
            cur_pe = _f(info.get("trailingPE"))
        if _ok(cur_pe) and cur_pe > 0:
            fair_pe = 0.5 * growth_pe + 0.5 * cur_pe
            why = (f"Ileri EPS {fwd_eps:.2f} x F/K {fair_pe:.1f} "
                   f"(mevcut {cur_pe:.1f} ile makul {growth_pe:.0f} arasi)")
        else:
            fair_pe = growth_pe
            why = f"Ileri EPS {fwd_eps:.2f} x makul F/K {fair_pe:.0f} (PEG 1.5)"

        val = fwd_eps * fair_pe
        # 12 aylik taban senaryoyu makul bantta tut
        val = max(price * 0.70, min(price * 1.80, val))
        # Analist hedefi varsa degerleme ikincil kalir (daha gurultulu bir tahmin)
        w = 0.7 if any(k == "analist" for k, _, _, _ in methods) else 1.2
        methods.append(("degerleme", val, w, why))

    if not methods:
        return {"available": False}

    tw = sum(w for _, _, w, _ in methods)
    target = sum(v * w for _, v, w, _ in methods) / tw

    return {
        "available": True,
        "target": round(target, 4),
        "upside_pct": round((target / price - 1) * 100, 2),
        "horizon_tr": "12 ay",
        "analyst_high": _n(tgt.get("high")),
        "analyst_low": _n(tgt.get("low")),
        "candidates": [{"yontem": k, "hedef": round(v, 4), "agirlik": w, "aciklama": d}
                       for k, v, w, d in methods],
        # Analist bandi da bir yontem gibi degerlendirilir: hedefin aralik
        # olarak sunulmasi icin en genis bilgi kaynagi odur.
        **_dispersion([v for _, v, _, _ in methods]
                      + [x for x in (_f(tgt.get("low")), _f(tgt.get("high")))
                         if _ok(x) and x > 0], target),
    }


# =============================================================================
#  Stop seviyeleri
# =============================================================================
def stop_levels(df: pd.DataFrame, price: float, entry_price: float | None
                ) -> dict[str, Any]:
    """Baslangic ve takip eden (trailing) stop seviyeleri.

    * Baslangic stop : giristeki teknik kirilma noktasi. Uc aday icinden
                       fiyata EN YAKIN olani secilir (en siki koruma).
    * Chandelier stop: son 22 gunun zirvesi - 3xATR. Fiyat yukseldikce yukselir,
                       asla asagi inmez -> kari korur.
    * Kullanilan stop: ikisinin YUKSEGI (pozisyon ilerledikce koruma siki
                       tutulur; klasik "stop'u yukari cek" davranisi).
    """
    high, low, close = df["High"], df["Low"], df["Close"]
    atr14 = _f(ta.atr(high, low, close, 14).dropna().iloc[-1]) if len(df) > 20 else float("nan")
    if not _ok(atr14) or atr14 <= 0:
        return {"available": False}

    base = entry_price if (entry_price and entry_price > 0) else price
    cands: list[tuple[str, float, str]] = []

    # --- 2xATR altinda
    cands.append(("atr", base - 2.0 * atr14, "Giris - 2xATR (normal dalgalanma disi)"))

    # --- son swing dip
    _, lows = swing_levels(df, window=8, lookback=126)
    below = [lv for lv in lows if lv < price]
    if below:
        sl = max(below)
        cands.append(("swing_dip", sl - 0.4 * atr14, "Son onemli dip - 0.4xATR"))

    # --- MA50 altinda
    if len(close) >= 50:
        ma50 = _f(ta.sma(close, 50).iloc[-1])
        if _ok(ma50) and ma50 < price:
            cands.append(("ma50", ma50 * 0.97, "MA50'nin %3 altinda"))

    valid = [(k, v, d) for k, v, d in cands if v < price]
    if not valid:
        valid = [("atr", price - 2.0 * atr14, "Giris - 2xATR")]

    # fiyata en yakin = en siki koruma
    init_name, init_stop, init_why = max(valid, key=lambda c: c[1])

    # --- chandelier (takip eden)
    hh22 = float(close.tail(22).max())
    chandelier = hh22 - 3.0 * atr14

    active = max(init_stop, chandelier) if chandelier < price else init_stop
    active_kind = "chandelier" if active == chandelier else init_name

    risk_pct = (base / active - 1) * 100 if active > 0 else float("nan")

    return {
        "available": True,
        "initial_stop": round(init_stop, 4),
        "initial_method": init_name,
        "initial_why": init_why,
        "chandelier_stop": round(chandelier, 4),
        "active_stop": round(active, 4),
        "active_method": active_kind,
        "distance_pct": round((price / active - 1) * 100, 2),
        "risk_per_share_pct": _n(risk_pct),
        "atr14": round(atr14, 4),
        "atr_pct": round(100 * atr14 / price, 2),
        "candidates": [{"yontem": k, "seviye": round(v, 4), "aciklama": d}
                       for k, v, d in valid],
    }


# =============================================================================
#  Satis sinyalleri ve risk seviyesi
# =============================================================================
def sell_signals(df: pd.DataFrame, bundle: dict, price: float,
                 stops: dict, st_target: dict, lt_target: dict,
                 entry_price: float | None,
                 score_now: float | None, score_at_entry: float | None
                 ) -> dict[str, Any]:
    """Somut, kural tabanli satis ve risk sinyalleri.

    Her sinyalin bir AGIRLIGI vardir; toplam agirlik risk seviyesini belirler.
    Boylece "his" degil, sayilabilir gerekceler uzerinden karar verilir.
    """
    close, high, low = df["Close"], df["High"], df["Low"]
    signals: list[dict] = []

    def add(code: str, sev: int, title: str, detail: str) -> None:
        signals.append({"kod": code, "siddet": sev, "baslik": title, "aciklama": detail})

    # ---------------------------------------------------------------- SAT (5)
    active_stop = stops.get("active_stop")
    if active_stop and price <= active_stop:
        add("STOP_VURDU", 5, "Stop seviyesi kirildi",
            f"Fiyat {price:.2f}, stop {active_stop:.2f}. Cikis sarti olustu — "
            f"tez yanlislandi, pozisyon kapatilmali.")

    ma150 = _f(ta.sma(close, 150).iloc[-1]) if len(close) >= 150 else float("nan")
    ma200 = _f(ta.sma(close, 200).iloc[-1]) if len(close) >= 200 else float("nan")

    if _ok(ma200) and price < ma200:
        slope200 = ta.slope_pct(ta.sma(close, 200), 25)
        if _ok(slope200) and slope200 < 0:
            add("ASAMA4", 5, "Asama 4 — dusus trendi",
                f"Fiyat MA200'un ({ma200:.2f}) altinda ve MA200 asagi egimli. "
                f"Weinstein'a gore dusus asamasi; tutmanin teknik gerekcesi kalmadi.")

    # ------------------------------------------------------------ YUKSEK (4)
    if _ok(ma150) and price < ma150:
        add("MA150_ALTI", 4, "MA150 altina sarkti",
            f"Fiyat {price:.2f} < MA150 {ma150:.2f}. Orta vadeli trend bozuldu; "
            f"Asama 2'den cikis sinyali.")

    if st_target.get("available") and price >= st_target["target"]:
        add("KISA_HEDEF", 4, "Kisa vadeli hedefe ulasildi",
            f"Hedef {st_target['target']:.2f} asildi. Kismi kar realizasyonu ve "
            f"stop'u girise cekmek degerlendirilmeli.")

    # dagitim: fiyat yukari, OBV asagi
    if "Volume" in df and len(df) > 60:
        obv_slope = ta.slope_pct(ta.obv(close, df["Volume"]), 40)
        px_slope = ta.slope_pct(close, 40)
        if _ok(obv_slope) and _ok(px_slope) and px_slope > 0.05 and obv_slope < -0.05:
            add("DAGITIM", 4, "Dagitim — negatif hacim uyumsuzlugu",
                "Fiyat yukselirken OBV dusuyor: yukselis hacimle desteklenmiyor, "
                "kurumsal cikis olabilir.")

    # skor bozulmasi
    if score_now is not None and score_at_entry is not None:
        drop = score_at_entry - score_now
        if drop >= 12:
            add("SKOR_DUSTU", 4, "Toplam puan belirgin dustu",
                f"Giriste {score_at_entry:.1f} -> simdi {score_now:.1f} ({drop:.1f} puan). "
                f"Alis gerekceleri zayifladi.")
        elif drop >= 6:
            add("SKOR_ZAYIF", 2, "Toplam puan zayifladi",
                f"Giriste {score_at_entry:.1f} -> simdi {score_now:.1f} ({drop:.1f} puan).")

    # ------------------------------------------------------------- DIKKAT (3)
    ma50 = _f(ta.sma(close, 50).iloc[-1]) if len(close) >= 50 else float("nan")
    if _ok(ma50) and price < ma50:
        add("MA50_ALTI", 3, "MA50 altina sarkti",
            f"Fiyat {price:.2f} < MA50 {ma50:.2f}. Kisa vadeli trend zayifladi.")

    if _ok(ma50) and ma50 > 0:
        ext = 100 * (price / ma50 - 1)
        if ext > 40:
            add("PARABOLIK", 3, "Parabolik uzama",
                f"Fiyat MA50'nin %{ext:.0f} uzerinde. Ortalamaya donus riski yuksek; "
                f"kismi kar almak icin uygun bolge.")

    rsi14 = _f(ta.rsi(close, 14).iloc[-1])
    if _ok(rsi14) and rsi14 > 78:
        add("ASIRI_ALIM", 2, "Asiri alim bolgesi",
            f"RSI(14) = {rsi14:.0f}. Kisa vadeli geri cekilme olasiligi artti.")

    # ATR genislemesi = artan oynaklik
    atr_now = ta.atr(high, low, close, 14).dropna()
    if len(atr_now) > 90:
        a_short = float(atr_now.tail(10).mean())
        a_long = float(atr_now.tail(90).mean())
        if a_long > 0 and a_short / a_long > 1.5:
            add("OYNAKLIK", 2, "Oynaklik sicramasi",
                f"Son 10 gun ATR, 3 aylik ortalamanin %{100*(a_short/a_long-1):.0f} "
                f"uzerinde. Belirsizlik artti, pozisyon boyutu gozden gecirilmeli.")

    # MACD negatife dondu
    line, sig, _ = ta.macd(close)
    lv, sv = _f(line.iloc[-1]), _f(sig.iloc[-1])
    if _ok(lv) and _ok(sv) and lv < sv:
        prev_l, prev_s = _f(line.iloc[-6]), _f(sig.iloc[-6])
        if _ok(prev_l) and _ok(prev_s) and prev_l > prev_s:
            add("MACD_KESISIM", 2, "MACD asagi kesti",
                "MACD sinyal cizgisinin altina indi — momentum yon degistiriyor.")

    # bilanco riski
    cal = bundle.get("calendar") or {}
    ed = cal.get("Earnings Date")
    if isinstance(ed, (list, tuple)) and ed:
        ed = ed[0]
    if ed is not None:
        try:
            dt = pd.Timestamp(ed)
            if dt.tzinfo is None:
                dt = dt.tz_localize("UTC")
            days = int((dt - pd.Timestamp.now(tz="UTC")).days)
            if 0 <= days <= 7:
                add("BILANCO", 2, f"{days} gun icinde bilanco",
                    "Ongorulemez olay riski. Pozisyon boyutunu kucultmek veya "
                    "bilanco sonrasina beklemek degerlendirilmeli.")
        except Exception:
            pass

    # uzun vadeli hedef kisa vadelinin altinda = sinirli yapisal potansiyel
    if (st_target.get("available") and lt_target.get("available")
            and lt_target["target"] < st_target["target"]):
        add("HEDEF_TERS", 2, "Uzun vadeli hedef kisa vadelinin altinda",
            f"12 aylik hedef {lt_target['target']:.2f} < 1-3 aylik hedef "
            f"{st_target['target']:.2f}. Teknik olarak yukari alan var ama temel "
            f"olarak sinirli — bu bir ALIM-TUT degil, kisa vadeli islem adayi.")

    # stop'a yakinlik
    if active_stop and price > active_stop:
        dist = 100 * (price / active_stop - 1)
        if dist < 4:
            add("STOP_YAKIN", 3, "Stop seviyesine cok yakin",
                f"Fiyat stop'un sadece %{dist:.1f} uzerinde. Kucuk bir dususte cikis gelir.")

    # ---------------------------------------------------------------- sonuc
    signals.sort(key=lambda s: -s["siddet"])
    max_sev = max((s["siddet"] for s in signals), default=0)
    total = sum(s["siddet"] for s in signals)

    if max_sev >= 5:
        level = "SAT"
    elif max_sev >= 4 or total >= 9:
        level = "YUKSEK_RISK"
    elif max_sev >= 3 or total >= 5:
        level = "DIKKAT"
    elif signals:
        level = "IZLE"
    else:
        level = "GUVENLI"

    # eylem onerisi
    if level == "SAT":
        action = "Cikis sarti olustu — pozisyonu kapat."
    elif level == "YUKSEK_RISK":
        action = "Kismi kar al ve stop'u yukari cek; yeni alim yapma."
    elif level == "DIKKAT":
        action = "Yakin takip; stop'u sikilastir, pozisyon buyutme."
    elif level == "IZLE":
        action = "Trend saglam, kucuk uyarilar var. Plana sadik kal."
    else:
        action = "Trend saglam, uyari yok. Stop'u takip et."

    return {
        "risk_level": level,
        "risk_level_tr": RISK_TR[level],
        "risk_index": RISK_LEVELS.index(level),
        "signals": signals,
        "signal_count": len(signals),
        "severity_total": total,
        "action_tr": action,
    }


# =============================================================================
#  Tek giris noktasi
# =============================================================================
def earnings_countdown(bundle: dict) -> dict[str, Any]:
    """Bilancoya kalan gun (bulgu 3.4).

    Bu bilgi sistemde zaten vardi ama yalnizca bir CEZA olarak (-0.20 sigma)
    kullaniliyordu: kullanici skorun neden dustugunu goruyor, ama TARIHI
    gormuyordu. Oysa 21 gunluk ufukta getiriyi en cok belirleyen tek olay
    budur -- pozisyon boyutu ve zamanlama kararini dogrudan etkiler.
    """
    cal = bundle.get("calendar") or {}
    raw = None
    for key in ("Earnings Date", "earningsDate"):
        v = cal.get(key)
        if isinstance(v, (list, tuple)) and v:
            v = v[0]
        if v is not None:
            raw = v
            break
    if raw is None:
        return {"available": False}

    try:
        d = pd.Timestamp(raw).normalize()
    except Exception:
        return {"available": False}

    days = int((d - pd.Timestamp.utcnow().normalize().tz_localize(None)).days)
    if days < -3:
        return {"available": False}      # gecmis tarih: veri bayat

    if days <= 2:
        sev, tr = "yuksek", "Bilanco cok yakin - tek gunde buyuk hareket olabilir"
    elif days <= 7:
        sev, tr = "orta", "Bilanco bu hafta - pozisyon boyutu kucultulmeli"
    elif days <= 21:
        sev, tr = "dusuk", "Bilanco 21 gunluk ufuk icinde"
    else:
        sev, tr = "yok", "Bilanco yakin degil"

    return {"available": True, "date": d.strftime("%Y-%m-%d"), "days": days,
            "severity": sev, "note_tr": tr}


def analyze(df: pd.DataFrame, bundle: dict, entry_price: float | None = None,
            score_now: float | None = None,
            score_at_entry: float | None = None) -> dict[str, Any]:
    """Bir hisse icin hedef + stop + sinyal paketinin tamami."""
    close = df["Close"].dropna()
    if len(close) < 60:
        return {"available": False, "reason": "yetersiz fiyat gecmisi"}

    price = float(close.iloc[-1])
    st = short_term_target(df, price)
    lt = long_term_target(df, bundle, price)
    stops = stop_levels(df, price, entry_price)
    sig = sell_signals(df, bundle, price, stops, st, lt,
                       entry_price, score_now, score_at_entry)

    out: dict[str, Any] = {
        "available": True,
        "price": round(price, 4),
        "short_term": st,
        "long_term": lt,
        "stops": stops,
        **sig,
    }

    # --- islem maliyeti: hedefler NET olarak da gosterilir (bulgu D3)
    adv = None
    if len(df) >= 30 and "Volume" in df:
        try:
            adv = float((df["Close"] * df["Volume"]).tail(30).mean())
        except Exception:
            adv = None
    costs = estimate_costs(price, adv)
    out["costs"] = costs
    if costs.get("available"):
        rt = costs["round_trip_pct"]
        for key in ("short_term", "long_term"):
            blk = out.get(key) or {}
            if blk.get("available") and blk.get("upside_pct") is not None:
                blk["net_upside_pct"] = round(blk["upside_pct"] - rt, 2)
                # Maliyet, getirinin ucte birinden fazlasini yiyorsa bunu
                # sessizce gecmek yaniltici olur.
                blk["cost_heavy"] = bool(blk["upside_pct"] > 0 and
                                         rt > blk["upside_pct"] / 3.0)

    # --- bilancoya kalan gun (kartta gosterilir)
    out["earnings"] = earnings_countdown(bundle)

    # --- pozisyon matematigi
    if entry_price and entry_price > 0:
        out["entry_price"] = round(float(entry_price), 4)
        out["pnl_pct"] = round((price / entry_price - 1) * 100, 2)
        if stops.get("available"):
            risk = entry_price - stops["active_stop"]
            if st.get("available") and risk > 0:
                reward = st["target"] - price
                out["risk_reward"] = round(reward / risk, 2)
            out["stop_pnl_pct"] = round((stops["active_stop"] / entry_price - 1) * 100, 2)

    # --- gunluk teknik ozet
    out["technical"] = {
        "rsi14": _n(ta.rsi(close, 14).iloc[-1]),
        "ma20": _n(ta.sma(close, 20).iloc[-1]) if len(close) >= 20 else None,
        "ma50": _n(ta.sma(close, 50).iloc[-1]) if len(close) >= 50 else None,
        "ma150": _n(ta.sma(close, 150).iloc[-1]) if len(close) >= 150 else None,
        "ma200": _n(ta.sma(close, 200).iloc[-1]) if len(close) >= 200 else None,
        "atr14": stops.get("atr14"),
        "atr_pct": stops.get("atr_pct"),
        "52w_high": _n(close.tail(252).max()),
        "52w_low": _n(close.tail(252).min()),
        "change_1d_pct": _n(100 * (close.iloc[-1] / close.iloc[-2] - 1)) if len(close) > 1 else None,
        "change_5d_pct": _n(100 * (close.iloc[-1] / close.iloc[-6] - 1)) if len(close) > 5 else None,
        "change_21d_pct": _n(100 * (close.iloc[-1] / close.iloc[-22] - 1)) if len(close) > 22 else None,
    }
    return out
