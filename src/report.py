"""Sonuclari kendi kendine yeten (self-contained) bir HTML panoya cevirir.

Harici CSS/JS/font yok — dosya cift tiklanarak acilir, internet gerektirmez.
Gorsel dil: karanlik vaporwave + kutsal geometri (bkz. theme.py).

Ana gorsel: siralanmis YATAY YIGILI CUBUK grafik.
  * cubuk uzunlugu  = toplam etki puani (siralamanin kendisi)
  * yigin dilimleri = her parametre kategorisinin katkisi
Boylece "kim onde" ve "neden onde" ayni gorselde okunur.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .theme import CRIMSON, PLANE, SERIES, SURFACE, hero_svg, sigil_svg

# Kategori -> etiket + seri rengi. Renkler validate_palette.js ile #120609
# zeminine karsi dogrulandi (tum kontroller PASS).
CATEGORY_STYLE = [
    ("technical",  "TEKNIK",     "∿", SERIES[0]),
    ("emergence",  "YUKSELEN",   "↗", SERIES[1]),
    ("potential",  "POTANSIYEL", "Δ", SERIES[2]),
    ("quality",    "KALITE",     "Ω", SERIES[3]),
    ("analyst",    "ANALIST",    "Σ", SERIES[4]),
    ("valuation",  "DEGERLEME",  "λ", SERIES[5]),
    ("momentum",   "MOMENTUM",   "Φ", SERIES[6]),
    ("risk",       "RISK/DIGER", "σ", SERIES[7]),
]
# growth -> YUKSELEN ; sentiment (WSB, squeeze) ve preference -> RISK/DIGER
CATEGORY_MERGE = {"growth": "emergence", "sentiment": "risk",
                  "preference": "risk", "model": "potential"}


def _clean(obj):
    """NaN / Inf / numpy tiplerini JSON'a uygun hale getirir.

    json.dumps(allow_nan=False) NaN gorunce patlar; Yahoo verisinde NaN her
    yerden sizabildigi icin payload'i tek noktadan temizliyoruz.
    """
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return f if math.isfinite(f) else None
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if obj is None or isinstance(obj, (str, int)):
        return obj
    if isinstance(obj, np.ndarray):
        return _clean(obj.tolist())
    if pd.isna(obj) is True:
        return None
    return str(obj)


def _payload(df: pd.DataFrame, diagnostics: dict, top_n: int) -> dict:
    cat_order = [c[0] for c in CATEGORY_STYLE]
    rows = []

    # Ilk N + izleme listesindekilerin TAMAMI. Liste gunluk degisir, ama
    # kullanicinin sectigi hisse siralamada geriye dusse bile gorunur kalir.
    shown = df.head(top_n)
    if "pinned" in df.columns:
        extra = df[df["pinned"].fillna(False) & ~df["ticker"].isin(shown["ticker"])]
        if len(extra):
            shown = pd.concat([shown, extra], ignore_index=True)
            shown = shown.sort_values("total_score", ascending=False, na_position="last")

    for _, r in shown.iterrows():
        contrib = {c: 0.0 for c in cat_order}
        factors = []

        for fid, fd in (r.get("factors") or {}).items():
            cat = CATEGORY_MERGE.get(fd.get("category"), fd.get("category"))
            if cat in contrib and fd.get("contribution") is not None:
                contrib[cat] += float(fd["contribution"])
            factors.append({
                "id": fid,
                "name": fd.get("name_tr", fid),
                "category": cat,
                "weight": fd.get("weight"),
                "score": fd.get("score"),
                "contribution": fd.get("contribution"),
                "available": bool(fd.get("available")),
                "raw": fd.get("raw"),
                "band": fd.get("band_label"),
            })

        factors.sort(key=lambda x: -(x["weight"] or 0))
        rets = r.get("returns") or {}

        # Ham katkilarin toplami, kapsama <%100 oldugunda toplam puandan kucuk
        # kalir (eksik faktorlerin agirligi yeniden dagitilir). Cubugun boyu
        # tablodaki puanla birebir ayni olsun diye dilimleri toplam puana
        # olcekliyoruz — oranlar korunur, uzunluk dogru okunur.
        total = r.get("total_score")
        s = sum(contrib.values())
        if total and s > 0:
            k = float(total) / s
            contrib = {c: v * k for c, v in contrib.items()}

        rows.append({
            "rank": int(r["rank"]),
            "ticker": r["ticker"],
            "name": r.get("name") or r["ticker"],
            "sector": r.get("sector"),
            "price": r.get("price"),
            "currency": r.get("currency") or "USD",
            "marketCap": r.get("market_cap"),
            "total": r.get("total_score"),
            "base": r.get("base_score"),
            "penalty": r.get("penalty"),
            "penaltiesHit": r.get("penalties_hit") or [],
            "coverage": r.get("coverage"),
            "lowConfidence": bool(r.get("low_confidence")),
            "pinned": bool(r.get("pinned")),
            "dollarVolume": r.get("avg_dollar_volume"),
            "maxPosition": r.get("max_position_usd"),
            "turnover": r.get("turnover_daily"),
            "prevRank": r.get("prev_rank"),
            "rankChange": r.get("rank_change"),
            "scoreChange": r.get("score_change"),
            "isNew": bool(r.get("is_new")),
            "contrib": {k2: round(v, 3) for k2, v in contrib.items()},
            "factors": factors,
            "categoryScores": r.get("category_scores") or {},
            "returns": {k2: rets.get(k2) for k2 in ("1m", "3m", "6m", "12m")},
            "rsi": r.get("rsi14"),
            "daysToEarnings": r.get("days_to_earnings"),
        })

    # Her hissenin yuzdelik dilimi: "1. sira" yerine "ilk %3" demek daha durust,
    # cunku tepe bolgesinde puan farklari gunluk gurultunun altinda (bulgu K2).
    n_scored = max(1, len(df))
    for r in rows:
        r["percentile"] = round(100.0 * r["rank"] / n_scored, 1)

    return {
        "generatedAt": datetime.now(timezone.utc).strftime("%Y.%m.%d · %H:%M UTC"),
        "generatedTs": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rows": rows,
        "categories": [{"id": c[0], "label": c[1], "glyph": c[2], "color": c[3]}
                       for c in CATEGORY_STYLE],
        "diagnostics": diagnostics,
        "totalScored": int(len(df)),
        "validation": _validation_state(),
        "noise": _score_noise(df, diagnostics),
        "paper": _paper_state(),
        "factorIC": _factor_ic_state(),
        "factorTime": _factor_time_state(),
        "regime": diagnostics.get("regime") or {},
        "health": _health_state(diagnostics),
    }


def _paper_state() -> dict | None:
    """Kagit uzerinde defterin ozeti (bkz. src/paper.py)."""
    try:
        from .paper import load_summary
        return load_summary()
    except Exception:
        return None


def _factor_ic_state() -> dict | None:
    """Parametre bazli IC tablosu.

    Bu olcum sistemde bastan beri vardi (`run.py learn`) ama HICBIR YERDE
    gorunmuyordu: kullanici 28 parametreye bakip hangisinin ise yaradigini
    bilemiyordu. Panoda gosterilmesi, "kesinlik cilasi" elestirisine verilen
    en dogrudan cevap.
    """
    p = Path(__file__).resolve().parents[1] / "data" / "faktor_ic.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _factor_time_state() -> dict | None:
    """Parametre gucunun zaman ve rejim kirilimi (bkz. src/faktor_zaman.py).

    IC tablosu "ortalama guc" soruyordu; bu dosya "ortalama gercek mi, artiyor
    mu azaliyor mu, hangi ortamda" sorulariyla ayni tabloyu tamamliyor. Ayri
    bir bolum acmak yerine ayni tabloya sutun olarak giriyor -- kullanicinin
    iki listeyi kafasinda eslestirmesi gerekmesin.
    """
    p = Path(__file__).resolve().parents[1] / "data" / "faktor_zaman.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _health_state(diagnostics: dict) -> dict:
    """Sistemin kendi sagligi — terminale girmeden gorunsun (bkz. scripts/durum.py)."""
    out: dict = {
        "fetchRate": diagnostics.get("fetch_success_rate"),
        "yahooRate": diagnostics.get("yahoo_success_rate"),
        "fallbackUsed": diagnostics.get("fallback_used"),
        "universeFull": diagnostics.get("universe_full"),
        "scored": diagnostics.get("scored_universe"),
        "rateLimited": diagnostics.get("fetch_rate_limited"),
        "aborted": bool(diagnostics.get("fetch_aborted")),
        "coveragePct": diagnostics.get("universe_coverage_pct"),
        # Satirlarin hepsi bugunun fiyatindan gelmiyor: donusumlu tarama bir
        # turda evrenin bir dilimini ceker, gerisi onbellekten katilir. Kac
        # satirin ne kadar eski oldugu SOYLENMEZSE sira "bugunun siralamasi"
        # gibi okunur. run.py diag["data_age"] icinde sayiyor.
        "dataAge": diagnostics.get("data_age"),
    }
    # DIKKAT: run_status.json BURADAN OKUNMAZ. O dosya panodan SONRA yazilir,
    # yani burada okunan kayit her zaman bir ONCEKI calismaya aittir ve pano
    # "son calisma hatali" gibi yanlis bir sey gosterir. Bu taramanin zamani
    # zaten `generatedAt` icinde; pano onu kullaniyor.
    try:
        from .delisting import info as _dl_info
        out["delisting"] = _dl_info()
    except Exception:
        pass
    try:
        from .fundamentals import info as _f_info
        out["fundamentals"] = _f_info()
    except Exception:
        pass
    return out


def _validation_state() -> dict:
    """Sistemin dogrulanma durumu (denetim bulgusu K1).

    Agirliklar uzman gorusu oldugu ve hicbiri ileri getiriyle test edilmedigi
    surece bu acikca gosterilmeli — aksi halde pano hak etmedigi bir kesinlik
    izlenimi verir.
    """
    from .ml import FEATURE_STORE

    snaps = sorted(FEATURE_STORE.glob("snapshot_*.csv")) if FEATURE_STORE.exists() else []
    dates = sorted({p.stem.replace("snapshot_", "")[:10] for p in snaps})
    span = 0
    if len(dates) >= 2:
        try:
            span = (datetime.strptime(dates[-1], "%Y-%m-%d")
                    - datetime.strptime(dates[0], "%Y-%m-%d")).days
        except ValueError:
            span = 0

    learned = Path(__file__).resolve().parents[1] / "output" / "learned_weights.json"
    ic_done = learned.exists()

    # Ogrenilen modelin durumu (kendi kendini besleyen dongu)
    champ = None
    try:
        from .training import champion
        champ = champion()
    except Exception:
        champ = None

    needed_days, needed_snaps = 120, 60
    return {
        "snapshots": len(dates),
        "span_days": span,
        "ic_measured": ic_done,
        "validated": bool((ic_done or champ) and span >= needed_days
                          and len(dates) >= needed_snaps),
        "needed_days": needed_days,
        "needed_snapshots": needed_snaps,
        "progress_pct": round(100 * min(1.0, min(span / needed_days,
                                                 len(dates) / needed_snaps)), 1),
        "model": None if not champ else {
            "name": champ.get("model"),
            "weight": champ.get("weight"),
            "ic": champ.get("ic"),
            "icir": champ.get("icir"),
            "promoted_at": champ.get("promoted_at"),
        },
        **_countdown(len(dates), span, needed_snaps, needed_days),
        "pretrain": _pretrain_state(),
    }


def _countdown(snaps: int, span: int, need_snaps: int, need_days: int) -> dict:
    """Sayacin ne zaman dolacagi.

    "%2.5" tek basina anlamsiz — kullanicinin bilmek istedigi sey NE ZAMAN
    bitecegi ve bunun icin bir sey yapmasi gerekip gerekmedigi.

    Anlik goruntu is gunlerinde birikir; kalan takvim gunu bu yuzden 7/5
    ile carpilir.
    """
    miss_snaps = max(0, need_snaps - snaps)
    miss_days = max(0, need_days - span)
    if not miss_snaps and not miss_days:
        return {"days_left": 0, "eta": None}

    left = max(int(round(miss_snaps * 7 / 5)), miss_days)
    eta = datetime.now() + timedelta(days=left)
    return {
        "days_left": left,
        "eta": eta.strftime("%d.%m.%Y"),
        "missing_snapshots": miss_snaps,
        "missing_days": miss_days,
    }


def _pretrain_state() -> dict | None:
    """Gecmise donuk on egitim paneli var mi?

    Kullanici sayaci gorunce "bekleyecek miyim?" diye soruyor. Cevabin yarisi
    bu: beklemeden denenebilecek bir panel var mi, varsa ne kadar buyuk.
    """
    try:
        from .backfill import info as _bf_info
        m = _bf_info()
    except Exception:
        return None
    if not m or not m.get("ok"):
        return None
    return {
        "snapshots": m.get("snapshots"),
        "rows": m.get("rows"),
        "tickers": m.get("tickers_used"),
        "first_date": m.get("first_date"),
        "last_date": m.get("last_date"),
        "partial": bool(m.get("partial") or not m.get("complete", True)),
    }


def _score_noise(df: pd.DataFrame, diagnostics: dict) -> dict:
    """Gunluk tipik puan oynamasi — belirsizlik bandi icin (bulgu K2).

    Kaynak, compute_deltas'in zaten hesapladigi `score_change` sutunudur.
    Dosyadan okumak yanlis olurdu: pano CSV'den once yazildigi icin bir
    onceki calismanin dosyasi okunurdu.
    """
    if "score_change" not in df.columns:
        return {"available": False}
    d = pd.to_numeric(df["score_change"], errors="coerce").abs().dropna()
    if len(d) < 20:
        return {"available": False}
    return {
        "available": True,
        "median_abs_change": round(float(d.median()), 2),
        "p90_abs_change": round(float(d.quantile(0.90)), 2),
        "compared_to": (diagnostics.get("deltas") or {}).get("compared_to"),
        "n": int(len(d)),
    }


_CSS = f"""
:root{{
  color-scheme: dark;
  --plane:{PLANE}; --surface:{SURFACE};
  --ink:#f4e9e6; --ink-2:#b09a97; --ink-3:#6f5c5c;
  --crimson:{CRIMSON}; --crimson-dim:#8f2118; --crimson-glow:rgba(240,72,58,.28);
  --steel:#3d6b8a; --steel-dim:#1e3547;
  --rule:rgba(240,72,58,.20); --rule-2:rgba(160,140,140,.13);
  --track:#1e1013;
  --good:#3fbf6a; --bad:#f0483a;
  --disp:"Impact","Haettenschweiler","Franklin Gothic Bold","Arial Narrow Bold",
         "Oswald",sans-serif;
  --mono:"Cascadia Mono","Consolas","SF Mono",ui-monospace,monospace;
  --body:"Segoe UI",-apple-system,"Helvetica Neue",sans-serif;
}}
*{{box-sizing:border-box}}
html{{background:var(--plane)}}
body{{
  margin:0; overflow-x:hidden; background:var(--plane); color:var(--ink);
  font:14px/1.6 var(--body);
  -webkit-font-smoothing:antialiased;
}}
/* Sabit atmosfer katmani. 'background-attachment:fixed' uzun sayfalarda her
   kaydirmada tam yeniden boyama tetikledigi icin ayri bir katman kullaniliyor. */
body::before{{
  content:"";position:fixed;inset:0;z-index:-1;pointer-events:none;
  background:
    radial-gradient(ellipse 90% 55% at 50% 0%, rgba(143,33,24,.26), transparent 70%),
    radial-gradient(ellipse 60% 40% at 85% 12%, rgba(61,107,138,.09), transparent 70%),
    var(--plane);
}}
.wrap{{max-width:1280px;margin:0 auto;padding:0 22px 100px}}

/* ============ BASLIK ============ */
.hero{{position:relative;width:100%;overflow:hidden;
  border-bottom:1px solid var(--rule);min-height:min(74vh,520px);
  display:flex;align-items:flex-end}}
.hero-svg{{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}}
.hero-svg .dome ellipse,.hero-svg .dome path{{fill:none;stroke:var(--steel-dim);stroke-width:.7}}
/* Not: bu yollara drop-shadow UYGULANMAZ. Binlerce noktali bir yolda filtre
   her kaydirmada yeniden rasterize edilir ve sayfayi kilitler. Parlama etkisi
   .hero::after ile ucuz bir radyal degradeden geliyor. */
.hero-svg .traces path{{fill:none;stroke:var(--crimson);stroke-width:.75;opacity:.5}}
.hero::after{{content:"";position:absolute;left:50%;top:6%;width:min(760px,68%);
  height:52%;transform:translateX(-50%);pointer-events:none;
  background:radial-gradient(ellipse at center,var(--crimson-glow),transparent 68%)}}
.hero-svg .floor line{{stroke:var(--steel);stroke-width:.55;opacity:.45}}
.hero-svg .ridge{{fill:#05070a}}
.hero-inner{{position:relative;width:100%;max-width:1280px;margin:0 auto;padding:0 22px 40px}}
.eyebrow{{font:600 11px/1 var(--mono);letter-spacing:.42em;text-transform:uppercase;
  color:var(--crimson);margin-bottom:14px;display:flex;align-items:center;gap:12px}}
.eyebrow::after{{content:"";height:1px;flex:1;background:linear-gradient(90deg,var(--rule),transparent)}}
h1{{font:400 clamp(46px,9.5vw,116px)/.80 var(--disp);letter-spacing:-.015em;
  text-transform:uppercase;margin:0;color:var(--ink);
  text-shadow:0 0 44px rgba(240,72,58,.34), 3px 0 0 rgba(240,72,58,.42), -3px 0 0 rgba(26,159,184,.30)}}
h1 em{{font-style:normal;color:var(--crimson);display:block}}
.hero-meta{{display:flex;flex-wrap:wrap;gap:8px 26px;margin-top:20px;
  font:500 11px/1 var(--mono);letter-spacing:.20em;text-transform:uppercase;color:var(--ink-2)}}
.hero-meta b{{color:var(--ink);font-weight:600}}

/* ============ BOLUM ============ */
section{{position:relative;margin-top:62px}}
.sec-head{{display:flex;align-items:baseline;gap:16px;margin-bottom:6px}}
.sec-num{{font:400 13px/1 var(--mono);color:var(--crimson);letter-spacing:.1em;
  border:1px solid var(--rule);padding:6px 9px;flex:0 0 auto}}
h2{{font:400 clamp(24px,3.6vw,40px)/1 var(--disp);letter-spacing:.005em;
  text-transform:uppercase;margin:0;color:var(--ink)}}
.sec-note{{color:var(--ink-2);font-size:13px;margin:10px 0 24px;max-width:78ch}}
.sec-rule{{height:1px;background:linear-gradient(90deg,var(--crimson),var(--rule) 30%,transparent);
  margin:14px 0 22px}}

/* ============ PANEL ============ */
.panel{{background:linear-gradient(160deg,rgba(30,16,19,.72),rgba(10,4,6,.55));
  border:1px solid var(--rule-2);padding:24px;position:relative;
  clip-path:polygon(0 0,calc(100% - 20px) 0,100% 20px,100% 100%,20px 100%,0 calc(100% - 20px))}}
.panel::before{{content:"";position:absolute;top:0;right:0;width:20px;height:20px;
  background:linear-gradient(225deg,var(--crimson) 50%,transparent 51%);opacity:.55}}

/* ============ DURUM SERIDI ============ */
.st{{border:1px solid var(--rule-2);border-left:3px solid var(--c,var(--crimson));
  padding:15px 19px;margin:0 0 12px;background:linear-gradient(160deg,rgba(30,16,19,.6),rgba(10,4,6,.4))}}
.st.warn{{--c:{CRIMSON}}} .st.info{{--c:#3d6b8a}} .st.ok{{--c:#1ba372}}
/* Yalnizca DOGRUDAN cocuk <b> etikettir ve blok olur. Paragraf icindeki <b>
   vurgular satir ici kalmali — aksi halde cumleler parcalanir. */
.st > b{{display:block;font:600 10px/1 var(--mono);letter-spacing:.2em;
  color:var(--c,var(--crimson));margin-bottom:8px}}
.st p{{margin:0;font-size:12.5px;color:var(--ink-2);line-height:1.65}}
.st p b{{display:inline;color:var(--ink);font-weight:600;letter-spacing:0;font-family:inherit}}
.st code{{font:400 11.5px/1 var(--mono);color:var(--ink);background:var(--plane);
  padding:3px 7px;border:1px solid var(--rule-2)}}
.stbar{{display:block;height:4px;background:var(--track);margin:9px 0;
  outline:1px solid rgba(160,140,140,.08)}}
.stbar>i{{display:block;height:100%;background:var(--c,var(--crimson))}}

/* ============ KARNE / IC / SAGLIK TABLOLARI ============ */
.muted{{color:var(--ink-3,#8b7d7d);font-size:11.5px;line-height:1.6;margin:10px 0 0}}
.panel-sub{{margin:14px 0 0;padding:12px 14px;border-radius:8px;background:rgba(255,255,255,.03);border:1px solid rgba(255,255,255,.06);font-size:12px;color:var(--ink-2)}}
ul.tight{{margin:8px 0 0;padding-left:18px}}
ul.tight li{{margin:5px 0;line-height:1.55}}
table.mini{{width:100%;border-collapse:collapse;margin:10px 0 0;
  font:400 12px/1.5 var(--sans,inherit)}}
table.mini td,table.mini th{{padding:7px 10px;border-bottom:1px solid var(--rule-2);
  text-align:left;vertical-align:top}}
table.mini thead th{{font:600 10px/1 var(--mono);letter-spacing:.14em;
  color:var(--ink-3,#8b7d7d);text-transform:uppercase}}
table.mini td:first-child{{color:var(--ink-3,#8b7d7d);white-space:nowrap;width:34%}}
table.mini.wide td:first-child{{color:var(--ink);width:auto;font:400 12px/1.5 var(--mono)}}
table.mini .num{{text-align:right;font:400 12px/1.5 var(--mono);white-space:nowrap}}
/* IC satirlari: gurultu olan parametreler goze carpsin — asil bilgi budur */
tr.ic-warn td{{color:var(--ink-3,#8b7d7d)}}
tr.ic-ok td:nth-child(2){{color:#1ba372;font-weight:600}}
tr.ic-info td:nth-child(2){{color:#c9a227}}

/* ============ SAYAC KUTULARI ============ */
.tiles{{display:grid;grid-template-columns:repeat(auto-fit,minmax(158px,1fr));gap:1px;
  background:var(--rule-2);border:1px solid var(--rule-2);margin-top:-1px}}
.tile{{background:var(--surface);padding:20px 18px;position:relative}}
.tile .k{{font:600 10px/1 var(--mono);letter-spacing:.22em;text-transform:uppercase;
  color:var(--ink-3)}}
.tile .v{{font:400 40px/1 var(--disp);margin-top:12px;color:var(--ink);letter-spacing:.01em}}
.tile:nth-child(1) .v{{color:var(--crimson)}}
.tile .glyph{{position:absolute;top:14px;right:14px;font:400 15px/1 var(--mono);color:var(--ink-3);opacity:.6}}

/* ============ LEJANT ============ */
.legend{{display:flex;flex-wrap:wrap;gap:0;margin:0 0 22px;border:1px solid var(--rule-2)}}
.legend button{{display:flex;align-items:center;gap:9px;background:transparent;border:0;
  border-right:1px solid var(--rule-2);padding:11px 15px;cursor:pointer;color:var(--ink-2);
  font:600 10px/1 var(--mono);letter-spacing:.17em;text-transform:uppercase;flex:1 1 auto;
  transition:background .12s,color .12s}}
.legend button:last-child{{border-right:0}}
.legend button:hover{{background:rgba(240,72,58,.07);color:var(--ink)}}
.legend button[aria-pressed="false"]{{opacity:.3}}
.legend .gl{{font-size:13px;line-height:1}}
.swatch{{width:10px;height:10px;flex:0 0 auto;transform:rotate(45deg)}}

/* ============ GRAFIK ============ */
.chart-scroll{{overflow-x:auto}}
.bars{{min-width:750px}}
.row{{display:grid;grid-template-columns:34px 146px 82px 1fr 68px 74px;align-items:center;
  gap:14px;padding:6px 0;border-bottom:1px solid rgba(160,140,140,.06)}}
.px{{text-align:right;line-height:1.15}}
.px b{{display:block;font:600 14px/1 var(--mono);letter-spacing:.02em;color:var(--ink)}}
.px small{{display:block;font:500 8.5px/1 var(--mono);letter-spacing:.16em;
  color:var(--ink-3);margin-top:4px}}
.row:hover{{background:rgba(240,72,58,.05)}}
.row .rk{{font:400 20px/1 var(--disp);color:var(--ink-3);text-align:right;letter-spacing:.02em;
  position:relative}}
.row:nth-child(-n+3) .rk{{color:var(--crimson)}}
/* izleme listesindekiler: siralamadan dusseler bile listede kalir */
.row.pinned{{background:linear-gradient(90deg,rgba(240,72,58,.09),transparent 60%);
  box-shadow:inset 2px 0 0 0 var(--crimson)}}
.pin{{color:var(--crimson);margin-right:5px;font-size:11px;vertical-align:1px}}
/* gunluk hareket rozeti */
.dlt{{display:block;font:600 8px/1 var(--mono);letter-spacing:.06em;font-style:normal;
  margin-top:4px;text-align:right}}
.dlt.up{{color:var(--good)}} .dlt.dn{{color:var(--bad)}}
.dlt.new{{color:var(--crimson);letter-spacing:.1em}}
td .dlt{{display:inline;margin:0 0 0 7px}}
.lbl{{min-width:0}}
.lbl .tk{{font:600 14px/1.15 var(--mono);letter-spacing:.06em;color:var(--ink)}}
.lbl .nm{{color:var(--ink-3);font-size:10.5px;overflow:hidden;text-overflow:ellipsis;
  white-space:nowrap;margin-top:3px}}
.bar{{position:relative;height:19px;display:flex;background:var(--track);
  outline:1px solid rgba(160,140,140,.09)}}
.seg{{height:100%;box-shadow:2px 0 0 0 var(--plane)}}
.seg:last-child{{box-shadow:none}}
.val{{font:400 22px/1 var(--disp);text-align:right;color:var(--ink);letter-spacing:.02em}}
.val .pm{{font:500 8.5px/1 var(--mono);font-style:normal;color:var(--ink-3);
  letter-spacing:.02em;margin-left:3px}}
.val small{{display:block;font:500 8px/1 var(--mono);letter-spacing:.11em;
  color:var(--ink-3);margin-top:5px;text-transform:uppercase}}
.flag{{font:600 8.5px/1 var(--mono);letter-spacing:.13em;padding:3px 5px;
  border:1px solid var(--ink-3);color:var(--ink-3);margin-left:7px;vertical-align:2px}}
.flag.pen{{color:var(--crimson);border-color:var(--crimson-dim)}}

/* ============ AGIRLIK LISTESI ============ */
.wrow{{display:grid;grid-template-columns:26px 1fr 62px 132px;gap:14px;align-items:center;
  padding:10px 0;border-bottom:1px solid var(--rule-2)}}
.wrow:last-child{{border-bottom:0}}
.wrow .wi{{font:400 14px/1 var(--mono);color:var(--ink-3);text-align:right}}
.wrow .wn{{font-size:13.5px;color:var(--ink)}}
.wrow .wn small{{display:block;color:var(--ink-3);font:500 9.5px/1.5 var(--mono);
  letter-spacing:.13em;text-transform:uppercase;margin-top:3px}}
.wrow .wv{{font:400 24px/1 var(--disp);text-align:right;color:var(--ink)}}
.wbar{{height:5px;background:var(--track);outline:1px solid rgba(160,140,140,.08)}}
.wbar>i{{display:block;height:100%}}
.note{{border-left:2px solid var(--crimson);padding:11px 15px;margin:10px 0;
  background:rgba(240,72,58,.06);font-size:12.5px;color:var(--ink-2)}}
.note b{{color:var(--ink);font-weight:600}}
/* Dogrudan cocuk <b> bir BASLIKTIR ve kendi satirini alir */
.note > b:first-child{{display:block;font:600 9.5px/1 var(--mono);letter-spacing:.19em;
  text-transform:uppercase;color:var(--crimson);margin-bottom:7px}}

/* ============ TABLO ============ */
.controls{{display:flex;flex-wrap:wrap;gap:10px;align-items:center;margin-bottom:18px}}
input[type=search],select{{font:500 11px/1 var(--mono);letter-spacing:.13em;
  text-transform:uppercase;padding:10px 13px;border:1px solid var(--rule-2);
  background:var(--surface);color:var(--ink);outline:none}}
input[type=search]:focus,select:focus{{border-color:var(--crimson)}}
input[type=search]::placeholder{{color:var(--ink-3)}}
.cnt{{font:500 10px/1 var(--mono);letter-spacing:.18em;text-transform:uppercase;color:var(--ink-3)}}
.tbl-scroll{{overflow-x:auto;border:1px solid var(--rule-2)}}
table{{border-collapse:collapse;width:100%;min-width:940px}}
th,td{{padding:11px 13px;text-align:right;border-bottom:1px solid var(--rule-2);white-space:nowrap}}
th:nth-child(-n+3),td:nth-child(-n+3){{text-align:left}}
th{{font:600 9.5px/1 var(--mono);letter-spacing:.18em;text-transform:uppercase;
  color:var(--ink-3);cursor:pointer;user-select:none;position:sticky;top:0;
  background:#150a0d;border-bottom:1px solid var(--rule)}}
th:hover{{color:var(--crimson)}}
td{{font:400 13px/1.3 var(--mono);color:var(--ink-2)}}
td:nth-child(2){{color:var(--ink);font-weight:600;letter-spacing:.05em}}
td.big{{font:400 19px/1 var(--disp);color:var(--ink)}}
tbody tr{{cursor:pointer}}
tbody tr:hover{{background:rgba(240,72,58,.06)}}
tbody tr.pinrow{{box-shadow:inset 2px 0 0 0 var(--crimson);background:rgba(240,72,58,.05)}}
.pos{{color:var(--good)}} .neg{{color:var(--bad)}}
.det td{{white-space:normal;padding:20px 22px;background:rgba(240,72,58,.04);
  border-bottom:1px solid var(--rule)}}
.det h3{{font:400 22px/1 var(--disp);letter-spacing:.02em;text-transform:uppercase;margin:0 0 4px}}
.det .dm{{font:500 10px/1.7 var(--mono);letter-spacing:.15em;text-transform:uppercase;
  color:var(--ink-3);margin-bottom:16px}}

/* ============ FAKTOR IZGARASI ============ */
.fgrid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(268px,1fr));gap:1px;
  background:var(--rule-2);border:1px solid var(--rule-2)}}
.fitem{{display:grid;grid-template-columns:12px 1fr 44px;gap:10px;align-items:center;
  padding:10px 13px;background:var(--surface)}}
.fitem .fn{{min-width:0;font-size:12px;color:var(--ink-2)}}
.fitem .fn span{{display:block;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}
.fitem .fn small{{color:var(--ink-3);font:500 9px/1.4 var(--mono);letter-spacing:.11em}}
.fitem .fs{{font:400 21px/1 var(--disp);text-align:right;color:var(--ink)}}
.fitem.na{{opacity:.34}}
.fitem.na .fs{{font:500 9px/1 var(--mono);letter-spacing:.1em;color:var(--ink-3)}}

/* ============ IPUCU ============ */
.tooltip{{position:fixed;pointer-events:none;z-index:99;background:#160a0e;
  border:1px solid var(--crimson-dim);padding:12px 15px;font-size:12px;
  box-shadow:0 12px 44px rgba(0,0,0,.72);opacity:0;transition:opacity .1s;max-width:310px}}
.tooltip.on{{opacity:1}}
.tooltip b.tt-h{{display:block;margin-bottom:8px;font:600 10px/1 var(--mono);
  letter-spacing:.19em;text-transform:uppercase;color:var(--crimson)}}
.tt-row{{display:flex;justify-content:space-between;gap:18px;color:var(--ink-2);
  font:400 11.5px/1.9 var(--mono)}}
.tt-row b{{color:var(--ink)}}

/* ============ UYARI / ALTLIK ============ */
.disclaimer{{border:1px solid var(--rule);padding:16px 20px;margin-top:26px;
  font-size:12.5px;color:var(--ink-2);background:rgba(240,72,58,.05)}}
.disclaimer b{{color:var(--crimson);font:600 10px/1 var(--mono);letter-spacing:.19em;
  text-transform:uppercase;display:block;margin-bottom:7px}}
footer{{margin-top:70px;padding-top:22px;border-top:1px solid var(--rule-2);
  font:500 10px/1.9 var(--mono);letter-spacing:.17em;text-transform:uppercase;color:var(--ink-3);
  display:flex;flex-wrap:wrap;gap:8px 30px;justify-content:space-between}}
.sigil{{width:64px;height:64px;position:absolute;top:16px;right:20px;opacity:.30}}
.sigil circle{{fill:none;stroke:var(--crimson);stroke-width:.8}}

@media (max-width:720px){{
  .row{{grid-template-columns:26px 84px 62px 1fr 48px 62px;gap:7px}}
  .px b{{font-size:12px}}
  .wrow{{grid-template-columns:22px 1fr 50px;}}
  .wrow .wbar{{display:none}}
  .hero{{min-height:300px}}
}}
"""


def build_html(df: pd.DataFrame, diagnostics: dict, top_n: int = 40,
               title: str = "SIGMA / HISSE SIRALAMA MOTORU") -> str:
    data = _clean(_payload(df, diagnostics, top_n))
    data_json = json.dumps(data, ensure_ascii=False, allow_nan=False)

    return f"""<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<style>{_CSS}</style>
<div class="hero">
  {hero_svg()}
  <div class="hero-inner">
    <div class="eyebrow">Cok faktorlu agirlikli skorlama motoru</div>
    <h1>HISSE<em>SIRALAMA</em></h1>
    <div class="hero-meta">
      <span>ANLIK GORUNTU <b id="gen"></b></span>
      <span>EVREN <b id="m-univ"></b></span>
      <span>PARAMETRE <b id="m-fact"></b></span>
      <span>OLCEK <b>0&#8212;100</b></span>
    </div>
  </div>
</div>

<div class="wrap">
  <div id="statusBar"></div>
  <div class="tiles" id="tiles"></div>

  <section>
    <div class="sec-head"><span class="sec-num">I</span><h2>Toplam Etki Puani</h2></div>
    <div class="sec-rule"></div>
    <p class="sec-note">Cubuk uzunlugu toplam puani, dilimler her parametre kategorisinin
      katkisini gosterir. Lejanda tiklayarak kategori ac/kapa; dilim uzerine gelince kirilim
      acilir. Sagdaki <b>+ EKLE</b> ile hisseyi izleme listesine al.</p>
    <p class="sec-note" id="deltaNote" style="margin-top:-14px"></p>
    <div class="legend" id="legend"></div>
    <div class="chart-scroll"><div class="bars" id="bars"></div></div>
  </section>

  <section>
    <div class="sec-head"><span class="sec-num">II</span><h2>Parametre Agirliklari</h2></div>
    <div class="sec-rule"></div>
    <p class="sec-note">Etki puanlari yuksekten dusuge. Kapsama, parametrenin evrenin
      yuzde kacinda olculebildigini gosterir; eksik parametrenin agirligi digerlerine dagitilir.</p>
    <div class="panel">{sigil_svg()}<div id="weights"></div><div id="disabled"></div></div>
  </section>

  <section>
    <div class="sec-head"><span class="sec-num">III</span><h2>Tum Sonuclar</h2></div>
    <div class="sec-rule"></div>
    <div class="controls">
      <input type="search" id="q" placeholder="Sembol / sirket ara">
      <select id="sec"><option value="">Tum sektorler</option></select>
      <span class="cnt" id="cnt"></span>
    </div>
    <div class="tbl-scroll"><table id="tbl">
      <thead><tr>
        <th data-k="rank">#</th><th data-k="ticker">Sembol</th><th data-k="sector">Sektor</th>
        <th data-k="price">Fiyat</th><th data-k="total">Puan</th>
        <th data-k="base">Ham</th><th data-k="penalty">Ceza</th>
        <th data-k="r1">1A %</th><th data-k="r3">3A %</th><th data-k="r12">12A %</th>
        <th data-k="dollarVolume" title="30 gunluk ortalama gunluk dolar hacmi">Hacim/gun</th>
        <th data-k="maxPosition" title="Gunluk hacmin %5'i — piyasayi bozmadan girilebilecek kaba ust sinir">Azami poz.</th>
        <th data-k="coverage">Kapsama</th><th>Izle</th>
      </tr></thead><tbody></tbody>
    </table></div>
    <p class="sec-note" style="margin-top:14px">Satira tiklayarak parametre kirilimini ac.
      <b>+ EKLE</b> ile hisseyi izleme listesine al &mdash; sagdaki panelden alis fiyatini
      girip komutu kopyalayabilirsin.</p>
  </section>
  <section id="secPaper" hidden>
    <div class="sec-head"><span class="sec-num">IV</span><h2>Karne &mdash; Ilk 20 Ne Yapti</h2></div>
    <div class="sec-rule"></div>
    <p class="sec-note">Her tarama gunu listenin ilk 20'si bir <b>kohort</b> olarak
      deftere yazilir, 21 islem gunu tutulur ve SPY'a karsi olculur. Portfoy
      simulasyonu degildir: sermaye, pozisyon boyutu ve nakit yonetimi devre disidir
      &mdash; olculen tek sey <b>siralamanin kendisi</b>.</p>
    <div id="paperBody"></div>
  </section>

  <section id="secIC" hidden>
    <div class="sec-head"><span class="sec-num">V</span><h2>Parametreler Gercekten Calisiyor mu</h2></div>
    <div class="sec-rule"></div>
    <p class="sec-note">Bilgi Katsayisi (IC), bir parametrenin skoru ile o hissenin
      ileri getirisi arasindaki siralama korelasyonudur. Olcut:
      <b>|IC| &gt; 0.03</b> zayif ama kullanilabilir, <b>&gt; 0.05</b> iyi,
      <b>&gt; 0.10</b> cok iyi. Bunun altindaki her sey gurultudur ve o parametrenin
      agirligi savunulamaz.</p>
    <div class="panel"><div id="icBody"></div></div>
  </section>

  <section id="secHealth" hidden>
    <div class="sec-head"><span class="sec-num">VI</span><h2>Sistem Sagligi</h2></div>
    <div class="sec-rule"></div>
    <p class="sec-note">Otomasyonun kendi durumu &mdash; terminale girmeden gorunsun.</p>
    <div class="panel"><div id="healthBody"></div></div>
  </section>

__ASSISTANT_HTML__

  <div class="disclaimer">
    <b>Uyari</b>
    Bu arac yatirim tavsiyesi degildir. Skorlar gecmis ve kamuya acik verilerden
    hesaplanan istatistiksel gostergelerdir; gelecek getiriyi garanti etmez.
    Karar oncesi kendi arastirmani yap.
  </div>

  <footer>
    <span>Kaynaklar · Yahoo Finance · ApeWisdom · Tradestie</span>
    <span id="foot-gen"></span>
  </footer>
</div>
<div class="tooltip" id="tt"></div>
<script>
const DATA = {data_json};
const CATS = DATA.categories;
const CAT_MERGE = {json.dumps(CATEGORY_MERGE)};
const CAT_BY_ID = Object.fromEntries(CATS.map(c => [c.id, c]));
/* Bazi faktor kategorileri (sentiment, preference) grafikte tek dilimde
   birlestirilir; bu yardimci her zaman gecerli bir kategori dondurur. */
const catOf = id => CAT_BY_ID[CAT_MERGE[id] || id] || CATS[CATS.length - 1];
const off = new Set();
const ROMAN = ['I','II','III','IV','V','VI','VII','VIII','IX','X','XI','XII',
               'XIII','XIV','XV','XVI','XVII','XVIII','XIX','XX'];
const fmtPct = v => v == null ? '—' : (v*100).toFixed(1);
const fmtNum = (v,d=1) => v == null ? '—' : Number(v).toFixed(d);
const fmtCap = v => {{
  if (v == null) return '—';
  for (const [n,s] of [[1e12,'T'],[1e9,'Mr'],[1e6,'Mn']]) if (v >= n) return (v/n).toFixed(1)+s;
  return String(Math.round(v));
}};

const D = DATA.diagnostics;
document.getElementById('gen').textContent = DATA.generatedAt;
document.getElementById('foot-gen').textContent = DATA.generatedAt;
document.getElementById('m-univ').textContent = D.universe_size ?? DATA.totalScored;
document.getElementById('m-fact').textContent = (D.active_factors||[]).length;

/* ---------- sayac kutulari ---------- */
(function tiles() {{
  const rows = DATA.rows;
  const best = rows.length ? rows[0] : null;
  const dl = D.deltas || {{}};
  const nNew = rows.filter(r => r.isNew).length;
  const nPin = rows.filter(r => r.pinned).length;
  const t = [
    ['En yuksek puan', best ? fmtNum(best.total,1) : '—', 'Φ'],
    ['Skorlanan hisse', DATA.totalScored, 'Σ'],
    ['Bugun listeye giren', nNew, '✦'],
    ['Izleme listem', nPin, '★'],
    ['Aktif parametre', (D.active_factors||[]).length, 'λ'],
  ];
  document.getElementById('tiles').innerHTML = t.map(([k,v,g]) =>
    `<div class="tile"><span class="glyph">${{g}}</span>
       <div class="k">${{k}}</div><div class="v">${{v}}</div></div>`).join('');
}})();

/* ---------- durum seridi: dogrulama + veri tazeligi ---------- */
(function statusBar() {{
  const v = DATA.validation || {{}};
  const parts = [];

  // Piyasa rejimi — siralamayi DEGISTIRMEZ, hangi ortamda uretildigini soyler.
  // Momentum agirlikli bir siralama dusus rejiminde tarihsel olarak en kotu
  // sonucu verir; bunu bilmeden listeye bakmak eksik bilgiyle bakmaktir.
  const rg = DATA.regime || {{}};
  if (rg.label && rg.label !== 'BILINMIYOR') {{
    const cls = rg.label === 'DUSUS' ? 'warn' : (rg.label === 'YUKSELIS' ? 'ok' : 'info');
    const bits = [];
    if (rg.vs_ma200_pct != null) bits.push(`endeks 200g ort. ${{rg.vs_ma200_pct > 0 ? '+' : ''}}${{rg.vs_ma200_pct}}%`);
    if (rg.breadth_pct != null) bits.push(`genislik %${{rg.breadth_pct}}`);
    if (rg.vol20_annual_pct != null) bits.push(`oynaklik %${{rg.vol20_annual_pct}}`);
    parts.push(`<div class="st ${{cls}}"><b>PIYASA REJIMI &middot; ${{rg.label_tr || rg.label}}</b>
      <p>${{rg.detail_tr || ''}}<br><span class="muted">${{bits.join(' &middot; ')}}</span></p></div>`);
  }}

  // Dogrulama durumu — sistemin en buyuk eksigi, en gorunur yerde
  if (!v.validated) {{
    const p = v.progress_pct || 0;
    // Sayacin kendisi soru birakiyor: ne zaman dolacak, benim bir sey yapmam
    // gerekiyor mu? Ikisinin de cevabi burada olmali.
    const eta = v.eta
      ? `Su anki hizla <b>${{v.eta}}</b> dolar (yaklasik <b>${{v.days_left}}</b> gun).`
      : '';
    parts.push(`<div class="st warn">
      <b>DOGRULANMAMIS</b>
      <p>Parametre agirliklari uzman gorusudur; hicbiri ileri getiriyle test
      edilmedi. Bu skorlar <b>hipotez</b>, tahmin degildir.
      <span class="stbar"><i style="width:${{p}}%"></i></span>
      Dogrulama icin ${{v.needed_snapshots}} anlik goruntu / ${{v.needed_days}} gun gerekiyor —
      su an <b>${{v.snapshots || 0}}</b> goruntu, <b>${{v.span_days || 0}}</b> gun (%${{p}}).
      ${{eta}}</p>
      <p><b>Senin yapman gereken bir sey yok.</b> Sayaci gunluk tarama
      ilerletiyor; her calisma bir anlik goruntu ekliyor. Tek sart bilgisayarin
      gun icinde bir kez acik olmasi.</p>
    </div>`);
  }} else {{
    parts.push(`<div class="st ok"><b>DOGRULANDI</b>
      <p>Agirliklar ${{v.snapshots}} anlik goruntu uzerinden olculdu.</p></div>`);
  }}

  // Ogrenilen model — kendi kendini besleyen dongunun durumu
  if (v.model) {{
    parts.push(`<div class="st ok"><b>OGRENILEN MODEL ETKIN</b>
      <p>Sampiyon <b>${{v.model.name}}</b> &middot; skor agirligi
      <b>${{v.model.weight}}</b> &middot; olculen IC <b>${{v.model.ic}}</b>,
      ICIR <b>${{v.model.icir}}</b>. Agirlik, sizintisiz ileri yuruyuslu
      degerlendirmede olculen beceriyle orantilidir.</p></div>`);
  }} else {{
    // Beklemenin alternatifi varsa soylenmeli: onbellekteki fiyat gecmisinden
    // uretilmis panel, sayac dolmadan mimari denemeye izin verir.
    const pt = v.pretrain;
    const ptLine = pt
      ? `<br>Beklemeden deneyebilirsin: gecmise donuk panel hazir —
         <b>${{pt.snapshots}}</b> goruntu, <b>${{pt.tickers}}</b> hisse
         (${{pt.first_date}} → ${{pt.last_date}})${{pt.partial ? ', uretim suruyor' : ''}}.
         <code>python run.py ml train --pretrain</code>. Bu panel hayatta kalma
         yanliligi tasidigi icin <b>sampiyon uretemez</b>; yalnizca hangi model
         turunun sinyal yakaladigini gosterir.`
      : `<br>Beklemeden denemek icin: <code>python run.py history</code> ile
         onbellekteki fiyat gecmisinden gecmise donuk panel uretilebilir.`;
    parts.push(`<div class="st info"><b>OGRENILEN MODEL YOK</b>
      <p>Model, kanit olmadan skorlamaya <b>katilmaz</b>. Gunluk anlik
      goruntuler yeterli sayiya ulasinca egitim <b>kendiliginden</b> baslar
      (gunluk is her 5 taramada bir dener) ve yalnizca esikleri gecen model
      devreye girer.${{ptLine}}</p></div>`);
  }}

  // Veri tazeligi
  const gen = new Date(DATA.generatedTs);
  const ageH = (Date.now() - gen.getTime()) / 36e5;
  if (ageH > 30) {{
    parts.push(`<div class="st warn"><b>VERI ESKI</b>
      <p>Bu tarama <b>${{Math.floor(ageH/24)}} gun ${{Math.floor(ageH%24)}} saat</b> once
      calisti. Gunluk is calismamis olabilir:
      <code>python run.py daily</code></p></div>`);
  }}

  // Evren kapsamasi — donusumlu tarama ve/veya hiz siniri
  const d = DATA.diagnostics || {{}};
  const partial = (d.fetch_success_rate != null && d.fetch_success_rate < 0.85);
  if (partial || d.batched) {{
    const bits = [];
    if (d.batched) {{
      bits.push(`Evren <b>${{d.universe_full}}</b> sembol; bu turda en uzun suredir
        taranmamis <b>${{d.universe_size}}</b> tanesi tarandi (donusumlu tarama).`);
    }}
    if (partial) {{
      bits.push(`Denenenlerin <b>%${{Math.round(d.fetch_success_rate*100)}}</b>'i
        cekilebildi (hiz siniri).`);
    }}
    if (d.universe_coverage_pct != null) {{
      bits.push(`Evrenin <b>%${{d.universe_coverage_pct}}</b>'i bugune kadar en az
        bir kez tarandi${{d.never_scanned ? `, <b>${{d.never_scanned}}</b> sembol henuz hic` : ''}}.`);
    }}
    parts.push(`<div class="st info"><b>EVREN KAPSAMASI</b><p>${{bits.join(' ')}}
      Siralama yalnizca gorulen kismi kapsar; her calistirmada evren tamamlanir.</p></div>`);
  }}
  document.getElementById('statusBar').innerHTML = parts.join('');
}})();

/* ---------- karne: kagit uzerinde defter ---------- */
(function paperPanel() {{
  const P = DATA.paper;
  if (!P) return;
  const box = document.getElementById('paperBody');
  const sec = document.getElementById('secPaper');

  const pct = v => (v == null ? '&mdash;' : (v > 0 ? '+' : '') + v.toFixed(2) + '%');

  function card(s, title, note) {{
    if (!s || !s.ok) {{
      return `<div class="st info"><b>${{title}}</b><p>${{(s && s.reason) || 'veri yok'}}
        ${{s && s.bekleyen ? ` (${{s.bekleyen}} pozisyon ufkunu bekliyor)` : ''}}</p></div>`;
    }}
    // Anlamlilik: t < 2 ise fark gurultuden ayirt edilemez. Bu satiri
    // gizlemek, karneyi oldugundan guclu gostermek olurdu.
    const tOk = s.t_stat != null && Math.abs(s.t_stat) >= 2;
    const tTxt = s.t_stat == null
      ? 'anlamlilik olculemedi (az kohort)'
      : (tOk ? `t = ${{s.t_stat}} &rarr; <b>gurultuden ayirt edilebilir</b>`
             : `t = ${{s.t_stat}} &rarr; <b>gurultuden ayirt EDILEMEZ</b>`);
    return `<div class="st ${{tOk && s.excess_pct > 0 ? 'ok' : 'info'}}">
      <b>${{title}}</b>
      <p>${{note || ''}}</p>
      <table class="mini"><tbody>
        <tr><td>Donem</td><td>${{s.first_date}} &rarr; ${{s.last_date}}
            (${{s.cohorts}} kohort, ${{s.positions}} pozisyon)</td></tr>
        <tr><td>Ortalama getiri</td><td><b>${{pct(s.mean_pct)}}</b>
            &nbsp; SPY ${{pct(s.bench_mean_pct)}}</td></tr>
        <tr><td>Endeks farki</td><td><b>${{pct(s.excess_pct)}}</b>
            &nbsp;(pozisyonlarin %${{s.excess_positive_pct}}'i endeksi yendi)</td></tr>
        <tr><td>Isabet</td><td>%${{s.hit_rate_pct}} pozitif &middot;
            kazanan ort ${{pct(s.avg_win_pct)}} &middot;
            kaybeden ort ${{pct(s.avg_loss_pct)}}</td></tr>
        <tr><td>Dagilim</td><td>en iyi ${{pct(s.best_pct)}} &middot;
            en kotu ${{pct(s.worst_pct)}} &middot; std %${{s.std_pct}}</td></tr>
        <tr><td>Anlamlilik</td><td>${{tTxt}}</td></tr>
        ${{s.delisted_closed ? `<tr><td>Kote disi kapanan</td><td>${{s.delisted_closed}}
            pozisyon son bilinen fiyattan kapatildi</td></tr>` : ''}}
      </tbody></table>
      ${{s.bias_warning ? `<p class="muted"><b>Uyari:</b> ${{s.bias_warning}}</p>` : ''}}
    </div>`;
  }}

  const html = [
    card(P.live, 'GERCEK TARAMALAR',
         '28 parametrenin tamami, cezalar dahil. Yanlilik tasimaz; ama sistem yeni oldugu icin az kohort var.'),
    card(P.panel, 'GECMISE DONUK PANEL (ust sinir)',
         'Onbellekteki fiyat gecmisinden uretildi. Uzun donem gorunur ama sonuc oldugundan iyidir.'),
  ];
  box.innerHTML = html.join('');
  sec.hidden = false;
}})();

/* ---------- parametre IC tablosu ---------- */
(function icPanel() {{
  const F = DATA.factorIC;
  if (!F || !F.factors || !F.factors.length) return;
  const rows = F.factors.slice().sort((a, b) => (b.ic_mean || 0) - (a.ic_mean || 0));

  /* Zaman/rejim kirilimi ayri bir dosyadan gelir ve OLMAYABILIR (eski kurulum,
     ya da olcum henuz kosmamis). Sutunlar bu yuzden kosullu ekleniyor; veri
     yoksa tablo eskisi gibi gorunur. */
  const T = DATA.factorTime;
  const tmap = {{}};
  ((T && T.factors) || []).forEach(r => {{ tmap[r.factor] = r; }});
  const hasT = Object.keys(tmap).length > 0;
  const num = (v, d) => (v == null || !isFinite(v)) ? '—' : v.toFixed(d);
  const rejimHucre = r => {{
    const g = (r && r.by_regime) || {{}};
    const ks = Object.keys(g);
    if (!ks.length) return '—';
    return ks.sort().map(k => `${{k.slice(0, 3)}} ${{num(g[k].ic, 3)}}`).join(' · ');
  }};

  const verdict = ic => {{
    const a = Math.abs(ic || 0);
    if (a >= 0.10) return ['cok iyi', 'ok'];
    if (a >= 0.05) return ['iyi', 'ok'];
    if (a >= 0.03) return ['zayif ama kullanilabilir', 'info'];
    return ['gurultu', 'warn'];
  }};

  const body = rows.map(r => {{
    const [txt, cls] = verdict(r.ic_mean);
    const neg = (r.ic_mean || 0) < 0;
    const z = tmap[r.factor];
    const esik = (T && T.t_threshold) || 2;
    const gecti = z && z.t_nw != null && Math.abs(z.t_nw) >= esik;
    const ekstra = hasT ? `
      <td class="num">${{z ? (gecti ? '<b>' + num(z.t_nw, 2) + '</b>' : num(z.t_nw, 2)) : '—'}}</td>
      <td class="num">${{z ? num(z.ic_ilk_yari, 3) + ' &rarr; ' + num(z.ic_son_yari, 3) : '—'}}</td>
      <td class="num">${{rejimHucre(z)}}</td>` : '';
    return `<tr class="ic-${{cls}}">
      <td>${{r.factor}}</td>
      <td class="num">${{(r.ic_mean == null ? '—' : r.ic_mean.toFixed(4))}}</td>
      <td class="num">${{(r.icir == null ? '—' : r.icir.toFixed(2))}}</td>
      <td class="num">${{r.periods}}</td>${{ekstra}}
      <td class="num">${{r.weight == null ? '—' : r.weight}}</td>
      <td>${{neg ? '<b>ters yonde</b> &mdash; ' : ''}}${{txt}}</td>
    </tr>`;
  }}).join('');

  const basliklar = hasT
    ? `<th class="num" title="Ortusen etiketler icin duzeltilmis t degeri (Newey-West). |t| ${{(T.t_threshold || 2)}} ve uzeri: gurultuden ayirt edilebilir.">t (duz)</th>
       <th class="num" title="Donemin ilk ve ikinci yarisindaki IC ortalamasi. Isaret degisiyorsa parametre kararsizdir.">1.yari &rarr; 2.yari</th>
       <th class="num" title="Piyasa rejimine gore IC. Rejim, endeksin kendi fiyat gecmisinden ayni kuralla uretilir.">Rejime gore</th>`
    : '';

  const notlar = ((T && T.notes_tr) || []).map(n => `<li>${{n}}</li>`).join('');
  const zamanNot = notlar
    ? `<div class="panel-sub"><b>Zaman ve rejim okumasi</b><ul class="tight">${{notlar}}</ul></div>`
    : '';

  document.getElementById('icBody').innerHTML = `
    <p class="muted">Kaynak: <b>${{F.source === 'panel' ? 'gecmise donuk panel' : 'gercek taramalar'}}</b>
      &middot; ufuk ${{F.horizon}} islem gunu &middot; ${{F.dates}} tarih,
      ${{F.labeled_rows.toLocaleString('tr-TR')}} etiketli satir.<br>${{F.note_tr || ''}}</p>
    <div class="tbl-scroll"><table class="mini wide"><thead><tr>
      <th>Parametre</th><th class="num">IC</th><th class="num">ICIR</th>
      <th class="num">Donem</th>${{basliklar}}<th class="num">Agirlik</th><th>Yorum</th>
    </tr></thead><tbody>${{body}}</tbody></table></div>
    ${{zamanNot}}
    <p class="muted">Hicbir agirlik bu tabloya bakilarak <b>otomatik degistirilmez</b>.
      Olcum bir oneridir; degisiklik senin kararindir.</p>`;
  document.getElementById('secIC').hidden = false;
}})();

/* ---------- sistem sagligi ---------- */
(function healthPanel() {{
  const H = DATA.health || {{}};
  const p = v => (v == null ? '—' : '%' + Math.round(v * 100));
  const rows = [
    ['Bu tarama', DATA.generatedAt || '—'],
    ['Evren', `${{H.universeFull ?? '—'}} sembol &middot; ${{H.scored ?? '—'}} skorlandi`],
    ['Cekim basarisi', `${{p(H.fetchRate)}}` +
      (H.fallbackUsed ? ` (Yahoo ${{p(H.yahooRate)}} + yedek kaynaktan ${{H.fallbackUsed}})` : '')],
    ['Hiz siniri', `${{H.rateLimited ?? 0}} ret${{H.aborted ? ' &middot; <b>devre kesici devreye girdi</b>' : ''}}`],
    ['Evren kapsamasi', H.coveragePct != null ? `%${{H.coveragePct}} en az bir kez tarandi` : '—'],
  ];
  /* Veri tazeligi: siradaki her satir bugunun fiyatindan gelmiyor. Bunu
     soylemek zorunlu — aksi halde iki haftalik fiyattan uretilmis bir sira
     "bugunun siralamasi" gibi okunur. */
  if (H.dataAge) {{
    const a = H.dataAge;
    const uyari = a.stale_over_7d > 0
      ? ` &middot; <b>${{a.stale_over_7d}} satir 7+ gunluk veriden</b>` : '';
    rows.push(['Veri tazeligi',
      `${{a.fresh_today}} hisse bugun cekildi &middot; ${{a.stale_over_3d}} hisse
       3+ gunluk &middot; medyan ${{a.median_days}} gun${{uyari}}`]);
  }}
  if (H.fundamentals && H.fundamentals.snapshots) {{
    const f = H.fundamentals;
    rows.push(['Temel veri arsivi',
      `${{f.snapshots}} gun &middot; ${{f.rows_last}} hisse &middot; ${{f.mb}} MB
       (${{f.first_date}} &rarr; ${{f.last_date}})`]);
  }}
  if (H.delisting) {{
    rows.push(['Kote disi takibi',
      `${{H.delisting.confirmed}} kesinlesti, ${{H.delisting.pending}} izleniyor
       (${{H.delisting.confirm_days}} gun kurali)`]);
  }}
  document.getElementById('healthBody').innerHTML =
    `<table class="mini"><tbody>` +
    rows.map(r => `<tr><td>${{r[0]}}</td><td>${{r[1]}}</td></tr>`).join('') +
    `</tbody></table>`;
  document.getElementById('secHealth').hidden = false;
}})();

/* ---------- gunluk degisim ozeti ---------- */
(function deltaNote() {{
  const el = document.getElementById('deltaNote');
  const dl = D.deltas || {{}};
  const pin = DATA.rows.filter(r => r.pinned).length;
  const pinTxt = pin
    ? ` <span style="color:var(--crimson)">★ ${{pin}} hisse izleme listenden</span> — `
      + `siralamada geriye dusseler bile listede kalirlar.`
    : '';
  if (!dl.compared_to) {{
    el.innerHTML = 'Bu ilk tarama; karsilastirilacak onceki gun yok. '
      + 'Yarin tekrar calistirdiginda hangi hisselerin girdigi ve siralamanin '
      + 'nasil degistigi burada gorunecek.' + pinTxt;
    return;
  }}
  el.innerHTML = `<b>${{dl.compared_to}}</b> tarihli taramaya gore: `
    + `<b style="color:var(--crimson)">${{dl.new_count}}</b> yeni giris, `
    + `<b style="color:var(--good)">${{dl.moved_up}}</b> yukari, `
    + `<b style="color:var(--bad)">${{dl.moved_down}}</b> asagi.` + pinTxt;
}})();

/* ---------- lejant ---------- */
function renderLegend() {{
  document.getElementById('legend').innerHTML = CATS.map(c =>
    `<button data-c="${{c.id}}" aria-pressed="${{!off.has(c.id)}}">
       <span class="swatch" style="background:${{c.color}}"></span>
       <span class="gl" style="color:${{c.color}}">${{c.glyph}}</span>${{c.label}}</button>`).join('');
  document.querySelectorAll('.legend button').forEach(b => b.onclick = () => {{
    const id = b.dataset.c;
    off.has(id) ? off.delete(id) : off.add(id);
    renderLegend(); renderBars();
  }});
}}

/* ---------- ipucu ---------- */
const tt = document.getElementById('tt');
function showTip(html, e) {{
  tt.innerHTML = html; tt.classList.add('on');
  const r = tt.getBoundingClientRect();
  let x = e.clientX + 16, y = e.clientY + 16;
  if (x + r.width > innerWidth - 10) x = e.clientX - r.width - 16;
  if (y + r.height > innerHeight - 10) y = e.clientY - r.height - 16;
  tt.style.left = x + 'px'; tt.style.top = y + 'px';
}}
const hideTip = () => tt.classList.remove('on');

/* Gunluk tipik oynama — puanin yanina belirsizlik bandi olarak yazilir.
   Denetim bulgusu K2: tepe bolgesindeki puan farklari bu bandin altinda
   kaldigi icin "1. sira" ile "15. sira" ayrimi guvenilir degil. */
const NOISE = (DATA.noise && DATA.noise.available) ? DATA.noise.median_abs_change : null;
function noiseBand() {{
  return NOISE ? `<em class="pm">±${{NOISE.toFixed(1)}}</em>` : '';
}}

/* Onceki gune gore hareket rozeti. rankChange POZITIF = yukari cikti. */
function deltaBadge(r) {{
  if (r.isNew) return '<i class="dlt new" title="Listeye bugun girdi">YENI</i>';
  if (r.rankChange == null || r.rankChange === 0) return '';
  const up = r.rankChange > 0;
  return `<i class="dlt ${{up ? 'up' : 'dn'}}" title="Onceki tarama: ${{r.prevRank}}. sira">`
       + `${{up ? '▲' : '▼'}}${{Math.abs(r.rankChange)}}</i>`;
}}

/* ---------- ana grafik ---------- */
function renderBars() {{
  const rows = DATA.rows;
  const max = Math.max(...rows.map(r => r.total || 0), 1);
  document.getElementById('bars').innerHTML = rows.map(r => {{
    const shown = CATS.filter(c => !off.has(c.id));
    const segs = shown.map(c => {{
      const v = r.contrib[c.id] || 0;
      if (v <= 0.001) return '';
      return `<i class="seg" style="width:${{(v/max)*100}}%;background:${{c.color}}"
                 data-t="${{r.ticker}}" data-c="${{c.id}}"></i>`;
    }}).join('');
    const visTotal = shown.reduce((a,c)=>a+(r.contrib[c.id]||0),0);
    const pen = r.penalty < 0 ? `<span class="flag pen">${{r.penalty}}</span>` : '';
    const lc = r.lowConfidence ? '<span class="flag">DUSUK VERI</span>' : '';
    return `<div class="row${{r.pinned ? ' pinned' : ''}}">
      <div class="rk">${{r.rank}}${{deltaBadge(r)}}</div>
      <div class="lbl"><div class="tk">${{r.pinned ? '<span class="pin">★</span>' : ''}}${{r.ticker}}${{pen}}${{lc}}</div>
        <div class="nm">${{r.name}}</div></div>
      <div class="px"><b>${{fmtNum(r.price,2)}}</b><small>${{r.currency}}</small></div>
      <div class="bar">${{segs}}</div>
      <div class="val">${{fmtNum(visTotal,1)}}${{noiseBand()}}
        <small>ilk %${{fmtNum(r.percentile,1)}}</small></div>
      <div><button class="addbtn" data-add="${{r.ticker}}">+ EKLE</button></div></div>`;
  }}).join('');

  document.querySelectorAll('#bars .addbtn').forEach(b => b.onclick = ev => {{
    ev.stopPropagation();
    toggleBasket(b.dataset.add);
  }});
  syncAddButtons();

  document.querySelectorAll('.seg').forEach(s => {{
    s.onmousemove = e => {{
      const r = DATA.rows.find(x => x.ticker === s.dataset.t);
      const c = CATS.find(x => x.id === s.dataset.c);
      const fs = r.factors.filter(f => f.category === c.id && f.available)
        .sort((a,b) => (b.contribution||0)-(a.contribution||0));
      showTip(`<b class="tt-h">${{r.ticker}} · ${{c.label}}</b>` +
        `<div class="tt-row"><span>Kategori katkisi</span>` +
        `<b>${{fmtNum(r.contrib[c.id],2)}} / ${{fmtNum(r.total,1)}}</b></div>` +
        fs.map(f => `<div class="tt-row"><span>${{f.name.slice(0,30)}}</span>` +
          `<span>${{fmtNum(f.score,0)}} × ${{fmtNum(f.weight,1)}}</span></div>`).join(''), e);
    }};
    s.onmouseleave = hideTip;
  }});
}}

/* ---------- agirlik listesi ---------- */
(function weights() {{
  const af = D.active_factors || [];
  const max = Math.max(...af.map(f => f.weight || 0), 1);
  document.getElementById('weights').innerHTML = af.map((f,i) => {{
    const c = catOf(f.category);
    return `<div class="wrow">
      <div class="wi">${{ROMAN[i] || (i+1)}}</div>
      <div class="wn">${{f.name_tr}}
        <small style="color:${{c.color}}">${{c.glyph}} ${{c.label}} · KAPSAMA %${{Math.round((f.coverage||0)*100)}}</small></div>
      <div class="wv">${{fmtNum(f.weight,1)}}</div>
      <div class="wbar"><i style="width:${{(f.weight/max)*100}}%;background:${{c.color}}"></i></div>
    </div>`;
  }}).join('');

  const dis = D.auto_disabled || [];
  document.getElementById('disabled').innerHTML = dis.length
    ? '<div style="margin-top:22px">' + dis.map(d =>
        `<div class="note"><b>DEVRE DISI · ${{d.name_tr}}</b><br>${{d.reason_tr}}</div>`).join('') + '</div>'
    : '<div class="note" style="margin-top:22px">Tum parametreler yeterli veri kapsamina sahip.</div>';
}})();

/* ---------- tablo ---------- */
let sortK = 'rank', sortAsc = true;
const getV = (r,k) => ({{r1:r.returns['1m'], r3:r.returns['3m'], r12:r.returns['12m']}})[k] ?? r[k];

function renderTable() {{
  const q = document.getElementById('q').value.trim().toUpperCase();
  const sec = document.getElementById('sec').value;
  let rows = DATA.rows.filter(r =>
    (!q || r.ticker.includes(q) || (r.name||'').toUpperCase().includes(q)) &&
    (!sec || r.sector === sec));

  rows.sort((a,b) => {{
    const x = getV(a,sortK), y = getV(b,sortK);
    if (x == null) return 1;
    if (y == null) return -1;
    const c = typeof x === 'string' ? x.localeCompare(y) : x - y;
    return sortAsc ? c : -c;
  }});

  document.getElementById('cnt').textContent = `${{rows.length}} / ${{DATA.rows.length}} kayit`;
  const cls = v => v == null ? '' : (v >= 0 ? 'pos' : 'neg');
  document.querySelector('#tbl tbody').innerHTML = rows.map(r => `
    <tr data-t="${{r.ticker}}"${{r.pinned ? ' class="pinrow"' : ''}}>
      <td>${{r.rank}}${{deltaBadge(r)}}</td>
      <td>${{r.pinned ? '<span class="pin">★</span>' : ''}}${{r.ticker}}</td>
      <td>${{r.sector || '—'}}</td>
      <td>${{fmtNum(r.price,2)}}</td>
      <td class="big">${{fmtNum(r.total,1)}}</td>
      <td>${{fmtNum(r.base,1)}}</td>
      <td class="${{r.penalty<0?'neg':''}}">${{r.penalty ? fmtNum(r.penalty,0) : '0'}}</td>
      <td class="${{cls(r.returns['1m'])}}">${{fmtPct(r.returns['1m'])}}</td>
      <td class="${{cls(r.returns['3m'])}}">${{fmtPct(r.returns['3m'])}}</td>
      <td class="${{cls(r.returns['12m'])}}">${{fmtPct(r.returns['12m'])}}</td>
      <td>${{fmtCap(r.dollarVolume)}}</td>
      <td>${{fmtCap(r.maxPosition)}}</td>
      <td>${{fmtNum((r.coverage||0)*100,0)}}%</td>
      <td><button class="addbtn" data-add="${{r.ticker}}">+ EKLE</button></td>
    </tr>`).join('');

  document.querySelectorAll('#tbl tbody tr').forEach(tr => tr.onclick = ev => {{
    // Ekle butonuna basildiysa satir detayini acma
    if (ev.target.closest('.addbtn')) return;
    toggleDetail(tr);
  }});
  document.querySelectorAll('#tbl .addbtn').forEach(b => b.onclick = ev => {{
    ev.stopPropagation();
    toggleBasket(b.dataset.add);
  }});
  syncAddButtons();
}}

function toggleDetail(tr) {{
  const nxt = tr.nextElementSibling;
  if (nxt && nxt.classList.contains('det')) {{ nxt.remove(); return; }}
  document.querySelectorAll('tr.det').forEach(e => e.remove());
  const r = DATA.rows.find(x => x.ticker === tr.dataset.t);

  const items = r.factors.map(f => {{
    const c = catOf(f.category);
    return `<div class="fitem ${{f.available?'':'na'}}"
        title="Agirlik ${{fmtNum(f.weight,1)}} · katki ${{fmtNum(f.contribution,2)}}">
      <span class="swatch" style="background:${{c.color}}"></span>
      <div class="fn"><span>${{f.name}}</span>
        <small>${{c.glyph}} AGIRLIK ${{fmtNum(f.weight,1)}}${{f.band ? ' · '+f.band.toUpperCase() : ''}}</small></div>
      <div class="fs">${{f.available ? fmtNum(f.score,0) : 'YOK'}}</div></div>`;
  }}).join('');

  const pen = r.penaltiesHit.length
    ? `<div class="note" style="margin-top:16px"><b>CEZALAR</b><br>` +
      r.penaltiesHit.map(p => `${{p.name_tr}} (${{p.points}})`).join(' · ') + '</div>' : '';

  const row = document.createElement('tr');
  row.className = 'det';
  row.innerHTML = `<td colspan="14">
    <h3>${{r.ticker}} · ${{r.name}}</h3>
    <div class="dm">${{r.sector}} — PIYASA DEGERI ${{fmtCap(r.marketCap)}} ${{r.currency}}
      — RSI ${{fmtNum(r.rsi,0)}}${{r.daysToEarnings!=null ? ' — BILANCOYA '+r.daysToEarnings+' GUN' : ''}}
      — KAPSAMA %${{fmtNum((r.coverage||0)*100,0)}} — DILIM ILK %${{fmtNum(r.percentile,1)}}</div>
    <div class="note" style="margin:0 0 16px">
      <b>ISLENEBILIRLIK</b>Gunluk hacim <b>${{fmtCap(r.dollarVolume)}} ${{r.currency}}</b>
      · devir hizi <b>${{r.turnover!=null ? (r.turnover*100).toFixed(2)+'%' : '—'}}</b>
      · piyasayi bozmadan girilebilecek kaba ust sinir
      <b>${{fmtCap(r.maxPosition)}} ${{r.currency}}</b> (gunluk hacmin %5'i).
    </div>
    <div class="fgrid">${{items}}</div>${{pen}}
    <p class="sec-note" style="margin:16px 0 0">Puanlar 0—100 arasi, evren icindeki goreli
      konuma gore. "YOK" = veri bulunamadi; agirligi diger parametrelere dagitildi.</p></td>`;
  tr.after(row);
}}

/* ---------- baglama ---------- */
document.querySelectorAll('#tbl th').forEach(th => th.onclick = () => {{
  const k = th.dataset.k;
  sortAsc = sortK === k ? !sortAsc : (k === 'rank' || k === 'ticker' || k === 'sector');
  sortK = k; renderTable();
}});
document.getElementById('q').oninput = renderTable;
document.getElementById('sec').onchange = renderTable;
document.getElementById('sec').innerHTML = '<option value="">Tum sektorler</option>' +
  [...new Set(DATA.rows.map(r => r.sector).filter(Boolean))].sort()
    .map(s => `<option>${{s}}</option>`).join('');

/* Asistan ve sepet kodu ILK yuklenir. Sebep: renderTable() -> syncAddButtons()
   zinciri `basket` degiskenini okuyor; `let` ile bildirilen bir degiskene
   bildiriminden once erisilirse ReferenceError firlar ve script komple durur
   (o zaman sohbet formu da baglanmaz). Ilk cizimler bu yuzden en sona alindi. */
__ASSISTANT_JS__

renderLegend(); renderBars(); renderTable();
</script>"""


def _inject_assistant(html: str) -> str:
    """Asistan ve izleme-listesi arayuzunu sayfaya yerlestirir.

    Duz .replace kullaniliyor cunku bu parcalarda JS suslu parantezleri var;
    f-string icine konsalar hepsinin kacislanmasi gerekirdi.
    """
    from . import assistant_ui as ui

    html = html.replace("__ASSISTANT_HTML__", ui.HTML)
    html = html.replace("__ASSISTANT_JS__", ui.JS)
    # Asistan stillerini mevcut <style> blogunun sonuna ekle
    return html.replace("</style>", ui.CSS + "\n</style>", 1)


def write_html(df: pd.DataFrame, diagnostics: dict, path: Path,
               top_n: int = 40, title: str = "SIGMA / HISSE SIRALAMA MOTORU") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    html = _inject_assistant(build_html(df, diagnostics, top_n, title))
    path.write_text(html, encoding="utf-8")
    return path


def write_csv(df: pd.DataFrame, path: Path) -> Path:
    """Duz CSV — Excel/Sheets icin."""
    flat = []
    for _, r in df.iterrows():
        row = {
            "rank": r["rank"], "ticker": r["ticker"], "name": r.get("name"),
            "sector": r.get("sector"), "price": r.get("price"),
            "market_cap": r.get("market_cap"),
            "total_score": r.get("total_score"), "base_score": r.get("base_score"),
            "penalty": r.get("penalty"), "coverage": r.get("coverage"),
        }
        for k, v in (r.get("returns") or {}).items():
            row[f"return_{k}"] = v
        for fid, fd in (r.get("factors") or {}).items():
            row[f"score__{fid}"] = fd.get("score")
            row[f"raw__{fid}"] = fd.get("raw")
        flat.append(row)

    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(flat).to_csv(path, index=False, encoding="utf-8-sig")
    return path
