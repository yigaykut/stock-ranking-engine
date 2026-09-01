"""Kagit uzerinde portfoy defteri — sistemin kendi karnesi.

NEDEN VAR
---------
Denetim raporunun en kritik bulgusu (K1) suydu: sistemin ise yarayip
yaramadigina dair TEK BIR OLCUM YOK. Ogrenme katmani bunu IC/ICIR ile
olcecek, ama 60 anlik goruntu birikmesini bekliyor -- yani Aralik'i.

Oysa "1 Eylul'de listenin ilk 20'sini alsaydim bugun ne olurdu?" sorusunun
cevabi ilk 21 gunun sonunda vardir ve fiyat verisi zaten diskte duruyor. Bu
modul o soruyu cevaplar.

YONTEM — kohort defteri
-----------------------
Portfoy simulasyonu degil, KOHORT takibi yapilir. Her tarama gunu, o gunun
ilk N hissesi bir "kohort" olarak deftere yazilir. Kohort H islem gunu
tutulur ve kapanir. Boylece:

  * sermaye kisiti, pozisyon boyutu, nakit yonetimi gibi -- olculmek istenen
    seyle ilgisi olmayan -- degiskenler devre disi kalir,
  * her kohort bagimsiz bir OLCUMDUR; 70 kohort, 70 ayri denemedir,
  * sonuc dogrudan siralamanin kalitesini olcer, portfoy yonetimi becerisini
    degil.

Her pozisyon ayrica SPY'a karsi olculur. Ham getiri piyasa yukselirken zaten
pozitif cikar; anlamli olan tek sayi FARKTIR (excess).

IKI KAYNAK, IKI FARKLI GUVEN SEVIYESI
-------------------------------------
  live  : gercek gunluk taramanin ilk N'i. 28 parametrenin tamami, cezalar
          dahil. Az sayida gun var (sistem yeni) ama YANLILIK TASIMAZ.
  panel : geriye donuk uretilmis panel. 73 tarih, 11 ay. Ama:
            - yalnizca 11 FIYAT faktoru var (temel veri gecmise donuk
              bilinemiyor), yani gercek siralamanin bir yaklasikligidir,
            - ceza kurallari yok,
            - HAYATTA KALMA YANLILIGI tasir: onbellek yalnizca bugun kote
              olan sirketleri icerir, batanlar hic gorunmez.
          Bu yuzden panel sonucu OLDUGUNDAN IYI cikar. Panoda da, burada da
          bu acikca yazilir; gizlenirse ozellik faydadan cok zarar verir.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "data" / "paper"
COHORTS = PAPER / "kohortlar.csv"
RESULTS = PAPER / "sonuclar.csv"
SUMMARY = PAPER / "ozet.json"

DEFAULT_TOP_N = 20
HORIZONS = (5, 21, 63)
BENCHMARK = "SPY"

COHORT_COLS = ["snapshot_date", "ticker", "rank", "score", "sector", "price",
               "source", "top_n"]


# =============================================================================
#  1) DEFTERE YAZMA
# =============================================================================
def _load(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _append(path: Path, rows: pd.DataFrame, keys: list[str]) -> int:
    """Satirlari ekler; ayni anahtarli eski satirlari YENISIYLE degistirir."""
    if rows.empty:
        return 0
    PAPER.mkdir(parents=True, exist_ok=True)
    old = _load(path)
    if not old.empty:
        mask = pd.Series(True, index=old.index)
        new_keys = set(map(tuple, rows[keys].astype(str).values))
        old_keys = list(map(tuple, old[keys].astype(str).values))
        mask = pd.Series([k not in new_keys for k in old_keys], index=old.index)
        rows = pd.concat([old[mask], rows], ignore_index=True, sort=False)
    rows = rows.sort_values(["snapshot_date", "rank"], na_position="last")
    rows.to_csv(path, index=False, encoding="utf-8")
    return len(rows)


def record_live(result_df: pd.DataFrame, top_n: int = DEFAULT_TOP_N,
                date: str | None = None) -> dict:
    """Bugunun ilk N'ini deftere yazar. cmd_scan sonunda cagrilir."""
    if result_df is None or result_df.empty:
        return {"ok": False, "reason": "bos siralama"}

    date = date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    df = result_df.head(top_n).copy()
    rows = pd.DataFrame({
        "snapshot_date": date,
        "ticker": df["ticker"].astype(str).values,
        "rank": range(1, len(df) + 1),
        "score": pd.to_numeric(df.get("total_score"), errors="coerce").values,
        "sector": df.get("sector", pd.Series(["Bilinmiyor"] * len(df))).values,
        "price": pd.to_numeric(df.get("price"), errors="coerce").values,
        "source": "live",
        "top_n": top_n,
    })
    total = _append(COHORTS, rows[COHORT_COLS], ["snapshot_date", "ticker", "source"])
    return {"ok": True, "date": date, "added": len(rows), "total_rows": total}


# =============================================================================
#  2) GERIYE DONUK DEFTER
# =============================================================================
def _score_panel_snapshot(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Geriye donuk panelin bir gununu, GERCEK skorlayiciyla siralar.

    Ayri bir skorlama kopyasi yazilmadi: yuzdelik siralama, sektor
    notrlestirmesi, kume butcesi ve agirlik yeniden dagitimi burada da aynen
    calissin diye scoring.Scorer'in kendisi kullaniliyor. Panelde bulunmayan
    faktorler (temel veri) sutun olmadigi icin kendiliginden devre disi kalir
    ve agirliklari mevcut faktorlere yeniden dagitilir.
    """
    from . import scoring

    raw_cols = [c for c in df.columns if c.startswith("raw_")]
    records = []
    for row in df.itertuples(index=False):
        d = row._asdict()
        raw = {c[4:]: d.get(c) for c in raw_cols}
        records.append({
            "ticker": d["ticker"],
            "ok": True,
            "sector": d.get("sector") or "Bilinmiyor",
            "price": d.get("price"),
            "raw": raw,
            "meta": {},
            "penalties": [],
        })

    scorer = scoring.Scorer(cfg)
    scored, _ = scorer.score(records)
    return scored


def build_from_panel(cfg: dict, top_n: int = DEFAULT_TOP_N,
                     store: Path | None = None, progress: bool = True) -> dict:
    """Geriye donuk panelden kohort defteri uretir (kaynak: 'panel')."""
    store = store or (ROOT / "data" / "backfill_store")
    files = sorted(store.glob("snapshot_*.csv"))
    if not files:
        return {"ok": False, "reason": "gecmise donuk panel yok "
                                       "(once: python run.py history)"}

    out = []
    for i, p in enumerate(files, 1):
        try:
            snap = pd.read_csv(p)
        except Exception:
            continue
        if snap.empty:
            continue
        date = str(snap["snapshot_date"].iloc[0])[:10]
        scored = _score_panel_snapshot(snap, cfg)
        if scored.empty:
            continue
        head = scored.head(top_n)
        out.append(pd.DataFrame({
            "snapshot_date": date,
            "ticker": head["ticker"].astype(str).values,
            "rank": range(1, len(head) + 1),
            "score": pd.to_numeric(head.get("total_score"), errors="coerce").values,
            "sector": head.get("sector", pd.Series(["Bilinmiyor"] * len(head))).values,
            "price": pd.to_numeric(head.get("price"), errors="coerce").values,
            "source": "panel",
            "top_n": top_n,
        }))
        if progress and i % 10 == 0:
            print(f"      {i}/{len(files)} tarih siralandi")

    if not out:
        return {"ok": False, "reason": "panelden siralama uretilemedi"}

    rows = pd.concat(out, ignore_index=True)
    total = _append(COHORTS, rows[COHORT_COLS], ["snapshot_date", "ticker", "source"])
    return {"ok": True, "dates": len(out), "added": len(rows), "total_rows": total}


def build_from_feature_store(top_n: int = DEFAULT_TOP_N) -> dict:
    """Birikmis GERCEK anlik goruntulerden kohort defteri uretir.

    Anlik goruntuler `total_score` ve `rank` sutunlarini zaten tasiyor; yani
    gecmis gunlerin gercek siralamasi elimizde. Defter bu yuzden sistemin
    ilk gunune kadar geriye dogru doldurulabilir.
    """
    from . import ml

    files = sorted(ml.FEATURE_STORE.glob("snapshot_*.csv"))
    if not files:
        return {"ok": False, "reason": "anlik goruntu yok"}

    out = []
    for p in files:
        try:
            snap = pd.read_csv(p)
        except Exception:
            continue
        if snap.empty or "total_score" not in snap.columns:
            continue
        date = p.stem.replace("snapshot_", "")[:10]
        head = (snap.sort_values("total_score", ascending=False, na_position="last")
                    .head(top_n))
        out.append(pd.DataFrame({
            "snapshot_date": date,
            "ticker": head["ticker"].astype(str).values,
            "rank": range(1, len(head) + 1),
            "score": pd.to_numeric(head["total_score"], errors="coerce").values,
            "sector": head.get("sector", pd.Series(["Bilinmiyor"] * len(head))).values,
            "price": pd.to_numeric(head.get("price"), errors="coerce").values,
            "source": "live",
            "top_n": top_n,
        }))

    if not out:
        return {"ok": False, "reason": "anlik goruntulerde skor sutunu yok"}

    rows = pd.concat(out, ignore_index=True)
    total = _append(COHORTS, rows[COHORT_COLS], ["snapshot_date", "ticker", "source"])
    return {"ok": True, "dates": len(out), "added": len(rows), "total_rows": total}


# =============================================================================
#  3) PIYASAYA GORE DEGERLEME
# =============================================================================
def _closes(ticker: str) -> pd.Series | None:
    """Onbellekteki kapanis serisi. AG ISTEGI YAPILMAZ.

    Defter her gun yeniden hesaplanir; her hesapta 2000 hisse icin ag istegi
    atmak hem hiz sinirini yakar hem de gereksizdir: gunluk tarama fiyatlari
    zaten onbellege yaziyor.
    """
    from .providers import yahoo

    b = yahoo.fetch_cached(ticker, "2y", max_age_seconds=30 * 24 * 3600)
    if not b:
        return None
    h = b.get("history")
    if h is None or len(h) < 5 or "Close" not in h:
        return None
    s = pd.to_numeric(h["Close"], errors="coerce").dropna()
    s.index = pd.to_datetime(s.index).tz_localize(None)
    return s if len(s) >= 5 else None


def _bench_closes(symbol: str = BENCHMARK) -> pd.Series | None:
    """Karsilastirma endeksinin kapanis serisi.

    Endeks AYRI bir onbellek alaninda tutuluyor (yahoo_bench), hisselerle ayni
    yerde degil. Tek sembol oldugu icin onbellekte yoksa aga cikmak serbest --
    2000 hisse icin yasak olan sey, bir endeks icin sorun degil.
    """
    from .providers import cache as _cache
    from .providers import yahoo

    hit = _cache.peek("yahoo_bench", f"{symbol}:2y")
    h = hit[0] if hit else None
    if h is None or len(h) < 30:
        h = yahoo.fetch_benchmark(symbol, "2y")
    if h is None or "Close" not in h:
        return None
    s = pd.to_numeric(h["Close"], errors="coerce").dropna()
    s.index = pd.to_datetime(s.index).tz_localize(None)
    return s if len(s) >= 30 else None


def _forward(series: pd.Series, date: pd.Timestamp, horizon: int
             ) -> tuple[float | None, float | None, str | None]:
    """(giris fiyati, cikis fiyati, cikis tarihi) — H ISLEM GUNU sonrasi.

    Takvim gunu degil islem gunu sayilir; aksi halde tatiller ufku kaydirir.
    """
    idx = series.index.searchsorted(date)
    if idx >= len(series):
        return None, None, None
    entry = float(series.iloc[idx])
    j = idx + horizon
    if j >= len(series):
        return entry, None, None            # ufuk henuz dolmadi
    return entry, float(series.iloc[j]), series.index[j].strftime("%Y-%m-%d")


def mark(horizons: tuple[int, ...] = HORIZONS, progress: bool = True) -> dict:
    """Defterdeki her pozisyonun ileriye donuk getirisini hesaplar."""
    coh = _load(COHORTS)
    if coh.empty:
        return {"ok": False, "reason": "defter bos"}

    bench = _bench_closes()
    if bench is None:
        return {"ok": False, "reason": f"{BENCHMARK} fiyat gecmisi onbellekte yok"}

    coh["snapshot_date"] = pd.to_datetime(coh["snapshot_date"])
    rows = []
    tickers = sorted(coh["ticker"].astype(str).unique())
    cache: dict[str, pd.Series | None] = {}

    for i, tk in enumerate(tickers, 1):
        cache[tk] = _closes(tk)
        if progress and i % 200 == 0:
            print(f"      {i}/{len(tickers)} sembol okundu")

    # Kote disi kalanlar: seri kesildigi icin ufuk hicbir zaman "dolmuyor" ve
    # pozisyon sessizce olcumun disinda kaliyor. Iste hayatta kalma yanliligi
    # tam olarak boyle olusur. Bu pozisyonlar son bilinen fiyattan kapatilir.
    try:
        from . import delisting
        delisted = delisting.confirmed()
    except Exception:
        delisted = {}

    missing = 0
    closed_delisted = 0
    for r in coh.itertuples(index=False):
        s = cache.get(str(r.ticker))
        if s is None:
            missing += 1
            continue
        gone = str(r.ticker).upper() in delisted
        rec = {"snapshot_date": r.snapshot_date.strftime("%Y-%m-%d"),
               "ticker": r.ticker, "rank": r.rank, "score": r.score,
               "sector": r.sector, "source": r.source,
               "kote_disi": bool(gone)}
        any_h = False
        for h in horizons:
            e, x, xd = _forward(s, r.snapshot_date, h)
            if x is None and gone and e is not None and len(s):
                x = float(s.iloc[-1])       # son bilinen fiyattan kapat
                xd = s.index[-1].strftime("%Y-%m-%d")
                closed_delisted += 1
            be, bx, _ = _forward(bench, r.snapshot_date, h)
            if e is None or x is None or be is None or bx is None:
                rec[f"ret_{h}"] = None
                rec[f"bench_{h}"] = None
                rec[f"excess_{h}"] = None
                continue
            any_h = True
            ret = x / e - 1.0
            bret = bx / be - 1.0
            rec[f"ret_{h}"] = round(ret, 6)
            rec[f"bench_{h}"] = round(bret, 6)
            rec[f"excess_{h}"] = round(ret - bret, 6)
            rec[f"exit_date_{h}"] = xd
        rec["entry_price"] = round(_forward(s, r.snapshot_date, 0)[0] or 0, 4)
        rec["olgun"] = any_h
        rows.append(rec)

    res = pd.DataFrame(rows)
    PAPER.mkdir(parents=True, exist_ok=True)
    res.to_csv(RESULTS, index=False, encoding="utf-8")
    return {"ok": True, "positions": len(res), "missing_price": missing,
            "closed_delisted": closed_delisted, "horizons": list(horizons)}


# =============================================================================
#  4) OZET
# =============================================================================
def _curve(df: pd.DataFrame, horizon: int) -> list[dict]:
    """Kohort ortalamalarindan birikimli egri.

    Her tarih bir kohorttur ve kohortlar ortusur (her gun yeni bir tane
    aciliyor, her biri H gun tutuluyor). Birikimli carpim, kohort basina
    ortalama getirinin ust uste binmesidir; gercek bir portfoyun egrisi
    degil, siralamanin zaman icinde tutarliligini gosteren bir izdir.
    """
    col = f"ret_{horizon}"
    bcol = f"bench_{horizon}"
    if col not in df.columns:
        return []
    g = df.dropna(subset=[col]).groupby("snapshot_date")
    out, cum, bcum = [], 1.0, 1.0
    for date, part in g:
        m = float(part[col].mean())
        b = float(part[bcol].mean()) if bcol in part else 0.0
        cum *= (1 + m / horizon)          # gunluk esdeger, ortusmeyi telafi eder
        bcum *= (1 + b / horizon)
        out.append({"date": str(date), "sistem": round((cum - 1) * 100, 2),
                    "spy": round((bcum - 1) * 100, 2), "n": int(len(part))})
    return out


def summary(horizon: int = 21, source: str | None = None) -> dict:
    """Defterin karnesi. source: 'live' | 'panel' | None (hepsi)."""
    df = _load(RESULTS)
    if df.empty:
        return {"ok": False, "reason": "defter degerlenmemis (once: mark)"}

    if source:
        df = df[df["source"] == source]
    col, bcol, ecol = f"ret_{horizon}", f"bench_{horizon}", f"excess_{horizon}"
    if col not in df.columns:
        return {"ok": False, "reason": f"{horizon} gunluk ufuk hesaplanmamis"}

    d = df.dropna(subset=[col])
    if d.empty:
        pend = int(df[col].isna().sum())
        return {"ok": False, "reason": f"{horizon} gunluk ufuk henuz dolmadi",
                "bekleyen": pend}

    ret = d[col].astype(float)
    exc = d[ecol].astype(float) if ecol in d else pd.Series(dtype=float)
    wins, losses = ret[ret > 0], ret[ret <= 0]

    by_sector = []
    if "sector" in d.columns:
        gs = d.groupby("sector")[col]
        for sec, part in gs:
            if len(part) < 5:
                continue
            by_sector.append({"sector": str(sec), "n": int(len(part)),
                              "mean_pct": round(100 * float(part.mean()), 2)})
        by_sector.sort(key=lambda x: -x["mean_pct"])

    return {
        "ok": True,
        "horizon": horizon,
        "source": source or "hepsi",
        "cohorts": int(d["snapshot_date"].nunique()),
        "positions": int(len(d)),
        "first_date": str(d["snapshot_date"].min()),
        "last_date": str(d["snapshot_date"].max()),
        "pending": int(df[col].isna().sum()),
        "mean_pct": round(100 * float(ret.mean()), 2),
        "median_pct": round(100 * float(ret.median()), 2),
        "bench_mean_pct": round(100 * float(d[bcol].astype(float).mean()), 2)
        if bcol in d else None,
        "excess_pct": round(100 * float(exc.mean()), 2) if len(exc) else None,
        "excess_positive_pct": round(100 * float((exc > 0).mean()), 1)
        if len(exc) else None,
        "hit_rate_pct": round(100 * float((ret > 0).mean()), 1),
        "avg_win_pct": round(100 * float(wins.mean()), 2) if len(wins) else None,
        "avg_loss_pct": round(100 * float(losses.mean()), 2) if len(losses) else None,
        "payoff": round(abs(float(wins.mean()) / float(losses.mean())), 2)
        if len(wins) and len(losses) and float(losses.mean()) != 0 else None,
        "best_pct": round(100 * float(ret.max()), 2),
        "worst_pct": round(100 * float(ret.min()), 2),
        "std_pct": round(100 * float(ret.std()), 2),
        # Kohort ortalamalarinin t istatistigi: "bu fark sifirdan ayirt
        # edilebilir mi" sorusunun kaba ama durust cevabi.
        "t_stat": _t_stat(d, ecol),
        "delisted_closed": int(d["kote_disi"].sum()) if "kote_disi" in d else 0,
        "by_sector": by_sector[:8],
        "curve": _curve(d, horizon),
        "bias_warning": _bias_warning(source),
    }


def _t_stat(d: pd.DataFrame, ecol: str) -> float | None:
    """Kohort BAZINDA t istatistigi.

    Pozisyon bazinda hesaplanamaz: ayni gunun 20 hissesi bagimsiz degildir
    (hepsi ayni piyasa gununu yasar). Kohort ortalamalari kullanilir; bu da
    ortusen ufuklar yuzunden hala iyimserdir, ama pozisyon bazindan cok daha
    durusttur.
    """
    if ecol not in d.columns:
        return None
    per = d.groupby("snapshot_date")[ecol].mean().astype(float).dropna()
    if len(per) < 3 or per.std() == 0:
        return None
    return round(float(per.mean() / (per.std() / np.sqrt(len(per)))), 2)


def _bias_warning(source: str | None) -> str | None:
    if source == "live":
        return None
    return ("Geriye donuk panel HAYATTA KALMA YANLILIGI tasir (batmis sirketler "
            "onbellekte yok) ve yalnizca 11 fiyat faktoru kullanir; cezalar "
            "yoktur. Sonuc oldugundan IYI cikar, ust sinir olarak okunmalidir.")


def refresh(horizon: int = 21) -> dict:
    """Defteri degerler ve ozeti diske yazar (gunluk isin cagirdigi fonksiyon)."""
    marked = mark(progress=False)
    if not marked.get("ok"):
        return marked
    out = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "marked": marked,
           "live": summary(horizon, "live"),
           "panel": summary(horizon, "panel")}
    PAPER.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str),
                       encoding="utf-8")
    return out


def load_summary() -> dict | None:
    if not SUMMARY.exists():
        return None
    try:
        return json.loads(SUMMARY.read_text(encoding="utf-8"))
    except Exception:
        return None
