"""GECMISE DONUK ANLIK GORUNTU URETIMI — ogrenmenin baslangic sermayesi.

Sorun
-----
Feature store gunde bir satir buyur. Egitim kapisi 60 anlik goruntu ve 120
gunluk aralik ister; bu, sifirdan baslayan bir sistemde ~4 ay demektir. Bu
sure boyunca model hic egitilemez, dolayisiyla dizi (GRU) mimarisinin ise
yarayip yaramadigi bile olculemez.

Cozum
-----
Onbellekte her hisse icin 2 yillik gunluk OHLCV zaten var. Fiyattan turetilen
faktorler GECMISTEKI HERHANGI BIR GUN icin yeniden hesaplanabilir: seriyi o
gune kadar kes, ayni fonksiyonu cagir. Boylece bir yillik gecmis anlik
goruntu birkac dakikada uretilir.

Kritik ayrim: BURADA CANLI SISTEMDEN FARKLI BIR FORMUL KULLANILMAZ. Ayni
factors.f_* fonksiyonlari, kesilmis bir DataFrame ile cagrilir. Ayri bir
"vektorize" surum yazmak hizli olurdu ama egitim ile canli tahmin arasinda
sessiz bir formul farki dogururdu — modelin ogrendigi ozellik ile gordugu
ozellik ayni olmazdi. Hiz icin dogruluk feda edilmiyor.

NEYIN DISARIDA BIRAKILDIGI ve NEDEN
-----------------------------------
Yalnizca fiyat/hacimden turetilen faktorler uretilir. Temel veriler (F/K,
analist notu, EPS revizyonu, kisa pozisyon orani, kurumsal sahiplik) Yahoo
tarafindan yalnizca BUGUNKU haliyle veriliyor — gecmisteki degerleri yok.
Bunlari geriye tasimak, "sirketin bugun bilinen karliligini bir yil onceki
gune yazmak" olurdu; bu klasik gelecege bakis (look-ahead) hatasidir ve
modeli gercek disi basarili gosterir. Bu yuzden hic dahil edilmiyorlar.

YANLILIK UYARISI — durustce
---------------------------
1. HAYATTA KALMA YANLILIGI. Onbellekteki evren BUGUN kote olan hisselerden
   olusur. Gecen yil ici cokup kote disi kalanlar burada yok. Bu, gecmise
   donuk panelde olculen basariyi YUKARI yanli yapar.
2. EVREN YANLILIGI. Tarama donusumlu oldugu icin onbellekteki hisseler
   evrenin rastgele bir ornegi degil, en son taranan dilimidir.
3. Bu veri, modelin ON EGITIMI ve mimari secimi icindir. Nihai karar —
   modelin skorlamaya girip girmeyecegi — GERCEK ileriye donuk anlik
   goruntularle verilmelidir. Bu yuzden ayri bir depoya yazilir ve sampiyon
   secimi varsayilan olarak buradan yapilmaz.
"""
from __future__ import annotations

import json
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BACKFILL_STORE = ROOT / "data" / "backfill_store"

# Yalnizca OHLCV'den turetilen, gecmise donuk yeniden hesaplanabilir faktorler.
PIT_FACTORS = (
    "price_momentum_12_1",
    "relative_strength",
    "trend_structure",
    "breakout_setup",
    "stage2_breakout",
    "chart_position",
    "momentum_persistence",
    "volume_accumulation",
    "risk_drawdown",
    "nominal_price_fit",
    "technical_oscillators",
)

# En uzun geriye bakis f_price_momentum_12_1'de: 260 bar.
MIN_BARS = 260


def _num(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


# =============================================================================
#  Tek bir (hisse, tarih) satiri
# =============================================================================
def point_in_time_row(ticker: str, df: pd.DataFrame, bench_close: pd.Series | None,
                      as_of: pd.Timestamp, sector: str = "Bilinmiyor",
                      min_bars: int = MIN_BARS) -> dict | None:
    """Seriyi `as_of` gunune kadar keser ve o gun BILINEBILIR faktorleri uretir.

    Kesme islemi katidir: `as_of` gununun kapanisi dahil, sonrasi haric.
    Canli taramada gunun yarim bari atildigi gibi burada da tamamlanmis
    barlarla calisilir.
    """
    from . import factors as fx
    from . import investing_summary

    d = df.loc[df.index <= as_of]
    if len(d) < min_bars:
        return None

    close = d["Close"]
    bench = None
    if bench_close is not None:
        bench = bench_close.loc[bench_close.index <= as_of]
        if len(bench) < 70:
            bench = None

    # bundle={} bilincli: temel veri gecmise donuk bilinemez. Bu iki faktor
    # eksik bundle ile de fiyat bileseniyle calisir (beta / analist hedefi
    # olmadan), digerleri zaten bundle kullanmiyor.
    empty: dict = {}

    raw: dict[str, float | None] = {
        "price_momentum_12_1": _num(fx.f_price_momentum_12_1(close)[0]),
        "relative_strength": _num(fx.f_relative_strength(close, bench)[0]),
        "trend_structure": _num(fx.f_trend_structure(d)[0]),
        "breakout_setup": _num(fx.f_breakout_setup(d)[0]),
        "stage2_breakout": _num(fx.f_stage2_breakout(d)[0]),
        "chart_position": _num(fx.f_chart_position(d, empty)[0]),
        "momentum_persistence": _num(fx.f_momentum_persistence(d)[0]),
        "volume_accumulation": _num(fx.f_volume_accumulation(d)[0]),
        "risk_drawdown": _num(fx.f_risk_drawdown(d, empty)[0]),
        "nominal_price_fit": _num(fx.f_nominal_price_fit(d)[0]),
    }

    tech = investing_summary.compute(d)
    raw["technical_oscillators"] = _num(tech.get("score")) if tech.get("available") else None

    row: dict[str, Any] = {
        "snapshot_date": as_of.strftime("%Y-%m-%d"),
        "ticker": ticker,
        "sector": sector,
        "price": _num(close.iloc[-1]),
        "bars_used": len(d),
    }
    for fid in PIT_FACTORS:
        v = raw.get(fid)
        row[f"raw_{fid}"] = v
        row[f"has_{fid}"] = 1 if v is not None else 0

    # Fiyattan turetilen ek olcumler (canli tarafta da mevcut)
    if "Volume" in d:
        dv = (d["Close"] * d["Volume"]).tail(30)
        row["raw_dollar_volume_30d"] = _num(dv.mean())
    return row


# =============================================================================
#  Islemci havuzu icin isci
# =============================================================================
_BENCH: pd.Series | None = None
_DATES: list[pd.Timestamp] = []
_MIN_BARS = MIN_BARS


def _init(bench: pd.Series | None, dates: list[pd.Timestamp], min_bars: int) -> None:
    global _BENCH, _DATES, _MIN_BARS
    _BENCH, _DATES, _MIN_BARS = bench, dates, min_bars


def _worker(ticker: str) -> list[dict]:
    """Bir hisse icin tum tarih izgarasindaki satirlari uretir."""
    from .providers import yahoo

    try:
        bundle = yahoo.fetch_cached(ticker, "2y", max_age_seconds=365 * 24 * 3600)
    except Exception:
        return []
    if not bundle:
        return []

    hist = bundle.get("history")
    if not isinstance(hist, pd.DataFrame) or "Close" not in hist:
        return []

    df = hist.dropna(subset=["Close"]).copy()
    try:
        df.index = pd.to_datetime(df.index).tz_localize(None)
    except (TypeError, ValueError):
        df.index = pd.to_datetime(df.index)
    if len(df) < _MIN_BARS:
        return []

    sector = ((bundle.get("info") or {}).get("sector")) or "Bilinmiyor"

    out = []
    for d in _DATES:
        try:
            row = point_in_time_row(ticker, df, _BENCH, d, sector, _MIN_BARS)
        except Exception:
            row = None
        if row is not None:
            out.append(row)
    return out


# =============================================================================
#  Tarih izgarasi
# =============================================================================
def build_date_grid(bench_index: pd.DatetimeIndex, step: int = 3,
                    max_snapshots: int = 90, horizon: int = 21,
                    min_bars: int = MIN_BARS) -> list[pd.Timestamp]:
    """Tum hisselerin PAYLASTIGI islem gunu izgarasi.

    Iki ucu da bilincli olarak kirpilir:
      * bas: `min_bars` gun oncesine kadar faktorler hesaplanamaz
      * son: son `horizon` gun icin ILERI GETIRI henuz olusmamistir; etiketsiz
        satir uretmek dosyayi sisirir ve panelde ise yaramaz
    """
    idx = pd.DatetimeIndex(bench_index).sort_values()
    lo = min_bars
    hi = len(idx) - horizon - 1
    if hi <= lo:
        return []
    usable = list(idx[lo:hi])
    picked = usable[::-1][::step][:max_snapshots]      # sondan basa, seyrelterek
    return sorted(picked)


# =============================================================================
#  Ana giris
# =============================================================================
def _parts_dir(store: Path) -> Path:
    return store / "_parts"


def _done_tickers(store: Path) -> set[str]:
    """Onceki calismalarda islenmis semboller."""
    p = _parts_dir(store) / "islenen.txt"
    if not p.exists():
        return set()
    try:
        return {ln.strip() for ln in p.read_text(encoding="utf-8").splitlines()
                if ln.strip()}
    except OSError:
        return set()


def _flush(store: Path, rows: list[dict], names: list[str], seq: int) -> None:
    """Bir yigin sonucu diske yazar ve islenen sembolleri isaretler.

    Once veri, sonra isaretleme: sira tersine donerse islem yarida kesildiginde
    sembol "islendi" gorunur ama satirlari kaybolur.
    """
    pdir = _parts_dir(store)
    pdir.mkdir(parents=True, exist_ok=True)
    if rows:
        pd.DataFrame(rows).to_csv(pdir / f"part_{seq:04d}.csv", index=False,
                                  encoding="utf-8")
    with (pdir / "islenen.txt").open("a", encoding="utf-8") as fh:
        for n in names:
            fh.write(f"{n}\n")


def _merge_parts(store: Path) -> pd.DataFrame:
    """Tum yiginlari tek panelde birlestirir."""
    pdir = _parts_dir(store)
    if not pdir.exists():
        return pd.DataFrame()
    frames = []
    for p in sorted(pdir.glob("part_*.csv")):
        try:
            frames.append(pd.read_csv(p))
        except Exception:
            continue
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _next_seq(store: Path) -> int:
    existing = sorted(_parts_dir(store).glob("part_*.csv")) \
        if _parts_dir(store).exists() else []
    return len(existing)


def build(step: int = 3, max_snapshots: int = 90, horizon: int = 21,
          workers: int = 4, tickers: list[str] | None = None,
          min_bars: int = MIN_BARS, store: Path | None = None,
          progress: bool = True, resume: bool = True,
          chunk: int = 150) -> dict:
    """Gecmise donuk anlik goruntuleri uretir ve tarih basina CSV yazar.

    KESINTIYE DAYANIKLI. Tum evren icin bu is bir saati bulur; makinenin
    kapanmasi veya islemin oldurulmesi bir saatlik hesabi cope atmamali. Her
    `chunk` sembolde sonuc diske yazilir ve islenen semboller kaydedilir;
    tekrar calistirildiginda kaldigi yerden devam eder.

    Yarim kalmis bir calisma da kullanilabilir panel birakir — anlik goruntu
    dosyalari her calismanin sonunda birikmis TUM yiginlardan yeniden uretilir,
    sadece icindeki hisse sayisi azdir. `resume=False` bastan baslatir.
    """
    from . import scanlog
    from .providers import yahoo

    store = store or BACKFILL_STORE
    if tickers is None:
        tickers = sorted(scanlog.load().keys())
    if not tickers:
        return {"ok": False, "reason": "taranmis sembol kaydi yok — once "
                                       "'python run.py daily' calistir"}

    store.mkdir(parents=True, exist_ok=True)
    if not resume:
        pdir = _parts_dir(store)
        if pdir.exists():
            for old in pdir.iterdir():
                old.unlink()

    yahoo.ensure_ssl_env()
    bench = yahoo.fetch_benchmark("SPY", "2y", use_cache=True)
    if bench is None or len(bench) < min_bars:
        return {"ok": False, "reason": "SPY gecmisi onbellekte yok — goreli guc "
                                       "ve tarih izgarasi kurulamaz"}

    bench_close = bench["Close"] if isinstance(bench, pd.DataFrame) else bench
    bench_close = bench_close.dropna()
    try:
        bench_close.index = pd.to_datetime(bench_close.index).tz_localize(None)
    except (TypeError, ValueError):
        bench_close.index = pd.to_datetime(bench_close.index)

    dates = build_date_grid(bench_close.index, step=step,
                            max_snapshots=max_snapshots, horizon=horizon,
                            min_bars=min_bars)
    if not dates:
        return {"ok": False, "reason": f"onbellekteki gecmis {min_bars}+{horizon} "
                                       f"bardan kisa — izgara kurulamadi"}

    already = _done_tickers(store) if resume else set()
    todo = [t for t in tickers if t not in already]
    if progress and already:
        print(f"      devam ediliyor: {len(already)} sembol daha once islendi, "
              f"{len(todo)} kaldi")

    seq = _next_seq(store)
    batch_rows: list[dict] = []
    batch_names: list[str] = []
    skipped = 0
    done = 0

    if todo:
        with ProcessPoolExecutor(max_workers=max(1, workers), initializer=_init,
                                 initargs=(bench_close, dates, min_bars)) as pool:
            futs = {pool.submit(_worker, t): t for t in todo}
            try:
                for fut in as_completed(futs):
                    done += 1
                    name = futs[fut]
                    try:
                        got = fut.result()
                    except Exception:
                        got = []
                    if got:
                        batch_rows.extend(got)
                    else:
                        skipped += 1
                    batch_names.append(name)

                    if len(batch_names) >= chunk:
                        _flush(store, batch_rows, batch_names, seq)
                        seq += 1
                        batch_rows, batch_names = [], []
                        if progress:
                            print(f"      {done}/{len(todo)} hisse islendi "
                                  f"(diske yazildi)", flush=True)
            except KeyboardInterrupt:
                # Elde ne varsa kaydet, sonra cik. Kesinti veri kaybettirmemeli.
                _flush(store, batch_rows, batch_names, seq)
                batch_rows, batch_names = [], []
                raise
            finally:
                if batch_names:
                    _flush(store, batch_rows, batch_names, seq)

    df = _merge_parts(store)
    if df.empty:
        return {"ok": False, "reason": "hicbir hisse icin yeterli gecmis yok",
                "tickers": len(tickers), "skipped": skipped}

    for old in store.glob("snapshot_*.csv"):
        old.unlink()

    written = 0
    for date, grp in df.groupby("snapshot_date"):
        grp = grp.sort_values("ticker")
        grp.to_csv(store / f"snapshot_{date}.csv", index=False, encoding="utf-8")
        written += 1

    processed = _done_tickers(store)
    remaining = [t for t in tickers if t not in processed]

    meta = {
        "ok": True,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "store": str(store),
        "snapshots": written,
        "rows": len(df),
        "tickers_requested": len(tickers),
        "tickers_processed": len(processed),
        "tickers_remaining": len(remaining),
        "complete": not remaining,
        "tickers_used": int(df["ticker"].nunique()),
        "tickers_skipped": skipped,
        "first_date": str(df["snapshot_date"].min()),
        "last_date": str(df["snapshot_date"].max()),
        "step_trading_days": step,
        "horizon": horizon,
        "factors": list(PIT_FACTORS),
        "rows_per_snapshot_median": int(df.groupby("snapshot_date").size().median()),
        "bias_warning": (
            "Hayatta kalma yanliligi: onbellek yalnizca bugun kote olan "
            "hisseleri icerir. Temel veri yok (gecmise donuk bilinemez). "
            "Bu panel ON EGITIM icindir; sampiyon secimi gercek ileriye "
            "donuk anlik goruntularle yapilmalidir."
        ),
    }
    (store / "backfill_bilgisi.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def materialize(store: Path | None = None, step: int = 3, horizon: int = 21
                ) -> dict:
    """Birikmis yiginlari, is DEVAM EDERKEN bile panele cevirir.

    Uzun bir uretim sirasinda elde olani kullanabilmek icin: yigin dosyalari
    zaten diskte, bu fonksiyon onlari tarih basina anlik goruntulere yazar.
    Calisan surece dokunmaz; o kendi yiginlarini yazmaya devam eder ve bitince
    ayni dosyalari tekrar uretir.
    """
    store = store or BACKFILL_STORE
    df = _merge_parts(store)
    if df.empty:
        return {"ok": False, "reason": "birikmis yigin yok"}

    store.mkdir(parents=True, exist_ok=True)
    for old in store.glob("snapshot_*.csv"):
        old.unlink()

    written = 0
    for date, grp in df.groupby("snapshot_date"):
        grp.sort_values("ticker").to_csv(store / f"snapshot_{date}.csv",
                                         index=False, encoding="utf-8")
        written += 1

    processed = _done_tickers(store)
    meta = {
        "ok": True,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "store": str(store),
        "snapshots": written,
        "rows": len(df),
        "tickers_processed": len(processed),
        "tickers_used": int(df["ticker"].nunique()),
        "first_date": str(df["snapshot_date"].min()),
        "last_date": str(df["snapshot_date"].max()),
        "step_trading_days": step,
        "horizon": horizon,
        "factors": list(PIT_FACTORS),
        "rows_per_snapshot_median": int(df.groupby("snapshot_date").size().median()),
        "partial": True,
        "bias_warning": (
            "Hayatta kalma yanliligi: onbellek yalnizca bugun kote olan "
            "hisseleri icerir. Temel veri yok (gecmise donuk bilinemez). "
            "Bu panel ON EGITIM icindir; sampiyon secimi gercek ileriye "
            "donuk anlik goruntularle yapilmalidir."
        ),
    }
    (store / "backfill_bilgisi.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def info(store: Path | None = None) -> dict | None:
    """Uretilmis panelin ozeti (yoksa None)."""
    store = store or BACKFILL_STORE
    p = store / "backfill_bilgisi.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
