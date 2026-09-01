"""Gecmise donuk panel uretiminin dogrulugu.

En kritik soru: uretilen satirlar GERCEKTEN o gunun bilgisiyle mi
hesaplaniyor? Bir tek bar sizarsa, egitimde olculen basari sahte olur ve
sistem kendi kendini kandirir. Bunu dogrulamanin kesin yolu, ayni satiri
iki kez hesaplamaktir:

  A) tam seriden, `as_of` gunune kesilerek
  B) zaten `as_of` gununde BITEN bir seriden

Ikisi ayni cikmiyorsa, hesap gelecege bakiyor demektir.

Ayrica on egitim panelinin sampiyon uretememesi de burada test edilir —
o yanlilik freni sessizce kaybolursa, hayatta kalma yanliligi tasiyan bir
model canli skorlamaya girer.

Calistir:  python tests/test_backfill.py
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import backfill as bf         # noqa: E402
from src import dataset as ds          # noqa: E402
from src import ml                     # noqa: E402
from src import training as tr         # noqa: E402

fails = 0


def check(name, cond, detail=""):
    global fails
    if cond:
        print(f"  PASS  {name}" + (f"  {detail}" if detail else ""))
    else:
        print(f"  FAIL  {name}  {detail}")
        fails += 1


# ---------------------------------------------------------------------------
def synthetic_ohlcv(n=600, seed=7, start="2024-01-02"):
    """Gercekci bir fiyat serisi — trend + oynaklik + hacim."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n)
    ret = rng.normal(0.0006, 0.018, n)
    close = 40 * np.exp(np.cumsum(ret))
    high = close * (1 + np.abs(rng.normal(0, 0.008, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.008, n)))
    vol = rng.lognormal(13.5, 0.4, n)
    return pd.DataFrame({"Open": close, "High": high, "Low": low,
                         "Close": close, "Volume": vol}, index=idx)


def synthetic_bench(idx, seed=11):
    rng = np.random.default_rng(seed)
    ret = rng.normal(0.0004, 0.010, len(idx))
    return pd.Series(30 * np.exp(np.cumsum(ret)), index=idx)


# ===========================================================================
print("\n=== 1. GELECEGE BAKIS YOK MU ===")

df = synthetic_ohlcv()
bench = synthetic_bench(df.index)

# Izgaranin ortasindan birkac gun sec
for offset in (30, 90, 160):
    as_of = df.index[-offset]

    a = bf.point_in_time_row("TEST", df, bench, as_of)
    # B) seri zaten o gun bitiyor — gelecek fiziksel olarak yok
    b = bf.point_in_time_row("TEST", df.loc[:as_of], bench.loc[:as_of], as_of)

    same = True
    diffs = []
    for k in a:
        if not k.startswith("raw_"):
            continue
        va, vb = a[k], b[k]
        if va is None and vb is None:
            continue
        if va is None or vb is None or abs(va - vb) > 1e-9:
            same = False
            diffs.append(f"{k}: {va} vs {vb}")
    check(f"as_of=-{offset}: kesilmis seri ile ayni sonuc", same,
          "; ".join(diffs[:3]))

# Gelecegi DEGISTIRMEK satiri degistirmemeli (en dogrudan sizinti testi)
as_of = df.index[-120]
tampered = df.copy()
future = tampered.index > as_of
tampered.loc[future, ["Open", "High", "Low", "Close"]] *= 3.0
tampered.loc[future, "Volume"] *= 50.0

orig = bf.point_in_time_row("TEST", df, bench, as_of)
tamp = bf.point_in_time_row("TEST", tampered, bench, as_of)
def _differs(a, b) -> bool:
    if a is None or b is None:
        return (a is None) != (b is None)
    return abs(a - b) > 1e-9


leaked = [k for k in orig
          if k.startswith("raw_") and _differs(orig[k], tamp[k])]
check("gelecek barlar 3 katina cikarilinca satir DEGISMIYOR", not leaked,
      f"sizan: {leaked[:4]}")

# Benchmark'in gelecegi de sizmamali (goreli guc faktoru)
tb = bench.copy()
tb.loc[tb.index > as_of] *= 5.0
tamp_b = bf.point_in_time_row("TEST", df, tb, as_of)
check("benchmark'in gelecegi de sizmiyor",
      abs((orig["raw_relative_strength"] or 0) -
          (tamp_b["raw_relative_strength"] or 0)) < 1e-9)


# ===========================================================================
print("\n=== 2. YETERSIZ GECMIS REDDEDILIYOR MU ===")

short = df.iloc[:200]
check("200 barlik seride satir uretilmiyor",
      bf.point_in_time_row("TEST", short, bench, short.index[-1]) is None)
check("tam serinin ilk gunlerinde satir uretilmiyor",
      bf.point_in_time_row("TEST", df, bench, df.index[100]) is None)
row = bf.point_in_time_row("TEST", df, bench, df.index[bf.MIN_BARS])
check("MIN_BARS esiginde satir uretiliyor", row is not None,
      f"bars={None if row is None else row['bars_used']}")


# ===========================================================================
print("\n=== 3. TARIH IZGARASI DOGRU KIRPILIYOR MU ===")

grid = bf.build_date_grid(df.index, step=3, max_snapshots=90, horizon=21)
check("izgara kuruldu", len(grid) > 0, f"{len(grid)} tarih")
check("izgara sirali", list(grid) == sorted(grid))
check("ilk tarih MIN_BARS sonrasinda",
      grid[0] >= df.index[bf.MIN_BARS],
      f"{grid[0].date()} >= {df.index[bf.MIN_BARS].date()}")
check("son tarih, ufuk kadar sondan uzakta",
      grid[-1] <= df.index[-22],
      f"{grid[-1].date()} <= {df.index[-22].date()}")

gaps = pd.Series(grid).diff().dropna().dt.days
check("adim araligi uygulandi", (gaps >= 3).all(), f"min gap {gaps.min()} gun")

few = bf.build_date_grid(df.index, step=3, max_snapshots=5, horizon=21)
check("max_snapshots siniri uygulaniyor", len(few) == 5, f"{len(few)}")
check("sinirli izgara EN YENI tarihleri aliyor", few[-1] == grid[-1])

tiny = bf.build_date_grid(df.index[:100], step=3, max_snapshots=90, horizon=21)
check("kisa seride bos izgara doner", tiny == [])


# ===========================================================================
print("\n=== 4. AYRI DEPO GERCEKTEN AYRI MI ===")

tmp = Path(tempfile.mkdtemp(prefix="bf_test_"))
try:
    # 100 is gunu ~138 takvim gunu eder — dogrulama esigi (60 goruntu / 120 gun)
    # bilincli olarak takvim gunu de sorar; ikisini birden gecmeli.
    dates = pd.bdate_range("2025-01-06", periods=100)
    for i, d in enumerate(dates):
        pd.DataFrame({
            "snapshot_date": [d.strftime("%Y-%m-%d")] * 40,
            "ticker": [f"T{j:02d}" for j in range(40)],
            "raw_trend_structure": np.random.default_rng(i).normal(size=40),
        }).to_csv(tmp / f"snapshot_{d:%Y-%m-%d}.csv", index=False)

    got = ml.load_all_snapshots(tmp)
    check("alternatif depodan okunuyor", len(got) == 100 * 40, f"{len(got)} satir")
    check("canli depo etkilenmiyor",
          len(ml.load_all_snapshots()) != len(got) or ml.FEATURE_STORE == tmp)

    r = ds.readiness(21, store=tmp)
    check("readiness alternatif depoyu okuyor", r["snapshots"] == 100,
          f"{r['snapshots']} goruntu / {r['span_days']} gun")
    check("bu panel egitime hazir sayiliyor", r["ready_to_train"])
    check("bu panel dogrulamaya da hazir", r["ready_to_validate"])

    live = ds.readiness(21)
    check("canli depo readiness'i degismedi", live["snapshots"] != 100
          or ds.FEATURE_STORE == tmp, f"canli {live['snapshots']}")
finally:
    shutil.rmtree(tmp, ignore_errors=True)


# ===========================================================================
print("\n=== 5. ON EGITIM SAMPIYON URETEMIYOR MU ===")

strong = {"ok": True, "model": "seq", "horizon": 21, "folds": 5,
          "positive_folds": 5, "ic_mean": 0.25, "icir": 2.4,
          "top_decile_spread": 0.09, "pretrain": True,
          "store": "data/backfill_store"}
dec = tr.promotion_check(strong)
check("cok yuksek IC'li on egitim modeli bile terfi ALMIYOR",
      dec["promote"] is False, f"IC={strong['ic_mean']}")
check("reddetme gerekcesi yanliligi soyluyor",
      any("yanlilig" in r for r in dec["reasons"]), dec["reasons"][0])
check("on egitim modeline agirlik verilmiyor",
      dec["suggested_weight"] == 0.0)

same_live = dict(strong)
same_live.pop("pretrain")
same_live.pop("store")
dec_live = tr.promotion_check(same_live)
check("ayni sonuc CANLI panelde terfi aliyor", dec_live["promote"] is True,
      f"agirlik {dec_live['suggested_weight']}")


# ===========================================================================
print("\n=== 6. FAKTOR SETI TEMEL VERI ICERMIYOR MU ===")

fundamental = {"valuation_composite", "quality_profitability", "financial_health",
               "growth_quality", "eps_revision_momentum", "analyst_consensus",
               "analyst_upside", "earnings_surprise", "short_squeeze",
               "institutional_ownership", "size_opportunity", "revenue_scaling",
               "rule_of_40", "undiscovered", "cash_runway", "liquidity",
               "reddit_wsb_attention"}
overlap = fundamental & set(bf.PIT_FACTORS)
check("gecmise donuk sette temel/analist faktoru YOK", not overlap,
      f"sizan: {sorted(overlap)}")

row = bf.point_in_time_row("TEST", df, bench, df.index[-40])
cols = {k for k in row if k.startswith("raw_")}
expected = {f"raw_{f}" for f in bf.PIT_FACTORS}
check("beklenen tum faktorler uretiliyor", expected <= cols,
      f"eksik: {sorted(expected - cols)}")
check("her faktorun kapsama bayragi var",
      all(f"has_{f}" in row for f in bf.PIT_FACTORS))


# ===========================================================================
print("\n=== 7. KESINTIYE DAYANIKLILIK ===")
# Tum evren icin bu is bir saati buluyor. Kesilirse bir saatlik hesap
# kaybolmamali; ikinci calisma kaldigi yerden devam etmeli ve satir
# kaybetmemeli. (Gercek uretimde bir kere oldu: is oldu, her sey gitti.)

tmp7 = Path(tempfile.mkdtemp(prefix="bfresume_"))
try:
    rng7 = np.random.default_rng(3)

    def fake_rows(name, n):
        return [{"snapshot_date": f"2026-01-{d + 1:02d}", "ticker": name,
                 "raw_trend_structure": float(rng7.normal())} for d in range(n)]

    # 1. calisma: 2 yigin yazildi, sonra kesildi
    bf._flush(tmp7, fake_rows("AAA", 3) + fake_rows("BBB", 3), ["AAA", "BBB"], 0)
    bf._flush(tmp7, fake_rows("CCC", 3), ["CCC"], 1)

    done = bf._done_tickers(tmp7)
    check("islenen semboller kaydedildi", done == {"AAA", "BBB", "CCC"},
          f"{sorted(done)}")
    check("kesinti sonrasi veri diskte", len(bf._merge_parts(tmp7)) == 9,
          f"{len(bf._merge_parts(tmp7))} satir")

    # 2. calisma: yalnizca kalanlar islenmeli
    universe7 = ["AAA", "BBB", "CCC", "DDD", "EEE"]
    todo = [t for t in universe7 if t not in bf._done_tickers(tmp7)]
    check("devam ederken islenmisler atlaniyor", todo == ["DDD", "EEE"], f"{todo}")

    bf._flush(tmp7, fake_rows("DDD", 3) + fake_rows("EEE", 3), todo,
              bf._next_seq(tmp7))
    merged = bf._merge_parts(tmp7)
    check("iki calisma birlestiginde satir kaybi yok", len(merged) == 15,
          f"{len(merged)} satir")
    check("tum evren islenmis sayiliyor",
          bf._done_tickers(tmp7) == set(universe7))
    check("her sembol tam", merged.groupby("ticker").size().eq(3).all(),
          str(dict(merged.groupby("ticker").size())))

    # Yigin numarasi cakismamali — cakisirsa onceki yigin ustune yazilir
    seqs = sorted(p.name for p in bf._parts_dir(tmp7).glob("part_*.csv"))
    check("yigin dosyalari benzersiz", len(seqs) == 3, f"{seqs}")

    # Uretim DEVAM EDERKEN elde olan panele cevrilebilmeli
    m = bf.materialize(tmp7)
    check("yarim panel materyalize ediliyor", m.get("ok") is True,
          f"{m.get('snapshots')} goruntu / {m.get('rows')} satir")
    check("materyalize edilen panel yarim isaretli", m.get("partial") is True)
    snaps = sorted(tmp7.glob("snapshot_*.csv"))
    check("anlik goruntu dosyalari yazildi", len(snaps) == 3, f"{len(snaps)}")
    check("dosya basina tum semboller var",
          len(pd.read_csv(snaps[0])) == 5, f"{len(pd.read_csv(snaps[0]))}")
    check("readiness yarim paneli gorebiliyor",
          ds.readiness(21, store=tmp7)["snapshots"] == 3)
finally:
    shutil.rmtree(tmp7, ignore_errors=True)


print()
print(f"{'TUM BACKFILL TESTLERI GECTI' if not fails else str(fails) + ' TEST BASARISIZ'}\n")
sys.exit(1 if fails else 0)
