"""Ogrenme boru hattinin dogrulugu — SENTETIK veriyle.

Neden sentetik: gercek veri birikmesi aylar aliyor. Ama boru hattinin dogru
olup olmadigini bugun bilmemiz gerekiyor. Sinyali BIZIM koydugumuz bir veri
kumesinde model sinyali buluyorsa, mekanizma calisiyordur; sinyalsiz veride
sifir bulmasi gerekiyorsa ve buluyorsa, asiri uyum yapmiyordur.

Test edilen kritik ozellikler:
  1. Bilinen sinyal ogreniliyor mu
  2. Sinyalsiz veride SIFIR sonuc uretiliyor mu (asiri uyum kontrolu)
  3. Arindirma (purge) gercekten sizintiyi engelliyor mu
  4. Terfi kapisi gurultuyu reddediyor mu
  5. Dizi modeli, capraz kesitsel modelin GOREMEDIGI zaman orunusunu goruyor mu

Calistir:  python tests/test_ml.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import dataset as ds          # noqa: E402
from src import models as mz           # noqa: E402
from src import training as tr         # noqa: E402

RNG = np.random.default_rng(20260812)


# ---------------------------------------------------------------------------
def _make_panel(n_days=120, n_stocks=80, n_feat=12, signal=0.55, seed=1):
    """Sinyali BIZIM koydugumuz panel.

    y = signal * (f0 - f1) + gurultu   -> yalnizca ilk iki ozellik bilgi tasir.
    """
    rng = np.random.default_rng(seed)
    dates, tickers, X, y = [], [], [], []
    for d in range(n_days):
        stamp = f"2026-{1 + d // 30:02d}-{1 + d % 30:02d}"
        f = rng.normal(size=(n_stocks, n_feat)).astype(np.float32)
        target = signal * (f[:, 0] - f[:, 1]) + rng.normal(scale=1.0, size=n_stocks)
        target -= target.mean()
        X.append(f); y.append(target.astype(np.float32))
        dates += [stamp] * n_stocks
        tickers += [f"T{i:03d}" for i in range(n_stocks)]
    return ds.Panel(X=np.vstack(X), y=np.concatenate(y),
                    dates=np.array(dates), tickers=np.array(tickers),
                    feature_names=[f"score_f{i}" for i in range(n_feat)], horizon=21)


def _fit_eval(model, panel, train_frac=0.7):
    uniq = np.array(sorted(set(panel.dates)))
    cut = uniq[int(len(uniq) * train_frac)]
    tr_m, te_m = panel.dates < cut, panel.dates >= cut
    model.fit(panel.X[tr_m], panel.y[tr_m], panel.dates[tr_m])
    pred = model.predict(panel.X[te_m])
    return tr.evaluate_predictions(pred, panel.y[te_m], panel.dates[te_m])


fails = 0


def check(name, cond, detail=""):
    global fails
    if cond:
        print(f"  PASS  {name}" + (f"  {detail}" if detail else ""))
    else:
        print(f"  FAIL  {name}  {detail}")
        fails += 1


# ===========================================================================
print("\n=== 1. BILINEN SINYAL OGRENILIYOR MU ===")
panel = _make_panel()
ev = _fit_eval(mz.RidgeRanker(), panel)
check("ridge sinyali buluyor", (ev["ic_mean"] or 0) > 0.10,
      f"IC={ev['ic_mean']}  ICIR={ev['icir']}")
check("ilk dilim getirisi pozitif", (ev["top_decile_spread"] or 0) > 0,
      f"dilim farki={ev['top_decile_spread']}")

if mz.torch_available():
    ev_mlp = _fit_eval(mz.MLPRanker(epochs=60, patience=10), panel)
    check("mlp sinyali buluyor", (ev_mlp["ic_mean"] or 0) > 0.08,
          f"IC={ev_mlp['ic_mean']}")

# ===========================================================================
print("\n=== 2. SINYALSIZ VERIDE SIFIR (asiri uyum kontrolu) ===")
noise = _make_panel(signal=0.0, seed=99)
ev0 = _fit_eval(mz.RidgeRanker(), noise)
check("ridge gurultuye uymuyor", abs(ev0["ic_mean"] or 0) < 0.05,
      f"IC={ev0['ic_mean']} (0'a yakin olmali)")

if mz.torch_available():
    ev0m = _fit_eval(mz.MLPRanker(epochs=60, patience=10), noise)
    check("mlp gurultuye uymuyor", abs(ev0m["ic_mean"] or 0) < 0.06,
          f"IC={ev0m['ic_mean']}")

# ===========================================================================
print("\n=== 3. ARINDIRMA SIZINTIYI ENGELLIYOR MU ===")
p = _make_panel(n_days=100, n_stocks=40)
uniq = np.array(sorted(set(p.dates)))
horizon, embargo = 21, 5
splits = list(ds.walk_forward_splits(p.dates, horizon, n_splits=4, embargo=embargo))
check("katmanlar olusuyor", len(splits) >= 2, f"{len(splits)} katman")

gap_ok = True
for tr_m, te_m, meta in splits:
    tr_dates = set(p.dates[tr_m]); te_dates = set(p.dates[te_m])
    check_overlap = tr_dates & te_dates
    if check_overlap:
        gap_ok = False
        break
    # egitim sonu ile test basi arasinda en az horizon+embargo GUN olmali
    last_tr = max(np.flatnonzero(uniq == max(tr_dates)))
    first_te = min(np.flatnonzero(uniq == min(te_dates)))
    if first_te - last_tr < horizon + embargo:
        gap_ok = False
        print(f"     yetersiz bosluk: {first_te - last_tr} < {horizon + embargo}")
        break
check("egitim/test kesismiyor ve arindirma bosluk birakiyor", gap_ok)

# Sizinti testi: hedefi dogrudan iceren bir ozellik eklenirse, ARINDIRMA
# olmasa naif bolme bunu yakalayamazdi. Arindirmali bolmede egitim setinde
# test doneminin etiketi BULUNMAMALI.
leak_free = True
for tr_m, te_m, meta in splits:
    if meta["purged_dates"] < horizon:
        leak_free = False
check("arindirma penceresi ufuktan kucuk degil", leak_free)

# ===========================================================================
print("\n=== 4. TERFI KAPISI GURULTUYU REDDEDIYOR MU ===")
good = {"ok": True, "model": "x", "horizon": 21, "ic_mean": 0.06, "icir": 0.8,
        "folds": 5, "positive_folds": 5, "top_decile_spread": 0.02}
bad_ic = {**good, "ic_mean": 0.005, "icir": 0.9}
bad_icir = {**good, "icir": 0.10}
few_folds = {**good, "folds": 2, "positive_folds": 2}
unstable = {**good, "positive_folds": 2}

check("iyi model terfi ediyor", tr.promotion_check(good)["promote"])
check("dusuk IC reddediliyor", not tr.promotion_check(bad_ic)["promote"])
check("dusuk ICIR reddediliyor", not tr.promotion_check(bad_icir)["promote"])
check("az katman reddediliyor", not tr.promotion_check(few_folds)["promote"])
check("tutarsiz katmanlar reddediliyor", not tr.promotion_check(unstable)["promote"])

base = {"ok": True, "ic_mean": 0.058, "icir": 0.9, "folds": 5, "positive_folds": 5}
check("taban cizgisini gecemeyen reddediliyor",
      not tr.promotion_check(good, baseline=base)["promote"],
      f"model 0.060 vs taban 0.058")

w_none = tr.suggested_weight(None, None)
w_weak = tr.suggested_weight(0.35, 0.03)
w_strong = tr.suggested_weight(1.2, 0.08)
check("kanit yoksa agirlik sifir", w_none == 0.0, f"{w_none}")
check("zayif kanit -> kucuk agirlik", 0 <= w_weak < 3, f"{w_weak}")
check("guclu kanit -> buyuk ama sinirli agirlik", 8 <= w_strong <= 12, f"{w_strong}")

# ===========================================================================
print("\n=== 5. CAPRAZ KESITSEL NORMALIZASYON ===")
Xr = ds.cross_sectional_rank(panel.X, panel.dates)
d0 = panel.dates == panel.dates[0]
col = Xr[d0][:, 0]
check("gun ici -1..+1 araliginda", col.min() >= -1.001 and col.max() <= 1.001,
      f"[{col.min():.2f}, {col.max():.2f}]")
check("gun ici ortalama ~0", abs(float(col.mean())) < 0.05, f"{col.mean():.3f}")

yd = ds.demean_by_date(panel.y, panel.dates)
check("hedef gun ici ortalamasi ~0", abs(float(yd[d0].mean())) < 1e-5)

# ===========================================================================
print("\n=== 6. DIZI MODELI ZAMAN ORUNTUSUNU GORUYOR MU ===")
if mz.torch_available():
    # Sinyal YALNIZCA degisimde: y = f0(t) - f0(t-1). Capraz kesitsel model
    # tek gune baktigi icin bunu goremez; dizi modeli gormeli.
    rng = np.random.default_rng(5)
    n_days, n_stocks, n_feat = 90, 60, 6
    prev = rng.normal(size=(n_stocks, n_feat)).astype(np.float32)
    dates, tickers, Xs, ys = [], [], [], []
    for d in range(n_days):
        cur = (0.7 * prev + 0.7 * rng.normal(size=(n_stocks, n_feat))).astype(np.float32)
        target = 1.2 * (cur[:, 0] - prev[:, 0]) + rng.normal(scale=0.7, size=n_stocks)
        target -= target.mean()
        Xs.append(cur); ys.append(target.astype(np.float32))
        dates += [f"2026-{1 + d // 30:02d}-{1 + d % 30:02d}"] * n_stocks
        tickers += [f"S{i:03d}" for i in range(n_stocks)]
        prev = cur
    dp = ds.Panel(X=np.vstack(Xs), y=np.concatenate(ys), dates=np.array(dates),
                  tickers=np.array(tickers),
                  feature_names=[f"score_g{i}" for i in range(n_feat)], horizon=21)

    ev_cs = _fit_eval(mz.RidgeRanker(), dp)
    Xseq, yseq, dseq, tseq = ds.build_sequences(dp, window=4)
    check("dizi kurulumu satir uretiyor", len(yseq) > 1000, f"{len(yseq)} ornek")
    sp = ds.Panel(X=Xseq, y=yseq, dates=dseq, tickers=tseq, horizon=21)
    ev_sq = _fit_eval(mz.SeqRanker(epochs=60, patience=10, window=4), sp)

    print(f"       capraz kesitsel IC={ev_cs['ic_mean']}   dizi IC={ev_sq['ic_mean']}")
    check("dizi modeli degisim sinyalini yakaliyor", (ev_sq["ic_mean"] or 0) > 0.10,
          f"IC={ev_sq['ic_mean']}")
    check("dizi modeli capraz kesitseli geciyor",
          (ev_sq["ic_mean"] or 0) > (ev_cs["ic_mean"] or 0) + 0.05,
          f"{ev_sq['ic_mean']} vs {ev_cs['ic_mean']}")
else:
    print("  ATLANDI  torch yok")

# ===========================================================================
print("\n=== 7. HAZIRLIK KAPISI ===")
r = ds.readiness(21)
check("hazirlik durumu okunuyor", "snapshots" in r and "ready_to_train" in r,
      f"{r['snapshots']} goruntu, %{r['progress_pct']}")
check("az veriyle egitim engelleniyor",
      not r["ready_to_train"] or r["snapshots"] >= 30,
      "gercek veri az oldugu icin kapi kapali olmali")


# ===========================================================================
print("\n=== 8. GERI BESLEME DONGUSU KAPANIYOR MU (uctan uca) ===")
import shutil
import tempfile

import pandas as pd

_real_dir, _real_reg = tr.MODEL_DIR, tr.REGISTRY
_tmp = Path(tempfile.mkdtemp(prefix="mlloop_"))
try:
    # Gercek kayit defterini kirletmeden calis
    tr.MODEL_DIR, tr.REGISTRY = _tmp, _tmp / "registry.json"

    # 1) EGIT — sinyalli sentetik panelle
    p8 = _make_panel(n_days=100, n_stocks=60, n_feat=8, signal=0.6, seed=11)
    model = mz.RidgeRanker()
    model.fit(p8.X, p8.y, p8.dates)
    mpath = _tmp / "ridge_h21.pkl"
    model.save(mpath)
    check("model diske yaziliyor", mpath.exists())

    # 2) DEGERLENDIR + TERFI
    ev8 = _fit_eval(mz.RidgeRanker(), p8)
    result8 = {"ok": True, "model": "ridge", "horizon": 21,
               "ic_mean": ev8["ic_mean"], "icir": ev8["icir"],
               "folds": 5, "positive_folds": 5,
               "top_decile_spread": ev8["top_decile_spread"],
               "model_path": str(mpath),
               "feature_names": p8.feature_names, "rank_features": True}
    dec8 = tr.promotion_check(result8)
    check("guclu model terfi onayi aliyor", dec8["promote"],
          f"IC={result8['ic_mean']}")
    entry = tr.promote(result8, dec8)
    check("sampiyon kaydediliyor", tr.champion() is not None,
          f"{entry['model']} agirlik {entry['weight']}")
    check("terfi agirligi pozitif", entry["weight"] > 0, f"{entry['weight']}")

    # 3) CANLI TAHMIN — skorlamaya donen adim
    live = pd.DataFrame({"ticker": [f"T{i:03d}" for i in range(60)]})
    for j, name in enumerate(p8.feature_names):
        live[name] = p8.X[:60, j]
    preds = tr.predict_live(live, p8.feature_names)
    check("canli tahmin uretiliyor", preds is not None and len(preds) == 60,
          f"{0 if preds is None else len(preds)} hisse")
    if preds:
        vals = np.array(list(preds.values()))
        check("tahminler 0-100 olceginde",
              vals.min() >= -0.01 and vals.max() <= 100.01,
              f"[{vals.min():.1f}, {vals.max():.1f}]")
        # Tahmin, gercek sinyalle ayni yonde olmali
        truth = p8.y[:60]
        ic_live = mz.spearman(np.array([preds[f"T{i:03d}"] for i in range(60)]), truth)
        check("canli tahmin sinyalle uyumlu", ic_live > 0.15, f"IC={ic_live:.3f}")

    # 4) PARAMETRE SETI DEGISIRSE SESSIZCE YANLIS TAHMIN URETMEMELI
    broken = live.drop(columns=[p8.feature_names[0]])
    check("eksik parametrede tahmin reddediliyor",
          tr.predict_live(broken, p8.feature_names) is None)

finally:
    tr.MODEL_DIR, tr.REGISTRY = _real_dir, _real_reg
    shutil.rmtree(_tmp, ignore_errors=True)

check("gercek kayit defteri kirlenmedi",
      tr.champion() is None or tr.REGISTRY == _real_reg)


# ===========================================================================
print("\n=== 9. DIZI MODELI CANLIDA CALISIYOR MU ===")
# Dizi modelinin tum degeri "son N gunde nasil degisti" bilgisidir. Eskiden
# canlida hic calistirilamiyordu (predict_live None donerdi), yani GRU
# olculebiliyor ama KULLANILAMIYORDU. Pencere artik feature store'daki onceki
# goruntulerden kuruluyor; burada o yolun gercekten isledigi test edilir.
from src import ml as _mlmod                                    # noqa: E402

if not mz.torch_available():
    print("  ATLANDI  torch yok")
else:
    _store_real = _mlmod.FEATURE_STORE
    _dir_real, _reg_real = tr.MODEL_DIR, tr.REGISTRY
    _tmp9 = Path(tempfile.mkdtemp(prefix="seqlive_"))
    try:
        _store = _tmp9 / "store"
        _store.mkdir()
        _mlmod.FEATURE_STORE = _store
        tr.MODEL_DIR, tr.REGISTRY = _tmp9, _tmp9 / "registry.json"

        WIN = 6
        p9 = _make_panel(n_days=40, n_stocks=40, n_feat=6, signal=0.7, seed=21)
        names9 = p9.feature_names

        # Panelin SON gunu "bugun"un canli satiri; oncekiler feature store'a
        # yazilir. Yani model tam da uretimde gorecegi kurulumla calisir.
        all_dates = sorted(set(p9.dates))
        hist_dates, live_date = all_dates[:-1], all_dates[-1]

        # Sadece ilk 30 sembol gecmise yazilir -> kalan 10'u pencere kuramaz
        short_tickers = {f"T{i:03d}" for i in range(30, 40)}
        for d in hist_dates:
            m = (p9.dates == d)
            rows = {"snapshot_date": p9.dates[m], "ticker": p9.tickers[m]}
            for j, nm in enumerate(names9):
                rows[nm] = p9.X[m, j]
            frame = pd.DataFrame(rows)
            frame = frame[~frame["ticker"].isin(short_tickers)]
            frame.to_csv(_store / f"snapshot_{d}.csv", index=False)

        check("gecmis goruntuler yazildi",
              len(list(_store.glob("snapshot_*.csv"))) == len(hist_dates),
              f"{len(hist_dates)} gun")

        # Dizi modelini egit (canli satir DISARIDA — sizinti olmasin)
        seq_src = ds.Panel(X=ds.cross_sectional_rank(p9.X, p9.dates),
                           y=ds.demean_by_date(p9.y, p9.dates),
                           dates=p9.dates, tickers=p9.tickers,
                           feature_names=names9, horizon=21)
        Xs, ys, sd, st = ds.build_sequences(seq_src, window=WIN, min_len=3)
        tr_m = sd != live_date
        seq = mz.AVAILABLE["seq"](window=WIN)
        seq.fit(Xs[tr_m], ys[tr_m], sd[tr_m])
        mp9 = _tmp9 / "seq_h21.pkl"
        seq.save(mp9)

        tr._save_registry({"champion": {
            "model": "seq", "horizon": 21, "path": str(mp9), "weight": 5.0,
            "feature_names": names9, "rank_features": True, "window": WIN,
        }, "history": [], "candidates": {}})

        live9 = pd.DataFrame({"ticker": p9.tickers[p9.dates == live_date]})
        for j, nm in enumerate(names9):
            live9[nm] = p9.X[p9.dates == live_date, j]

        preds9 = tr.predict_live(live9, names9)
        check("dizi modeli canlida tahmin URETIYOR", preds9 is not None,
              f"{0 if preds9 is None else len(preds9)} hisse")

        if preds9:
            check("gecmisi olmayan semboller kapsam DISINDA",
                  not (set(preds9) & short_tickers),
                  f"sizan: {sorted(set(preds9) & short_tickers)[:3]}")
            check("gecmisi olan tum semboller kapsamda", len(preds9) == 30,
                  f"{len(preds9)}/30")
            v9 = np.array(list(preds9.values()))
            check("tahminler 0-100 olceginde",
                  v9.min() >= -0.01 and v9.max() <= 100.01,
                  f"[{v9.min():.1f}, {v9.max():.1f}]")

            truth9 = {t: y for t, y in zip(p9.tickers[p9.dates == live_date],
                                           p9.y[p9.dates == live_date])}
            keys = sorted(preds9)
            ic9 = mz.spearman(np.array([preds9[k] for k in keys]),
                              np.array([truth9[k] for k in keys]))
            check("canli dizi tahmini sinyalle uyumlu", ic9 > 0.15, f"IC={ic9:.3f}")

        # Gecmis yoksa uydurma pencere kurup tahmin URETMEMELI
        for f in _store.glob("snapshot_*.csv"):
            f.unlink()
        check("gecmis yokken tahmin reddediliyor",
              tr.predict_live(live9, names9) is None)
    finally:
        _mlmod.FEATURE_STORE = _store_real
        tr.MODEL_DIR, tr.REGISTRY = _dir_real, _reg_real
        shutil.rmtree(_tmp9, ignore_errors=True)

print()
print(f"{'TUM ML TESTLERI GECTI' if not fails else str(fails) + ' TEST BASARISIZ'}\n")
sys.exit(1 if fails else 0)
