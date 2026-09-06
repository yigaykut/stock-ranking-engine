"""Meta-model plumbing: the split, the rescaling, and the metrics.

The model itself is measured against real data by `run.py meta`. What's tested
here is the machinery around it, because two of these pieces went wrong in ways
that looked fine from the outside.

The training split has to be by time. Split at random and the same day lands on
both sides, so the model gets scored on days it already trained on and every
number after that is inflated.

Platt rescaling has to shrink overconfidence without inventing skill. The
sequence model was predicting anywhere from 6% to 91% while the real rate never
left 43-47%, and Brier punished that hard. Rescaling should pull the range back
and leave ranking ability exactly where it was.

Run:  python tests/test_meta_model.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import meta_model as mm      # noqa: E402

fails = 0


def check(name, cond, extra=""):
    global fails
    if cond:
        print(f"  OK    {name}" + (f"  {extra}" if extra else ""))
    else:
        print(f"  HATA  {name}" + (f"  {extra}" if extra else ""))
        fails += 1


print("=" * 72)
print("1) VALIDATION SPLIT IS BY TIME")
print("=" * 72)

gun = pd.DatetimeIndex(sorted(pd.date_range("2025-01-01", periods=200).repeat(20)))
tr, val = mm._dogrulama_bol(gun, pay=0.15)
check("both sides are non-empty", tr.any() and val.any(),
      f"{tr.sum()} / {val.sum()}")
check("validation is roughly the requested share",
      0.10 < val.mean() < 0.22, f"{val.mean():.3f}")
check("no day appears on both sides",
      not (set(gun[tr]) & set(gun[val])),
      str(sorted(set(gun[tr]) & set(gun[val]))[:2]))
check("validation days all come after training days",
      max(gun[tr]) < min(gun[val]))
check("too few rows -> no split at all",
      not mm._dogrulama_bol(gun[:100])[1].any())

print()
print("=" * 72)
print("2) PLATT SHRINKS FAKE CONFIDENCE")
print("=" * 72)

rng = np.random.default_rng(0)
y = (rng.random(5000) < 0.47).astype(float)
p = np.clip(0.47 + (rng.random(5000) - 0.5) * 0.9, 0.01, 0.99)
a, b = mm._platt(p, y)
q = mm._uygula(p, (a, b))
print(f"        raw  {p.min():.3f}-{p.max():.3f} brier {mm.brier(p, y):.5f}")
print(f"        cal  {q.min():.3f}-{q.max():.3f} brier {mm.brier(q, y):.5f}")
check("range collapses toward the base rate",
      (q.max() - q.min()) < 0.25 * (p.max() - p.min()),
      f"{p.max()-p.min():.3f} -> {q.max()-q.min():.3f}")
check("brier improves a lot", mm.brier(q, y) < mm.brier(p, y) - 0.05)
check("brier lands near the constant baseline",
      abs(mm.brier(q, y) - mm.brier(np.full(len(y), y.mean()), y)) < 0.002)
check("ranking ability is not created",
      abs(mm.auc(q, y) - 0.5) < 0.03, f"{mm.auc(q, y):.4f}")
check("no calibration is a no-op",
      bool(np.array_equal(mm._uygula(p, None), p)))

# When predictions really do carry signal, rescaling must keep it.
sinyal = rng.normal(0, 1, 5000)
y2 = (rng.random(5000) < 1 / (1 + np.exp(-sinyal))).astype(float)
p2 = np.clip(0.5 + sinyal * 0.02, 0.01, 0.99)      # right order, wrong scale
q2 = mm._uygula(p2, mm._platt(p2, y2))
check("real ranking survives rescaling",
      abs(mm.auc(q2, y2) - mm.auc(p2, y2)) < 0.01,
      f"{mm.auc(p2, y2):.3f} -> {mm.auc(q2, y2):.3f}")
check("and the scale gets fixed", mm.brier(q2, y2) < mm.brier(p2, y2),
      f"{mm.brier(p2, y2):.5f} -> {mm.brier(q2, y2):.5f}")

print()
print("=" * 72)
print("3) METRICS")
print("=" * 72)

yy = np.array([0, 0, 1, 1, 1.0])
check("brier is zero on perfect predictions", mm.brier(yy, yy) == 0.0)
check("brier of a coin flip is 0.25",
      abs(mm.brier(np.full(5, 0.5), yy) - 0.25) < 1e-9)
check("auc is 1 when ranking is perfect",
      mm.auc(np.array([0.1, 0.2, 0.8, 0.9, 0.95]), yy) == 1.0)
check("auc is 0 when ranking is reversed",
      mm.auc(np.array([0.95, 0.9, 0.2, 0.1, 0.05]), yy) == 0.0)
check("auc is nan with a single class",
      not np.isfinite(mm.auc(np.arange(5.0), np.ones(5))))

# Daily aggregation: an improvement spread over few days shouldn't look strong
t_gun = pd.DatetimeIndex(["2025-01-01"] * 500 + ["2025-01-02"] * 500)
yb = (np.random.default_rng(1).random(1000) < 0.5).astype(float)
pm = np.full(1000, 0.5)
pt = np.full(1000, 0.5)
f, t, n = mm.gunluk_fark(pm, pt, yb, t_gun)
check("fewer than five days -> no t value", not np.isfinite(t), f"{n} days")

print()
print("=" * 72)
print("4) FEATURE COLUMNS EXCLUDE IDS AND LABELS")
print("=" * 72)

df = pd.DataFrame({
    "ticker": ["A"], "tarih": ["2025-01-01"], "zaman": ["2025-01-01 10:00"],
    "frekans": ["1h"], "kurulum": ["cekic"], "yon": ["long"],
    "guc": [0.5], "rsi14": [50.0],
    "fazla_5g": [0.01], "kazanc_5g": [1.0],
})
oz = mm._ozellik_sutunlari(df)
check("ids are excluded", not (set(mm.KIMLIK) & set(oz)), str(oz))
check("labels are excluded",
      not any(c.startswith(("fazla_", "kazanc_")) for c in oz))
check("real features survive", set(oz) == {"guc", "rsi14"}, str(oz))

print()
print("=" * 72)
print("5) TOP-DECILE RETURN")
print("=" * 72)

# This is the metric the decision now rests on, so it has to do three things:
# find a real edge, stay quiet on a fake one, and show where costs kill a
# small one. Hit rate can't show the third.
rng2 = np.random.default_rng(3)
n = 8000
gun = pd.DatetimeIndex(np.repeat(pd.date_range("2025-01-01", periods=400), 20))
sinyal = rng2.normal(0, 1, n)
getiri = 0.004 * sinyal + rng2.normal(0, 0.02, n)
tahmin = 0.5 + 0.1 * sinyal + rng2.normal(0, 0.02, n)

d = mm.dilim_getirisi(tahmin, getiri, gun, maliyet_bp=10)
check("a planted edge is found", d["ok"] and d["t_nw"] >= 2, str(d.get("t_nw")))
check("top decile beats the base rate", d["getiri"] > d["taban"],
      f"{100*d['getiri']:.3f}% vs {100*d['taban']:.3f}%")
check("costs come off the gross number",
      abs((d["brut"] - d["getiri"]) - 0.001) < 1e-9,
      f"{100*d['brut']:.3f}% -> {100*d['getiri']:.3f}%")
getiriler = [x["getiri"] for x in d["dilimler"]]
check("deciles are ordered", getiriler[-1] > getiriler[0],
      f"{100*getiriler[0]:.2f}% .. {100*getiriler[-1]:.2f}%")

bos = mm.dilim_getirisi(rng2.normal(0.5, 0.1, n), getiri, gun, maliyet_bp=10)
check("no edge when predictions are noise", abs(bos["t_nw"]) < 2,
      str(bos["t_nw"]))

kucuk = 0.0003 * sinyal + rng2.normal(0, 0.02, n)
sifir = mm.dilim_getirisi(tahmin, kucuk, gun, maliyet_bp=0)
yirmi = mm.dilim_getirisi(tahmin, kucuk, gun, maliyet_bp=20)
check("a small edge survives at zero cost and dies at 20bp",
      sifir["getiri"] > 0 > yirmi["getiri"],
      f"{100*sifir['getiri']:+.4f}% -> {100*yirmi['getiri']:+.4f}%")

check("too few rows is refused",
      not mm.dilim_getirisi(tahmin[:50], getiri[:50], gun[:50])["ok"])
check("too few days is refused",
      not mm.dilim_getirisi(tahmin[:500], getiri[:500],
                            pd.DatetimeIndex(["2025-01-01"] * 500))["ok"])

print()
if fails:
    print(f"{fails} KONTROL BASARISIZ")
    raise SystemExit(1)
print("TUM META MODEL TESTLERI GECTI")
