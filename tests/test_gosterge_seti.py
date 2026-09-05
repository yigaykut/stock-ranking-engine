"""The wide feature set: does it look forward, and does it stay comparable?

Two things can go wrong here and neither shows up as an error.

An indicator that peeks at future bars makes the model look brilliant in
backtest and useless live. The check is the same one used for the setup
detectors: cut the series at a bar, recompute, and every feature at that bar
has to come out identical.

And a feature that isn't scale-free quietly encodes "this is an expensive
stock". The model would learn the ticker, not the pattern.

Run:  python tests/test_gosterge_seti.py
"""
from __future__ import annotations

import io
import re
import sys
import tokenize
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import gosterge_seti as gs     # noqa: E402

fails = 0


def check(name, cond, extra=""):
    global fails
    if cond:
        print(f"  OK    {name}" + (f"  {extra}" if extra else ""))
    else:
        print(f"  HATA  {name}" + (f"  {extra}" if extra else ""))
        fails += 1


def barlar(n=600, seed=0, fiyat=40.0):
    r = np.random.default_rng(seed)
    c = fiyat * np.exp(np.cumsum(r.normal(0.0004, 0.011, n)))
    onceki = np.concatenate([[fiyat], c[:-1]])
    o = onceki * (1 + r.normal(0, 0.002, n))
    h = np.maximum(o, c) * (1 + np.abs(r.normal(0, 0.005, n)))
    l = np.minimum(o, c) * (1 - np.abs(r.normal(0, 0.005, n)))
    v = np.exp(r.normal(13.5, 0.6, n))
    return pd.DataFrame({"Open": o, "High": h, "Low": l, "Close": c,
                         "Volume": v},
                        index=pd.bdate_range("2024-01-01", periods=n))


print("=" * 72)
print("1) SHAPE AND SANITY")
print("=" * 72)

df = barlar()
F = gs.olustur(df)
check("plenty of features", F.shape[1] >= 70, f"{F.shape[1]} columns")
check("one row per bar", len(F) == len(df))
check("no NaN", int(F.isna().sum().sum()) == 0)
check("no inf", int(np.isinf(F.to_numpy()).sum()) == 0)
check("names can be listed without real bars",
      len(gs.adlar()) == F.shape[1])
check("index is preserved", bool((F.index == df.index).all()))

print()
print("=" * 72)
print("2) NO LOOKAHEAD")
print("=" * 72)

kesikler = [300, 400, 500, 599]
sapan = []
for kes in kesikler:
    kismi = gs.olustur(df.iloc[:kes + 1])
    a = F.iloc[kes]
    b = kismi.iloc[-1]
    for sut in F.columns:
        if abs(float(a[sut]) - float(b[sut])) > 1e-4:
            sapan.append((sut, kes, float(a[sut]), float(b[sut])))
check("cutting the series leaves each feature unchanged", not sapan,
      str(sapan[:3]))

bozuk = df.copy()
bozuk.iloc[450:] = bozuk.iloc[450:] * 4.0
Fb = gs.olustur(bozuk)
fark = [s for s in F.columns
        if abs(float(F.iloc[449][s]) - float(Fb.iloc[449][s])) > 1e-4]
check("changing later bars leaves earlier features alone", not fark,
      str(fark[:3]))

kod = []
with io.open(ROOT / "src" / "gosterge_seti.py", encoding="utf-8") as f:
    for tok in tokenize.generate_tokens(f.readline):
        if tok.type not in (tokenize.COMMENT, tokenize.STRING):
            kod.append(tok.string)
kod = " ".join(kod)
check("no negative shift in the source", not re.search(r"shift\s*\(\s*-", kod))
check("no center=True", not re.search(r"center\s*=\s*True", kod))
check("no bfill", not re.search(r"(bfill|backfill)", kod))

print()
print("=" * 72)
print("3) SCALE-FREE")
print("=" * 72)

# Same shape, ten times the price. Features should barely move.
ucuz = barlar(seed=5, fiyat=8.0)
pahali = ucuz.copy()
for k in ("Open", "High", "Low", "Close"):
    pahali[k] = pahali[k] * 40.0
Fu, Fp = gs.olustur(ucuz).iloc[-1], gs.olustur(pahali).iloc[-1]
kayan = []
for sut in Fu.index:
    if sut == "dolar_hacim" or sut.endswith("dolar_hacim"):
        continue                      # log of price*volume, expected to shift
    a, b = float(Fu[sut]), float(Fp[sut])
    if abs(a - b) > max(1e-3, 0.01 * abs(a)):
        kayan.append((sut, round(a, 4), round(b, 4)))
check("40x the price changes nothing but dollar volume", not kayan,
      str(kayan[:4]))

# Same for volume scale.
bol = ucuz.copy()
bol["Volume"] = bol["Volume"] * 100.0
Fv = gs.olustur(bol).iloc[-1]
hacim_kayan = [s for s in Fu.index
               if not s.startswith("dolar")
               and abs(float(Fu[s]) - float(Fv[s])) > max(1e-3,
                                                          0.01 * abs(float(Fu[s])))]
check("100x the volume changes nothing but dollar volume",
      not hacim_kayan, str(hacim_kayan[:4]))

print()
print("=" * 72)
print("4) CANDLE PATTERNS")
print("=" * 72)

o = np.array([10.0, 10.0, 9.5, 10.0])
h = np.array([10.2, 10.1, 10.6, 10.1])
l = np.array([9.8, 9.9, 9.4, 9.9])
c = np.array([10.1, 9.6, 10.5, 10.0])
m = gs.mum_kaliplari(o, h, l, c)
check("bullish engulfing is caught", m["yutan_boga_sekli"][2] == 1.0)
check("first two bars never fire",
      all(v[0] == 0.0 and v[1] == 0.0 for v in m.values()))
check("flags are 0 or 1",
      all(set(np.unique(v)) <= {0.0, 1.0} for v in m.values()))
check("there are a good number of patterns", len(m) >= 15, f"{len(m)}")

oranlar = {k: float(gs.olustur(df)[k].mean()) for k in m}
cok_sik = {k: round(v, 3) for k, v in oranlar.items() if v > 0.4}
check("no pattern fires on more than 40% of bars", not cok_sik, str(cok_sik))
hic = [k for k, v in oranlar.items() if v == 0.0]
check("no pattern is completely dead", not hic, str(hic))

print()
if fails:
    print(f"{fails} KONTROL BASARISIZ")
    raise SystemExit(1)
print("TUM GOSTERGE SETI TESTLERI GECTI")
