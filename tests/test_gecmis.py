"""Uzun gunluk gecmis — indirilen barin DOGRU bar oldugunu dogrular.

Bu modulun tehlikeli yeri tek bir yerde toplaniyor: bolunme duzeltmesi.
Yahoo ham OHLC'yi duzeltmeden verir ve adjclose'u ayri bir dizide tutar.
Olcekleme yapilmazsa her hisse bolunmesi, gostergelerin gercek sandigi
devasa bir bosluk olur -- %50'lik bir dusus gibi gorunur ve model onu bir
kalip olarak ogrenir. Hicbir sey hata vermez.

Ikinci tehlike sessiz veri kaybi: kisa gecmisli bir sembolun, daha uzun
gecmisi olan bir anahtarin uzerine yazilmasi.

Ag istegi YAPILMAZ: hepsi sentetik cevaplarla.

Calistir:  python tests/test_gecmis.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import gecmis as gc          # noqa: E402

fails = 0


def check(ad, kosul, ek=""):
    global fails
    ok = bool(kosul)
    if not ok:
        fails += 1
    print(f"  {'OK  ' if ok else 'HATA'}  {ad}" + (f"  {ek}" if ek else ""))


def cevap(kapanis, adjclose=None, hacim=None, baslangic="2020-01-01"):
    """Yahoo chart ucunun dondurdugu seklin sentetik hali."""
    n = len(kapanis)
    ts = [int(t.timestamp()) for t in
          pd.date_range(baslangic, periods=n, freq="D", tz="UTC")]
    def olcek(c, k):
        return None if c is None else c * k

    q = {"open": list(kapanis), "high": [olcek(c, 1.01) for c in kapanis],
         "low": [olcek(c, 0.99) for c in kapanis], "close": list(kapanis),
         "volume": list(hacim if hacim is not None else [1e6] * n)}
    ind = {"quote": [q]}
    if adjclose is not None:
        ind["adjclose"] = [{"adjclose": list(adjclose)}]
    return {"chart": {"result": [{"timestamp": ts, "indicators": ind}]}}


print("=" * 72)
print("1) THE FRAME COMES OUT IN THE RIGHT SHAPE")
print("=" * 72)

d = gc._cerceve(cevap([10.0, 11.0, 12.0, 13.0]))
check("a frame is produced", d is not None)
check("every OHLCV column is there",
      list(d.columns) == ["Open", "High", "Low", "Close", "Volume"],
      str(list(d.columns)))
check("the index is naive dates", d.index.tz is None and
      (d.index == d.index.normalize()).all())
check("rows are in order", d.index.is_monotonic_increasing)
check("a broken payload returns nothing",
      gc._cerceve({"chart": {"result": []}}) is None)
check("an empty payload returns nothing", gc._cerceve({}) is None)

print()
print("=" * 72)
print("2) SPLITS ARE ADJUSTED AWAY")
print("=" * 72)

# A 2-for-1 split halfway through. Raw close halves; adjclose scales the
# earlier half down so the series is continuous.
ham = [100.0, 102.0, 104.0, 53.0, 54.0, 55.0]
adj = [50.0, 51.0, 52.0, 53.0, 54.0, 55.0]
d = gc._cerceve(cevap(ham, adjclose=adj))

check("the adjusted close is used",
      np.allclose(d["Close"].to_numpy(), adj),
      f"{d['Close'].tolist()}")

getiri = d["Close"].pct_change().dropna().abs()
check("the split is no longer a 50% crash", getiri.max() < 0.10,
      f"largest daily move {100 * getiri.max():.1f}%")

ham_getiri = pd.Series(ham).pct_change().dropna().abs()
check("without adjusting it would have been one", ham_getiri.max() > 0.40,
      f"{100 * ham_getiri.max():.0f}%")

# Open/High/Low have to move with Close, or the bar stops being a bar.
check("the whole bar is scaled, not just the close",
      bool((d["High"] >= d["Close"]).all() and (d["Low"] <= d["Close"]).all()),
      "high/low must still bracket the close after scaling")
oran = (d["Open"] / d["Close"]).round(6)
check("the open keeps its ratio to the close", oran.nunique() == 1,
      f"{sorted(oran.unique())}")

# Volume is a count, not a price. Scaling it would be wrong.
check("volume is left alone", bool((d["Volume"] == 1e6).all()))

print()
print("=" * 72)
print("3) MISSING OR ODD ADJCLOSE DOESN'T CORRUPT ANYTHING")
print("=" * 72)

d = gc._cerceve(cevap([10.0, 11.0, 12.0]))
check("no adjclose -> raw prices, not zeros",
      np.allclose(d["Close"].to_numpy(), [10.0, 11.0, 12.0]))

d = gc._cerceve(cevap([10.0, 11.0, 12.0], adjclose=[None, 11.0, 0.0]))
check("a null or zero ratio falls back to 1 rather than blanking the bar",
      np.allclose(d["Close"].to_numpy(), [10.0, 11.0, 12.0]),
      f"{d['Close'].tolist()}")

d = gc._cerceve(cevap([10.0, None, 12.0]))
check("bars with no price are dropped", len(d) == 2, str(len(d)))

print()
print("=" * 72)
print("4) A SHORT HISTORY NEVER OVERWRITES A LONG ONE")
print("=" * 72)

# The cache key is shared with the rest of the system. Writing 60 bars over
# a key that held 2500 would lose data silently, so short histories are
# counted and skipped instead.
cagrilar = []
kisa = cevap([10.0 + i for i in range(60)])
uzun = cevap([10.0 + i for i in range(400)])

gercek_istek, gercek_put = gc._istek, gc._cache.put
try:
    gc._istek = lambda tk, aralik, zaman_asimi=20: (
        kisa if tk == "KISA" else uzun)
    gc._cache.put = lambda ns, ident, val: cagrilar.append(ident)
    gc._cache.peek = lambda ns, ident: None

    d = gc.cek(["KISA", "UZUN"], period="10y", bekle=0.0, en_az_bar=250)
finally:
    gc._istek, gc._cache.put = gercek_istek, gercek_put

check("the long history is written", "UZUN:10y" in cagrilar, str(cagrilar))
check("the short one is not", "KISA:10y" not in cagrilar)
check("and it is reported rather than hidden", d["kisa"] == 1, str(d))
check("the written count is right", d["yazildi"] == 1, str(d["yazildi"]))

print()
print("=" * 72)
print("5) RATE LIMITING STOPS THE RUN INSTEAD OF FIGHTING IT")
print("=" * 72)

# The system lost four days once because something kept retrying into a wall.
# Hitting 429 has to end the pass, not extend the ban -- and the next run
# picks up where this one stopped because finished symbols are skipped.
def _patlat(tk, aralik, zaman_asimi=20):
    raise gc.HizSiniri("429")


gercek_istek, gercek_peek = gc._istek, gc._cache.peek
try:
    gc._istek = _patlat
    gc._cache.peek = lambda ns, ident: None
    d = gc.cek(["A", "B", "C"], period="10y", bekle=0.0)
finally:
    gc._istek, gc._cache.peek = gercek_istek, gercek_peek

check("the run stops", d["durduruldu"] is not None, str(d["durduruldu"]))
check("nothing is claimed to be written", d["yazildi"] == 0)

# Already-cached symbols are skipped, which is what makes resuming cheap.
gercek_peek = gc._cache.peek
try:
    gc._cache.peek = lambda ns, ident: ({"history": pd.DataFrame()}, 0.0)
    d = gc.cek(["A", "B"], period="10y", bekle=0.0)
finally:
    gc._cache.peek = gercek_peek

check("cached symbols are skipped, so a rerun resumes", d["atlandi"] == 2,
      str(d["atlandi"]))

print()
if fails:
    print(f"{fails} KONTROL BASARISIZ")
    raise SystemExit(1)
print("TUM GECMIS TESTLERI GECTI")
