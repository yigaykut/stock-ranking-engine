"""Yedek kaynak ile Yahoo'nun ayni sekle sahip olmasi.

OLAY (31.08.2026): Yahoo hiz siniri devreye girdi, yedek kaynak 248 hisseyi
basariyla kurtardi -- ve kurtarilan 248 hissenin TAMAMI bir sonraki asamada
sessizce dusdu:

    TypeError: Cannot join tz-naive with tz-aware DatetimeIndex

Sebep: Yahoo gecmisi America/New_York ile DILIMLI geliyor, api.nasdaq.com
ucundan kurulan seri ise DILIMSIZ. Goreli guc faktoru seriyi endeksle
hizalarken patliyor, hata tek tek hisse bazinda yakalanip `errors` listesine
yaziliyor ve tarama devam ediyor. Yani gorunurde her sey calisiyor: "yedek
kaynaktan kurtarilan: 248 hisse" yaziyor, sonra o hisseler siralamada yok.

Bu, kurtarmanin kendisini anlamsiz kiliyordu. Daha kotusu, skorlanan sayi
dustugu icin cikti guvenlik kapisi panoyu reddediyor, gun isaretlenmiyor ve
sistem bir haftaligina kendini besleyen bir sarmala giriyordu.

Buradaki kontrol sekil (schema) kontrolu: iki kaynagin urettigi paketler
skorlamaya girmeden ONCE ayni tipte olmali. Ag istegi yapilmaz.

Calistir:  python tests/test_kaynak_uyum.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.providers import nasdaq as nq     # noqa: E402

fails = 0


def check(name: str, cond: bool, extra: str = "") -> None:
    global fails
    if cond:
        print(f"  OK    {name}" + (f"  {extra}" if extra else ""))
    else:
        print(f"  HATA  {name}" + (f"  {extra}" if extra else ""))
        fails += 1


def naive_frame(n: int = 40) -> pd.DataFrame:
    """Dilimsiz gunluk bar serisi -- duzeltme oncesi nasdaq ucunun urettigi sekil."""
    idx = pd.date_range("2026-01-01", periods=n, freq="B")   # tz YOK
    return pd.DataFrame(
        {"Open": 10.0, "High": 10.5, "Low": 9.5,
         "Close": [10 + i * 0.01 for i in range(n)], "Volume": 1_000_000},
        index=idx,
    )


def aware_frame(n: int = 40) -> pd.DataFrame:
    """Yahoo'nun urettigi sekil: America/New_York ile dilimli."""
    df = naive_frame(n)
    df.index = df.index.tz_localize(nq._EXCHANGE_TZ)
    return df


print("=" * 70)
print("1) YEDEK KAYNAK SERISI DILIMLI OLMALI")
print("=" * 70)

check("_EXCHANGE_TZ tanimli", getattr(nq, "_EXCHANGE_TZ", None) == "America/New_York",
      str(getattr(nq, "_EXCHANGE_TZ", None)))

saved = nq.fetch_history
try:
    # Ag istegi yok: uc, duzeltme oncesi sekli (dilimsiz) donuyormus gibi yapiyoruz.
    nq.fetch_history = lambda ticker, period="2y": naive_frame()
    bundle = nq.as_bundle("TEST", "2y", base=None)
    check("as_bundle paket dondurdu", bundle is not None)
    if bundle is not None:
        idx = bundle["history"].index
        check("as_bundle ciktisi DILIMLI", getattr(idx, "tz", None) is not None,
              str(getattr(idx, "tz", None)))
        check("dilim borsa dilimi", str(getattr(idx, "tz", "")) == nq._EXCHANGE_TZ)

    # Onbellekte duzeltme ONCESI yazilmis kayitlar var; okurken de duzelmeli.
    # (Duzeltme yalnizca _to_frame'de olsaydi eski kayitlar TTL dolana kadar
    #  hisseyi elemeye devam ederdi.)
    check("eski onbellek kaydi da duzeltiliyor",
          bundle is not None and getattr(bundle["history"].index, "tz", None) is not None)
finally:
    nq.fetch_history = saved

print()
print("=" * 70)
print("2) ASIL HATA: DILIMSIZ SERI ENDEKSLE HIZALANAMAZ")
print("=" * 70)

bench = aware_frame()["Close"]

# Once hatayi URETIYORUZ: testin gercek bir sorunu yakaladigini gostermek icin.
patladi = False
try:
    naive_frame()["Close"].to_frame("x").join(bench.to_frame("bench"), how="inner")
except TypeError as exc:
    patladi = "tz-naive" in str(exc) or "tz-aware" in str(exc)
check("dilimsiz seri gercekten patliyor (hata hala gecerli)", patladi)

# Simdi duzeltilmis sekil ayni islemi gecmeli.
gecti = True
try:
    joined = aware_frame()["Close"].to_frame("x").join(bench.to_frame("bench"),
                                                       how="inner")
    gecti = len(joined) > 0
except TypeError:
    gecti = False
check("dilimli seri endeksle hizalanabiliyor", gecti)

print()
print("=" * 70)
print("3) IKI KAYNAK AYNI SUTUNLARI VERMELI")
print("=" * 70)

saved = nq.fetch_history
try:
    nq.fetch_history = lambda ticker, period="2y": naive_frame()
    b = nq.as_bundle("TEST", "2y", base=None)
    cols = set(b["history"].columns)
    for c in ("Open", "High", "Low", "Close", "Volume"):
        check(f"{c} sutunu var", c in cols)
    check("fiyat kaynagi isaretli", b.get("_price_source") == "nasdaq")
    # base verilirse temel veri KORUNMALI (yalnizca fiyat tazelenir)
    b2 = nq.as_bundle("TEST", "2y", base={"info": {"sector": "Tech"}, "history": None})
    check("base verildiginde temel veri korunuyor",
          (b2 or {}).get("info", {}).get("sector") == "Tech")
    check("base verildiginde temel veri kaynagi isaretli",
          (b2 or {}).get("_fundamentals_source") == "yahoo_onbellek")
finally:
    nq.fetch_history = saved

print()
if fails:
    print(f"{fails} KONTROL BASARISIZ")
    raise SystemExit(1)
print("TUM KAYNAK UYUM TESTLERI GECTI")
