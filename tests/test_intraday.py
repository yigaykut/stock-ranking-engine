"""Gun ici veri katmani — sinirlari VARSAYMAK yerine OLCMEK.

Bu modulun en kritik iki davranisi:

  1. `yf.download` cok seviyeli sutun donduruyor. Duzlestirilmezse
     df["Close"] bir Series degil DataFrame olur; gostergeler patlamaz,
     SESSIZCE sacmalar. Bu, yakalanmasi en zor hata tipi.

  2. Zaman dilimi karari BAR sayisina degil FARKLI GUN sayisina bakmali.
     1 dakikalik veride 2730 bar olabilir; hepsi 7 gunden geliyorsa bagimsiz
     gozlem sayisi 7'dir. Bar sayisina bakan bir kural, en olculemez araligi
     en iyi sanir.

Ag istegi YAPILMAZ: hepsi sentetik veriyle.

Calistir:  python tests/test_intraday.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import intraday as idy      # noqa: E402

fails = 0


def check(name: str, cond: bool, extra: str = "") -> None:
    global fails
    if cond:
        print(f"  OK    {name}" + (f"  {extra}" if extra else ""))
    else:
        print(f"  HATA  {name}" + (f"  {extra}" if extra else ""))
        fails += 1


def saatlik(gun: int, bar_gun: int = 7, sembol: str = "AAPL",
            coklu: bool = True) -> pd.DataFrame:
    """yf.download ciktisini taklit eder (cok seviyeli sutunlarla)."""
    idx = []
    for g in pd.bdate_range("2025-01-01", periods=gun):
        for b in range(bar_gun):
            idx.append(pd.Timestamp(g) + pd.Timedelta(hours=9 + b))
    n = len(idx)
    r = np.random.default_rng(0)
    c = 100 * np.exp(np.cumsum(r.normal(0, 0.002, n)))
    veri = {"Open": c * 0.999, "High": c * 1.002, "Low": c * 0.998,
            "Close": c, "Volume": np.full(n, 1e5)}
    if coklu:
        kol = pd.MultiIndex.from_product([list(veri), [sembol]])
        return pd.DataFrame(np.column_stack(list(veri.values())),
                            index=pd.DatetimeIndex(idx), columns=kol)
    return pd.DataFrame(veri, index=pd.DatetimeIndex(idx))


print("=" * 72)
print("1) COK SEVIYELI SUTUN DUZLESTIRME")
print("=" * 72)

ham = saatlik(5)
check("girdi gercekten cok seviyeli",
      isinstance(ham.columns, pd.MultiIndex))
d = idy._duzlestir(ham, "AAPL")
check("sutunlar duzlesti", not isinstance(d.columns, pd.MultiIndex))
check("beklenen sutunlar var",
      list(d.columns) == ["Open", "High", "Low", "Close", "Volume"],
      str(list(d.columns)))
check("Close bir Series (DataFrame DEGIL)",
      isinstance(d["Close"], pd.Series), type(d["Close"]).__name__)
check("tek seviyeli girdi bozulmuyor",
      list(idy._duzlestir(saatlik(3, coklu=False), "AAPL").columns)
      == ["Open", "High", "Low", "Close", "Volume"])
check("bilinmeyen sembolde de duzlesiyor",
      not isinstance(idy._duzlestir(ham, "YOK").columns, pd.MultiIndex))

print()
print("=" * 72)
print("2) KAPSAM OLCUMU")
print("=" * 72)

o = idy.olcum(d)
check("bar sayisi dogru", o["bar"] == 35, str(o["bar"]))
check("farkli gun sayisi dogru", o["gun"] == 5, str(o["gun"]))
check("bar/gun dogru", o["bar_gun"] == 7.0, str(o["bar_gun"]))
check("tarih araligi var", o["ilk"] <= o["son"])
check("bos girdide sifir", idy.olcum(None)["bar"] == 0)
check("bos DataFrame'de sifir", idy.olcum(pd.DataFrame())["gun"] == 0)

print()
print("=" * 72)
print("3) ARALIK SECIMI — gun sayisina bakmali, bara degil")
print("=" * 72)

# 1 dakikalik: cok bar, az gun. Saatlik: az bar/gun ama cok gun.
k = {"araliklar": {
    "1m": {"gun": 7, "bar": 2730},
    "5m": {"gun": 60, "bar": 4680},
    "15m": {"gun": 60, "bar": 1560},
    "1h": {"gun": 500, "bar": 3500},
}}
oneri = idy.onerilen_aralik(k)
check("saatlik secildi", oneri["birincil"] == "1h", str(oneri.get("birincil")))
check("en cok BAR veren 5m secilmedi", oneri["birincil"] != "5m")
check("yetersiz araliklar adaylardan cikti",
      set(oneri["adaylar"]) == {"1h"}, str(oneri["adaylar"]))

# Hicbiri yeterli degilse oneri verilmemeli
az = {"araliklar": {"1m": {"gun": 7, "bar": 2730},
                    "15m": {"gun": 60, "bar": 1560}}}
o2 = idy.onerilen_aralik(az)
check("hicbiri yeterli degilse ok=False", o2["ok"] is False)
check("sebep aciklaniyor", "gun" in o2["reason"], o2["reason"][:60])
check("kapsam yoksa ok=False", idy.onerilen_aralik({})["ok"] is False)

# Esit gun sayisinda daha cok bar veren (daha ince kesit) tercih edilmeli
esit = {"araliklar": {"1h": {"gun": 500, "bar": 3500},
                      "30m": {"gun": 500, "bar": 7000}}}
check("esit gunde daha ince aralik tercih ediliyor",
      idy.onerilen_aralik(esit)["birincil"] == "30m")

print()
print("=" * 72)
print("4) KAPSAM KAYDI — eski olcumler korunuyor")
print("=" * 72)

with tempfile.TemporaryDirectory() as td:
    p = Path(td) / "kapsam.json"
    idy.kapsam_kaydet({"sembol": "SPY",
                       "araliklar": {"1h": {"gun": 500, "bar": 3500}}}, p)
    idy.kapsam_kaydet({"sembol": "SPY",
                       "araliklar": {"1h": {"gun": 300, "bar": 2100}}}, p)
    g = idy.kapsam_yukle(p)
    check("son olcum guncel", g["araliklar"]["1h"]["gun"] == 300)
    check("onceki olcum gecmiste duruyor", len(g["gecmis"]) == 1,
          str(len(g.get("gecmis", []))))
    check("gecmisteki olcum dogru",
          g["gecmis"][0]["araliklar"]["1h"]["gun"] == 500)
    check("olmayan dosyada None", idy.kapsam_yukle(Path(td) / "yok.json") is None)

print()
print("=" * 72)
print("5) YAPILANDIRMA")
print("=" * 72)

check("her aralik icin istek suresi tanimli",
      set(idy.ISTEK) == {"1m", "5m", "15m", "30m", "1h"}, str(sorted(idy.ISTEK)))
check("MIN_GUN anlamli bir esik", idy.MIN_GUN >= 60, str(idy.MIN_GUN))
check("istekler arasi bekleme var", idy.BEKLEME > 0)
check("gun ici TTL gunluk veriden kisa", idy.TTL <= 24 * 3600)

# Hiz siniri yayilmali: havuz_cek DURMALI, sessizce bos donmemeli
import re                                        # noqa: E402
kaynak = (ROOT / "src" / "intraday.py").read_text(encoding="utf-8")
check("hiz siniri yakalanip DURULUYOR",
      "except RateLimited" in kaynak and "hiz siniri" in kaynak)
check("Ticker.history kullanilmiyor (bu surumde bozuk)",
      not re.search(r"\.history\s*\(", kaynak))
check("yf.download kullaniliyor", "yf.download" in kaynak)

print()
if fails:
    print(f"{fails} KONTROL BASARISIZ")
    raise SystemExit(1)
print("TUM GUN ICI TESTLERI GECTI")
