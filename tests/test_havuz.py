"""Benzer sirketlerden test havuzu kurulumu.

Havuzun tek isi HOMOJENLIK. Testler bunu iddia olarak degil olcum olarak
kovaliyor:

  1. Sert filtreler   : ucuz ve ince hisseler havuza HIC girmemeli
  2. Sektor sarti     : bir havuzda tek sektor olmali
  3. Gercekten dar mi : bilerek iki kumeye ayrilmis sentetik evrende havuz
                        TEK kumeden secilmeli, iki kumeye yayilmamali
  4. Agirlik ise yariyor mu : bir ekseni agirlastirinca havuz o eksende
                        daralmali (agirliklar suslemesi degil)
  5. Kararlilik       : ayni girdi ayni havuzu vermeli

Calistir:  python tests/test_havuz.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import havuz as hv          # noqa: E402

fails = 0


def check(name: str, cond: bool, extra: str = "") -> None:
    global fails
    if cond:
        print(f"  OK    {name}" + (f"  {extra}" if extra else ""))
    else:
        print(f"  HATA  {name}" + (f"  {extra}" if extra else ""))
        fails += 1


def hisse(fiyat: float, hacim: float, oynaklik: float, n: int = 260,
          seed: int = 0) -> pd.DataFrame:
    r = np.random.default_rng(seed)
    c = fiyat * np.exp(np.cumsum(r.normal(0.0005, oynaklik, n)))
    c = c * (fiyat / c[-1])                      # son fiyat hedeflenen olsun
    o = np.concatenate([[c[0]], c[:-1]])
    h = np.maximum(o, c) * (1 + oynaklik)
    l = np.minimum(o, c) * (1 - oynaklik)
    v = np.full(n, hacim / fiyat)
    return pd.DataFrame({"Open": o, "High": h, "Low": l, "Close": c,
                         "Volume": v},
                        index=pd.bdate_range("2024-01-01", periods=n))


def paket(fiyat, hacim, oynaklik, sektor, mcap, seed=0) -> dict:
    return {"history": hisse(fiyat, hacim, oynaklik, seed=seed),
            "info": {"sector": sektor, "marketCap": mcap}}


print("=" * 72)
print("1) NITELIK CIKARIMI")
print("=" * 72)

b = {f"T{i:02d}": paket(50 + i, 5e6, 0.02, "Technology", 2e9, seed=i)
     for i in range(20)}
n = hv.nitelikler(b)
check("nitelik tablosu doldu", len(n) == 20, f"{len(n)} satir")
check("log sutunlari uretildi",
      {"log_mcap", "log_dolar_hacim", "log_fiyat"} <= set(n.columns))
check("dolar hacim makul", bool((n["dolar_hacim"] > 1e6).all()),
      f"medyan {n['dolar_hacim'].median():,.0f}")
check("atr pozitif", bool((n["atr_pct"] > 0).all()))
check("kisa gecmis eleniyor",
      len(hv.nitelikler({"X": {"history": hisse(50, 5e6, 0.02, n=50),
                               "info": {"sector": "Technology"}}})) == 0)

print()
print("=" * 72)
print("2) SERT FILTRELER")
print("=" * 72)

karisik = dict(b)
karisik["UCUZ"] = paket(1.2, 5e6, 0.02, "Technology", 2e9, seed=90)
karisik["INCE"] = paket(60.0, 2e5, 0.02, "Technology", 2e9, seed=91)
karisik["SEKTORSUZ"] = {"history": hisse(60, 5e6, 0.02, seed=92), "info": {}}

d = hv.kur(karisik, boyut=12)
uyeler = set(hv.semboller(d))
check("ucuz hisse havuza girmedi", "UCUZ" not in uyeler)
check("ince hisse havuza girmedi", "INCE" not in uyeler)
check("sektoru bilinmeyen havuza girmedi", "SEKTORSUZ" not in uyeler)

print()
print("=" * 72)
print("3) SEKTOR SARTI")
print("=" * 72)

iki_sektor = {}
for i in range(20):
    iki_sektor[f"TEC{i:02d}"] = paket(50, 5e6, 0.02, "Technology", 2e9, seed=i)
for i in range(20):
    iki_sektor[f"FIN{i:02d}"] = paket(50, 5e6, 0.02, "Financial Services",
                                      2e9, seed=100 + i)
d2 = hv.kur(iki_sektor, boyut=12)
check("iki ayri havuz kuruldu", d2["havuz_sayisi"] == 2,
      str(d2["havuz_sayisi"]))
for h in d2["havuzlar"]:
    onek = "TEC" if h["sektor"] == "Technology" else "FIN"
    check(f"{h['sektor']} havuzunda tek sektor var",
          all(u.startswith(onek) for u in h["uyeler"]),
          str([u for u in h["uyeler"] if not u.startswith(onek)][:3]))

print()
print("=" * 72)
print("4) GERCEKTEN DAR MI — iki kumeli evren")
print("=" * 72)

# Ayni sektorde bilerek IKI ayri kume: kucuk/ince ve buyuk/kalin.
# Dogru davranis, havuzun TEK kumeden secilmesi. Karisik secim, esleme
# yapmiyor demektir.
iki_kume = {}
for i in range(18):                              # kucuk kume
    iki_kume[f"KUCUK{i:02d}"] = paket(12 + i * 0.1, 2e6, 0.045,
                                      "Technology", 3e8, seed=200 + i)
for i in range(18):                              # buyuk kume
    iki_kume[f"BUYUK{i:02d}"] = paket(180 + i, 8e7, 0.012,
                                      "Technology", 6e10, seed=300 + i)
d3 = hv.kur(iki_kume, boyut=14)
uy = hv.semboller(d3)
kucuk = sum(1 for u in uy if u.startswith("KUCUK"))
buyuk = sum(1 for u in uy if u.startswith("BUYUK"))
check("havuz tek kumeden secildi", kucuk == 0 or buyuk == 0,
      f"kucuk {kucuk}, buyuk {buyuk}")
h3 = d3["havuzlar"][0]
check("mcap ekseninde evrenden belirgin dar",
      h3["daralma"]["log_mcap"] < 0.4, str(h3["daralma"]["log_mcap"]))
check("hacim ekseninde evrenden belirgin dar",
      h3["daralma"]["log_dolar_hacim"] < 0.4,
      str(h3["daralma"]["log_dolar_hacim"]))

print()
print("=" * 72)
print("5) AGIRLIKLAR GERCEKTEN ETKILI Mi")
print("=" * 72)

# Oynaklikta ayrisan, buyuklukte ayni bir evren. Oynaklik agirligi
# yukseltilince havuz oynaklik ekseninde DAHA DAR olmali.
oyn = {}
for i in range(30):
    oyn[f"O{i:02d}"] = paket(50, 5e6, 0.008 + 0.0025 * i, "Technology",
                             2e9, seed=400 + i)

az = hv.kur(oyn, boyut=12, agirlik={"log_mcap": 0.9, "log_dolar_hacim": 0.05,
                                    "atr_pct": 0.01, "log_fiyat": 0.04})
cok = hv.kur(oyn, boyut=12, agirlik={"log_mcap": 0.05, "log_dolar_hacim": 0.05,
                                     "atr_pct": 0.85, "log_fiyat": 0.05})
d_az = az["havuzlar"][0]["dagilim"]["atr_pct"]["havuz"]["ceyrekler_arasi"]
d_cok = cok["havuzlar"][0]["dagilim"]["atr_pct"]["havuz"]["ceyrekler_arasi"]
check("oynaklik agirligi artinca o eksen daraliyor", d_cok < d_az,
      f"agirlik dusukken {d_az:.4f} -> yuksekken {d_cok:.4f}")

print()
print("=" * 72)
print("6) KARARLILIK VE YAPI")
print("=" * 72)

check("ayni girdi ayni havuzu veriyor",
      hv.semboller(hv.kur(iki_kume, boyut=14)) == uy)
check("havuz boyutu istenen kadar",
      all(h["boyut"] == 14 for h in d3["havuzlar"]),
      str([h["boyut"] for h in d3["havuzlar"]]))
check("uyeler benzersiz", len(set(uy)) == len(uy))
check("dagilim uc seviyeyi de tasiyor",
      set(h3["dagilim"]["log_mcap"]) == {"havuz", "sektor", "evren"})
check("agirliklar ciktida kayitli", d3["agirlik"] == hv.AGIRLIK)
check("yetersiz adayda havuz kurulmuyor",
      hv.kur({f"A{i}": paket(50, 5e6, 0.02, "Technology", 2e9, seed=i)
              for i in range(5)}, boyut=25)["havuz_sayisi"] == 0)
check("bos girdide ok=False", hv.kur({})["ok"] is False)

with tempfile.TemporaryDirectory() as td:
    p = Path(td) / "havuz.json"
    hv.kaydet(d3, p)
    geri = hv.yukle(p)
    check("kayit/okuma ayni", hv.semboller(geri) == uy)
    check("olmayan dosyada None", hv.yukle(Path(td) / "yok.json") is None)
    check("havuz kimligiyle suzme calisiyor",
          set(hv.semboller(geri, havuz_id=d3["havuzlar"][0]["id"]))
          == set(d3["havuzlar"][0]["uyeler"]))

print()
print("=" * 72)
print("7) DURUM NITELIGI KULLANILMIYOR")
print("=" * 72)

# Havuz secimi ZAMANLA DEGISEN niteliklere bakmamali. Bunu davranisla
# kontrol etmek zor, ama kaynakta trend/momentum sinyali kullanilmadigini
# gorebiliriz: nitelikler() yalnizca uzun pencereli medyanlar uretiyor.
import re                                        # noqa: E402
kaynak = (ROOT / "src" / "havuz.py").read_text(encoding="utf-8")
govde = kaynak.split("def nitelikler")[1].split("def _uygun")[0]
check("nitelik cikariminda trend/momentum gostergesi yok",
      not re.search(r"\b(sma|ema|rsi|macd|momentum|donchian)\s*\(", govde),
      "nitelikler() yalnizca fiyat/hacim/atr kullanmali")
check("nitelikler son gunun degil TIPIK halin olcusu",
      "tail(120)" in govde)

print()
if fails:
    print(f"{fails} KONTROL BASARISIZ")
    raise SystemExit(1)
print("TUM HAVUZ TESTLERI GECTI")
