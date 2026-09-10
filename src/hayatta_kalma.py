"""Hayatta kalma yanliligini OLCER (gideremez).

NEDEN
-----
Bu projede olculen her pozitif sonuc ayni yere cikti: kenar, en ince
isimlerde. En son kesitsel modelde de oyleydi -- dolar hacmi ve amihud
cikarilinca dilimler tersine dondu. Aciklama iki tane ve iceriden ayni
gorunuyorlar: ya likidite primi, ya fiyat onbelleginin yalnizca BUGUN kote
olan sirketleri tutmasi.

Ayirt etmenin yolu, olenleri saymak. Fiyatlari yok ama SEC kayitlari var.

DUZELTILEMIYOR
--------------
Yahoo kote disi sembollere 404 donuyor (TWTR, ATVI, CERN, XLNX, SIVB, FRC
denendi). Daha kotusu, bazi sembollere 200 donuyor ama bunlar YENIDEN
KULLANILMIS semboller: SBNY 2024-08'de, BBBY 2026-07'de basliyor, ikisi de
bambaska sirket. Korlemesine cekmek, olmus bir sirketin yerine yenisinin
fiyatlarini koymak olurdu -- yanliligi duzeltmek yerine veriyi bozmak.

NASIL OLCULUYOR
---------------
Kohort: 2016 sonunda XBRL ile bilanco raporlayan her sirket (Assets
CY2016Q4I, tek istekte 6.672 sirket). Hayatta kalan: 2025 sonunda hala
raporlayan. Ikisi de tek bir uctan geliyor, hicbir sey ornekleme degil.

Buyukluk olcusu GELIR degil TOPLAM VARLIK. Geliri olmayan biyoteknoloji
sirketleri gelir cercevesinde hic gorunmuyor ve mikro-kap bandi tam orasi.

"Raporlamayi birakmak" olmek demek DEGIL: satin alinmak da bir cikis ve
genellikle primli. Yanlilik bu yuzden tek yonlu degil -- hem batanlari hem
satin alinanlari siliyor. Ayirmak icin cikan sirketin son bildirimlerine
bakiliyor: 2.01 (devralma tamamlandi) satin alinma, 3.01 (kotasyon
uyumsuzlugu) batma isareti.
"""
from __future__ import annotations

import pandas as pd
import requests

from .providers import cache as _cache
from .olay import _tanitici, cik_haritasi, oku as olay_oku

CERCEVE = "https://data.sec.gov/api/xbrl/frames/us-gaap/Assets/USD/"
ALAN = "sec_kohort"


def _cerceve(donem: str, yenile: bool = False) -> pd.DataFrame:
    """Bir donemde bilanco raporlayan her sirket: CIK, ad, toplam varlik."""
    if not yenile:
        hit = _cache.peek(ALAN, donem)
        if hit and hit[0] is not None:
            return pd.DataFrame(hit[0])
    r = requests.get(f"{CERCEVE}{donem}.json",
                     headers={"User-Agent": _tanitici(),
                              "Accept-Encoding": "gzip, deflate"}, timeout=60)
    r.raise_for_status()
    d = r.json().get("data") or []
    df = pd.DataFrame({"cik": [x["cik"] for x in d],
                       "ad": [x.get("entityName", "") for x in d],
                       "varlik": [float(x["val"]) for x in d]})
    _cache.put(ALAN, donem, df.to_dict("list"))
    return df


def kohort(baslangic: str = "CY2016Q4I", bitis: str = "CY2025Q4I",
           period: str = "10y", kova: int = 10) -> dict:
    """Buyukluge gore on yillik hayatta kalma orani.

    Ucuncu sutun asil mesele: kohorttaki sirketlerin kaci BUGUN fiyat
    onbelleginde. Model yalnizca onlari goruyor.
    """
    k = _cerceve(baslangic)
    yasayan = set(_cerceve(bitis)["cik"])
    if k.empty:
        return {"ok": False, "reason": "kohort bos"}
    k = k[k["varlik"] > 0].copy()
    k["yasiyor"] = k["cik"].isin(yasayan)

    harita = cik_haritasi()
    onbellekte = {int(harita[tk]) for tk in harita
                  if _cache.peek("yahoo", f"{tk}:{period}") is not None}
    k["onbellekte"] = k["cik"].isin(onbellekte)
    k["kova"] = pd.qcut(k["varlik"], kova, labels=False, duplicates="drop")

    t = (k.groupby("kova")
         .agg(n=("cik", "size"), ortanca_varlik=("varlik", "median"),
              yasayan=("yasiyor", "mean"), onbellekte=("onbellekte", "mean"))
         .reset_index())
    return {
        "ok": True, "baslangic": baslangic, "bitis": bitis,
        "kohort": int(len(k)),
        "yasayan_oran": round(float(k["yasiyor"].mean()), 4),
        "onbellek_oran": round(float(k["onbellekte"].mean()), 4),
        "kovalar": [{"kova": int(r["kova"]), "n": int(r["n"]),
                     "ortanca_varlik": float(r["ortanca_varlik"]),
                     "yasayan": round(float(r["yasayan"]), 4),
                     "onbellekte": round(float(r["onbellekte"]), 4)}
                    for _, r in t.iterrows()],
    }


def cikis_nedeni(cikler, ornek: int = 400) -> dict:
    """Cikanlar batti mi, satin mi alindi.

    Onbellekteki 8-K kayitlarindan bakiliyor, yani yalnizca bir zamanlar
    evrende olup sonradan dusen semboller icin cevap verebilir. Hic
    girmemis sirketler icin bildirim indirmek gerekir; bu fonksiyon ag
    istegi YAPMAZ, elindekiyle cevap verir.
    """
    satin = batma = belirsiz = bakilan = 0
    for tk in list(cikler)[:ornek]:
        d = olay_oku(tk)
        if d is None or d.empty:
            continue
        bakilan += 1
        son = " ".join(d.tail(6)["maddeler"].astype(str))
        if "2.01" in son:
            satin += 1
        elif "3.01" in son:
            batma += 1
        else:
            belirsiz += 1
    return {"bakilan": bakilan, "satin_alinma": satin, "kotasyon_sorunu": batma,
            "belirsiz": belirsiz}
