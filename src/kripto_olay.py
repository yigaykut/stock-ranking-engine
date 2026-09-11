"""Kriptoda "olay": listelenme yasi.

NEDEN BOYLE
-----------
Hisse tarafinda olay katmani SEC 8-K bildirimleriydi: sirketin "maddi bir
sey oldu" diye kendi duyurusu, saniye hassasiyetinde zaman damgali. Kriptoda
bunun karsiligi yok. En yakini borsanin kendi duyurulari ve Binance'in
arsivi gercekten 2017-07'ye kadar gidiyor (2.253 listeleme, 431 delisting
duyurusu) -- ama uc birkac istekten sonra bos govdeyle 400 donmeye basladi,
yani duzenli bir besleme olarak guvenilmez.

Elde kalan ve ONUN KADAR kesin olan bir olay var: paritenin BORSAYA GIRDIGI
gun. Barlarin ilki tam olarak odur. Ucretsiz, tam, zaten indirilmis, ve
zaman damgasi tartismasiz.

Yeni listelenen bir coinin ilk haftalari literaturde ayri davraniyor --
duyuru sicramasi, sonrasinda surukleme. Kesitte "bu parite ne kadar
zamandir burada" sorusu, hisse tarafindaki "son bildirimden bu yana kac
gun" sorusunun dogrudan esi.

ILERIYE BAKMAYAN TARAF
----------------------
Listelenme gunu GECMISTE. Paritenin KAPANDIGI gun ise gelecekte ve
ozelliklerde KULLANILMAZ. Ayarttir: elimizde olu paritelerin tam omru var,
yani "kac gun sonra kapanacak" hesaplanabilir ve muhtesem bir tahminci
olurdu. Tamamen ileriye bakis olurdu. Bu dosyada o sutun yok ve olmayacak.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def ozellikler(gunler: pd.DatetimeIndex, tavan: int = 365) -> pd.DataFrame:
    """Bar bazinda listelenme yasi ozellikleri.

    - kripto_yas      : kacinci bar (log olcekli, tavanli)
    - kripto_yeni30   : ilk 30 bar icinde mi
    - kripto_yeni90   : ilk 90 bar icinde mi

    Yas log olcekli, cunku 10. gun ile 40. gun arasindaki fark, 800. gun ile
    830. gun arasindakinden cok daha anlamli; ham sayi kesitte butun eski
    pariteleri tek bir uca yigar.
    """
    n = len(gunler)
    d = pd.DataFrame(index=pd.DatetimeIndex(gunler))
    if n == 0:
        for c in ("kripto_yas", "kripto_yeni30", "kripto_yeni90"):
            d[c] = pd.Series(dtype="float32")
        return d
    bar = np.arange(n, dtype=float)
    d["kripto_yas"] = np.log1p(np.minimum(bar, tavan)).astype(np.float32)
    d["kripto_yeni30"] = (bar < 30).astype(np.float32)
    d["kripto_yeni90"] = (bar < 90).astype(np.float32)
    return d


def panele_ekle(yol, baslangic: str = "2017-08-01", parca: int = 250_000
                ) -> dict:
    """Var olan kripto paneline listelenme sutunlarini ekler.

    `olay.panele_ekle` ile ayni gerekce: bu sutunlar yalnizca (parite, tarih)
    ciftine bagli, gostergelerin hesabina hicbir sey borclu degiller. Agir
    dongunun icinde durmalari icin bir sebep yok.

    Yas, paritenin TAM barlari uzerinden sayiliyor, panelin tarihleri
    uzerinden degil: kesit paneli her besinci gunu tutuyor ve "kacinci bar"
    orada sayilsaydi butun yaslar bese bolunurdu.
    """
    from pathlib import Path

    from . import kripto as kr

    yol = Path(yol)
    if not yol.exists():
        return {"ok": False, "reason": f"{yol} yok"}

    anahtar = pd.read_csv(yol, usecols=["ticker", "tarih"],
                          parse_dates=["tarih"])
    parcalar, kapsanan = [], 0
    for tk, alt in anahtar.groupby("ticker", sort=False):
        bar = kr.oku(tk, baslangic)
        if bar is None or not len(bar):
            continue
        f = ozellikler(pd.DatetimeIndex(bar.index).normalize())
        alt_f = f.reindex(pd.DatetimeIndex(alt["tarih"]).normalize())
        alt_f.index = alt.index
        parcalar.append(alt_f)
        kapsanan += 1
    if not parcalar:
        return {"ok": False, "reason": "hicbir parite eslesmedi"}

    tablo = pd.concat(parcalar).reindex(anahtar.index)
    del parcalar
    # Eslesmeyen satir "cok eski" sayilir: bir paritenin barlarinda
    # bulunmayan bir tarih, o parite icin zaten yoktur.
    varsayilan = {"kripto_yas": float(np.log1p(365)), "kripto_yeni30": 0.0,
                  "kripto_yeni90": 0.0}
    for c in tablo.columns:
        tablo[c] = tablo[c].fillna(varsayilan[c]).astype("float32")

    gecici = yol.with_suffix(".yasli.csv")
    gecici.unlink(missing_ok=True)
    yazilan = 0
    for blok in pd.read_csv(yol, chunksize=parca):
        blok = blok.drop(columns=[c for c in tablo.columns
                                  if c in blok.columns])
        birlesik = pd.concat(
            [blok.reset_index(drop=True),
             tablo.iloc[yazilan:yazilan + len(blok)].reset_index(drop=True)],
            axis=1)
        birlesik.to_csv(gecici, index=False, mode="a", header=not yazilan,
                        float_format="%.6g")
        yazilan += len(blok)
        del blok, birlesik
    gecici.replace(yol)
    return {"ok": True, "yol": str(yol), "satir": yazilan,
            "parite": kapsanan, "sutun": list(tablo.columns)}
