"""Kripto piyasa rejimi: boga, ayi, karisik.

NEDEN
-----
Kesitsel model butun rejimlerde ayni agirliklarla calisiyor ve tek bir
ortalama rapor ediyor. Ama bir siralamanin cokerken de yukselirken de ayni
ise yaramasi icin bir sebep yok: kriptoda ayi piyasasinda her sey birlikte
duser, kesit daralir ve "hangisi daha iyi" sorusunun cevabi degersizlesir.

Bu modul rejimi ETIKETLER. Strateji degistirmez, hisse secmez; olcumun
rejim rejim okunabilmesini saglar.

`src/regime.py` ile karistirilmamali: o, hisse tarafinda SPY'a bakiyor ve
panoya baglam yaziyor. Burasi kripto icin ve olcume giriyor.

GERIYE BAKIYOR
--------------
Rejim, o gunun KAPANISINDA bilinebilecek seylerden kuruluyor: BTC'nin 200
gunluk ortalamasina gore yeri, o ortalamanin son 21 gundeki egimi, ve
paritelerin yuzde kacinin kendi 50 gunluk ortalamasinin ustunde oldugu.
Hicbiri gelecege bakmiyor -- bakan bir rejim etiketi, butun rejim
ayrimini anlamsiz kilardi cunku "cokus rejimi" etiketini cokusu gorduken
sonra koymus olurduk.

UCE BOLMEK BIR TESTTIR
----------------------
Uc kovaya bolup birinde iyi sonuc bulmak, uc ayri test yapmaktir. Her
kovada null kontrolu ayri bakilmali; yoksa sansin uc denemeden birini
sectigini goremeyiz.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def seri(btc: pd.Series, genislik: "pd.Series | None" = None,
         uzun: int = 200, egim: int = 21) -> pd.DataFrame:
    """Gun bazinda rejim etiketi ve onu kuran olcumler.

    boga   : BTC uzun ortalamanin USTUNDE ve ortalama YUKSELIYOR
    ayi    : BTC uzun ortalamanin ALTINDA ve ortalama DUSUYOR
    karisik: gerisi (donus bolgeleri, yatay)

    Genislik verilirse ucuncu bir kosul olarak giriyor: endeks yukarida ama
    paritelerin yalnizca ucte biri kendi ortalamasinin ustundeyse, bu bir
    boga degil birkac buyuk ismin tasidigi bir yukselistir.
    """
    b = pd.Series(btc).astype(float).sort_index()
    ma = b.rolling(uzun, min_periods=uzun).mean()
    d = pd.DataFrame(index=b.index)
    d["btc_ma_uzaklik"] = b / ma - 1.0
    d["btc_ma_egim"] = ma / ma.shift(egim) - 1.0
    ust = d["btc_ma_uzaklik"] > 0
    yukari = d["btc_ma_egim"] > 0

    if genislik is not None:
        g = pd.Series(genislik).reindex(d.index).ffill()
        d["genislik"] = g
        genis_iyi = g > 0.5
        genis_kotu = g < 0.35
    else:
        d["genislik"] = np.nan
        genis_iyi = pd.Series(True, index=d.index)
        genis_kotu = pd.Series(True, index=d.index)

    rejim = pd.Series("karisik", index=d.index, dtype=object)
    rejim[ust & yukari & genis_iyi] = "boga"
    rejim[(~ust) & (~yukari) & genis_kotu] = "ayi"
    # Ortalama henuz tanimsizken rejim de tanimsiz. "karisik" demek, bilgi
    # yokken bir rejim uydurmak olurdu.
    rejim[ma.isna()] = "bilinmiyor"
    d["rejim"] = rejim
    return d


def panele_ekle(yol, btc_seri: pd.Series, genislik=None,
                parca: int = 250_000) -> dict:
    """Kripto paneline rejim sutunu ekler."""
    from pathlib import Path

    yol = Path(yol)
    if not yol.exists():
        return {"ok": False, "reason": f"{yol} yok"}
    r = seri(btc_seri, genislik)
    harita = r["rejim"]
    harita.index = pd.DatetimeIndex(harita.index).normalize()
    harita = harita[~harita.index.duplicated(keep="last")]

    gecici = yol.with_suffix(".rejimli.csv")
    gecici.unlink(missing_ok=True)
    yazilan = 0
    sayim: dict = {}
    for blok in pd.read_csv(yol, chunksize=parca, parse_dates=["tarih"]):
        blok = blok.drop(columns=[c for c in ("rejim",) if c in blok.columns])
        rj = harita.reindex(
            pd.DatetimeIndex(blok["tarih"]).normalize()).to_numpy()
        blok["rejim"] = pd.Series(rj, index=blok.index).fillna("bilinmiyor")
        for k, v in blok["rejim"].value_counts().items():
            sayim[k] = sayim.get(k, 0) + int(v)
        blok.to_csv(gecici, index=False, mode="a", header=not yazilan,
                    float_format="%.6g")
        yazilan += len(blok)
        del blok
    gecici.replace(yol)
    return {"ok": True, "yol": str(yol), "satir": yazilan, "dagilim": sayim}
