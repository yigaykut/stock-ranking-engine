"""Meta-model: "bu kurulum TUTACAK mi?"

NE OGRENIYOR
------------
Kurulum dedektorleri birincil sinyali uretir: "burada bir cekic olustu".
Kalibrasyon o sinyalin gecmiste ne yaptigini SAYAR ve kova bazinda bir
olasilik verir: "orta oynaklikta, 5 gunde, %52".

Bu modul bir kat daha asagi iniyor: ayni kurulum, ayni kova icinde bile bazi
gunler tutup bazi gunler tutmuyor olabilir. Model, sinyal anindaki HAM
ozelliklerden (ATR%, RSI, hacim orani, MA uzakligi, gucu...) o ayrimi
ogrenmeye calisiyor.

Yazindaki adi META-ETIKETLEME: birincil sinyal kaliptan gelir, ikincil model
o sinyalin tutup tutmayacagini kestirir. Sorusu "hangi hisse yukselecek"
degil "bu kurulum bu ortamda tutar mi" -- cok daha dar ve cok daha
ogrenilebilir.

TABAN CIZGISI 0.5 DEGIL, KOVANIN KENDISI
----------------------------------------
Modelin "iyi" sayilmasi icin yazi-turadan iyi olmasi yetmez. Zaten elimizde
calisan bir cevap var: kova bazli kalibrasyon. Model, ONUN uzerine bir sey
koymak zorunda. Bu yuzden butun karsilastirmalar kova olasiligina karsi
yapiliyor.

Olcut Brier skoru: ortalama (tahmin - gerceklesen)^2. Olasilik tahminleri icin
uygun (proper) bir skor -- yani en iyi skoru, gercek olasiligi soyleyerek
alirsin; blof yapmak cezalandirilir.

FARKIN ANLAMLILIGI — buradaki en onemli nokta
---------------------------------------------
"Model Brier'i 0.2481'den 0.2467'ye dusurdu" cumlesi tek basina hicbir sey
anlatmaz. 100 bin satirda olculmus olabilir ama o satirlar bagimsiz degildir:
ayni gunun sinyalleri ayni piyasa gununu yasar, ardisik gunlerin sonuclari
ortusur. Sistemin her yerinde oldugu gibi burada da olcum GUN BAZINDA
yapiliyor: her gun icin (kova_brier - model_brier) farki hesaplaniyor, sonra o
gunluk seride Newey-West duzeltmeli t degeri aliniyor.

SIZINTI
-------
Ileri yuruyus, arindirma (purge) ve tampon (embargo) ile. Egitim kumesi, test
penceresiyle ETIKETI ORTUSEN satirlari icermez. Bu yapilmazsa model, test
doneminin sonucunu egitimde gormus olur ve skor sahte cikar.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"

# Modelin egitilebilmesi icin gereken en az satir. Altinda sonuc raporlanmaz.
MIN_SATIR = 2000

# Egitimde kullanilmayacak sutunlar: kimlik ve etiketler.
KIMLIK = ("ticker", "tarih", "zaman", "frekans", "kurulum", "yon")


def cikti_yolu(frekans: str, dizi: bool = False,
               etiket: str = "kazanc") -> Path:
    ek = "_dizi" if dizi else ""
    et = "" if etiket == "kazanc" else f"_{etiket}"
    return DATA / f"meta_model_{frekans}{ek}{et}.json"


# =============================================================================
#  Veri
# =============================================================================
def panel_yukle(frekans: str = "1d") -> pd.DataFrame | None:
    from . import kalibrasyon as kb

    p = kb.panel_yolu(frekans)
    if not p.exists():
        return None
    df = pd.read_csv(p)
    if "tarih" in df.columns:
        df["tarih"] = pd.to_datetime(df["tarih"], errors="coerce")
    return df.dropna(subset=["tarih"])


def _ozellik_sutunlari(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns
            if c not in KIMLIK
            and not c.startswith(("fazla_", "kazanc_", "bariyer", "akran_"))]


def hazirla(df: pd.DataFrame, ufuk: int,
            etiket_tipi: str = "kazanc") -> dict | None:
    """Bir ufuk icin X, y, tarih ve kova taban olasiligi.

    Kategorik sutunlar (kurulum, oynaklik, likidite, trend_konumu) one-hot
    ediliyor. Kurulum kimliginin ozellik olmasi SART: model "hangi kurulum"
    bilgisini gormeden, kurulumlar arasi farki ogrenemez.
    """
    etiket = f"{etiket_tipi}_{ufuk}g"
    if etiket not in df.columns:
        return None
    alt = df[df[etiket].notna()].copy()
    if len(alt) < MIN_SATIR:
        return None

    oz = _ozellik_sutunlari(alt)
    kategorik = [c for c in ("oynaklik", "likidite", "trend_konumu")
                 if c in alt.columns]
    sayisal = [c for c in oz if c not in kategorik]

    X = alt[sayisal].apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True)).fillna(0.0)

    parcalar = [X]
    for c in kategorik + ["kurulum"]:
        if c in alt.columns:
            parcalar.append(pd.get_dummies(alt[c].astype(str), prefix=c[:4]))
    XX = pd.concat(parcalar, axis=1).astype(np.float32)

    zaman = (pd.DatetimeIndex(pd.to_datetime(alt["zaman"], errors="coerce"))
             if "zaman" in alt.columns else None)
    # The label says whether it worked; the return says by how much. A model
    # can be right barely more than half the time and still be worth having if
    # its wins are bigger than its losses, and a hit rate alone can't show
    # that. Prefer the peer-demeaned return when it's there.
    getiri_ad = next((c for c in (f"akran_{ufuk}g", f"fazla_{ufuk}g")
                      if c in alt.columns), None)
    getiri = (alt[getiri_ad].to_numpy(dtype=np.float64)
              if getiri_ad else np.zeros(len(alt)))
    if getiri_ad and zaman is not None:
        getiri = _budanmis(getiri, pd.DatetimeIndex(alt["tarih"]).normalize())
    # Some label columns are already 0/1 (kazanc, bariyer). The peer one is a
    # return, and BCE needs a class, so it gets thresholded at zero -- "did it
    # beat its peer group" rather than "by how much". The size is kept
    # separately in `getiri` and that is what the top-decile metric reads.
    ham_y = alt[etiket].to_numpy(dtype=np.float64)
    benzersiz = np.unique(ham_y[np.isfinite(ham_y)])
    if not set(benzersiz.tolist()) <= {0.0, 1.0}:
        ham_y = (ham_y > 0).astype(np.float64)

    return {
        "X": XX.to_numpy(dtype=np.float32),
        "y": ham_y.astype(np.float32),
        "tarih": pd.DatetimeIndex(alt["tarih"]).normalize(),
        "zaman": zaman,
        "ticker": alt["ticker"].to_numpy(),
        "getiri": getiri,
        "getiri_ad": getiri_ad,
        "kurulum": alt["kurulum"].to_numpy(),
        "kosullar": {c: alt[c].astype(str).to_numpy() for c in kategorik},
        "ozellik_adlari": list(XX.columns),
        "satir": len(alt),
    }


def kova_olasiligi(kalib: dict | None, kurulum: np.ndarray, ufuk: int,
                   kosullar: dict, taban: float) -> np.ndarray:
    """Her satir icin KOVA taban cizgisi olasiligi.

    guven() satir satir cagrilirsa yuz binlerce sorgu olur; burada kova
    tablosu bir kez sozluge cevriliyor. Sonuc ayni: once satirin kendi
    kosuluna ait kova, yoksa kurulumun geneli, o da yoksa taban orani.
    """
    n = len(kurulum)
    out = np.full(n, taban, dtype=np.float64)
    if not kalib or not kalib.get("kovalar"):
        return out

    idx = {}
    for k in kalib["kovalar"]:
        if k["ufuk"] == ufuk and k["durum"] == "olculdu":
            idx[(k["kurulum"], k["kosul"], k["deger"])] = k["p"]

    for i in range(n):
        kid = kurulum[i]
        p = None
        for ad, dizi in kosullar.items():
            p = idx.get((kid, ad, dizi[i]))
            if p is not None:
                break
        if p is None:
            p = idx.get((kid, "*", "*"))
        if p is not None:
            out[i] = p
    return out


# =============================================================================
#  Olcutler
# =============================================================================
def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def auc(p: np.ndarray, y: np.ndarray) -> float:
    """ROC egrisi altindaki alan; siralama gucunun olcusu.

    scipy'siz: Mann-Whitney U ile esdeger sira toplami hesabi.
    """
    y = np.asarray(y, float)
    if y.min() == y.max():
        return float("nan")
    sira = pd.Series(p).rank().to_numpy()
    n1 = float(y.sum())
    n0 = float(len(y) - n1)
    return float((sira[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def guvenilirlik(p: np.ndarray, y: np.ndarray, dilim: int = 10) -> list[dict]:
    """Guvenilirlik egrisi: tahmin edilen olasilik vs GERCEKLESEN oran.

    Kullanicinin "hangi durumlarda daha accurate calisiyor" sorusunun
    dogrudan cevabi. Iyi kalibre bir modelde, %60 dedigi satirlarin %60'i
    tutar. Tutmuyorsa modelin sayisi olasilik degil yalnizca siralama olur.
    """
    s = pd.DataFrame({"p": p, "y": y})
    try:
        s["kova"] = pd.qcut(s["p"], dilim, labels=False, duplicates="drop")
    except ValueError:
        return []
    out = []
    for kv_, g in s.groupby("kova"):
        out.append({"dilim": int(kv_), "n": int(len(g)),
                    "tahmin": round(float(g["p"].mean()), 4),
                    "gerceklesen": round(float(g["y"].mean()), 4)})
    return out


def _budanmis(getiri: np.ndarray, gun: pd.DatetimeIndex,
              alt_p: float = 0.01, ust_p: float = 0.99) -> np.ndarray:
    """Clip each day's cross-section at its 1st and 99th percentile.

    Without this the average is a fiction. Over 63 days the peer-excess return
    has a mean of +0.56% and a median of -2.11%: the typical stock loses to its
    group and the positive average comes from a handful of names, one of them
    up 2479%. The top 1% of rows carry four times the entire sum. Moves that
    size are usually a reverse split that never got adjusted, and no portfolio
    ever collects them anyway.

    Clipping uses only the same day's cross-section, so nothing from the future
    enters, and it's applied to every row alike -- the top decile and the base
    rate are trimmed by the same rule.
    """
    out = getiri.astype(np.float64, copy=True)
    s = pd.Series(out)
    g = s.groupby(np.asarray(gun))
    lo = g.transform(lambda x: x.quantile(alt_p))
    hi = g.transform(lambda x: x.quantile(ust_p))
    return s.clip(lower=lo, upper=hi).to_numpy()


def dilim_getirisi(p: np.ndarray, getiri: np.ndarray,
                   tarih: pd.DatetimeIndex, dilim: int = 10,
                   maliyet_bp: float = 0.0, ufuk_gun: int = 1) -> dict:
    """What the top slice actually returned, net of costs.

    This is the number that decides whether the model is worth anything.
    Brier and AUC both treat every row as equally important, and in practice
    only the rows you'd act on matter -- you don't trade the middle of the
    distribution. A model with AUC 0.51 whose top decile earns is useful; one
    with AUC 0.55 whose top decile earns nothing is not.

    Costs come off as a round-trip in basis points. Reported at several levels
    because where an edge dies is more informative than whether it exists at
    zero cost.

    The t value is over DAILY means, Newey-West corrected -- signals on the
    same day share one market day, and a per-row t would inflate the sample
    roughly tenfold.

    The lag has to come from the horizon, not from the sample size. Today's
    63-day forward return and tomorrow's share 62 of their 63 days, so the
    daily series stays correlated out to about 63 lags. The textbook
    4*(T/100)^(2/9) rule looks at how much data there is rather than how it
    was generated and picks 4, which understates the standard error by around
    a factor of four at that horizon.
    """
    from .faktor_zaman import newey_west_t

    n = len(p)
    if n < 100:
        return {"ok": False, "reason": f"{n} satir"}
    esik = np.quantile(p, 1.0 - 1.0 / dilim)
    ust = p >= esik
    if ust.sum() < 50:
        return {"ok": False, "reason": "ust dilim cok kucuk"}

    net = getiri[ust] - maliyet_bp / 10000.0
    gunluk = pd.DataFrame({"gun": tarih[ust], "r": net}).groupby("gun")["r"].mean()
    if len(gunluk) < 5:
        return {"ok": False, "reason": f"{len(gunluk)} gun"}
    tv, _, gecikme = newey_west_t(gunluk.to_numpy(),
                                  lag=max(1, int(ufuk_gun)))

    dilimler = []
    try:
        kova = pd.qcut(p, dilim, labels=False, duplicates="drop")
        for k, g in pd.DataFrame({"k": kova, "r": getiri}).groupby("k"):
            dilimler.append({"dilim": int(k), "n": int(len(g)),
                             "getiri": round(float(g["r"].mean()), 6)})
    except ValueError:
        pass

    return {
        "ok": True,
        "n": int(ust.sum()),
        "gun": int(len(gunluk)),
        "getiri": round(float(net.mean()), 6),
        "brut": round(float(getiri[ust].mean()), 6),
        "taban": round(float(getiri.mean()), 6),
        # A mean alone hid a distribution where 1% of rows carried 400% of the
        # total. The median says what the typical pick did; if the two
        # disagree badly the average is being carried by a few names.
        "ortanca": round(float(np.median(net)), 6),
        "taban_ortanca": round(float(np.median(getiri)), 6),
        "t_nw": None if not np.isfinite(tv) else round(float(tv), 2),
        "gecikme": int(gecikme),
        "maliyet_bp": maliyet_bp,
        "dilimler": dilimler,
    }


def gunluk_fark(p_model: np.ndarray, p_taban: np.ndarray, y: np.ndarray,
                tarih: pd.DatetimeIndex,
                ufuk_gun: int = 1) -> tuple[float, float, int]:
    """Gun bazinda Brier farki -> Newey-West duzeltmeli t.

    Neden gun bazinda: ayni gunun satirlari bagimsiz degil. Satir bazinda
    bir t hesabi, orneklem buyuklugunu on kat sisirir ve her kucuk farki
    anlamli gosterir.

    Doner: (ortalama_fark, t, gun_sayisi). Pozitif fark = model daha iyi.
    """
    from .faktor_zaman import newey_west_t

    d = pd.DataFrame({
        "gun": tarih,
        "fark": (p_taban - y) ** 2 - (p_model - y) ** 2,
    }).groupby("gun")["fark"].mean()
    if len(d) < 5:
        return float("nan"), float("nan"), len(d)
    # Same overlap as the returns: consecutive days score labels that share
    # almost all of their future window.
    t, _, _ = newey_west_t(d.to_numpy(), lag=max(1, int(ufuk_gun)))
    return float(d.mean()), float(t), int(len(d))


# =============================================================================
#  Model
# =============================================================================
def _dogrulama_bol(tarih, pay: float = 0.15):
    """Split training rows by time: the last slice is held out.

    Has to be by time, not at random. A random split leaks — the same day
    lands on both sides, and the model gets scored on days it already saw.
    """
    import numpy as _np

    n = len(tarih)
    if n < 500:
        return _np.ones(n, dtype=bool), _np.zeros(n, dtype=bool)
    gunler = _np.array(sorted(set(tarih)))
    kes = gunler[max(1, int(len(gunler) * (1 - pay)))]
    val = _np.array([t >= kes for t in tarih])
    if val.sum() < 100 or (~val).sum() < 400:
        return _np.ones(n, dtype=bool), _np.zeros(n, dtype=bool)
    return ~val, val


def _platt(p: "np.ndarray", y: "np.ndarray") -> tuple:
    """One-dimensional logistic fit that rescales predictions.

    The sequence model was spreading its predictions from 6% to 91% while the
    real rate never left 43-47%. That spread is fabricated, and Brier punishes
    it hard. Platt pulls the scale back to whatever the held-out slice
    supports. It can't invent ranking ability, and it isn't meant to — if AUC
    is 0.50 it stays 0.50, only the numbers stop lying about their own
    certainty.
    """
    eps = 1e-6
    z = np.log(np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps)))
    a, b = 1.0, 0.0
    for _ in range(200):
        q = 1.0 / (1.0 + np.exp(-(a * z + b)))
        g = q - y
        ga, gb = float(np.mean(g * z)), float(np.mean(g))
        h = q * (1 - q)
        haa = float(np.mean(h * z * z)) + 1e-6
        hbb = float(np.mean(h)) + 1e-6
        a -= ga / haa
        b -= gb / hbb
    return float(a), float(b)


def _uygula(p: "np.ndarray", kal: tuple | None) -> "np.ndarray":
    if not kal:
        return p
    a, b = kal
    eps = 1e-6
    z = np.log(np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps)))
    return 1.0 / (1.0 + np.exp(-(a * z + b)))


def _egit(X: np.ndarray, y: np.ndarray, seed: int = 7, epochs: int = 200,
          gizli: int = 128, lr: float = 2e-3, tarih=None,
          sabir: int = 15) -> "object | None":
    """Small MLP with a logistic output. Table in, 0/1 out.

    Deliberately small: with a noisy target a big net memorises instead of
    learning, and the baseline it's up against is simple anyway.

    Training stops on a held-out tail of the training window rather than
    running a fixed number of epochs, and the output gets rescaled on that
    same tail. Without either, the model just keeps fitting noise and its
    confidence drifts away from anything real.
    """
    from . import models as mz

    if not mz.torch_available():
        return None
    import torch
    import torch.nn as nn

    mz._tohumla(seed)
    mu, sd = X.mean(0), X.std(0)
    sd[sd < 1e-9] = 1.0
    Z = (X - mu) / sd

    tr, val = _dogrulama_bol(tarih if tarih is not None else np.arange(len(y)))

    # Three layers rather than two, and wider. Capacity on its own doesn't
    # find signal that isn't there -- it finds more ways to memorise the
    # training window -- so the size is only safe because early stopping
    # decides when to quit and the output gets rescaled afterwards.
    net = nn.Sequential(
        nn.Linear(Z.shape[1], gizli), nn.LayerNorm(gizli), nn.GELU(),
        nn.Dropout(0.25),
        nn.Linear(gizli, gizli), nn.LayerNorm(gizli), nn.GELU(),
        nn.Dropout(0.15),
        nn.Linear(gizli, gizli // 2), nn.GELU(),
        nn.Linear(gizli // 2, 1),
    )
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-3)
    kayip = nn.BCEWithLogitsLoss()
    Xt = torch.tensor(Z, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    Xtr, ytr = Xt[tr], yt[tr]
    rng = np.random.default_rng(seed)
    n = len(Xtr)
    yigin = 4096

    en_iyi, en_iyi_skor, bekleme = None, np.inf, 0
    for _ in range(epochs):
        net.train()
        sira = rng.permutation(n)
        for i in range(0, n, yigin):
            idx = sira[i:i + yigin]
            opt.zero_grad()
            kayip(net(Xtr[idx]), ytr[idx]).backward()
            opt.step()
        if not val.any():
            continue
        net.eval()
        with torch.no_grad():
            pv = torch.sigmoid(net(Xt[val]).squeeze(-1)).numpy()
        skor = brier(pv, y[val])
        if skor < en_iyi_skor - 1e-6:
            en_iyi_skor, bekleme = skor, 0
            en_iyi = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bekleme += 1
            if bekleme >= sabir:
                break

    if en_iyi is not None:
        net.load_state_dict(en_iyi)
    net.eval()

    kal = None
    if val.any():
        with torch.no_grad():
            pv = torch.sigmoid(net(Xt[val]).squeeze(-1)).numpy()
        kal = _platt(pv, y[val])
    return {"net": net, "mu": mu, "sd": sd, "kal": kal}


def _egit_dizi(Xs: np.ndarray, Xd: np.ndarray, y: np.ndarray, seed: int = 7,
               epochs: int = 120, gizli: int = 128, lr: float = 2e-3,
               tarih=None, sabir: int = 12):
    """Same job as _egit, but it also reads the bars before the signal.

    A small 1-D conv runs over the window and gets pooled down to a vector,
    which is then glued to the scalar features and fed through the same head.
    Conv rather than a recurrent net because it trains a lot faster here and
    there's no long-range dependency to chase across 24 bars.
    """
    from . import models as mz

    if not mz.torch_available():
        return None
    import torch
    import torch.nn as nn

    mz._tohumla(seed)
    mu, sd = Xs.mean(0), Xs.std(0)
    sd[sd < 1e-9] = 1.0
    Zs = (Xs - mu) / sd

    # Per-channel scaling over the whole window, so one loud channel
    # (volume, usually) doesn't drown the rest.
    dmu = Xd.reshape(-1, Xd.shape[2]).mean(0)
    dsd = Xd.reshape(-1, Xd.shape[2]).std(0)
    dsd[dsd < 1e-9] = 1.0
    Zd = (Xd - dmu) / dsd

    kanal = Xd.shape[2]

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv1d(kanal, 32, 3, padding=1), nn.GELU(),
                nn.Conv1d(32, 32, 3, padding=1), nn.GELU(),
                nn.AdaptiveAvgPool1d(1),
            )
            self.head = nn.Sequential(
                nn.Linear(32 + Xs.shape[1], gizli), nn.LayerNorm(gizli),
                nn.GELU(), nn.Dropout(0.2),
                nn.Linear(gizli, gizli // 2), nn.GELU(),
                nn.Linear(gizli // 2, 1),
            )

        def forward(self, d, s):
            z = self.conv(d.transpose(1, 2)).squeeze(-1)
            return self.head(torch.cat([z, s], dim=1))

    net = Net()
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-2)
    kayip = nn.BCEWithLogitsLoss()
    St = torch.tensor(Zs, dtype=torch.float32)
    Dt = torch.tensor(Zd, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    tr, val = _dogrulama_bol(tarih if tarih is not None else np.arange(len(y)))
    rng = np.random.default_rng(seed)
    idx_tr = np.flatnonzero(tr)
    n = len(idx_tr)
    yigin = 2048

    en_iyi, en_iyi_skor, bekleme = None, np.inf, 0
    for _ in range(epochs):
        net.train()
        sira = rng.permutation(n)
        for i in range(0, n, yigin):
            idx = idx_tr[sira[i:i + yigin]]
            opt.zero_grad()
            kayip(net(Dt[idx], St[idx]), yt[idx]).backward()
            opt.step()
        if not val.any():
            continue
        net.eval()
        with torch.no_grad():
            pv = torch.sigmoid(
                net(Dt[val], St[val]).squeeze(-1)).numpy()
        skor = brier(pv, y[val])
        if skor < en_iyi_skor - 1e-6:
            en_iyi_skor, bekleme = skor, 0
            en_iyi = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bekleme += 1
            if bekleme >= sabir:
                break

    if en_iyi is not None:
        net.load_state_dict(en_iyi)
    net.eval()

    kal = None
    if val.any():
        with torch.no_grad():
            pv = torch.sigmoid(net(Dt[val], St[val]).squeeze(-1)).numpy()
        kal = _platt(pv, y[val])
    return {"net": net, "mu": mu, "sd": sd, "dmu": dmu, "dsd": dsd,
            "kal": kal, "dizi": True}


def _tahmin(model: dict, X: np.ndarray, Xd: np.ndarray | None = None
            ) -> np.ndarray:
    import torch

    Z = (X - model["mu"]) / model["sd"]
    with torch.no_grad():
        St = torch.tensor(Z, dtype=torch.float32)
        if model.get("dizi"):
            Zd = (Xd - model["dmu"]) / model["dsd"]
            z = model["net"](torch.tensor(Zd, dtype=torch.float32), St)
        else:
            z = model["net"](St)
        p = torch.sigmoid(z.squeeze(-1)).numpy().astype(np.float64)
    return _uygula(p, model.get("kal"))


# =============================================================================
#  Ileri yuruyus
# =============================================================================
def walk_forward(veri: dict, ufuk: int, kalib: dict | None, taban: float,
                 n_kat: int = 4, embargo_gun: int = 5,
                 bar_gun: float = 1.0, seed: int = 7,
                 Xd: "np.ndarray | None" = None,
                 maliyetler: "tuple[float, ...]" = (0.0, 10.0, 20.0),
                 gizli: int = 128, devir: int = 200, sabir: int = 15) -> dict:
    """Zaman sirali katmanlar; arindirma + tampon ile.

    Her katmanda: o katmandan ONCEKI gunlerle egit, katmanin kendisinde olc.
    Egitim kumesinin sonundan, etiket ufku + tampon kadar gun ATILIR --
    yoksa egitim satirlarinin etiketi test penceresine tasar.
    """
    gunler = np.array(sorted(pd.unique(veri["tarih"])))
    if len(gunler) < 40:
        return {"ok": False, "reason": f"yalnizca {len(gunler)} farkli gun"}

    # The horizon is counted in TRADING days; the gap below is subtracted in
    # CALENDAR days. Those are not the same unit and treating them as one let
    # roughly a fifth of the training labels reach into the test window -- a
    # 63-bar horizon is about 91 calendar days, and we were dropping 68. The
    # leak grew with the horizon, which is exactly the shape the first run
    # showed: 21 bars looked plausible, 63 bars returned an absurd 10%.
    ufuk_gun = max(1, int(np.ceil(ufuk / max(bar_gun, 1e-9))))
    arindirma = int(np.ceil(ufuk_gun * 7.0 / 5.0)) + embargo_gun

    parcalar = np.array_split(gunler, n_kat + 1)
    katlar = []
    tum_p, tum_b, tum_y, tum_t, tum_r = [], [], [], [], []

    for k in range(1, n_kat + 1):
        test_gun = set(pd.Timestamp(g) for g in parcalar[k])
        egitim_sonu = parcalar[k][0] - np.timedelta64(arindirma, "D")
        egitim_maske = veri["tarih"].to_numpy() <= egitim_sonu
        test_maske = np.array([t in test_gun for t in veri["tarih"]])
        if egitim_maske.sum() < MIN_SATIR or test_maske.sum() < 200:
            continue

        egitim_tarih = veri["tarih"][egitim_maske]
        if Xd is not None:
            model = _egit_dizi(veri["X"][egitim_maske], Xd[egitim_maske],
                               veri["y"][egitim_maske], seed=seed,
                               tarih=egitim_tarih, gizli=gizli,
                               epochs=max(40, devir // 2), sabir=sabir)
        else:
            model = _egit(veri["X"][egitim_maske], veri["y"][egitim_maske],
                          seed=seed, tarih=egitim_tarih, gizli=gizli,
                          epochs=devir, sabir=sabir)
        if model is None:
            return {"ok": False, "reason": "torch yok"}

        p = _tahmin(model, veri["X"][test_maske],
                    Xd[test_maske] if Xd is not None else None)
        y = veri["y"][test_maske]
        b = kova_olasiligi(kalib, veri["kurulum"][test_maske], ufuk,
                           {a: d[test_maske] for a, d in veri["kosullar"].items()},
                           taban)
        katlar.append({
            "kat": k,
            "egitim": int(egitim_maske.sum()),
            "test": int(test_maske.sum()),
            "test_gun": int(len(test_gun)),
            "brier_model": round(brier(p, y), 5),
            "brier_kova": round(brier(b, y), 5),
            "brier_taban": round(brier(np.full(len(y), taban), y), 5),
            "auc_model": round(auc(p, y), 4),
            "auc_kova": round(auc(b, y), 4),
        })
        tum_p.append(p); tum_b.append(b); tum_y.append(y)
        tum_t.append(veri["tarih"][test_maske])
        tum_r.append(veri["getiri"][test_maske])

    if not katlar:
        return {"ok": False, "reason": "hicbir katman kurulamadi"}

    P = np.concatenate(tum_p); B = np.concatenate(tum_b)
    Y = np.concatenate(tum_y); T = pd.DatetimeIndex(np.concatenate(tum_t))
    R = np.concatenate(tum_r)
    fark, t, n_gun = gunluk_fark(P, B, Y, T, ufuk_gun=ufuk_gun)
    dilim = {f"{int(c)}bp": dilim_getirisi(P, R, T, maliyet_bp=c,
                                           ufuk_gun=ufuk_gun)
             for c in maliyetler}

    return {
        "ok": True,
        "ufuk": ufuk,
        "dizi": Xd is not None,
        "satir": int(len(Y)),
        "gun": n_gun,
        "katlar": katlar,
        "brier_model": round(brier(P, Y), 5),
        "brier_kova": round(brier(B, Y), 5),
        "brier_taban": round(brier(np.full(len(Y), taban), Y), 5),
        "auc_model": round(auc(P, Y), 4),
        "auc_kova": round(auc(B, Y), 4),
        "gunluk_fark": None if not np.isfinite(fark) else round(fark, 6),
        "t_nw": None if not np.isfinite(t) else round(t, 2),
        "brier_daha_iyi": bool(np.isfinite(t) and t >= 2.0),
        "dilim": dilim,
        # The headline call. Brier says whether the probabilities are better
        # calibrated; this says whether the rows you'd act on made money after
        # costs. The second question is the one that matters.
        "model_daha_iyi": bool(
            (dilim.get("10bp", {}).get("t_nw") or 0) >= 2.0
            and (dilim.get("10bp", {}).get("getiri") or 0) > 0),
        "guvenilirlik": guvenilirlik(P, Y),
        "tahmin_araligi": [round(float(P.min()), 4), round(float(P.max()), 4)],
    }


def _barlari_yukle(frekans: str, semboller: set) -> dict:
    """Cached bars for the symbols the panel actually references."""
    from . import intraday as idy
    from .providers import cache as _c

    out = {}
    for tk in sorted(semboller):
        if frekans == "1d":
            hit = _c.peek("yahoo", f"{tk}:2y")
            h = (hit[0] or {}).get("history") if hit else None
        else:
            h = idy.oku(tk, frekans, max_gun=30)
        if h is not None and len(h):
            out[tk] = h
    return out


def calistir(frekans: str = "1d", ufuklar: "tuple[int, ...] | None" = None,
             n_kat: int = 4, seed: int = 7, dizi: bool = False,
             pencere: int = 24, etiket: str = "kazanc",
             maliyetler: "tuple[float, ...]" = (0.0, 10.0, 20.0),
             gizli: int = 128, devir: int = 200, sabir: int = 15) -> dict:
    """Panelden meta-modeli egitir ve kova taban cizgisine karsi olcer.

    dizi=True feeds the bars leading up to each signal as well; see src/dizi.py.
    """
    from . import kalibrasyon as kb
    from . import kisa_vade as kv

    df = panel_yukle(frekans)
    if df is None or df.empty:
        return {"ok": False,
                "reason": f"panel yok — once: python run.py kisa panel "
                          f"--frekans {frekans}"}
    kalib = kb.yukle(frekans=frekans)
    ufuklar = ufuklar or kv.ufuklar(frekans)
    bg = kv.bar_gun(frekans)

    bars = _barlari_yukle(frekans, set(df["ticker"].unique())) if dizi else {}
    if dizi and not bars:
        return {"ok": False,
                "reason": f"{frekans} bars are not cached - run: "
                          f"python run.py intraday cek --interval {frekans}"}

    sonuc = []
    pencere_ozet = None
    for u in ufuklar:
        veri = hazirla(df, u, etiket)
        if veri is None:
            sonuc.append({"ufuk": u, "ok": False, "reason": "yeterli satir yok"})
            continue

        Xd = None
        if dizi:
            from . import dizi as dz

            if veri.get("zaman") is None:
                return {"ok": False,
                        "reason": "panel has no 'zaman' column - rebuild it "
                                  "with: python run.py kisa panel"}
            anah = list(zip(veri["ticker"], veri["zaman"]))
            Xd, bulundu = dz.pencereler(bars, anah, pencere)
            pencere_ozet = dz.ozet(Xd, bulundu)
            # Rows without a full window would train the model on blank
            # history, so drop them instead of zero-filling.
            if int(bulundu.sum()) < MIN_SATIR:
                sonuc.append({"ufuk": u, "ok": False,
                              "reason": f"only {int(bulundu.sum())} rows have "
                                        f"a full window"})
                continue
            Xd = Xd[bulundu]
            veri = {**veri, "X": veri["X"][bulundu], "y": veri["y"][bulundu],
                    "tarih": veri["tarih"][bulundu],
                    "kurulum": veri["kurulum"][bulundu],
                    "kosullar": {a: d[bulundu]
                                 for a, d in veri["kosullar"].items()}}

        # The bucket calibration answers "did it beat the index after N
        # bars". The barrier label answers a different question and has a
        # different base rate -- around 40% rather than 48%, because the stop
        # sits closer than the target. Scoring it against buckets built for
        # the other question would flatter the model: the baseline would be
        # aiming at the wrong number and losing to anything.
        #
        # There are no barrier-calibrated buckets, so for that label both
        # baselines collapse to the label's own base rate, and the comparison
        # is honestly model-vs-constant.
        if etiket == "kazanc":
            taban = float((kalib or {}).get("taban", {}).get(str(u), 0.5))
            kalib_kul = kalib
        else:
            taban = float(np.nanmean(veri["y"]))
            kalib_kul = None
        r = walk_forward(veri, u, kalib_kul, taban, n_kat=n_kat,
                         bar_gun=bg, seed=seed, Xd=Xd,
                         maliyetler=maliyetler, gizli=gizli, devir=devir,
                         sabir=sabir)
        r["getiri_ad"] = veri.get("getiri_ad")
        r["taban_orani"] = round(taban, 4)
        r["ozellik"] = len(veri["ozellik_adlari"])
        r.setdefault("ufuk", u)
        sonuc.append(r)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "frekans": frekans,
        "ok": any(r.get("ok") for r in sonuc),
        "panel_satir": int(len(df)),
        "kalibrasyon": (kalib or {}).get("generated_at"),
        "ufuklar": list(ufuklar),
        "etiket": etiket,
        "model": {"gizli": gizli, "devir": devir, "sabir": sabir},
        "dizi": bool(dizi),
        "pencere": pencere if dizi else None,
        "pencere_ozet": pencere_ozet,
        "sonuclar": sonuc,
        "notlar_tr": _notlar(sonuc),
    }


def _notlar(sonuc: list) -> list[str]:
    out = []
    calisan = [r for r in sonuc if r.get("ok")]
    if not calisan:
        return ["Hicbir ufukta model olculemedi."]
    iyi = [r for r in calisan if r.get("model_daha_iyi")]
    out.append(
        f"{len(calisan)} ufuk olculdu, {len(iyi)} tanesinde ust dilim 10bp "
        "maliyetten sonra pozitif ve gun bazinda t>=2.")
    for r in calisan:
        d = (r.get("dilim") or {}).get("10bp") or {}
        if d.get("ok"):
            out.append(
                f"Ufuk {r['ufuk']}: ust dilim brut %{100*d['brut']:.3f}, "
                f"10bp sonrasi %{100*d['getiri']:.3f}, taban "
                f"%{100*d['taban']:.3f}, t={d['t_nw']}, {d['n']} satir / "
                f"{d['gun']} gun.")
            out.append(
                f"Ufuk {r['ufuk']}: ORTANCA ust dilim %{100*d['ortanca']:.3f} "
                f"/ taban %{100*d['taban_ortanca']:.3f}. Ortalama ile ortanca "
                "birbirinden uzaksa getiriyi birkac isim tasiyordur.")
    if not iyi:
        out.append(
            "Model, kova bazli kalibrasyonun uzerine OLCULEBILIR bir sey "
            "koymuyor. Bu bir basarisizlik degil bir sonuc: elimizdeki "
            "ozelliklerde, kovanin ayirt edemedigini ayirt eden bir yapi yok.")
    for r in calisan:
        g = r.get("guvenilirlik") or []
        if len(g) >= 3:
            alt, ust = g[0], g[-1]
            out.append(
                f"Ufuk {r['ufuk']}: en dusuk dilimde tahmin "
                f"%{100*alt['tahmin']:.0f} / gerceklesen %{100*alt['gerceklesen']:.0f}, "
                f"en yuksekte %{100*ust['tahmin']:.0f} / %{100*ust['gerceklesen']:.0f}. "
                + ("Siralama dogru yonde." if ust["gerceklesen"] > alt["gerceklesen"]
                   else "Siralama TERS — modelin yuksek dedigi daha az tutuyor."))
    out.append(
        "Brier farki GUN bazinda olculdu: ayni gunun sinyalleri bagimsiz "
        "degil, satir bazinda bir t degeri orneklemi on kat sisirirdi.")
    return out


def kaydet(payload: dict, path: Path | None = None) -> Path:
    p = path or cikti_yolu(payload.get("frekans", "1d"),
                           bool(payload.get("dizi")),
                           payload.get("etiket", "kazanc"))
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                 encoding="utf-8")
    return p


def yukle(frekans: str = "1d", path: Path | None = None,
          dizi: bool = False, etiket: str = "kazanc") -> dict | None:
    p = path or cikti_yolu(frekans, dizi, etiket)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
