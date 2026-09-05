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
            and not c.startswith(("fazla_", "kazanc_", "bariyer_"))]


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
    return {
        "X": XX.to_numpy(dtype=np.float32),
        "y": alt[etiket].to_numpy(dtype=np.float32),
        "tarih": pd.DatetimeIndex(alt["tarih"]).normalize(),
        "zaman": zaman,
        "ticker": alt["ticker"].to_numpy(),
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


def gunluk_fark(p_model: np.ndarray, p_taban: np.ndarray, y: np.ndarray,
                tarih: pd.DatetimeIndex) -> tuple[float, float, int]:
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
    t, _, _ = newey_west_t(d.to_numpy(), lag=None)
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


def _egit(X: np.ndarray, y: np.ndarray, seed: int = 7, epochs: int = 60,
          gizli: int = 48, lr: float = 2e-3, tarih=None,
          sabir: int = 6) -> "object | None":
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

    net = nn.Sequential(
        nn.Linear(Z.shape[1], gizli), nn.LayerNorm(gizli), nn.GELU(),
        nn.Dropout(0.2),
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
               epochs: int = 40, gizli: int = 48, lr: float = 2e-3,
               tarih=None, sabir: int = 5):
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
                 Xd: "np.ndarray | None" = None) -> dict:
    """Zaman sirali katmanlar; arindirma + tampon ile.

    Her katmanda: o katmandan ONCEKI gunlerle egit, katmanin kendisinde olc.
    Egitim kumesinin sonundan, etiket ufku + tampon kadar gun ATILIR --
    yoksa egitim satirlarinin etiketi test penceresine tasar.
    """
    gunler = np.array(sorted(pd.unique(veri["tarih"])))
    if len(gunler) < 40:
        return {"ok": False, "reason": f"yalnizca {len(gunler)} farkli gun"}

    ufuk_gun = max(1, int(np.ceil(ufuk / max(bar_gun, 1e-9))))
    arindirma = ufuk_gun + embargo_gun

    parcalar = np.array_split(gunler, n_kat + 1)
    katlar = []
    tum_p, tum_b, tum_y, tum_t = [], [], [], []

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
                               tarih=egitim_tarih)
        else:
            model = _egit(veri["X"][egitim_maske], veri["y"][egitim_maske],
                          seed=seed, tarih=egitim_tarih)
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

    if not katlar:
        return {"ok": False, "reason": "hicbir katman kurulamadi"}

    P = np.concatenate(tum_p); B = np.concatenate(tum_b)
    Y = np.concatenate(tum_y); T = pd.DatetimeIndex(np.concatenate(tum_t))
    fark, t, n_gun = gunluk_fark(P, B, Y, T)

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
        "model_daha_iyi": bool(np.isfinite(t) and t >= 2.0),
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
             pencere: int = 24, etiket: str = "kazanc") -> dict:
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

        taban = float((kalib or {}).get("taban", {}).get(str(u), 0.5))
        r = walk_forward(veri, u, kalib, taban, n_kat=n_kat,
                         bar_gun=bg, seed=seed, Xd=Xd)
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
        f"{len(calisan)} ufuk olculdu, {len(iyi)} tanesinde model kova taban "
        "cizgisini gun bazinda |t|>=2 ile geciyor.")
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
