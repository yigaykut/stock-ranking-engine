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


def cikti_yolu(frekans: str) -> Path:
    return DATA / f"meta_model_{frekans}.json"


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
            and not c.startswith(("fazla_", "kazanc_"))]


def hazirla(df: pd.DataFrame, ufuk: int) -> dict | None:
    """Bir ufuk icin X, y, tarih ve kova taban olasiligi.

    Kategorik sutunlar (kurulum, oynaklik, likidite, trend_konumu) one-hot
    ediliyor. Kurulum kimliginin ozellik olmasi SART: model "hangi kurulum"
    bilgisini gormeden, kurulumlar arasi farki ogrenemez.
    """
    etiket = f"kazanc_{ufuk}g"
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

    return {
        "X": XX.to_numpy(dtype=np.float32),
        "y": alt[etiket].to_numpy(dtype=np.float32),
        "tarih": pd.DatetimeIndex(alt["tarih"]).normalize(),
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
def _egit(X: np.ndarray, y: np.ndarray, seed: int = 7, epochs: int = 60,
          gizli: int = 48, lr: float = 2e-3) -> "object | None":
    """Kucuk MLP + lojistik cikti. Girdi tablo, hedef 0/1.

    Kucuk tutuluyor: 14 ozellik ve gurultulu bir hedef icin buyuk bir ag,
    ogrenmekten cok ezberler. Karsilastirilan taban cizgisi de zaten basit.
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

    rng = np.random.default_rng(seed)
    n = len(Xt)
    yigin = 4096
    net.train()
    for _ in range(epochs):
        sira = rng.permutation(n)
        for i in range(0, n, yigin):
            idx = sira[i:i + yigin]
            opt.zero_grad()
            kayip(net(Xt[idx]), yt[idx]).backward()
            opt.step()
    net.eval()
    return {"net": net, "mu": mu, "sd": sd}


def _tahmin(model: dict, X: np.ndarray) -> np.ndarray:
    import torch

    Z = (X - model["mu"]) / model["sd"]
    with torch.no_grad():
        z = model["net"](torch.tensor(Z, dtype=torch.float32)).squeeze(-1)
        return torch.sigmoid(z).numpy().astype(np.float64)


# =============================================================================
#  Ileri yuruyus
# =============================================================================
def walk_forward(veri: dict, ufuk: int, kalib: dict | None, taban: float,
                 n_kat: int = 4, embargo_gun: int = 5,
                 bar_gun: float = 1.0, seed: int = 7) -> dict:
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

        model = _egit(veri["X"][egitim_maske], veri["y"][egitim_maske], seed=seed)
        if model is None:
            return {"ok": False, "reason": "torch yok"}

        p = _tahmin(model, veri["X"][test_maske])
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


def calistir(frekans: str = "1d", ufuklar: "tuple[int, ...] | None" = None,
             n_kat: int = 4, seed: int = 7) -> dict:
    """Panelden meta-modeli egitir ve kova taban cizgisine karsi olcer."""
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

    sonuc = []
    for u in ufuklar:
        veri = hazirla(df, u)
        if veri is None:
            sonuc.append({"ufuk": u, "ok": False, "reason": "yeterli satir yok"})
            continue
        taban = float((kalib or {}).get("taban", {}).get(str(u), 0.5))
        r = walk_forward(veri, u, kalib, taban, n_kat=n_kat,
                         bar_gun=bg, seed=seed)
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
    p = path or cikti_yolu(payload.get("frekans", "1d"))
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                 encoding="utf-8")
    return p


def yukle(frekans: str = "1d", path: Path | None = None) -> dict | None:
    p = path or cikti_yolu(frekans)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
