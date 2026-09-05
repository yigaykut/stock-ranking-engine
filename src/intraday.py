"""Gun ici bar cekimi — ve saglayicinin GERCEK sinirlarini ogrenme.

NEDEN AYRI BIR MODUL
--------------------
Gunluk barlar `providers/yahoo.py` uzerinden geliyor ve orasi `Ticker.history`
kullaniyor. Gun ici icin iki sey farkli:

  1. `Ticker.history` bu ortamda BOZUK (yfinance 1.0): her cagride
     `TypeError: 'NoneType' object is not subscriptable` firlatiyor.
     `yf.download` ayni veriyi doner ve calisir. Gun ici yolu bu yuzden
     ayri.
  2. Gun ici gecmis, aralik basina FARKLI uzunlukta veriliyor ve bu sinir
     saglayicinin kararidir, bizim degil.

SINIRI VARSAYMIYORUZ, OLCUYORUZ
-------------------------------
"1 dakikalik veri 7 gun, saatlik 730 gun" gibi sayilar belgelenmis olsa da
degisebilir ve dogrulanmadan koda gomulmemeli. Gomulurse sinir daraldiginda
sistem sessizce eksik veriyle calisir ve bunu kimse fark etmez.

Bunun yerine her basarili cekimde GORULEN kapsam kaydediliyor
(data/intraday_kapsam.json): kac bar geldi, kac farkli gun, ilk ve son tarih.
Karar bu olculere gore verilir. Ilk kosuda bilgi yoksa cikti bunu acikca
soyler.

HIZ SINIRI
----------
Bu sistemin en kirilgan kaynagi Yahoo hiz siniri; gunluk taramanin butcesini
paylasiyoruz. Bu yuzden:
  - havuz disinda sembol cekilmez (2755 degil ~150 sembol)
  - istekler arasi bekleme var
  - 429 gorulunce DEVAM EDILMEZ, durulur (RateLimited)
  - basarili cikti onbelleklenir, ayni gun tekrar cekilmez
"""
from __future__ import annotations

import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .providers import cache
from .providers.yahoo import RateLimited, _is_rate_limit

DATA = Path(__file__).resolve().parents[1] / "data"
KAPSAM = DATA / "intraday_kapsam.json"

# Denenecek araliklar ve saglayicinin BELGELEDIGI ust sinir. Bunlar birer
# ISTEK, garanti degil; gercekte ne geldigi kapsam dosyasina yazilir.
ISTEK = {
    "1m": "7d",
    "5m": "60d",
    "15m": "60d",
    "30m": "60d",
    "1h": "730d",
}

# Bir araligin "olculebilir" sayilmasi icin gereken en az FARKLI GUN sayisi.
# Bar sayisi degil gun sayisi onemli: ayni gunun barlari tek bir piyasa
# gunudur (bkz. kalibrasyon.etkin_n).
MIN_GUN = 120

BEKLEME = 1.2          # istekler arasi saniye
TTL = 12 * 3600        # gun ici veri gun icinde tazelenmeli


def _indir(sembol: str, interval: str, period: str,
           attempts: int = 3) -> pd.DataFrame | None:
    """yf.download ile cekim. 429 gorulurse RateLimited firlatir."""
    import warnings

    import yfinance as yf

    for i in range(attempts):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                h = yf.download(sembol, period=period, interval=interval,
                                progress=False, auto_adjust=True,
                                threads=False)
            if isinstance(h, pd.DataFrame) and len(h):
                return _duzlestir(h, sembol)
        except Exception as exc:
            if _is_rate_limit(exc):
                raise RateLimited(str(exc)[:120]) from exc
        if i < attempts - 1:
            time.sleep(1.5 * (2 ** i) + random.random())
    return None


def _duzlestir(h: pd.DataFrame, sembol: str) -> pd.DataFrame:
    """yf.download cok seviyeli sutun dondurur; tek hisseye indiriyoruz.

    Duzlestirilmezse sutun adlari ('Close', 'AAPL') gibi demet olur ve
    df["Close"] bir DataFrame doner -- gostergeler sessizce sacmalar.
    """
    if isinstance(h.columns, pd.MultiIndex):
        try:
            h = h.xs(sembol, axis=1, level=-1)
        except (KeyError, ValueError):
            h.columns = [c[0] for c in h.columns]
    sut = [c for c in ("Open", "High", "Low", "Close", "Volume")
           if c in h.columns]
    return h[sut].dropna()


def cek(sembol: str, interval: str = "1h", period: str | None = None,
        use_cache: bool = True) -> pd.DataFrame | None:
    """Tek sembol icin gun ici bar. Onbellekli."""
    p = period or ISTEK.get(interval, "60d")
    return cache.get_or_fetch(
        "yahoo_intraday", f"{sembol}:{interval}:{p}",
        lambda: _indir(sembol, interval, p),
        ttl_seconds=TTL, enabled=use_cache,
        should_cache=lambda h: h is not None and len(h) >= 50)


def oku(sembol: str, interval: str = "1h", period: str | None = None,
        max_gun: float = 3.0) -> pd.DataFrame | None:
    """ONBELLEKTEN okur, ag istegi YAPMAZ.

    Tarama ile cekim ayri isler. `cek()` gerekirse aga cikar; tarama bunu
    yapmamali -- yoksa "bugunku kurulumlari goster" komutu sessizce 150
    sembollük bir indirme isine donusur ve hiz sinirini yer. Veri yoksa
    cevap None'dur ve cagiran taraf kullaniciyi cekime yonlendirir.
    """
    p = period or ISTEK.get(interval, "60d")
    hit = cache.peek("yahoo_intraday", f"{sembol}:{interval}:{p}")
    if not hit:
        return None
    veri, yas = hit
    if yas > max_gun * 24 * 3600:
        return None
    return veri if isinstance(veri, pd.DataFrame) and len(veri) else None


def olcum(h: pd.DataFrame | None) -> dict:
    """Bir cekimin kapsami: bar, farkli gun, tarih araligi, bar/gun."""
    if h is None or not len(h):
        return {"bar": 0, "gun": 0}
    idx = pd.DatetimeIndex(h.index)
    gun = int(pd.Series(idx).dt.date.nunique())
    return {
        "bar": int(len(h)),
        "gun": gun,
        "ilk": str(idx[0].date()),
        "son": str(idx[-1].date()),
        "bar_gun": round(len(h) / max(gun, 1), 1),
    }


def kapsam_olc(sembol: str = "SPY", araliklar: "list[str] | None" = None,
               use_cache: bool = False) -> dict:
    """Her aralikta GERCEKTE ne kadar gecmis geldigini olcer ve kaydeder.

    Tek sembolle yapilir: amac veri toplamak degil, sinirlari ogrenmek.
    """
    out = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "sembol": sembol, "araliklar": {}}
    for interval in (araliklar or list(ISTEK)):
        p = ISTEK.get(interval, "60d")
        try:
            h = cek(sembol, interval, p, use_cache=use_cache)
            o = olcum(h)
            o["istenen"] = p
            o["olculebilir"] = bool(o.get("gun", 0) >= MIN_GUN)
            out["araliklar"][interval] = o
        except RateLimited as exc:
            out["araliklar"][interval] = {"hata": "hiz siniri", "detay": str(exc)[:80]}
            break
        except Exception as exc:
            out["araliklar"][interval] = {"hata": type(exc).__name__}
        time.sleep(BEKLEME)
    kapsam_kaydet(out)
    return out


def kapsam_kaydet(payload: dict, path: Path | None = None) -> Path:
    p = path or KAPSAM
    p.parent.mkdir(parents=True, exist_ok=True)
    # Eski olcumler korunur: sinir zamanla degisirse gormek isteriz.
    eski = kapsam_yukle(p) or {}
    gecmis = eski.get("gecmis") or []
    if eski.get("araliklar"):
        gecmis.append({k: v for k, v in eski.items() if k != "gecmis"})
    payload = {**payload, "gecmis": gecmis[-10:]}
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                 encoding="utf-8")
    return p


def kapsam_yukle(path: Path | None = None) -> dict | None:
    p = path or KAPSAM
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def onerilen_aralik(kapsam: dict | None = None) -> dict:
    """Olculen kapsama gore hangi aralikta calisilmali.

    Olcut BAR SAYISI DEGIL, FARKLI GUN SAYISI. 1 dakikalik veride 2730 bar
    olabilir ama 7 gunden geliyorsa bagimsiz gozlem sayisi 7'dir; ustune
    ileri getiri ortusmesi de binince olculecek bir sey kalmaz.
    """
    # `kapsam is None` ile bos sozluk AYNI SEY DEGIL. `kapsam or ...`
    # yazilirsa, cagiran taraf bilerek bos bir kapsam gecse bile fonksiyon
    # gidip diskteki dosyayi okur -- yani "veri yok" durumu test edilemez
    # hale gelir. (Kapsam dosyasi olusana kadar fark etmiyordu.)
    k = kapsam_yukle() if kapsam is None else kapsam
    if not k or not k.get("araliklar"):
        return {"ok": False,
                "reason": "kapsam olcumu yok — once: python run.py intraday kapsam"}
    uygun = {i: o for i, o in k["araliklar"].items()
             if isinstance(o, dict) and o.get("gun", 0) >= MIN_GUN}
    if not uygun:
        return {"ok": False, "reason": f"hicbir aralik {MIN_GUN} farkli gune ulasmiyor",
                "olculen": k["araliklar"]}
    # En cok gun veren aralik; esitlikte daha ince olani (daha cok kesit).
    sira = sorted(uygun.items(), key=lambda kv: (-kv[1]["gun"], -kv[1]["bar"]))
    return {"ok": True, "birincil": sira[0][0],
            "adaylar": [i for i, _ in sira],
            "olculen": {i: {"gun": o["gun"], "bar": o["bar"]}
                        for i, o in uygun.items()}}


def havuz_cek(semboller: "list[str]", interval: str = "1h",
              period: str | None = None, bekleme: float = BEKLEME,
              ilerleme: "callable | None" = None) -> dict:
    """Havuzdaki sembollerin gun ici barlari. Hiz sinirinda DURUR."""
    out: dict[str, pd.DataFrame] = {}
    hata = 0
    for i, s in enumerate(semboller):
        try:
            h = cek(s, interval, period)
            if h is not None and len(h):
                out[s] = h
            else:
                hata += 1
        except RateLimited:
            return {"bundles": out, "durum": "hiz siniri",
                    "cekilen": len(out), "kalan": len(semboller) - i - 1,
                    "hatali": hata}
        except Exception:
            hata += 1
        if ilerleme and (i + 1) % 25 == 0:
            ilerleme(i + 1, len(out))
        time.sleep(bekleme)
    return {"bundles": out, "durum": "tamam", "cekilen": len(out),
            "kalan": 0, "hatali": hata}
