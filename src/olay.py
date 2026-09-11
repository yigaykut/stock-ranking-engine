"""Sirkete ozel olaylar: SEC 8-K bildirimleri.

NEDEN BU KAYNAK
---------------
Bir formasyon, sirket hakkinda o sirada cikan bir haberle birlikte bambaska
bir sey ifade edebilir. Bunu OLCMEK icin gereken sey haberin kendisi degil,
gecmise donuk ve zaman damgasi guvenilir bir kayit.

8-K tam olarak bu: sirketin "maddi bir olay oldu" diye kendi yaptigi duyuru.
Bilanco sonucu, yonetici ayriligi, onemli sozlesme, ihrac, denetci
degisikligi. Haber sitelerinden kazinmis basliklara gore uc ustunlugu var:

- Kabul zamani saniye hassasiyetinde ve kaynagindan geliyor. Haber
  arsivlerinde yayin saati cogu zaman yaklasik, bazen sonradan duzeltilmis.
- Sirkete CIK ile bagli, isim eslestirmesi yok. "Apple" gecen her makaleyi
  toplamak baska bir sey.
- Kote disi kalmis sirketlerin bildirimleri de duruyor. Fiyat onbelleginde
  olmayan hayatta kalma yanliligi burada yok.

Karsiliginda kaybettigimiz sey basligin metni: 8-K "yonetici ayrildi" der,
"neden" demez. Yani olayin TURUNU biliyoruz, TONUNU bilmiyoruz.

SIZINTI — BU DOSYANIN ASIL ISI
------------------------------
Bir 8-K'nin kabul saati cogunlukla 20:00-22:00 UTC, yani 16:00-18:00 ET:
kapanistan SONRA. Bildirimi dosyalama gunune yazarsak, o gunun kapanisinda
hesaplanan bir ozellik henuz olmamis bir olayi gormus olur -- ve 2.02
(bilanco) bildirimlerinin neredeyse tamami boyle. Olculen kenar da tam
oradan gelir.

Bu yuzden `etkin_gun`: kabul saati ET ile 16:00'dan onceyse dosyalama gunu,
degilse BIR SONRAKI gun. Bar hizalamasi her zaman bu sutunla yapilir,
`filingDate` ile degil.

HIZ SINIRI
----------
SEC saniyede 10 istege kadar izin veriyor ve gercek bir User-Agent istiyor.
Varsayilan tanitici kisisel bilgi icermez; SEC kendi politikasinda iletisim
adresi istedigi icin SEC_USER_AGENT ortam degiskeniyle degistirilebilir.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

from .providers import cache as _cache

UC = "https://data.sec.gov/submissions/"
ARSIV = "https://www.sec.gov/Archives/edgar/data/"
HARITA = "https://www.sec.gov/files/company_tickers.json"
BEKLE = 0.12                      # ~8 istek/saniye, SEC siniri 10
ALAN = "sec_olay"

# 8-K madde kodlari. Hepsini tek tek tasimak yerine anlamli obeklere
# indiriliyor: 2.02 ile 2.06 ayri seyler ama 40 ayri sutun, her biri yilda
# birkac kez atesleyen, olculebilir hicbir sey uretmez.
OBEK = {
    "sonuc":     ("2.02",),                    # faaliyet sonuclari
    "yonetim":   ("5.02",),                    # yonetici/direktor degisikligi
    "sozlesme":  ("1.01", "1.02"),             # onemli sozlesme, feshi
    "varlik":    ("2.01", "2.05", "2.06"),     # devralma/elden cikarma, deger dususu
    "ihrac":     ("3.02", "3.03"),             # hisse ihraci, hak degisikligi
    "kotasyon":  ("3.01",),                    # kotasyon/kural uyumsuzlugu
    "denetci":   ("4.01", "4.02"),             # denetci degisikligi, guvenilmezlik
    "regfd":     ("7.01",),                    # Reg FD aciklamasi
    "diger":     ("8.01",),                    # diger olaylar
}


class HizSiniri(RuntimeError):
    """SEC 429 dondurdu — pas bitirilir, kaldigi yerden devam edilir."""


def _tanitici() -> str:
    return os.environ.get(
        "SEC_USER_AGENT",
        "invest-research (kisisel arastirma; iletisim icin SEC_USER_AGENT)")


def _istek(url: str, zaman_asimi: int = 20) -> dict | None:
    r = requests.get(url, headers={"User-Agent": _tanitici(),
                                   "Accept-Encoding": "gzip, deflate"},
                     timeout=zaman_asimi)
    if r.status_code == 429:
        raise HizSiniri("SEC 429")
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return r.json()


def cik_haritasi(yenile: bool = False) -> dict:
    """Sembol -> sifir dolgulu CIK."""
    if not yenile:
        hit = _cache.peek(ALAN, "cik_haritasi")
        if hit and hit[0]:
            return hit[0]
    d = _istek(HARITA)
    harita = {str(v["ticker"]).upper(): str(v["cik_str"]).zfill(10)
              for v in (d or {}).values() if v.get("ticker")}
    if harita:
        _cache.put(ALAN, "cik_haritasi", harita)
    return harita


def _etkin_gun(kabul: str) -> str:
    """Bildirimin islem yapilabilir hale geldigi gun.

    Kabul saati UTC geliyor. ET'ye cevirip 16:00'dan sonraysa bir sonraki
    gune atiliyor. Hafta sonu duzeltmesi burada YAPILMIYOR: bar hizalamasi
    zaten yalnizca islem gunlerini biliyor ve cumartesiye dusen bir olayi
    ilk sonraki bara tasiyor. Burada iki kez duzeltmek, pazartesi olaylarini
    saliya kaydirirdi.
    """
    try:
        t = datetime.fromisoformat(kabul.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return kabul[:10]
    # Gercek New York saati, sabit bir fark degil.
    #
    # Ilk surum yaz saatini gormezden gelip her zaman UTC-5 kullaniyordu ve
    # yorumunda bunun "temkinli" oldugu yaziyordu. Tam tersiymis: yazin
    # 20:30 UTC gercekte 16:30 EDT, yani kapanistan sonra, ama UTC-5 ile
    # 15:30 hesaplaniyor ve AYNI gune yaziliyor. Bilanco bildirimlerinin
    # yarisi tam o saatte dosyalaniyor, yani bu dogrudan ileriye bakis
    # olurdu. ZoneInfo yoksa UTC-4'e dusuluyor: bu yonde hata olayi bir gun
    # GEC gostermek demek, ki kaybedilen bilgidir, uydurulan degil.
    try:
        et = t.astimezone(ZoneInfo("America/New_York"))
    except Exception:
        et = t.astimezone(timezone(timedelta(hours=-4)))
    gun = et.date()
    if et.hour >= 16:
        gun = gun + timedelta(days=1)
    return gun.isoformat()


def _kayitlari_ac(blok: dict) -> list[dict]:
    """EDGAR'in sutun-bazli kayitlarini satirlara cevirir."""
    form = blok.get("form") or []
    n = len(form)
    kabul = blok.get("acceptanceDateTime") or [""] * n
    tarih = blok.get("filingDate") or [""] * n
    madde = blok.get("items") or [""] * n
    erisim = blok.get("accessionNumber") or [""] * n
    out = []
    for i in range(n):
        if form[i] != "8-K":
            continue
        out.append({
            "form": form[i],
            "dosyalama": tarih[i],
            "kabul": kabul[i],
            "etkin_gun": _etkin_gun(kabul[i] or tarih[i]),
            "maddeler": madde[i] or "",
            # Belge metnini cekebilmek icin gerekiyor; 8-K'nin kendisi
            # cogunlukla XBRL etiketi, anlati EX-99.1'de.
            "erisim": erisim[i] or "",
        })
    return out


def cek_bir(ticker: str, cik: str, baslangic: str = "2016-01-01",
            bekle: float = BEKLE) -> list[dict]:
    """Bir sirketin 8-K gecmisi, istenen tarihe kadar geriye.

    `recent` blogu son 1000 bildirimi tutuyor; cok bildirim yapan bir
    sirkette bu yalnizca birkac yil demek (PLUG'da 2017'ye kadar). Bu yuzden
    en eski kayit hala istenen baslangictan sonraysa arsiv dosyalari da
    cekiliyor -- yoksa hizli bildirim yapanlarin ilk yillari sessizce bos
    kalirdi ve tam da onlar en cok olay yasayanlar.
    """
    d = _istek(f"{UC}CIK{cik}.json")
    if not d:
        return []
    kayitlar = _kayitlari_ac((d.get("filings") or {}).get("recent") or {})
    en_eski = min((k["dosyalama"] for k in kayitlar), default="9999")
    for ek in (d.get("filings") or {}).get("files", []):
        if en_eski <= baslangic:
            break
        time.sleep(bekle)
        blok = _istek(f"{UC}{ek['name']}")
        if not blok:
            continue
        yeni = _kayitlari_ac(blok)
        kayitlar.extend(yeni)
        en_eski = min((k["dosyalama"] for k in kayitlar), default="9999")
    kayitlar = [k for k in kayitlar if k["dosyalama"] >= baslangic]
    kayitlar.sort(key=lambda k: k["etkin_gun"])
    return kayitlar


def cek(semboller, baslangic: str = "2016-01-01", yenile: bool = False,
        bekle: float = BEKLE, ilerleme=None) -> dict:
    """Evrenin 8-K gecmisi. Onbellekte olani atlar, yani tekrar calisir."""
    harita = cik_haritasi()
    istenen = list(semboller)
    yazildi = atlandi = ciksiz = hatali = 0
    durduruldu = None
    for i, tk in enumerate(istenen):
        anahtar = f"{tk}:{baslangic}"
        if not yenile and _cache.peek(ALAN, anahtar):
            atlandi += 1
            continue
        cik = harita.get(str(tk).upper())
        if not cik:
            ciksiz += 1
            continue
        try:
            kayitlar = cek_bir(tk, cik, baslangic=baslangic, bekle=bekle)
        except HizSiniri as exc:
            durduruldu = str(exc)
            break
        except Exception:
            hatali += 1
            time.sleep(bekle)
            continue
        # Bos liste de yazilir: "bu sirketin 8-K'si yok" ile "bakmadik"
        # ayri seyler ve ikincisini her calistirmada tekrar denemek gerekir.
        _cache.put(ALAN, anahtar, {"kayitlar": kayitlar})
        yazildi += 1
        time.sleep(bekle)
        if ilerleme and (i + 1) % 200 == 0:
            ilerleme(i + 1, len(istenen), yazildi, atlandi)
    return {"istenen": len(istenen), "yazildi": yazildi, "atlandi": atlandi,
            "ciksiz": ciksiz, "hatali": hatali, "durduruldu": durduruldu}


def oku(ticker: str, baslangic: str = "2016-01-01") -> pd.DataFrame | None:
    """Onbellekten bir sirketin olaylari. Ag istegi YOK."""
    hit = _cache.peek(ALAN, f"{ticker}:{baslangic}")
    if not hit or not hit[0]:
        return None
    kayitlar = (hit[0] or {}).get("kayitlar") or []
    if not kayitlar:
        return pd.DataFrame(columns=["etkin_gun", "maddeler"])
    df = pd.DataFrame(kayitlar)
    df["etkin_gun"] = pd.to_datetime(df["etkin_gun"], errors="coerce")
    return df.dropna(subset=["etkin_gun"]).sort_values("etkin_gun")


def _obek_maskesi(maddeler: pd.Series) -> dict:
    """Madde kodu metnini obek bayraklarina cevirir."""
    m = maddeler.fillna("").astype(str)
    return {ad: m.apply(lambda s: any(k in s for k in kodlar))
            for ad, kodlar in OBEK.items()}


def ozellikler(olaylar: "pd.DataFrame | None", gunler: pd.DatetimeIndex,
               pencere: int = 21, uzun_pencere: int = 252) -> pd.DataFrame:
    """Bar bazinda olay ozellikleri.

    Hepsi GERIYE bakar ve `etkin_gun` ile hizalanir, `filingDate` ile degil --
    kapanistan sonra dosyalanan bir bildirim o gunun barinda henuz yoktur.

    - olay_gun_once : en son olaydan bu yana kac bar (yoksa `pencere` * 3)
    - olay_sayi21   : son `pencere` barda kac olay
    - olay_var5     : son 5 barda olay oldu mu
    - olay_yogunluk : son 21 bardaki oran, kendi 252 barlik ortalamasina gore
    - olay_<obek>   : son `pencere` barda o turden olay oldu mu
    """
    n = len(gunler)
    bos = pd.DataFrame(index=gunler)
    bos["olay_gun_once"] = float(pencere * 3)
    bos["olay_sayi21"] = 0.0
    bos["olay_var5"] = 0.0
    bos["olay_yogunluk"] = 1.0
    for ad in OBEK:
        bos[f"olay_{ad}"] = 0.0
    if olaylar is None or olaylar.empty or n == 0:
        return bos

    gun = pd.DatetimeIndex(gunler).normalize()
    # Olayi bara tasi: etkin gunu bir islem gunune denk gelmiyorsa (hafta
    # sonu, tatil) ILK SONRAKI bara. searchsorted "left" tam da bunu yapar.
    yer = np.searchsorted(gun.to_numpy(),
                          pd.DatetimeIndex(olaylar["etkin_gun"]).normalize()
                          .to_numpy(), side="left")
    gecerli = yer < n
    yer = yer[gecerli]
    if len(yer) == 0:
        return bos

    say = np.zeros(n)
    np.add.at(say, yer, 1.0)
    s = pd.Series(say, index=gun)
    bos["olay_sayi21"] = s.rolling(pencere, min_periods=1).sum().to_numpy()
    bos["olay_var5"] = (s.rolling(5, min_periods=1).sum() > 0).astype(float).to_numpy()
    uzun = s.rolling(uzun_pencere, min_periods=60).mean() * pencere
    with np.errstate(divide="ignore", invalid="ignore"):
        yog = bos["olay_sayi21"].to_numpy() / np.maximum(uzun.to_numpy(), 1e-9)
    bos["olay_yogunluk"] = np.where(np.isfinite(yog), yog, 1.0)

    # Son olaydan bu yana gecen bar sayisi.
    son = np.full(n, -1.0)
    son[yer] = yer
    son = pd.Series(son).replace(-1.0, np.nan).ffill().to_numpy()
    idx = np.arange(n, dtype=float)
    onceki = idx - son
    bos["olay_gun_once"] = np.where(np.isfinite(onceki), onceki,
                                    float(pencere * 3))

    maskeler = _obek_maskesi(olaylar["maddeler"])
    for ad, m in maskeler.items():
        v = np.zeros(n)
        secili = yer[m.to_numpy()[gecerli]]
        if len(secili):
            np.add.at(v, secili, 1.0)
        bos[f"olay_{ad}"] = (pd.Series(v, index=gun)
                             .rolling(pencere, min_periods=1).sum()
                             .gt(0).astype(float).to_numpy())
    return bos


def panele_ekle(yol, period: str = "10y", baslangic: str = "2016-01-01",
                parca: int = 250_000) -> dict:
    """Var olan bir panele olay sutunlarini ekler.

    Bu is neden panel kurulumunun ICINDE degil: olay ozellikleri yalnizca
    (hisse, tarih) ciftine bagli. Gostergelerin hesabina, kovalara, etikete
    hicbir sey borclu degiller. Agir dongunun icine konduklarinda her yeni
    bildirim 2600 hissenin 93 gostergesini bastan hesaplatiyordu -- kirk
    dakika, ve makinede bellek dararsa hic bitmiyor.

    Pencere hesabi HISSENIN TAM GUNLUK SERISI uzerinde yapilir, panelin
    tarihleri uzerinde degil. Kesit paneli her besinci gunu tutuyor; "son 21
    bar" o tarihler uzerinde sayilsaydi 105 takvim gunu demek olurdu ve
    bayraklar dort kat uzun yasardi.
    """
    yol = Path(yol)
    if not yol.exists():
        return {"ok": False, "reason": f"{yol} yok"}

    anahtar = pd.read_csv(yol, usecols=["ticker", "tarih"],
                          parse_dates=["tarih"])
    parcalar, kapsanan, olaysiz = [], 0, 0
    for tk, alt in anahtar.groupby("ticker", sort=False):
        hit = _cache.peek("yahoo", f"{tk}:{period}")
        if not hit or not hit[0] or hit[0].get("history") is None:
            continue
        tam = pd.DatetimeIndex(hit[0]["history"].index)
        try:
            tam = tam.tz_localize(None) if tam.tz is None else tam.tz_convert(None)
        except (TypeError, AttributeError):
            pass
        tam = tam.normalize()
        d = oku(tk, baslangic)
        if d is None:
            olaysiz += 1
        f = ozellikler(d, tam)
        # Panelin tarihlerine indir. reindex, panelde olup barlarda olmayan
        # bir tarih varsa NaN birakir; onlar asagida "olay yok" ile dolar.
        alt_f = f.reindex(pd.DatetimeIndex(alt["tarih"]).normalize())
        alt_f.index = alt.index
        parcalar.append(alt_f)
        kapsanan += 1
    if not parcalar:
        return {"ok": False, "reason": "hicbir hisse eslesmedi"}

    tablo = pd.concat(parcalar).reindex(anahtar.index)
    del parcalar
    varsayilan = ozellikler(None, pd.DatetimeIndex([pd.Timestamp("2020-01-01")]))
    for c in tablo.columns:
        tablo[c] = tablo[c].fillna(float(varsayilan[c].iloc[0])).astype("float32")

    gecici = yol.with_suffix(".olayli.csv")
    gecici.unlink(missing_ok=True)
    yazilan = 0
    for blok in pd.read_csv(yol, chunksize=parca):
        blok = blok.drop(columns=[c for c in tablo.columns if c in blok.columns])
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
            "hisse": kapsanan, "olaysiz_hisse": olaysiz,
            "sutun": list(tablo.columns)}


def etki(panel: pd.DataFrame, etiket: str = "akranmed_21g",
         ufuk_gun: int = 21) -> dict:
    """Bir olayin cevresinde ne oluyor — modelden bagimsiz olarak.

    Uc soru, uc tablo:

    1. Son olaydan bu yana gecen sureye gore ileri getiri. Olay gunu, ertesi
       hafta, ertesi ay, hic olay yok.
    2. Olayin TURUNE gore. Bilanco ile yonetici ayriligi ayni sey degil.
    3. Kurulum satirlarinda etkilesim: ayni kurulum, son bes gunde bir
       bildirim varken ve yokken.

    Ucuncusu kullanicinin asil sordugu sey: formasyon dogru gorunuyor olabilir
    ama sirket ici bir haberle birlikte baska bir sey ifade eder.

    Ortalamalarin yaninda MEDYAN ve gun sayisi da veriliyor. Bu panelde bir
    avuc satirin ortalamayi tasidigini bir kez gorduk; olay satirlarinda risk
    daha da buyuk, cunku en buyuk hareketler tam oralarda.
    """
    from .faktor_zaman import newey_west_t

    if etiket not in panel.columns or "olay_gun_once" not in panel.columns:
        return {"ok": False, "reason": f"{etiket} veya olay sutunlari yok"}
    d = panel[panel[etiket].notna()]
    if len(d) < 1000:
        return {"ok": False, "reason": f"{len(d)} satir"}

    gun = pd.DatetimeIndex(d["tarih"]).normalize()
    y = pd.to_numeric(d[etiket], errors="coerce").to_numpy(dtype=float)
    # Gunluk ortalamanin Newey-West t'si; satir bazinda bir t, ayni gunun
    # satirlarini bagimsiz sayip orneklemi kat kat sisirirdi.
    def _t(maske):
        if maske.sum() < 50:
            return None
        g = pd.Series(y[maske]).groupby(gun[maske].to_numpy()).mean()
        if len(g) < 10:
            return None
        tv, _, _ = newey_west_t(g.to_numpy(), lag=max(1, int(ufuk_gun)))
        return None if not np.isfinite(tv) else round(float(tv), 2)

    def _gunluk(maske):
        if not maske.any():
            return None
        return pd.Series(y[maske]).groupby(gun[maske].to_numpy()).mean()

    def _fark(maske, taban_maske):
        """Bu kovanin gunluk ortalamasi eksi taban kovanin AYNI gundeki hali.

        Sifira karsi test etmek burada yanlis soru. Etiket akran medyanina
        gore arindirilmis ve dagilimi saga carpik, yani her kova pozitif bir
        ortalama veriyor; ilginc olan kovalar ARASINDAKI fark. Gun bazinda
        eslestirmek piyasa gununu de aradan cikariyor.
        """
        a, b = _gunluk(maske), _gunluk(taban_maske)
        if a is None or b is None:
            return None, None
        d_ = (a - b).dropna()
        if len(d_) < 20:
            return None, None
        tv, _, _ = newey_west_t(d_.to_numpy(), lag=max(1, int(ufuk_gun)))
        return (round(float(d_.mean()), 5),
                None if not np.isfinite(tv) else round(float(tv), 2))

    def _satir(ad, maske, taban_maske=None):
        # Budanmis ortalama: ust ve alt %1 disarida. Ham ortalamanin yaninda
        # duruyor cunku bu panelde bir avuc satirin ortalamayi tasidigini
        # gorduk ve olay satirlarinda risk daha buyuk -- en buyuk hareketler
        # tam oralarda. Ikisi cok ayrisiyorsa sayi birkac isme aittir.
        v = y[maske] if maske.any() else np.array([])
        if len(v) > 100:
            alt, ust = np.nanpercentile(v, [1, 99])
            budanmis = float(np.nanmean(np.clip(v, alt, ust)))
        else:
            budanmis = float(np.nanmean(v)) if len(v) else None
        fark, fark_t = (_fark(maske, taban_maske)
                        if taban_maske is not None else (None, None))
        return {"kova": ad, "n": int(maske.sum()),
                "gun": int(gun[maske].nunique()) if maske.any() else 0,
                "ortalama": round(float(np.nanmean(v)), 5) if len(v) else None,
                "budanmis": round(budanmis, 5) if budanmis is not None else None,
                "pozitif": round(float(np.nanmean(v > 0)), 4) if len(v) else None,
                "t_nw": _t(maske), "fark": fark, "fark_t": fark_t}

    once = pd.to_numeric(d["olay_gun_once"], errors="coerce").to_numpy()
    kovalar = [
        ("olay gunu", once == 0),
        ("1-4 bar sonra", (once >= 1) & (once <= 4)),
        ("5-20 bar sonra", (once >= 5) & (once <= 20)),
        ("21-62 bar sonra", (once >= 21) & (once < 63)),
        ("olay yok / cok eski", once >= 63),
    ]
    # Taban: son 62 barda hicbir bildirim olmayan satirlar. Her sey buna
    # gore olculuyor.
    taban = once >= 63
    mesafe = [_satir(ad, m, None if ad.startswith("olay yok") else taban)
              for ad, m in kovalar]

    tur = []
    for ad in OBEK:
        s_ = f"olay_{ad}"
        if s_ not in d.columns:
            continue
        m = pd.to_numeric(d[s_], errors="coerce").fillna(0).to_numpy() > 0
        if m.sum() >= 200:
            tur.append(_satir(ad, m, taban))
    tur.sort(key=lambda r: -(r["fark"] or 0))

    etkilesim = []
    if "kurulum" in d.columns:
        var5 = pd.to_numeric(d["olay_var5"], errors="coerce").fillna(0).to_numpy() > 0
        for kid in sorted(pd.unique(d["kurulum"].astype(str))):
            k = (d["kurulum"].astype(str) == kid).to_numpy()
            ile, siz = k & var5, k & ~var5
            if ile.sum() < 200 or siz.sum() < 200:
                continue
            a, b = _satir(f"{kid} · olayli", ile, siz), _satir(
                f"{kid} · olaysiz", siz)
            etkilesim.append({"kurulum": kid, "olayli": a, "olaysiz": b,
                              "fark": a["fark"], "fark_t": a["fark_t"]})
        etkilesim.sort(key=lambda r: -abs(r["fark"] or 0))

    return {"ok": True, "etiket": etiket, "satir": int(len(d)),
            "olay_orani": round(float((once < 63).mean()), 4),
            "mesafe": mesafe, "tur": tur, "etkilesim": etkilesim}


def kapsam(semboller, baslangic: str = "2016-01-01") -> dict:
    """Onbellekte ne var: kac sirket, kac olay, hangi araliktan."""
    hisse = olay = 0
    ilk, son = None, None
    for tk in semboller:
        d = oku(tk, baslangic)
        if d is None:
            continue
        hisse += 1
        olay += len(d)
        if len(d):
            a, b = d["etkin_gun"].min(), d["etkin_gun"].max()
            ilk = a if ilk is None or a < ilk else ilk
            son = b if son is None or b > son else son
    return {"ok": hisse > 0, "hisse": hisse, "olay": olay,
            "olay_ortalama": round(olay / hisse, 1) if hisse else 0.0,
            "ilk": str(ilk)[:10] if ilk is not None else None,
            "son": str(son)[:10] if son is not None else None}
