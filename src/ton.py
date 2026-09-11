"""Bildirimin TONU — 8-K'nin soyledigi seyin metni.

NEDEN
-----
`olay.py` bir 8-K'nin TURUNU biliyor: bilanco, yonetici degisikligi, onemli
sozlesme. Tonunu bilmiyor. "Yonetici ayrildi" ile "yonetici gorevden
alindi", madde kodunda ayni satir.

Bu modul o eksigi kapatmayi deniyor: bildirimin basin bulteni metnini alip
finansal sozlukle puanliyor.

METIN NEREDE
------------
8-K'nin birincil belgesi cogunlukla XBRL etiketinden ibaret -- AAPL'in
bilanco bildiriminde 4.200 karakter ve tek cumle anlati yok. Asil metin
EX-99.1 ekinde (ayni bildirimde 10.400 karakter). Tam gonderi dosyasi
(.txt) tek istekte hepsini veriyor ve icinden EX-99 blogu ayiklanabiliyor;
ayri ayri indirmek iki kat istek eder.

SOZLUK HAKKINDA DURUST OLMAK
----------------------------
Asagidaki liste elle derlenmis, ~200 kelimelik bir finans sozlugu. Akademik
standart olan Loughran-McDonald sozlugu ~2.700 kelime ve bu onun yerine
gecmez; buradaki amac "ton olculebilir bir sey tasiyor mu" sorusuna ucuz bir
on cevap vermek. Tasiyorsa tam sozlukle tekrar edilmeli.

Iki bilinen zayiflik, bastan:

  - Olumsuzlama okunmuyor. "no material weakness" ile "material weakness"
    ayni puani alir. LM literaturu bunun etkisinin sanildigindan kucuk
    oldugunu soyluyor ama sifir degil.
  - Bilanco bultenlerinde "increase"/"decrease" surekli geciyor ve puanin
    buyuk kismini onlar belirliyor. Bu aslinda istenen sey -- ceyrek iyi mi
    kotu mu -- ama "ton" degil "sonuc" olculdugu anlamina da geliyor.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
import requests

from .olay import ALAN as OLAY_ALAN, _tanitici          # noqa: F401
from .providers import cache as _cache

ARSIV = "https://www.sec.gov/Archives/edgar/data"
ALAN = "sec_ton"

OLUMSUZ = """
loss losses lost decline declined declines decrease decreased decreasing
weak weaker weakness weakened deteriorate deteriorated deterioration
impairment impaired writedown write-down writeoff charge charges
litigation lawsuit lawsuits sued defendant settlement penalty penalties
investigation subpoena inquiry violation violations breach breached
default defaulted delinquent bankruptcy insolvency restructuring
terminate terminated termination resign resigned resignation departure
dismissed removed vacated discontinued discontinuation shutdown closure
downgrade downgraded negative adverse adversely unfavorable unfavorably
shortfall miss missed below disappointing disappointed concern concerns
risk risks uncertain uncertainty uncertainties doubt doubts
delay delayed delays postpone postponed suspension suspended
recall recalled defect defective failure failed failing
decrease reduction reduced cut cuts curtail curtailed
covenant waiver going-concern dilution dilutive
lay-off layoff layoffs workforce-reduction furlough
warning warned caution cautioned cautionary
""".split()

OLUMLU = """
record records growth grew growing increase increased increases increasing
strong stronger strength improve improved improvement improvements
exceed exceeded exceeding beat outperform outperformed surpass surpassed
gain gains profit profits profitable profitability margin-expansion
expand expanded expansion accelerate accelerated acceleration
upgrade upgraded favorable favorably positive positively
success successful successfully achieve achieved achievement
milestone breakthrough approval approved authorized authorization
partnership collaboration agreement award awarded contract contracts
dividend buyback repurchase guidance-raised raised raising
efficient efficiency optimize optimized robust solid
demand momentum leadership innovative innovation
""".split()

_KELIME = re.compile(r"[a-z][a-z\-]+")


def _metni_ayikla(ham: str) -> str:
    """Tam gonderiden EX-99 bloklarinin duz metni.

    EX-99 yoksa bos doner. Birincil belgeye DUSULMEZ: orasi XBRL etiketi ve
    sayisal bir ton uretmez, uretse de baska bir seyi olcmus oluruz.
    """
    parcalar = []
    for blok in re.findall(r"(?is)<DOCUMENT>(.*?)</DOCUMENT>", ham):
        tip = re.search(r"(?i)<TYPE>\s*([^\s<]+)", blok)
        if not tip or not tip.group(1).upper().startswith("EX-99"):
            continue
        b = re.sub(r"(?is)<(script|style).*?</\1>", " ", blok)
        b = re.sub(r"(?is)<XBRL>.*?</XBRL>", " ", b)
        b = re.sub(r"<[^>]+>", " ", b)
        b = re.sub(r"&[a-z]+;|&#\d+;", " ", b)
        parcalar.append(b)
    return re.sub(r"\s+", " ", " ".join(parcalar)).strip()


def belge(cik: str, erisim: str, yenile: bool = False) -> str | None:
    """Bir bildirimin basin bulteni metni. Onbelleklenir."""
    if not erisim:
        return None
    anahtar = erisim
    if not yenile:
        hit = _cache.peek(ALAN, anahtar)
        if hit and hit[0] is not None:
            return hit[0].get("metin")
    from .olay import HizSiniri

    kisa = erisim.replace("-", "")
    u = f"{ARSIV}/{int(cik)}/{kisa}/{erisim}.txt"
    # Bazi gonderiler on megabayti geciyor (gomulu gorsel, tam mali tablolar)
    # ve tek bir yavas istek butun pasi durduruyordu. Iki deneme, sonra pas
    # gec -- ama BOS OLARAK ONBELLEGE YAZMA: yazsak, gecici bir zaman asimi
    # kalici bir "bu bildirimde metin yok" kaydina donusurdu.
    r = None
    for deneme in (0, 1):
        try:
            r = requests.get(u, headers={"User-Agent": _tanitici(),
                                         "Accept-Encoding": "gzip, deflate"},
                             timeout=25)
            break
        except requests.exceptions.RequestException:
            if deneme:
                return None
    if r is None:
        return None
    if r.status_code == 429:
        raise HizSiniri("SEC 429")
    if r.status_code != 200:
        _cache.put(ALAN, anahtar, {"metin": ""})
        return ""
    m = _metni_ayikla(r.text)
    _cache.put(ALAN, anahtar, {"metin": m})
    return m


def puan(metin: str | None) -> dict:
    """Metnin ton puani.

    `ton` = (olumlu - olumsuz) / (olumlu + olumsuz). Toplam kelimeye degil
    DUYGU YUKLU kelime sayisina bolunuyor: uzun bir bulten daha cok her
    turden kelime icerir ve uzunluga bolmek puani metnin boyuna baglar.
    Ayrica `olumsuz_oran` veriliyor, cunku literaturde ise yarayan taraf
    genellikle olumsuz kelimelerin YOGUNLUGU.
    """
    if not metin:
        return {"ton": np.nan, "olumsuz_oran": np.nan, "kelime": 0}
    kelimeler = _KELIME.findall(metin.lower())
    n = len(kelimeler)
    if n < 50:
        return {"ton": np.nan, "olumsuz_oran": np.nan, "kelime": n}
    art = sum(1 for k in kelimeler if k in _OLUMLU)
    eks = sum(1 for k in kelimeler if k in _OLUMSUZ)
    yuklu = art + eks
    return {
        "ton": (art - eks) / yuklu if yuklu >= 5 else np.nan,
        "olumsuz_oran": eks / n,
        "kelime": n,
    }


_OLUMLU = frozenset(OLUMLU)
_OLUMSUZ = frozenset(OLUMSUZ)


def cek(kayitlar, cik: str, bekle: float = 0.12, ilerleme=None) -> dict:
    """Bir sirketin bildirimleri icin metinleri indirir (onbellekli)."""
    import time

    from .olay import HizSiniri

    alinan = atlanan = bos = 0
    for i, k in enumerate(kayitlar):
        e = k.get("erisim")
        if not e:
            continue
        if _cache.peek(ALAN, e):
            atlanan += 1
            continue
        try:
            m = belge(cik, e)
        except HizSiniri:
            return {"alinan": alinan, "atlanan": atlanan, "bos": bos,
                    "durduruldu": "429"}
        except Exception:
            m = None
        if m is None:
            continue
        alinan += 1
        if not m:
            bos += 1
        time.sleep(bekle)
        if ilerleme and (i + 1) % 100 == 0:
            ilerleme(i + 1, len(kayitlar), alinan)
    return {"alinan": alinan, "atlanan": atlanan, "bos": bos,
            "durduruldu": None}


def cerceve(kayitlar) -> pd.DataFrame:
    """Onbellekteki metinlerden ton tablosu. Ag istegi YOK."""
    satir = []
    for k in kayitlar:
        e = k.get("erisim")
        if not e:
            continue
        hit = _cache.peek(ALAN, e)
        if not hit or hit[0] is None:
            continue
        p = puan((hit[0] or {}).get("metin"))
        satir.append({"etkin_gun": k["etkin_gun"], "maddeler": k["maddeler"],
                      **p})
    if not satir:
        return pd.DataFrame(columns=["etkin_gun", "maddeler", "ton",
                                     "olumsuz_oran", "kelime"])
    d = pd.DataFrame(satir)
    d["etkin_gun"] = pd.to_datetime(d["etkin_gun"], errors="coerce")
    return d.dropna(subset=["etkin_gun"]).sort_values("etkin_gun")
