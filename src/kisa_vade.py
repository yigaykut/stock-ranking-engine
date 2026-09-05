"""Kisa vadeli kurulum (setup) yakalama — gunluk barlar uzerinde.

BU MODUL NEDEN AYRI
-------------------
Sistemin geri kalani CAPRAZ KESITSEL ve 21 gunluk ufuklu: "bugun hangi hisse
digerlerinden daha iyi duruyor". Kisa vade bambaska bir soru soruyor: "bu
hissede su anda, onumuzdeki 3-10 gunde ise yarayabilecek bir kurulum var mi".

Ikisi karistirilmamali. Uzun vadeli skor bir SIRALAMADIR; buradaki cikti ise
bir OLAY tespitidir (var / yok). Ayni tabloya konursa ikisi de anlamini
kaybeder, cunku bir hisse hem siralamada 400. olup hem de bugun temiz bir
kurulum gosterebilir.

ONEMLI: BU MODUL TEK BASINA BIR SEY IDDIA ETMEZ
------------------------------------------------
Burada yalnizca kurulumun VAR OLDUGU tespit edilir. "Ise yarar mi" sorusunun
cevabi bu dosyada YOKTUR -- o, src/kalibrasyon.py'nin isi ve olculmus gecmis
frekansa dayanir. Bir kurulumun tespit edilmesi, o kurulumun kazandirdigi
anlamina gelmez; olculene kadar guven degeri "bilinmiyor" doner.

ILERIYE BAKIS YOK — TASARIMLA
------------------------------
Butun dedektorler VEKTORELDIR ve yalnizca geriye bakan pencereler kullanir
(rolling, shift(+n), ewm). Kodda `shift(-n)` veya `center=True` YOKTUR ve
olmamalidir; testler bunu ayrica kontrol eder. Boylece bir tarihteki sinyal,
seri o tarihte kesildiginde de aynidir.

KURULUMLARIN KAYNAGI
--------------------
Hepsi yaygin, nesnel olarak tanimlanabilir ve gunluk barla hesaplanabilir
kaliplar: mum formasyonlari (yutan boga, cekic), oynaklik sikismasi
(Bollinger/NR7), Connors tipi asiri satim geri cekilmesi, hacimli kirilim.
Hicbiri yeni degil; yeni olan, her birinin bu evrende OLCULUYOR olmasi.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from . import indicators as ind

# Kurulum bulunmasi icin gereken en az bar sayisi. 200 gunluk ortalama
# kullanan dedektorler var; altinda hepsi NaN doner.
MIN_BAR = 220

# Kisa vade ufuklari (islem gunu). Kalibrasyon bunlarin hepsini olcer;
# varsayilan raporlama 5 gun.
UFUKLAR = (3, 5, 10)
VARSAYILAN_UFUK = 5


# =============================================================================
#  Yardimcilar
# =============================================================================
def _govde(o: pd.Series, c: pd.Series) -> pd.Series:
    return (c - o).abs()


def _menzil(h: pd.Series, l: pd.Series) -> pd.Series:
    return (h - l).replace(0.0, np.nan)


def _0_1(x: pd.Series) -> pd.Series:
    """Guvenli sikistirma: NaN'lari 0, araligi [0, 1]."""
    return x.fillna(0.0).clip(0.0, 1.0)


def _dolar_hacim(c: pd.Series, v: pd.Series, n: int = 20) -> pd.Series:
    return (c * v).rolling(n).median()


# =============================================================================
#  Kurulum kaydi
# =============================================================================
@dataclass(frozen=True)
class Kurulum:
    id: str
    ad_tr: str
    yon: str                      # "long" | "short"
    dedektor: Callable[[pd.DataFrame], tuple[pd.Series, pd.Series]]
    aciklama_tr: str
    # Kurulumun dogal ufku (islem gunu). Kalibrasyon yine hepsini olcer ama
    # panoda bu one cikar.
    ufuk: int = VARSAYILAN_UFUK


KAYIT: dict[str, Kurulum] = {}


def kaydet(k: Kurulum) -> Kurulum:
    if k.id in KAYIT:
        raise ValueError(f"kurulum kimligi tekrar etti: {k.id}")
    KAYIT[k.id] = k
    return k


# =============================================================================
#  Dedektorler
#
#  Her dedektor (var, guc) dondurur:
#    var : bool Series   -- o barda kurulum olustu mu
#    guc : float Series  -- [0,1], kurulum ne kadar temiz olustu
#
#  `guc` bir olasilik DEGILDIR. Olasilik kalibrasyondan gelir.
# =============================================================================
def _boga_yutan(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Yutan boga: dun dusus mumu, bugun onu tamamen kapsayan yukselis mumu.

    Ek sart: kisa vadeli bir geri cekilmenin ARDINDAN gelmesi. Yukselisin
    ortasinda olusan yutan mum, literaturde de en zayif hali.
    """
    o, h, l, c = df["Open"], df["High"], df["Low"], df["Close"]
    dun_dusus = c.shift(1) < o.shift(1)
    bugun_yukselis = c > o
    kapsiyor = (o <= c.shift(1)) & (c >= o.shift(1))
    geri_cekilmis = c.shift(1) < c.shift(4)          # son 3 gunde asagi
    trend = c > ind.sma(c, 200)

    var = dun_dusus & bugun_yukselis & kapsiyor & geri_cekilmis & trend
    # Guc: bugunun govdesi dunun govdesinin kac kati (2 kat -> 1.0)
    oran = _govde(o, c) / _govde(o, c).shift(1).replace(0.0, np.nan)
    return var.fillna(False), _0_1((oran - 1.0) / 1.0)


def _cekic(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Cekic: uzun alt fitil, kucuk govde, gunun ust yarisinda kapanis.

    Yalnizca kisa vadeli DIP civarinda anlamli; ortada olusan ayni sekil
    bilgi tasimaz. Bu yuzden 10 gunun en dusugune yakinlik sarti var.
    """
    o, h, l, c = df["Open"], df["High"], df["Low"], df["Close"]
    menzil = _menzil(h, l)
    govde = _govde(o, c)
    alt_fitil = (o.where(o < c, c) - l)
    ust_fitil = (h - c.where(c > o, o))

    # Ust fitil sarti MENZILE gore olmali, govdeye gore degil: govde zaten
    # kucuk oldugu icin "ust fitil < govde" neredeyse hicbir cekicte saglanmaz
    # ve dedektor hicbir sey yakalamaz. Standart tanim menzilin ~%15'i.
    var = (
        (govde <= 0.35 * menzil)
        & (alt_fitil >= 2.0 * govde)
        & (ust_fitil <= 0.15 * menzil)
        & (c >= l + 0.55 * menzil)
        & (l <= l.rolling(10).min() * 1.01)         # 10 gunun dibine yakin
        & (c > ind.sma(c, 200))
    )
    guc = alt_fitil / menzil                        # fitil ne kadar baskin
    return var.fillna(False), _0_1((guc - 0.5) / 0.4)


def _ic_bar_nr7(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """NR7 + ic bar: son 7 gunun en dar menzili ve onceki barin icinde.

    Oynaklik sikismasi. Yonu kendisi soylemez; yukari trend sartiyla
    long tarafa cevriliyor.
    """
    o, h, l, c = df["Open"], df["High"], df["Low"], df["Close"]
    menzil = h - l
    nr7 = menzil <= menzil.rolling(7).min()
    ic_bar = (h <= h.shift(1)) & (l >= l.shift(1))
    trend = (c > ind.sma(c, 50)) & (ind.sma(c, 50) > ind.sma(c, 200))

    var = nr7 & ic_bar & trend
    # Guc: menzil, son 20 gun ortalamasinin ne kadar altinda
    daralma = 1.0 - (menzil / menzil.rolling(20).mean())
    return var.fillna(False), _0_1(daralma / 0.6)


def _hacimli_kirilim(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """20 gunun en yuksegini hacimle kirmak.

    Hacim sarti onemli: hacimsiz kirilim, literaturde en sik geri donen
    kurulumlardan biri.
    """
    h, c, v = df["High"], df["Close"], df["Volume"]
    onceki_zirve = h.rolling(20).max().shift(1)
    hacim_med = v.rolling(20).median()

    var = (c > onceki_zirve) & (v > 1.5 * hacim_med) & (c > ind.sma(c, 200))
    guc = v / hacim_med
    return var.fillna(False), _0_1((guc - 1.5) / 2.0)


def _ma20_geri_cekilme(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Yukselen MA20'ye geri cekilip ustunde kapanmak.

    "Trende geri donus" kurulumu. MA20'nin YUKSELIYOR olmasi sart; yatay
    ortalamaya deger her hisse bu kalibi doldurur ve hicbir sey anlatmaz.
    """
    l, c = df["Low"], df["Close"]
    ma20 = ind.sma(c, 20)
    ma20_yukseliyor = ma20 > ma20.shift(5)
    dokundu = l <= ma20 * 1.01
    kapanis_ustte = c > ma20
    trend = c > ind.sma(c, 200)

    var = ma20_yukseliyor & dokundu & kapanis_ustte & trend
    # Guc: temas ne kadar hassas (tam degip donduyse yuksek)
    sapma = (c - ma20).abs() / ma20
    return var.fillna(False), _0_1(1.0 - sapma / 0.04)


def _bollinger_sikismasi(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Bant genisligi son 120 gunun en dar %10'unda.

    Oynaklik dongusu: sikisma sonrasi genisleme. Yon soylemez; trend
    sartiyla long tarafa aliniyor.
    """
    c = df["Close"]
    ust, orta, alt = ind.bollinger(c, 20, 2.0)
    genislik = (ust - alt) / orta
    esik = genislik.rolling(120).quantile(0.10)
    trend = c > ind.sma(c, 200)

    var = (genislik <= esik) & trend
    guc = 1.0 - (genislik / genislik.rolling(120).median())
    return var.fillna(False), _0_1(guc / 0.6)


def _rsi2_asiri_satim(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """RSI(2) < 10, fiyat MA200 ustunde (Connors tipi).

    Uzun vadeli trend yukari, kisa vadede asiri satim. Ortalamaya donus
    ailesinin en cok test edilmis kurulumlarindan.
    """
    c = df["Close"]
    r2 = ind.rsi(c, 2)
    var = (r2 < 10) & (c > ind.sma(c, 200)) & (c > ind.sma(c, 50) * 0.92)
    return var.fillna(False), _0_1((10.0 - r2) / 10.0)


def _bosluk_dolumu(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Asagi boslukla acilip boslugu kapatarak yukari donmek."""
    o, h, l, c = df["Open"], df["High"], df["Low"], df["Close"]
    onceki_kapanis = c.shift(1)
    bosluk = (onceki_kapanis - o) / onceki_kapanis

    var = (bosluk > 0.03) & (c > o) & (c >= onceki_kapanis - 0.5 *
                                       (onceki_kapanis - o)) & (c > ind.sma(c, 200))
    return var.fillna(False), _0_1((bosluk - 0.03) / 0.05)


def _uc_gun_geri_cekilme(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Yukselen trendde ust uste 3 dusen kapanis."""
    c = df["Close"]
    dus = c < c.shift(1)
    var = dus & dus.shift(1) & dus.shift(2) & (c > ind.sma(c, 200)) & \
        (ind.sma(c, 50) > ind.sma(c, 50).shift(10))
    toplam = (c.shift(3) - c) / c.shift(3)
    return var.fillna(False), _0_1(toplam / 0.08)


def _hacim_kurumasi(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Yukselen MA50 destegi civarinda hacmin kurumasi.

    Satis baskisinin bitmesinin klasik isareti: fiyat destege gelir ama
    kimse satmiyordur.
    """
    l, c, v = df["Low"], df["Close"], df["Volume"]
    ma50 = ind.sma(c, 50)
    hacim_med = v.rolling(50).median()

    var = (
        (ma50 > ma50.shift(10))
        & (l <= ma50 * 1.03) & (c > ma50)
        & (v < 0.6 * hacim_med)
        & (c > ind.sma(c, 200))
    )
    return var.fillna(False), _0_1((0.6 - v / hacim_med) / 0.4)


# --- Uyari tarafi (izleme listesi cikis sinyalleri icin) --------------------
def _yutan_ayi(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Yutan ayi: zirve civarinda, dunun yukselis mumunu yutan dusus mumu."""
    o, h, l, c = df["Open"], df["High"], df["Low"], df["Close"]
    var = (
        (c.shift(1) > o.shift(1)) & (c < o)
        & (o >= c.shift(1)) & (c <= o.shift(1))
        & (h >= h.rolling(20).max() * 0.98)
    )
    oran = _govde(o, c) / _govde(o, c).shift(1).replace(0.0, np.nan)
    return var.fillna(False), _0_1((oran - 1.0) / 1.0)


def _dagitim_gunu(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Zirve civarinda yuksek hacimli dusus gunu (kurumsal satis izi)."""
    h, c, v = df["High"], df["Close"], df["Volume"]
    hacim_med = v.rolling(20).median()
    var = (
        (c < c.shift(1) * 0.98)
        & (v > 1.5 * hacim_med)
        & (c >= c.rolling(60).max() * 0.90)
    )
    return var.fillna(False), _0_1((v / hacim_med - 1.5) / 2.0)


# =============================================================================
#  Kayit
# =============================================================================
kaydet(Kurulum("boga_yutan", "Yutan boga mumu", "long", _boga_yutan,
               "Dun dusus, bugun onu tamamen kapsayan yukselis mumu; kisa bir "
               "geri cekilmenin ardindan ve MA200 ustunde.", 5))
kaydet(Kurulum("cekic", "Cekic mumu", "long", _cekic,
               "Uzun alt fitil, kucuk govde, gunun ust yarisinda kapanis; 10 "
               "gunun dibine yakin.", 5))
kaydet(Kurulum("nr7_ic_bar", "NR7 + ic bar sikismasi", "long", _ic_bar_nr7,
               "Son 7 gunun en dar menzili ve onceki barin icinde: oynaklik "
               "sikismasi.", 10))
kaydet(Kurulum("hacimli_kirilim", "Hacimli 20 gun kirilimi", "long",
               _hacimli_kirilim,
               "20 gunun zirvesini medyan hacmin 1.5 katiyla kirmak.", 10))
kaydet(Kurulum("ma20_geri_cekilme", "MA20 geri cekilmesi", "long",
               _ma20_geri_cekilme,
               "Yukselen 20 gunluk ortalamaya deyip ustunde kapanmak.", 5))
kaydet(Kurulum("bollinger_sikismasi", "Bollinger sikismasi", "long",
               _bollinger_sikismasi,
               "Bant genisligi son 120 gunun en dar %10'unda.", 10))
kaydet(Kurulum("rsi2_asiri_satim", "RSI(2) asiri satim", "long",
               _rsi2_asiri_satim,
               "RSI(2) 10'un altinda ve fiyat MA200 ustunde.", 3))
kaydet(Kurulum("bosluk_dolumu", "Asagi bosluk dolumu", "long", _bosluk_dolumu,
               "%3+ asagi boslukla acilip boslugu kapatarak yukari donmek.", 3))
kaydet(Kurulum("uc_gun_geri_cekilme", "Uc gun geri cekilme", "long",
               _uc_gun_geri_cekilme,
               "Yukselen trendde ust uste uc dusen kapanis.", 5))
kaydet(Kurulum("hacim_kurumasi", "MA50'de hacim kurumasi", "long",
               _hacim_kurumasi,
               "Yukselen MA50 destegi civarinda hacmin medyanin %60'inin "
               "altina dusmesi.", 10))
kaydet(Kurulum("yutan_ayi", "Yutan ayi mumu", "short", _yutan_ayi,
               "Zirve civarinda dunun yukselis mumunu yutan dusus mumu.", 5))
kaydet(Kurulum("dagitim_gunu", "Dagitim gunu", "short", _dagitim_gunu,
               "Zirve civarinda yuksek hacimli dusus gunu.", 5))


# =============================================================================
#  Kosullar — kalibrasyonun "hangi durumda" tarafi
# =============================================================================
def kosullar(df: pd.DataFrame) -> pd.DataFrame:
    """Her bar icin, kurulumun olustugu ORTAMI tarif eden kategoriler.

    Kalibrasyon bunlara gore kirilim yapar: ayni kurulum sakin piyasada
    baska, oynak piyasada baska calisiyor olabilir. Sayisal degerler degil
    KOVA adlari tutulur; kova sayisi az olsun ki her kovada olcum yapacak
    kadar gozlem birikSin.
    """
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]
    atr_pct = ind.atr(h, l, c, 14) / c
    dv = _dolar_hacim(c, v, 20)
    ma200 = ind.sma(c, 200)
    uzaklik = (c / ma200 - 1.0)

    out = pd.DataFrame(index=df.index)
    out["oynaklik"] = pd.cut(atr_pct, [-np.inf, 0.02, 0.045, np.inf],
                             labels=["sakin", "orta", "oynak"]).astype(object)
    out["likidite"] = pd.cut(dv, [-np.inf, 2e6, 2e7, np.inf],
                             labels=["ince", "orta", "kalin"]).astype(object)
    out["trend_konumu"] = pd.cut(uzaklik, [-np.inf, 0.0, 0.15, np.inf],
                                 labels=["ma200_alti", "yakin", "uzak"]).astype(object)
    return out


# =============================================================================
#  Tarama
# =============================================================================
def tespit(df: pd.DataFrame, kurulumlar: "list[str] | None" = None
           ) -> pd.DataFrame:
    """Butun barlar icin butun kurulumlari isaretler.

    Doner: satir = bar, sutun = MultiIndex (kurulum_id, 'var'|'guc')
    Bu VEKTOREL bir hesaptir; tek bir gun icin de tum seri hesaplanip son
    satir okunur. Boylece "bugun" ve "gecmis" ayni kodla uretilir -- iki ayri
    yol olsaydi biri digerinden sapabilir ve kalibrasyon yanlis kurulumu
    olcerdi.
    """
    ids = kurulumlar or list(KAYIT)
    parcalar = {}
    for kid in ids:
        k = KAYIT[kid]
        try:
            var, guc = k.dedektor(df)
        except Exception:
            var = pd.Series(False, index=df.index)
            guc = pd.Series(0.0, index=df.index)
        parcalar[(kid, "var")] = var.reindex(df.index).fillna(False).astype(bool)
        parcalar[(kid, "guc")] = guc.reindex(df.index).fillna(0.0).astype(float)
    out = pd.DataFrame(parcalar, index=df.index)
    out.columns = pd.MultiIndex.from_tuples(out.columns)
    return out


def bugun(df: pd.DataFrame, ticker: str = "") -> list[dict]:
    """Serinin SON barinda olusan kurulumlar."""
    if df is None or len(df) < MIN_BAR:
        return []
    t = tespit(df)
    ks = kosullar(df)
    son = df.index[-1]
    out = []
    for kid, k in KAYIT.items():
        if not bool(t[(kid, "var")].iloc[-1]):
            continue
        out.append({
            "ticker": ticker,
            "tarih": str(pd.Timestamp(son).date()),
            "kurulum": kid,
            "ad_tr": k.ad_tr,
            "yon": k.yon,
            "ufuk": k.ufuk,
            "guc": round(float(t[(kid, "guc")].iloc[-1]), 3),
            "oynaklik": ks["oynaklik"].iloc[-1],
            "likidite": ks["likidite"].iloc[-1],
            "trend_konumu": ks["trend_konumu"].iloc[-1],
            "fiyat": round(float(df["Close"].iloc[-1]), 2),
        })
    return out


def tara(bundles: dict, kurulumlar: "list[str] | None" = None) -> pd.DataFrame:
    """Butun evrende bugunku kurulumlar."""
    satirlar = []
    for tk, b in (bundles or {}).items():
        h = (b or {}).get("history")
        if h is None or len(h) < MIN_BAR:
            continue
        try:
            satirlar.extend(bugun(h, tk))
        except Exception:
            continue
    if not satirlar:
        return pd.DataFrame(columns=["ticker", "tarih", "kurulum", "ad_tr", "yon",
                                     "ufuk", "guc", "oynaklik", "likidite",
                                     "trend_konumu", "fiyat"])
    return pd.DataFrame(satirlar)
