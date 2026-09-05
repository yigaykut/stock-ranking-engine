"""Kisa vadeli kurulum dedektorleri.

En kritik test ILERIYE BAKIS testidir. Bir dedektor yanlislikla gelecege
bakarsa -- ornegin `center=True` bir rolling ya da `shift(-1)` -- gecmiste
mukemmel calisir, canlida hicbir sey uretmez ve bunu fark etmek cok zordur.
Cunku hata sessizdir: kod calisir, sayilar makul gorunur, kalibrasyon parlak
sonuclar verir. Bu yuzden iki katmanli kontrol var:

  1. DAVRANIS: seriyi t tarihinde KESIP yeniden hesaplayinca t'deki sinyal
     ayni cikmali. Gelecege bakan bir dedektor burada duser.
  2. KAYNAK KODU: dosyada negatif shift veya center=True gecmemeli. Davranis
     testi orneklem bazlidir; kaynak kontrolu kalibi komple yasaklar.

Geri kalan testler her dedektorun ADINI HAK EDIP ETMEDIGINI kovaliyor:
kurulumu elle kurulmus barlarda yakalamali, duz/rastgele seride ise nadiren
tetiklenmeli. "Calisti, patlamadi" yeterli degil.

Calistir:  python tests/test_kisa_vade.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import kisa_vade as kv        # noqa: E402

fails = 0


def check(name: str, cond: bool, extra: str = "") -> None:
    global fails
    if cond:
        print(f"  OK    {name}" + (f"  {extra}" if extra else ""))
    else:
        print(f"  HATA  {name}" + (f"  {extra}" if extra else ""))
        fails += 1


# ---------------------------------------------------------------------------
#  Bar uretici
# ---------------------------------------------------------------------------
def taban(n: int = 300, egim: float = 0.0018, seed: int = 0,
          oynaklik: float = 0.007) -> pd.DataFrame:
    """Yukselen, gurultulu bir seri. Kurulumlar buna eklenir.

    Egim/oynaklik orani bilerek yuksek: dedektorlerin cogunda MA200 ustunde
    olma sarti var, ve zayif egimli bir rastgele yuruyus tohum sansina gore
    MA200'un ALTINDA bitebiliyor. O zaman test, dedektoru degil tohumu olcer.
    """
    r = np.random.default_rng(seed)
    getiri = r.normal(egim, oynaklik, n)
    kapanis = 50.0 * np.exp(np.cumsum(getiri))
    acilis = kapanis * (1 + r.normal(0, 0.003, n))
    yuksek = np.maximum(acilis, kapanis) * (1 + np.abs(r.normal(0, 0.004, n)))
    dusuk = np.minimum(acilis, kapanis) * (1 - np.abs(r.normal(0, 0.004, n)))
    hacim = r.integers(800_000, 1_200_000, n).astype(float)
    idx = pd.bdate_range("2024-01-01", periods=n)
    return pd.DataFrame({"Open": acilis, "High": yuksek, "Low": dusuk,
                         "Close": kapanis, "Volume": hacim}, index=idx)


def bar_yaz(df: pd.DataFrame, i: int, o, h, l, c, v=None) -> None:
    """Belirli bir bari elle ayarla (fiyatlar son kapanisa oranli)."""
    df.iloc[i, df.columns.get_loc("Open")] = o
    df.iloc[i, df.columns.get_loc("High")] = h
    df.iloc[i, df.columns.get_loc("Low")] = l
    df.iloc[i, df.columns.get_loc("Close")] = c
    if v is not None:
        df.iloc[i, df.columns.get_loc("Volume")] = v


print("=" * 72)
print("1) KAYIT DEFTERI")
print("=" * 72)

check("kurulum var", len(kv.KAYIT) >= 10, f"{len(kv.KAYIT)} kurulum")
check("kimlikler benzersiz", len(set(kv.KAYIT)) == len(kv.KAYIT))
check("yonler gecerli",
      all(k.yon in ("long", "short") for k in kv.KAYIT.values()))
check("hepsinin aciklamasi var",
      all(len(k.aciklama_tr) > 25 for k in kv.KAYIT.values()))
check("hepsinin ufku makul",
      all(1 <= k.ufuk <= 20 for k in kv.KAYIT.values()))
try:
    kv.kaydet(kv.Kurulum("cekic", "x", "long", lambda d: (None, None), "y" * 30))
    check("ayni kimlik iki kez kaydedilemiyor", False)
except ValueError:
    check("ayni kimlik iki kez kaydedilemiyor", True)

print()
print("=" * 72)
print("2) ILERIYE BAKIS YOK — kaynak kodu")
print("=" * 72)

def sadece_kod(yol: Path) -> str:
    """Yorumlari ve metin sabitlerini atarak kaynagi dondurur.

    Duz metin aramasi yetmiyor: modulun kendi aciklamasinda "center=True
    YOKTUR" yaziyor ve naif bir arama bunu ihlal sanip patliyor. Tokenize ile
    ayiklamak, testi "kodda var mi" sorusuna geri getiriyor.
    """
    import io
    import tokenize
    parcalar = []
    with io.open(yol, encoding="utf-8") as f:
        for tok in tokenize.generate_tokens(f.readline):
            if tok.type in (tokenize.COMMENT, tokenize.STRING):
                continue
            parcalar.append(tok.string)
    return " ".join(parcalar)


kod = sadece_kod(ROOT / "src" / "kisa_vade.py")
check("negatif shift yok", not re.search(r"shift\s*\(\s*-", kod))
check("center=True yok", not re.search(r"center\s*=\s*True", kod))
check("bfill/backfill yok", not re.search(r"(bfill|backfill)", kod))
check("ileri getiri hesabi burada YOK",
      not re.search(r"fwd|forward|ileri_getiri", kod))

print()
print("=" * 72)
print("3) ILERIYE BAKIS YOK — davranis")
print("=" * 72)

df = taban(400, seed=5)
tam = kv.tespit(df)
sapan = []
for kes in (250, 300, 330, 370, 399):
    kismi = kv.tespit(df.iloc[:kes + 1])
    for kid in kv.KAYIT:
        a = bool(tam[(kid, "var")].iloc[kes])
        b = bool(kismi[(kid, "var")].iloc[-1])
        if a != b:
            sapan.append((kid, kes, a, b))
check("seri kesilince sinyal degismiyor", not sapan, str(sapan[:3]))

# Guc degerleri de degismemeli
guc_sapan = []
for kes in (300, 370):
    kismi = kv.tespit(df.iloc[:kes + 1])
    for kid in kv.KAYIT:
        a = float(tam[(kid, "guc")].iloc[kes])
        b = float(kismi[(kid, "guc")].iloc[-1])
        if abs(a - b) > 1e-9:
            guc_sapan.append((kid, kes, round(a, 4), round(b, 4)))
check("guc degeri de degismiyor", not guc_sapan, str(guc_sapan[:3]))

# GELECEGI BOZ: t'den SONRAKI barlari degistir, t'deki sinyal ayni kalmali
bozuk = df.copy()
bozuk.iloc[320:] = bozuk.iloc[320:] * 3.0
bozuk_t = kv.tespit(bozuk)
fark = [kid for kid in kv.KAYIT
        if bool(tam[(kid, "var")].iloc[319]) != bool(bozuk_t[(kid, "var")].iloc[319])]
check("sonraki barlari 3x yapmak onceki sinyali degistirmiyor", not fark,
      str(fark))

print()
print("=" * 72)
print("4) DEDEKTORLER — kurulumu YAKALIYOR mu")
print("=" * 72)

# --- yutan boga: uc gun dususten sonra kapsayan yukselis mumu
d = taban(seed=1)
i = len(d) - 1
p = float(d["Close"].iloc[i - 5])
for j, oran in enumerate([0.98, 0.96, 0.945]):           # geri cekilme
    bar_yaz(d, i - 3 + j, p * oran * 1.005, p * oran * 1.01, p * oran * 0.99, p * oran)
bar_yaz(d, i - 1, p * 0.955, p * 0.958, p * 0.938, p * 0.94)   # dusus mumu
bar_yaz(d, i, p * 0.935, p * 0.985, p * 0.933, p * 0.98)       # yutan
t = kv.tespit(d)
check("yutan boga yakalandi", bool(t[("boga_yutan", "var")].iloc[-1]),
      f"guc {t[('boga_yutan', 'guc')].iloc[-1]:.2f}")

# --- cekic: 10 gunun dibinde uzun alt fitil
d = taban(seed=2)
i = len(d) - 1
p = float(d["Close"].iloc[i - 12])
# Inis sig tutuluyor: dedektorde MA200 ustunde olma sarti var, derin bir
# geri cekilme fiyati MA200 altina indirir ve kurulum zaten gecersizlesir.
for j in range(10):                                   # dibe dogru inis
    q = p * (1 - 0.003 * j)
    bar_yaz(d, i - 10 + j, q, q * 1.004, q * 0.996, q * 0.999)
dip = float(d["Close"].iloc[i - 1])
bar_yaz(d, i, dip * 1.0005, dip * 1.002, dip * 0.965, dip * 1.0)
t = kv.tespit(d)
check("cekic yakalandi", bool(t[("cekic", "var")].iloc[-1]),
      f"guc {t[('cekic', 'guc')].iloc[-1]:.2f}")

# --- hacimli kirilim: 20 gunun zirvesi + 3x hacim
d = taban(seed=3)
i = len(d) - 1
zirve = float(d["High"].iloc[i - 20:i].max())
bar_yaz(d, i, zirve * 1.001, zirve * 1.05, zirve * 0.999, zirve * 1.04,
        v=float(d["Volume"].iloc[i - 20:i].median()) * 3.0)
t = kv.tespit(d)
check("hacimli kirilim yakalandi", bool(t[("hacimli_kirilim", "var")].iloc[-1]),
      f"guc {t[('hacimli_kirilim', 'guc')].iloc[-1]:.2f}")

# --- NR7 + ic bar: son 7 gunun en dar bari, onceki barin icinde
d = taban(seed=4)
i = len(d) - 1
p = float(d["Close"].iloc[i - 1])
# Son bar, onceki 6 barin HEPSINDEN dar olmali (NR7 tanimi). Taban serinin
# gunluk menzili ~%0.7, o yuzden buradaki bar belirgin sekilde daha dar.
bar_yaz(d, i - 1, p * 0.99, p * 1.02, p * 0.98, p)
bar_yaz(d, i, p * 1.0002, p * 1.0008, p * 0.9995, p * 1.0004)
t = kv.tespit(d)
check("NR7 + ic bar yakalandi", bool(t[("nr7_ic_bar", "var")].iloc[-1]),
      f"guc {t[('nr7_ic_bar', 'guc')].iloc[-1]:.2f}")

# --- asagi bosluk dolumu
d = taban(seed=6)
i = len(d) - 1
onceki = float(d["Close"].iloc[i - 1])
bar_yaz(d, i, onceki * 0.95, onceki * 1.005, onceki * 0.948, onceki * 1.002)
t = kv.tespit(d)
check("bosluk dolumu yakalandi", bool(t[("bosluk_dolumu", "var")].iloc[-1]),
      f"guc {t[('bosluk_dolumu', 'guc')].iloc[-1]:.2f}")

# --- uc gun geri cekilme
d = taban(seed=7)
i = len(d) - 1
p = float(d["Close"].iloc[i - 3])
for j, oran in enumerate([0.985, 0.97, 0.958]):
    bar_yaz(d, i - 2 + j, p * oran * 1.004, p * oran * 1.008, p * oran * 0.996, p * oran)
t = kv.tespit(d)
check("uc gun geri cekilme yakalandi",
      bool(t[("uc_gun_geri_cekilme", "var")].iloc[-1]))

# --- yutan ayi (short tarafi)
d = taban(seed=8)
i = len(d) - 1
zirve = float(d["High"].iloc[i - 20:i].max())
bar_yaz(d, i - 1, zirve * 0.99, zirve * 1.001, zirve * 0.985, zirve * 1.0)
bar_yaz(d, i, zirve * 1.005, zirve * 1.006, zirve * 0.96, zirve * 0.975)
t = kv.tespit(d)
check("yutan ayi yakalandi", bool(t[("yutan_ayi", "var")].iloc[-1]))

# --- dagitim gunu
d = taban(seed=9)
i = len(d) - 1
onceki = float(d["Close"].iloc[i - 1])
bar_yaz(d, i, onceki, onceki * 1.001, onceki * 0.955, onceki * 0.96,
        v=float(d["Volume"].iloc[i - 20:i].median()) * 2.5)
t = kv.tespit(d)
check("dagitim gunu yakalandi", bool(t[("dagitim_gunu", "var")].iloc[-1]))

print()
print("=" * 72)
print("5) YANLIS ALARM — duz seride tetiklenmemeli")
print("=" * 72)

# Tamamen duz seri: hicbir kurulum anlamli degil
n = 300
duz = pd.DataFrame({
    "Open": 100.0, "High": 100.0, "Low": 100.0, "Close": 100.0,
    "Volume": 1_000_000.0,
}, index=pd.bdate_range("2024-01-01", periods=n))
td = kv.tespit(duz)
tetik = {kid: int(td[(kid, "var")].sum()) for kid in kv.KAYIT}
check("duz seride hicbir kurulum tetiklenmiyor",
      sum(tetik.values()) == 0, str({k: v for k, v in tetik.items() if v}))

# Rastgele seride oranlar makul olmali (her gun tetiklenen dedektor ise yaramaz)
r = taban(1200, egim=0.0, seed=11)
tr = kv.tespit(r)
kullanilabilir = len(r) - kv.MIN_BAR
oranlar = {kid: float(tr[(kid, "var")].iloc[kv.MIN_BAR:].mean())
           for kid in kv.KAYIT}
cok_sik = {k: round(v, 3) for k, v in oranlar.items() if v > 0.25}
check("hicbir kurulum gunlerin %25'inden fazlasinda tetiklenmiyor",
      not cok_sik, str(cok_sik))
print("        tetiklenme oranlari: "
      + ", ".join(f"{k}={v:.3f}" for k, v in sorted(oranlar.items(),
                                                    key=lambda x: -x[1])[:5]))

print()
print("=" * 72)
print("6) KOSULLAR")
print("=" * 72)

ks = kv.kosullar(taban(300, seed=12))
check("uc kosul ekseni var",
      set(ks.columns) == {"oynaklik", "likidite", "trend_konumu"},
      str(list(ks.columns)))
check("oynaklik kovalari bilinen degerlerden",
      set(ks["oynaklik"].dropna().unique()) <= {"sakin", "orta", "oynak"},
      str(set(ks["oynaklik"].dropna().unique())))
check("likidite kovalari bilinen degerlerden",
      set(ks["likidite"].dropna().unique()) <= {"ince", "orta", "kalin"})
check("son barin kosullari dolu",
      all(pd.notna(ks[c].iloc[-1]) for c in ks.columns))

print()
print("=" * 72)
print("7) TARAMA ARAYUZU")
print("=" * 72)

check("kisa seride bugun() bos doner", kv.bugun(taban(50)) == [])
check("None girdide bugun() bos doner", kv.bugun(None) == [])

d = taban(seed=3)
i = len(d) - 1
zirve = float(d["High"].iloc[i - 20:i].max())
bar_yaz(d, i, zirve * 1.001, zirve * 1.05, zirve * 0.999, zirve * 1.04,
        v=float(d["Volume"].iloc[i - 20:i].median()) * 3.0)
sinyaller = kv.bugun(d, "TEST")
check("bugun() sinyal donduruyor", len(sinyaller) >= 1, f"{len(sinyaller)} sinyal")
if sinyaller:
    s = sinyaller[0]
    check("sinyalde gerekli alanlar var",
          {"ticker", "tarih", "kurulum", "yon", "guc", "oynaklik",
           "likidite", "trend_konumu", "fiyat"} <= set(s), str(sorted(s)))
    check("guc 0-1 araliginda", 0.0 <= s["guc"] <= 1.0, str(s["guc"]))

tablo = kv.tara({"A": {"history": d}, "B": {"history": taban(50)},
                 "C": {"history": None}, "D": None})
check("tara() yalnizca gecerli hisseleri isliyor",
      set(tablo["ticker"].unique()) <= {"A"} if len(tablo) else True,
      str(tablo["ticker"].unique().tolist() if len(tablo) else []))
check("tara() bos girdide bos tablo doner", len(kv.tara({})) == 0)

print()
print("=" * 72)
print("8) HER DEDEKTOR GERCEKTEN CALISIYOR MU")
print("=" * 72)

# 05.09.2026: bollinger_sikismasi, ind.bollinger()'in BES deger dondurmesine
# ragmen uc degere cozumleniyordu. tespit() istisnayi yutuyordu, dedektor
# aylarca olu kalabilirdi ve "hic sinyal uretmeyen dedektor" ile "bozuk
# dedektor" ayirt edilemiyordu. Artik hatalar KAYIT altinda birikiyor.
def gercekci(n: int = 900, seed: int = 21) -> pd.DataFrame:
    """Hacim dalgalanmasi ve bosluk iceren seri.

    taban() duz hacimli ve bosluksuz; hacme veya bosluga bakan dedektorler
    orada YAPISAL olarak tetiklenemez. Onlari "olu" saymak dedektoru degil
    fikstur'u olcmek olurdu.
    """
    r = np.random.default_rng(seed)
    getiri = r.normal(0.0015, 0.011, n)
    bosluk = (r.random(n) < 0.05) * r.normal(0, 0.045, n)   # ara sira bosluk
    kapanis = 40.0 * np.exp(np.cumsum(getiri))
    # ACILIS, ONCEKI kapanistan tureMELI. Bugunun kapanisindan turetilirse
    # bosluksuz gunlerde Open == Close olur, hicbir bar "yukselis mumu"
    # sayilmaz ve yutan mum dedektorleri yapisal olarak hic tetiklenemez.
    onceki = np.concatenate([[40.0], kapanis[:-1]])
    acilis = onceki * (1 - bosluk)
    yuksek = np.maximum(acilis, kapanis) * (1 + np.abs(r.normal(0, 0.006, n)))
    dusuk = np.minimum(acilis, kapanis) * (1 - np.abs(r.normal(0, 0.006, n)))
    # Hacim: log-normal + ara sira patlama ve kuraklik
    hacim = np.exp(r.normal(13.8, 0.5, n))
    hacim *= np.where(r.random(n) < 0.06, r.uniform(2.5, 5.0, n), 1.0)
    hacim *= np.where(r.random(n) < 0.10, r.uniform(0.25, 0.5, n), 1.0)
    return pd.DataFrame({"Open": acilis, "High": yuksek, "Low": dusuk,
                         "Close": kapanis, "Volume": hacim},
                        index=pd.bdate_range("2022-01-03", periods=n))


kv.HATALAR.clear()
uzun = gercekci()
tt = kv.tespit(uzun)
check("hicbir dedektor hata vermiyor", not kv.HATALAR, str(kv.HATALAR))

sayim = {kid: int(tt[(kid, "var")].sum()) for kid in kv.KAYIT}
olu = [kid for kid, n_ in sayim.items() if n_ == 0]
check("hicbir dedektor gercekci seride TAMAMEN olu degil", not olu, str(olu))
print("        tetik sayilari: "
      + ", ".join(f"{k}={v}" for k, v in sorted(sayim.items(),
                                                key=lambda x: -x[1])))

# Bozuk bir dedektor SESSIZ kalmamali
kv.HATALAR.clear()


def _bozuk(df):
    raise RuntimeError("bilerek bozuk")


kv.KAYIT["_bozuk"] = kv.Kurulum("_bozuk", "Bozuk", "long", _bozuk, "test" * 10)
try:
    kv.tespit(uzun)
    check("bozuk dedektor hata kaydina dusuyor", "_bozuk" in kv.HATALAR,
          str(kv.HATALAR))
    check("bozuk dedektor taramayi durdurmuyor",
          bool(kv.tespit(uzun)[("cekic", "var")].sum() >= 0))
finally:
    kv.KAYIT.pop("_bozuk", None)
    kv.HATALAR.clear()

print()
print("=" * 72)
print("9) SAYISAL OZELLIKLER")
print("=" * 72)

oz = kv.ozellikler(taban(300, seed=22))
bekle = {"atr_pct", "bb_genislik", "dolar_hacim", "ma200_uzaklik",
         "ma50_uzaklik", "rsi14", "hacim_orani", "getiri_5g", "getiri_20g",
         "govde_orani"}
check("beklenen ozellik sutunlari var", bekle <= set(oz.columns),
      str(sorted(set(oz.columns))))
check("son barda hepsi dolu",
      all(pd.notna(oz[c].iloc[-1]) for c in oz.columns),
      str([c for c in oz.columns if pd.isna(oz[c].iloc[-1])]))
check("RSI 0-100 araliginda",
      bool(oz["rsi14"].dropna().between(0, 100).all()))
check("ATR yuzdesi pozitif", bool((oz["atr_pct"].dropna() > 0).all()))

# kosullar(), ozellikler() uzerine oturmali -- iki ayri hesap olmamali
ks2 = kv.kosullar(taban(300, seed=22))
check("kosullar ile ozellikler ayni indekste",
      list(ks2.index) == list(oz.index))

print()
if fails:
    print(f"{fails} KONTROL BASARISIZ")
    raise SystemExit(1)
print("TUM KISA VADE TESTLERI GECTI")
