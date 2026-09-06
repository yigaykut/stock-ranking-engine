"""Kisa vadeli kurulumlarin GUVEN degeri — iddia degil, olculmus frekans.

SORU
----
"Bu kurulum olustugunda ne olur?" Cevap bir tahmin degil, bir SAYIMDIR:
gecmiste bu kurulum kac kez olustu, kacinda onumuzdeki N gunde endeksten iyi
performans cikti.

NEDEN HAM ORAN YETMEZ — uc ayri tuzak
--------------------------------------
1. TABAN ORANI. Yukselen bir piyasada rastgele bir hisse gunu bile %52
   olasilikla endeksi gecebilir. %53 tutturan bir kurulum "iyi" degildir; TABAN
   ORANININ USTUNDE ne kadar oldugu onemlidir. Bu yuzden her kova, kendi
   ufkunun taban oraniyla birlikte raporlanir ve asil sayi `edge = p - taban`.

2. KUCUK ORNEKLEM. 7 gozlemde 5 kazanc %71 eder ve hicbir sey anlatmaz.
   Cozum iki katmanli: (a) Wilson araligi -- kucuk n'de dogru davranan
   guven araligi, normal yaklasimin aksine [0,1] disina tasmaz; (b) taban
   oranina dogru BUZME (empirical Bayes) -- n kucukken raporlanan olasilik
   tabana yaklasir, veri biriktikce ham orana yakinsar.

3. GOZLEMLER BAGIMSIZ DEGIL. Iki ayri sekilde:
   - Ayni gun 50 hissede ayni kurulum olusur; hepsi ayni piyasa gununu
     yasar. Bu 50 gozlem degil, kabaca 1 gozlemdir.
   - Etiket N gunluk ileri getiri, sinyaller gunluk. Ardisik N gunun
     sonuclari ayni gelecegi paylasir.
   Ikisi birlikte ETKIN ORNEKLEM BUYUKLUGUNU dusurur: n_etkin ~ farkli
   gun sayisi / ufuk. Aralik bu sayiyla hesaplanir. Sonuc her zaman daha
   GENIS bir araliktir; yani duzeltme sistemi daha ihtiyatli yapar.

BU DOSYA HICBIR SEY TAVSIYE ETMEZ. Frekans uretir. Karar kullanicinindir.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"
CIKTI = DATA / "kisa_vade_kalibrasyon.json"

# Buzme gucu: kova, taban oranina "kac sanal gozlem" kadar cekilir.
# 30 secildi cunku bir kovanin taban orandan ayrilmasi icin en az o
# mertebede gercek gozlem gormesini istiyoruz.
BUZME = 30.0

# Bir kovanin "olculdu" sayilmasi icin gereken en az ETKIN gozlem.
MIN_ETKIN = 8

# Kovaya girmek icin gereken en az ham gozlem (etkin sayidan once).
MIN_HAM = 25


# =============================================================================
#  Wilson araligi
# =============================================================================
def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Bir oran icin Wilson skor araligi.

    Normal yaklasim (p +- z*sqrt(p(1-p)/n)) kucuk n'de ve p uclara yakinken
    [0,1] disina tasar; 0/10 icin sifir genislikte bir aralik verir ki bu
    acikca yanlistir. Wilson ikisinde de dogru davranir.
    """
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1.0 + z * z / n
    merkez = (p + z * z / (2 * n)) / d
    yari = (z / d) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, merkez - yari), min(1.0, merkez + yari))


def etkin_n(n_sinyal: int, n_gun: int, ufuk: int,
            bar_gun: float = 1.0) -> float:
    """Bagimsiz sayilabilecek gozlem sayisi.

    Iki kirilim birden:
      - ayni gun icindeki sinyaller tek bir piyasa gunu yasar -> gun sayisi
      - ardisik gunlerin sonuclari ortusur -> gun / (ufkun GUN cinsinden hali)

    `bar_gun`: gunde kac bar. Saatlik veride ufuk BAR cinsindendir; 21 barlik
    bir ufuk 21 gun degil ~3 gun ortusme demektir. Gunluk veride bar_gun=1
    oldugu icin formul eskisiyle ayni kalir.

    Alt sinir 1; ust sinir ham sinyal sayisi (etkin, hamdan buyuk olamaz).
    """
    if n_sinyal <= 0 or n_gun <= 0:
        return 0.0
    ufuk_gun = float(ufuk) / max(1e-9, float(bar_gun))
    return float(max(1.0, min(float(n_sinyal), n_gun / max(1.0, ufuk_gun))))


def buzulmus(k: int, n: int, taban: float, guc: float = BUZME) -> float:
    """Taban orana dogru buzulmus olasilik (empirical Bayes).

    n buyudukce ham orana, kucukken tabana yaklasir. Boylece "7 gozlemde 5
    kazanc" %71 degil, tabana yakin bir sayi olarak raporlanir.
    """
    if n <= 0:
        return float(taban)
    return float((k + guc * taban) / (n + guc))


# =============================================================================
#  Sonuc etiketleme
# =============================================================================
def ileri_getiri(close: pd.Series, ufuk: int) -> pd.Series:
    """t barindan t+ufuk barina getiri. SON ufuk kadar bar NaN kalir.

    shift(-ufuk) burada ILERIYE BAKIS DEGIL, etiketin ta kendisidir: sonucu
    olcuyoruz. Kritik olan, DEDEKTORLERIN bunu gormemesi -- kisa_vade.py'de
    hicbir negatif shift yok ve testi var.
    """
    return close.shift(-ufuk) / close - 1.0


def _kosul_dilimleri(ks: pd.DataFrame) -> "list[tuple[str, pd.Series]]":
    """Kalibrasyonun kirilim eksenleri.

    TEK BOYUT: (oynaklik x likidite x trend) capraz carpimi 27 kova eder ve
    her biri bos kalir. Tek boyutlu dilimler daha az bilgi tasir ama
    OLCULEBILIR; olculemeyen kirilimin degeri yoktur.
    """
    out: list[tuple[str, pd.Series]] = []
    for sutun in ("oynaklik", "likidite", "trend_konumu"):
        if sutun in ks.columns:
            out.append((sutun, ks[sutun]))
    return out


def _zaman_indeks(idx) -> "pd.DatetimeIndex":
    """Dilimsiz zaman damgasi (gune INDIRGEMEDEN).

    Gun ici olcumde sart: saatlik bar ile saatlik endeksi gun bazinda
    hizalarsak bir gunun butun barlari ayni endeks degerini alir, endeks
    getirisi gun icinde SIFIR cikar ve "endeksten iyi" olcusu sessizce
    "yukari gitti"ye donusur.
    """
    d = pd.DatetimeIndex(idx)
    try:
        d = d.tz_localize(None) if d.tz is None else d.tz_convert(None)
    except (TypeError, AttributeError):
        pass
    return d


def _gunluk_indeks(idx) -> "pd.DatetimeIndex":
    """Dilimli/dilimsiz karisik indeksi, dilimsiz gune indirger.

    Yahoo gecmisi America/New_York dilimli gelir, endeks serisi ayri
    kurulur. Ikisi ayni forma sokulmazsa reindex sessizce hepsini NaN
    doldurur ve butun etiketler kaybolur.
    """
    d = pd.DatetimeIndex(idx)
    try:
        d = d.tz_localize(None) if d.tz is None else d.tz_convert(None)
    except (TypeError, AttributeError):
        pass
    return d.normalize()


def _bench_hazirla(bench_close, gun_bazli: bool = True) -> "pd.Series | None":
    """Endeks serisini hizalamaya hazirlar.

    gun_bazli=True  -> indeks gune indirgenir (gunluk olcum)
    gun_bazli=False -> zaman damgasi korunur (gun ici olcum)
    """
    if bench_close is None or not len(bench_close):
        return None
    b = pd.to_numeric(bench_close, errors="coerce").dropna()
    if b.empty:
        return None
    idx = _gunluk_indeks(b.index) if gun_bazli else _zaman_indeks(b.index)
    out = pd.Series(b.to_numpy(), index=idx)
    return out[~out.index.duplicated(keep="last")].sort_index()


# Which raw indicators get turned into within-peer-group ranks. Not all 78 --
# ranking every column would triple the panel for little gain, and these are
# the ones where "where does this stand against its peers today" is a question
# with an obvious meaning.
CAPRAZ_OZELLIKLER = (
    "g_rsi7", "g_rsi14", "g_roc5", "g_roc10", "g_roc20",
    "g_ma20_uzaklik", "g_ma50_uzaklik", "g_ma200_uzaklik",
    "g_atr14", "g_bb_pct", "g_bb_genislik",
    "g_hacim10", "g_hacim50", "g_yukari_hacim_pay",
    "g_adx", "g_di_fark",
    "g_tepe20_uzaklik", "g_dip20_uzaklik", "g_donchian20", "g_sikisma",
)


def capraz_kesit(uzun: pd.DataFrame, ufuklar) -> pd.DataFrame:
    """Peer-relative ranks, and the label with the peer move taken out.

    Two things happen here and they're the reason this pass exists at all.

    The ranks answer a question none of the existing features do. Everything
    else is absolute and per-stock -- "RSI is 32" -- and says nothing about
    whether that's unusual for this group today. `x_` columns are the
    within-(group, timestamp) percentile, which is the same idea the long-term
    scorer has used from the start (src/scoring.py) and the short-term side
    never had.

    The label loses the peer move. The old one is excess over SPY, but SPY is
    not what these stocks have in common -- their sector is. Subtracting the
    group mean at each timestamp strips out the market and the sector and
    leaves the idiosyncratic part, which is the only part a cross-sectional
    model could predict in the first place.

    Not lookahead: a rank at time t uses other stocks at time t, all of it
    known at t. What has to be backward-looking is the features being ranked,
    and that is locked down in tests/test_gosterge_seti.py.
    """
    if uzun.empty:
        return uzun
    g = uzun.groupby(["grup", "zaman"], sort=False)

    sut = [c for c in CAPRAZ_OZELLIKLER if c in uzun.columns]
    if sut:
        siralar = g[sut].rank(pct=True)
        siralar.columns = [f"x_{c[2:]}" if c.startswith("g_") else f"x_{c}"
                           for c in sut]
        uzun = pd.concat([uzun, siralar], axis=1)

    for u in ufuklar:
        ad = f"fazla_{u}g"
        if ad in uzun.columns:
            uzun[f"akran_{u}g"] = uzun[ad] - g[ad].transform("mean")
    return uzun


# =============================================================================
#  Toplama
# =============================================================================
class Toplayici:
    """Hisse hisse gezerken sayimlari biriktirir.

    Bellekte tek seferde tum evrenin barlarini tutmak yerine akis halinde
    toplaniyor: 2800 hisse x 500 bar x 12 kurulum tek bir tabloya sigmaz.
    """

    def __init__(self, ufuklar: "tuple[int, ...]", bar_gun: float = 1.0):
        self.ufuklar = tuple(ufuklar)
        self.bar_gun = float(bar_gun)
        # anahtar: (kurulum, ufuk, kosul_adi, kosul_degeri)
        self._sayim: dict[tuple, dict] = {}
        # taban oran icin: tum bar-gun ciftleri
        self._taban: dict[int, dict] = {u: {"n": 0, "k": 0} for u in self.ufuklar}

    def _kova(self, anahtar: tuple) -> dict:
        d = self._sayim.get(anahtar)
        if d is None:
            d = {"n": 0, "k": 0, "gunler": set(), "getiriler": [],
                 "yon": "long"}
            self._sayim[anahtar] = d
        return d

    def taban_ekle(self, ufuk: int, fazla: pd.Series) -> None:
        """Taban orani: rastgele bir bar-hisse, endeksi gecme orani.

        Kisa taraf icin tabani ayrica tutmuyoruz; P(fazla<0) = 1 - P(fazla>0)
        (tam sifirlar ihmal edilebilir) oldugu icin `taban_orani` yonu
        parametre aliyor.
        """
        g = fazla.dropna()
        if not len(g):
            return
        self._taban[ufuk]["n"] += int(len(g))
        self._taban[ufuk]["k"] += int((g > 0).sum())

    def ekle(self, kurulum: str, ufuk: int, tarihler: pd.Index,
             kazanc: pd.Series, getiri: pd.Series,
             kosullar: "list[tuple[str, pd.Series]]",
             yon: str = "long") -> None:
        """Bir hissenin bir kurulumu icin sayimlari ekler.

        `kazanc` YONE GORE tanimlanmis olarak gelir: uzun tarafta "endeksi
        gecti", kisa tarafta "endeksin ALTINDA kaldi". Ikisini de "endeksi
        gecti" diye saymak, cikis sinyallerini tersten okumak olurdu --
        dagitim gunu icin +%5 kenar, kurulumun CALISTIGINI degil
        CALISMADIGINI gosterirdi.
        """
        gecerli = kazanc.notna()
        if not gecerli.any():
            return
        kz = kazanc[gecerli]
        gt = getiri[gecerli]
        td = pd.DatetimeIndex(tarihler[gecerli])

        def _yaz(anahtar, maske=None):
            d = self._kova(anahtar)
            kk = kz if maske is None else kz[maske]
            gg = gt if maske is None else gt[maske]
            tt = td if maske is None else td[maske]
            if not len(kk):
                return
            d["n"] += int(len(kk))
            d["k"] += int(kk.sum())
            d["yon"] = yon
            d["gunler"].update(pd.Series(tt).dt.date.tolist())
            # Getiri dagilimi icin ornek tutulur; hepsini tutmak gereksiz.
            if len(d["getiriler"]) < 5000:
                d["getiriler"].extend(gg.tolist())

        _yaz((kurulum, ufuk, "*", "*"))
        for ad, seri in kosullar:
            s = seri[gecerli]
            for deger in pd.unique(s.dropna()):
                _yaz((kurulum, ufuk, ad, str(deger)), (s == deger).to_numpy())

    def taban_orani(self, ufuk: int, yon: str = "long") -> float:
        t = self._taban.get(ufuk) or {}
        p = (t["k"] / t["n"]) if t.get("n") else 0.5
        return p if yon != "short" else 1.0 - p

    def rapor(self) -> list[dict]:
        out = []
        for (kurulum, ufuk, kosul, deger), d in self._sayim.items():
            n, k = d["n"], d["k"]
            if n < MIN_HAM:
                continue
            taban = self.taban_orani(ufuk, d.get("yon", "long"))
            ne = etkin_n(n, len(d["gunler"]), ufuk, self.bar_gun)
            lo, hi = wilson(int(round(k * ne / n)), int(round(ne)))
            p = buzulmus(k, n, taban)
            gt = np.asarray(d["getiriler"], dtype=float)
            out.append({
                "kurulum": kurulum,
                "yon": d.get("yon", "long"),
                "ufuk": int(ufuk),
                "kosul": kosul,
                "deger": deger,
                "n": int(n),
                "n_gun": int(len(d["gunler"])),
                "n_etkin": round(ne, 1),
                "kazanc": int(k),
                "p_ham": round(k / n, 4),
                "p": round(p, 4),
                "taban": round(taban, 4),
                "edge": round(p - taban, 4),
                "alt": round(lo, 4),
                "ust": round(hi, 4),
                "medyan_getiri": round(float(np.median(gt)), 5) if gt.size else None,
                "durum": ("olculdu" if ne >= MIN_ETKIN else "az veri"),
            })
        out.sort(key=lambda r: (-abs(r["edge"]), -r["n_etkin"]))
        return out


# =============================================================================
#  Kalibrasyon kurulumu
# =============================================================================
def kur(bundles: dict, bench_close: "pd.Series | None" = None,
        ufuklar: "tuple[int, ...]" = (3, 5, 10),
        min_bar: int = 220, ilerleme: "callable | None" = None,
        frekans: str = "1d") -> dict:
    """Onbellekteki gunluk barlardan kalibrasyon uretir.

    `bench_close` verilirse kazanc "endeksten iyi" demektir; verilmezse
    "pozitif getiri". Endeksli olcum dogru olanidir: yukselen piyasada her
    kurulum iyi gorunur.
    """
    from . import kisa_vade as kv

    top = Toplayici(ufuklar, bar_gun=kv.bar_gun(frekans))
    gun_bazli = (frekans == "1d")
    bench = _bench_hazirla(bench_close, gun_bazli)

    islenen = hatali = 0
    for i, (tk, bundle) in enumerate(sorted((bundles or {}).items())):
        h = (bundle or {}).get("history")
        if h is None or len(h) < min_bar:
            continue
        try:
            df = h[["Open", "High", "Low", "Close", "Volume"]].dropna()
            if len(df) < min_bar:
                continue
            t = kv.tespit(df)
            ks = kv.kosullar(df)
            dilimler = _kosul_dilimleri(ks)

            gunler = _gunluk_indeks(df.index)          # kova/gun sayimi icin
            hiza = gunler if gun_bazli else _zaman_indeks(df.index)

            for ufuk in ufuklar:
                getiri = ileri_getiri(df["Close"], ufuk)
                if bench is not None:
                    bh = bench.reindex(hiza).to_numpy()
                    bser = pd.Series(bh, index=df.index)
                    bgetiri = bser.shift(-ufuk) / bser - 1.0
                    fazla = getiri - bgetiri
                else:
                    fazla = getiri
                top.taban_ekle(ufuk, fazla)
                uzun = (fazla > 0).where(fazla.notna())
                kisa = (fazla < 0).where(fazla.notna())
                for kid, kur_ in kv.KAYIT.items():
                    var = t[(kid, "var")].to_numpy()
                    if not var.any():
                        continue
                    kazanc = kisa if kur_.yon == "short" else uzun
                    top.ekle(kid, ufuk, df.index[var], kazanc[var],
                             fazla[var],
                             [(ad, s[var]) for ad, s in dilimler],
                             yon=kur_.yon)
            islenen += 1
        except Exception:
            hatali += 1
            continue
        if ilerleme and (i + 1) % 200 == 0:
            ilerleme(i + 1, islenen)

    kovalar = top.rapor()
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "frekans": frekans,
        "bar_gun": kv.bar_gun(frekans),
        "ufuk_birimi": "bar",
        "kaynak": f"onbellek {frekans} barlar",
        "kazanc_tanimi": ("endeksten iyi" if bench is not None
                          else "pozitif getiri (ENDEKSSIZ - zayif olcum)"),
        "hisse": islenen,
        "hatali": hatali,
        "ufuklar": list(ufuklar),
        "taban": {str(u): round(top.taban_orani(u), 4) for u in ufuklar},
        "min_ham": MIN_HAM,
        "min_etkin": MIN_ETKIN,
        "buzme": BUZME,
        "kovalar": kovalar,
        "notlar_tr": _notlar(kovalar, top, ufuklar),
    }


def _notlar(kovalar: list, top: "Toplayici", ufuklar) -> list[str]:
    out = []
    if not kovalar:
        return ["Hicbir kova olcum esigini gecmedi."]
    olculen = [k for k in kovalar if k["durum"] == "olculdu"]
    out.append(f"{len(kovalar)} kova sayildi, {len(olculen)} tanesi etkin "
               f"orneklem esigini ({MIN_ETKIN}) geciyor.")
    if olculen:
        pozitif = [k for k in olculen if k["alt"] > k["taban"]]
        if pozitif:
            out.append("Alt guven siniri taban oranin USTUNDE kalan kovalar: "
                       + ", ".join(f"{k['kurulum']}/{k['ufuk']}g"
                                   + ("" if k["kosul"] == "*"
                                      else f" [{k['kosul']}={k['deger']}]")
                                   for k in pozitif[:8]))
        else:
            out.append("Hicbir kovanin alt guven siniri taban oranin ustunde "
                       "degil. Yani olculen hicbir kurulum, 'rastgele bir gun' "
                       "olmaktan ayirt edilemiyor.")
    n_kova = len(kovalar)
    beklenen = n_kova * 0.025           # tek yonlu, %95 aralik
    gecen = [k for k in kovalar if k["durum"] == "olculdu"
             and k["alt"] > k["taban"]]
    out.append(
        f"COKLU TEST: {n_kova} kova %95 araligiyla test edildi. Hicbir kenar "
        f"olmasa bile sansa {beklenen:.0f} civari kova esigi gecerdi. "
        f"Gecen: {len(gecen)}. "
        + ("Bu, sanstan BEKLENENIN ALTINDA -- yani gecenler de kanit sayilmaz."
           if len(gecen) <= beklenen else
           "Gecen sayisi sans beklentisinin uzerinde, ama tek tek hangisinin "
           "gercek oldugu bu tabloyla soylenemez."))
    out.append("Guven degeri bir SAYIMDIR, tahmin degil: gecmiste bu kurulum "
               "olustugunda kac kez endeks gecildi. Gelecek icin garanti "
               "vermez.")
    out.append("Etkin orneklem, ham sayimdan cok daha kucuktur: ayni gun olusan "
               "sinyaller tek bir piyasa gunu yasar ve ardisik gunlerin "
               "sonuclari ortusur. Araliklar bu kucuk sayiyla hesaplandi.")
    return out


# =============================================================================
#  Okuma / yazma ve sorgulama
# =============================================================================
def _frekans_yolu(frekans: str) -> Path:
    return DATA / f"kisa_vade_kalibrasyon_{frekans}.json"


def kaydet(payload: dict, path: Path | None = None) -> Path:
    """Kalibrasyonu yazar: frekansa ozel arsiv + "en son kosan" kopyasi.

    Frekans basina ayri dosya SART. Gunluk ve saatlik kalibrasyon farkli
    seyler olcuyor -- dedektorlerin trend suzgeci bar-goreli, yani saatlikte
    ~6 haftalik, gunlukte ~10 aylik bir trendi tarif ediyor. Tek dosyada
    tutulsalardi ikinci kosu birincisini ezerdi; ayni hatayi faktor
    analizinde bir kez yaptik (bkz. faktor_zaman.save).
    """
    p = path or CIKTI
    p.parent.mkdir(parents=True, exist_ok=True)
    govde = json.dumps(payload, ensure_ascii=False, indent=1)
    p.write_text(govde, encoding="utf-8")
    if path is None and payload.get("frekans"):
        _frekans_yolu(payload["frekans"]).write_text(govde, encoding="utf-8")
    return p


def yukle(path: Path | None = None, frekans: str | None = None) -> dict | None:
    """frekans verilirse o frekansin arsivi, verilmezse en son kosan.

    ESKI DOSYA GERI UYUMU: frekans alani, gun ici destegi eklenince (05.09.2026)
    geldi. Ondan onceki kalibrasyonlarda alan yok ve hepsi GUNLUKTU. Arsiv
    dosyasi bulunamazsa kanonik dosyaya bakilir; oradaki kayit istenen
    frekanstaysa (ya da alansizsa ve "1d" isteniyorsa) o kullanilir.
    Bu olmadan, gun ici destegi eklendigi anda gunluk guven degerleri
    "bilinmiyor"a duserdi -- calisan bir seyi bozmak.
    """
    if path is not None:
        p = path
    elif frekans:
        p = _frekans_yolu(frekans)
        if not p.exists():
            kanonik = _oku(CIKTI)
            if kanonik and (kanonik.get("frekans") or "1d") == frekans:
                return kanonik
            return None
    else:
        p = CIKTI
    return _oku(p)


def _oku(p: Path) -> dict | None:
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def kayitli_frekanslar() -> list[str]:
    out = []
    for f in DATA.glob("kisa_vade_kalibrasyon_*.json"):
        ad = f.stem.replace("kisa_vade_kalibrasyon_", "")
        if ad:
            out.append(ad)
    return sorted(out)


def guven(kalib: dict | None, kurulum: str, ufuk: int,
          kosullar: "dict[str, str] | None" = None) -> dict:
    """Bir sinyal icin guven degeri.

    EN OZEL KOVADAN GENELE: once sinyalin kendi kosuluna ait kova aranir
    (ornegin oynaklik=oynak), bulunamazsa veya olcum esigini gecmiyorsa
    kurulumun geneline dusulur. Hicbiri yoksa "bilinmiyor" doner -- uydurma
    bir sayi yerine acikca bilinmedigini soylemek, bu sistemde kural.
    """
    bos = {"p": None, "edge": None, "alt": None, "ust": None, "n": 0,
           "n_etkin": 0.0, "durum": "bilinmiyor", "kova": None,
           "aciklama_tr": "Bu kurulum icin henuz olcum yok."}
    if not kalib or not kalib.get("kovalar"):
        return bos

    idx = {(k["kurulum"], k["ufuk"], k["kosul"], k["deger"]): k
           for k in kalib["kovalar"]}
    adaylar = []
    for ad, deger in (kosullar or {}).items():
        if deger is None:
            continue
        k = idx.get((kurulum, ufuk, ad, str(deger)))
        if k and k["durum"] == "olculdu":
            adaylar.append(k)
    genel = idx.get((kurulum, ufuk, "*", "*"))

    # En dar (en az etkin gozlemli degil, en COK edge tasiyan degil) --
    # en COK etkin gozlemli ozel kova secilir: ozel olmak tek basina yeterli
    # degil, guvenilir de olmali.
    sec = max(adaylar, key=lambda k: k["n_etkin"]) if adaylar else genel
    if not sec:
        return bos

    taban = sec["taban"]
    if sec["durum"] != "olculdu":
        return {**bos, "n": sec["n"], "n_etkin": sec["n_etkin"],
                "durum": "az veri", "kova": _kova_adi(sec),
                "aciklama_tr": (f"{sec['n']} sinyal sayildi ama etkin orneklem "
                                f"{sec['n_etkin']} — guven degeri verilemez.")}

    ustunde = sec["alt"] > taban
    return {
        "p": sec["p"], "edge": sec["edge"],
        "alt": sec["alt"], "ust": sec["ust"],
        "taban": taban, "n": sec["n"], "n_etkin": sec["n_etkin"],
        "medyan_getiri": sec.get("medyan_getiri"),
        "durum": "olculdu",
        "kova": _kova_adi(sec),
        "ayirt_edilebilir": bool(ustunde),
        "aciklama_tr": (
            f"Gecmiste bu kurulum {sec['n']} kez olustu ({sec['n_gun']} farkli "
            f"gun, etkin {sec['n_etkin']}). {sec['ufuk']} gun sonunda endeksi "
            f"gecme orani %{100 * sec['p']:.0f}; ayni donemde rastgele bir gunun "
            f"orani %{100 * taban:.0f}. "
            + ("Alt guven siniri taban oranin ustunde."
               if ustunde else
               "Aralik taban orani iceriyor, yani farki gurultuden ayirt "
               "edilemiyor.")
        ),
    }


def _kova_adi(k: dict) -> str:
    if k["kosul"] == "*":
        return f"{k['kurulum']} / {k['ufuk']}g / genel"
    return f"{k['kurulum']} / {k['ufuk']}g / {k['kosul']}={k['deger']}"


# =============================================================================
#  Meta-etiket paneli — ileride egitilecek model icin
# =============================================================================
PANEL = DATA / "kisa_vade_panel.csv"


def panel_yolu(frekans: str) -> Path:
    return DATA / f"kisa_vade_panel_{frekans}.csv"


def panel(bundles: dict, bench_close: "pd.Series | None" = None,
          ufuklar: "tuple[int, ...]" = (3, 5, 10), min_bar: int = 220,
          yol: Path | None = None,
          ilerleme: "callable | None" = None,
          frekans: str = "1d", gruplar: dict | None = None) -> dict:
    """Kurulum basina SATIR SATIR ozellik + sonuc tablosu.

    NEDEN AYRI BIR CIKTI
    --------------------
    kur() kova bazinda SAYIM uretir: "bu kurulum, oynak piyasada, 5 gunde
    %57". Bu, insanin okuyup karar verebilecegi bir ozet -- ve kovalar
    bilerek kaba tutuldu, cunku ince kirilim olculemiyor.

    Bir model ise kovaya ihtiyac duymaz; ham sayilardan kendi esiklerini
    ogrenir. Bu dosya ona o ham hali verir. Iki cikti ayni olcum
    disiplininden gelir (ayni dedektorler, ayni ozellikler, ayni etiket) --
    yani model, kalibrasyonun olctugunden BASKA bir dunyayi ogrenmez.

    NE ISE YARAR: "kurulum olustu" ile "kurulum ise yarayacak" ayri
    sorulardir. Ikincisini ogrenen model, literaturde meta-etiketleme diye
    geciyor: birincil sinyal kaliptan gelir, ikincil model o sinyalin
    tutup tutmayacagini kestirir. Bu dosya o ikinci modelin egitim kumesi.

    SIZINTI: ozellikler sinyal gunune kadar (dahil) hesaplanir, etiketler
    sinyal gununden SONRASINI olcer. Iki taraf hicbir yerde karismaz.

    Doner: ozet sozluk. Tablo diske yazilir (data/ gitignore altinda).
    """
    from . import kisa_vade as kv

    gun_bazli = (frekans == "1d")
    bench = _bench_hazirla(bench_close, gun_bazli)
    gruplar = gruplar or {}
    parcalar: list[pd.DataFrame] = []
    # Every bar of every stock, not just the signal bars -- a peer rank has to
    # be taken over the whole cross-section, and signal rows alone are one or
    # two names at any given timestamp.
    capraz_parcalar: list[pd.DataFrame] = []
    islenen = hatali = 0

    for i, (tk, bundle) in enumerate(sorted((bundles or {}).items())):
        h = (bundle or {}).get("history")
        if h is None or len(h) < min_bar:
            continue
        try:
            df = h[["Open", "High", "Low", "Close", "Volume"]].dropna()
            if len(df) < min_bar:
                continue
            t = kv.tespit(df)
            oz = kv.ozellikler(df)
            ks = kv.kosullar(df)
            # The full indicator sweep goes in as well, prefixed so it can't
            # collide with the bucket columns above. Those stay as they are
            # because the calibration splits on them and changing them would
            # silently move every bucket.
            try:
                from . import gosterge_seti as gsx

                genis = gsx.olustur(df).add_prefix("g_")
            except Exception:
                genis = None
            gunler = _gunluk_indeks(df.index)
            hiza = gunler if gun_bazli else _zaman_indeks(df.index)

            # Second label alongside the first: did the target come before
            # the stop. Kept as extra columns rather than replacing anything,
            # so both questions can be measured on the same rows.
            try:
                bariyer = bariyer_etiketleri(df, ufuklar)
            except Exception:
                bariyer = {}

            etiketler = {}
            for ufuk in ufuklar:
                getiri = ileri_getiri(df["Close"], ufuk)
                if bench is not None:
                    bser = pd.Series(bench.reindex(hiza).to_numpy(),
                                     index=df.index)
                    getiri = getiri - (bser.shift(-ufuk) / bser - 1.0)
                etiketler[ufuk] = getiri

            for kid in kv.KAYIT:
                var = t[(kid, "var")].to_numpy()
                if not var.any():
                    continue
                # Build the columns in a dict and make the frame once.
                # Adding ~110 columns one at a time to a DataFrame fragments
                # it badly, and this runs per setup per ticker.
                kol: dict = {
                    "ticker": tk,
                    "tarih": gunler[var],
                    "zaman": _zaman_indeks(df.index)[var],
                    "kurulum": kid,
                    "yon": kv.KAYIT[kid].yon,
                    "guc": t[(kid, "guc")].to_numpy()[var],
                }
                for sut in oz.columns:
                    kol[sut] = oz[sut].to_numpy()[var]
                if genis is not None:
                    for sut in genis.columns:
                        kol[sut] = genis[sut].to_numpy()[var]
                for sut in ks.columns:
                    kol[sut] = ks[sut].to_numpy()[var]
                for ufuk, g_ in etiketler.items():
                    fz = g_.to_numpy()[var]
                    kol[f"fazla_{ufuk}g"] = fz
                    kz = (fz > 0).astype(float)
                    kz[pd.isna(fz)] = np.nan
                    kol[f"kazanc_{ufuk}g"] = kz
                    for ad in ("bariyer", "bariyertam"):
                        b = bariyer.get((ad, ufuk))
                        if b is not None:
                            kol[f"{ad}_{ufuk}g"] = b.to_numpy()[var]
                alt = pd.DataFrame(kol)
                parcalar.append(alt)

            if gruplar and genis is not None and tk in gruplar:
                sut = [c for c in CAPRAZ_OZELLIKLER if c in genis.columns]
                cx = pd.DataFrame({"ticker": tk, "grup": gruplar[tk],
                                   "zaman": _zaman_indeks(df.index)})
                for c in sut:
                    cx[c] = genis[c].to_numpy()
                for ufuk, g_ in etiketler.items():
                    cx[f"fazla_{ufuk}g"] = g_.to_numpy()
                capraz_parcalar.append(cx)
            islenen += 1
        except Exception:
            hatali += 1
            continue
        if ilerleme and (i + 1) % 200 == 0:
            ilerleme(i + 1, islenen)

    if not parcalar:
        return {"ok": False, "reason": "hicbir kurulum bulunamadi"}

    tablo = pd.concat(parcalar, ignore_index=True)
    tablo.insert(3, "frekans", frekans)

    capraz_bilgi = None
    if capraz_parcalar:
        uzun = pd.concat(capraz_parcalar, ignore_index=True)
        uzun = capraz_kesit(uzun, ufuklar)
        tut = (["ticker", "zaman"]
               + [c for c in uzun.columns if c.startswith(("x_", "akran_"))])
        tablo = tablo.merge(uzun[tut], on=["ticker", "zaman"], how="left")
        capraz_bilgi = {
            "grup": int(uzun["grup"].nunique()),
            "hisse": int(uzun["ticker"].nunique()),
            "bar": int(len(uzun)),
            "sutun": [c for c in uzun.columns if c.startswith("x_")],
            "etiket": [c for c in uzun.columns if c.startswith("akran_")],
        }
        del uzun
    # FREKANS BASINA AYRI DOSYA. Ayni hatayi ufuk arsivlerinde ve
    # kalibrasyonda birer kez yaptik: tek dosyaya yazilan iki olcum,
    # ikincisi birincisini yok ediyor ve bunu hicbir sey soylemiyor.
    p = yol or panel_yolu(frekans)
    p.parent.mkdir(parents=True, exist_ok=True)
    tablo.to_csv(p, index=False)
    return {
        "ok": True,
        "yol": str(p),
        "frekans": frekans,
        "satir": int(len(tablo)),
        "hisse": islenen,
        "hatali": hatali,
        "kurulum": int(tablo["kurulum"].nunique()),
        "tarih_araligi": [str(tablo["tarih"].min())[:10],
                          str(tablo["tarih"].max())[:10]],
        "ozellikler": [c for c in tablo.columns
                       if c not in ("ticker", "tarih", "zaman", "kurulum",
                                    "yon", "frekans")
                       and not c.startswith(("fazla_", "kazanc_", "bariyer",
                                             "akran_"))],
        "capraz": capraz_bilgi,
        "etiketler": [c for c in tablo.columns
                      if c.startswith(("fazla_", "kazanc_", "bariyer",
                                       "akran_"))],
        "bariyer_oran": {
            f"{u}g": round(float(tablo[f"bariyer_{u}g"].notna().mean()), 4)
            for u in ufuklar if f"bariyer_{u}g" in tablo.columns},
        "etiketli_oran": {
            f"{u}g": round(float(tablo[f"kazanc_{u}g"].notna().mean()), 4)
            for u in ufuklar},
    }


# =============================================================================
#  Triple-barrier labels
# =============================================================================
def uc_bariyer(high: pd.Series, low: pd.Series, close: pd.Series,
               atr: pd.Series, ufuk: int, ust_k: float = 1.5,
               alt_k: float = 1.0, dikey_isaret: bool = False) -> pd.Series:
    """Which came first: the target, the stop, or the clock?

    The label everywhere else is "was the return positive after exactly N
    bars". That's a snapshot, and a noisy one — a signal can be right for four
    bars and still get measured on the fifth after it gave everything back.

    This asks the question a trade actually asks. From each bar, walk forward
    until price touches close*(1 + ust_k*ATR%) or close*(1 - alt_k*ATR%), or
    until N bars are gone. 1 if the upper came first, 0 if the lower did, NaN
    if neither did and the clock ran out — those rows carry no answer and
    shouldn't be counted as losses.

    Barriers scale with ATR rather than being fixed percentages, so a quiet
    stock and a volatile one are asked an equally hard question.

    The default is asymmetric on purpose: a wider target than stop is the shape
    most rules of this kind take, and it means the label isn't a coin flip by
    construction.

    dikey_isaret closes the hole that leaves. With it off, a short horizon only
    labels the rows where price moved decisively and fast, and a model scored
    on those rows is being asked an easier question than the one you face live
    — you can't know in advance whether a trade will resolve quickly. With it
    on, rows that ran out of clock are labelled by the sign of the return at
    the vertical barrier, so every row gets an answer and the selection
    disappears. If an edge only shows up with it off, the edge was selection.
    """
    h = high.to_numpy(float)
    l = low.to_numpy(float)
    c = close.to_numpy(float)
    a = atr.to_numpy(float)
    n = len(c)
    out = np.full(n, np.nan)

    for i in range(n - 1):
        if not np.isfinite(a[i]) or a[i] <= 0 or not np.isfinite(c[i]):
            continue
        ust = c[i] + ust_k * a[i]
        alt = c[i] - alt_k * a[i]
        son = min(i + ufuk, n - 1)
        for j in range(i + 1, son + 1):
            if h[j] >= ust:
                out[i] = 1.0
                break
            if l[j] <= alt:
                out[i] = 0.0
                break
        else:
            if dikey_isaret and son > i:
                out[i] = 1.0 if c[son] > c[i] else 0.0
    return pd.Series(out, index=close.index)


def bariyer_etiketleri(df: pd.DataFrame, ufuklar, ust_k: float = 1.5,
                       alt_k: float = 1.0) -> dict:
    """Barrier labels per horizon, in both variants.

    Returns {("bariyer", u): ..., ("bariyertam", u): ...}. The second one
    labels every row; the first leaves unresolved ones blank. Having both on
    the same rows is what makes the selection question answerable.
    """
    from . import indicators as ind

    atr = ind.atr(df["High"], df["Low"], df["Close"], 14)
    out = {}
    for u in ufuklar:
        out[("bariyer", u)] = uc_bariyer(df["High"], df["Low"], df["Close"],
                                         atr, u, ust_k, alt_k, False)
        out[("bariyertam", u)] = uc_bariyer(df["High"], df["Low"], df["Close"],
                                            atr, u, ust_k, alt_k, True)
    return out
