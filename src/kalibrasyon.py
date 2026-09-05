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


def etkin_n(n_sinyal: int, n_gun: int, ufuk: int) -> float:
    """Bagimsiz sayilabilecek gozlem sayisi.

    Iki kirilim birden:
      - ayni gun icindeki sinyaller tek bir piyasa gunu yasar -> gun sayisi
      - ardisik gunlerin N gunluk sonuclari ortusur      -> gun / ufuk

    Alt sinir 1; ust sinir ham sinyal sayisi (etkin, hamdan buyuk olamaz).
    """
    if n_sinyal <= 0 or n_gun <= 0:
        return 0.0
    return float(max(1.0, min(float(n_sinyal), n_gun / max(1.0, float(ufuk)))))


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


# =============================================================================
#  Toplama
# =============================================================================
class Toplayici:
    """Hisse hisse gezerken sayimlari biriktirir.

    Bellekte tek seferde tum evrenin barlarini tutmak yerine akis halinde
    toplaniyor: 2800 hisse x 500 bar x 12 kurulum tek bir tabloya sigmaz.
    """

    def __init__(self, ufuklar: "tuple[int, ...]"):
        self.ufuklar = tuple(ufuklar)
        # anahtar: (kurulum, ufuk, kosul_adi, kosul_degeri)
        self._sayim: dict[tuple, dict] = {}
        # taban oran icin: tum bar-gun ciftleri
        self._taban: dict[int, dict] = {u: {"n": 0, "k": 0} for u in self.ufuklar}

    def _kova(self, anahtar: tuple) -> dict:
        d = self._sayim.get(anahtar)
        if d is None:
            d = {"n": 0, "k": 0, "gunler": set(), "getiriler": []}
            self._sayim[anahtar] = d
        return d

    def taban_ekle(self, ufuk: int, kazanc: pd.Series) -> None:
        g = kazanc.dropna()
        if not len(g):
            return
        self._taban[ufuk]["n"] += int(len(g))
        self._taban[ufuk]["k"] += int((g > 0).sum())

    def ekle(self, kurulum: str, ufuk: int, tarihler: pd.Index,
             kazanc: pd.Series, getiri: pd.Series,
             kosullar: "list[tuple[str, pd.Series]]") -> None:
        """Bir hissenin bir kurulumu icin sayimlari ekler."""
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
            d["gunler"].update(pd.Series(tt).dt.date.tolist())
            # Getiri dagilimi icin ornek tutulur; hepsini tutmak gereksiz.
            if len(d["getiriler"]) < 5000:
                d["getiriler"].extend(gg.tolist())

        _yaz((kurulum, ufuk, "*", "*"))
        for ad, seri in kosullar:
            s = seri[gecerli]
            for deger in pd.unique(s.dropna()):
                _yaz((kurulum, ufuk, ad, str(deger)), (s == deger).to_numpy())

    def taban_orani(self, ufuk: int) -> float:
        t = self._taban.get(ufuk) or {}
        return (t["k"] / t["n"]) if t.get("n") else 0.5

    def rapor(self) -> list[dict]:
        out = []
        for (kurulum, ufuk, kosul, deger), d in self._sayim.items():
            n, k = d["n"], d["k"]
            if n < MIN_HAM:
                continue
            taban = self.taban_orani(ufuk)
            ne = etkin_n(n, len(d["gunler"]), ufuk)
            lo, hi = wilson(int(round(k * ne / n)), int(round(ne)))
            p = buzulmus(k, n, taban)
            gt = np.asarray(d["getiriler"], dtype=float)
            out.append({
                "kurulum": kurulum,
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
        min_bar: int = 220, ilerleme: "callable | None" = None) -> dict:
    """Onbellekteki gunluk barlardan kalibrasyon uretir.

    `bench_close` verilirse kazanc "endeksten iyi" demektir; verilmezse
    "pozitif getiri". Endeksli olcum dogru olanidir: yukselen piyasada her
    kurulum iyi gorunur.
    """
    from . import kisa_vade as kv

    top = Toplayici(ufuklar)
    bench = None
    if bench_close is not None and len(bench_close):
        b = pd.to_numeric(bench_close, errors="coerce").dropna()
        idx = pd.DatetimeIndex(b.index)
        try:
            idx = idx.tz_localize(None) if idx.tz is None else idx.tz_convert(None)
        except (TypeError, AttributeError):
            pass
        bench = pd.Series(b.to_numpy(), index=idx.normalize())
        bench = bench[~bench.index.duplicated(keep="last")].sort_index()

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

            gunler = pd.DatetimeIndex(df.index)
            try:
                gunler = (gunler.tz_localize(None) if gunler.tz is None
                          else gunler.tz_convert(None)).normalize()
            except (TypeError, AttributeError):
                pass

            for ufuk in ufuklar:
                getiri = ileri_getiri(df["Close"], ufuk)
                if bench is not None:
                    bh = bench.reindex(gunler).to_numpy()
                    bser = pd.Series(bh, index=df.index)
                    bgetiri = bser.shift(-ufuk) / bser - 1.0
                    fazla = getiri - bgetiri
                else:
                    fazla = getiri
                kazanc = (fazla > 0).where(fazla.notna())

                top.taban_ekle(ufuk, kazanc)
                for kid in kv.KAYIT:
                    var = t[(kid, "var")].to_numpy()
                    if not var.any():
                        continue
                    top.ekle(kid, ufuk, df.index[var], kazanc[var],
                             fazla[var],
                             [(ad, s[var]) for ad, s in dilimler])
            islenen += 1
        except Exception:
            hatali += 1
            continue
        if ilerleme and (i + 1) % 200 == 0:
            ilerleme(i + 1, islenen)

    kovalar = top.rapor()
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "kaynak": "onbellek gunluk barlar",
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
def kaydet(payload: dict, path: Path | None = None) -> Path:
    p = path or CIKTI
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                 encoding="utf-8")
    return p


def yukle(path: Path | None = None) -> dict | None:
    p = path or CIKTI
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


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
