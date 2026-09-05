"""Benzer sirketlerden olusan TEST HAVUZU.

NEDEN HAVUZ
-----------
Kisa vadeli bir kenari olcerken en buyuk tehlike, olctugun seyin kurulum degil
SIRKET FARKI olmasi. 2 milyon dolar hacimli bir biyoteknoloji ile 50 milyon
dolar hacimli bir bankayi ayni tabloya koyup "bu kurulum %52 tutturuyor"
demek, iki tamamen farkli dunyanin ortalamasini almaktir. O ortalama ikisini
de tarif etmez.

Havuz bunu kesiyor: ayni sektorden, benzer buyuklukte, benzer likiditede ve
benzer oynaklikta sirketler. Boylece havuz icindeki fark, sirketten degil
kurulumdan geliyor olma sansina sahip olur.

YAPISAL NITELIK ile DURUM NITELIGI AYRIMI — buradaki en onemli karar
-------------------------------------------------------------------
Havuz YAPISAL niteliklerle kurulur: sektor, piyasa degeri, tipik likidite,
tipik oynaklik, fiyat seviyesi. Bunlar aylar boyunca yavas degisir.

Havuz DURUM nitelikleriyle KURULMAZ: "MA200 ustunde olanlar", "yukselen
trendde olanlar", "son 3 ayda kazandiranlar". Bunlar zamanla degisir ve
bugunku degerleri, olcecegimiz donemin sonucuyla kismen ayni seyden beslenir.
Trende gore hisse secmek, sessiz bir ileriye bakistir: "yukselen trendde olan
hisseler" kumesi, o trendin devam ettigi donemde secilmis olur.

Trend bir SECIM olcutu degil, bir KOSUL etiketidir. Kalibrasyon zaten her
sinyali kendi ortamiyla (oynaklik/likidite/trend konumu) kaydediyor.

AGIRLIKLAR
----------
Sektor bir agirlik degil, SART: sektor karisirsa olculen sey sektor
rotasyonu olur ve her seyi bastirir.

Kalan dort eksen agirlikli oklid uzakligiyla eslesiyor:

    log piyasa degeri   0.30   buyukluk, capraz kesitte getiri farkinin
                               sektorden sonraki en guclu belirleyicisi
    log dolar hacim     0.30   olculen kenarin ISLEM EDILEBILIR olup
                               olmadigini bu belirler; ince ve kalin
                               hisselerin mikroyapisi bambaska
    oynaklik (ATR%)     0.25   getirilerin BUYUKLUGU karsilastirilabilir
                               olsun; %1 oynayan ile %8 oynayanin ayni
                               tabloda ortalanmasi anlamsiz
    log fiyat           0.15   dusuk fiyatta tik boyutu ve makas, olculen
                               kenardan buyuk olabilir

Likiditeye buyuklukle ESIT agirlik verilmesi bilincli: bu sistemin kisa vade
tarafinin butun iddiasi "olculen kenar maliyeti asiyor mu" sorusu. Gunluk
500 bin dolarlik hisseyle 50 milyon dolarliği ayni havuza koyarsan o soruyu
soramazsin bile.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"
CIKTI = DATA / "havuz.json"

# Eslesme agirliklari (yukaridaki gerekce). Toplami 1.0 olmak zorunda degil;
# uzaklik hesabinda goreli buyuklukleri onemli.
AGIRLIK = {
    "log_mcap": 0.30,
    "log_dolar_hacim": 0.30,
    "atr_pct": 0.25,
    "log_fiyat": 0.15,
}

# Sert filtreler. Havuza girmek icin gereken asgari sartlar; bunlar
# "benzerlik" degil "olculebilirlik" sartlari.
MIN_FIYAT = 5.0            # tik boyutu / makas etkisi altinda olcum anlamsiz
MIN_DOLAR_HACIM = 1e6      # gunluk 1M$ altinda makas kenardan buyuk olur
MIN_BAR = 220              # gostergelerin oturmasi icin
MIN_HAVUZ = 12             # bundan kucuk kesitte capraz kesitsel olcum anlamsiz


# =============================================================================
#  Nitelikler
# =============================================================================
def nitelikler(bundles: dict, min_bar: int = MIN_BAR) -> pd.DataFrame:
    """Her hisse icin YAPISAL nitelikler.

    Hepsi uzun pencereli medyan/ortalama: gunluk dalgalanma degil, hissenin
    tipik hali olcuyor. Tek bir gunun degerine gore eslesme yapmak, havuzu
    o gunun gurultusune gore kurmak olurdu.
    """
    from . import indicators as ind

    satirlar = []
    for tk, b in sorted((bundles or {}).items()):
        h = (b or {}).get("history")
        if h is None or len(h) < min_bar:
            continue
        try:
            c = pd.to_numeric(h["Close"], errors="coerce").dropna()
            if len(c) < min_bar:
                continue
            v = pd.to_numeric(h["Volume"], errors="coerce").reindex(c.index)
            hi = pd.to_numeric(h["High"], errors="coerce").reindex(c.index)
            lo = pd.to_numeric(h["Low"], errors="coerce").reindex(c.index)

            fiyat = float(c.iloc[-1])
            dv = float((c * v).tail(120).median())
            atr = float((ind.atr(hi, lo, c, 14) / c).tail(120).median())
            info = (b or {}).get("info") or {}
            mcap = info.get("marketCap")
            if not mcap:
                sh = info.get("sharesOutstanding")
                mcap = float(sh) * fiyat if sh else None

            satirlar.append({
                "ticker": tk,
                "sektor": str(info.get("sector") or "BILINMIYOR"),
                "sanayi": str(info.get("industry") or ""),
                "fiyat": fiyat,
                "mcap": float(mcap) if mcap else np.nan,
                "dolar_hacim": dv,
                "atr_pct": atr,
                "bar": int(len(c)),
            })
        except Exception:
            continue

    df = pd.DataFrame(satirlar)
    if df.empty:
        return df
    df["log_mcap"] = np.log10(df["mcap"].where(df["mcap"] > 0))
    df["log_dolar_hacim"] = np.log10(df["dolar_hacim"].where(df["dolar_hacim"] > 0))
    df["log_fiyat"] = np.log10(df["fiyat"].where(df["fiyat"] > 0))
    return df


def _uygun(df: pd.DataFrame, min_fiyat: float, min_dv: float) -> pd.DataFrame:
    """Olculebilirlik sartlari. Benzerlikten ONCE gelir."""
    if df.empty:
        return df
    m = (
        (df["fiyat"] >= min_fiyat)
        & (df["dolar_hacim"] >= min_dv)
        & df["log_mcap"].notna()
        & df["atr_pct"].notna()
        & (df["atr_pct"] > 0)
        & (df["sektor"] != "BILINMIYOR")
    )
    return df[m].copy()


# =============================================================================
#  Esleme
# =============================================================================
def _saglam_z(s: pd.Series) -> pd.Series:
    """Medyan/MAD ile standartlastirma.

    Ortalama/std kullanilmiyor: piyasa degeri ve hacim dagilimlari agir
    kuyruklu, tek bir dev sirket ortalamayi ve std'yi ele geciriyor ve
    geri kalan herkes birbirine yapisik gorunuyor.
    """
    med = float(s.median())
    mad = float((s - med).abs().median())
    if not np.isfinite(mad) or mad < 1e-12:
        sd = float(s.std(ddof=0))
        return (s - med) / (sd if sd > 1e-12 else 1.0)
    return (s - med) / (1.4826 * mad)


def _uzakliklar(alt: pd.DataFrame, agirlik: dict) -> np.ndarray:
    """Agirlikli oklid uzaklik matrisi (standartlastirilmis eksenlerde)."""
    eksenler = [k for k in agirlik if k in alt.columns]
    Z = np.column_stack([_saglam_z(alt[k]).to_numpy() for k in eksenler])
    Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)
    w = np.array([agirlik[k] for k in eksenler], dtype=float)
    fark = Z[:, None, :] - Z[None, :, :]
    return np.sqrt(((fark ** 2) * w).sum(axis=2))


def _en_sik_kume(alt: pd.DataFrame, k: int, agirlik: dict) -> list[str]:
    """Sektor icinde birbirine EN YAKIN k hisseyi bulur.

    Yontem: her aday icin, kendisine en yakin k-1 komsuya olan uzakliklarin
    toplamina bak; en kucugu tohum olarak sec ve o komsulari havuz yap.

    Neden "en yakin k komsu" da "en yakin ortalamaya k hisse" degil: ortalama
    (centroid) hicbir sirket degildir ve dagilimin bos bir bolgesine dusebilir.
    O noktaya en yakin k hisse birbirine yakin OLMAYABILIR. Komsuluk toplami
    dogrudan "bu k tanesi birbirine ne kadar yakin" sorusunu olcer.
    """
    n = len(alt)
    if n < k:
        return []
    D = _uzakliklar(alt, agirlik)
    np.fill_diagonal(D, np.inf)
    sirali = np.sort(D, axis=1)[:, :k - 1]
    toplam = sirali.sum(axis=1)
    tohum = int(np.argmin(toplam))
    komsu = np.argsort(D[tohum])[:k - 1]
    idx = [tohum] + list(komsu)
    return alt.iloc[idx]["ticker"].tolist()


def _dagilim(alt: pd.DataFrame, eksen: str) -> dict:
    s = alt[eksen].dropna()
    if s.empty:
        return {}
    q1, q3 = float(s.quantile(0.25)), float(s.quantile(0.75))
    return {"medyan": round(float(s.median()), 4),
            "ceyrekler_arasi": round(q3 - q1, 4),
            "min": round(float(s.min()), 4), "max": round(float(s.max()), 4)}


def kur(bundles: dict, boyut: int = 25, en_fazla_sektor: int = 6,
        agirlik: dict | None = None, min_fiyat: float = MIN_FIYAT,
        min_dv: float = MIN_DOLAR_HACIM) -> dict:
    """Sektor basina bir havuz kurar ve homojenligini olcer.

    Homojenlik IDDIA EDILMEZ, OLCULUR: her eksende havuzun ceyrekler arasi
    genisligi, ayni sektorun ve tum evrenin genisligiyle birlikte raporlanir.
    Havuz gercekten dar degilse bu sayilar onu soyler.
    """
    agirlik = agirlik or AGIRLIK
    ham = nitelikler(bundles)
    if ham.empty:
        return {"ok": False, "reason": "nitelik cikarilamadi"}
    uygun = _uygun(ham, min_fiyat, min_dv)
    if uygun.empty:
        return {"ok": False, "reason": "sert filtreleri gecen hisse yok"}

    sayim = uygun["sektor"].value_counts()
    sektorler = [s for s, n in sayim.items() if n >= max(boyut, MIN_HAVUZ)]
    sektorler = sektorler[:en_fazla_sektor]

    havuzlar = []
    for sek in sektorler:
        alt = uygun[uygun["sektor"] == sek].reset_index(drop=True)
        uyeler = _en_sik_kume(alt, boyut, agirlik)
        if len(uyeler) < MIN_HAVUZ:
            continue
        hav = alt[alt["ticker"].isin(uyeler)]
        havuzlar.append({
            "id": _kimlik(sek),
            "sektor": sek,
            "boyut": len(uyeler),
            "uyeler": sorted(uyeler),
            "aday_havuzu": int(len(alt)),
            "dagilim": {e: {"havuz": _dagilim(hav, e),
                            "sektor": _dagilim(alt, e),
                            "evren": _dagilim(uygun, e)}
                        for e in ("log_mcap", "log_dolar_hacim", "atr_pct",
                                  "log_fiyat")},
            "daralma": _daralma(hav, alt, uygun),
        })

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ok": bool(havuzlar),
        "agirlik": agirlik,
        "sert_filtre": {"min_fiyat": min_fiyat, "min_dolar_hacim": min_dv,
                        "min_bar": MIN_BAR},
        "evren": int(len(ham)),
        "uygun": int(len(uygun)),
        "havuz_sayisi": len(havuzlar),
        "havuzlar": havuzlar,
        "notlar_tr": _notlar(havuzlar, ham, uygun),
    }


def _kimlik(sektor: str) -> str:
    return ("".join(ch.lower() if ch.isalnum() else "_" for ch in sektor)
            .strip("_")[:28])


def _daralma(hav: pd.DataFrame, sektor: pd.DataFrame,
             evren: pd.DataFrame) -> dict:
    """Havuzun ceyrekler arasi genisligi, evrenin kacta kaci?

    1'e yakin = havuz evrenden daha dar DEGIL, yani esleme ise yaramamis.
    Kucuk = havuz gercekten dar.
    """
    out = {}
    for e in ("log_mcap", "log_dolar_hacim", "atr_pct", "log_fiyat"):
        eh = _dagilim(hav, e).get("ceyrekler_arasi")
        ee = _dagilim(evren, e).get("ceyrekler_arasi")
        if eh is not None and ee:
            out[e] = round(eh / ee, 3)
    return out


def _notlar(havuzlar: list, ham: pd.DataFrame, uygun: pd.DataFrame) -> list[str]:
    out = [f"{len(ham)} hisseden {len(uygun)} tanesi olculebilirlik "
           f"filtrelerini gecti (fiyat, likidite, sektor bilgisi)."]
    if not havuzlar:
        out.append("Hicbir sektorde havuz kuracak kadar aday yok.")
        return out
    out.append(f"{len(havuzlar)} havuz kuruldu, her biri tek bir sektorden.")
    en_iyi = min(havuzlar,
                 key=lambda h: np.mean(list(h["daralma"].values()) or [1.0]))
    out.append(
        "En dar havuz: " + en_iyi["sektor"] + " — eksenlerin evrene gore "
        "ceyrekler arasi genisligi " +
        ", ".join(f"{k.replace('log_', '')} {v:.2f}x"
                  for k, v in en_iyi["daralma"].items()) + ".")
    gevsek = [h["sektor"] for h in havuzlar
              if np.mean(list(h["daralma"].values()) or [1.0]) > 0.5]
    if gevsek:
        out.append("Su havuzlar evrenin yarisindan dar DEGIL, yani esleme "
                   "beklendigi kadar sikmamis: " + ", ".join(gevsek) + ".")
    out.append("Havuzlar YAPISAL niteliklerle kuruldu (sektor, buyukluk, "
               "likidite, oynaklik, fiyat). Trend/momentum gibi ZAMANLA "
               "DEGISEN nitelikler bilerek kullanilmadi: onlara gore secim "
               "yapmak, olculecek donemin sonucuna gore secim yapmak olurdu.")
    return out


# =============================================================================
#  Kayit
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


def semboller(payload: dict | None = None, havuz_id: str | None = None
              ) -> list[str]:
    """Havuz(lar)daki sembol listesi. Veri cekimi bunu kullanir."""
    d = payload or yukle()
    if not d or not d.get("havuzlar"):
        return []
    out: list[str] = []
    for h in d["havuzlar"]:
        if havuz_id and h["id"] != havuz_id:
            continue
        out.extend(h["uyeler"])
    return sorted(dict.fromkeys(out))
