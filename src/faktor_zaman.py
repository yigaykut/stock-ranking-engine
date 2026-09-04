"""Parametrelerin ZAMAN icindeki gucu — "hangi agirlik gercekten ise yariyor".

NEDEN AYRI BIR MODUL
--------------------
data/faktor_ic.json her parametre icin TEK bir IC ortalamasi tutuyor. O sayi
uc soruyu birden gizliyor:

  1. Bu ortalama sifirdan gercekten farkli mi?  73 gunluk bir IC serisinin
     ortalamasi +0.026 cikabilir ve tamamen gurultu olabilir. Ortalamanin
     yaninda hata payi olmadan "bu parametre calisiyor" denemez.

  2. Gucu artiyor mu, azaliyor mu?  Ilk yarida +0.05, ikinci yarida -0.01 olan
     bir parametrenin ortalamasi +0.02 cikar ve iyi gorunur. Oysa o parametre
     artik calismiyordur.

  3. Hangi ortamda calisiyor?  Momentum'un dusus rejiminde coktugu literaturun
     en iyi belgelenmis bulgularindan biri. Rejimden bagimsiz tek ortalama, iki
     farkli dunyanin ortalamasidir ve ikisini de tarif etmez.

ORTUSEN ETIKET SORUNU — buradaki en onemli teknik nokta
-------------------------------------------------------
Etiket 21 islem gunu ileri getiri, anlik goruntuler ise ~3 gunde bir. Yani
ardisik yaklasik 7 IC olcumu AYNI gelecek donemin parcalarini paylasir. Bu
olcumler bagimsiz DEGIL. Siradan t = ort / (std/sqrt(n)) formulu bagimsizlik
varsayar ve burada t'yi ciddi sekilde SISIRIR -- 73 gozlem varmis gibi davranir,
oysa bagimsiz gozlem sayisi 10 civarindadir.

Duzeltme: Newey-West (Bartlett cekirdegi). Ortalamanin varyansina, ortusme
uzunlugu kadar gecikmeli otokovaryanslar da katilir. Sonuc t degeri neredeyse
her zaman daha KUCUKTUR; yani duzeltme sistemi daha ihtiyatli yapar, daha
iddiali degil.

Bu modul hicbir agirligi degistirmez. Olcum uretir; karar kullanicinindir.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"
CIKTI = DATA / "faktor_zaman.json"

# |t| bu esigi gecerse "gurultuden ayirt edilebilir" deriz. 2.0 kabaca %5
# anlamlilik. Tek bir esik butun kararlari vermez ama bir yerden baslamak sart.
T_ESIK = 2.0


# =============================================================================
#  Newey-West duzeltilmis t
# =============================================================================
def newey_west_t(x: "np.ndarray | list[float]", lag: int | None = None
                 ) -> tuple[float, float, int]:
    """Ortalamanin ortusme-duzeltilmis t degeri.

    Doner: (t, standart_hata, kullanilan_gecikme)

    lag=None ise Newey-West'in klasik kural-i kaidesi: 4*(T/100)^(2/9).
    Ortusme uzunlugunu BILIYORSAK (bkz. ortusme_gecikmesi) onu vermek daha
    dogrudur -- kural-i kaide veri uzunluguna bakar, veri URETIMINE degil.
    """
    a = np.asarray(x, dtype=float)
    a = a[np.isfinite(a)]
    n = a.size
    if n < 3:
        return float("nan"), float("nan"), 0

    if lag is None:
        lag = int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0)))
    lag = int(max(0, min(lag, n - 2)))

    d = a - a.mean()
    # gamma_0 = varyans; buyuk orneklem formulu (n ile bolunur)
    s = float((d * d).mean())
    for l in range(1, lag + 1):
        g = float((d[l:] * d[:-l]).mean())
        # Bartlett agirligi: uzak gecikmeler daha az sayar, boylece tahmin
        # pozitif tanimli kalir.
        s += 2.0 * (1.0 - l / (lag + 1.0)) * g

    if not np.isfinite(s) or s <= 0:
        return float("nan"), float("nan"), lag
    se = float(np.sqrt(s / n))
    if se < 1e-12:
        return float("nan"), se, lag
    return float(a.mean() / se), se, lag


def ortusme_gecikmesi(dates: "list[str]", horizon: int) -> int:
    """Kac ardisik olcum ayni gelecek donemi paylasiyor?

    Ornek: etiket 21 islem gunu, anlik goruntuler 3 gunde bir -> 7. Takvim
    gunu farkindan islem gunune kabaca 5/7 ile geciyoruz.
    """
    if len(dates) < 2:
        return 0
    ts = pd.to_datetime(pd.Series(sorted(set(dates))), errors="coerce").dropna()
    if len(ts) < 2:
        return 0
    takvim_adim = float(ts.diff().dt.days.dropna().median() or 1.0)
    islem_adim = max(1.0, takvim_adim * 5.0 / 7.0)
    return int(max(0, min(np.ceil(horizon / islem_adim) - 1, len(ts) - 2)))


# =============================================================================
#  Analiz
# =============================================================================
def _yarim_yarim(dates: "list[str]", ic: "list[float]") -> tuple[float, float]:
    """Serinin ilk ve ikinci yarisinin IC ortalamasi (zayiflama kontrolu)."""
    if len(ic) < 6:
        return float("nan"), float("nan")
    k = len(ic) // 2
    a, b = np.asarray(ic[:k], float), np.asarray(ic[k:], float)
    return float(np.nanmean(a)), float(np.nanmean(b))


def _rejime_gore(dates: "list[str]", ic: "list[float]",
                 rejim: "dict[str, str]") -> dict:
    grup: dict[str, list[float]] = {}
    for d, v in zip(dates, ic):
        lab = rejim.get(d)
        if lab:
            grup.setdefault(lab, []).append(float(v))
    out = {}
    for lab, vals in grup.items():
        if len(vals) >= 5:      # 5 gunden az kesitte ortalama anlamsiz
            out[lab] = {"ic": round(float(np.mean(vals)), 4), "n": len(vals)}
    return out


def _karar(t_nw: float, ic_mean: float, ilk: float, son: float) -> str:
    if not np.isfinite(t_nw):
        return "olculemedi"
    if abs(t_nw) < T_ESIK:
        return "gurultuden ayirt edilemiyor"
    yon = "pozitif" if ic_mean > 0 else "TERS YONDE"
    if np.isfinite(ilk) and np.isfinite(son) and ilk * son < 0:
        return f"{yon} ama yon yari yariya degisiyor — kararsiz"
    return f"{yon} ve gurultuden ayirt edilebilir"


def analyze(labeled: pd.DataFrame, factor_ids: "list[str]", label_col: str,
            horizon: int, weights: "dict[str, float]" | None = None,
            rejim: "dict[str, str]" | None = None,
            directions: "dict[str, str]" | None = None,
            source: str = "canli") -> dict:
    """Her parametre icin IC serisi + anlamlilik + zayiflama + rejim kirilimi.

    directions: {faktor_id: 'higher_better' | 'lower_better'}. ISARET DUZELTMESI
    icin sart. Gecmise donuk panelde IC, HAM deger uzerinden hesaplanir;
    'lower_better' bir parametrede (ornegin degerleme carpani) ham degerin
    dusugu iyidir, dolayisiyla ham IC dogal olarak NEGATIF cikar. Duzeltme
    yapilmazsa saglikli bir parametre "ters yonde calisiyor" diye raporlanir --
    tam da bakilarak agirlik degistirilecek tabloda en pahali yanlis okuma.
    Duzeltmeden sonra isaret her yerde ayni anlama gelir: pozitif = parametre
    SKORLANDIGI YONDE calisiyor.
    """
    from . import ml

    weights = weights or {}
    rejim = rejim or {}
    directions = directions or {}
    rows = []
    ilk_seri: list[str] = []

    for fid in factor_ids:
        ser = ml.factor_ic_series(labeled, fid, label_col)
        if not ser or len(ser["ic"]) < 5:
            continue
        dates, ic = ser["dates"], ser["ic"]
        # Yalnizca HAM sutunda duzeltme gerekir; score_* zaten yonu uygulanmis
        # haldedir (bkz. scoring.py, 'rank' yonteminde ters cevirme).
        ters = (str(ser["column"]).startswith("raw_")
                and directions.get(fid) == "lower_better")
        if ters:
            ic = [-v for v in ic]
        if not ilk_seri:
            ilk_seri = dates

        arr = np.asarray(ic, dtype=float)
        lag = ortusme_gecikmesi(dates, horizon)
        t_nw, se_nw, lag_used = newey_west_t(arr, lag)
        std = float(arr.std(ddof=1)) if arr.size > 1 else float("nan")
        t_naive = (float(arr.mean() / (std / np.sqrt(arr.size)))
                   if np.isfinite(std) and std > 1e-12 else float("nan"))
        ilk, son = _yarim_yarim(dates, ic)

        rows.append({
            "factor": fid,
            "column": ser["column"],
            "sign_flipped": bool(ters),
            "weight": round(float(weights.get(fid, 0.0)), 3),
            "periods": int(arr.size),
            "ic_mean": round(float(arr.mean()), 4),
            "ic_std": None if not np.isfinite(std) else round(std, 4),
            "icir": (None if not np.isfinite(std) or std < 1e-12
                     else round(float(arr.mean() / std), 3)),
            # Bagimsizlik varsayan t. Yalnizca duzeltmenin ne kadar fark
            # ettigini gostermek icin duruyor; karar t_nw ile verilir.
            "t_naive": None if not np.isfinite(t_naive) else round(t_naive, 2),
            "t_nw": None if not np.isfinite(t_nw) else round(t_nw, 2),
            "se_nw": None if not np.isfinite(se_nw) else round(se_nw, 5),
            "nw_lag": lag_used,
            "ic_ilk_yari": None if not np.isfinite(ilk) else round(ilk, 4),
            "ic_son_yari": None if not np.isfinite(son) else round(son, 4),
            "by_regime": _rejime_gore(dates, ic, rejim),
            "verdict_tr": _karar(t_nw, float(arr.mean()), ilk, son),
        })

    rows.sort(key=lambda r: abs(r["t_nw"] or 0.0), reverse=True)

    rejim_sayim: dict[str, int] = {}
    for d in ilk_seri:
        lab = rejim.get(d)
        if lab:
            rejim_sayim[lab] = rejim_sayim.get(lab, 0) + 1

    gecen = [r["factor"] for r in rows if abs(r["t_nw"] or 0.0) >= T_ESIK]

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "horizon": horizon,
        "source": source,
        "periods": len(ilk_seri),
        "first_date": ilk_seri[0] if ilk_seri else None,
        "last_date": ilk_seri[-1] if ilk_seri else None,
        "t_threshold": T_ESIK,
        "regime_counts": rejim_sayim,
        "passing": gecen,
        "factors": rows,
        "notes_tr": _notlar(rows, rejim_sayim, source),
    }
    return payload


def _notlar(rows: list, rejim_sayim: dict, source: str) -> list[str]:
    out = []
    if not rows:
        return ["Olcum yok."]

    gecen = [r for r in rows if abs(r["t_nw"] or 0.0) >= T_ESIK]
    out.append(
        f"{len(rows)} parametrenin {len(gecen)} tanesi |t|>={T_ESIK:.0f} esigini geciyor "
        "(ortusen etiketler icin Newey-West duzeltmesi uygulanmis halde)."
    )

    sisme = [r for r in rows
             if r["t_naive"] and r["t_nw"] and abs(r["t_naive"]) >= T_ESIK > abs(r["t_nw"])]
    if sisme:
        out.append(
            "Ortusme duzeltmesi olmasaydi su parametreler yanlislikla anlamli "
            "gorunecekti: " + ", ".join(r["factor"] for r in sisme) + "."
        )

    ters = [r for r in rows if abs(r["t_nw"] or 0.0) >= T_ESIK and (r["ic_mean"] or 0) < 0
            and (r["weight"] or 0) >= 3]
    if ters:
        out.append(
            "TERS YONDE ve agirligi 3'ten buyuk: "
            + ", ".join(f"{r['factor']} (agirlik {r['weight']})" for r in ters)
            + ". Bu parametreler siralamayi olcum yonunun tersine cekiyor."
        )

    kararsiz = [r for r in rows if r["ic_ilk_yari"] is not None
                and r["ic_son_yari"] is not None
                and r["ic_ilk_yari"] * r["ic_son_yari"] < 0
                and (r["weight"] or 0) >= 4]
    if kararsiz:
        out.append(
            "Ilk yaridan ikinci yariya YON DEGISTIREN yuksek agirlikli "
            "parametreler: " + ", ".join(r["factor"] for r in kararsiz)
            + ". Tek pencerelik ortalamalari yaniltici."
        )

    if rejim_sayim and "DUSUS" not in rejim_sayim:
        out.append(
            "Bu pencerede HIC dusus rejimi yok ("
            + ", ".join(f"{k}: {v}" for k, v in sorted(rejim_sayim.items()))
            + "). Yani buradaki hicbir sonuc, piyasa duserken ne olacagini "
            "soylemiyor — trend/momentum agirlikli bir sistemde en kritik soru bu."
        )

    if source == "panel":
        out.append(
            "Kaynak gecmise donuk panel: bugun kote olan hisselerden uretildi, "
            "hayatta kalma yanliligi tasir. Yon gostergesi olarak okunmali, "
            "kesin buyukluk olarak degil."
        )
    return out


def save(payload: dict, path: Path | None = None) -> Path:
    p = path or CIKTI
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    return p


def load(path: Path | None = None) -> dict | None:
    p = path or CIKTI
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


# =============================================================================
#  Konsol ciktisi
# =============================================================================
def print_table(payload: dict) -> None:
    rows = payload.get("factors") or []
    if not rows:
        print("Zaman analizi icin yeterli olcum yok.")
        return

    print("=" * 78)
    print("PARAMETRE GUCU — ZAMAN VE REJIM KIRILIMI")
    print("=" * 78)
    rc = payload.get("regime_counts") or {}
    print(f"  {payload.get('periods')} donem  "
          f"({payload.get('first_date')} -> {payload.get('last_date')}), "
          f"ufuk {payload.get('horizon')} gun, kaynak: {payload.get('source')}")
    if rc:
        print("  rejim dagilimi: " + ", ".join(f"{k} {v}" for k, v in sorted(rc.items())))
    print()
    print(f"  {'PARAMETRE':<26}{'IC':>8}{'t(ham)':>8}{'t(duz)':>8}"
          f"{'1.yari':>8}{'2.yari':>8}{'AGIRLIK':>9}")
    print("  " + "-" * 74)
    for r in rows:
        def f(v, w=8, d=4):
            return f"{v:>{w}.{d}f}" if isinstance(v, (int, float)) else f"{'-':>{w}}"
        print(f"  {r['factor'][:26]:<26}{f(r['ic_mean'])}{f(r['t_naive'], 8, 2)}"
              f"{f(r['t_nw'], 8, 2)}{f(r['ic_ilk_yari'])}{f(r['ic_son_yari'])}"
              f"{f(r['weight'], 9, 2)}")
    print()
    for n in payload.get("notes_tr") or []:
        print("  * " + n)
    print()
