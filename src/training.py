"""Egitim, degerlendirme ve KENDI KENDINI BESLEYEN geri besleme dongusu.

Dongu
-----
    1. TOPLA    gunluk tarama -> feature store              (run.py daily)
    2. ETIKETLE ufku dolan satirlara ileri getiri           (otomatik)
    3. EGIT     sizintisiz ileri yuruyuslu bolmelerle       (run.py train)
    4. DEGERLE  OOS: IC, ICIR, ilk-dilim getiri farki
    5. TERFI    sadece TABAN CIZGISINI ve esikleri gecerse  (run.py promote)
    6. UYGULA   model tahmini bir FAKTOR olarak skora girer
    7. IZLE     canli tahmin vs gerceklesen -> bozulma      (run.py ml-status)
       -> bozulma varsa 3'e don

Guvenlik freni
--------------
Modelin skora etkisi, OLCULEN OOS becerisiyle ORANTILIDIR. Kanit yoksa agirlik
sifirdir. Bu, "model kurdum, artik ona guveniyorum" hatasini yapisal olarak
imkansiz kilar.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from . import dataset as ds
from . import models as mz

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "data" / "models"
REGISTRY = MODEL_DIR / "registry.json"

# --- Terfi esikleri ----------------------------------------------------------
# Kantitatif finansta |IC| ~0.03 zayif-ama-kullanilabilir kabul edilir.
# ICIR (IC / IC'nin std'si) tutarliligi olcer; 0.3 alti gurultuden ayirt edilemez.
MIN_IC = 0.02
MIN_ICIR = 0.30
MIN_FOLDS = 3
MIN_BEAT_BASELINE = 0.005          # taban cizgisini en az bu kadar gecmeli


# =============================================================================
#  Degerlendirme
# =============================================================================
def evaluate_predictions(pred: np.ndarray, y: np.ndarray, dates: np.ndarray) -> dict:
    """Gun bazli IC ve ilk/son dilim getiri farki.

    Tek bir toplu korelasyon YANILTICIDIR: gunler arasi piyasa hareketi
    korelasyonu sisirir. Bu yuzden her gun ayri hesaplanip ortalanir —
    kantitatif finansin standart yaklasimi.
    """
    ics, spreads, tops = [], [], []
    for d in np.unique(dates):
        m = dates == d
        if m.sum() < 10:
            continue
        p, t = pred[m], y[m]
        ic = mz.spearman(p, t)
        if np.isfinite(ic):
            ics.append(ic)
        k = max(1, int(0.1 * len(p)))
        order = np.argsort(-p)
        top, bot = t[order[:k]], t[order[-k:]]
        spreads.append(float(top.mean() - bot.mean()))
        tops.append(float(top.mean()))

    if not ics:
        return {"days": 0, "ic_mean": None, "icir": None,
                "top_decile_spread": None, "hit_rate": None}

    arr = np.asarray(ics, dtype=float)
    sd = float(arr.std(ddof=1)) if len(arr) > 1 else float("nan")
    return {
        "days": len(arr),
        "ic_mean": round(float(arr.mean()), 5),
        "ic_std": None if not np.isfinite(sd) else round(sd, 5),
        "icir": (None if (not np.isfinite(sd) or sd < 1e-9)
                 else round(float(arr.mean()) / sd, 3)),
        "hit_rate": round(float((arr > 0).mean()), 3),
        "top_decile_spread": round(float(np.mean(spreads)), 5) if spreads else None,
        "top_decile_return": round(float(np.mean(tops)), 5) if tops else None,
    }


# =============================================================================
#  Ileri yuruyuslu egitim
# =============================================================================
def walk_forward(model_name: str = "ridge", horizon: int = 21, n_splits: int = 5,
                 embargo: int = 5, window: int = 10, use_cache: bool = True,
                 rank_features: bool = True, store: Path | None = None,
                 min_rows_per_date: int = 30, force: bool = False,
                 collect_predictions: bool = False,
                 panel: "ds.Panel | None" = None,
                 panel_info: dict | None = None) -> dict:
    """Bir modeli sizintisiz ileri yuruyusle egitir ve OOS degerlendirir.

    `store` verilirse egitim, canli feature store yerine o dizindeki panelden
    yapilir (gecmise donuk on egitim). Sonuc sozlugu bunu `store` alaninda
    tasir; terfi kararinda ayirt edilebilsin diye.

    `panel` verilirse yeniden KURULMAZ. Dort modeli tek komutta olcerken bu
    fark saatlerce: panel kurulumunun pahali kismi etiketleme ve etiketleme
    ONBELLEKTE OLMAYAN sembollerde aga cikiyor. Ayni paneli dort kez kurmak,
    ayni ag isini dort kez yapmak demekti. Panel modelden bagimsizdir --
    modele ozel tek is, dizi modelleri icin pencereye cevirmek, o da zaten
    asagida ayrica yapiliyor.
    """
    # `force`, CLI'daki --force ile ayni anlama gelir. Eskiden CLI kapiyi
    # aciyor ama burasi yine kapatiyordu; --force sessizce ise yaramiyordu.
    ready = ds.readiness(horizon, store=store)
    if not ready["ready_to_train"] and not force:
        return {"ok": False, "reason": "yetersiz veri", "readiness": ready}

    if panel is None:
        panel, info = ds.load_panel(horizon=horizon, use_cache=use_cache,
                                    store=store,
                                    min_rows_per_date=min_rows_per_date)
    else:
        info = dict(panel_info or {})
        info["reused"] = True
    if panel is None:
        return {"ok": False, "reason": info.get("reason", "panel kurulamadi"),
                "readiness": ready, "panel_info": info}

    X = ds.cross_sectional_rank(panel.X, panel.dates) if rank_features else panel.X
    y = ds.demean_by_date(panel.y, panel.dates)
    dates, tickers = panel.dates, panel.tickers

    if model_name not in mz.AVAILABLE:
        return {"ok": False, "reason": f"model yok: {model_name}",
                "available": sorted(mz.AVAILABLE)}

    cls = mz.AVAILABLE[model_name]
    needs_seq = getattr(cls, "needs_sequence", False)
    if needs_seq:
        seq_panel = ds.Panel(X=X, y=y, dates=dates, tickers=tickers,
                             feature_names=panel.feature_names, horizon=horizon)
        X, y, dates, tickers = ds.build_sequences(seq_panel, window=window)
        if len(y) == 0:
            return {"ok": False, "reason": "dizi kurulamadi (yetersiz gecmis)"}

    folds, fold_info = [], []
    predictions: list[dict] = []
    for tr, te, meta in ds.walk_forward_splits(dates, horizon, n_splits, embargo):
        model = cls(**({"window": window} if needs_seq else {}))
        try:
            trained = model.fit(X[tr], y[tr], dates[tr])
            # Capraz kesitsel dikkat modeli gun sinirlarini bilmek ZORUNDA:
            # test dilimi birden cok tarih icerir ve gunler birbirine
            # karisirsa model egitildiginden farkli bir sey gorur.
            pred = (model.predict(X[te], dates[te])
                    if getattr(cls, "needs_dates", False)
                    else model.predict(X[te]))
        except Exception as exc:                      # pragma: no cover
            fold_info.append({**meta, "error": f"{type(exc).__name__}: {exc}"})
            continue
        ev = evaluate_predictions(pred, y[te], dates[te])
        folds.append(ev)
        fold_info.append({**meta, **ev, "train": {k: v for k, v in trained.items()
                                                 if k != "history"}})
        if collect_predictions:
            # Topluluk (ensemble) icin: tahminler (tarih, sembol) anahtariyla
            # saklanir. Dizi modeli farkli bir satir kumesi uzerinde calisiyor
            # (pencere kadar gecmisi olmayan satirlar dusuyor), bu yuzden
            # indeksle degil ANAHTARLA hizalanmalari sart.
            predictions.append({
                "fold": len(folds) - 1,
                "keys": list(zip(dates[te].tolist(), tickers[te].tolist())),
                "pred": pred.astype(float).tolist(),
                "y": y[te].astype(float).tolist(),
            })

    if not folds:
        return {"ok": False, "reason": "hicbir katman kurulamadi (veri araligi dar)",
                "readiness": ready, "folds": fold_info}

    ics = [f["ic_mean"] for f in folds if f.get("ic_mean") is not None]
    icirs = [f["icir"] for f in folds if f.get("icir") is not None]
    spreads = [f["top_decile_spread"] for f in folds
               if f.get("top_decile_spread") is not None]

    summary = {
        "ok": True,
        "model": model_name,
        "horizon": horizon,
        "folds": len(folds),
        "ic_mean": round(float(np.mean(ics)), 5) if ics else None,
        "ic_std_across_folds": (round(float(np.std(ics, ddof=1)), 5)
                                if len(ics) > 1 else None),
        "icir": round(float(np.mean(icirs)), 3) if icirs else None,
        "top_decile_spread": round(float(np.mean(spreads)), 5) if spreads else None,
        "positive_folds": int(sum(1 for i in ics if i > 0)),
        "fold_detail": fold_info,
        "panel": info,
        "features": len(panel.feature_names),
        "rank_features": rank_features,
        "window": window if needs_seq else None,
        "store": None if store is None else str(store),
        "pretrain": store is not None,
    }
    if collect_predictions:
        summary["_predictions"] = predictions

    # Son modeli TUM veriyle yeniden egit (canli tahmin icin)
    final = cls(**({"window": window} if needs_seq else {}))
    try:
        final.fit(X, y, dates)
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        # On egitim modeli canli modelin uzerine YAZMAZ; ayri dosyada durur.
        tag = "_pretrain" if store is not None else ""
        path = MODEL_DIR / f"{model_name}_h{horizon}{tag}.pkl"
        final.save(path)
        summary["model_path"] = str(path)
        summary["feature_names"] = panel.feature_names
    except Exception as exc:                          # pragma: no cover
        summary["final_fit_error"] = f"{type(exc).__name__}: {exc}"

    return summary


# =============================================================================
#  Topluluk (ensemble)
# =============================================================================
def _to_rank(values: np.ndarray) -> np.ndarray:
    """Degerleri 0-1 yuzdelik sirasina cevirir."""
    n = len(values)
    if n < 2:
        return np.zeros(n, dtype=float)
    order = np.argsort(np.argsort(values))
    return order.astype(float) / (n - 1)


def ensemble(results: dict[str, dict]) -> dict:
    """Birden cok modelin tahminini harmanlar ve tek aday olarak degerlendirir.

    NEDEN: sistem bugune kadar SAMPIYON-HEPSINI-ALIR calisiyordu. Uc modelden
    biri secilip digerleri atiliyordu. Oysa birbirinden farkli hatalar yapan
    modellerin ortalamasi, tek tek hepsinden daha KARARLI olma egilimindedir;
    terfi kapisindaki asil zorluk da IC'nin buyuklugu degil TUTARLILIGI
    (ICIR) oldugu icin bu dogrudan ise yarar.

    NASIL: ham tahminler degil, gun ici YUZDELIK SIRALARI ortalanir. Ham
    tahminler harmanlanamaz -- ridge'in cikti olcegi ile sinir aginin olcegi
    farklidir, buyuk olcekli olan digerini ezerdi.

    HIZALAMA: dizi modeli, pencere kadar gecmisi olmayan satirlari dusurur;
    yani modeller ayni satir kumesi uzerinde calismaz. Bu yuzden harmanlama
    (tarih, sembol) anahtariyla yapilir ve yalnizca TUM modellerde bulunan
    satirlar kullanilir.
    """
    usable = {name: r for name, r in results.items()
              if r.get("ok") and r.get("_predictions")}
    if len(usable) < 2:
        return {"ok": False, "reason": "topluluk icin en az iki model gerekli",
                "models": sorted(usable)}

    names = sorted(usable)

    # KATMANLARI SIRAYLA ESLESTIRMEK YANLISTI (duzeltildi 04.09.2026).
    # Dizi modeli, pencere kadar gecmisi olmayan gunleri dusuyor; bu yuzden
    # ayni komutta ridge/mlp/attn 3 katman kurarken seq 2 katman kurabiliyor.
    # Eski kod katmanlari INDEKSLE esliyordu -- seq'in 0. katmani ridge'in
    # 0. katmaniyla eslesiyordu, oysa ikisi farkli zaman pencereleriydi.
    # 04.09'daki dort modelli olcumde bu "hicbir katmanda ortak satir yok"
    # diye patladi. Patlamasi sansti: pencereler KISMEN ortusseydi hata
    # sessiz kalir ve topluluk, farkli donemlerin tahminlerini harmanlayarak
    # anlamsiz ama makul gorunen bir sonuc uretirdi.
    #
    # Dogrusu: bir modelin katmanlari zaman olarak AYRIK, yani tahminleri tek
    # havuzda (tarih, sembol) anahtariyla toplanabilir ve havuzdaki her satir
    # o model icin ORNEKLEM DISIDIR. Once havuzlar kesistirilir, sonra ortak
    # gunler zaman sirasina gore dilimlenir. Dilimler yalnizca DAGILIM
    # (ICIR) hesabi icindir; hepsi ornek dISI oldugu icin bu mesru.
    havuz: dict[str, dict[tuple, float]] = {}
    truth: dict[tuple, float] = {}
    for name in names:
        acc: dict[tuple, float] = {}
        for blk in usable[name]["_predictions"]:
            ks = [tuple(k) for k in blk["keys"]]
            acc.update(dict(zip(ks, blk["pred"])))
            truth.update(dict(zip(ks, blk["y"])))
        havuz[name] = acc

    ortak = set.intersection(*(set(m) for m in havuz.values()))
    if len(ortak) < 30:
        return {"ok": False,
                "reason": ("modellerin ornek disi satirlari ortusmuyor "
                           f"({len(ortak)} ortak satir)"),
                "common": len(ortak),
                "per_model": {n: len(havuz[n]) for n in names}}

    keys = sorted(ortak)
    kdates = np.array([k[0] for k in keys])

    # Once TUM ortak satirlarda gun ici yuzdelik siralar. Siralama gun
    # icindedir; dilimleme sonradan yapilir ki dilim siniri siralamayi
    # degistirmesin.
    blended = np.zeros(len(keys), dtype=float)
    for name in names:
        vals = np.array([havuz[name][k] for k in keys], dtype=float)
        ranked = np.zeros(len(keys), dtype=float)
        for d in np.unique(kdates):
            m = kdates == d
            ranked[m] = _to_rank(vals[m])
        blended += ranked
    blended /= len(names)
    y_all = np.array([truth[k] for k in keys], dtype=float)

    gunler = np.unique(kdates)
    n_dilim = min(min(len(usable[n]["_predictions"]) for n in names), len(gunler))
    n_dilim = max(1, n_dilim)

    folds, fold_info = [], []
    for fi, parca in enumerate(np.array_split(gunler, n_dilim)):
        if not len(parca):
            continue
        m = np.isin(kdates, parca)
        if int(m.sum()) < 30:
            fold_info.append({"fold": fi, "skipped": "dilimde satir < 30",
                              "common": int(m.sum())})
            continue
        ev = evaluate_predictions(blended[m], y_all[m], kdates[m])
        folds.append(ev)
        fold_info.append({"fold": fi, "common": int(m.sum()),
                          "first_date": str(parca[0]), "last_date": str(parca[-1]),
                          **ev})

    if not folds:
        return {"ok": False, "reason": "hicbir dilimde yeterli ortak satir yok",
                "folds": fold_info}

    ics = [f["ic_mean"] for f in folds if f.get("ic_mean") is not None]
    icirs = [f["icir"] for f in folds if f.get("icir") is not None]
    spreads = [f["top_decile_spread"] for f in folds
               if f.get("top_decile_spread") is not None]
    any_res = usable[names[0]]

    return {
        "ok": True,
        "model": "topluluk(" + "+".join(names) + ")",
        "members": names,
        "horizon": any_res.get("horizon"),
        "folds": len(folds),
        "ic_mean": round(float(np.mean(ics)), 5) if ics else None,
        "icir": round(float(np.mean(icirs)), 3) if icirs else None,
        "top_decile_spread": round(float(np.mean(spreads)), 5) if spreads else None,
        "positive_folds": int(sum(1 for i in ics if i > 0)),
        "fold_detail": fold_info,
        "pretrain": any_res.get("pretrain", False),
        # Ozellik seti tum uyelerde aynidir (ayni panelden geliyorlar); canli
        # tahminde uyelerin ayni sutunlarla beslenmesi icin tasinir.
        "feature_names": any_res.get("feature_names"),
        "rank_features": any_res.get("rank_features", True),
        "window": next((usable[n].get("window") for n in names
                        if usable[n].get("window")), None),
        # Topluluk tek bir .pkl olarak saklanmaz: uyelerin kendi dosyalari
        # zaten diskte ve canli tahmin bunlarin yuzdelik ortalamasidir.
        "model_path": None,
    }


# =============================================================================
#  Terfi kapisi — guvenlik freni
# =============================================================================
def _registry() -> dict:
    if REGISTRY.exists():
        try:
            return json.loads(REGISTRY.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"champion": None, "history": [], "candidates": {}}


def _save_registry(reg: dict) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REGISTRY.write_text(json.dumps(reg, ensure_ascii=False, indent=2), encoding="utf-8")


def suggested_weight(icir: float | None, ic: float | None) -> float:
    """Olculen beceriden skor agirligi turetir.

    Kanit ne kadar guclu ise agirlik o kadar buyuk — ama ust sinir bilincli
    olarak dusuk (12). Model, insan tarafindan okunabilir 27 parametrenin
    yerini almaz; onlara EK bir gorus katar.
    """
    if icir is None or ic is None or ic <= MIN_IC:
        return 0.0
    scaled = (icir - MIN_ICIR) / 0.7          # ICIR 1.0 -> ~1.0
    return float(round(max(0.0, min(12.0, 12.0 * max(0.0, scaled))), 1))


def promotion_check(result: dict, baseline: dict | None = None) -> dict:
    """Bir modelin skorlamaya katilmaya hak kazanip kazanmadigina karar verir."""
    reasons: list[str] = []
    if not result.get("ok"):
        return {"promote": False, "reasons": [result.get("reason", "egitim basarisiz")]}

    # GECMISE DONUK PANEL SAMPIYON URETEMEZ.
    # backfill.py'nin urettigi panelde (a) kote disi kalmis hisseler yok
    # (hayatta kalma yanliligi), (b) temel veri sutunlari hic yok. Orada
    # olculen IC gercekte elde edilebilecegin uzerindedir. Bu panel mimari
    # secimi ve on egitim icindir; skorlamaya giris kararini veremez.
    if result.get("pretrain"):
        return {
            "promote": False,
            "reasons": ["gecmise donuk on egitim paneli — hayatta kalma "
                        "yanliligi tasir, sampiyon secimi yapilamaz"],
            "suggested_weight": 0.0,
            "metrics": {"ic": result.get("ic_mean"), "icir": result.get("icir"),
                        "folds": result.get("folds", 0),
                        "positive_folds": result.get("positive_folds", 0)},
        }

    ic, icir = result.get("ic_mean"), result.get("icir")
    folds = result.get("folds", 0)
    pos = result.get("positive_folds", 0)

    if folds < MIN_FOLDS:
        reasons.append(f"yetersiz katman: {folds} < {MIN_FOLDS}")
    if ic is None or ic < MIN_IC:
        reasons.append(f"IC dusuk: {ic} < {MIN_IC}")
    if icir is None or icir < MIN_ICIR:
        reasons.append(f"ICIR dusuk: {icir} < {MIN_ICIR}")
    if folds and pos / folds < 0.6:
        reasons.append(f"katmanlarin yalnizca {pos}/{folds}'inde pozitif")

    if baseline and baseline.get("ok") and baseline.get("ic_mean") is not None:
        if ic is not None and ic < baseline["ic_mean"] + MIN_BEAT_BASELINE:
            reasons.append(
                f"taban cizgisini gecemedi: {ic:.4f} vs ridge {baseline['ic_mean']:.4f} "
                f"(+{MIN_BEAT_BASELINE} gerekli)")

    ok = not reasons
    return {
        "promote": ok,
        "reasons": reasons if reasons else ["tum esikler saglandi"],
        "suggested_weight": suggested_weight(icir, ic) if ok else 0.0,
        "metrics": {"ic": ic, "icir": icir, "folds": folds, "positive_folds": pos},
    }


def promote(result: dict, decision: dict) -> dict:
    """Modeli sampiyon yapar ve kayit defterine yazar."""
    reg = _registry()
    entry = {
        "model": result["model"],
        "horizon": result["horizon"],
        "path": result.get("model_path"),
        "promoted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ic": result.get("ic_mean"),
        "icir": result.get("icir"),
        "folds": result.get("folds"),
        "weight": decision.get("suggested_weight", 0.0),
        "feature_names": result.get("feature_names"),
        "rank_features": result.get("rank_features", True),
        "window": result.get("window"),
        # Topluluk sampiyonda tek bir .pkl yoktur; uye adlari saklanir ve
        # canli tahmin uyeleri diskten tek tek yukler.
        "members": result.get("members"),
    }
    prev = reg.get("champion")
    if prev:
        reg.setdefault("history", []).append(prev)
    reg["champion"] = entry
    _save_registry(reg)
    return entry


def champion() -> dict | None:
    return _registry().get("champion")


def record_candidate(result: dict, decision: dict) -> None:
    """Terfi etmeyen adayi da kaydeder — bozulma/ilerleme takibi icin."""
    reg = _registry()
    reg.setdefault("candidates", {})[result.get("model", "?")] = {
        "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ic": result.get("ic_mean"),
        "icir": result.get("icir"),
        "folds": result.get("folds"),
        "promoted": decision.get("promote", False),
        "reasons": decision.get("reasons"),
    }
    _save_registry(reg)


# =============================================================================
#  Canli tahmin — skorlamaya baglanti
# =============================================================================
def predict_live(feature_rows: "Any", feature_names: list[str]) -> dict[str, float] | None:
    """Sampiyon model ile guncel tarama icin tahmin uretir.

    feature_rows: to_feature_matrix ciktisi (DataFrame)
    Doner: {ticker: skor} ya da sampiyon yoksa None.
    """
    champ = champion()
    if not champ:
        return None

    # Topluluk sampiyon ise: uyeler tek tek tahmin eder, sonuclar YUZDELIK
    # siraya cevrilip ortalanir. Egitimde harmanlama nasil yapildiysa canlida
    # da aynen oyle yapilmali; aksi halde olculen beceri ile uretilen tahmin
    # ayni seyin olcumu olmaz.
    if champ.get("members"):
        return _predict_live_ensemble(champ, feature_rows, feature_names)

    if not champ.get("path"):
        return None
    path = Path(champ["path"])
    if not path.exists():
        return None

    import pandas as pd

    need = champ.get("feature_names") or feature_names
    missing = [c for c in need if c not in feature_rows.columns]
    if missing:
        # Parametre seti degistiyse model gecersizdir — sessizce yanlis
        # tahmin uretmektense hic uretmemek dogrusudur.
        return None

    try:
        model = mz.BaseModel.load(path)
    except Exception:
        return None

    sub = feature_rows[need].copy()
    X = sub.to_numpy(dtype=np.float32)
    X = ds._fill_missing(X, need)
    dates = np.array(["live"] * len(X))
    if champ.get("rank_features", True):
        X = ds.cross_sectional_rank(X, dates)

    if getattr(model, "needs_sequence", False):
        # Dizi modeli tek gunluk veriyle calisamaz: gecmis pencere gerekir.
        # Pencere, feature store'daki onceki anlik goruntulerden kurulur.
        return _predict_live_sequence(model, champ, feature_rows, need)

    try:
        pred = model.predict(X)
    except Exception:
        return None

    tick = feature_rows["ticker"].astype(str).to_numpy()
    return _to_percentile(tick, pred)


def _predict_live_ensemble(champ: dict, feature_rows: "Any",
                           feature_names: list[str]) -> dict[str, float] | None:
    """Topluluk sampiyonun canli tahmini: uyelerin yuzdelik ortalamasi.

    Bir uye tahmin uretemezse (dosyasi silinmis, ozellik seti degismis)
    kalanlarla devam edilir. HICBIRI uretemezse None doner -- yanlis bir
    tahmin uretmektense hic uretmemek dogrusudur.
    """
    members = champ.get("members") or []
    horizon = champ.get("horizon", 21)
    parts: list[dict[str, float]] = []

    for name in members:
        path = MODEL_DIR / f"{name}_h{horizon}.pkl"
        if not path.exists():
            continue
        sub = dict(champ)
        sub.pop("members", None)
        sub["model"] = name
        sub["path"] = str(path)
        p = _predict_one(sub, feature_rows, feature_names)
        if p:
            parts.append(p)

    if not parts:
        return None

    common = set.intersection(*(set(p) for p in parts))
    if not common:
        return None
    return {t: round(sum(p[t] for p in parts) / len(parts), 4) for t in sorted(common)}


def _predict_one(champ: dict, feature_rows: "Any",
                 feature_names: list[str]) -> dict[str, float] | None:
    """Tek bir model dosyasiyla tahmin (topluluk uyeleri icin)."""
    path = Path(champ["path"])
    if not path.exists():
        return None
    need = champ.get("feature_names") or feature_names
    if [c for c in need if c not in feature_rows.columns]:
        return None
    try:
        model = mz.BaseModel.load(path)
    except Exception:
        return None

    X = ds._fill_missing(feature_rows[need].to_numpy(dtype=np.float32), need)
    dates = np.array(["live"] * len(X))
    if champ.get("rank_features", True):
        X = ds.cross_sectional_rank(X, dates)
    if getattr(model, "needs_sequence", False):
        return _predict_live_sequence(model, champ, feature_rows, need)
    try:
        pred = model.predict(X)
    except Exception:
        return None
    return _to_percentile(feature_rows["ticker"].astype(str).to_numpy(), pred)


def _to_percentile(tickers: "Any", pred: np.ndarray) -> dict[str, float]:
    """Ham tahmini capraz kesitsel yuzdelige cevirir (0-100).

    Modelin cikisi keyfi olcektedir; diger 28 parametre 0-100 oldugu icin
    agirliklandirma ancak ayni olcekte anlamli olur.
    """
    order = pred.argsort().argsort().astype(float)
    score = 100.0 * order / max(1, len(order) - 1)
    return {str(t): float(s) for t, s in zip(tickers, score)}


def _predict_live_sequence(model: "Any", champ: dict, feature_rows: "Any",
                           need: list[str]) -> dict[str, float] | None:
    """GRU icin canli tahmin — bugunun satirini gecmis goruntulerle birlestirir.

    Dizi modelinin tum anlami "son N gunde NASIL DEGISTI" bilgisidir; canlida
    calistirilabilmesi icin her sembolun gecmis ozellik satirlari gerekir.
    Bunlar feature store'da zaten var — her gunun taramasi oraya yaziliyor.

    Kurulum egitimdekiyle AYNI olmak zorunda: once gun ici capraz kesitsel
    yuzdelik, sonra pencereleme. Sirasi degisirse model, egitimde gordugunden
    baska bir sey gorur ve tahminleri sessizce anlamsizlasir.

    Gecmisi yetersiz semboller icin tahmin URETILMEZ (None doner ve o hisse
    parametreyi "eksik" olarak alir) — kisa pencereyi tekrarla doldurup
    uydurma bir dizi vermek, yanlis guven uretirdi.
    """
    import pandas as pd

    from . import ml as _ml

    window = int(champ.get("window") or 10)
    # Pencerenin en az bu kadari GERCEK gozlem olmali; gerisi ilk satirin
    # tekrariyla doldurulur (egitimdeki build_sequences ile ayni kural).
    min_len = max(3, window // 2)

    # Yalnizca gereken kadar dosya okunur. Bugunun dosyasi da yazilmis
    # olabilecegi icin bir fazlasi alinip tarihe gore eleniyor.
    hist = _ml.load_recent_snapshots(window)
    if hist.empty or "snapshot_date" not in hist.columns:
        return None
    if [c for c in need if c not in hist.columns]:
        return None

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    hist = hist[hist["snapshot_date"].astype(str) < today]
    if hist.empty:
        return None

    # Her sembolun en son window-1 goruntusu yeter
    hist = hist.sort_values("snapshot_date")
    hist = hist.groupby("ticker", group_keys=False).tail(window - 1)

    live = feature_rows.copy()
    live["snapshot_date"] = today

    cols = ["snapshot_date", "ticker"] + need
    panel = pd.concat([hist[cols], live[cols]], ignore_index=True)
    panel["snapshot_date"] = panel["snapshot_date"].astype(str)
    panel["ticker"] = panel["ticker"].astype(str)
    # Ayni gun ayni sembol iki kez olmasin (tarama gun icinde tekrarlanabilir)
    panel = panel.drop_duplicates(subset=["ticker", "snapshot_date"], keep="last")

    X = panel[need].to_numpy(dtype=np.float32)
    X = ds._fill_missing(X, need)
    dates = panel["snapshot_date"].to_numpy()
    tickers = panel["ticker"].to_numpy()
    if champ.get("rank_features", True):
        X = ds.cross_sectional_rank(X, dates)

    seq_panel = ds.Panel(X=X, y=np.zeros(len(X), dtype=np.float32), dates=dates,
                         tickers=tickers, feature_names=need)
    Xs, _, sdates, stick = ds.build_sequences(seq_panel, window=window,
                                              min_len=min_len)
    if len(Xs) == 0:
        return None

    mask = sdates == today
    if not mask.any():
        return None

    try:
        pred = model.predict(Xs[mask])
    except Exception:
        return None
    return _to_percentile(stick[mask], pred)
