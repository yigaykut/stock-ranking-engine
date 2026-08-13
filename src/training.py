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
                 rank_features: bool = True) -> dict:
    """Bir modeli sizintisiz ileri yuruyusle egitir ve OOS degerlendirir."""
    ready = ds.readiness(horizon)
    if not ready["ready_to_train"]:
        return {"ok": False, "reason": "yetersiz veri", "readiness": ready}

    panel, info = ds.load_panel(horizon=horizon, use_cache=use_cache)
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
    for tr, te, meta in ds.walk_forward_splits(dates, horizon, n_splits, embargo):
        model = cls(**({"window": window} if needs_seq else {}))
        try:
            trained = model.fit(X[tr], y[tr], dates[tr])
            pred = model.predict(X[te])
        except Exception as exc:                      # pragma: no cover
            fold_info.append({**meta, "error": f"{type(exc).__name__}: {exc}"})
            continue
        ev = evaluate_predictions(pred, y[te], dates[te])
        folds.append(ev)
        fold_info.append({**meta, **ev, "train": {k: v for k, v in trained.items()
                                                 if k != "history"}})

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
    }

    # Son modeli TUM veriyle yeniden egit (canli tahmin icin)
    final = cls(**({"window": window} if needs_seq else {}))
    try:
        final.fit(X, y, dates)
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        path = MODEL_DIR / f"{model_name}_h{horizon}.pkl"
        final.save(path)
        summary["model_path"] = str(path)
        summary["feature_names"] = panel.feature_names
    except Exception as exc:                          # pragma: no cover
        summary["final_fit_error"] = f"{type(exc).__name__}: {exc}"

    return summary


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
    if not champ or not champ.get("path"):
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
        # Dizi modeli canlida tek gunluk veriyle calisamaz: gecmis pencere gerekir.
        # Bu surumde dizi modeli yalnizca degerlendirme icin kullanilir.
        return None

    try:
        pred = model.predict(X)
    except Exception:
        return None

    tick = feature_rows["ticker"].astype(str).to_numpy()
    # Capraz kesitsel yuzdelik (0-100) — diger parametrelerle ayni olcek
    order = pred.argsort().argsort().astype(float)
    score = 100.0 * order / max(1, len(order) - 1)
    return {t: float(s) for t, s in zip(tick, score)}
