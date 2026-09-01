"""Topluluk (ensemble) harmanlamasinin dogrulugu.

Uc soru test edilir:

 1. HIZALAMA. Dizi modeli, pencere kadar gecmisi olmayan satirlari dusurur;
    yani modeller AYNI SATIR KUMESI uzerinde calismaz. Harmanlama indeksle
    yapilirsa farkli hisselerin tahminleri toplanir ve sonuc sessizce sacma
    olur. Anahtarla (tarih, sembol) hizalandigi dogrulanir.

 2. OLCEK BAGIMSIZLIGI. Ham tahminler harmanlanamaz: ridge'in cikti olcegi
    ile sinir aginin olcegi farklidir, buyuk olcekli olan digerini ezer.
    Bir modelin tahminleri 1000 ile carpildiginda sonucun DEGISMEMESI
    gerekir -- cunku yalnizca siralamalar harmanlanir.

 3. FAYDA. Birbirinden bagimsiz hata yapan iki zayif tahminci harmanlandiginda
    sonuc, tek tek her ikisinden de iyi olmali.

Calistir:  python tests/test_topluluk.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import training as tr        # noqa: E402

fails = 0


def check(name, cond, detail=""):
    global fails
    if cond:
        print(f"  PASS  {name}" + (f"  {detail}" if detail else ""))
    else:
        print(f"  FAIL  {name}  {detail}")
        fails += 1


def make_result(keys, pred, y, ok=True):
    """walk_forward ciktisinin topluluk icin gereken en kucuk hali."""
    return {"ok": ok, "horizon": 21, "rank_features": True,
            "feature_names": ["a", "b"], "window": None, "pretrain": False,
            "_predictions": [{"fold": 0, "keys": keys,
                              "pred": list(map(float, pred)),
                              "y": list(map(float, y))}]}


rng = np.random.default_rng(11)
DATES = ["2026-01-05", "2026-01-06"]
TICKS = [f"T{i:03d}" for i in range(60)]
KEYS = [(d, t) for d in DATES for t in TICKS]

# Gercek sinyal + iki BAGIMSIZ gurultu -> iki zayif tahminci
truth = rng.normal(0, 1, len(KEYS))
noise_a = rng.normal(0, 1.6, len(KEYS))
noise_b = rng.normal(0, 1.6, len(KEYS))
pred_a = truth + noise_a
pred_b = truth + noise_b

print("=" * 70)
print("TOPLULUK HARMANLAMASI")
print("=" * 70)

res = {"a": make_result(KEYS, pred_a, truth),
       "b": make_result(KEYS, pred_b, truth)}
ens = tr.ensemble(res)
check("topluluk kuruluyor", ens.get("ok"), str(ens.get("reason")))
check("uyeler kaydediliyor", ens.get("members") == ["a", "b"], str(ens.get("members")))

ic_a = tr.evaluate_predictions(np.array(pred_a), truth,
                               np.array([k[0] for k in KEYS]))["ic_mean"]
ic_b = tr.evaluate_predictions(np.array(pred_b), truth,
                               np.array([k[0] for k in KEYS]))["ic_mean"]
check("topluluk IC her iki uyeden de iyi",
      ens["ic_mean"] > max(ic_a, ic_b),
      f"topluluk {ens['ic_mean']:.4f} vs a {ic_a:.4f} / b {ic_b:.4f}")

# --- Olcek bagimsizligi
res_scaled = {"a": make_result(KEYS, pred_a * 1000.0, truth),
              "b": make_result(KEYS, pred_b, truth)}
ens_scaled = tr.ensemble(res_scaled)
check("olcek degisimi sonucu degistirmiyor",
      abs(ens_scaled["ic_mean"] - ens["ic_mean"]) < 1e-9,
      f"{ens_scaled['ic_mean']} vs {ens['ic_mean']}")

# --- Hizalama: b modeli satirlarin yarisini goremiyor (dizi modeli gibi)
half = KEYS[::2]
res_partial = {"a": make_result(KEYS, pred_a, truth),
               "b": make_result(half, pred_b[::2], truth[::2])}
ens_partial = tr.ensemble(res_partial)
check("eksik satirli uye ile topluluk kuruluyor", ens_partial.get("ok"),
      str(ens_partial.get("reason")))
check("yalnizca ORTAK satirlar kullaniliyor",
      ens_partial["fold_detail"][0]["common"] == len(half),
      f"{ens_partial['fold_detail'][0]['common']} vs {len(half)}")

# Karistirilmis anahtar sirasi sonucu DEGISTIRMEMELI (indeksle hizalama olsaydi
# degisirdi -- bu testin asil yakalamak istedigi hata budur).
order = rng.permutation(len(KEYS))
res_shuffled = {
    "a": make_result([KEYS[i] for i in order], pred_a[order], truth[order]),
    "b": make_result(KEYS, pred_b, truth),
}
ens_shuffled = tr.ensemble(res_shuffled)
check("satir sirasi sonucu degistirmiyor (anahtarla hizalama)",
      abs(ens_shuffled["ic_mean"] - ens["ic_mean"]) < 1e-9,
      f"{ens_shuffled['ic_mean']} vs {ens['ic_mean']}")

# --- Tek model ile topluluk kurulamaz
one = tr.ensemble({"a": make_result(KEYS, pred_a, truth)})
check("tek modelle topluluk kurulmaz", not one.get("ok"), str(one.get("reason")))

# --- Basarisiz model disarida kalir
res_bad = {"a": make_result(KEYS, pred_a, truth),
           "b": {"ok": False, "reason": "yetersiz veri"}}
bad = tr.ensemble(res_bad)
check("basarisiz model topluluga alinmaz", not bad.get("ok"), str(bad.get("reason")))

# --- Ortak satir cok azsa katman atlanir
tiny = KEYS[:10]
res_tiny = {"a": make_result(KEYS, pred_a, truth),
            "b": make_result(tiny, pred_b[:10], truth[:10])}
ens_tiny = tr.ensemble(res_tiny)
check("ortak satir 30'un altindaysa katman atlanir",
      not ens_tiny.get("ok"), str(ens_tiny.get("reason")))

# --- Topluluk on egitim bayragini tasiyor mu (terfi freni icin sart)
res_pre = {"a": {**make_result(KEYS, pred_a, truth), "pretrain": True},
           "b": {**make_result(KEYS, pred_b, truth), "pretrain": True}}
ens_pre = tr.ensemble(res_pre)
check("on egitim bayragi topluluga tasiniyor", ens_pre.get("pretrain") is True)
dec = tr.promotion_check(ens_pre)
check("on egitim toplulugu TERFI EDEMEZ", dec["promote"] is False,
      "; ".join(dec["reasons"]))

print()
if fails:
    print(f"{fails} KONTROL BASARISIZ")
    raise SystemExit(1)
print("TUM TOPLULUK TESTLERI GECTI")
