"""Skorlama motorunun cekirdek davranislari.

Calistir:  python -m pytest tests/ -q      (pytest varsa)
       veya python tests/test_scoring.py   (pytest yoksa)
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.scoring import Scorer  # noqa: E402


def _cfg(clusters=None, penalty_sigma=None, **meta) -> dict:
    base_meta = {"auto_disable_coverage_below": 0.15,
                 "min_coverage_for_confidence": 0.60,
                 "sector_neutralize": False}
    base_meta.update(meta)
    pen = {"id": "bad", "name_tr": "Kotu"}
    if penalty_sigma is None:
        pen["points"] = -10
    else:
        pen["sigma"] = penalty_sigma
    cfg = {
        "meta": base_meta,
        "factors": [
            {"id": "a", "name_tr": "A", "category": "momentum", "weight": 60.0,
             "direction": "higher_better", "norm": "rank"},
            {"id": "b", "name_tr": "B", "category": "quality", "weight": 30.0,
             "direction": "higher_better", "norm": "rank"},
            {"id": "c", "name_tr": "C (seyrek)", "category": "sentiment", "weight": 10.0,
             "direction": "higher_better", "norm": "rank"},
        ],
        "penalties": [pen],
    }
    if clusters:
        cfg["clusters"] = clusters
    return cfg


def _rec(tk: str, a=None, b=None, c=None, penalty=False, sector="Tech") -> dict:
    return {
        "ticker": tk, "ok": True, "name": tk, "sector": sector, "industry": "x",
        "price": 100.0, "currency": "USD", "market_cap": 1e9,
        "avg_dollar_volume": 1e7, "rsi14": 50.0, "days_to_earnings": None,
        "returns": {"1m": 0.0, "3m": 0.0, "6m": 0.0, "12m": 0.0},
        "raw": {"a": a, "b": b, "c": c},
        "meta": {},
        "penalty_flags": {"bad": penalty},
        "snapshot_date": "2026-08-11",
    }


# ---------------------------------------------------------------------------
def test_ranking_is_descending():
    recs = [_rec(f"T{i}", a=float(i), b=float(i)) for i in range(10)]
    out, _ = Scorer(_cfg()).score(recs)
    scores = out["total_score"].tolist()
    assert scores == sorted(scores, reverse=True), "siralama azalan olmali"
    assert out.iloc[0]["ticker"] == "T9", "en yuksek ham deger basta olmali"
    assert list(out["rank"]) == list(range(1, 11))


def test_missing_factor_weight_is_redistributed():
    """Eksik veri CEZA OLMAMALI: agirligi diger faktorlere dagitilmali."""
    recs = [_rec(f"T{i}", a=float(i), b=float(i), c=float(i)) for i in range(10)]
    # T9 icin 'c' eksik ama a ve b'de hala en iyi -> yine 1. olmali
    recs[9]["raw"]["c"] = None

    out, _ = Scorer(_cfg()).score(recs)
    top = out.iloc[0]
    assert top["ticker"] == "T9", "eksik veri hisseyi geriye dusurmemeli"
    assert top["coverage"] < 1.0, "kapsama eksigi raporlanmali"
    # a ve b'de %100 persentilde oldugu icin skoru hala tepe seviyede
    assert top["total_score"] > 90, f"beklenmedik dusus: {top['total_score']}"


def test_sparse_factor_is_auto_disabled():
    """Kapsama %15 altindaysa faktor devre disi kalmali (WSB senaryosu)."""
    recs = [_rec(f"T{i}", a=float(i), b=float(i)) for i in range(20)]
    recs[0]["raw"]["c"] = 5.0            # 20'de 1 = %5 kapsama

    out, diag = Scorer(_cfg()).score(recs)
    disabled_ids = [d["id"] for d in diag["auto_disabled"]]
    assert "c" in disabled_ids, "seyrek faktor devre disi birakilmaliydi"
    assert "c" not in diag["weights_applied"]

    # kalan agirliklar 100'e normalize edilmeli
    # (weights_applied 3 haneye yuvarlanmis raporlama degeridir -> tolerans gevsek)
    assert abs(sum(diag["weights_applied"].values()) - 100.0) < 1e-3
    # a:b orani 60:30 = 2:1 korunmali
    w = diag["weights_applied"]
    assert abs(w["a"] / w["b"] - 2.0) < 1e-3
    assert out.iloc[0]["coverage"] == 1.0, "devre disi faktor kapsamayi dusurmemeli"


def test_manual_disable_and_weight_override():
    recs = [_rec(f"T{i}", a=float(i), b=float(9 - i), c=float(i)) for i in range(10)]

    out_a, diag_a = Scorer(_cfg(), disabled={"b"}).score(recs)
    assert "b" not in diag_a["weights_applied"], "elle devre disi birakma calismali"
    assert out_a.iloc[0]["ticker"] == "T9"

    # b'yi cok agirlastirinca sira tersine donmeli (b, a'nin tersi)
    out_b, diag_b = Scorer(_cfg(), weight_overrides={"b": 900.0}).score(recs)
    assert out_b.iloc[0]["ticker"] == "T0", "agirlik degisimi siralamayi degistirmeli"
    assert diag_b["weights_applied"]["b"] > 90


def test_penalty_is_subtracted():
    recs = [_rec(f"T{i}", a=float(i), b=float(i)) for i in range(10)]
    clean, _ = Scorer(_cfg()).score(recs)
    top_clean = float(clean.iloc[0]["total_score"])

    recs[9]["penalty_flags"]["bad"] = True
    penalized, _ = Scorer(_cfg()).score(recs)
    row = penalized[penalized["ticker"] == "T9"].iloc[0]

    assert row["penalty"] == -10.0
    assert abs(float(row["total_score"]) - (top_clean - 10.0)) < 1e-6
    assert len(row["penalties_hit"]) == 1


def test_weights_normalize_to_100():
    recs = [_rec(f"T{i}", a=float(i), b=float(i), c=float(i)) for i in range(20)]
    _, diag = Scorer(_cfg()).score(recs)
    assert abs(sum(diag["weights_applied"].values()) - 100.0) < 1e-6
    # etki puanlari yuksekten dusuge sirali gelmeli
    ws = [f["weight"] for f in diag["active_factors"]]
    assert ws == sorted(ws, reverse=True), "aktif faktorler agirliga gore sirali olmali"


def test_score_bounds():
    recs = [_rec(f"T{i}", a=float(i), b=float(i), penalty=(i % 2 == 0)) for i in range(20)]
    out, _ = Scorer(_cfg()).score(recs)
    assert out["total_score"].between(0, 100).all(), "skorlar 0-100 disina cikmamali"


def test_low_confidence_flag():
    recs = [_rec(f"T{i}", a=float(i), b=float(i), c=float(i)) for i in range(20)]
    # T5: sadece 'c' var (agirlik 10/100) -> kapsama %10 < %60
    recs[5]["raw"]["a"] = None
    recs[5]["raw"]["b"] = None

    out, _ = Scorer(_cfg()).score(recs)
    row = out[out["ticker"] == "T5"].iloc[0]
    assert row["low_confidence"] is True or bool(row["low_confidence"])
    assert row["coverage"] < 0.6


def test_low_coverage_is_shrunk_toward_neutral():
    """Verisinin yarisi eksik bir hisse, elindeki birkac faktor yuksek diye
    listenin basina cikamamali (guven duzeltmesi)."""
    cfg = _cfg(coverage_shrink_power=1.0, min_coverage_to_include=0.0)

    # T0..T8 tam veriyle, T9 sadece 'a' ile (agirlik 60/100 -> kapsama %60)
    recs = [_rec(f"T{i}", a=float(i), b=float(i), c=float(i)) for i in range(10)]
    recs[9]["raw"]["b"] = None
    recs[9]["raw"]["c"] = None

    out, _ = Scorer(cfg).score(recs)
    t9 = out[out["ticker"] == "T9"].iloc[0]
    t8 = out[out["ticker"] == "T8"].iloc[0]

    assert t9["coverage"] < 0.7, "kapsama dusuk olmali"
    # Tum faktorlerde tepe olmasina ragmen tam veriye sahip T8'i gecmemeli
    assert t9["total_score"] < t8["total_score"], (
        f"dusuk kapsamali T9 ({t9['total_score']}) tam kapsamali "
        f"T8'i ({t8['total_score']}) gecmemeliydi")
    # 50'ye dogru cekilmis olmali ama 50'nin altina inmemeli (ham skoru yuksekti)
    assert 50.0 < t9["total_score"] < 100.0


def test_shrinkage_is_symmetric():
    """Buzulme bir CEZA degil: dusuk kapsamali KOTU skor da 50'ye dogru yukselir."""
    cfg = _cfg(coverage_shrink_power=1.0, min_coverage_to_include=0.0)
    recs = [_rec(f"T{i}", a=float(i), b=float(i), c=float(i)) for i in range(10)]
    # T0 en kotu; ayrica verisi eksik olsun
    recs[0]["raw"]["b"] = None
    recs[0]["raw"]["c"] = None

    out, _ = Scorer(cfg).score(recs)
    t0 = out[out["ticker"] == "T0"].iloc[0]
    t1 = out[out["ticker"] == "T1"].iloc[0]
    assert t0["total_score"] > t1["total_score"], "kotu skor notre dogru YUKSELMELI"
    assert t0["total_score"] < 50.0


def test_shrinkage_barely_affects_one_sparse_factor():
    """Tek seyrek faktorun (orn. WSB) eksikligi siralamayi bozmamali."""
    cfg = _cfg(coverage_shrink_power=1.0, min_coverage_to_include=0.0)
    recs = [_rec(f"T{i}", a=float(i), b=float(i), c=float(i)) for i in range(10)]
    recs[9]["raw"]["c"] = None          # sadece 'c' (agirlik 10/100) eksik

    out, _ = Scorer(cfg).score(recs)
    assert out.iloc[0]["ticker"] == "T9", "tek kucuk faktor eksikligi 1.ligi bozmamali"
    assert out.iloc[0]["coverage"] >= 0.85


def test_very_low_coverage_is_excluded():
    cfg = _cfg(min_coverage_to_include=0.55)
    recs = [_rec(f"T{i}", a=float(i), b=float(i), c=float(i)) for i in range(10)]
    recs[4]["raw"]["a"] = None
    recs[4]["raw"]["b"] = None          # sadece 'c' kalir -> kapsama %10

    out, diag = Scorer(cfg).score(recs)
    assert "T4" not in set(out["ticker"]), "cok dusuk kapsamali hisse elenmeliydi"
    assert any(x["ticker"] == "T4" for x in diag.get("excluded_low_coverage", []))
    assert list(out["rank"]) == list(range(1, len(out) + 1)), "siralama yeniden numaralanmali"


def test_pinned_survives_low_coverage_exclusion():
    """Izleme listesindeki hisse, verisi cok eksik olsa bile listeden atilmaz.

    Kullanici bir hisseyi listesine aldiysa gozden kaybolmasi kabul edilemez;
    tam tersine kotulesen durumu gormesi gerekir.
    """
    cfg = _cfg(min_coverage_to_include=0.55)
    recs = [_rec(f"T{i}", a=float(i), b=float(i), c=float(i)) for i in range(10)]
    recs[4]["raw"]["a"] = None
    recs[4]["raw"]["b"] = None          # kapsama %10 -> normalde elenirdi

    out_free, _ = Scorer(cfg).score(recs)
    assert "T4" not in set(out_free["ticker"]), "on kosul: sabitlenmemisken elenmeli"

    out_pin, diag = Scorer(cfg, pinned={"T4"}).score(recs)
    assert "T4" in set(out_pin["ticker"]), "sabitlenmis hisse elenmemeliydi"
    row = out_pin[out_pin["ticker"] == "T4"].iloc[0]
    assert bool(row["pinned"]) is True
    assert not any(x["ticker"] == "T4" for x in diag.get("excluded_low_coverage", []))


def test_pinned_flag_only_on_pinned():
    recs = [_rec(f"T{i}", a=float(i), b=float(i)) for i in range(6)]
    out, _ = Scorer(_cfg(), pinned={"T1", "T3"}).score(recs)
    pinned = set(out[out["pinned"]]["ticker"])
    assert pinned == {"T1", "T3"}, f"beklenmedik sabitleme: {pinned}"


def test_pinned_does_not_change_score():
    """Sabitleme sadece GORUNURLUK saglar, puani sismez."""
    recs = [_rec(f"T{i}", a=float(i), b=float(i)) for i in range(8)]
    plain, _ = Scorer(_cfg()).score(recs)
    pinned, _ = Scorer(_cfg(), pinned={"T2"}).score(recs)
    a = float(plain[plain["ticker"] == "T2"].iloc[0]["total_score"])
    b = float(pinned[pinned["ticker"] == "T2"].iloc[0]["total_score"])
    assert abs(a - b) < 1e-9, "sabitleme puani degistirmemeli"


def test_cluster_budget_caps_correlated_group():
    """Denetim bulgusu Y1: birbirinin tekrari parametrelerin agirliklari
    toplanip gizli bir tek-bahis olusturuyordu. Kume butcesi bunu keser."""
    clusters = [{"id": "trend", "name_tr": "Trend", "budget": 20.0,
                 "members": ["a", "b"]}]          # a+b = 90 -> 20
    recs = [_rec(f"T{i}", a=float(i), b=float(i), c=float(9 - i)) for i in range(10)]

    _, plain = Scorer(_cfg()).score(recs)
    _, capped = Scorer(_cfg(clusters=clusters)).score(recs)

    share_before = plain["weights_applied"]["a"] + plain["weights_applied"]["b"]
    share_after = capped["weights_applied"]["a"] + capped["weights_applied"]["b"]
    assert share_before > 85, f"on kosul: kume butcesiz agirlikli olmali ({share_before})"
    assert 60 < share_after < 70, f"kume butceye cekilmeliydi, gelen: {share_after}"

    # Uyelerin KENDI aralarindaki oran korunmali (a:b = 2:1)
    w = capped["weights_applied"]
    assert abs(w["a"] / w["b"] - 2.0) < 1e-2

    info = capped["clusters"][0]
    assert info["weight_before"] == 90.0 and info["weight_after"] == 20.0


def test_penalty_scales_with_score_spread():
    """Denetim bulgusu Y2: sabit puan ceza, dagilim daraldikca orantisiz
    buyuyordu. Sigma cinsinden ceza dagilimla birlikte olceklenmeli."""
    cfg = _cfg(penalty_sigma=-0.5)

    wide = [_rec(f"T{i}", a=float(i * 100), b=float(i * 100)) for i in range(12)]
    wide[0]["penalty_flags"]["bad"] = True
    out_w, diag_w = Scorer(cfg).score(wide)
    pen_w = float(out_w[out_w["ticker"] == "T0"].iloc[0]["penalty"])

    # Ayni siralama, ama skorlari birbirine cok yakin bir evren kurgulanamaz
    # (rank normalizasyonu ayni dagilimi verir) -> sigma'yi dogrudan dogrula
    assert diag_w["penalty_sigma"] > 0
    expected = -0.5 * diag_w["penalty_sigma"]
    # penalty sonuc satirinda 2 haneye yuvarlanir; tolerans ona gore
    assert abs(pen_w - expected) < 0.02, f"ceza sigma ile olceklenmeli: {pen_w} vs {expected}"

    hit = out_w[out_w["ticker"] == "T0"].iloc[0]["penalties_hit"][0]
    assert hit["sigma"] == -0.5
    assert abs(hit["points"] - expected) < 0.01


def test_penalty_total_is_capped():
    cfg = _cfg(penalty_sigma=-2.0, penalty_total_cap_sigma=0.8)
    recs = [_rec(f"T{i}", a=float(i), b=float(i)) for i in range(10)]
    recs[5]["penalty_flags"]["bad"] = True
    out, diag = Scorer(cfg).score(recs)
    pen = float(out[out["ticker"] == "T5"].iloc[0]["penalty"])
    # hem penalty hem penalty_sigma raporlamada yuvarlanir
    assert abs(pen) <= 0.8 * diag["penalty_sigma"] + 0.02, (
        f"ceza tavani asildi: {abs(pen):.3f} > {0.8 * diag['penalty_sigma']:.3f}")


def test_legacy_points_penalty_still_works():
    """Eski 'points' alanli config'ler bozulmamali (geriye uyum)."""
    recs = [_rec(f"T{i}", a=float(i), b=float(i)) for i in range(10)]
    recs[9]["penalty_flags"]["bad"] = True
    out, _ = Scorer(_cfg()).score(recs)          # points: -10
    pen = float(out[out["ticker"] == "T9"].iloc[0]["penalty"])
    assert abs(pen - (-10.0)) < 1e-6, f"eski puan cezasi korunmali, gelen {pen}"


def test_band_label_kept_when_norm_is_rank():
    """Denetim bulgusu O4: analyst_consensus rank'e gecti ama 'Al/Guclu Al'
    etiketi kaybolmamali."""
    cfg = _cfg()
    cfg["factors"].append({
        "id": "rating", "name_tr": "Tavsiye", "category": "analyst", "weight": 20.0,
        "direction": "lower_better", "norm": "rank",
        "bands": [{"max": 1.5, "score": 100, "label_tr": "Guclu Al"},
                  {"max": 2.5, "score": 60, "label_tr": "Al"},
                  {"max": 5.1, "score": 0, "label_tr": "Sat"}],
    })
    recs = [_rec(f"T{i}", a=1.0, b=1.0) for i in range(6)]
    for i, v in enumerate([1.2, 2.0, 3.0, 1.4, 4.5, 2.2]):
        recs[i]["raw"]["rating"] = v

    out, _ = Scorer(cfg).score(recs)
    labels = {r["ticker"]: r["factors"]["rating"]["band_label"] for _, r in out.iterrows()}
    assert labels["T0"] == "Guclu Al" and labels["T4"] == "Sat", labels
    # lower_better + rank: dusuk ham deger yuksek skor almali
    sc = {r["ticker"]: r["factors"]["rating"]["score"] for _, r in out.iterrows()}
    assert sc["T0"] > sc["T4"], "1.2 notlu hisse, 4.5 notludan yuksek skor almali"


def test_empty_input():
    out, diag = Scorer(_cfg()).score([])
    assert out.empty and "error" in diag


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {fn.__name__}: {e}")
        except Exception as e:
            failed += 1
            print(f"  ERROR {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns)-failed}/{len(fns)} gecti")
    raise SystemExit(1 if failed else 0)
