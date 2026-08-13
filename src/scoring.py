"""Skorlama motoru: ham faktorler -> normalize skor -> agirlikli toplam.

Tasarim ilkeleri
----------------
1. HICBIR FAKTOR ZORUNLU DEGIL. Bir hisse icin veri yoksa o faktorun agirligi
   diger faktorlere ORANTILI DAGITILIR. Boylece veri eksikligi cezaya donusmez.
   (Kullanicinin "cok kisitlayan parametre gozden cikarilabilsin" istegi.)

2. OTOMATIK DEVRE DISI BIRAKMA. Bir faktor evrenin %15'inden azinda mevcutsa
   (orn. WSB anmalari) tumuyle devre disi kalir ve raporda belirtilir.

3. CAPRAZ KESITSEL SIRALAMA. 'rank' normalizasyonu yuzdelik sira kullanir;
   aykiri degerlere dayaniklidir ve olcek birimi farklarini yok eder.

4. SEKTOR NOTRLUGU. Acikken faktor skorlari sektor icinde z-skorlanir; boylece
   "tum teknoloji hisseleri pahali" gibi sistematik kaymalar temizlenir.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


# =============================================================================
#  Normalizasyon
# =============================================================================
def _percentile_scores(values: pd.Series) -> pd.Series:
    """Yuzdelik sira -> 0-100. Eksikler NaN kalir."""
    v = pd.to_numeric(values, errors="coerce")
    if v.notna().sum() == 0:
        return pd.Series(np.nan, index=values.index, dtype=float)
    if v.notna().sum() == 1:
        # Tek gozlem: siralama anlamsiz -> notr
        return v.where(v.isna(), 50.0)
    return v.rank(pct=True, na_option="keep") * 100.0


def _band_score(value: float, bands: list[dict]) -> float:
    """Mutlak esiklerden puan (siralamadan bagimsiz)."""
    if value is None or not np.isfinite(value):
        return np.nan
    for b in bands:
        if value <= float(b["max"]):
            return float(b["score"])
    return float(bands[-1]["score"])


def _band_label(value: float, bands: list[dict]) -> str | None:
    if value is None or not np.isfinite(value):
        return None
    for b in bands:
        if value <= float(b["max"]):
            return b.get("label_tr")
    return bands[-1].get("label_tr")


def _sector_neutralize(scores: pd.Series, sectors: pd.Series) -> pd.Series:
    """Sektor icinde z-skorla, sonra 0-100'e geri esle."""
    out = scores.copy().astype(float)
    for sec, idx in sectors.groupby(sectors).groups.items():
        sub = scores.loc[idx].dropna()
        if len(sub) < 5:          # kucuk sektor -> dokunma
            continue
        sd = sub.std(ddof=0)
        if sd is None or not np.isfinite(sd) or sd < 1e-9:
            continue
        z = (sub - sub.mean()) / sd
        out.loc[sub.index] = (50.0 + 15.0 * z).clip(0, 100)
    return out


# =============================================================================
#  Ana skorlayici
# =============================================================================
class Scorer:
    def __init__(self, config: dict, disabled: set[str] | None = None,
                 weight_overrides: dict[str, float] | None = None,
                 pinned: set[str] | None = None):
        self.config = config
        self.meta_cfg = config.get("meta", {}) or {}
        self.disabled = set(disabled or ())
        self.weight_overrides = weight_overrides or {}
        # Izleme listesindeki semboller dusuk kapsama nedeniyle elenmez.
        self.pinned = set(pinned or ())

        self.factors = [f for f in config["factors"] if f["id"] not in self.disabled]
        self.penalties = {p["id"]: p for p in config.get("penalties", [])}
        self.clusters = config.get("clusters", []) or []

    def _weight(self, fid: str, default: float) -> float:
        return float(self.weight_overrides.get(fid, default))

    # ------------------------------------------------------------------ ana
    def score(self, records: list[dict]) -> tuple[pd.DataFrame, dict[str, Any]]:
        """records: factors.compute_all ciktilarinin listesi.

        Doner: (siralanmis DataFrame, tani/diagnostik sozlugu)
        """
        ok = [r for r in records if r.get("ok")]
        if not ok:
            return pd.DataFrame(), {"error": "skorlanabilir hisse yok"}

        index = [r["ticker"] for r in ok]
        raw_df = pd.DataFrame([r["raw"] for r in ok], index=index)
        sectors = pd.Series([r.get("sector", "Bilinmiyor") for r in ok], index=index)

        n = len(ok)
        diagnostics: dict[str, Any] = {
            "universe_size": n,
            "auto_disabled": [],
            "factor_coverage": {},
            "active_factors": [],
        }

        min_cov = float(self.meta_cfg.get("auto_disable_coverage_below", 0.15))
        neutralize = bool(self.meta_cfg.get("sector_neutralize", True))

        norm_scores: dict[str, pd.Series] = {}
        band_labels: dict[str, pd.Series] = {}
        active: list[dict] = []

        # --- 1) Her faktoru 0-100'e normalize et -----------------------------
        for f in self.factors:
            fid = f["id"]
            if fid not in raw_df.columns:
                continue

            col = pd.to_numeric(raw_df[fid], errors="coerce")
            coverage = float(col.notna().sum()) / n
            diagnostics["factor_coverage"][fid] = round(coverage, 4)

            # --- otomatik devre disi birakma (kullanicinin istegi)
            if coverage < min_cov:
                diagnostics["auto_disabled"].append({
                    "id": fid,
                    "name_tr": f.get("name_tr", fid),
                    "coverage": round(coverage, 4),
                    "reason_tr": f"Kapsama %{coverage * 100:.1f} < esik %{min_cov * 100:.0f} "
                                 f"- agirlik diger faktorlere dagitildi",
                })
                continue

            method = f.get("norm", "rank")
            # Bantlar tanimliysa ETIKET her zaman uretilir — norm 'rank' olsa
            # bile kullanici "Al / Guclu Al" karsiligini gormeye devam eder.
            if f.get("bands"):
                band_labels[fid] = col.apply(lambda v: _band_label(v, f["bands"]))

            if method == "band":
                s = col.apply(lambda v: _band_score(v, f["bands"]))
            elif method == "raw":
                s = col.clip(0, 100)
            else:  # rank
                s = _percentile_scores(col)

            # analyst_consensus icin yon: dusuk deger (1=Guclu Al) daha iyi.
            # band puanlari zaten bunu icerir; sadece 'rank' icin cevirmek gerekir.
            if method == "rank" and f.get("direction") == "lower_better":
                s = 100.0 - s

            if neutralize and method == "rank" and f.get("category") not in ("preference",):
                s = _sector_neutralize(s, sectors)

            norm_scores[fid] = s
            active.append(f)

        if not active:
            return pd.DataFrame(), {"error": "aktif faktor kalmadi"}

        # --- 2) Agirliklari normalize et -------------------------------------
        weights = {f["id"]: self._weight(f["id"], f["weight"]) for f in active}

        # --- 2a) Korelasyon kumesi butcesi (denetim bulgusu Y1) --------------
        # Birbirinin tekrari olan parametreler tek tek agirlik tasidiginda
        # etkileri toplanip gizli bir tek-bahis olusturuyordu. Kume uyelerinin
        # kendi aralarindaki oranlari korunur, toplamlari butceye cekilir.
        cluster_info = []
        for cl in self.clusters:
            members = [m for m in cl.get("members", []) if m in weights]
            if len(members) < 2:
                continue
            before = sum(weights[m] for m in members)
            budget = float(cl.get("budget", before))
            if before <= 0 or budget <= 0:
                continue
            k = budget / before
            for m in members:
                weights[m] *= k
            cluster_info.append({
                "id": cl.get("id"),
                "name_tr": cl.get("name_tr", cl.get("id")),
                "members": members,
                "weight_before": round(before, 2),
                "weight_after": round(budget, 2),
                "scale": round(k, 3),
            })
        if cluster_info:
            diagnostics["clusters"] = cluster_info

        total_w = sum(weights.values())
        if total_w <= 0:
            return pd.DataFrame(), {"error": "toplam agirlik sifir"}
        weights = {k: 100.0 * v / total_w for k, v in weights.items()}

        diagnostics["active_factors"] = [
            {"id": f["id"], "name_tr": f.get("name_tr", f["id"]),
             "category": f.get("category"), "weight": round(weights[f["id"]], 2),
             "coverage": diagnostics["factor_coverage"].get(f["id"]),
             "rationale_tr": (f.get("rationale_tr") or "").strip()}
            for f in sorted(active, key=lambda x: -weights[x["id"]])
        ]

        score_df = pd.DataFrame(norm_scores, index=index)

        # --- 3) Eksik faktor agirligini yeniden dagit ------------------------
        w_vec = pd.Series(weights)
        present = score_df.notna()
        available_w = present.mul(w_vec, axis=1).sum(axis=1)     # hisse basina mevcut agirlik
        coverage_ratio = (available_w / 100.0).clip(0, 1)

        contrib = score_df.fillna(0.0).mul(w_vec, axis=1)
        # Yeniden dagitim = mevcut agirliklarla agirlikli ortalama
        raw_base = contrib.sum(axis=1) / available_w.replace(0, np.nan)

        # --- KAPSAMA BUZULMESI (shrinkage) ---------------------------------
        # Yeniden dagitim tek basina yeterli degil: verisinin yarisi eksik olan
        # bir hisse, sadece elinde kalan birkac faktor yuksek diye listenin
        # basina cikabiliyordu. Bunu engellemek icin skor, eksik kapsama
        # oraninda NOTR degere (50) dogru cekilir.
        #
        # Bu bir CEZA DEGILDIR, guven duzeltmesidir ve simetriktir: kotu skorlu
        # bir hisse de ayni sekilde 50'ye dogru YUKSELIR. Mantik: veri yoksa
        # hissenin iyi oldugunu da kotu oldugunu da guclu bicimde iddia edemeyiz.
        #
        # Tek bir seyrek faktorun (orn. WSB, agirlik ~2) eksikligi kapsamayi
        # %98'de tutar ve etkisi ihmal edilebilir — kullanicinin "cok kisitlayan
        # parametre gozden cikarilabilsin" istegi korunur.
        shrink_power = float(self.meta_cfg.get("coverage_shrink_power", 1.0))
        if shrink_power > 0:
            shrink = coverage_ratio.clip(0, 1) ** shrink_power
            base_score = 50.0 + (raw_base - 50.0) * shrink
        else:
            base_score = raw_base

        # --- 4) Cezalari uygula ---------------------------------------------
        # Cezalar SIGMA cinsinden tanimlanir ve bu taramanin gercek skor
        # standart sapmasiyla puana cevrilir. Sabit puan kullanilsaydi, skor
        # dagilimi daraldikca cezalar orantisiz agirlasirdi (bulgu Y2).
        sigma = float(base_score.std(ddof=0))
        if not np.isfinite(sigma) or sigma < 1e-6:
            sigma = 1.0
        cap = float(self.meta_cfg.get("penalty_total_cap_sigma", 3.0))

        penalty_total = pd.Series(0.0, index=index)
        penalty_detail: dict[str, list[dict]] = {}
        for r in ok:
            tk = r["ticker"]
            hits = []
            sig_sum = 0.0
            for pid, flag in (r.get("penalty_flags") or {}).items():
                if not flag or pid not in self.penalties:
                    continue
                p = self.penalties[pid]
                # sigma alani yoksa eski 'points' alanina duser (geriye uyum)
                s_val = p.get("sigma")
                s_val = float(s_val) if s_val is not None else float(p.get("points", 0)) / sigma
                sig_sum += s_val
                hits.append({"id": pid, "name_tr": p.get("name_tr", pid),
                             "sigma": round(s_val, 3),
                             "points": round(s_val * sigma, 2)})

            sig_sum = max(sig_sum, -abs(cap))          # guvenlik siniri
            penalty_total[tk] = sig_sum * sigma
            penalty_detail[tk] = hits

        diagnostics["penalty_sigma"] = round(sigma, 3)
        diagnostics["penalty_cap_sigma"] = cap

        final = (base_score + penalty_total).clip(0, 100)

        # --- 5) Sonuc tablosu -------------------------------------------------
        by_ticker = {r["ticker"]: r for r in ok}
        min_conf = float(self.meta_cfg.get("min_coverage_for_confidence", 0.60))

        # Verisi neredeyse hic olmayan hisseler siralamadan tamamen cikarilir.
        min_include = float(self.meta_cfg.get("min_coverage_to_include", 0.0))
        excluded = [tk for tk in index
                    if coverage_ratio[tk] < min_include and tk not in self.pinned]
        if excluded:
            diagnostics["excluded_low_coverage"] = [
                {"ticker": tk, "coverage": round(float(coverage_ratio[tk]), 3)}
                for tk in excluded
            ]
        keep = [tk for tk in index if tk not in set(excluded)]
        if not keep:
            return pd.DataFrame(), {"error": "tum hisseler kapsama esiginin altinda"}

        rows = []
        for tk in keep:
            r = by_ticker[tk]
            per_factor = {}
            for f in active:
                fid = f["id"]
                s = score_df.at[tk, fid]
                per_factor[fid] = {
                    "name_tr": f.get("name_tr", fid),
                    "category": f.get("category"),
                    "weight": round(weights[fid], 2),
                    "raw": r["raw"].get(fid),
                    "score": None if pd.isna(s) else round(float(s), 2),
                    "contribution": None if pd.isna(s) else round(float(s) * weights[fid] / 100.0, 3),
                    "available": bool(not pd.isna(s)),
                    "band_label": (band_labels[fid].get(tk) if fid in band_labels else None),
                    "meta": r["meta"].get(fid),
                }

            rows.append({
                "ticker": tk,
                "name": r.get("name"),
                "sector": r.get("sector"),
                "industry": r.get("industry"),
                "price": r.get("price"),
                "currency": r.get("currency"),
                "market_cap": r.get("market_cap"),
                "avg_dollar_volume": r.get("avg_dollar_volume"),
                "max_position_usd": r.get("max_position_usd"),
                "turnover_daily": r.get("turnover_daily"),
                "rsi14": r.get("rsi14"),
                "days_to_earnings": r.get("days_to_earnings"),
                "returns": r.get("returns"),
                "base_score": None if pd.isna(base_score[tk]) else round(float(base_score[tk]), 2),
                "penalty": round(float(penalty_total[tk]), 2),
                "penalties_hit": penalty_detail.get(tk, []),
                "total_score": None if pd.isna(final[tk]) else round(float(final[tk]), 2),
                "coverage": round(float(coverage_ratio[tk]), 3),
                "low_confidence": bool(coverage_ratio[tk] < min_conf),
                "pinned": tk in self.pinned,
                "factors": per_factor,
                "snapshot_date": r.get("snapshot_date"),
            })

        out = pd.DataFrame(rows).sort_values("total_score", ascending=False, na_position="last")
        out.insert(0, "rank", range(1, len(out) + 1))

        # Kategori bazli alt skorlar (radar/karsilastirma icin)
        cats: dict[str, list[str]] = {}
        for f in active:
            cats.setdefault(f.get("category", "other"), []).append(f["id"])

        cat_scores = []
        for _, row in out.iterrows():
            cs = {}
            for cat, fids in cats.items():
                vals, ws = [], []
                for fid in fids:
                    fd = row["factors"].get(fid)
                    if fd and fd["available"]:
                        vals.append(fd["score"])
                        ws.append(fd["weight"])
                cs[cat] = round(float(np.average(vals, weights=ws)), 1) if vals else None
            cat_scores.append(cs)
        out["category_scores"] = cat_scores

        diagnostics["weights_applied"] = {k: round(v, 3) for k, v in
                                          sorted(weights.items(), key=lambda x: -x[1])}
        diagnostics["mean_coverage"] = round(float(coverage_ratio.mean()), 3)
        diagnostics["sector_neutralized"] = neutralize

        return out, diagnostics
