"""Yapay zeka modeline aktarim katmani.

Bu modul sistemi "ML-hazir" yapan kismidir. Uc is yapar:

1. EXPORT  — her tarama sonucunu tarih damgali, tidy bir ozellik matrisine cevirir
             (feature store). Zamanla biriken bu dosyalar egitim setini olusturur.
2. LABEL   — gecmis anlik goruntuleri ILERI GETIRI ile etiketler (denetimli ogrenme
             hedefi). Look-ahead bias'a dusmeden.
3. LEARN   — faktor agirliklarini veriden ogrenir (IC tabanli veya ridge regresyon)
             ve weights.yaml'a yazilabilecek yeni agirliklar uretir.

Kritik tasarim notu: uretilen ozellik matrisi "point-in-time"dir — her satir o gun
BILINEBILIR bilgiyi tasir. Etiket ise gelecekten gelir. Bu ayrim korunmazsa model
gercek disi yuksek basari gosterir.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

FEATURE_STORE = Path(__file__).resolve().parents[1] / "data" / "feature_store"


# =============================================================================
#  1) EXPORT
# =============================================================================
def to_feature_matrix(result_df: pd.DataFrame, factor_ids: list[str]) -> pd.DataFrame:
    """Skorlama ciktisini duz (flat) ozellik matrisine cevirir.

    Her faktor icin iki sutun:
        raw_<id>   -> ham deger (olcek korunur, agac modelleri icin iyi)
        score_<id> -> 0-100 normalize (dogrusal modeller icin iyi)
    Ayrica kapsama maskesi: has_<id> (0/1) — eksik veriyi model ogrenebilsin.
    """
    rows = []
    for _, r in result_df.iterrows():
        row = {
            "snapshot_date": r.get("snapshot_date"),
            "ticker": r["ticker"],
            "sector": r.get("sector"),
            "industry": r.get("industry"),
            "price": r.get("price"),
            "market_cap": r.get("market_cap"),
            "total_score": r.get("total_score"),
            "base_score": r.get("base_score"),
            "penalty": r.get("penalty"),
            "coverage": r.get("coverage"),
            "rank": r.get("rank"),
        }
        rets = r.get("returns") or {}
        for k, v in rets.items():
            row[f"trailing_return_{k}"] = v

        # Ham seri ozellikleri: modelin 28 insan hipotezinin disinda gordugu
        # tek sey. Skorlamaya girmez; feature store'a yazilir ve egitimde
        # kullanilir (bkz. factors.series_features).
        for k, v in (r.get("series") or {}).items():
            row[f"series_{k}"] = v

        factors = r.get("factors") or {}
        for fid in factor_ids:
            fd = factors.get(fid)
            row[f"raw_{fid}"] = fd["raw"] if fd else None
            row[f"score_{fid}"] = fd["score"] if fd else None
            row[f"has_{fid}"] = 1 if (fd and fd["available"]) else 0

        for pid, hit in (
            {p["id"]: True for p in (r.get("penalties_hit") or [])}
        ).items():
            row[f"penalty_{pid}"] = 1
        rows.append(row)

    df = pd.DataFrame(rows)
    # Isaretlenmeyen ceza sutunlarini 0 yap
    for c in [c for c in df.columns if c.startswith("penalty_")]:
        df[c] = df[c].fillna(0).astype(int)
    return df


def save_snapshot(feature_df: pd.DataFrame, tag: str = "") -> Path:
    """Ozellik matrisini feature store'a tarih damgali kaydeder.

    Ayni gun icinde tekrar calistirilirsa dosyanin UZERINE YAZMAZ, BIRLESTIRIR.
    Donusumlu tarama her turda farkli sembolleri gezdigi icin uzerine yazmak,
    o gun daha once toplanan hisseleri silerdi (bulgu D1 + donusumlu taramanin
    yan etkisi).
    """
    FEATURE_STORE.mkdir(parents=True, exist_ok=True)
    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    suffix = f"_{tag}" if tag else ""
    path = FEATURE_STORE / f"snapshot_{date}{suffix}.csv"

    out = feature_df
    if path.exists():
        try:
            old = pd.read_csv(path)
            if "ticker" in old.columns and "ticker" in feature_df.columns:
                # Ayni sembol iki turda da varsa YENISI kalir
                keep = old[~old["ticker"].isin(feature_df["ticker"])]
                out = pd.concat([feature_df, keep], ignore_index=True, sort=False)
                if "total_score" in out.columns:
                    out = out.sort_values("total_score", ascending=False,
                                          na_position="last")
                    out["rank"] = range(1, len(out) + 1)
        except Exception:
            out = feature_df

    out.to_csv(path, index=False, encoding="utf-8")
    return path


def previous_snapshot(before: str | None = None) -> tuple[pd.DataFrame | None, str | None]:
    """Bugunden ONCEKI en son anlik goruntuyu dondurur.

    Gunluk "ne degisti" karsilastirmasinin temeli. Ayni gun icinde tekrar
    tarama yapilirsa dosya uzerine yazildigi icin, karsilastirma her zaman
    onceki bir GUNE karsi yapilir — kendine karsi degil.
    """
    if not FEATURE_STORE.exists():
        return None, None

    cutoff = before or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    dated: list[tuple[str, Path]] = []
    for p in FEATURE_STORE.glob("snapshot_*.csv"):
        stamp = p.stem.replace("snapshot_", "")[:10]
        if len(stamp) == 10 and stamp < cutoff:
            dated.append((stamp, p))

    if not dated:
        return None, None

    stamp, path = max(dated, key=lambda x: x[0])
    try:
        return pd.read_csv(path), stamp
    except Exception:
        return None, None


def _last_seen_map(before: str | None = None) -> tuple[dict[str, dict], list[str]]:
    """Her sembolun EN SON hangi anlik goruntude, hangi konumda goruldugu.

    Neden tek bir "dunku dosya" yetmiyor: donusumlu tarama her turda evrenin
    farkli bir dilimini geziyor. Sabit bir onceki gune bakmak, ortak sembol
    sayisini neredeyse sifira dusurur (olculdu: 429 ve 248 satir arasinda
    yalnizca 3 ortak sembol). Bu yuzden sembol basina "en son gorulme" aranir.
    """
    if not FEATURE_STORE.exists():
        return {}, []

    cutoff = before or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    files: list[tuple[str, Path]] = []
    for p in FEATURE_STORE.glob("snapshot_*.csv"):
        stamp = p.stem.replace("snapshot_", "")[:10]
        if len(stamp) == 10 and stamp < cutoff:
            files.append((stamp, p))

    seen: dict[str, dict] = {}
    used: list[str] = []
    for stamp, path in sorted(files):          # eskiden yeniye: yeni olan ezer
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if "ticker" not in df.columns or "rank" not in df.columns:
            continue
        n = max(1, len(df))
        used.append(stamp)
        for _, row in df.iterrows():
            tk = row.get("ticker")
            rk, ts = row.get("rank"), row.get("total_score")
            if pd.isna(tk) or pd.isna(rk):
                continue
            seen[str(tk)] = {
                "date": stamp,
                "rank": float(rk),
                "n": n,
                "percentile": float(rk) / n,
                "score": float(ts) if pd.notna(ts) else None,
            }
    return seen, used


def compute_deltas(result_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Her hisseyi KENDI son gorulme anina gore karsilastirir.

    Eklenen sutunlar:
        prev_rank, prev_date, rank_change, prev_score, score_change, is_new

    Siralamalar farkli buyuklukteki evrenlerden geldigi icin dogrudan
    karsilastirilamaz; bu yuzden karsilastirma YUZDELIK DILIM uzerinden yapilir
    ve sonuc bugunun evren buyuklugune cevrilir. Boylece "500 hisseden 50."
    ile "800 hisseden 80." dogru biçimde esdeger sayilir.
    """
    out = result_df.copy()
    seen, used = _last_seen_map()

    if not seen:
        for c in ("prev_rank", "prev_date", "rank_change", "prev_score", "score_change"):
            out[c] = None
        out["is_new"] = True
        return out, {"compared_to": None, "reason": "karsilastirilacak onceki tarama yok"}

    n_now = max(1, len(out))
    prev_rank, prev_date, prev_score, rank_chg, score_chg, is_new = [], [], [], [], [], []

    for tk, rk, sc in zip(out["ticker"], out["rank"], out["total_score"]):
        rec = seen.get(str(tk))
        if rec is None:
            prev_rank.append(None); prev_date.append(None); prev_score.append(None)
            rank_chg.append(None); score_chg.append(None); is_new.append(True)
            continue
        # Onceki yuzdelik dilimi bugunun evrenine tasi
        equiv_prev = rec["percentile"] * n_now
        prev_rank.append(int(round(rec["rank"])))
        prev_date.append(rec["date"])
        prev_score.append(rec["score"])
        rank_chg.append(int(round(equiv_prev - float(rk))))     # + = yukari cikti
        score_chg.append(None if rec["score"] is None
                         else round(float(sc) - rec["score"], 2))
        is_new.append(False)

    out["prev_rank"] = prev_rank
    out["prev_date"] = prev_date
    out["prev_score"] = prev_score
    out["rank_change"] = rank_chg
    out["score_change"] = score_chg
    out["is_new"] = is_new

    matched = int(sum(1 for x in is_new if not x))
    return out, {
        "compared_to": used[-1] if used else None,
        "snapshots_used": len(used),
        "matched": matched,
        "new_count": int(sum(is_new)),
        "moved_up": int(sum(1 for c in rank_chg if c is not None and c > 0)),
        "moved_down": int(sum(1 for c in rank_chg if c is not None and c < 0)),
    }


def load_all_snapshots(store: Path | None = None) -> pd.DataFrame:
    """Biriken tum anlik goruntuleri tek DataFrame'de birlestirir.

    `store`, canli feature store yerine baska bir dizinden okumaya yarar —
    ornegin backfill.py'nin urettigi gecmise donuk panel. Iki depo bilincli
    olarak AYRI tutulur: gecmise donuk satirlarda temel veri sutunlari hic
    yoktur ve birlestirilirlerse model, "sutun var mi yok mu" bilgisinden
    tarihi cikarabilir hale gelir.
    """
    store = store or FEATURE_STORE
    if not store.exists():
        return pd.DataFrame()
    frames = []
    for p in sorted(store.glob("snapshot_*.csv")):
        try:
            frames.append(pd.read_csv(p))
        except Exception:
            continue
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_recent_snapshots(n: int, store: Path | None = None) -> pd.DataFrame:
    """Yalnizca SON n anlik goruntuyu okur.

    Dizi modelinin canli tahmini icin gereken pencere sabit uzunluktadir; tum
    depoyu okumak gereksizdir. Bir yil biriktiginde feature store yuzlerce MB
    olur ve her taramada bastan okunmasi taramaya dakikalar ekler.
    """
    store = store or FEATURE_STORE
    if not store.exists() or n <= 0:
        return pd.DataFrame()
    paths = sorted(store.glob("snapshot_*.csv"))[-n:]
    frames = []
    for p in paths:
        try:
            frames.append(pd.read_csv(p))
        except Exception:
            continue
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# =============================================================================
#  2) LABEL
# =============================================================================
def label_forward_returns(snapshots: pd.DataFrame, horizon_days: int = 21,
                          use_cache: bool = True,
                          cache_max_age_days: int = 7) -> pd.DataFrame:
    """Her (tarih, hisse) satirina ileri getiri etiketi ekler.

    Yalnizca anlik goruntu tarihinden EN AZ horizon_days gun gecmis satirlar
    etiketlenir; digerleri NaN kalir (henuz sonuc belli degil).

    ONBELLEK ONCELIKLI. Etiket, GECMIS fiyattan hesaplanir: bir anlik
    goruntunun ufku dolmussa gereken fiyat zaten onbellekteki gecmiste vardir.
    Normal `fetch` 6 saatlik TTL kullandigi icin binlerce sembol icin
    gereksiz ag istegi dogurur ve hiz sinirina takilir — bir keresinde egitim
    bu yuzden dakikalarca ag bekledi. Once uzun omurlu onbellek denenir, ag
    yalnizca hic kayit yoksa kullanilir.
    """
    from .providers import yahoo

    if snapshots.empty:
        return snapshots

    df = snapshots.copy()
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"], errors="coerce")
    out_col = f"fwd_return_{horizon_days}d"
    df[out_col] = np.nan

    rate_limited = 0
    from_cache = 0
    from_network = 0
    for ticker, grp in df.groupby("ticker"):
        bundle = None
        if use_cache:
            try:
                bundle = yahoo.fetch_cached(
                    str(ticker), "2y",
                    max_age_seconds=cache_max_age_days * 24 * 3600)
            except Exception:
                bundle = None
            if bundle:
                from_cache += 1

        if not bundle:
            # Hiz siniri etiketlemeyi DURDURMAMALI: elde ne varsa onunla devam
            # edilir, kac sembolun atlandigi sonunda raporlanir.
            try:
                bundle = yahoo.fetch(str(ticker), period="2y", use_cache=use_cache)
                from_network += 1
            except yahoo.RateLimited:
                rate_limited += 1
                continue
            except Exception:
                continue

        hist = (bundle or {}).get("history")
        if not isinstance(hist, pd.DataFrame) or hist.empty:
            continue

        close = hist["Close"].copy()
        close.index = pd.to_datetime(close.index).tz_localize(None)

        for idx, row in grp.iterrows():
            d0 = row["snapshot_date"]
            if pd.isna(d0):
                continue
            past = close.loc[close.index <= d0]
            if past.empty:
                continue
            start_pos = close.index.get_loc(past.index[-1])
            end_pos = start_pos + horizon_days
            if end_pos >= len(close):
                continue  # gelecek henuz gerceklesmedi
            p0, p1 = float(close.iloc[start_pos]), float(close.iloc[end_pos])
            if p0 > 0:
                df.at[idx, out_col] = p1 / p0 - 1

    if from_network:
        print(f"      etiketleme: {from_cache} sembol onbellekten, "
              f"{from_network} sembol agdan")
    if rate_limited:
        print(f"      UYARI: {rate_limited} sembol hiz siniri nedeniyle "
              f"etiketlenemedi; bir sure sonra tekrar calistir.")

    # Capraz kesitsel goreli etiket (piyasa yonunden arindirilmis)
    df[f"{out_col}_excess"] = df.groupby("snapshot_date")[out_col].transform(
        lambda s: s - s.mean()
    )
    return df


# =============================================================================
#  3) LEARN
# =============================================================================
def factor_ic_series(labeled: pd.DataFrame, factor_id: str,
                     label_col: str, min_rows: int = 10) -> dict | None:
    """Tek bir faktorun GUN GUN IC serisi.

    information_coefficients() bu serinin ortalamasini alir. Seriyi ayrica
    dondurmemizin sebebi: ortalama, zamanla degisen her seyi gizler. Bir
    faktorun gucu azaliyor mu, yalnizca belirli bir rejimde mi calisiyor,
    ortalama gercekten sifirdan farkli mi -- hicbiri tek sayidan okunamaz.

    Doner: {'column': kullanilan sutun, 'dates': [...], 'ic': [...]}
    Sutun bulunamazsa None.
    """
    col = f"score_{factor_id}"
    # Gecmise donuk panelde yalnizca HAM deger var (score_* sutunlari
    # skorlama sirasinda uretiliyor, panelde skorlama yok). IC bir
    # SIRALAMA korelasyonu oldugu icin ham deger uzerinden de dogru
    # hesaplanir -- tek fark yon: 'lower_better' faktorlerde isaret ters
    # cikar ve bu, yorumlarken bilinmesi gereken bir seydir.
    if col not in labeled.columns:
        col = f"raw_{factor_id}"
    if col not in labeled.columns or label_col not in labeled.columns:
        return None

    dates, ics = [], []
    for d, grp in labeled.groupby("snapshot_date"):
        sub = grp[[col, label_col]].dropna()
        if len(sub) >= min_rows and sub[col].nunique() > 2:
            v = sub[col].corr(sub[label_col], method="spearman")
            if np.isfinite(v):
                dates.append(str(pd.Timestamp(d).date()))
                ics.append(float(v))
    return {"column": col, "dates": dates, "ic": ics}


def information_coefficients(labeled: pd.DataFrame, factor_ids: list[str],
                             label_col: str,
                             directions: dict[str, str] | None = None) -> pd.DataFrame:
    """Her faktorun Bilgi Katsayisi (IC) = skor ile ileri getiri Spearman korelasyonu.

    IC, kantitatif finansta bir faktorun ongoru gucunun standart olcusudur.
    |IC| > 0.03 zayif ama kullanilabilir, > 0.05 iyi, > 0.10 cok iyi kabul edilir.
    """
    directions = directions or {}
    rows = []
    for fid in factor_ids:
        ser = factor_ic_series(labeled, fid, label_col)
        if ser is None:
            continue
        col, per_date = ser["column"], ser["ic"]
        # HAM sutunda 'lower_better' parametrenin IC'si dogal olarak negatif
        # cikar (dusuk deger iyi demek). Duzeltilmezse saglikli bir parametre
        # "ters yonde" gorunur ve _save_factor_ic ona "agirligi dusurulmeli"
        # onerisi yazar. score_* sutununda yon zaten uygulanmistir.
        if col.startswith("raw_") and directions.get(fid) == "lower_better":
            per_date = [-v for v in per_date]
        if per_date:
            arr = np.array(per_date, dtype=float)
            arr = arr[np.isfinite(arr)]
            if arr.size:
                mean_ic = float(arr.mean())
                std_ic = float(arr.std(ddof=1)) if arr.size > 1 else float("nan")
                rows.append({
                    "factor": fid,
                    "source_col": col,
                    "ic_mean": round(mean_ic, 4),
                    "ic_std": None if not np.isfinite(std_ic) else round(std_ic, 4),
                    # ICIR = IC / std(IC): tutarlilik olcusu, Sharpe'in faktor karsiligi
                    "icir": None if (not np.isfinite(std_ic) or std_ic < 1e-9)
                            else round(mean_ic / std_ic, 3),
                    "periods": int(arr.size),
                })

    return pd.DataFrame(rows).sort_values("ic_mean", ascending=False) if rows else pd.DataFrame()


def learn_weights(labeled: pd.DataFrame, factor_ids: list[str], label_col: str,
                  method: str = "ic", total_weight: float = 100.0) -> dict[str, float]:
    """Veriden agirlik ogrenir.

    method="ic"    : IC ile orantili agirlik (basit, gurbuz, az veriyle calisir)
    method="ridge" : ridge regresyon katsayilari (sklearn varsa; daha cok veri ister)

    Ogrenilen agirliklar weights.yaml'a yazilabilir; sistem o andan itibaren
    uzman gorusu yerine veriyle calisir.
    """
    if method == "ridge":
        try:
            return _ridge_weights(labeled, factor_ids, label_col, total_weight)
        except ImportError:
            pass  # sklearn yok -> IC'ye dus

    ic = information_coefficients(labeled, factor_ids, label_col)
    if ic.empty:
        return {}

    # Sadece pozitif IC'ler agirlik alir; ICIR ile guven duzeltmesi yapilir
    ic = ic.copy()
    ic["confidence"] = ic["icir"].fillna(0.0).abs().clip(0, 2.0) / 2.0 + 0.5
    ic["w"] = ic["ic_mean"].clip(lower=0.0) * ic["confidence"]

    total = ic["w"].sum()
    if total <= 0:
        return {}
    return {r["factor"]: round(total_weight * r["w"] / total, 3) for _, r in ic.iterrows()}


def _ridge_weights(labeled, factor_ids, label_col, total_weight) -> dict[str, float]:
    from sklearn.linear_model import Ridge  # noqa: PLC0415

    cols = [f"score_{f}" for f in factor_ids if f"score_{f}" in labeled.columns]
    sub = labeled[cols + [label_col]].dropna()
    if len(sub) < 50:
        raise ImportError("ridge icin yetersiz veri")

    X = ((sub[cols] - sub[cols].mean()) / sub[cols].std(ddof=0).replace(0, 1)).to_numpy()
    y = sub[label_col].to_numpy()

    model = Ridge(alpha=10.0).fit(X, y)
    coefs = np.clip(model.coef_, 0, None)
    if coefs.sum() <= 0:
        raise ImportError("ridge pozitif katsayi uretmedi")

    return {fid.replace("score_", ""): round(total_weight * c / coefs.sum(), 3)
            for fid, c in zip(cols, coefs)}


# =============================================================================
#  Disa aktarim yardimcilari
# =============================================================================
def export_for_llm(result_df: pd.DataFrame, diagnostics: dict, path: Path,
                   top_n: int = 30) -> Path:
    """Bir dil modeline (LLM) dogrudan verilebilecek kompakt JSON.

    Amac: token israfi yapmadan, modelin akil yurutebilecegi yapisal ozet.
    """
    top = result_df.head(top_n)
    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "methodology": {
            "description": "Agirlikli cok faktorlu hisse skorlama. Her faktor evren "
                           "icinde 0-100 normalize edilir, agirlikla carpilir; eksik "
                           "faktorlerin agirligi mevcut faktorlere yeniden dagitilir.",
            "score_range": [0, 100],
            "factors": diagnostics.get("active_factors", []),
            "auto_disabled_factors": diagnostics.get("auto_disabled", []),
            "universe_size": diagnostics.get("universe_size"),
        },
        "results": [],
    }

    for _, r in top.iterrows():
        strengths, weaknesses = [], []
        for fid, fd in (r.get("factors") or {}).items():
            if not fd.get("available"):
                continue
            s = fd["score"]
            entry = {"factor": fid, "name_tr": fd["name_tr"], "score": s,
                     "weight": fd["weight"], "contribution": fd["contribution"]}
            if s >= 70:
                strengths.append(entry)
            elif s <= 30:
                weaknesses.append(entry)

        payload["results"].append({
            "rank": int(r["rank"]),
            "ticker": r["ticker"],
            "name": r.get("name"),
            "sector": r.get("sector"),
            "price": r.get("price"),
            "total_score": r.get("total_score"),
            "coverage": r.get("coverage"),
            "low_confidence": bool(r.get("low_confidence")),
            "category_scores": r.get("category_scores"),
            "trailing_returns": r.get("returns"),
            "strengths": sorted(strengths, key=lambda x: -(x["contribution"] or 0))[:5],
            "weaknesses": sorted(weaknesses, key=lambda x: (x["contribution"] or 0))[:5],
            "penalties": r.get("penalties_hit"),
        })

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
