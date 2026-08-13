"""Egitim veri kumesi — panel kurulumu, etiketleme ve SIZINTISIZ bolme.

Bu dosyanin en onemli isi bolme (split) mantigidir. Finansal panel verisinde
naif capraz dogrulama, modeli oldugundan iyi gosterir; sebebi uc tanedir:

1. ORTUSEN ETIKETLER
   21 gunluk ileri getiri, ardisik gunlerin etiketlerini %95 ortusturur.
   t gunu icin etiket t+21'e kadar olan fiyati kullanir. Egitim setinde
   t, test setinde t+3 varsa, egitim etiketinin icinde test doneminin
   fiyati vardir -> dogrudan sizinti.
   COZUM: purge (arindirma) — test baslangicindan geriye dogru `horizon`
   kadar gunluk egitim verisi atilir.

2. SERI KORELASYON
   Arindirma sonrasi bile sinir bolgesindeki ornekler benzer piyasa
   rejimini paylasir.
   COZUM: embargo — arindirmanin ustune ek tampon gun.

3. GELECEGE BAKIS
   Ozellikler o gun BILINEBILIR olmali. Feature store zaten her gunun
   anlik goruntusunu ayri sakladigi icin bu saglaniyor.

Ayrica capraz kesitsel calisildigi icin etiket olarak HAM getiri degil,
o gunun evren ortalamasindan ARINDIRILMIS getiri kullanilir; boylece model
piyasa yonunu degil, GORELI performansi ogrenir.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

from .ml import FEATURE_STORE

# Ozellik olarak KULLANILMAYACAK sutunlar (kimlik, hedef, sizinti riski)
_EXCLUDE = {
    "snapshot_date", "ticker", "sector", "industry", "rank",
    "total_score", "base_score",          # skorun kendisi hedefle donguye girer
    "price", "market_cap",                # olcek; log'u ayrica turetilir
}


@dataclass
class Panel:
    """Egitime hazir panel veri."""
    X: np.ndarray                  # (n, f) ozellikler
    y: np.ndarray                  # (n,)   hedef (arindirilmis ileri getiri)
    dates: np.ndarray              # (n,)   snapshot tarihi
    tickers: np.ndarray            # (n,)
    feature_names: list[str] = field(default_factory=list)
    horizon: int = 21

    def __len__(self) -> int:
        return len(self.y)

    @property
    def unique_dates(self) -> np.ndarray:
        return np.array(sorted(set(self.dates)))


# =============================================================================
#  Panel kurulumu
# =============================================================================
def load_panel(horizon: int = 21, min_rows_per_date: int = 30,
               use_cache: bool = True, label_col: str | None = None
               ) -> tuple[Panel | None, dict]:
    """Tum anlik goruntuleri birlestirip etiketli panel uretir.

    Doner: (Panel | None, tani sozlugu)
    Etiketlenebilir satir yoksa Panel None doner — bu bir hata degil, sistemin
    henuz yeterince beklememis olmasidir.
    """
    from .ml import label_forward_returns, load_all_snapshots

    snaps = load_all_snapshots()
    info: dict = {"snapshots": 0, "rows": 0, "labeled": 0, "features": 0}
    if snaps.empty:
        info["reason"] = "feature store bos"
        return None, info

    dates = sorted(snaps["snapshot_date"].dropna().astype(str).unique())
    info["snapshots"] = len(dates)
    info["rows"] = len(snaps)
    info["first_date"], info["last_date"] = dates[0], dates[-1]

    labeled = label_forward_returns(snaps, horizon_days=horizon, use_cache=use_cache)
    col = label_col or f"fwd_return_{horizon}d_excess"
    if col not in labeled.columns:
        info["reason"] = f"etiket sutunu yok: {col}"
        return None, info

    df = labeled[labeled[col].notna()].copy()
    info["labeled"] = len(df)
    if df.empty:
        info["reason"] = ("hicbir satirin ufku dolmadi — en eski anlik goruntu "
                          f"{horizon} islem gununden yeni")
        return None, info

    # Cok az satirli gunler capraz kesitsel siralamayi anlamsiz kilar
    counts = df.groupby("snapshot_date")[col].transform("size")
    df = df[counts >= min_rows_per_date]
    if df.empty:
        info["reason"] = f"hicbir gunde {min_rows_per_date}+ etiketli hisse yok"
        return None, info

    feats = _feature_columns(df)
    if not feats:
        info["reason"] = "kullanilabilir ozellik sutunu yok"
        return None, info

    X = df[feats].to_numpy(dtype=np.float32)
    # Eksik degerler: 'has_*' sutunlari zaten eksikligi kodluyor, bu yuzden
    # sayisal doldurma notr degerle yapilir (skorlar 0-100 olcekli -> 50).
    X = _fill_missing(X, feats)

    panel = Panel(
        X=X,
        y=df[col].to_numpy(dtype=np.float32),
        dates=df["snapshot_date"].astype(str).to_numpy(),
        tickers=df["ticker"].astype(str).to_numpy(),
        feature_names=feats,
        horizon=horizon,
    )
    info["features"] = len(feats)
    info["dates_usable"] = int(df["snapshot_date"].nunique())
    return panel, info


def _feature_columns(df: pd.DataFrame) -> list[str]:
    """Sayisal, hedefe sizmayan ozellik sutunlari."""
    out = []
    for c in df.columns:
        if c in _EXCLUDE or c.startswith("fwd_return"):
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        # Tamamen bos veya tek degerli sutun bilgi tasimaz
        s = df[c]
        if s.notna().sum() == 0 or s.nunique(dropna=True) <= 1:
            continue
        out.append(c)
    return sorted(out)


def _fill_missing(X: np.ndarray, names: list[str]) -> np.ndarray:
    X = X.copy()
    for j, name in enumerate(names):
        col = X[:, j]
        bad = ~np.isfinite(col)
        if not bad.any():
            continue
        if name.startswith("score_"):
            col[bad] = 50.0                    # notr yuzdelik
        elif name.startswith("has_") or name.startswith("penalty_"):
            col[bad] = 0.0
        else:
            med = np.nanmedian(col[np.isfinite(col)]) if np.isfinite(col).any() else 0.0
            col[bad] = med
        X[:, j] = col
    return X


# =============================================================================
#  Capraz kesitsel normalizasyon
# =============================================================================
def cross_sectional_rank(X: np.ndarray, dates: np.ndarray) -> np.ndarray:
    """Her ozelligi GUN ICINDE yuzdelik siraya cevirir (-1..+1).

    Neden: model, "F/K 12'dir" gibi mutlak seviyeleri degil, "bu hisse bugun
    digerlerine gore ucuz" bilgisini ogrenmeli. Ayrica rejim degisimlerinde
    (tum piyasanin carpanlari sistematik kaydiginda) ozellik dagilimi sabit
    kalir — bu, modelin zaman icinde bozulmasini yavaslatir.
    """
    out = np.zeros_like(X, dtype=np.float32)
    for d in np.unique(dates):
        m = dates == d
        block = X[m]
        n = block.shape[0]
        if n < 2:
            out[m] = 0.0
            continue
        order = block.argsort(axis=0).argsort(axis=0).astype(np.float32)
        out[m] = 2.0 * (order / max(1, n - 1)) - 1.0
    return out


def demean_by_date(y: np.ndarray, dates: np.ndarray) -> np.ndarray:
    """Hedefi gun icinde ortalamadan arindirir (piyasa yonunu cikarir)."""
    out = y.astype(np.float32).copy()
    for d in np.unique(dates):
        m = dates == d
        out[m] -= out[m].mean()
    return out


# =============================================================================
#  SIZINTISIZ ILERI YURUYUSLU BOLME
# =============================================================================
def walk_forward_splits(dates: np.ndarray, horizon: int, n_splits: int = 5,
                        embargo: int = 5, min_train_dates: int = 20
                        ) -> Iterator[tuple[np.ndarray, np.ndarray, dict]]:
    """Arindirmali + embargolu ileri yuruyuslu bolme.

    Her katman icin:
        egitim  = baslangictan (test_basi - horizon - embargo) tarihine kadar
        test    = ardisik bir tarih blogu

    `horizon` kadar geri arindirma SART: 21 gunluk etiket, test doneminin
    fiyatlarini iceren egitim orneklerini disarida birakir.
    """
    uniq = np.array(sorted(set(dates)))
    n = len(uniq)
    if n < min_train_dates + n_splits + horizon:
        return

    # Test bloklarini sona esit araliklarla yerlestir
    test_size = max(1, (n - min_train_dates) // (n_splits + 1))
    for k in range(n_splits):
        test_start_i = n - (n_splits - k) * test_size
        test_end_i = test_start_i + test_size
        if test_start_i <= min_train_dates:
            continue

        # Arindirma: test baslangicindan `horizon + embargo` gun geri
        train_end_i = test_start_i - horizon - embargo
        if train_end_i < min_train_dates:
            continue

        train_dates = set(uniq[:train_end_i])
        test_dates = set(uniq[test_start_i:test_end_i])
        if not train_dates or not test_dates:
            continue

        tr = np.array([d in train_dates for d in dates])
        te = np.array([d in test_dates for d in dates])
        if tr.sum() < 100 or te.sum() < 20:
            continue

        yield tr, te, {
            "fold": k + 1,
            "train_dates": len(train_dates),
            "test_dates": len(test_dates),
            "train_rows": int(tr.sum()),
            "test_rows": int(te.sum()),
            "purged_dates": horizon + embargo,
            "train_end": str(uniq[train_end_i - 1]),
            "test_start": str(uniq[test_start_i]),
        }


# =============================================================================
#  Dizi (sequence) verisi — derin ogrenmenin gercekten katki verdigi yer
# =============================================================================
def build_sequences(panel: Panel, window: int = 10, min_len: int | None = None
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Her (tarih, hisse) icin son `window` gunun ozellik dizisini kurar.

    Capraz kesitsel bir model yalnizca "bugun neye benziyor" sorusunu gorur.
    Dizi modeli "son 10 gunde NASIL DEGISTI" sorusunu da gorur — momentumun
    hizlanmasi, skorun bozulmasi, kirilim kurulumunun olgunlasmasi gibi
    orunuler ancak burada ogrenilebilir. Derin ogrenmenin bu problemde
    tablo modellerine gore gercek avantaji budur.

    Doner: (Xs (n, window, f), y, dates, tickers)
    """
    min_len = min_len or window
    order = np.lexsort((panel.dates, panel.tickers))
    X, y = panel.X[order], panel.y[order]
    dates, tickers = panel.dates[order], panel.tickers[order]

    seqs, ys, ds, ts = [], [], [], []
    start = 0
    for i in range(1, len(tickers) + 1):
        if i == len(tickers) or tickers[i] != tickers[start]:
            block_X, block_y = X[start:i], y[start:i]
            block_d = dates[start:i]
            for j in range(len(block_X)):
                lo = j - window + 1
                if lo < 0:
                    if j + 1 < min_len:
                        continue
                    pad = np.repeat(block_X[0:1], window - (j + 1), axis=0)
                    seq = np.concatenate([pad, block_X[: j + 1]], axis=0)
                else:
                    seq = block_X[lo: j + 1]
                seqs.append(seq)
                ys.append(block_y[j])
                ds.append(block_d[j])
                ts.append(tickers[start])
            start = i

    if not seqs:
        empty_f = panel.X.shape[1]
        return (np.zeros((0, window, empty_f), np.float32), np.zeros(0, np.float32),
                np.zeros(0, dtype=object), np.zeros(0, dtype=object))

    return (np.asarray(seqs, dtype=np.float32), np.asarray(ys, dtype=np.float32),
            np.asarray(ds), np.asarray(ts))


# =============================================================================
#  Hazirlik kapisi
# =============================================================================
def readiness(horizon: int = 21) -> dict:
    """Egitim icin yeterli veri var mi? Yoksa NE KADAR eksik?

    Bu kapi bilincli olarak katidir. Az veriyle egitilen bir model, guvenilir
    gorunen ama tamamen gurultuye uydurulmus tahminler uretir — sistemin
    en buyuk riski budur.
    """
    if not FEATURE_STORE.exists():
        snaps, span = 0, 0
        dates: list[str] = []
    else:
        dates = sorted({p.stem.replace("snapshot_", "")[:10]
                        for p in FEATURE_STORE.glob("snapshot_*.csv")})
        snaps = len(dates)
        span = 0
        if snaps >= 2:
            span = (pd.Timestamp(dates[-1]) - pd.Timestamp(dates[0])).days

    # Esikler: capraz kesitsel bir modelin anlamli sekilde dogrulanabilmesi icin
    # en az birkac bagimsiz (ortusmeyen) test penceresi gerekir.
    need_snaps = 60          # ~3 ay gunluk tarama
    need_span = 120          # takvim gunu
    labelable = max(0, span - horizon)

    ready_train = snaps >= 30 and span >= horizon + 20
    ready_validate = snaps >= need_snaps and span >= need_span

    return {
        "snapshots": snaps,
        "span_days": span,
        "first_date": dates[0] if dates else None,
        "last_date": dates[-1] if dates else None,
        "horizon": horizon,
        "labelable_days": labelable,
        "ready_to_train": ready_train,
        "ready_to_validate": ready_validate,
        "need_snapshots": need_snaps,
        "need_span_days": need_span,
        "progress_pct": round(100 * min(1.0, min(snaps / need_snaps,
                                                 span / need_span)), 1),
        "missing_snapshots": max(0, need_snaps - snaps),
        "missing_days": max(0, need_span - span),
    }
