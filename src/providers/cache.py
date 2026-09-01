"""Basit disk onbellegi — ayni gun icinde tekrar cagrilarda API'yi yormamak icin."""
from __future__ import annotations

import hashlib
import json
import pickle
import time
from pathlib import Path
from typing import Any, Callable

CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "cache"


def _key(namespace: str, ident: str) -> Path:
    h = hashlib.sha1(f"{namespace}:{ident}".encode()).hexdigest()[:20]
    return CACHE_DIR / namespace / f"{h}.pkl"


def get_or_fetch(namespace: str, ident: str, fetch: Callable[[], Any],
                 ttl_seconds: int = 6 * 3600, enabled: bool = True,
                 should_cache: Callable[[Any], bool] | None = None) -> Any:
    """Onbellekte taze kayit varsa onu dondur, yoksa fetch() calistir ve sakla.

    should_cache: sonucu saklamadan once kontrol eder. BASARISIZ CEKIMLERI
    ONBELLEGE YAZMAMAK icin kritik — gecici bir hiz siniri hatasi aksi halde
    saatlerce "veri yok" olarak donerdi.
    """
    if not enabled:
        return fetch()

    path = _key(namespace, ident)
    if path.exists():
        try:
            if time.time() - path.stat().st_mtime < ttl_seconds:
                with path.open("rb") as fh:
                    return pickle.load(fh)
        except Exception:
            pass  # bozuk onbellek -> yeniden cek

    value = fetch()
    # Savunma: None ASLA onbellege yazilmaz. Basarisiz bir cekim onbellege
    # dusmus olsaydi, TTL boyunca "veri yok" olarak donerdi.
    if value is None:
        return None
    if should_cache is not None and not should_cache(value):
        return value
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump(value, fh)
    except Exception:
        pass
    return value


def peek(namespace: str, ident: str) -> tuple[Any, float] | None:
    """Onbellekteki kaydi YASINDAN BAGIMSIZ dondurur: (deger, yas_saniye).

    get_or_fetch tazelik esigini gecmeyen kaydi yok sayar. Ama bazi veriler
    icin bayat kayit, HIC KAYIT OLMAMASINDAN cok daha iyidir: borsa kotasyon
    listesi gunluk birkac sembol degisir, dunun listesi bugun de calisir.
    api.nasdaq.com erisilemedigi bir gunde bu fonksiyon olmadigi icin evren
    bos donuyor ve tarama, kullanicinin izleme listesindeki 4 hisseye
    cokuyordu (pano da o 4 hisseyle yeniden yaziliyordu).

    Doner: (deger, yas) veya kayit yoksa None.
    """
    path = _key(namespace, ident)
    if not path.exists():
        return None
    try:
        age = time.time() - path.stat().st_mtime
        with path.open("rb") as fh:
            return pickle.load(fh), age
    except Exception:
        return None


def put(namespace: str, ident: str, value: Any) -> None:
    """Onbellege dogrudan yazar (peek ile birlikte kullanilir)."""
    if value is None:
        return
    path = _key(namespace, ident)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump(value, fh)
    except Exception:
        pass


def clear(namespace: str | None = None) -> int:
    root = CACHE_DIR / namespace if namespace else CACHE_DIR
    if not root.exists():
        return 0
    n = 0
    for p in root.rglob("*.pkl"):
        p.unlink()
        n += 1
    return n


def purge_invalid(namespace: str = "yahoo") -> int:
    """Bos/bozuk onbellek kayitlarini siler.

    Sadece bir temizlik degil: boyle bir kayit varken ilgili hisse TTL boyunca
    "veri yok" kabul edilir ve sessizce taramanin disinda kalir.
    """
    root = CACHE_DIR / namespace
    if not root.exists():
        return 0

    removed = 0
    for p in root.rglob("*.pkl"):
        drop = False
        try:
            with p.open("rb") as fh:
                v = pickle.load(fh)
            if v is None:
                drop = True
            elif isinstance(v, dict):
                h = v.get("history")
                drop = h is None or len(h) < 30
        except Exception:
            drop = True          # bozuk dosya
        if drop:
            try:
                p.unlink()
                removed += 1
            except OSError:
                pass
    return removed
