"""Kote disi kalan sirketlerin takibi — hayatta kalma yanliligina karsi.

SORUN (denetim bulgusu Y3)
-------------------------
Evren, bugun kote olan sirketlerden olusuyor. Batmis, satin alinmis veya kote
disi kalmis sirketler hicbir zaman gorunmuyor. Tarama icin bu dogru davranis
(bugun alamayacagin hisseyi siralamanin anlami yok) ama OLCUM icin olumcul:
ileriye donuk getiri hesaplayan her sey -- IC, kagit defter, model egitimi --
yalnizca hayatta kalanlari gorur ve sistematik olarak YUKARI YANLI sonuc
uretir. Dahasi bunu fark edemez, cunku kaybolan sirketi hic gormez.

ONEMLI AYRIM
------------
"Evrenden dustu" ile "kote disi kaldi" AYNI SEY DEGILDIR. Evren piyasa degeri
bandiyla (300M - 20Mr) sinirli; degeri bandin ustune cikan bir sirket de
listeden duser, ama o sirket batmadi -- tam tersine cok iyi performans
gosterdi. Bu ikisi karistirilirsa yanlilik duzelmez, TERS YONE cevrilir.

Bu yuzden kontrol, band suzgecinden gecmis evrene degil, KOTASYON BESLEMESININ
TAMAMINA karsi yapilir: sembol borsa listesinin tamaminda yoksa kote disi
adayidir.

KESINLIK
--------
Tek bir gunun eksigi kanit degil (besleme hatasi olabilir). Bir sembol
`CONFIRM_DAYS` ayri tarama gununde listede gorunmezse "kote disi" sayilir.
Son bilinen fiyati da saklanir; boylece pozisyon o fiyattan kapatilabilir.

DURUSTLUK NOTU
--------------
Son bilinen fiyattan kapatmak IYIMSER bir varsayimdir: iflasta hisse cogu
zaman o fiyattan cok altina, bazen sifira gider. Ama kaydi tamamen SILMEKTEN
cok daha iyidir; silmek yanliligin ta kendisidir. Ozet ciktilarinda kac
pozisyonun boyle kapandigi ayrica raporlanir.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"
LEDGER = DATA / "kote_disi.json"

# Kac ayri tarama gununde gorunmezse "kote disi" sayilsin
CONFIRM_DAYS = 5


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def load() -> dict:
    if not LEDGER.exists():
        return {"symbols": {}, "listed_seen": {}, "updated_at": None}
    try:
        d = json.loads(LEDGER.read_text(encoding="utf-8")) or {}
    except Exception:
        return {"symbols": {}, "listed_seen": {}, "updated_at": None}
    d.setdefault("symbols", {})
    d.setdefault("listed_seen", {})
    return d


def _save(d: dict) -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    d["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    LEDGER.write_text(json.dumps(d, ensure_ascii=False, indent=0, sort_keys=True),
                      encoding="utf-8")


def _cached_last_close(ticker: str) -> float | None:
    """Onbellekteki son kapanis. Ag istegi YAPMAZ (sembol zaten kayboldu)."""
    try:
        from .providers import yahoo
        b = yahoo.fetch_cached(ticker, "2y", max_age_seconds=60 * 24 * 3600)
        h = (b or {}).get("history")
        if h is None or not len(h):
            return None
        return round(float(h["Close"].iloc[-1]), 4)
    except Exception:
        return None


def update(all_listed: set[str] | list[str], last_prices: dict[str, float] | None = None
           ) -> dict:
    """Kotasyon beslemesinin TAMAMIYLA karsilastirir ve defteri gunceller.

    all_listed: band suzgecinden GECMEMIS, borsadaki tum semboller.
    last_prices: sembol -> son bilinen kapanis (varsa saklanir).
    """
    listed = {str(s).upper() for s in all_listed}
    if len(listed) < 500:
        # Besleme yarim geldi; bu veriyle "kayboldu" karari verilemez.
        return {"ok": False, "reason": "kotasyon beslemesi yetersiz",
                "listed": len(listed)}

    d = load()
    today = _today()
    seen: dict[str, str] = d["listed_seen"]
    syms: dict[str, dict] = d["symbols"]
    last_prices = last_prices or {}

    known = set(seen)
    # 1) Bugun listede olanlarin son gorulme tarihini tazele
    for s in listed:
        seen[s] = today
        rec = syms.get(s)
        if rec and not rec.get("confirmed"):
            syms.pop(s, None)                 # geri geldi: yanlis alarm
        elif rec and rec.get("confirmed"):
            rec["returned_at"] = today        # yeniden kote olmus (nadir)
            rec["confirmed"] = False

    # 2) Daha once gorulmus ama bugun listede olmayanlar
    newly, confirmed = [], []
    for s in known - listed:
        rec = syms.setdefault(s, {"last_seen": seen.get(s), "missing_days": 0})
        # Ayni gun iki kez calisirsa sayaci iki kez artirma
        if rec.get("last_check") == today:
            continue
        rec["last_check"] = today
        rec["missing_days"] = int(rec.get("missing_days", 0)) + 1
        if s in last_prices and last_prices[s]:
            rec.setdefault("last_price", float(last_prices[s]))
        if rec["missing_days"] == 1:
            newly.append(s)
            # Kaybolan sembolun son fiyatini SIMDI yakala: onbellek kaydi
            # zamanla temizlenir, sonra bir daha bulunamaz. Pozisyonun hangi
            # fiyattan kapatilacagini belirleyen tek bilgi budur.
            if "last_price" not in rec:
                px = _cached_last_close(s)
                if px:
                    rec["last_price"] = px
        if rec["missing_days"] >= CONFIRM_DAYS and not rec.get("confirmed"):
            rec["confirmed"] = True
            rec["confirmed_at"] = today
            confirmed.append(s)

    d["listed_seen"] = seen
    d["symbols"] = syms
    _save(d)

    return {"ok": True, "listed": len(listed),
            "missing_today": len(known - listed),
            "newly_missing": sorted(newly)[:50],
            "newly_confirmed": sorted(confirmed),
            "confirmed_total": sum(1 for r in syms.values() if r.get("confirmed"))}


def confirmed() -> dict[str, dict]:
    """Kote disi oldugu KESINLESMIS semboller: sembol -> kayit."""
    return {s: r for s, r in load()["symbols"].items() if r.get("confirmed")}


def is_delisted(ticker: str) -> bool:
    return bool(load()["symbols"].get(str(ticker).upper(), {}).get("confirmed"))


def last_price(ticker: str) -> float | None:
    r = load()["symbols"].get(str(ticker).upper(), {})
    v = r.get("last_price")
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def info() -> dict:
    d = load()
    conf = [r for r in d["symbols"].values() if r.get("confirmed")]
    pending = [r for r in d["symbols"].values()
               if not r.get("confirmed") and r.get("missing_days")]
    return {
        "tracked": len(d["listed_seen"]),
        "confirmed": len(conf),
        "pending": len(pending),
        "confirm_days": CONFIRM_DAYS,
        "updated_at": d.get("updated_at"),
    }
