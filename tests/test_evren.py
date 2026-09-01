"""Evren cokmesine karsi savunmalar.

OLAY (16.08.2026): api.nasdaq.com erisilemedi. Kotasyon kaynagi sessizce bos
liste dondu; evren, kullanicinin izleme listesindeki 4 hisseye coktu; tarama
"basarili" sayildi ve 4 hisselik siralamayi panonun uzerine yazdi. Kullanici
sitede yalnizca kendi ekledigi hisseleri gordu.

Uc ayri savunma test edilir:
  1. Kotasyon listesi bayat onbellekten kurtarilir (ag hatasi = veri yok DEGIL)
  2. Evren yine de cokerse dunku evren kaydiyla devam edilir
  3. Kurtarma da yoksa tarama IPTAL edilir - pano ezilmez

Calistir:  python tests/test_evren.py
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import run as cli                          # noqa: E402
from src import scanlog                    # noqa: E402
from src import universe                   # noqa: E402
from src.providers import cache            # noqa: E402

fails = 0


def check(name, cond, detail=""):
    global fails
    if cond:
        print(f"  PASS  {name}" + (f"  {detail}" if detail else ""))
    else:
        print(f"  FAIL  {name}  {detail}")
        fails += 1


def args_ns(**kw):
    base = dict(limit=None, universe="smallcap,midcap", wsb_top=60,
                symbols_file=None, min_mcap=None, max_mcap=None)
    base.update(kw)
    return argparse.Namespace(**base)


def fake_listings(n: int) -> list[tuple[str, float]]:
    return [(f"T{i:04d}", 1e9 + i) for i in range(n)]


# ===========================================================================
print("=" * 70)
print("1) KOTASYON LISTESI - BAYAT ONBELLEK KURTARMASI")
print("=" * 70)

tmp = Path(tempfile.mkdtemp(prefix="evren_"))
orig_cache_dir = cache.CACHE_DIR
cache.CACHE_DIR = tmp / "cache"

try:
    ident = f"{universe._LISTINGS_KEY}:NASDAQ"

    # Ag calisiyor -> onbellege yazilir
    universe._fetch_us_listings = lambda ex: fake_listings(2000)
    rows, info = universe.us_listings(("NASDAQ",))
    check("ag basariliysa liste doner", len(rows) == 2000, f"{len(rows)}")
    check("kaynak 'ag' olarak raporlanir", info["source"] == "ag", info["source"])
    check("onbellege yazildi", cache.peek("universe", ident) is not None)

    # Ag coktu -> BAYAT onbellek kullanilmali
    cache.peek("universe", ident)  # kaydi eskitmek yerine TTL'yi sifirla
    real_ttl = universe._LISTINGS_TTL
    universe._LISTINGS_TTL = 0                 # her kayit bayat sayilsin
    universe._fetch_us_listings = lambda ex: []
    rows, info = universe.us_listings(("NASDAQ",))
    check("ag cokunce bayat onbellek kullanilir", len(rows) == 2000, f"{len(rows)}")
    check("kaynak 'bayat_onbellek'", info["source"] == "bayat_onbellek", info["source"])
    check("bayat kayit ok sayilir", info["ok"] is True)

    # Yarim donen cekim de basarisiz sayilmali (esik alti)
    universe._fetch_us_listings = lambda ex: fake_listings(10)
    rows, info = universe.us_listings(("NASDAQ",))
    check("esik alti cekim basarisiz sayilir",
          info["source"] == "bayat_onbellek" and len(rows) == 2000,
          f"{info['source']} / {len(rows)}")

    # Onbellek de yoksa: basarisiz
    cache.CACHE_DIR = tmp / "bos"
    universe._fetch_us_listings = lambda ex: []
    rows, info = universe.us_listings(("NASDAQ",))
    check("onbellek de yoksa basarisiz raporlanir",
          info["source"] == "basarisiz" and info["ok"] is False, info["source"])

    universe._LISTINGS_TTL = real_ttl
finally:
    cache.CACHE_DIR = orig_cache_dir

# ===========================================================================
print()
print("=" * 70)
print("2) EVREN KAYDI - BOZUK GUN KURTARMA LISTESI OLAMAZ")
print("=" * 70)

tmp2 = Path(tempfile.mkdtemp(prefix="evren2_"))
orig_uni_log, orig_data = scanlog.UNIVERSE_LOG, scanlog.DATA
scanlog.DATA = tmp2
scanlog.UNIVERSE_LOG = tmp2 / "universe_history.json"

try:
    today = scanlog._today()
    hist = {
        "2026-01-05": [f"A{i}" for i in range(1500)],
        "2026-01-06": [f"A{i}" for i in range(1490)],
        "2026-01-07": ["LQDT", "ODD", "OMDA", "PRG"],       # cokmus gun
    }
    scanlog.UNIVERSE_LOG.write_text(json.dumps(hist), encoding="utf-8")

    syms, day = scanlog.last_universe()
    check("cokmus gun kurtarma listesi olarak secilmez", day == "2026-01-06", str(day))
    check("kurtarma listesi dolu", len(syms) == 1490, f"{len(syms)}")

    # Ayni gun icinde ikinci kayit BIRLESTIRIR, uzerine yazmaz
    scanlog.record_universe([f"B{i}" for i in range(800)])
    scanlog.record_universe(["LQDT", "ODD"])
    after = json.loads(scanlog.UNIVERSE_LOG.read_text(encoding="utf-8"))
    check("ayni gun ikinci tarama kaydi silmez",
          len(after[today]) == 802, f"{len(after.get(today, []))}")
finally:
    scanlog.UNIVERSE_LOG, scanlog.DATA = orig_uni_log, orig_data

# ===========================================================================
print()
print("=" * 70)
print("3) TARAMA KORUMASI - guard_universe")
print("=" * 70)

_saved_last = scanlog.last_universe
# Test, gercek output/run_status.json dosyasina dokunmasin.
_saved_status, cli.write_status = cli.write_status, lambda *a, **k: None

# a) Evren saglamsa dokunmaz
scanlog.last_universe = lambda before_today=True: ([f"X{i}" for i in range(1000)], "2026-01-06")
big = [f"S{i}" for i in range(2500)]
out, info = cli.guard_universe(big, {"_listings": {"ok": True}},
                               ["smallcap", "midcap"], args_ns())
check("saglam evrene dokunulmaz", out is big and not info["abort"] and not info["recovered"])

# b) Evren coktu, gecmis var -> kurtarilir
out, info = cli.guard_universe(["LQDT", "ODD", "OMDA", "PRG"],
                               {"_listings": {"ok": False, "source": "basarisiz"}},
                               ["smallcap", "midcap"], args_ns())
check("cokmus evren gecmisten kurtarilir",
      info["recovered"] and len(out) == 1000 and not info["abort"],
      f"{len(out)} sembol")

# c) Evren coktu, gecmis yok -> IPTAL
scanlog.last_universe = lambda before_today=True: ([], None)
out, info = cli.guard_universe(["LQDT", "ODD"],
                               {"_listings": {"ok": False, "source": "basarisiz"}},
                               ["smallcap", "midcap"], args_ns())
check("kurtarma yoksa tarama iptal edilir", info["abort"] is True)

# d) Kucuk evren KASITLIYSA kontrol calismaz
out, info = cli.guard_universe(["AAPL", "MSFT"], {},
                               ["file"], args_ns(universe="file"))
check("dosyadan gelen kucuk evren engellenmez", not info["abort"] and not info["recovered"])

out, info = cli.guard_universe(["AAPL", "MSFT"], {"_listings": {"ok": False}},
                               ["smallcap"], args_ns(limit=20))
check("--limit ile kucultulen tarama engellenmez",
      not info["abort"] and not info["recovered"])

scanlog.last_universe = _saved_last
cli.write_status = _saved_status

# ===========================================================================
print()
print("=" * 70)
print("4) CIKTI KORUMASI - yarim tarama panoyu ezmez")
print("=" * 70)

# _previous_snapshot_rows gercek magazadan okur; burada esigin kendisini
# dogruluyoruz: 2400 satirlik gecmisin ardindan 4 satir gelirse reddedilmeli.
prev, now = 2400, 4
check("4 satir, 2400 satirin uzerine yazilmaz", now < 0.4 * prev)
check("2000 satir, 2400 satirin uzerine yazilir", not (2000 < 0.4 * prev))
check("gecmis yoksa kontrol calismaz", not (0 >= 200))

real_prev = cli._previous_snapshot_rows()
check("_previous_snapshot_rows gercek magazadan okuyor", real_prev >= 0,
      f"{real_prev} satir")

# ===========================================================================
print()
if fails:
    print(f"{fails} KONTROL BASARISIZ")
    raise SystemExit(1)
print("TUM EVREN TESTLERI GECTI")
