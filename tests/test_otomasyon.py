"""Gunluk otomasyonun dayanikliligi.

OLAY (15-17.08.2026): ogrenme adimindaki kozmetik bir print satiri istisna
atti. O gune ait tarama BASARIYLA bitmisti -- 2390 hisse skorlanmis, pano
uretilmisti -- ama surec 1 ile ciktigi icin gun isaretlenmedi ve sekiz tetigin
hepsi ayni isi bastan yapti. Uc gun boyunca gunluk is hic "tamamlandi"
sayilmadi.

Buradaki tek soru su: bir adimin cokusu, digerlerinin urettigi degeri
silebiliyor mu?

Calistir:  python tests/test_otomasyon.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import run as cli                          # noqa: E402

fails = 0


def check(name, cond, detail=""):
    global fails
    if cond:
        print(f"  PASS  {name}" + (f"  {detail}" if detail else ""))
    else:
        print(f"  FAIL  {name}  {detail}")
        fails += 1


def boom():
    raise UnicodeEncodeError("charmap", "x", 0, 1, "gercek olayin hatasi")


print("=" * 70)
print("GUNLUK IS - ADIM YALITIMI")
print("=" * 70)

# --- Ikincil adim coktugunde gun basarisiz OLMAZ
degraded: list[str] = []
rc = cli.run_stage("ogrenme dongusu", boom, essential=False, degraded=degraded)
check("ikincil adimin cokusu gunu dusurmez", rc == 0, f"rc={rc}")
check("cokme raporlanir", degraded == ["ogrenme dongusu"], str(degraded))

# --- Zorunlu adim coktugunde gun basarisiz OLUR
degraded = []
rc = cli.run_stage("tarama", boom, essential=True, degraded=degraded)
check("zorunlu adimin cokusu gunu dusurur", rc == 1, f"rc={rc}")
check("zorunlu cokme de raporlanir", degraded == ["tarama"], str(degraded))

# --- Basarili adim degeri gecirir
degraded = []
rc = cli.run_stage("tarama", lambda: 0, essential=True, degraded=degraded)
check("basarili adim 0 dondurur", rc == 0 and not degraded)

rc = cli.run_stage("tarama", lambda: 3, essential=True, degraded=degraded)
check("adimin kendi cikis kodu korunur", rc == 3, f"rc={rc}")
check("cikis kodu 0 degil diye 'cokme' sayilmaz", not degraded, str(degraded))

# --- None donduren adim (cogu komut boyle) 0 sayilir
rc = cli.run_stage("izleme listesi", lambda: None, essential=False, degraded=degraded)
check("None dondurmek 0 sayilir", rc == 0, f"rc={rc}")

# --- KeyboardInterrupt yutulmaz: kullanici durdurdugunda gercekten dursun
try:
    cli.run_stage("tarama", lambda: (_ for _ in ()).throw(KeyboardInterrupt()),
                  essential=True, degraded=degraded)
    caught = False
except KeyboardInterrupt:
    caught = True
check("KeyboardInterrupt yutulmaz", caught)

print()
if fails:
    print(f"{fails} KONTROL BASARISIZ")
    raise SystemExit(1)
print("TUM OTOMASYON TESTLERI GECTI")
