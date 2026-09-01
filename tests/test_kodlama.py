"""Kaynak dosyalarin kodlama saglami.

Bu testin varlik sebebi somut bir olaydir: run.py icine bir onarim sirasinda
U+FFFD (degistirme karakteri) sizdi. Tek bir karakter, gunluk otomasyonu uc gun
boyunca her calismada dusurdu -- tarama basariyla bitiyor, sonra kozmetik bir
print satiri UnicodeEncodeError atiyor, cikis kodu 1 oluyor, gun isaretlenmiyor
ve sekiz tetigin hepsi ayni isi bastan yapiyordu. Hata gozle gorulmuyordu cunku
karakter terminalde de editorde de bir soru isareti gibi duruyor.

Kontroller:
  1. Hicbir kaynak dosyada U+FFFD olmayacak (bozuk kodlama donusumunun izi).
  2. .bat dosyalari saf ASCII olacak (cmd.exe kod sayfasi ongorulemez).
  3. .py dosyalarinda BOM olmayacak (Python kabul eder, bazi araclar etmez).
  4. Kaynak dosyalar gecerli UTF-8 olacak.

Calistir:  python tests/test_kodlama.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Uretilen / disaridan gelen dizinler denetlenmez
SKIP_DIRS = {".git", "node_modules", "__pycache__", "data", "output",
             "publish", "logs", ".venv", "venv"}

TEXT_SUFFIXES = {".py", ".md", ".yaml", ".yml", ".bat", ".txt", ".json"}

fails = 0


def check(name, cond, detail=""):
    global fails
    if cond:
        print(f"  PASS  {name}" + (f"  {detail}" if detail else ""))
    else:
        print(f"  FAIL  {name}  {detail}")
        fails += 1


def source_files() -> list[Path]:
    out = []
    for p in ROOT.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in TEXT_SUFFIXES:
            continue
        if any(part in SKIP_DIRS for part in p.relative_to(ROOT).parts):
            continue
        out.append(p)
    return sorted(out)


def main() -> int:
    files = source_files()
    print("=" * 70)
    print(f"KODLAMA SAGLAMI  ({len(files)} dosya)")
    print("=" * 70)

    # --- 1) Gecerli UTF-8 mi
    bad_utf8 = []
    for p in files:
        try:
            p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            bad_utf8.append(p.relative_to(ROOT).as_posix())
    check("tum dosyalar gecerli UTF-8", not bad_utf8, ", ".join(bad_utf8[:5]))

    # --- 2) U+FFFD sizintisi
    # NOT: aranan karakter kaynakta duz yazilamaz, yoksa bu dosyanin kendisi
    # testi patlatir. chr() ile uretiliyor.
    replacement_char = chr(0xFFFD)
    hits = []
    for p in files:
        text = p.read_text(encoding="utf-8", errors="replace")
        for i, line in enumerate(text.splitlines(), 1):
            if replacement_char in line:
                hits.append(f"{p.relative_to(ROOT).as_posix()}:{i}")
    check("hicbir dosyada U+FFFD yok", not hits, ", ".join(hits[:5]))

    # --- 3) .bat saf ASCII
    non_ascii_bat = []
    for p in files:
        if p.suffix.lower() != ".bat":
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        for i, line in enumerate(text.splitlines(), 1):
            if any(ord(ch) > 127 for ch in line):
                non_ascii_bat.append(f"{p.relative_to(ROOT).as_posix()}:{i}")
    check(".bat dosyalari saf ASCII", not non_ascii_bat,
          ", ".join(non_ascii_bat[:5]))

    # --- 4) .py dosyalarinda BOM yok
    bom = []
    for p in files:
        if p.suffix.lower() != ".py":
            continue
        if p.read_bytes()[:3] == b"\xef\xbb\xbf":
            bom.append(p.relative_to(ROOT).as_posix())
    check(".py dosyalarinda BOM yok", not bom, ", ".join(bom[:5]))

    # --- 5) run.py konsolu UTF-8'e sabitliyor mu
    run_src = (ROOT / "run.py").read_text(encoding="utf-8")
    check("run.py konsolu UTF-8'e sabitliyor",
          "_force_utf8_console()" in run_src)

    print()
    if fails:
        print(f"{fails} KONTROL BASARISIZ")
        return 1
    print("TUM KODLAMA TESTLERI GECTI")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
