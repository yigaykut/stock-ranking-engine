"""Kagit uzerinde defterin dogrulugu.

Bir karne, yanlis hesaplandiginda en tehlikeli ciktilardan biridir: sayilar
inandirici gorunur ve kimse sorgulamaz. Bu yuzden asagidakiler ayri ayri
dogrulanir:

 1. Getiri, TAKVIM gunu degil ISLEM gunu sayilarak hesaplanmali. Tatiller
    ufku kaydirirsa 21 gunluk sonuc aslinda 15 veya 25 gunluk olur.
 2. Ufku dolmamis pozisyon "sifir getiri" degil BOS olmali; aksi halde
    ortalama sistematik olarak sifira cekilir.
 3. Endeks farki (excess), ham getiriden bagimsiz hesaplanmali.
 4. Kote disi kalan hisse, seri kesildigi icin sonsuza kadar "ufku dolmadi"
    durumunda kalmamali -- hayatta kalma yanliligi tam olarak boyle olusur.
 5. t istatistigi POZISYON degil KOHORT bazinda olmali: ayni gunun 20 hissesi
    bagimsiz gozlem degildir.

Calistir:  python tests/test_karne.py
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import paper                      # noqa: E402

fails = 0


def check(name, cond, detail=""):
    global fails
    if cond:
        print(f"  PASS  {name}" + (f"  {detail}" if detail else ""))
    else:
        print(f"  FAIL  {name}  {detail}")
        fails += 1


# ---------------------------------------------------------------------------
def series(start="2026-01-05", n=200, rate=0.001, gap_after=None):
    """Gunluk kapanis serisi. gap_after verilirse seri orada KESILIR."""
    idx = pd.bdate_range(start, periods=n)
    vals = 100 * np.exp(np.arange(n) * rate)
    s = pd.Series(vals, index=idx)
    return s.iloc[:gap_after] if gap_after else s


print("=" * 70)
print("1) ILERI GETIRI - ISLEM GUNU SAYIMI")
print("=" * 70)

s = series()
d0 = s.index[10]

# 21 islem gunu sonrasi: tam olarak 21 satir ileri
e, x, xd = paper._forward(s, d0, 21)
check("giris fiyati dogru gun", abs(e - float(s.iloc[10])) < 1e-9)
check("cikis fiyati 21 ISLEM gunu sonra",
      abs(x - float(s.iloc[31])) < 1e-9, f"{xd}")
check("cikis tarihi takvimden 21 gun DEGIL",
      (pd.Timestamp(xd) - d0).days > 21,
      f"{(pd.Timestamp(xd) - d0).days} takvim gunu")

# Ufuk seriden tasiyorsa cikis YOK (sifir degil)
e2, x2, _ = paper._forward(s, s.index[-3], 21)
check("ufku dolmayan pozisyon bos doner", e2 is not None and x2 is None)

# horizon=0 giris fiyatini verir (mark() bunu kullaniyor)
e3, x3, _ = paper._forward(s, d0, 0)
check("horizon 0 giris fiyatini verir", abs(e3 - float(s.iloc[10])) < 1e-9)


print()
print("=" * 70)
print("2) DEFTER - YAZMA VE DEGERLEME")
print("=" * 70)

tmp = Path(tempfile.mkdtemp(prefix="karne_"))
_saved = (paper.PAPER, paper.COHORTS, paper.RESULTS, paper.SUMMARY)
paper.PAPER = tmp
paper.COHORTS = tmp / "kohortlar.csv"
paper.RESULTS = tmp / "sonuclar.csv"
paper.SUMMARY = tmp / "ozet.json"

try:
    rank = pd.DataFrame({
        "ticker": ["AAA", "BBB", "CCC", "DDD"],
        "total_score": [90.0, 80.0, 70.0, 60.0],
        "sector": ["Tech", "Tech", "Health", "Health"],
        "price": [10.0, 20.0, 30.0, 40.0],
    })
    r = paper.record_live(rank, top_n=3, date="2026-02-02")
    check("ilk N deftere yaziliyor", r["added"] == 3, str(r))

    # Ayni gun tekrar yazmak SATIR COGALTMAMALI
    paper.record_live(rank, top_n=3, date="2026-02-02")
    coh = pd.read_csv(paper.COHORTS)
    check("ayni gun tekrar yazmak cogaltmiyor", len(coh) == 3, f"{len(coh)} satir")

    # --- Degerleme: AAA yukselen, BBB dusen, CCC kote disi
    up = series(start="2026-01-05", n=120, rate=0.004)
    down = series(start="2026-01-05", n=120, rate=-0.004)
    cut = series(start="2026-01-05", n=120, rate=0.0)[:25]     # 2026-02-06'da kesiliyor
    bench = series(start="2026-01-05", n=120, rate=0.001)

    fake = {"AAA": up, "BBB": down, "CCC": cut, "DDD": up}
    paper._closes = lambda t: fake.get(str(t))
    paper._bench_closes = lambda symbol=paper.BENCHMARK: bench

    class _NoDelist:
        @staticmethod
        def confirmed():
            return {}

    sys.modules["src.delisting"] = _NoDelist            # kote disi YOK senaryosu
    m = paper.mark(horizons=(21,), progress=False)
    check("degerleme calisiyor", m.get("ok"), str(m.get("reason")))

    res = pd.read_csv(paper.RESULTS)
    aaa = res[res["ticker"] == "AAA"].iloc[0]
    bbb = res[res["ticker"] == "BBB"].iloc[0]
    ccc = res[res["ticker"] == "CCC"].iloc[0]
    check("yukselen hisse pozitif", aaa["ret_21"] > 0, f"{aaa['ret_21']:.4f}")
    check("dusen hisse negatif", bbb["ret_21"] < 0, f"{bbb['ret_21']:.4f}")
    check("endeks farki ham getiriden farkli",
          abs(aaa["excess_21"] - aaa["ret_21"]) > 1e-6,
          f"ret {aaa['ret_21']:.4f} vs excess {aaa['excess_21']:.4f}")
    check("endeks farki = getiri - endeks",
          abs((aaa["ret_21"] - aaa["bench_21"]) - aaa["excess_21"]) < 1e-6)
    check("serisi kesilen hisse BOS kaliyor (kote disi bilinmiyorken)",
          pd.isna(ccc["ret_21"]), str(ccc["ret_21"]))

    # --- Ayni senaryo, ama CCC kote disi olarak isaretli
    class _Delisted:
        @staticmethod
        def confirmed():
            return {"CCC": {"confirmed": True}}

    sys.modules["src.delisting"] = _Delisted
    m2 = paper.mark(horizons=(21,), progress=False)
    res2 = pd.read_csv(paper.RESULTS)
    ccc2 = res2[res2["ticker"] == "CCC"].iloc[0]
    check("kote disi pozisyon son fiyattan KAPATILIYOR",
          not pd.isna(ccc2["ret_21"]), str(ccc2["ret_21"]))
    check("kapatma sayisi raporlaniyor", m2.get("closed_delisted", 0) > 0,
          str(m2.get("closed_delisted")))
    check("kote disi bayragi satirda", bool(ccc2["kote_disi"]))

    print()
    print("=" * 70)
    print("3) OZET - ANLAMLILIK KOHORT BAZINDA")
    print("=" * 70)

    s21 = paper.summary(21)
    check("ozet uretiliyor", s21.get("ok"), str(s21.get("reason")))
    check("kohort sayisi pozisyon sayisindan kucuk",
          s21["cohorts"] <= s21["positions"],
          f"{s21['cohorts']} kohort / {s21['positions']} pozisyon")
    check("tek kohortta t hesaplanmiyor", s21.get("t_stat") is None,
          str(s21.get("t_stat")))

    # Uc ayri gun ekleyip t'nin devreye girmesini bekle
    for i, d in enumerate(("2026-02-09", "2026-02-16", "2026-02-23", "2026-03-02")):
        paper.record_live(rank, top_n=3, date=d)
    paper.mark(horizons=(21,), progress=False)
    s21b = paper.summary(21)
    check("cok kohortta t hesaplaniyor", s21b.get("t_stat") is not None,
          str(s21b.get("t_stat")))
    check("kohort sayisi arttı", s21b["cohorts"] > s21["cohorts"],
          f"{s21['cohorts']} -> {s21b['cohorts']}")

    # Kaynak filtresi
    only_panel = paper.summary(21, source="panel")
    check("olmayan kaynak icin bos sonuc", not only_panel.get("ok"),
          str(only_panel.get("reason")))

    # Yanlilik uyarisi yalnizca panelde
    check("gercek tarama uyarisi tasimaz",
          paper._bias_warning("live") is None)
    check("panel uyarisi tasir", "YANLILIGI" in (paper._bias_warning("panel") or ""))

finally:
    sys.modules.pop("src.delisting", None)
    paper.PAPER, paper.COHORTS, paper.RESULTS, paper.SUMMARY = _saved
    shutil.rmtree(tmp, ignore_errors=True)

print()
if fails:
    print(f"{fails} KONTROL BASARISIZ")
    raise SystemExit(1)
print("TUM KARNE TESTLERI GECTI")
