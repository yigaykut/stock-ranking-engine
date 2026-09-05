"""Kisa vade guven degeri — olculmus frekans, uydurma skor degil.

Bu modulun tek isi "bu kurulum olustugunda ne oldu" sorusunu SAYARAK
cevaplamak. Testler uc seyi kovaliyor:

  1. ISTATISTIK DOGRU MU
     Wilson araligi kucuk orneklemde ve uclarda dogru davranmali (0/10 icin
     sifir genislikte aralik vermemeli). Buzme kucuk n'de tabana, buyuk n'de
     ham orana gitmeli. Etkin orneklem, ham sayimdan kucuk olmali.

  2. GERCEK KENARI BULUYOR MU
     Sonucu ONCEDEN EKILMIS bir seride, kalibrasyon o kenari gormeli.
     Gormezse modul yalnizca sayi uretiyor demektir.

  3. OLMAYAN KENARI UYDURMUYOR MU — asil onemli olan bu.
     Sonucu rastgele olan bir seride, kalibrasyon "bu kurulum calisiyor"
     DEMEMELI. Bir guven sistemi icin yanlis pozitif, kacirilan sinyalden
     cok daha pahalidir: insan o sayiya bakip para koyar.

Calistir:  python tests/test_kalibrasyon.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import kalibrasyon as kb      # noqa: E402
from src import kisa_vade as kv        # noqa: E402

fails = 0


def check(name: str, cond: bool, extra: str = "") -> None:
    global fails
    if cond:
        print(f"  OK    {name}" + (f"  {extra}" if extra else ""))
    else:
        print(f"  HATA  {name}" + (f"  {extra}" if extra else ""))
        fails += 1


print("=" * 72)
print("1) WILSON ARALIGI")
print("=" * 72)

lo, hi = kb.wilson(5, 10)
check("50/50'de aralik simetrik", abs((lo + hi) / 2 - 0.5) < 0.02,
      f"[{lo:.3f}, {hi:.3f}]")
lo0, hi0 = kb.wilson(0, 10)
check("0/10'da aralik SIFIR GENISLIKTE degil", hi0 > 0.15,
      f"[{lo0:.3f}, {hi0:.3f}]")
check("0/10'da alt sinir 0", lo0 == 0.0)
lo1, hi1 = kb.wilson(10, 10)
check("10/10'da ust sinir 1", hi1 == 1.0, f"[{lo1:.3f}, {hi1:.3f}]")
check("10/10'da alt sinir 1 DEGIL", lo1 < 0.85, f"{lo1:.3f}")

d_kucuk = kb.wilson(50, 100)
d_buyuk = kb.wilson(500, 1000)
check("n buyudukce aralik daraliyor",
      (d_buyuk[1] - d_buyuk[0]) < (d_kucuk[1] - d_kucuk[0]),
      f"{d_kucuk[1]-d_kucuk[0]:.3f} -> {d_buyuk[1]-d_buyuk[0]:.3f}")
check("aralik her zaman [0,1] icinde",
      all(0 <= a <= 1 and 0 <= b <= 1
          for a, b in (kb.wilson(k, 12) for k in range(13))))
check("n=0'da NaN", not np.isfinite(kb.wilson(0, 0)[0]))

print()
print("=" * 72)
print("2) ETKIN ORNEKLEM")
print("=" * 72)

check("ayni gunde 500 sinyal, 200 gun, 5g ufuk -> 40",
      kb.etkin_n(500, 200, 5) == 40.0, str(kb.etkin_n(500, 200, 5)))
check("etkin, ham sayimdan buyuk olamaz",
      kb.etkin_n(10, 500, 1) == 10.0, str(kb.etkin_n(10, 500, 1)))
check("ufuk buyudukce etkin kuculuyor",
      kb.etkin_n(500, 200, 10) < kb.etkin_n(500, 200, 3))
check("sinyal yoksa 0", kb.etkin_n(0, 100, 5) == 0.0)
check("en az 1", kb.etkin_n(5, 2, 100) >= 1.0)

print()
print("=" * 72)
print("3) BUZME")
print("=" * 72)

taban_oran = 0.52
kucuk = kb.buzulmus(5, 7, taban_oran)
buyuk = kb.buzulmus(700, 1000, taban_oran)
check("kucuk orneklem tabana yakin", abs(kucuk - taban_oran) < 0.06,
      f"5/7 ham %71 -> %{100*kucuk:.0f}")
check("buyuk orneklem ham orana yakin", abs(buyuk - 0.70) < 0.02,
      f"700/1000 ham %70 -> %{100*buyuk:.0f}")
check("n=0'da taban doner", kb.buzulmus(0, 0, taban_oran) == taban_oran)
check("buzme monoton", kb.buzulmus(70, 100, 0.5) < kb.buzulmus(700, 1000, 0.5))


# ---------------------------------------------------------------------------
#  Sentetik evren
# ---------------------------------------------------------------------------
def seri(n: int, tetik_gunleri: "set[int]", sonrasi: float,
         seed: int = 0, gurultu: float = 0.004) -> pd.DataFrame:
    """Belirli gunlerde %4 sicrama, ardindan `sonrasi` kadar gunluk surukleme.

    `sonrasi` > 0 ise kurulum sonrasi gercek bir kenar EKILMIS olur.
    """
    r = np.random.default_rng(seed)
    kapanis = np.zeros(n)
    kapanis[0] = 50.0
    for i in range(1, n):
        adim = r.normal(0.0, gurultu)
        if (i - 1) in tetik_gunleri:
            adim += 0.04                       # tetik gunu: %4 yukari
        # tetikten sonraki 5 gun surukleme
        if any((i - k) in range(1, 6) for k in tetik_gunleri):
            adim += sonrasi
        kapanis[i] = kapanis[i - 1] * (1 + adim)
    acilis = kapanis * (1 - 0.001)
    yuksek = np.maximum(acilis, kapanis) * 1.002
    dusuk = np.minimum(acilis, kapanis) * 0.998
    hacim = np.full(n, 1_000_000.0)
    return pd.DataFrame({"Open": acilis, "High": yuksek, "Low": dusuk,
                         "Close": kapanis, "Volume": hacim},
                        index=pd.bdate_range("2024-01-01", periods=n))


def _sicrama_dedektor(df: pd.DataFrame):
    """Test kurulumu: gunluk %3+ yukselis. Yalnizca geriye bakar."""
    c = df["Close"]
    var = c > c.shift(1) * 1.03
    return var.fillna(False), pd.Series(1.0, index=df.index)


TEST_ID = "_test_sicrama"


def evren(sonrasi: float, n_hisse: int = 12, n_bar: int = 420,
          seed0: int = 0) -> dict:
    out = {}
    for h in range(n_hisse):
        r = np.random.default_rng(seed0 + h)
        gunler = set(r.choice(np.arange(230, n_bar - 15), size=25, replace=False)
                     .tolist())
        out[f"T{h:02d}"] = {"history": seri(n_bar, gunler, sonrasi,
                                            seed=seed0 * 100 + h)}
    return out


print()
print("=" * 72)
print("4) EKILMIS KENARI BULUYOR MU")
print("=" * 72)

kv.KAYIT[TEST_ID] = kv.Kurulum(TEST_ID, "Test sicramasi", "long",
                               _sicrama_dedektor, "yalnizca test icin" * 3, 5)
try:
    bench = pd.Series(100.0, index=pd.bdate_range("2024-01-01", periods=420))

    kal_iyi = kb.kur(evren(sonrasi=0.006, seed0=1), bench, ufuklar=(5,),
                     min_bar=220)
    kovalar = {(k["kurulum"], k["kosul"]): k for k in kal_iyi["kovalar"]}
    g = kovalar.get((TEST_ID, "*"))
    check("kova olustu", g is not None)
    if g:
        print(f"        n={g['n']} n_gun={g['n_gun']} n_etkin={g['n_etkin']} "
              f"p={g['p']} taban={g['taban']} alt={g['alt']}")
        check("ekilmis kenar pozitif goruldu", g["edge"] > 0.15,
              f"edge {g['edge']}")
        check("alt guven siniri taban oranin ustunde", g["alt"] > g["taban"],
              f"alt {g['alt']} > taban {g['taban']}")
        check("kova olculdu sayiliyor", g["durum"] == "olculdu", g["durum"])
        check("medyan getiri pozitif", (g["medyan_getiri"] or 0) > 0,
              str(g["medyan_getiri"]))

    print()
    print("=" * 72)
    print("5) OLMAYAN KENARI UYDURUYOR MU — asil test")
    print("=" * 72)

    kal_bos = kb.kur(evren(sonrasi=0.0, seed0=50), bench, ufuklar=(5,),
                     min_bar=220)
    kovalar2 = {(k["kurulum"], k["kosul"]): k for k in kal_bos["kovalar"]}
    g2 = kovalar2.get((TEST_ID, "*"))
    check("kova olustu", g2 is not None)
    if g2:
        print(f"        n={g2['n']} n_etkin={g2['n_etkin']} p={g2['p']} "
              f"taban={g2['taban']} alt={g2['alt']} ust={g2['ust']}")
        check("kenar yok denildi (alt sinir tabani asmiyor)",
              g2["alt"] <= g2["taban"], f"alt {g2['alt']} taban {g2['taban']}")
        check("aralik tabani iceriyor",
              g2["alt"] <= g2["taban"] <= g2["ust"],
              f"[{g2['alt']}, {g2['ust']}] taban {g2['taban']}")

    print()
    print("=" * 72)
    print("6) GUVEN SORGUSU")
    print("=" * 72)

    bos = kb.guven(None, TEST_ID, 5)
    check("kalibrasyon yoksa 'bilinmiyor'", bos["durum"] == "bilinmiyor")
    check("bilinmiyorken olasilik uydurulmuyor", bos["p"] is None)

    s = kb.guven(kal_iyi, TEST_ID, 5)
    check("olculmus kurulumda durum 'olculdu'", s["durum"] == "olculdu",
          s["durum"])
    check("aciklama sayilari iceriyor", "kez olustu" in s["aciklama_tr"])
    check("ayirt_edilebilir bayragi acik", s.get("ayirt_edilebilir") is True)

    yok = kb.guven(kal_iyi, "boyle_bir_kurulum_yok", 5)
    check("bilinmeyen kurulumda 'bilinmiyor'", yok["durum"] == "bilinmiyor")

    # Kosullu kova genelden once gelmeli
    ozel = kb.guven(kal_iyi, TEST_ID, 5, {"oynaklik": "oynak"})
    check("kosul verilince kova secimi calisiyor",
          ozel["durum"] in ("olculdu", "az veri"), ozel["durum"])
    check("bilinmeyen kosul degeri genele dusuyor",
          kb.guven(kal_iyi, TEST_ID, 5, {"oynaklik": "boyle_kova_yok"}
                   )["kova"] == s["kova"])

    print()
    print("=" * 72)
    print("7) YAPI VE KAYIT")
    print("=" * 72)

    check("taban orani raporlandi", "5" in kal_iyi["taban"])
    check("kazanc tanimi endeksli", "endeks" in kal_iyi["kazanc_tanimi"])
    check("notlar var", len(kal_iyi["notlar_tr"]) >= 3)
    check("islenen hisse sayisi dogru", kal_iyi["hisse"] == 12,
          str(kal_iyi["hisse"]))
    check("her kovada gerekli alanlar",
          all({"kurulum", "ufuk", "kosul", "n", "n_etkin", "p", "taban",
               "edge", "alt", "ust", "durum"} <= set(k)
              for k in kal_iyi["kovalar"]))
    check("hicbir kovada n_etkin > n yok",
          all(k["n_etkin"] <= k["n"] for k in kal_iyi["kovalar"]))
    check("MIN_HAM altindaki kovalar elenmis",
          all(k["n"] >= kb.MIN_HAM for k in kal_iyi["kovalar"]))

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "kalib.json"
        kb.kaydet(kal_iyi, p)
        geri = kb.yukle(p)
        check("kayit/okuma ayni", geri["hisse"] == kal_iyi["hisse"])
        check("olmayan dosyada None", kb.yukle(Path(td) / "yok.json") is None)

    print()
    print("=" * 72)
    print("8) ENDEKSSIZ OLCUM ISARETLENIYOR MU")
    print("=" * 72)

    kal_ham = kb.kur(evren(sonrasi=0.0, seed0=50), None, ufuklar=(5,),
                     min_bar=220)
    check("endekssiz olcum acikca isaretleniyor",
          "ENDEKSSIZ" in kal_ham["kazanc_tanimi"], kal_ham["kazanc_tanimi"])

finally:
    kv.KAYIT.pop(TEST_ID, None)

check("test kurulumu kayittan silindi", TEST_ID not in kv.KAYIT)

print()
if fails:
    print(f"{fails} KONTROL BASARISIZ")
    raise SystemExit(1)
print("TUM KALIBRASYON TESTLERI GECTI")
