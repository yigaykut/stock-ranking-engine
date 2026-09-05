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

import json
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

    print()
    print("=" * 72)
    print("9) META-ETIKET PANELI")
    print("=" * 72)

    with tempfile.TemporaryDirectory() as td:
        yol = Path(td) / "panel.csv"
        oz = kb.panel(evren(sonrasi=0.006, seed0=1), bench, ufuklar=(3, 5),
                      min_bar=220, yol=yol)
        check("panel uretildi", oz.get("ok"), str(oz.get("reason")))
        if oz.get("ok"):
            t = pd.read_csv(yol)
            print(f"        {oz['satir']} satir, {len(oz['ozellikler'])} ozellik")
            check("satir sayisi uyuyor", len(t) == oz["satir"])
            check("kimlik sutunlari var",
                  {"ticker", "tarih", "kurulum", "yon"} <= set(t.columns))
            check("etiket sutunlari var",
                  {"fazla_3g", "kazanc_3g", "fazla_5g", "kazanc_5g"}
                  <= set(t.columns))
            check("sayisal ozellikler var",
                  {"atr_pct", "rsi14", "ma200_uzaklik", "hacim_orani"}
                  <= set(t.columns))
            check("kova sutunlari da tasiniyor",
                  {"oynaklik", "likidite", "trend_konumu"} <= set(t.columns))
            check("kazanc etiketi 0/1 (veya bos)",
                  set(t["kazanc_5g"].dropna().unique()) <= {0.0, 1.0},
                  str(set(t["kazanc_5g"].dropna().unique())))
            check("kazanc, fazla getirinin isaretiyle tutarli",
                  bool(((t["fazla_5g"] > 0).astype(float)
                        == t["kazanc_5g"]).where(t["fazla_5g"].notna())
                       .dropna().all()))
            check("ekilmis kenar panelde de gorunuyor",
                  float(t.loc[t["kurulum"] == TEST_ID, "kazanc_5g"].mean()) > 0.8,
                  f"%{100*float(t.loc[t['kurulum'] == TEST_ID, 'kazanc_5g'].mean()):.0f}")
            check("son barlarin etiketi BOS (ufuk dolmadi)",
                  bool(t["kazanc_5g"].isna().any()))
            check("ozellik listesinde etiket yok",
                  not any(c.startswith(("fazla_", "kazanc_"))
                          for c in oz["ozellikler"]))
            check("frekans sutunu var ve ozellik sayilmiyor",
                  "frekans" in t.columns
                  and "frekans" not in oz["ozellikler"])

    # Frekans basina AYRI dosya: saatlik panel gunlugu ezmemeli
    with tempfile.TemporaryDirectory() as td:
        eski_data = kb.DATA
        kb.DATA = Path(td)
        try:
            a = kb.panel(evren(sonrasi=0.0, seed0=11), bench, ufuklar=(5,),
                         min_bar=220, frekans="1d")
            b_ = kb.panel(evren(sonrasi=0.0, seed0=11), bench, ufuklar=(21,),
                          min_bar=220, frekans="1h")
            check("iki panel ayri dosyaya yazildi", a["yol"] != b_["yol"],
                  f"{Path(a['yol']).name} vs {Path(b_['yol']).name}")
            check("gunluk panel duruyor", Path(a["yol"]).exists())
            check("saatlik panel duruyor", Path(b_["yol"]).exists())
            check("dosya adinda frekans var",
                  "1h" in Path(b_["yol"]).name)
        finally:
            kb.DATA = eski_data

finally:
    kv.KAYIT.pop(TEST_ID, None)

check("test kurulumu kayittan silindi", TEST_ID not in kv.KAYIT)

print()
print("=" * 72)
print("9b) KISA TARAF — kazanc tanimi TERS olmali")
print("=" * 72)

# Cikis sinyalleri icin "kazanc", endeksi gecmek DEGIL endeksin ALTINDA
# kalmaktir. Ilk surumde hepsi "endeksi gecti" diye sayiliyordu ve saatlik
# olcumde dagitim gunu +%5 kenar gosterdi -- kurulumun CALISTIGINI degil
# tam tersini gosteren bir sayi.
KISA_ID = "_test_kisa"
kv.KAYIT[KISA_ID] = kv.Kurulum(KISA_ID, "Test cikis", "short",
                               _sicrama_dedektor, "yalnizca test" * 5, 5)
kv.KAYIT[TEST_ID] = kv.Kurulum(TEST_ID, "Test giris", "long",
                               _sicrama_dedektor, "yalnizca test" * 5, 5)
try:
    bench3 = pd.Series(100.0, index=pd.bdate_range("2024-01-01", periods=420))
    # Tetikten SONRA yukari giden seri: uzun taraf kazanmali, kisa taraf
    # KAYBETMELI. Ayni dedektor, ayni sinyaller, sadece yon farkli.
    kk = kb.kur(evren(sonrasi=0.006, seed0=3), bench3, ufuklar=(5,),
                min_bar=220)
    kov = {(x["kurulum"], x["kosul"]): x for x in kk["kovalar"]}
    u, k_ = kov.get((TEST_ID, "*")), kov.get((KISA_ID, "*"))
    check("iki yon de kovalandi", u is not None and k_ is not None)
    if u and k_:
        print(f"        uzun: p={u['p']:.3f} taban={u['taban']:.3f} "
              f"edge={u['edge']:+.3f}")
        print(f"        kisa: p={k_['p']:.3f} taban={k_['taban']:.3f} "
              f"edge={k_['edge']:+.3f}")
        check("uzun tarafta kenar POZITIF", u["edge"] > 0.15, str(u["edge"]))
        check("ayni sinyalde kisa taraf kenari NEGATIF", k_["edge"] < -0.15,
              str(k_["edge"]))
        check("kisa tarafin tabani uzunun tumleyeni",
              abs((u["taban"] + k_["taban"]) - 1.0) < 0.02,
              f"{u['taban']:.3f} + {k_['taban']:.3f}")
        check("kova yon bilgisini tasiyor",
              u["yon"] == "long" and k_["yon"] == "short")
    check("coklu test uyarisi var",
          any("COKLU TEST" in n for n in kk["notlar_tr"]),
          "; ".join(kk["notlar_tr"])[:80])
finally:
    kv.KAYIT.pop(KISA_ID, None)

print()
print("=" * 72)
print("9c) GUN ICI ENDEKS HIZALAMASI")
print("=" * 72)

# Saatlik barlar + saatlik endeks: endeks getirisi gun ICINDE de sifirdan
# farkli olmali. Gun bazinda hizalanirsa bir gunun tum barlari ayni endeks
# degerini alir, endeks getirisi 0 cikar ve "endeksten iyi" olcusu sessizce
# "yukari gitti"ye doner.
saat_idx = pd.DatetimeIndex(
    [pd.Timestamp("2025-01-02") + pd.Timedelta(hours=9 + b) + pd.Timedelta(days=g)
     for g in range(300) for b in range(7)])
bench_saat = pd.Series(np.linspace(100, 130, len(saat_idx)), index=saat_idx)

gun_bazli = kb._bench_hazirla(bench_saat, gun_bazli=True)
zaman_bazli = kb._bench_hazirla(bench_saat, gun_bazli=False)
check("gun bazli hizalama gun basina TEK deger birakiyor",
      len(gun_bazli) == 300, f"{len(gun_bazli)} (300 gun)")
check("zaman bazli hizalama tum barlari koruyor",
      len(zaman_bazli) == len(saat_idx), f"{len(zaman_bazli)}")
# Kayan nokta: ayni degerin std'si tam sifir degil ~1e-14 cikiyor.
check("gun bazli seride gun ici endeks getirisi SIFIR",
      float((gun_bazli.reindex(saat_idx.normalize()).to_numpy()[:7]).std()) < 1e-9,
      f"{float((gun_bazli.reindex(saat_idx.normalize()).to_numpy()[:7]).std()):.2e}")
check("zaman bazli seride gun ici endeks getirisi sifir DEGIL",
      float(zaman_bazli.to_numpy()[:7].std()) > 0)

print()
print("=" * 72)
print("10) FREKANS FARKINDALIGI")
print("=" * 72)

# Gun ici ufuklar BAR cinsindendir. 21 barlik bir saatlik ufuk 21 gun degil
# ~3 gun ortusme demek; etkin orneklem bunu bilmezse gereksiz yere kucuk
# cikar ve sistem hicbir seyi olculebilir bulmaz.
check("gunluk: 5 gun ufuk, 200 gun -> 40",
      kb.etkin_n(500, 200, 5, 1.0) == 40.0, str(kb.etkin_n(500, 200, 5, 1.0)))
check("saatlik: 21 bar (~3 gun), 200 gun -> ~67",
      abs(kb.etkin_n(500, 200, 21, 7.0) - 200 / 3) < 0.1,
      f"{kb.etkin_n(500, 200, 21, 7.0):.1f}")
check("gun altindaki ufukta sinir GUN sayisi olur",
      kb.etkin_n(500, 200, 3, 7.0) == 200.0,
      f"{kb.etkin_n(500, 200, 3, 7.0)}  (3 bar < 1 gun)")
check("bar_gun verilmezse gunluk gibi davranir",
      kb.etkin_n(500, 200, 5) == kb.etkin_n(500, 200, 5, 1.0))

kv.KAYIT[TEST_ID] = kv.Kurulum(TEST_ID, "Test", "long", _sicrama_dedektor,
                               "yalnizca test" * 5, 5)
try:
    bench2 = pd.Series(100.0, index=pd.bdate_range("2024-01-01", periods=420))
    k1d = kb.kur(evren(sonrasi=0.004, seed0=7), bench2, ufuklar=(5,),
                 min_bar=220, frekans="1d")
    k1h = kb.kur(evren(sonrasi=0.004, seed0=7), bench2, ufuklar=(21,),
                 min_bar=220, frekans="1h")
    check("frekans ciktida kayitli", k1d["frekans"] == "1d" and k1h["frekans"] == "1h")
    check("bar_gun ciktida kayitli", k1h["bar_gun"] == 7.0, str(k1h["bar_gun"]))
    check("ufuk birimi bar olarak isaretli", k1d["ufuk_birimi"] == "bar")

    g1d = next(x for x in k1d["kovalar"] if x["kosul"] == "*")
    g1h = next(x for x in k1h["kovalar"] if x["kosul"] == "*")
    check("ayni gun sayisinda saatlik ufuk daha COK etkin gozlem veriyor",
          g1h["n_etkin"] > g1d["n_etkin"],
          f"1d {g1d['n_etkin']} vs 1h {g1h['n_etkin']}")

    # Frekans basina AYRI dosya: ikinci kosu birincisini ezmemeli
    eski_data, eski_cikti = kb.DATA, kb.CIKTI
    with tempfile.TemporaryDirectory() as td:
        kb.DATA, kb.CIKTI = Path(td), Path(td) / "k.json"
        try:
            kb.kaydet(k1d)
            kb.kaydet(k1h)
            check("kanonik dosya en son kosani gosteriyor",
                  kb.yukle()["frekans"] == "1h")
            check("gunluk arsiv duruyor", kb.yukle(frekans="1d")["frekans"] == "1d")
            check("saatlik arsiv duruyor", kb.yukle(frekans="1h")["frekans"] == "1h")
            check("kayitli frekanslar listeleniyor",
                  kb.kayitli_frekanslar() == ["1d", "1h"],
                  str(kb.kayitli_frekanslar()))

            # ESKI DOSYA GERI UYUMU: frekans alani olmayan kayit gunluk sayilir
            kb.CIKTI.write_text(json.dumps({k: v for k, v in k1d.items()
                                            if k != "frekans"}),
                                encoding="utf-8")
            # Arsivlerin IKISI de silinmeli: yoksa 1h sorgusu kendi
            # arsivini bulur ve geri uyum yolu hic denenmemis olur.
            for f in Path(td).glob("kisa_vade_kalibrasyon_*.json"):
                f.unlink()
            check("frekanssiz eski kayit '1d' olarak okunuyor",
                  kb.yukle(frekans="1d") is not None)
            check("frekanssiz eski kayit '1h' diye okunMUYOR",
                  kb.yukle(frekans="1h") is None)
        finally:
            kb.DATA, kb.CIKTI = eski_data, eski_cikti
finally:
    kv.KAYIT.pop(TEST_ID, None)

print()
if fails:
    print(f"{fails} KONTROL BASARISIZ")
    raise SystemExit(1)
print("TUM KALIBRASYON TESTLERI GECTI")
