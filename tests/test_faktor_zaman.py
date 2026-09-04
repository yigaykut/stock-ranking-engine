"""Parametre gucunun zaman/rejim kirilimi (faktor_zaman).

Buradaki testlerin cogu TEK bir iddiayi kovaliyor: ortusen etiketler yuzunden
siradan t degeri sisiyor ve Newey-West duzeltmesi bunu geri aliyor. Bu, modulun
var olma sebebi; calistigini kanitlamazsak modul yalnizca sayi uretiyor demektir.

  1. Bagimsiz seride  : duzeltilmis t ~ ham t (duzeltme zarar vermemeli)
  2. Otokorelasyonlu  : duzeltilmis t belirgin KUCUK olmali
     seride
  3. Gecikme hesabi   : 3 gunde bir goruntu + 21 gun ufuk -> ~7 donem ortusme
  4. Zayiflama        : yon degistiren seri "kararsiz" olarak isaretlenmeli
  5. Rejim kirilimi   : yalnizca yeterli gozlem olan rejimler raporlanmali
  6. Gecmise donuk    : ayni kural, gecmis tarihte de ayni etiketi vermeli
     rejim etiketi      ve ILERIYE BAKMAMALI

Calistir:  python tests/test_faktor_zaman.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import faktor_zaman as fz     # noqa: E402
from src import regime as rg           # noqa: E402

fails = 0


def check(name: str, cond: bool, extra: str = "") -> None:
    global fails
    if cond:
        print(f"  OK    {name}" + (f"  {extra}" if extra else ""))
    else:
        print(f"  HATA  {name}" + (f"  {extra}" if extra else ""))
        fails += 1


print("=" * 70)
print("1) NEWEY-WEST — bagimsiz seride duzeltme zarar vermemeli")
print("=" * 70)

rng = np.random.default_rng(7)
bagimsiz = rng.normal(0.02, 0.10, 400)
t_ham = bagimsiz.mean() / (bagimsiz.std(ddof=1) / np.sqrt(bagimsiz.size))
t_nw, se, lag = fz.newey_west_t(bagimsiz, lag=0)
check("lag=0 iken duzeltilmis t ~ ham t", abs(t_nw - t_ham) < 0.02,
      f"ham {t_ham:.3f} vs duz {t_nw:.3f}")

t_nw6, _, _ = fz.newey_west_t(bagimsiz, lag=6)
check("bagimsiz seride gecikme eklemek t'yi ucurmuyor",
      abs(t_nw6 - t_ham) < 0.5 * abs(t_ham), f"{t_nw6:.3f}")

print()
print("=" * 70)
print("2) NEWEY-WEST — ortusen (otokorelasyonlu) seride t KUCULMELI")
print("=" * 70)

# Ortusen etiketi taklit et: her olcum, onceki 7 sokun hareketli ortalamasi.
# Gercek bagimsiz gozlem sayisi ~n/7, ama seri n uzunlugunda gorunuyor.
sok = rng.normal(0.02, 0.10, 406)
ortusen = np.convolve(sok, np.ones(7) / 7.0, mode="valid")   # 400 nokta
t_ham2 = ortusen.mean() / (ortusen.std(ddof=1) / np.sqrt(ortusen.size))
t_nw2, se2, lag2 = fz.newey_west_t(ortusen, lag=6)
check("ortusen seride duzeltilmis t, ham t'den kucuk",
      abs(t_nw2) < abs(t_ham2), f"ham {t_ham2:.2f} -> duz {t_nw2:.2f}")
check("kucultme ciddi (en az %30)", abs(t_nw2) < 0.7 * abs(t_ham2),
      f"oran {abs(t_nw2) / abs(t_ham2):.2f}")
# 7'lik hareketli ortalama, varyansi ~7 kat sisiriyor -> t ~ sqrt(7) kat
# kuculmeli. Genis bir bant biraktik; asil iddia YONUN dogrulugu.
check("kucultme buyuklugu makul aralikta (1.5x - 4x)",
      1.5 < abs(t_ham2) / abs(t_nw2) < 4.0,
      f"{abs(t_ham2) / abs(t_nw2):.2f}x")

check("cok kisa seride NaN doner", not np.isfinite(fz.newey_west_t([1.0, 2.0])[0]))

print()
print("=" * 70)
print("3) ORTUSME GECIKMESI")
print("=" * 70)

d3 = [str(x.date()) for x in pd.date_range("2026-01-01", periods=60, freq="3D")]
check("3 gunde bir goruntu + 21 gun ufuk -> 9 donem",
      fz.ortusme_gecikmesi(d3, 21) == 9, str(fz.ortusme_gecikmesi(d3, 21)))
d1 = [str(x.date()) for x in pd.date_range("2026-01-01", periods=60, freq="D")]
check("gunluk goruntude gecikme daha buyuk",
      fz.ortusme_gecikmesi(d1, 21) > fz.ortusme_gecikmesi(d3, 21),
      f"{fz.ortusme_gecikmesi(d1, 21)} > {fz.ortusme_gecikmesi(d3, 21)}")
check("tek tarihte gecikme 0", fz.ortusme_gecikmesi(["2026-01-01"], 21) == 0)


# ---------------------------------------------------------------------------
# Sentetik etiketli panel: uc parametre, uc farkli davranis.
#   guclu     -> her donem pozitif IC
#   kararsiz  -> ilk yari pozitif, ikinci yari negatif
#   gurultu   -> iliskisiz
# ---------------------------------------------------------------------------
def panel(n_gun: int = 40, per_gun: int = 80, seed: int = 3) -> pd.DataFrame:
    r = np.random.default_rng(seed)
    parcalar = []
    gunler = pd.date_range("2026-01-01", periods=n_gun, freq="3D")
    for i, g in enumerate(gunler):
        y = r.normal(0, 1, per_gun)
        isaret = 1.0 if i < n_gun // 2 else -1.0
        parcalar.append(pd.DataFrame({
            "snapshot_date": str(g.date()),
            "ticker": [f"T{j}" for j in range(per_gun)],
            "raw_guclu": 0.9 * y + r.normal(0, 0.5, per_gun),
            "raw_kararsiz": isaret * (0.9 * y) + r.normal(0, 0.5, per_gun),
            "raw_gurultu": r.normal(0, 1, per_gun),
            "fwd_return_21d_excess": y,
        }))
    return pd.concat(parcalar, ignore_index=True)


print()
print("=" * 70)
print("4) ANALIZ — zayiflama ve karar")
print("=" * 70)

df = panel()
ids = ["guclu", "kararsiz", "gurultu"]
rejim = {d: ("YUKSELIS" if i < 25 else "GECIS")
         for i, d in enumerate(sorted(df["snapshot_date"].unique()))}

out = fz.analyze(df, ids, "fwd_return_21d_excess", horizon=21,
                 weights={"guclu": 8.0, "kararsiz": 6.0, "gurultu": 2.0},
                 rejim=rejim, source="panel")
sat = {r["factor"]: r for r in out["factors"]}

check("uc parametre de raporlandi", len(sat) == 3, str(list(sat)))
check("guclu parametre esigi geciyor", abs(sat["guclu"]["t_nw"]) >= fz.T_ESIK,
      f"t={sat['guclu']['t_nw']}")
check("gurultu parametresi esigi GECMIYOR", abs(sat["gurultu"]["t_nw"]) < fz.T_ESIK,
      f"t={sat['gurultu']['t_nw']}")
check("kararsiz parametrenin yarilari ters isaretli",
      sat["kararsiz"]["ic_ilk_yari"] * sat["kararsiz"]["ic_son_yari"] < 0,
      f"{sat['kararsiz']['ic_ilk_yari']} / {sat['kararsiz']['ic_son_yari']}")
check("kararsiz parametrenin ortalamasi yaniltici sekilde ~0",
      abs(sat["kararsiz"]["ic_mean"]) < 0.1, str(sat["kararsiz"]["ic_mean"]))
check("guclu icin karar metni olumlu", "ayirt edilebilir" in sat["guclu"]["verdict_tr"],
      sat["guclu"]["verdict_tr"])
check("gurultu icin karar metni olumsuz",
      "ayirt edilemiyor" in sat["gurultu"]["verdict_tr"], sat["gurultu"]["verdict_tr"])
check("siralama |t| buyukten kucuge",
      abs(out["factors"][0]["t_nw"]) >= abs(out["factors"][-1]["t_nw"]))
check("duzeltilmis t, ham t'den kucuk (guclu)",
      abs(sat["guclu"]["t_nw"]) < abs(sat["guclu"]["t_naive"]),
      f"{sat['guclu']['t_naive']} -> {sat['guclu']['t_nw']}")

print()
print("=" * 70)
print("5) REJIM KIRILIMI")
print("=" * 70)

check("iki rejim de raporlandi", set(sat["guclu"]["by_regime"]) == {"YUKSELIS", "GECIS"},
      str(list(sat["guclu"]["by_regime"])))
check("rejim sayimlari toplami donem sayisina esit",
      sum(out["regime_counts"].values()) == out["periods"],
      f"{out['regime_counts']} / {out['periods']}")
check("kararsiz parametre rejimlere gore ters isaret veriyor",
      sat["kararsiz"]["by_regime"]["YUKSELIS"]["ic"] *
      sat["kararsiz"]["by_regime"]["GECIS"]["ic"] < 0)

az = {d: "DUSUS" for d in sorted(df["snapshot_date"].unique())[:3]}
out2 = fz.analyze(df, ids, "fwd_return_21d_excess", horizon=21, rejim=az)
check("5 gozlemden az olan rejim raporlanmiyor",
      "DUSUS" not in (out2["factors"][0]["by_regime"] or {}))

check("dusus rejimi yoksa not dusuluyor",
      any("dusus rejimi yok" in n for n in out["notes_tr"]),
      "; ".join(out["notes_tr"])[:90])
check("panel kaynagi icin yanlilik notu var",
      any("hayatta kalma" in n for n in out["notes_tr"]))

print()
print("=" * 70)
print("6) GECMISE DONUK REJIM ETIKETI")
print("=" * 70)

# 3 yillik yukselen seri: son gun mutlaka YUKSELIS olmali.
idx = pd.date_range("2024-01-01", periods=700, freq="B")
yukselen = pd.Series(np.linspace(100, 200, 700), index=idx)
hedef = [str(idx[300].date()), str(idx[500].date()), str(idx[-1].date())]
lab = rg.labels_for_dates(yukselen, hedef)
check("yukselen seride tum tarihler YUKSELIS",
      set(lab.values()) == {"YUKSELIS"}, str(lab))

# ILERIYE BAKIS: seriyi bir tarihte kesip ayni tarihi sorunca ayni cevap.
kesik = yukselen[yukselen.index <= idx[500]]
check("seri kesilince ayni tarih ayni etiketi aliyor (ileriye bakis yok)",
      rg.labels_for_dates(kesik, [hedef[1]]) == {hedef[1]: lab[hedef[1]]})

# Duseni de dogrula: son 250 gunde sert dusus -> DUSUS
duz = np.concatenate([np.linspace(100, 200, 450), np.linspace(200, 90, 250)])
dusen = pd.Series(duz, index=idx)
lab2 = rg.labels_for_dates(dusen, [str(idx[-1].date())])
check("cokusun sonunda DUSUS etiketi", list(lab2.values()) == ["DUSUS"], str(lab2))

check("210 gunden kisa seride etiket uretilmiyor",
      rg.labels_for_dates(yukselen.head(100), [str(idx[99].date())]) == {})

print()
print("=" * 70)
print("7) YAZ/OKU TURU")
print("=" * 70)

import tempfile                                    # noqa: E402
with tempfile.TemporaryDirectory() as td:
    p = Path(td) / "faktor_zaman.json"
    fz.save(out, p)
    geri = fz.load(p)
    check("kayit/okuma ayni icerigi veriyor",
          geri["factors"][0]["factor"] == out["factors"][0]["factor"])
    check("okunamayan dosyada None doner", fz.load(Path(td) / "yok.json") is None)

# Ufka ozel arsiv: 5 gunluk olcum 21 gunlugun uzerine YAZMAMALI.
# 04.09'da tam bu oldu ve 21 gunluk analiz kayboldu.
eski_data, eski_cikti = fz.DATA, fz.CIKTI
with tempfile.TemporaryDirectory() as td:
    fz.DATA = Path(td)
    fz.CIKTI = Path(td) / "faktor_zaman.json"
    try:
        h21 = dict(out, horizon=21)
        h5 = dict(out, horizon=5)
        fz.save(h21)
        fz.save(h5)
        check("kanonik dosya en son kosani gosteriyor", fz.load()["horizon"] == 5)
        check("21 gunluk arsiv duruyor", fz.load(horizon=21)["horizon"] == 21)
        check("5 gunluk arsiv duruyor", fz.load(horizon=5)["horizon"] == 5)
        check("kayitli ufuklar listeleniyor", fz.kayitli_ufuklar() == [5, 21],
              str(fz.kayitli_ufuklar()))
        check("olmayan ufukta None", fz.load(horizon=63) is None)
    finally:
        fz.DATA, fz.CIKTI = eski_data, eski_cikti

print()
fz.print_table(out)

if fails:
    print(f"{fails} KONTROL BASARISIZ")
    raise SystemExit(1)
print("TUM FAKTOR ZAMAN TESTLERI GECTI")
