"""Sirket olaylari — bildirimin BAR'a dogru gunden girdigini dogrular.

Bu modulun butun riski tek bir yerde: bir 8-K'nin hangi gune yazildigi.
Bildirimlerin cogu kapanistan sonra dosyalaniyor ve bilanco (2.02)
bildirimlerinin neredeyse tamami boyle. Dosyalama gunune yazarsak, o gunun
kapanisinda hesaplanan bir ozellik henuz olmamis bir olayi gormus olur; en
buyuk fiyat hareketi de tam o olayin ertesinde gerceklesir. Yani hata,
olculebilir ve tamamen sahte bir kenar uretir.

Ilk surum bunu yanlis yapiyordu: yaz saatini gormezden gelip sabit UTC-5
kullaniyor ve yorumunda bunun "temkinli" oldugu yaziyordu. Tam tersiydi --
20:30 UTC yazin 16:30 EDT'dir, kapanistan sonra, ama UTC-5 ile 15:30
hesaplanip AYNI gune yaziliyordu.

Ag istegi YAPILMAZ: hepsi sentetik kayitlarla.

Calistir:  python tests/test_olay.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import olay as ol           # noqa: E402

fails = 0


def check(ad, kosul, ek=""):
    global fails
    ok = bool(kosul)
    if not ok:
        fails += 1
    print(f"  {'OK  ' if ok else 'HATA'}  {ad}" + (f"  {ek}" if ek else ""))


print("=" * 72)
print("1) KAPANIS SONRASI BILDIRIM ERTESI GUNE YAZILIR")
print("=" * 72)

# Kis (EST, UTC-5): 21:31Z = 16:31 ET -> kapanistan sonra
check("kis, kapanis sonrasi bir gun ileri",
      ol._etkin_gun("2016-01-26T21:31:06.000Z") == "2016-01-27",
      ol._etkin_gun("2016-01-26T21:31:06.000Z"))
# Yaz (EDT, UTC-4): 20:30Z = 16:30 ET -> kapanistan sonra. Sabit UTC-5
# kullanan surum bunu 15:30 sanip AYNI gune yaziyordu.
check("yaz, kapanis sonrasi bir gun ileri",
      ol._etkin_gun("2016-07-26T20:30:55.000Z") == "2016-07-27",
      ol._etkin_gun("2016-07-26T20:30:55.000Z"))
# Yaz, gercekten kapanis oncesi: 14:36Z = 10:36 ET
check("yaz, kapanis oncesi ayni gun",
      ol._etkin_gun("2016-03-04T14:36:19.000Z") == "2016-03-04",
      ol._etkin_gun("2016-03-04T14:36:19.000Z"))
# Sinir: 19:59Z yazin 15:59 ET -> hala kapanis oncesi
check("15:59 ET hala ayni gun",
      ol._etkin_gun("2016-07-26T19:59:00.000Z") == "2016-07-26")
check("16:00 ET ertesi gun",
      ol._etkin_gun("2016-07-26T20:00:00.000Z") == "2016-07-27")
check("bozuk zaman damgasi patlamiyor",
      ol._etkin_gun("2016-07-26") == "2016-07-26")

print()
print("=" * 72)
print("2) OZELLIKLER GERIYE BAKAR")
print("=" * 72)

gun = pd.DatetimeIndex(pd.bdate_range("2024-01-01", periods=120))
# Olay 40. barda etkin oluyor.
olaylar = pd.DataFrame({"etkin_gun": [gun[40]], "maddeler": ["2.02,9.01"]})
f = ol.ozellikler(olaylar, gun)

check("olay gununden ONCE hicbir iz yok",
      float(f["olay_sonuc"].iloc[:40].max()) == 0.0
      and float(f["olay_sayi21"].iloc[:40].max()) == 0.0)
check("olay gununde gorunur",
      float(f["olay_sonuc"].iloc[40]) == 1.0
      and float(f["olay_gun_once"].iloc[40]) == 0.0)
check("bir gun sonra 'bir bar once' der",
      float(f["olay_gun_once"].iloc[41]) == 1.0)
check("21 bar sonra sayim dusuyor",
      float(f["olay_sayi21"].iloc[62]) == 0.0,
      f"{float(f['olay_sayi21'].iloc[62])}")
check("olay oncesi 'gun_once' varsayilan degerde",
      float(f["olay_gun_once"].iloc[0]) == 63.0,
      f"{float(f['olay_gun_once'].iloc[0])}")

# Ayni kilit, gosterge setindeki gibi: seriyi kesince onceki barlar
# degismemeli. Degisiyorsa bir yerde ileriye bakiliyordur.
kes = 80
kismi = ol.ozellikler(olaylar, gun[:kes])
sapan = [c for c in f.columns
         if not np.allclose(f[c].to_numpy()[:kes], kismi[c].to_numpy(),
                            equal_nan=True)]
check("seriyi kesmek onceki barlari degistirmiyor", not sapan, str(sapan))

print()
print("=" * 72)
print("3) ISLEM GUNU OLMAYAN BIR GUNE DUSEN OLAY")
print("=" * 72)

# Cumartesi etkin olan bir olay ilk SONRAKI bara tasinmali; kaybolmamali,
# geriye de gitmemeli.
cumartesi = pd.Timestamp("2024-01-06")
o2 = pd.DataFrame({"etkin_gun": [cumartesi], "maddeler": ["8.01"]})
f2 = ol.ozellikler(o2, gun)
ilk_sonra = int(np.searchsorted(gun.to_numpy(), cumartesi.to_datetime64(),
                                side="left"))
check("hafta sonu olayi kaybolmuyor", float(f2["olay_diger"].max()) == 1.0)
check("ilk sonraki bara tasindi",
      float(f2["olay_gun_once"].iloc[ilk_sonra]) == 0.0,
      f"bar {ilk_sonra} = {gun[ilk_sonra].date()}")
check("onceki bara SIZMADI",
      float(f2["olay_diger"].iloc[ilk_sonra - 1]) == 0.0)

# Barlarin bittigi tarihten sonraki bir olay hicbir seye dokunmamali.
o3 = pd.DataFrame({"etkin_gun": [gun[-1] + pd.Timedelta(days=30)],
                   "maddeler": ["8.01"]})
check("barlardan sonraki olay gormezden geliniyor",
      float(ol.ozellikler(o3, gun)["olay_diger"].max()) == 0.0)

print()
print("=" * 72)
print("4) MADDE KODLARI DOGRU OBEGE GIRIYOR")
print("=" * 72)

# Bayraklar 21 barlik bir pencereyi tariyor, o yuzden olaylar birbirinden
# pencereden uzaga konuyor -- yoksa "bulasma" diye olcecegim sey aslinda
# pencerenin dogru calismasi olurdu.
o4 = pd.DataFrame({
    "etkin_gun": [gun[10], gun[40], gun[70]],
    "maddeler": ["2.02,9.01", "5.02", "3.01,8.01"],
})
f4 = ol.ozellikler(o4, gun)
check("2.02 -> sonuc", float(f4["olay_sonuc"].iloc[10]) == 1.0)
check("5.02 -> yonetim", float(f4["olay_yonetim"].iloc[40]) == 1.0)
check("3.01 -> kotasyon", float(f4["olay_kotasyon"].iloc[70]) == 1.0)
check("8.01 -> diger", float(f4["olay_diger"].iloc[70]) == 1.0)
check("bir obek digerine bulasmiyor",
      float(f4["olay_yonetim"].iloc[10]) == 0.0
      and float(f4["olay_sonuc"].iloc[40]) == 0.0
      and float(f4["olay_kotasyon"].iloc[40]) == 0.0)
check("bayrak 21 bar sonra soner",
      float(f4["olay_sonuc"].iloc[10 + 21]) == 0.0,
      f"{float(f4['olay_sonuc'].iloc[31])}")
# 9.01 (ekler) her bildirimde var ve hicbir sey anlatmaz; kendi obegi yok.
check("9.01 icin obek yok", "olay_ekler" not in f4.columns)

print()
print("=" * 72)
print("5) OLAYI OLMAYAN SIRKET")
print("=" * 72)

# Bir sirketin hic 8-K'si olmamasi bir bilgidir, eksik veri degil. Sutunlar
# NaN degil, "olay yok" anlamina gelen degerlerle dolmali -- yoksa model
# medyanla doldurur ve olaysiz sirketler ortalama bir sirket gibi gorunur.
bos = ol.ozellikler(None, gun)
check("sutunlar yine de uretiliyor", bos.shape[1] == f.shape[1])
check("hicbiri NaN degil", int(bos.isna().sum().sum()) == 0)
check("sayimlar sifir", float(bos["olay_sayi21"].max()) == 0.0)
check("son olay 'cok uzakta' diyor", float(bos["olay_gun_once"].min()) == 63.0)
check("bos cerceve de ayni sonucu veriyor",
      ol.ozellikler(pd.DataFrame(columns=["etkin_gun", "maddeler"]),
                    gun).equals(bos))

print()
print("=" * 72)
print("6) YOGUNLUK KENDI GECMISINE GORE OLCULUYOR")
print("=" * 72)

# Duzenli bildirim yapan bir sirkette yogunluk 1 civarinda kalmali; ani bir
# yigilma bunu yukari cikarmali. Mutlak sayim bunu soyleyemez -- ayda bir
# bildiren bir sirket icin iki bildirim cok, on bildiren icin degil.
uzun_gun = pd.DatetimeIndex(pd.bdate_range("2022-01-01", periods=600))
duzenli = pd.DataFrame({"etkin_gun": uzun_gun[::21],
                        "maddeler": ["8.01"] * len(uzun_gun[::21])})
fd = ol.ozellikler(duzenli, uzun_gun)
sakin = float(fd["olay_yogunluk"].iloc[400:].median())
check("duzenli bildirimde yogunluk 1 civari", 0.5 < sakin < 1.6,
      f"{sakin:.2f}")

yigin = pd.concat([duzenli,
                   pd.DataFrame({"etkin_gun": uzun_gun[500:506],
                                 "maddeler": ["8.01"] * 6})])
fy = ol.ozellikler(yigin, uzun_gun)
check("ani yigilma yogunlugu yukari cikariyor",
      float(fy["olay_yogunluk"].iloc[506]) > 2 * sakin,
      f"{float(fy['olay_yogunluk'].iloc[506]):.2f} vs {sakin:.2f}")

print()
if fails:
    print(f"{fails} KONTROL BASARISIZ")
    raise SystemExit(1)
print("TUM OLAY TESTLERI GECTI")
