"""Gunluk dosyasi icin yerel-ayardan bagimsiz tarih damgasi.

gunluk.bat bunu cagirir. Ayri bir dosya olmasinin sebebi, batch icinde ic ice
tirnak kullanmadan tarih uretebilmek — %DATE% Windows yerel ayarina gore
degistigi icin (orn. "Per 13.08.2026") dogrudan kullanilamiyor.
"""
import datetime
import sys

sys.stdout.write(datetime.date.today().isoformat())
