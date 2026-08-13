"""Guclu bir pano parolasi uretir ve ortam degiskenine yazar.

Parola:
  * kriptografik olarak rastgele uretilir (secrets modulu)
  * ekrana YAZDIRILMAZ  -> sohbet kaydina/terminal gecmisine dusmez
  * yalnizca parola.txt dosyasina yazilir (bu dosya .gitignore'da)
  * ayrica DASHBOARD_PASSWORD ortam degiskenine kaydedilir

Ekrana yalnizca SHA-256 parmak izi basilir; bu, parolayi ele vermeden
"dogru parola ayarlandi mi" kontrolu yapmayi saglar.
"""
import hashlib
import pathlib
import secrets
import subprocess
import sys

ALPHABET = "abcdefghijkmnpqrstuvwxyz23456789"      # karisan harfler (l,o,0,1) yok
GROUPS, GROUP_LEN = 5, 5

pw = "-".join(
    "".join(secrets.choice(ALPHABET) for _ in range(GROUP_LEN))
    for _ in range(GROUPS)
)

bits = round(GROUPS * GROUP_LEN * (len(ALPHABET).bit_length() - 1))

out = pathlib.Path(__file__).resolve().parents[1] / "parola.txt"
out.write_text(
    "PANO PAROLASI\n"
    "=============\n\n"
    f"{pw}\n\n"
    "Bu dosya .gitignore'da - hicbir zaman GitHub'a gitmez.\n"
    "Parolayi bir parola yoneticisine kaydet, sonra bu dosyayi silebilirsin.\n"
    f"Guc: ~{bits} bit entropi\n",
    encoding="utf-8",
)

rc = subprocess.run(["setx", "DASHBOARD_PASSWORD", pw],
                    capture_output=True, text=True, shell=True).returncode

fp = hashlib.sha256(pw.encode()).hexdigest()[:8].upper()

print(f"Parola uretildi   : {GROUPS * GROUP_LEN + GROUPS - 1} karakter, ~{bits} bit")
print(f"Parmak izi        : {fp}")
print(f"Ortam degiskeni   : {'kaydedildi' if rc == 0 else 'KAYDEDILEMEDI'}")
print(f"Parolayi oku      : {out}")
print()
print("Parola ekrana YAZDIRILMADI. Yukaridaki dosyayi ac ve oku.")
sys.exit(0 if rc == 0 else 1)
