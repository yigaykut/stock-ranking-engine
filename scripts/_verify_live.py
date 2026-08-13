"""Canli sitedeki dosyanin gercekten sifreli ve cozulebilir oldugunu dogrular."""
import base64, json, os, pathlib, re
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes

h = pathlib.Path('canli.html').read_text(encoding='utf-8', errors='replace')
b = json.loads(re.search(r'const BLOB = (\{.*?\});', h, re.S).group(1))
d = base64.b64decode

def key(pw: bytes) -> bytes:
    return PBKDF2HMAC(algorithm=hashes.SHA256(), length=32,
                      salt=d(b['salt']), iterations=b['iter']).derive(pw)

pw = os.environ.get('DASHBOARD_PASSWORD', '').encode()
pt = AESGCM(key(pw)).decrypt(d(b['nonce']), d(b['ct']), None).decode()
print(f"  dogru parola  : acildi, {len(pt)//1024} KB pano")
print(f"  icerik saglam : {'Toplam Etki Puani' in pt and 'const DATA' in pt}")

try:
    AESGCM(key(b'yanlis-parola')).decrypt(d(b['nonce']), d(b['ct']), None)
    print("  yanlis parola : ACILDI - SORUN!")
except Exception:
    print("  yanlis parola : reddedildi (dogru)")
