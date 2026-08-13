"""Panoyu SIFRELI, tek dosyalik bir sayfaya cevirir.

Tasarim: ucdan uca sifreleme
----------------------------
Pano icerigi AES-256-GCM ile sifrelenir ve sifreli metin HTML'in icine gomulur.
Cozme islemi tamamen TARAYICIDA, WebCrypto ile yapilir. Parola hicbir yere
gonderilmez, hicbir sunucuda tutulmaz.

Bunun pratik anlami: dosyayi HERHANGI bir yere koyabilirsin — GitHub Pages,
bir web sunucusu, e-posta eki, bulut disk. Parolayi bilmeyen icin dosya
anlamsiz bayt yiginidir. Sunucuya guvenmek zorunda degilsin.

Guvenlik notu — durustce
------------------------
Guvenlik TAMAMEN parolanin gucune baglidir. Sifreli metin dosyanin icinde
oldugu icin saldirgan sinirsiz deneme yapabilir (cevrimdisi saldiri). Bu
yuzden:
  * PBKDF2-HMAC-SHA256, 600.000 tur  (OWASP 2023 onerisi)
  * Rastgele 16 baytlik tuz, her yayinda yeni
  * AES-256-GCM (butunluk dogrulamali — degistirilen dosya cozulmez)

Yine de kisa veya tahmin edilebilir bir parola bu korumayi ise yaramaz kilar.
En az 5 rastgele kelime ya da 16+ karakter kullan.

Parola, bu modul tarafindan HICBIR YERE yazilmaz ve loglanmaz.
"""
from __future__ import annotations

import base64
import json
import os
import secrets
from datetime import datetime, timezone
from pathlib import Path

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes

PBKDF2_ITERATIONS = 600_000          # OWASP 2023, SHA-256 icin
SALT_BYTES = 16
NONCE_BYTES = 12


def _derive(password: str, salt: bytes) -> bytes:
    kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt,
                     iterations=PBKDF2_ITERATIONS)
    return kdf.derive(password.encode("utf-8"))


def encrypt_payload(plaintext: str, password: str) -> dict:
    """AES-256-GCM ile sifreler; tarayicinin cozebilecegi bicimde doner."""
    salt = secrets.token_bytes(SALT_BYTES)
    nonce = secrets.token_bytes(NONCE_BYTES)
    key = _derive(password, salt)
    ct = AESGCM(key).encrypt(nonce, plaintext.encode("utf-8"), None)
    b64 = lambda b: base64.b64encode(b).decode("ascii")   # noqa: E731
    return {"v": 1, "kdf": "PBKDF2-SHA256", "iter": PBKDF2_ITERATIONS,
            "salt": b64(salt), "nonce": b64(nonce), "ct": b64(ct)}


# =============================================================================
#  Kilit ekrani — cozme tamamen tarayicida
# =============================================================================
_SHELL = """<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="robots" content="noindex,nofollow">
<title>__TITLE__</title>
<style>
:root{
  color-scheme: dark;
  --plane:#0a0406; --surface:#120609; --ink:#f4e9e6; --ink-2:#b09a97;
  --ink-3:#6f5c5c; --crimson:#f0483a; --rule:rgba(240,72,58,.20);
  --rule-2:rgba(160,140,140,.13);
  --disp:"Impact","Haettenschweiler","Arial Narrow Bold",sans-serif;
  --mono:"Cascadia Mono","Consolas",ui-monospace,monospace;
  --body:"Segoe UI",-apple-system,sans-serif;
}
*{box-sizing:border-box}
html,body{height:100%}
body{margin:0;background:var(--plane);color:var(--ink);font:15px/1.6 var(--body);
  display:flex;align-items:center;justify-content:center;padding:24px;
  background-image:radial-gradient(ellipse 80% 50% at 50% 0%,rgba(143,33,24,.28),transparent 70%)}
.lock{width:100%;max-width:430px}
.eyebrow{font:600 10px/1 var(--mono);letter-spacing:.4em;text-transform:uppercase;
  color:var(--crimson);margin-bottom:14px}
h1{font:400 clamp(34px,9vw,58px)/.88 var(--disp);letter-spacing:-.01em;
  text-transform:uppercase;margin:0 0 8px;
  text-shadow:0 0 34px rgba(240,72,58,.32),2px 0 0 rgba(240,72,58,.36),
             -2px 0 0 rgba(61,107,138,.26)}
h1 em{font-style:normal;color:var(--crimson);display:block}
.sub{color:var(--ink-2);font-size:13px;margin:16px 0 24px}
form{display:grid;gap:10px}
input{font:400 15px/1 var(--body);padding:15px 16px;border:1px solid var(--rule-2);
  background:var(--surface);color:var(--ink);outline:none;width:100%}
input:focus{border-color:var(--crimson)}
button{font:600 11px/1 var(--mono);letter-spacing:.2em;text-transform:uppercase;
  padding:16px;border:0;background:var(--crimson);color:#fff;cursor:pointer}
button:disabled{opacity:.5;cursor:progress}
button:focus-visible,input:focus-visible{outline:2px solid var(--crimson);outline-offset:3px}
.msg{font:500 11px/1.6 var(--mono);letter-spacing:.08em;min-height:34px;margin-top:6px}
.err{color:var(--crimson)}
.ok{color:#1ba372}
.meta{margin-top:26px;padding-top:18px;border-top:1px solid var(--rule-2);
  font:500 9.5px/1.9 var(--mono);letter-spacing:.14em;text-transform:uppercase;
  color:var(--ink-3)}
.bar{height:3px;background:#1e1013;margin-top:10px;overflow:hidden;display:none}
.bar.on{display:block}
.bar>i{display:block;height:100%;width:35%;background:var(--crimson);
  animation:slide 1.1s ease-in-out infinite}
@keyframes slide{0%{transform:translateX(-100%)}100%{transform:translateX(385%)}}
@media (prefers-reduced-motion:reduce){.bar>i{animation:none;width:100%}}
</style>

<div class="lock" id="lock">
  <div class="eyebrow">Sifreli pano</div>
  <h1>HISSE<em>SIRALAMA</em></h1>
  <p class="sub">Bu sayfanin icerigi ucdan uca sifrelidir. Cozme islemi
    tarayicinda yapilir; parola hicbir yere gonderilmez.</p>
  <form id="f" autocomplete="off">
    <input type="password" id="pw" placeholder="Parola" autocomplete="current-password"
           autofocus>
    <button type="submit" id="go">Coz ve ac</button>
  </form>
  <div class="bar" id="bar"><i></i></div>
  <div class="msg" id="msg"></div>
  <div class="meta">
    Uretildi __GENERATED__<br>
    AES-256-GCM &middot; PBKDF2-SHA256 __ITER__ tur
  </div>
</div>

<script>
const BLOB = __BLOB__;

const $ = id => document.getElementById(id);
const b64 = s => Uint8Array.from(atob(s), c => c.charCodeAt(0));

async function unlock(pw) {
  const enc = new TextEncoder();
  const base = await crypto.subtle.importKey('raw', enc.encode(pw), 'PBKDF2',
                                             false, ['deriveKey']);
  const key = await crypto.subtle.deriveKey(
    {name:'PBKDF2', salt:b64(BLOB.salt), iterations:BLOB.iter, hash:'SHA-256'},
    base, {name:'AES-GCM', length:256}, false, ['decrypt']);
  const plain = await crypto.subtle.decrypt(
    {name:'AES-GCM', iv:b64(BLOB.nonce)}, key, b64(BLOB.ct));
  return new TextDecoder().decode(plain);
}

$('f').onsubmit = async e => {
  e.preventDefault();
  const pw = $('pw').value;
  if (!pw) return;
  $('go').disabled = true;
  $('bar').classList.add('on');
  $('msg').className = 'msg';
  $('msg').textContent = 'Anahtar turetiliyor...';

  // Tarayicinin ekrani boyamasina izin ver (PBKDF2 birkac saniye surer)
  await new Promise(r => setTimeout(r, 30));
  try {
    const html = await unlock(pw);
    $('msg').className = 'msg ok';
    $('msg').textContent = 'Cozuldu, aciliyor...';
    // Cozulen sayfayi ayni sekmede goster
    document.open(); document.write(html); document.close();
  } catch (err) {
    $('bar').classList.remove('on');
    $('go').disabled = false;
    $('msg').className = 'msg err';
    $('msg').textContent = 'Parola yanlis veya dosya bozulmus.';
    $('pw').select();
  }
};
</script>
"""


def encrypt_html(source: Path, password: str, out: Path,
                 title: str = "Sifreli Pano") -> dict:
    """Bir HTML panosunu sifreleyip kilit ekranli tek dosyaya cevirir."""
    plaintext = source.read_text(encoding="utf-8")
    blob = encrypt_payload(plaintext, password)

    page = (_SHELL
            .replace("__TITLE__", title)
            .replace("__BLOB__", json.dumps(blob))
            .replace("__ITER__", f"{PBKDF2_ITERATIONS:,}".replace(",", "."))
            .replace("__GENERATED__",
                     datetime.now(timezone.utc).strftime("%Y.%m.%d %H:%M UTC")))

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    return {
        "source": str(source),
        "output": str(out),
        "source_kb": round(len(plaintext) / 1024, 1),
        "output_kb": round(len(page) / 1024, 1),
        "iterations": PBKDF2_ITERATIONS,
    }


# =============================================================================
#  Parola alma — parola sureclerarasi hicbir yerde saklanmaz
# =============================================================================
def get_password(env_var: str = "DASHBOARD_PASSWORD",
                 confirm: bool = False) -> str | None:
    """Paroayi ortam degiskeninden veya etkilesimli olarak alir.

    Komut satirinda --password ile GECIRILMEZ: kabuk gecmisine ve surec
    listesine dusmesini engellemek icin.
    """
    pw = os.environ.get(env_var)
    if pw:
        return pw

    import getpass
    try:
        pw = getpass.getpass("Pano parolasi: ")
    except (EOFError, KeyboardInterrupt):
        return None
    if not pw:
        return None
    if confirm:
        again = getpass.getpass("Parolayi tekrar gir: ")
        if again != pw:
            return None
    return pw


def password_strength(pw: str) -> dict:
    """Kaba parola gucu degerlendirmesi (uyari amacli)."""
    import math

    pools = 0
    if any(c.islower() for c in pw):
        pools += 26
    if any(c.isupper() for c in pw):
        pools += 26
    if any(c.isdigit() for c in pw):
        pools += 10
    if any(not c.isalnum() for c in pw):
        pools += 32
    bits = len(pw) * math.log2(pools) if pools else 0

    if bits < 45:
        level, note = "ZAYIF", "cevrimdisi saldiriya dayanmaz — mutlaka uzat"
    elif bits < 70:
        level, note = "ORTA", "kabul edilebilir, ama daha uzunu iyi olur"
    else:
        level, note = "GUCLU", "iyi"
    return {"bits": round(bits), "level": level, "note": note,
            "length": len(pw)}
