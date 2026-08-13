"""GitHub Pages yayini — SIZINTIYA KARSI KORUMALI.

Tasarim ilkesi: yayin dizini ASLA proje dizini degildir.

Proje kokunu depo yapip .gitignore'a guvenmek yaygin ama tehlikeli bir
yaklasimdir: tek bir yanlis desen, duz metin panoyu, izleme listeni veya
onbellegi herkese acik bir depoya gonderir ve git gecmisinden silmek zordur.

Bunun yerine ayri bir `publish/` dizini kurulur ve icine YALNIZCA sifreli
dosyalar KOPYALANIR. Duz metin oraya hic girmez.

Ustune bir de son kontrol vardir: gonderim oncesi dosyalar duz metin
belirteclerine karsi taranir. Bir tanesi bile bulunursa yayin DURDURULUR.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
PUBLISH = ROOT / "publish"

# Sifreli bir dosyada ASLA bulunmamasi gereken izler. Bulunursa, sifreleme
# yapilmamis veya bozulmus demektir.
LEAK_MARKERS = [
    "const DATA = {",           # panonun ham veri blogu
    "Toplam Etki Puani",        # bolum basligi
    "SIGMA / HISSE",            # pano basligi
    "izleme listenden",         # sabitleme metni
    "watchlist",                # izleme listesi ipucu
    "snapshot_date",
]

# Sifreli dosyada BULUNMASI gerekenler
REQUIRED_MARKERS = ["const BLOB = {", "AES-GCM", "PBKDF2"]


class LeakDetected(RuntimeError):
    """Duz metin sizintisi tespit edildi — yayin durduruldu."""


def verify_encrypted(path: Path) -> dict:
    """Bir dosyanin gercekten sifreli oldugunu dogrular."""
    text = path.read_text(encoding="utf-8")

    missing = [m for m in REQUIRED_MARKERS if m not in text]
    if missing:
        raise LeakDetected(
            f"{path.name}: sifreli dosya belirteci eksik {missing} — "
            f"bu dosya sifrelenmemis olabilir")

    # Sifreli blogu cikar, kalan duz metinde sizinti ara
    body = re.sub(r"const BLOB = \{.*?\};", "", text, flags=re.S)
    leaks = [m for m in LEAK_MARKERS if m in body]
    if leaks:
        raise LeakDetected(
            f"{path.name}: DUZ METIN SIZINTISI {leaks} — yayin durduruldu")

    m = re.search(r'"iter"\s*:\s*(\d+)', text)
    iterations = int(m.group(1)) if m else 0
    if iterations < 100_000:
        raise LeakDetected(
            f"{path.name}: PBKDF2 tur sayisi cok dusuk ({iterations})")

    return {"file": path.name, "kb": round(len(text) / 1024, 1),
            "iterations": iterations}


def build(index_from: str = "secure_dashboard.html") -> dict:
    """Yayin dizinini sifirdan kurar. Yalnizca sifreli dosyalar kopyalanir."""
    files = {
        index_from: "index.html",
        "secure_watchlist.html": "watchlist.html",
    }
    present = {src: dst for src, dst in files.items() if (OUT / src).exists()}
    if index_from not in present:
        raise FileNotFoundError(
            f"{index_from} yok. Once 'python run.py publish' calistir.")

    # Dizini tamamen sifirla — eski/yanlis dosya kalintisi olmasin
    if PUBLISH.exists():
        for p in PUBLISH.iterdir():
            if p.name == ".git":
                continue
            shutil.rmtree(p) if p.is_dir() else p.unlink()
    PUBLISH.mkdir(parents=True, exist_ok=True)

    checked = []
    for src, dst in present.items():
        s = OUT / src
        checked.append(verify_encrypted(s))       # KOPYALAMADAN ONCE dogrula
        shutil.copy2(s, PUBLISH / dst)

    # GitHub Pages Jekyll islemesini atlasin
    (PUBLISH / ".nojekyll").write_text("", encoding="utf-8")

    # Depoya yanlislikla baska bir sey eklenmesin
    (PUBLISH / ".gitignore").write_text(
        "# Bu dizine YALNIZCA sifreli dosyalar girer.\n"
        "*\n!index.html\n!watchlist.html\n!.nojekyll\n!.gitignore\n",
        encoding="utf-8")

    (PUBLISH / "yayin_bilgisi.json").write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "files": checked,
        "encryption": "AES-256-GCM + PBKDF2-HMAC-SHA256",
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    # Son kontrol: dizindeki HER dosyayi tekrar tara
    for p in PUBLISH.glob("*.html"):
        verify_encrypted(p)

    return {"dir": str(PUBLISH), "files": checked}


# =============================================================================
#  Git islemleri
# =============================================================================
def _run(args: list[str], cwd: Path) -> tuple[int, str]:
    p = subprocess.run(args, cwd=str(cwd), capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def git_push(repo: str, branch: str = "main", message: str | None = None) -> dict:
    """Yayin dizinini GitHub'a gonderir.

    repo: "kullanici/depo" bicimi
    """
    if not PUBLISH.exists():
        raise FileNotFoundError("yayin dizini yok — once build() calistir")

    # Guvenlik: gonderim oncesi SON kontrol
    for p in PUBLISH.glob("*.html"):
        verify_encrypted(p)

    log: list[str] = []
    if not (PUBLISH / ".git").exists():
        for args in (["git", "init"],
                     ["git", "checkout", "-B", branch],
                     ["git", "remote", "add", "origin",
                      f"https://github.com/{repo}.git"]):
            rc, out = _run(args, PUBLISH)
            log.append(f"$ {' '.join(args)}\n{out.strip()}")

    msg = message or f"Pano guncellemesi {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    steps = [
        ["git", "add", "-A"],
        ["git", "-c", "user.name=hisse-pano",
         "-c", "user.email=noreply@example.com", "commit", "-m", msg],
        ["git", "branch", "-M", branch],
        ["git", "push", "-u", "--force", "origin", branch],
    ]
    ok = True
    for args in steps:
        rc, out = _run(args, PUBLISH)
        log.append(f"$ {' '.join(args[:3])}...\n{out.strip()[:400]}")
        # "nothing to commit" hata degildir
        if rc != 0 and "nothing to commit" not in out.lower():
            if args[1] == "push":
                ok = False
            elif "commit" in args:
                continue
    return {"ok": ok, "log": log, "url": f"https://github.com/{repo}"}
